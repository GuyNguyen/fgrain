struct PoissonUniforms {
    grid_w: u32,
    grid_h: u32,
    grid_d: u32,
    phase: u32,

    cell_size: f32,
    r_min: f32,
    mean_radius: f32,
    size_variance: f32,

    film_w: f32,
    film_h: f32,
    film_d: f32,
    seed: u32,

    morphology: u32, // 0 = Cubic, 1 = Tabular, 2 = Octahedral
    tabular_aspect_ratio: f32,
    base_sensitivity: f32,
    pad: u32,
};

struct GpuGridCell {
    pos: vec4<f32>,       // xyz: physical pos (um), w: 0.0
    dims_exp: vec4<f32>,  // xyz: dimensions (radii), w: expansion
    rot: vec4<f32>,       // xyzw: quaternion
    flags: vec4<u32>,     // x: status (0=empty, 1=occupied), y: morphology, z: developed, w: absorbed
};

@group(0) @binding(0) var<uniform> uniforms: PoissonUniforms;
@group(0) @binding(1) var<storage, read_write> cells: array<GpuGridCell>;

fn hash_u32(state: u32) -> u32 {
    var x = state;
    x = x ^ (x >> 16u);
    x = x * 0x7feb352du;
    x = x ^ (x >> 15u);
    x = x * 0x846ca68bu;
    x = x ^ (x >> 16u);
    return x;
}

fn rand_f32(state: ptr<function, u32>) -> f32 {
    *state = hash_u32(*state);
    return f32(*state) / 4294967296.0;
}

const PI: f32 = 3.141592653589793;

@compute @workgroup_size(8, 8, 4)
fn clear_grid(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= uniforms.grid_w || id.y >= uniforms.grid_h || id.z >= uniforms.grid_d) {
        return;
    }
    let idx = (id.z * uniforms.grid_h + id.y) * uniforms.grid_w + id.x;
    cells[idx].pos = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    cells[idx].dims_exp = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    cells[idx].rot = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    cells[idx].flags = vec4<u32>(0u, 0u, 0u, 0u);
}

// Pass 1: Jittered Stratified Emulsion Synthesis
@compute @workgroup_size(8, 8, 4)
fn generate_candidates(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gx = global_id.x;
    let gy = global_id.y;
    let gz = global_id.z;

    if (gx >= uniforms.grid_w || gy >= uniforms.grid_h || gz >= uniforms.grid_d) {
        return;
    }

    let cell_idx = gz * (uniforms.grid_w * uniforms.grid_h) + gy * uniforms.grid_w + gx;

    var rng = uniforms.seed ^ (gx * 1973u + gy * 9277u * 65537u + gz * 26699u + 0x9E3779B9u);

    // Continuous uniform stochastic jitter in [0.05, 0.95]
    let offset_x = 0.05 + 0.90 * rand_f32(&rng);
    let offset_y = 0.05 + 0.90 * rand_f32(&rng);
    let offset_z = 0.05 + 0.90 * rand_f32(&rng);

    let cand_x = clamp((f32(gx) + offset_x) * uniforms.cell_size, 0.0, uniforms.film_w);
    let cand_y = clamp((f32(gy) + offset_y) * uniforms.cell_size, 0.0, uniforms.film_h);
    let cand_z = clamp((f32(gz) + offset_z) * uniforms.cell_size, 0.0, uniforms.film_d);
    let cand_pos = vec3<f32>(cand_x, cand_y, cand_z);

    // Log-normal crystal radius via Box-Muller transform
    let u1 = max(rand_f32(&rng), 1e-6);
    let u2 = rand_f32(&rng);
    let norm_z = sqrt(-2.0 * log(u1)) * cos(2.0 * PI * u2);
    let radius = clamp(
        exp(log(uniforms.mean_radius) + uniforms.size_variance * norm_z),
        0.15,
        uniforms.cell_size * 0.65
    );

    var dims = vec3<f32>(radius, radius, radius);
    var rot = vec4<f32>(0.0, 0.0, 0.0, 1.0);

    if (uniforms.morphology == 1u) {
        // Tabular T-Grain (T-Max): flat hexagonal/tabular tablet
        let aspect = max(uniforms.tabular_aspect_ratio, 1.0);
        let r_plane = radius * sqrt(aspect);
        let r_thick = radius / sqrt(aspect);
        dims = vec3<f32>(r_plane, r_plane, r_thick);

        let tilt_theta = rand_f32(&rng) * 0.20;
        let tilt_phi = rand_f32(&rng) * 2.0 * PI;
        let half_t = tilt_theta * 0.5;
        rot = vec4<f32>(
            sin(half_t) * cos(tilt_phi),
            sin(half_t) * sin(tilt_phi),
            0.0,
            cos(half_t)
        );
    } else {
        // Cubic (Tri-X) or Octahedral (HP5): isotropic 3D orientation on S^3
        let q1 = rand_f32(&rng);
        let q2 = rand_f32(&rng);
        let q3 = rand_f32(&rng);
        rot = vec4<f32>(
            sqrt(1.0 - q1) * sin(2.0 * PI * q2),
            sqrt(1.0 - q1) * cos(2.0 * PI * q2),
            sqrt(q1) * sin(2.0 * PI * q3),
            sqrt(q1) * cos(2.0 * PI * q3)
        );
    }

    cells[cell_idx].pos = vec4<f32>(cand_pos, 0.0);
    cells[cell_idx].dims_exp = vec4<f32>(dims, 0.0);
    cells[cell_idx].rot = rot;
    cells[cell_idx].flags = vec4<u32>(1u, uniforms.morphology, 0u, 0u);
}

// Pass 2: Colloidal Electrostatic Repulsion Relaxation
// Simulates the physical double-layer electrostatic repulsion of halide crystals in gelatin
@compute @workgroup_size(8, 8, 4)
fn resolve_conflicts(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gx = global_id.x;
    let gy = global_id.y;
    let gz = global_id.z;

    if (gx >= uniforms.grid_w || gy >= uniforms.grid_h || gz >= uniforms.grid_d) {
        return;
    }

    let cell_idx = gz * (uniforms.grid_w * uniforms.grid_h) + gy * uniforms.grid_w + gx;
    let my_pos = cells[cell_idx].pos.xyz;
    let r_min = uniforms.r_min;

    let igx = i32(gx);
    let igy = i32(gy);
    let igz = i32(gz);

    var force = vec3<f32>(0.0, 0.0, 0.0);

    for (var dz = -1; dz <= 1; dz++) {
        let nz = igz + dz;
        if (nz < 0 || nz >= i32(uniforms.grid_d)) { continue; }
        for (var dy = -1; dy <= 1; dy++) {
            let ny = igy + dy;
            if (ny < 0 || ny >= i32(uniforms.grid_h)) { continue; }
            for (var dx = -1; dx <= 1; dx++) {
                let nx = igx + dx;
                if (nx < 0 || nx >= i32(uniforms.grid_w)) { continue; }
                if (dx == 0 && dy == 0 && dz == 0) { continue; }

                let n_idx = u32(nz) * (uniforms.grid_w * uniforms.grid_h) + u32(ny) * uniforms.grid_w + u32(nx);
                let neighbor_pos = cells[n_idx].pos.xyz;
                let diff = my_pos - neighbor_pos;
                let dist_sq = dot(diff, diff);

                if (dist_sq < r_min * r_min && dist_sq > 1e-6) {
                    let dist = sqrt(dist_sq);
                    let push = (r_min - dist) / dist;
                    force += diff * push;
                }
            }
        }
    }

    let relaxed_pos = my_pos + force * 0.25;
    let clamped_pos = vec3<f32>(
        clamp(relaxed_pos.x, 0.0, uniforms.film_w),
        clamp(relaxed_pos.y, 0.0, uniforms.film_h),
        clamp(relaxed_pos.z, 0.0, uniforms.film_d)
    );

    cells[cell_idx].pos = vec4<f32>(clamped_pos, 0.0);
}
