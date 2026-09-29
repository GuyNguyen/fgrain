struct DevelopUniforms {
    grid_w: u32,
    grid_h: u32,
    grid_d: u32,
    threshold_photons: u32,

    width_px: u32,
    height_px: u32,
    seed: u32,
    stage: u32, // 0 = expose & develop, 1 = clumping

    film_w: f32,
    film_h: f32,
    film_d: f32,
    cell_size: f32,

    exposure: f32,
    filament_expansion: f32,
    clumping_factor: f32,
    base_sensitivity: f32,
};

struct GpuGridCell {
    pos: vec4<f32>,
    dims_exp: vec4<f32>,
    rot: vec4<f32>,
    flags: vec4<u32>,
};

@group(0) @binding(0) var<uniform> uniforms: DevelopUniforms;
@group(0) @binding(1) var<storage, read> input_image: array<f32>;
@group(0) @binding(2) var<storage, read_write> cells: array<GpuGridCell>;

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
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gx = global_id.x;
    let gy = global_id.y;
    let gz = global_id.z;

    if (gx >= uniforms.grid_w || gy >= uniforms.grid_h || gz >= uniforms.grid_d) {
        return;
    }

    let cell_idx = gz * (uniforms.grid_w * uniforms.grid_h) + gy * uniforms.grid_w + gx;
    var cell = cells[cell_idx];

    // Status: 0 = empty, 1 = occupied
    if (cell.flags.x == 0u) {
        return;
    }

    if (uniforms.stage == 0u) {
        // Pass 0: Exposure & Gurney-Mott latent image formation
        let pos = cell.pos.xyz;
        let u = clamp(pos.x / uniforms.film_w, 0.0, 1.0);
        let v = clamp(pos.y / uniforms.film_h, 0.0, 1.0);
        let px = clamp(u32(round(u * f32(uniforms.width_px - 1u))), 0u, uniforms.width_px - 1u);
        let py = clamp(u32(round(v * f32(uniforms.height_px - 1u))), 0u, uniforms.height_px - 1u);
        let pixel_val = input_image[py * uniforms.width_px + px];
        let local_irradiance = pixel_val * uniforms.exposure;

        // Physical photographic toe threshold:
        // Deep shadow regions (local_irradiance < 0.05, i.e. 8-bit digital pixel <= 13)
        // receive insufficient energy to form stable sub-latent specks (latent image regression).
        // Grains in unexposed areas remain completely unreduced and are dissolved by the fixer bath,
        // leaving the gelatin base optically transparent with zero noise variance (Base + Fog).
        if (local_irradiance < 0.05) {
            cell.flags.z = 0u; // developed = 0
            cell.dims_exp.w = 0.0;
            cell.flags.x = 0u; // dissolved by fixer bath!
            cells[cell_idx] = cell;
            return;
        }

        let depth_atten = exp(-0.10 * pos.z);
        let radius = cell.dims_exp.x;
        let area_factor = (radius * radius) / (0.55 * 0.55);
        let expected_photons = local_irradiance * depth_atten * area_factor * uniforms.base_sensitivity * 1.6;

        var rng = uniforms.seed ^ (gx * 1973u + gy * 9277u * 65537u + gz * 26699u + 0x85EBCA6Bu);

        // Stochastic Poisson quantum conversion
        let L = exp(-min(expected_photons, 30.0));
        var k = 0u;
        var p = 1.0;
        while (p > L && k < 40u) {
            k = k + 1u;
            p = p * rand_f32(&rng);
        }
        var absorbed = 0u;
        if (k > 0u) {
            absorbed = k - 1u;
        }
        if (expected_photons > 30.0) {
            let u1 = max(rand_f32(&rng), 1e-6);
            let u2 = rand_f32(&rng);
            let norm_z = sqrt(-2.0 * log(u1)) * cos(2.0 * PI * u2);
            let approx = expected_photons + sqrt(expected_photons) * norm_z;
            absorbed = u32(max(approx, 0.0));
        }

        cell.flags.w = absorbed;

        // Chemical sensitization & latent speck statistics:
        // Polydisperse silver halide crystals possess varying numbers of gold/sulfur sensitivity centers.
        // Fast sensitized grains develop at 3 photons (~30%), standard cubic grains require 4 photons (~70%).
        let speck_rand = rand_f32(&rng);
        let base_thresh = select(4.0, 3.0, speck_rand < 0.30);
        let size_ratio = clamp(0.55 / max(radius, 0.20), 0.75, 1.8);
        let dynamic_threshold = u32(max(round(base_thresh * size_ratio), 2.0));

        if (absorbed >= dynamic_threshold) {
            cell.flags.z = 1u; // developed = 1
            cell.dims_exp.w = uniforms.filament_expansion; // expansion
        } else {
            cell.flags.z = 0u; // developed = 0
            cell.dims_exp.w = 0.0;
            cell.flags.x = 0u; // dissolved by fixer bath!
        }
        cells[cell_idx] = cell;

    } else {
        // Pass 1: Multi-Grain Catalytic Clumping
        if (cell.flags.z == 0u) {
            return;
        }

        let curr_pos = cell.pos.xyz;
        var touching_count = 0u;
        var clump_center = curr_pos;

        let igx = i32(gx);
        let igy = i32(gy);
        let igz = i32(gz);

        // Effective filament mesh span for infectious catalytic clumping:
        // Metallic silver filaments extend 2.0x to 2.5x beyond the unreduced crystal radius,
        // allowing adjacent developing crystals in neighboring cells to entangle and coalesce.
        let clumping_reach = uniforms.cell_size * uniforms.clumping_factor * 1.50;

        for (var dz = -1; dz <= 1; dz++) {
            let nz = igz + dz;
            if (nz < 0 || nz >= i32(uniforms.grid_d)) {
                continue;
            }
            for (var dy = -1; dy <= 1; dy++) {
                let ny = igy + dy;
                if (ny < 0 || ny >= i32(uniforms.grid_h)) {
                    continue;
                }
                for (var dx = -1; dx <= 1; dx++) {
                    let nx = igx + dx;
                    if (nx < 0 || nx >= i32(uniforms.grid_w)) {
                        continue;
                    }
                    if (dx == 0 && dy == 0 && dz == 0) {
                        continue;
                    }

                    let n_idx = u32(nz) * (uniforms.grid_w * uniforms.grid_h) + u32(ny) * uniforms.grid_w + u32(nx);
                    let neighbor = cells[n_idx];
                    if (neighbor.flags.x == 1u && neighbor.flags.z == 1u) {
                        let diff = neighbor.pos.xyz - curr_pos;
                        if (dot(diff, diff) <= clumping_reach * clumping_reach) {
                            touching_count = touching_count + 1u;
                            clump_center = clump_center + neighbor.pos.xyz;
                        }
                    }
                }
            }
        }

        if (touching_count > 0u) {
            // Catalytic dendritic expansion: grains touching active neighbors grow extensive filament meshes,
            // forming ragged 5-to-15 grain macrograins
            let clump_boost = 1.12 + 0.08 * f32(min(touching_count, 8u));
            cell.dims_exp.w = cell.dims_exp.w * clump_boost;

            // Barycentric coalescence towards clump center (creates organic "wormy" silver clusters)
            let avg_center = clump_center / f32(touching_count + 1u);
            cell.pos = vec4<f32>(mix(curr_pos, avg_center, 0.35), 0.0);

            cells[cell_idx] = cell;
        }
    }
}
