struct ScanUniforms {
    width_px: u32,
    height_px: u32,
    samples_per_pixel: u32,
    grid_w: u32,

    grid_h: u32,
    grid_d: u32,
    seed: u32,
    pad0: u32,

    film_w: f32,
    film_h: f32,
    film_d: f32,
    cell_size: f32,

    silver_opacity: f32,
    pad1: f32,
    pad2: f32,
    pad3: f32,
};

struct GpuGridCell {
    pos: vec4<f32>,
    dims_exp: vec4<f32>,
    rot: vec4<f32>,
    flags: vec4<u32>,
};

@group(0) @binding(0) var<uniform> uniforms: ScanUniforms;
@group(0) @binding(1) var<storage, read> cells: array<GpuGridCell>;
@group(0) @binding(2) var<storage, read_write> out_density: array<f32>;

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

fn quat_inv_rot(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let qv = vec3<f32>(-q.x, -q.y, -q.z);
    let t = 2.0 * cross(qv, v);
    return v + q.w * t + cross(qv, t);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let px = global_id.x;
    let py = global_id.y;

    if (px >= uniforms.width_px || py >= uniforms.height_px) {
        return;
    }

    let w_px = uniforms.width_px;
    let h_px = uniforms.height_px;
    let spp = max(uniforms.samples_per_pixel, 1u);
    let grid_dim = max(u32(round(sqrt(f32(spp)))), 1u);
    let actual_spp = grid_dim * grid_dim;

    let pixel_w = uniforms.film_w / f32(w_px);
    let pixel_h = uniforms.film_h / f32(h_px);

    var rng_state = uniforms.seed ^ (px * 1973u + py * 9277u * 65537u + 1013904223u);

    var accumulated_transmittance = 0.0;

    for (var sy = 0u; sy < grid_dim; sy++) {
        for (var sx = 0u; sx < grid_dim; sx++) {
            let sub_x = (f32(sx) + rand_f32(&rng_state)) / f32(grid_dim);
            let sub_y = (f32(sy) + rand_f32(&rng_state)) / f32(grid_dim);

            let film_x = (f32(px) + sub_x) * pixel_w;
            let film_y = (f32(py) + sub_y) * pixel_h;

            let ray_orig = vec3<f32>(film_x, film_y, uniforms.film_d + 0.1);
            let ray_dir  = vec3<f32>(0.0, 0.0, -1.0);

            var total_path = 0.0;

            let cx_i = i32(floor(film_x / uniforms.cell_size));
            let cy_i = i32(floor(film_y / uniforms.cell_size));

            // Query neighborhood covering full bounding spheres of rotated crystals
            for (var dy = -2; dy <= 2; dy++) {
                let gy = cy_i + dy;
                if (gy < 0 || gy >= i32(uniforms.grid_h)) {
                    continue;
                }
                for (var dx = -2; dx <= 2; dx++) {
                    let gx = cx_i + dx;
                    if (gx < 0 || gx >= i32(uniforms.grid_w)) {
                        continue;
                    }

                    for (var gz = 0; gz < i32(uniforms.grid_d); gz++) {
                        let cell_idx = u32(gz) * (uniforms.grid_w * uniforms.grid_h) + u32(gy) * uniforms.grid_w + u32(gx);
                        let cell = cells[cell_idx];

                        if (cell.flags.x == 0u || cell.flags.z == 0u) {
                            continue;
                        }

                        let local_orig = quat_inv_rot(cell.rot, ray_orig - cell.pos.xyz);
                        let local_dir  = quat_inv_rot(cell.rot, ray_dir);
                        let dims = cell.dims_exp.xyz * cell.dims_exp.w;

                        if (cell.flags.y == 0u) {
                            // Cubic grain box slab test (Tri-X) with robust division protection
                            let sign_d = sign(local_dir);
                            let safe_sign = select(sign_d, vec3<f32>(1.0), sign_d == vec3<f32>(0.0));
                            let safe_dir = select(local_dir, safe_sign * 1e-6, abs(local_dir) < vec3<f32>(1e-6));
                            let inv_d = 1.0 / safe_dir;

                            let t1 = (-dims - local_orig) * inv_d;
                            let t2 = ( dims - local_orig) * inv_d;
                            let t_min_v = min(t1, t2);
                            let t_max_v = max(t1, t2);
                            let t_enter = max(max(t_min_v.x, t_min_v.y), t_min_v.z);
                            let t_exit  = min(min(t_max_v.x, t_max_v.y), t_max_v.z);

                            if (t_exit >= t_enter && t_exit > 0.0) {
                                let path = t_exit - max(t_enter, 0.0);
                                if (path > 0.0) {
                                    total_path += path;
                                }
                            }
                        } else {
                            // Tabular / Octahedral ellipsoid test (T-Max / HP5)
                            let p = local_orig / dims;
                            let d = local_dir / dims;
                            let a = dot(d, d);
                            if (a > 1e-6) {
                                let b = 2.0 * dot(p, d);
                                let c = dot(p, p) - 1.0;
                                let disc = b * b - 4.0 * a * c;
                                if (disc >= 0.0) {
                                    let sqrt_disc = sqrt(disc);
                                    let inv_2a = 0.5 / a;
                                    let t0 = (-b - sqrt_disc) * inv_2a;
                                    let t1 = (-b + sqrt_disc) * inv_2a;
                                    let t_enter = min(t0, t1);
                                    let t_exit  = max(t0, t1);
                                    if (t_exit > 0.0 && t_exit >= t_enter) {
                                        let path = t_exit - max(t_enter, 0.0);
                                        if (path > 0.0) {
                                            total_path += path;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            let safe_path = clamp(total_path, 0.0, uniforms.film_d);
            let t = exp(-uniforms.silver_opacity * safe_path);
            accumulated_transmittance += t;
        }
    }

    let mean_t = clamp(accumulated_transmittance / f32(actual_spp), 1e-4, 1.0);
    // Optical density D_neg = -log10(T) + D_min
    let d_val = -log(mean_t) / 2.302585092994046 + 0.05;

    let out_idx = py * uniforms.width_px + px;
    out_density[out_idx] = d_val;
}
