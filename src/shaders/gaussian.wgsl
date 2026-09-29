struct GaussianUniforms {
    width_px: u32,
    height_px: u32,
    direction: u32, // 0 = horizontal, 1 = vertical
    pad: u32,
};

@group(0) @binding(0) var<uniform> uniforms: GaussianUniforms;
@group(0) @binding(1) var<storage, read> in_data: array<f32>;
@group(0) @binding(2) var<storage, read_write> out_data: array<f32>;
@group(0) @binding(3) var<storage, read> orig_density: array<f32>; // used only in pass 2 to compute delta_d

// Precomputed 1D Gaussian kernel for sigma = 2.5 (radius 8, 17 taps, sum = 1.0)
const KERNEL_RADIUS: i32 = 8;
const KERNEL_WEIGHTS: array<f32, 17> = array<f32, 17>(
    0.002621, 0.007802, 0.019445, 0.040523, 0.070624, 0.103073, 0.125925, 0.134267,
    0.137440, // center (k = 0)
    0.134267, 0.125925, 0.103073, 0.070624, 0.040523, 0.019445, 0.007802, 0.002621
);

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let px = global_id.x;
    let py = global_id.y;

    if (px >= uniforms.width_px || py >= uniforms.height_px) {
        return;
    }

    let w = i32(uniforms.width_px);
    let h = i32(uniforms.height_px);
    let ix = i32(px);
    let iy = i32(py);

    var sum = 0.0;

    if (uniforms.direction == 0u) {
        // Horizontal pass
        for (var k = -KERNEL_RADIUS; k <= KERNEL_RADIUS; k++) {
            let sx = clamp(ix + k, 0, w - 1);
            let sample_idx = u32(iy * w + sx);
            let kw = KERNEL_WEIGHTS[k + KERNEL_RADIUS];
            sum += in_data[sample_idx] * kw;
        }
        let out_idx = py * uniforms.width_px + px;
        out_data[out_idx] = sum;
    } else {
        // Vertical pass: computes d_smooth, then outputs delta_d = orig_density - d_smooth
        for (var k = -KERNEL_RADIUS; k <= KERNEL_RADIUS; k++) {
            let sy = clamp(iy + k, 0, h - 1);
            let sample_idx = u32(sy * w + ix);
            let kw = KERNEL_WEIGHTS[k + KERNEL_RADIUS];
            sum += in_data[sample_idx] * kw;
        }
        let out_idx = py * uniforms.width_px + px;
        let d_orig = orig_density[out_idx];
        out_data[out_idx] = d_orig - sum;
    }
}
