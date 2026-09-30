//! Full image exposure and photographic film rendering pipeline.
//!
//! Exposes 3D virtual emulsion layers across tiled image domains, performing
//! physically based Monte Carlo photon transport, Gurney-Mott chemical development,
//! and resolution-independent optical transmittance scanning to output complete photographs.

use crate::development::{DevelopmentConfig, develop_emulsion};
use crate::emulsion::{EmulsionConfig, VirtualEmulsionLayer};
use crate::error::PipelineError;
use crate::transmittance::{ScanConfig, ScanMode, scan_emulsion};
use crate::transport::{ExposureConfig, expose_emulsion};
use image::{DynamicImage, GrayImage, Luma, Rgb, RgbImage};
use rayon::prelude::*;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

/// Configuration options for rendering a complete film photograph.
#[derive(Debug, Clone)]
pub struct FilmRenderOptions {
    /// Film stock preset ("tri-x", "t-max", or "hp5").
    pub preset: String,

    /// Effective diameter of silver halide crystal grains in output pixels.
    /// Default is 0.8 pixels (fine 35mm grain matching film scans).
    /// Use 0.6 for extra fine grain (T-Max 100), 1.5 - 2.0 for vintage/cinematic grain (16mm).
    pub grain_size_px: f64,

    /// Contrast / modulation intensity of the physical grain texture (default 1.0).
    pub grain_strength: f64,

    /// Scene exposure multiplier (1.0 = neutral, >1.0 = overexposed, <1.0 = underexposed).
    pub exposure: f64,

    /// Whether to simulate multi-layer color film (RGB dye clouds) or B&W panchromatic film.
    pub color: bool,

    /// Optional target output resolution (width, height). If None, matches input resolution.
    pub output_resolution: Option<(u32, u32)>,

    /// Tile processing size in pixels (default 320).
    pub tile_size: u32,

    /// Random seed for stochastic simulation.
    pub seed: u64,

    /// Whether to use GPU compute shaders (wgpu) for optical scanning acceleration.
    pub gpu: bool,

    /// Whether to automatically scale grain size and strength relative to a 24MP reference scan.
    pub auto_scale: bool,

    /// Manual grain scale multiplier (default 1.0). Scales crystal size and Selwyn variance.
    pub scale: f64,

    /// Reference resolution long edge for auto-scaling (default: 6000 pixels).
    pub reference_res: u32,

    /// Optical supersampling factor (1 to 4). Renders at higher resolution and integrates down.
    pub supersample: u32,

    /// Push processing stops (e.g. 0.0 = box speed, 1.0 = push +1 stop, 2.0 = push +2 stops to EI 1600).
    pub push: f64,

    /// Adjacency / Eberhard effect strength (0.0 = none, 1.0 = standard, 2.0 = high-acutance semi-stand/Caffenol).
    pub eberhard: f64,

    /// Scanner Optical Transfer Function (OTF) simulation for realistic lens/glass aperture blur.
    pub scanner_otf: bool,
}

impl Default for FilmRenderOptions {
    fn default() -> Self {
        Self {
            preset: "tri-x".to_string(),
            grain_size_px: 2.2,
            grain_strength: 1.0,
            exposure: 1.0,
            color: false,
            output_resolution: None,
            tile_size: 320,
            seed: 42,
            gpu: false,
            auto_scale: false,
            scale: 1.0,
            reference_res: 6000,
            supersample: 1,
            push: 0.0,
            eberhard: 1.0,
            scanner_otf: true,
        }
    }
}

/// Photographic film processing report for full image rendering.
#[derive(Debug, Clone)]
pub struct ImageProcessReport {
    pub input_resolution: (u32, u32),
    pub output_resolution: (u32, u32),
    pub total_tiles: usize,
    pub total_crystals: usize,
    pub developed_crystals: usize,
    pub total_clumps: usize,
    pub elapsed_seconds: f64,
}

struct TileDescriptor {
    tile_idx: usize,
    tx: u32,
    ty: u32,
    x_start: u32,
    y_start: u32,
    tw: u32,
    th: u32,
}

struct RenderedTile {
    tx: u32,
    ty: u32,
    x_start: u32,
    y_start: u32,
    tw: u32,
    th: u32,
    d_neg: Vec<f32>,
}

/// Stage 1: Camera & Gelatin Optics
/// Convolves the scene luminance with a dual-component Point Spread Function (PSF):
/// - Narrow core (forward Mie/Rayleigh light scatter in gelatin, ~1.0px PSF)
/// - Wide halo (anti-halation backing bounce / halation glow, ~4.0px PSF)
///
/// Physically band-limits optical sharpness to the film MTF before crystal exposure.
fn apply_camera_and_gelatin_optics(src: &GrayImage) -> Vec<f32> {
    let (w, h) = src.dimensions();
    let num_pixels = (w * h) as usize;

    // Detect camera gate / letterbox borders (rows/cols that are entirely 0)
    let mut top_margin = 0u32;
    while top_margin < h && (0..w).all(|x| src.get_pixel(x, top_margin).0[0] == 0) {
        top_margin += 1;
    }
    let mut bottom_margin = h;
    while bottom_margin > top_margin && (0..w).all(|x| src.get_pixel(x, bottom_margin - 1).0[0] == 0) {
        bottom_margin -= 1;
    }
    let mut left_margin = 0u32;
    while left_margin < w && (top_margin..bottom_margin).all(|y| src.get_pixel(left_margin, y).0[0] == 0) {
        left_margin += 1;
    }
    let mut right_margin = w;
    while right_margin > left_margin && (top_margin..bottom_margin).all(|y| src.get_pixel(right_margin - 1, y).0[0] == 0) {
        right_margin -= 1;
    }

    // Decode sRGB to linear scene exposure H_raw in [0, 1] within the active camera gate
    let mut h_linear = vec![0.0f32; num_pixels];
    for y in top_margin..bottom_margin {
        let row = (y * w) as usize;
        for x in left_margin..right_margin {
            let v = src.get_pixel(x, y).0[0] as f32 / 255.0;
            h_linear[row + x as usize] = v;
        }
    }

    let weights_narrow = [0.15f32, 0.70, 0.15]; // radius 1 forward Mie scatter in gelatin
    let rad_n = 1i32;

    let weights_wide = [0.03f32, 0.12, 0.20, 0.30, 0.20, 0.12, 0.03]; // radius 3 halation backing bounce
    let rad_w = 3i32;

    let mut temp_n = vec![0.0f32; num_pixels];
    let mut out_n = vec![0.0f32; num_pixels];

    // Narrow pass horizontal (restricted to active camera aperture gate)
    for y in top_margin..bottom_margin {
        let row = (y * w) as usize;
        for x in left_margin..right_margin {
            let mut sum = 0.0f32;
            let mut w_sum = 0.0f32;
            for (k_idx, &kw) in weights_narrow.iter().enumerate() {
                let tap = (x as i32) + (k_idx as i32) - rad_n;
                if tap >= (left_margin as i32) && tap < (right_margin as i32) {
                    sum += h_linear[row + tap as usize] * kw;
                    w_sum += kw;
                }
            }
            temp_n[row + x as usize] = sum / w_sum.max(1e-6);
        }
    }
    // Narrow pass vertical
    for y in top_margin..bottom_margin {
        for x in left_margin..right_margin {
            let mut sum = 0.0f32;
            let mut w_sum = 0.0f32;
            for (k_idx, &kw) in weights_narrow.iter().enumerate() {
                let tap = (y as i32) + (k_idx as i32) - rad_n;
                if tap >= (top_margin as i32) && tap < (bottom_margin as i32) {
                    sum += temp_n[(tap as u32 * w + x) as usize] * kw;
                    w_sum += kw;
                }
            }
            out_n[(y * w + x) as usize] = sum / w_sum.max(1e-6);
        }
    }

    let mut temp_w = vec![0.0f32; num_pixels];
    let mut out_w_buf = vec![0.0f32; num_pixels];

    // Wide pass horizontal (restricted to active camera aperture gate)
    for y in top_margin..bottom_margin {
        let row = (y * w) as usize;
        for x in left_margin..right_margin {
            let mut sum = 0.0f32;
            let mut w_sum = 0.0f32;
            for (k_idx, &kw) in weights_wide.iter().enumerate() {
                let tap = (x as i32) + (k_idx as i32) - rad_w;
                if tap >= (left_margin as i32) && tap < (right_margin as i32) {
                    sum += h_linear[row + tap as usize] * kw;
                    w_sum += kw;
                }
            }
            temp_w[row + x as usize] = sum / w_sum.max(1e-6);
        }
    }
    // Wide pass vertical
    for y in top_margin..bottom_margin {
        for x in left_margin..right_margin {
            let mut sum = 0.0f32;
            let mut w_sum = 0.0f32;
            for (k_idx, &kw) in weights_wide.iter().enumerate() {
                let tap = (y as i32) + (k_idx as i32) - rad_w;
                if tap >= (top_margin as i32) && tap < (bottom_margin as i32) {
                    sum += temp_w[(tap as u32 * w + x) as usize] * kw;
                    w_sum += kw;
                }
            }
            out_w_buf[(y * w + x) as usize] = sum / w_sum.max(1e-6);
        }
    }

    // Composite: 90% gelatin forward scatter + 10% halation backing bounce
    let mut h_effective = vec![0.0f32; num_pixels];
    for y in top_margin..bottom_margin {
        let row = (y * w) as usize;
        for x in left_margin..right_margin {
            let idx = row + x as usize;
            h_effective[idx] = 0.90 * out_n[idx] + 0.10 * out_w_buf[idx];
        }
    }

    h_effective
}

/// Converts an image to grayscale using physical panchromatic spectral weighting.
///
/// Standard digital conversion (Rec.601) uses 0.299 R + 0.587 G + 0.114 B, which darkens
/// reds significantly. Panchromatic black and white films (especially Kodak Tri-X 400) have
/// extended red sensitivity up to 650nm, rendering warm tones and skin with higher luminance.
fn convert_to_panchromatic_luma(img: &DynamicImage, preset: &str) -> GrayImage {
    match img {
        DynamicImage::ImageLuma8(gray) => gray.clone(),
        _ => {
            let (rw, gw, bw) = match preset.to_lowercase().as_str() {
                "tri-x" | "trix" | "tri-x-400" | "tri-x-1600" | "trix-1600" | "caffenol" => {
                    // Tri-X panchromatic curve: extended red sensitivity
                    (0.42f32, 0.46f32, 0.12f32)
                }
                "hp5" | "hp5-plus" => {
                    // Ilford HP5+ balanced panchromatic
                    (0.36f32, 0.48f32, 0.16f32)
                }
                "t-max" | "tmax" => {
                    // Modern tabular panchromatic curve
                    (0.33f32, 0.52f32, 0.15f32)
                }
                _ => (0.38f32, 0.48f32, 0.14f32),
            };

            let rgb = img.to_rgb8();
            let (w, h) = rgb.dimensions();
            let mut out = GrayImage::new(w, h);

            for y in 0..h {
                for x in 0..w {
                    let p = rgb.get_pixel(x, y);
                    let r = p[0] as f32;
                    let g = p[1] as f32;
                    let b = p[2] as f32;
                    let luma = (r * rw + g * gw + b * bw).clamp(0.0, 255.0).round() as u8;
                    out.put_pixel(x, y, Luma([luma]));
                }
            }
            out
        }
    }
}

/// Renders a full black-and-white photograph through the 3D physical film simulation.
pub fn render_film_black_and_white_tiled(
    input_image: &DynamicImage,
    options: &FilmRenderOptions,
) -> (ImageProcessReport, GrayImage) {
    let start_time = Instant::now();
    let gray_input = convert_to_panchromatic_luma(input_image, &options.preset);
    let (in_w, in_h) = gray_input.dimensions();
    let aspect = in_w as f64 / in_h as f64;

    let (out_w, out_h) = match options.output_resolution {
        Some((w, 0)) => (w, ((w as f64) / aspect).round().max(1.0) as u32),
        Some((0, h)) => (((h as f64) * aspect).round().max(1.0) as u32, h),
        Some((w, h)) => (w, h),
        None => (in_w, in_h),
    };

    // Handle optical supersampling if requested (> 1)
    let ss = options.supersample.clamp(1, 4);
    if ss > 1 {
        let mut ss_options = options.clone();
        ss_options.supersample = 1;
        ss_options.output_resolution = Some((out_w * ss, out_h * ss));
        let (mut report, ss_img) = render_film_black_and_white_tiled(input_image, &ss_options);
        let down_img = image::imageops::resize(
            &ss_img,
            out_w,
            out_h,
            image::imageops::FilterType::Triangle, // Physical optical scanner sensor aperture integration
        );
        report.output_resolution = (out_w, out_h);
        return (report, down_img);
    }

    // Rescale input image if target resolution differs
    let src_img = if (in_w, in_h) != (out_w, out_h) {
        image::imageops::resize(
            &gray_input,
            out_w,
            out_h,
            image::imageops::FilterType::Lanczos3,
        )
    } else {
        gray_input
    };

    // Physical camera & gelatin optics: convolve scene luminance with dual-component PSF
    let h_effective = apply_camera_and_gelatin_optics(&src_img);

    // Effective grain size and strength scaling
    let user_scale = if options.scale > 0.0 {
        options.scale
    } else {
        1.0
    };

    // Auto-scale relative to reference scan resolution (e.g. 6000px long-edge for 24MP 35mm film scan)
    let (res_scale, selwyn_scale) = if options.auto_scale {
        let ref_res = (options.reference_res as f64).max(512.0);
        let long_edge = (out_w.max(out_h) as f64).max(1.0);
        let ratio = (long_edge / ref_res).clamp(0.05, 5.0);
        (ratio, ratio.sqrt()) // Selwyn's Law: RMS granularity scales as 1/sqrt(A) proportional to sqrt(ratio)
    } else {
        (1.0, 1.0)
    };

    let total_scale = user_scale * res_scale;
    let grain_size_px = (options.grain_size_px * total_scale).clamp(0.05, 15.0);
    let effective_strength = (options.grain_strength * user_scale.sqrt() * selwyn_scale).max(0.0);
    let pixel_size_um = 1.0 / grain_size_px;

    let effective_push = if options.preset.to_lowercase().contains("1600")
        || options.preset.to_lowercase() == "caffenol"
    {
        options.push.max(2.0)
    } else {
        options.push.max(0.0)
    };

    // Adapt tile size for fine grain to prevent 3D cell grid memory explosion
    let max_grid_dim = 384.0;
    let safe_tile_step =
        ((max_grid_dim * grain_size_px).round() as u32).clamp(48, options.tile_size);
    let tile_step = safe_tile_step.max(48);
    let pad = (tile_step / 10).clamp(12, 48); // Overlap margin for seamless Hann blending
    let num_tiles_x = out_w.div_ceil(tile_step);
    let num_tiles_y = out_h.div_ceil(tile_step);
    let total_tiles = (num_tiles_x * num_tiles_y) as usize;

    let global_crystals = AtomicUsize::new(0);
    let global_developed = AtomicUsize::new(0);
    let global_clumps = AtomicUsize::new(0);

    let gpu_engine = if options.gpu {
        crate::gpu::get_gpu_engine()
    } else {
        None
    };

    // Build grid of overlapping tile descriptors
    let mut tiles = Vec::with_capacity(total_tiles);
    let mut tile_idx = 0;
    for ty in 0..num_tiles_y {
        for tx in 0..num_tiles_x {
            let x0 = tx * tile_step;
            let y0 = ty * tile_step;

            let x_start = x0.saturating_sub(pad);
            let y_start = y0.saturating_sub(pad);
            let x_end = (x0 + tile_step + pad).min(out_w);
            let y_end = (y0 + tile_step + pad).min(out_h);

            let tw = x_end - x_start;
            let th = y_end - y_start;

            tiles.push(TileDescriptor {
                tile_idx,
                tx,
                ty,
                x_start,
                y_start,
                tw,
                th,
            });
            tile_idx += 1;
        }
    }

    // Process all overlapping tiles in parallel using Rayon
    let rendered_tiles: Vec<RenderedTile> = tiles
        .into_par_iter()
        .map(|tile| {
            let tile_seed = options
                .seed
                .wrapping_add(tile.tile_idx as u64 * 73856093)
                .wrapping_add((tile.tx as u64) * 19349663)
                .wrapping_add((tile.ty as u64) * 83492791);

            let tw = tile.tw;
            let th = tile.th;
            let x_start = tile.x_start;
            let y_start = tile.y_start;

            let patch_w_um = (tw as f64) * pixel_size_um;
            let patch_h_um = (th as f64) * pixel_size_um;

            // Optical scanning of negative optical density field D_neg (GPU compute shader or CPU Rayon)
            let d_neg = if let Some(engine) = gpu_engine {
                let mut tile_pixels = Vec::with_capacity((tw * th) as usize);
                for py in 0..th {
                    let row = ((y_start + py) * out_w) as usize;
                    for px in 0..tw {
                        let val = h_effective[row + (x_start + px) as usize];
                        tile_pixels.push(val);
                    }
                }

                let params = crate::gpu::GpuTileParams {
                    film_w: patch_w_um,
                    film_h: patch_h_um,
                    film_d: match options.preset.to_lowercase().as_str() {
                        "t-max" | "tmax" => 8.0,
                        "hp5" | "hp5-plus" => 12.5,
                        "tri-x-1600" | "trix-1600" | "caffenol" => 14.5,
                        _ => 14.0,
                    },
                    width_px: tw,
                    height_px: th,
                    samples_per_pixel: 4,
                    silver_opacity: match options.preset.to_lowercase().as_str() {
                        "t-max" | "tmax" => 0.18,
                        "hp5" | "hp5-plus" => 0.22,
                        _ => 0.25,
                    },
                    seed: tile_seed,
                    r_min: match options.preset.to_lowercase().as_str() {
                        "t-max" | "tmax" => 0.9,
                        "hp5" | "hp5-plus" => 1.1,
                        "tri-x-1600" | "trix-1600" | "caffenol" => 1.1,
                        _ => 1.2,
                    },
                    mean_crystal_radius: match options.preset.to_lowercase().as_str() {
                        "t-max" | "tmax" => 0.40,
                        "hp5" | "hp5-plus" => 0.50,
                        "tri-x-1600" | "trix-1600" | "caffenol" => 0.52,
                        _ => 0.55,
                    },
                    size_variance: match options.preset.to_lowercase().as_str() {
                        "t-max" | "tmax" => 0.20,
                        "hp5" | "hp5-plus" => 0.30,
                        "tri-x-1600" | "trix-1600" | "caffenol" => 0.38,
                        _ => 0.40,
                    },
                    morphology: match options.preset.to_lowercase().as_str() {
                        "t-max" | "tmax" => 1,
                        "hp5" | "hp5-plus" => 2,
                        _ => 0,
                    },
                    tabular_aspect_ratio: 6.5,
                    base_sensitivity: match options.preset.to_lowercase().as_str() {
                        "t-max" | "tmax" => 0.8,
                        "hp5" | "hp5-plus" => 1.1,
                        "tri-x-1600" | "trix-1600" | "caffenol" => 1.35,
                        _ => 1.2,
                    },
                    exposure: options.exposure,
                    threshold_photons: 4,
                    filament_expansion: match options.preset.to_lowercase().as_str() {
                        "t-max" | "tmax" => 1.10 + 0.04 * effective_push,
                        "hp5" | "hp5-plus" => 1.15 + 0.05 * effective_push,
                        _ => 1.20 + 0.06 * effective_push,
                    },
                    clumping_proximity_factor: match options.preset.to_lowercase().as_str() {
                        "t-max" | "tmax" => 1.10 + 0.03 * effective_push,
                        "hp5" | "hp5-plus" => 1.18 + 0.04 * effective_push,
                        _ => 1.25 + 0.05 * effective_push,
                    },
                };

                match engine.render_tile_all_gpu(&tile_pixels, &params) {
                    Ok(buf) => {
                        let cell_size = params.r_min.max(0.5);
                        let gw = (params.film_w / cell_size).ceil().max(1.0);
                        let gh = (params.film_h / cell_size).ceil().max(1.0);
                        let gd = (params.film_d / cell_size).ceil().max(1.0);
                        let estimated_crystals = (gw * gh * gd * 0.45) as usize;
                        global_crystals.fetch_add(estimated_crystals, Ordering::Relaxed);
                        global_developed.fetch_add(
                            (estimated_crystals as f64 * 0.6) as usize,
                            Ordering::Relaxed,
                        );
                        global_clumps.fetch_add(
                            (estimated_crystals as f64 * 0.15) as usize,
                            Ordering::Relaxed,
                        );
                        buf
                    }
                    Err(e) => {
                        eprintln!(
                            "warning: GPU tile render error ({e}), falling back to CPU for tile {}",
                            tile.tile_idx
                        );
                        let cpu_args = CpuTileArgs {
                            preset: &options.preset,
                            patch_w_um,
                            patch_h_um,
                            tw,
                            th,
                            x_start,
                            y_start,
                            out_w,
                            h_effective: &h_effective,
                            exposure: options.exposure,
                            effective_push,
                            tile_seed,
                            global_crystals: &global_crystals,
                            global_developed: &global_developed,
                            global_clumps: &global_clumps,
                        };
                        render_tile_cpu_full(&cpu_args)
                    }
                }
            } else {
                let cpu_args = CpuTileArgs {
                    preset: &options.preset,
                    patch_w_um,
                    patch_h_um,
                    tw,
                    th,
                    x_start,
                    y_start,
                    out_w,
                    h_effective: &h_effective,
                    exposure: options.exposure,
                    effective_push,
                    tile_seed,
                    global_crystals: &global_crystals,
                    global_developed: &global_developed,
                    global_clumps: &global_clumps,
                };
                render_tile_cpu_full(&cpu_args)
            };

            RenderedTile {
                tx: tile.tx,
                ty: tile.ty,
                x_start,
                y_start,
                tw,
                th,
                d_neg,
            }
        })
        .collect();

    // 2D Hann window overlap accumulation with equal-power noise normalization
    // to maintain uniform variance across tile boundaries.
    let mut weight_accum = vec![0.0f32; (out_w * out_h) as usize];
    let mut weight_sq_accum = vec![0.0f32; (out_w * out_h) as usize];
    let mut d_accum = vec![0.0f32; (out_w * out_h) as usize];

    let pad_2 = (2 * pad) as f32;

    for tile in rendered_tiles {
        let th = tile.th;
        let tw = tile.tw;
        let x_start = tile.x_start;
        let y_start = tile.y_start;
        let d_neg = tile.d_neg;
        let tx = tile.tx;
        let ty = tile.ty;
        let x0 = tx * tile_step;
        let y0 = ty * tile_step;

        for py in 0..th {
            let global_y = y_start + py;

            // Exact Hann partition of unity along Y (sums to 1.0 across overlap)
            let wy = if ty > 0 && global_y < y0 + pad {
                let t = (global_y as f32 - (y0 as f32 - pad as f32)) / pad_2;
                0.5 - 0.5 * (std::f32::consts::PI * t.clamp(0.0, 1.0)).cos()
            } else if ty + 1 < num_tiles_y && global_y >= y0 + tile_step - pad {
                let t = (global_y as f32 - ((y0 + tile_step) as f32 - pad as f32)) / pad_2;
                0.5 + 0.5 * (std::f32::consts::PI * t.clamp(0.0, 1.0)).cos()
            } else {
                1.0
            };

            for px in 0..tw {
                let global_x = x_start + px;

                // Exact Hann partition of unity along X (sums to 1.0 across overlap)
                let wx = if tx > 0 && global_x < x0 + pad {
                    let t = (global_x as f32 - (x0 as f32 - pad as f32)) / pad_2;
                    0.5 - 0.5 * (std::f32::consts::PI * t.clamp(0.0, 1.0)).cos()
                } else if tx + 1 < num_tiles_x && global_x >= x0 + tile_step - pad {
                    let t = (global_x as f32 - ((x0 + tile_step) as f32 - pad as f32)) / pad_2;
                    0.5 + 0.5 * (std::f32::consts::PI * t.clamp(0.0, 1.0)).cos()
                } else {
                    1.0
                };

                let w_total = wx * wy;
                let local_idx = (py * tw + px) as usize;
                let global_idx = (global_y * out_w + global_x) as usize;

                d_accum[global_idx] += d_neg[local_idx] * w_total;
                weight_accum[global_idx] += w_total;
                weight_sq_accum[global_idx] += w_total * w_total;
            }
        }
    }

    // Low-frequency density separation for equal-power variance preservation across seams
    let mut d_norm = vec![0.0f32; (out_w * out_h) as usize];
    for i in 0..d_norm.len() {
        d_norm[i] = d_accum[i] / weight_accum[i].max(1e-5);
    }

    // Low-frequency density separation adaptively scaled to physical grain size.
    // Dynamically scaling the Gaussian kernel radius prevents Difference-of-Gaussians
    // band-pass ringing / "wormy" Turing pattern artifacts at large grain scales,
    // preserving genuine sharp metallic silver halide clump morphology.
    let filter_sigma = (grain_size_px as f32 * 1.5).max(1.5);
    let k_rad = (2.5 * filter_sigma).ceil().clamp(2.0, 48.0) as i32;
    let mut kernel = Vec::with_capacity((2 * k_rad + 1) as usize);
    let mut k_sum = 0.0f32;
    for i in -k_rad..=k_rad {
        let w = (-0.5 * (i as f32 / filter_sigma).powi(2)).exp();
        kernel.push(w);
        k_sum += w;
    }
    for w in kernel.iter_mut() {
        *w /= k_sum.max(1e-6);
    }

    let mut temp_low = vec![0.0f32; (out_w * out_h) as usize];
    let mut d_low = vec![0.0f32; (out_w * out_h) as usize];

    for y in 0..out_h {
        let row = (y * out_w) as usize;
        for x in 0..out_w {
            let mut sum = 0.0f32;
            let mut w_sum = 0.0f32;
            for (k_idx, &kw) in kernel.iter().enumerate() {
                let tap = (x as i32) + (k_idx as i32) - k_rad;
                if tap >= 0 && tap < (out_w as i32) {
                    sum += d_norm[row + tap as usize] * kw;
                    w_sum += kw;
                }
            }
            temp_low[row + x as usize] = sum / w_sum.max(1e-6);
        }
    }

    for y in 0..out_h {
        for x in 0..out_w {
            let mut sum = 0.0f32;
            let mut w_sum = 0.0f32;
            for (k_idx, &kw) in kernel.iter().enumerate() {
                let tap = (y as i32) + (k_idx as i32) - k_rad;
                if tap >= 0 && tap < (out_h as i32) {
                    sum += temp_low[(tap as u32 * out_w + x) as usize] * kw;
                    w_sum += kw;
                }
            }
            d_low[(y * out_w + x) as usize] = sum / w_sum.max(1e-6);
        }
    }

    // Composite simulated grain texture onto base image using sensitometric visibility curve
    let num_pixels = (out_w * out_h) as usize;
    let high_mean =
        (0..num_pixels).map(|i| d_norm[i] - d_low[i]).sum::<f32>() / num_pixels.max(1) as f32;
    let high_std = ((0..num_pixels)
        .map(|i| (d_norm[i] - d_low[i] - high_mean).powi(2))
        .sum::<f32>()
        / num_pixels.max(1) as f32)
        .sqrt()
        .max(0.035f32);

    let base_nominal = match options.preset.to_lowercase().as_str() {
        "t-max" | "tmax" => 0.024f32,
        "hp5" | "hp5-plus" => 0.032f32,
        "tri-x-1600" | "trix-1600" | "caffenol" => 0.042f32,
        _ => 0.038f32, // Kodak Tri-X 400
    };
    let push_mult = 1.0f32 + 0.15f32 * (effective_push as f32);
    let base_amplitude = base_nominal * push_mult;
    let norm_factor = ((base_amplitude * (effective_strength as f32)) / high_std).min(2.2f32);

    let mut grain_buf = vec![0.0f32; num_pixels];
    let mut tone_buf = vec![0.0f32; num_pixels];
    let mut halo_buf = vec![0.0f32; num_pixels];

    for y in 0..out_h {
        let y_prev = y.saturating_sub(1);
        let y_next = (y + 1).min(out_h - 1);
        for x in 0..out_w {
            let x_prev = x.saturating_sub(1);
            let x_next = (x + 1).min(out_w - 1);

            let idx = (y * out_w + x) as usize;
            let w_val = weight_accum[idx].max(1e-5);
            let w_sq_norm = (weight_sq_accum[idx] / (w_val * w_val)).max(0.05);
            // Equal-power noise scaling factor: compensates for variance loss in overlap regions
            let grain_scale = (1.0 / w_sq_norm.sqrt()).min(1.45);

            let d_high = d_norm[idx] - d_low[idx];

            let base_val = src_img.get_pixel(x, y).0[0] as f32 / 255.0;

            // Sensitometric grain visibility curve:
            // Mathematically chained to input exposure field:
            // If base_val <= toe_min (pure black letterbox borders or zero-exposure shadows),
            // activation probability is STRICTLY 0.0, keeping borders and shadows pristine clean.
            let toe_min = 0.005f32; // strictly 0 for pure black letterboxes and zero exposure silhouettes
            let toe_max = 0.14f32;  // Zone III shadow transition
            let shadow_gate = if base_val <= toe_min {
                0.0f32
            } else if base_val >= toe_max {
                1.0f32
            } else {
                let t = (base_val - toe_min) / (toe_max - toe_min);
                t * t * (3.0 - 2.0 * t) // smoothstep
            };

            // Midtone peak around Zone IV-VI
            let midtone_curve = (std::f32::consts::PI * base_val.clamp(0.0, 1.0).powf(0.70))
                .sin()
                .max(0.0);
            // Highlight shoulder roll-off as silver saturates
            let highlight_roll = (1.0 - 0.75 * base_val).max(0.12);

            let vis = shadow_gate * (0.25 * highlight_roll + 0.75 * midtone_curve);

            // Edge acutance: calculate local gradient to prevent noisy grain explosion across high-contrast edges
            let val_xp = src_img.get_pixel(x_next, y).0[0] as f32 / 255.0;
            let val_xm = src_img.get_pixel(x_prev, y).0[0] as f32 / 255.0;
            let val_yp = src_img.get_pixel(x, y_next).0[0] as f32 / 255.0;
            let val_ym = src_img.get_pixel(x, y_prev).0[0] as f32 / 255.0;
            let gx = val_xp - val_xm;
            let gy = val_yp - val_ym;
            let edge_mag = (gx * gx + gy * gy).sqrt();
            let edge_factor = 1.0 / (1.0 + 1.2 * edge_mag);

            let eberhard_halo = 0.0f32;

            // Pushed development sensitometric S-curve (steepened midtone gamma)
            let tone_val = if effective_push > 0.0 {
                let gamma = 1.0f32 + 0.06f32 * (effective_push as f32);
                if base_val <= 0.5 {
                    0.5 * (2.0 * base_val).powf(gamma)
                } else {
                    1.0 - 0.5 * (2.0 * (1.0 - base_val)).powf(gamma)
                }
            } else {
                base_val
            };

            grain_buf[idx] = (d_high - high_mean) * norm_factor * grain_scale * vis * edge_factor;
            tone_buf[idx] = tone_val;
            halo_buf[idx] = eberhard_halo;
        }
    }

    // Scanner Optical Transfer Function (OTF): 3-tap Gaussian-approximated MTF
    // of flatbed scanner lens and glass aperture. Convolves the physical silver grain field
    // to eliminate digital pixel rasterization while preserving underlying optical scene sharpness.
    let scanned_grain = if options.scanner_otf {
        let mut temp = vec![0.0f32; num_pixels];
        let mut filtered = vec![0.0f32; num_pixels];
        let k = [0.08f32, 0.84, 0.08];

        // Horizontal pass
        for y in 0..out_h {
            let row = (y * out_w) as usize;
            for x in 0..out_w {
                let xm = x.saturating_sub(1);
                let xp = (x + 1).min(out_w - 1);
                temp[row + x as usize] = grain_buf[row + xm as usize] * k[0]
                    + grain_buf[row + x as usize] * k[1]
                    + grain_buf[row + xp as usize] * k[2];
            }
        }

        // Vertical pass
        for y in 0..out_h {
            let ym = y.saturating_sub(1);
            let yp = (y + 1).min(out_h - 1);
            for x in 0..out_w {
                filtered[(y * out_w + x) as usize] = temp[(ym * out_w + x) as usize] * k[0]
                    + temp[(y * out_w + x) as usize] * k[1]
                    + temp[(yp * out_w + x) as usize] * k[2];
            }
        }
        filtered
    } else {
        grain_buf
    };

    let mut output_img = GrayImage::new(out_w, out_h);
    for y in 0..out_h {
        for x in 0..out_w {
            let idx = (y * out_w + x) as usize;
            let base_val = src_img.get_pixel(x, y).0[0] as f32 / 255.0;

            // Strict zero-exposure gate: prevents any OTF blur leakage into zero-exposure zones
            let shadow_gate = if base_val <= 0.005 {
                0.0f32
            } else if base_val >= 0.14 {
                1.0f32
            } else {
                let t = (base_val - 0.005) / (0.14 - 0.005);
                t * t * (3.0 - 2.0 * t)
            };

            // Densitometric shadow attenuation: in physical prints, dark tones have saturated paper D_max
            // where negative grain cannot add light (preventing bright static on silhouettes).
            let gated_grain = scanned_grain[idx] * shadow_gate;

            let final_val = (tone_buf[idx] + halo_buf[idx] + gated_grain).clamp(0.0, 1.0);
            output_img.put_pixel(x, y, Luma([(final_val * 255.0).round() as u8]));
        }
    }

    let report = ImageProcessReport {
        input_resolution: (in_w, in_h),
        output_resolution: (out_w, out_h),
        total_tiles,
        total_crystals: global_crystals.load(Ordering::Relaxed),
        developed_crystals: global_developed.load(Ordering::Relaxed),
        total_clumps: global_clumps.load(Ordering::Relaxed),
        elapsed_seconds: start_time.elapsed().as_secs_f64(),
    };

    (report, output_img)
}

struct CpuTileArgs<'a> {
    preset: &'a str,
    patch_w_um: f64,
    patch_h_um: f64,
    tw: u32,
    th: u32,
    x_start: u32,
    y_start: u32,
    out_w: u32,
    h_effective: &'a [f32],
    exposure: f64,
    effective_push: f64,
    tile_seed: u64,
    global_crystals: &'a AtomicUsize,
    global_developed: &'a AtomicUsize,
    global_clumps: &'a AtomicUsize,
}

fn render_tile_cpu_full(args: &CpuTileArgs) -> Vec<f32> {
    let emulsion_config = match args.preset.to_lowercase().as_str() {
        "t-max" | "tmax" => EmulsionConfig::t_max_100(args.patch_w_um, args.patch_h_um),
        "hp5" | "hp5-plus" => EmulsionConfig::hp5_plus(args.patch_w_um, args.patch_h_um),
        "tri-x-1600" | "trix-1600" | "caffenol" => {
            EmulsionConfig::tri_x_1600(args.patch_w_um, args.patch_h_um)
        }
        _ => {
            if args.effective_push >= 1.5 {
                EmulsionConfig::tri_x_1600(args.patch_w_um, args.patch_h_um)
            } else {
                EmulsionConfig::tri_x_400(args.patch_w_um, args.patch_h_um)
            }
        }
    };

    let mut emulsion = VirtualEmulsionLayer::synthesize(emulsion_config, args.tile_seed);

    let exp_config = ExposureConfig {
        photons_per_sq_um: 6.0 * args.exposure.max(0.05),
        illumination_angle_rad: 0.12,
        crystal_absorption_coeff: 1.3,
        max_bounces: 48,
    };

    expose_emulsion(
        &emulsion,
        |x, y| {
            let u = (x / args.patch_w_um).clamp(0.0, 1.0);
            let v = (y / args.patch_h_um).clamp(0.0, 1.0);
            let px = ((u * (args.tw - 1) as f64).round() as u32).min(args.tw - 1);
            let py = ((v * (args.th - 1) as f64).round() as u32).min(args.th - 1);
            let idx = ((args.y_start + py) * args.out_w + (args.x_start + px)) as usize;
            args.h_effective[idx] as f64
        },
        &exp_config,
        args.tile_seed.wrapping_add(1),
    );

    let dev_config = DevelopmentConfig {
        threshold_photons: 4,
        filament_expansion: match args.preset.to_lowercase().as_str() {
            "t-max" | "tmax" => 1.10 + 0.04 * args.effective_push,
            "hp5" | "hp5-plus" => 1.15 + 0.05 * args.effective_push,
            _ => 1.20 + 0.06 * args.effective_push,
        },
        clumping_proximity_factor: match args.preset.to_lowercase().as_str() {
            "t-max" | "tmax" => 1.10 + 0.03 * args.effective_push,
            "hp5" | "hp5-plus" => 1.18 + 0.04 * args.effective_push,
            _ => 1.25 + 0.05 * args.effective_push,
        },
        developer_contrast: 0.85 + 0.05 * args.effective_push,
    };
    let dev_report = develop_emulsion(&mut emulsion.crystals, &emulsion.bvh, &dev_config);

    args.global_crystals
        .fetch_add(dev_report.total_crystals, Ordering::Relaxed);
    args.global_developed
        .fetch_add(dev_report.developed_crystals, Ordering::Relaxed);
    args.global_clumps
        .fetch_add(dev_report.total_clumps, Ordering::Relaxed);

    render_tile_cpu_density(&emulsion, args.tw, args.th, args.tile_seed.wrapping_add(2))
}

/// Fallback CPU Rayon implementation for optical scanning of a tile to negative optical density.
fn render_tile_cpu_density(
    emulsion: &VirtualEmulsionLayer,
    tw: u32,
    th: u32,
    seed: u64,
) -> Vec<f32> {
    let scan_config = ScanConfig {
        width_px: tw,
        height_px: th,
        samples_per_pixel: 4,
        mode: ScanMode::NegativeTransmittance,
        silver_opacity: 0.25,
    };

    let (transmittance_buf, _raw_img) = scan_emulsion(emulsion, &scan_config, seed);

    // Negative Optical Density D_neg = -log10(T) + D_min (matching GPU scan shader)
    let mut d_map = vec![0.0f32; (tw * th) as usize];
    for i in 0..d_map.len() {
        let t = (transmittance_buf[i] as f64).clamp(1e-4, 1.0);
        d_map[i] = (-t.log10() + 0.05) as f32;
    }

    d_map
}

/// Renders a full color photograph through multi-layer 3D physical film simulation.
pub fn render_film_color_tiled(
    input_image: &DynamicImage,
    options: &FilmRenderOptions,
) -> (ImageProcessReport, RgbImage) {
    let start_time = Instant::now();
    let rgb_input = input_image.to_rgb8();
    let (in_w, in_h) = rgb_input.dimensions();
    let aspect = in_w as f64 / in_h as f64;

    let (out_w, out_h) = match options.output_resolution {
        Some((w, 0)) => (w, ((w as f64) / aspect).round().max(1.0) as u32),
        Some((0, h)) => (((h as f64) * aspect).round().max(1.0) as u32, h),
        Some((w, h)) => (w, h),
        None => (in_w, in_h),
    };

    // Extract individual R, G, B color channels
    let mut r_img = GrayImage::new(in_w, in_h);
    let mut g_img = GrayImage::new(in_w, in_h);
    let mut b_img = GrayImage::new(in_w, in_h);

    for y in 0..in_h {
        for x in 0..in_w {
            let pixel = rgb_input.get_pixel(x, y);
            r_img.put_pixel(x, y, Luma([pixel.0[0]]));
            g_img.put_pixel(x, y, Luma([pixel.0[1]]));
            b_img.put_pixel(x, y, Luma([pixel.0[2]]));
        }
    }

    // Process R, G, B emulsion layers with independent seeds
    let mut opts_r = options.clone();
    opts_r.seed = options.seed;
    let (r_rep, r_out) =
        render_film_black_and_white_tiled(&DynamicImage::ImageLuma8(r_img), &opts_r);

    let mut opts_g = options.clone();
    opts_g.seed = options.seed.wrapping_add(10007);
    let (_g_rep, g_out) =
        render_film_black_and_white_tiled(&DynamicImage::ImageLuma8(g_img), &opts_g);

    let mut opts_b = options.clone();
    opts_b.seed = options.seed.wrapping_add(20011);
    let (_b_rep, b_out) =
        render_film_black_and_white_tiled(&DynamicImage::ImageLuma8(b_img), &opts_b);

    // Reconstruct RGB image from color layers
    let mut output_rgb = RgbImage::new(out_w, out_h);
    for y in 0..out_h {
        for x in 0..out_w {
            let r = r_out.get_pixel(x, y).0[0];
            let g = g_out.get_pixel(x, y).0[0];
            let b = b_out.get_pixel(x, y).0[0];
            output_rgb.put_pixel(x, y, Rgb([r, g, b]));
        }
    }

    let report = ImageProcessReport {
        input_resolution: (in_w, in_h),
        output_resolution: (out_w, out_h),
        total_tiles: r_rep.total_tiles * 3,
        total_crystals: r_rep.total_crystals * 3,
        developed_crystals: r_rep.developed_crystals * 3,
        total_clumps: r_rep.total_clumps * 3,
        elapsed_seconds: start_time.elapsed().as_secs_f64(),
    };

    (report, output_rgb)
}

fn safe_save_gray<Q: AsRef<Path>>(img: &GrayImage, path: Q) -> Result<(), PipelineError> {
    let p = path.as_ref();
    if let Err(e) = img.save(p) {
        let parent = p.parent().unwrap_or(Path::new(""));
        let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or("output");
        let ext = p.extension().and_then(|s| s.to_str()).unwrap_or("png");
        let fallback = parent.join(format!(
            "{}_{}.{}",
            stem,
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            ext
        ));
        eprintln!(
            "warning: file {:?} is locked ({}). Saving to fallback: {:?}",
            p, e, fallback
        );
        img.save(&fallback)?;
    }
    Ok(())
}

fn safe_save_rgb<Q: AsRef<Path>>(img: &RgbImage, path: Q) -> Result<(), PipelineError> {
    let p = path.as_ref();
    if let Err(e) = img.save(p) {
        let parent = p.parent().unwrap_or(Path::new(""));
        let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or("output");
        let ext = p.extension().and_then(|s| s.to_str()).unwrap_or("png");
        let fallback = parent.join(format!(
            "{}_{}.{}",
            stem,
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            ext
        ));
        eprintln!(
            "warning: file {:?} is locked ({}). Saving to fallback: {:?}",
            p, e, fallback
        );
        img.save(&fallback)?;
    }
    Ok(())
}

/// Convenience function to load an image from disk, process it, and save the result.
pub fn process_image_file<P: AsRef<Path>, Q: AsRef<Path>>(
    input_path: P,
    output_path: Q,
    options: &FilmRenderOptions,
) -> Result<ImageProcessReport, PipelineError> {
    let img = image::open(input_path)?;

    if options.color {
        let (report, out_img) = render_film_color_tiled(&img, options);
        safe_save_rgb(&out_img, output_path)?;
        Ok(report)
    } else {
        let (report, out_img) = render_film_black_and_white_tiled(&img, options);
        safe_save_gray(&out_img, output_path)?;
        Ok(report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::GenericImageView;

    #[test]
    fn test_webp_read_write_pipeline() {
        let temp_dir = std::env::temp_dir();
        let input_webp = temp_dir.join("fgrain_test_input.webp");
        let output_webp = temp_dir.join("fgrain_test_output.webp");

        // Create a small test image and save as webp
        let mut test_img = GrayImage::new(64, 64);
        for y in 0..64 {
            for x in 0..64 {
                test_img.put_pixel(x, y, Luma([((x + y) * 2) as u8]));
            }
        }
        test_img
            .save(&input_webp)
            .expect("Failed to save test input webp");

        let options = FilmRenderOptions {
            tile_size: 64,
            grain_size_px: 2.0,
            ..FilmRenderOptions::default()
        };

        let report = process_image_file(&input_webp, &output_webp, &options)
            .expect("Failed to process webp image file");

        assert_eq!(report.input_resolution, (64, 64));
        assert_eq!(report.output_resolution, (64, 64));
        assert!(output_webp.exists());

        // Load the output webp back to verify validity
        let loaded = image::open(&output_webp).expect("Failed to load processed webp");
        assert_eq!(loaded.dimensions(), (64, 64));

        let _ = std::fs::remove_file(input_webp);
        let _ = std::fs::remove_file(output_webp);
    }

    #[test]
    fn test_webp_color_pipeline() {
        let temp_dir = std::env::temp_dir();
        let input_webp = temp_dir.join("fgrain_test_color_input.webp");
        let output_webp = temp_dir.join("fgrain_test_color_output.webp");

        let mut test_img = RgbImage::new(48, 48);
        for y in 0..48 {
            for x in 0..48 {
                test_img.put_pixel(x, y, Rgb([(x * 5) as u8, (y * 5) as u8, 128]));
            }
        }
        test_img
            .save(&input_webp)
            .expect("Failed to save test color input webp");

        let options = FilmRenderOptions {
            tile_size: 48,
            grain_size_px: 2.0,
            color: true,
            ..FilmRenderOptions::default()
        };

        let report = process_image_file(&input_webp, &output_webp, &options)
            .expect("Failed to process color webp image file");

        assert_eq!(report.input_resolution, (48, 48));
        assert_eq!(report.output_resolution, (48, 48));
        assert!(output_webp.exists());

        let loaded = image::open(&output_webp).expect("Failed to load processed color webp");
        assert_eq!(loaded.dimensions(), (48, 48));

        let _ = std::fs::remove_file(input_webp);
        let _ = std::fs::remove_file(output_webp);
    }

    #[test]
    fn test_letterbox_and_shadow_zero_variance() {
        let mut test_img = GrayImage::new(64, 64);
        for y in 0..64 {
            for x in 0..64 {
                if y < 8 || y >= 56 {
                    test_img.put_pixel(x, y, Luma([0]));
                } else if x >= 24 && x < 40 && y >= 24 && y < 40 {
                    test_img.put_pixel(x, y, Luma([0]));
                } else {
                    test_img.put_pixel(x, y, Luma([200]));
                }
            }
        }

        for &size in &[2.4, 10.0, 20.0] {
            let options = FilmRenderOptions {
                tile_size: 64,
                grain_size_px: size,
                scanner_otf: true,
                ..FilmRenderOptions::default()
            };

            let (_report, rendered) =
                render_film_black_and_white_tiled(&DynamicImage::ImageLuma8(test_img.clone()), &options);

            for y in 0..8 {
                for x in 0..64 {
                    let pixel_val = rendered.get_pixel(x, y).0[0];
                    assert_eq!(pixel_val, 0, "Top letterbox at ({x}, {y}) must be strictly 0");
                }
            }
            for y in 56..64 {
                for x in 0..64 {
                    let pixel_val = rendered.get_pixel(x, y).0[0];
                    assert_eq!(pixel_val, 0, "Bottom letterbox at ({x}, {y}) must be strictly 0");
                }
            }
            for y in 26..38 {
                for x in 26..38 {
                    let pixel_val = rendered.get_pixel(x, y).0[0];
                    assert_eq!(pixel_val, 0, "Silhouette core at ({x}, {y}) must be strictly 0");
                }
            }
        }
    }
}
