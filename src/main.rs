//! # fgrain CLI - Physically Based Film Grain Simulator
//!
//! Offline CLI renderer for physical film grain simulation.

use fgrain::{
    DevelopmentConfig, EmulsionConfig, ExposureConfig, ScanConfig, ScanMode, VirtualEmulsionLayer,
    develop_emulsion, expose_emulsion, process_image_file, scan_emulsion,
};
use std::env;
use std::time::Instant;

fn main() {
    println!("============================================================");
    println!(" fgrain: Physically based film grain simulation engine");
    println!("============================================================");

    let args: Vec<String> = env::args().collect();
    let mode = args.get(1).map(|s| s.as_str()).unwrap_or("gradient");

    match mode {
        "apply" | "image" => run_image_apply(&args[2..]),
        "tri-x" => run_preset_demo("tri-x"),
        "t-max" => run_preset_demo("t-max"),
        "halation" => run_halation_demo(),
        _ => {
            let lower = mode.to_lowercase();
            if lower.ends_with(".png")
                || lower.ends_with(".jpg")
                || lower.ends_with(".jpeg")
                || lower.ends_with(".webp")
            {
                run_image_apply(&args[1..]);
            } else {
                run_gradient_demo();
            }
        }
    }
}

/// Applies the physically based film grain simulation to an arbitrary input image file
/// (supports PNG, JPEG, WebP).
///
/// Usage:
///   fgrain apply <input.png|jpg|webp> [output.png|jpg|webp] [--preset tri-x|t-max|hp5] [--color] [--exposure 1.0] [--grain-size 2.5] [--grain-strength 0.28] [--width 512] [--height 512]
fn run_image_apply(args: &[String]) {
    if args.is_empty() {
        println!("\nUsage: fgrain apply <input_image> [output_image] [options]");
        println!("Supported formats: PNG, JPEG, WebP (.png, .jpg, .jpeg, .webp)");
        println!("Options:");
        println!(
            "  --preset <tri-x|tri-x-1600|caffenol|t-max|hp5> Film emulsion stock (default: tri-x)"
        );
        println!("  --format <35mm|120|4x5>      Negative format scaling (default: 35mm)");
        println!(
            "  --push <0..3>                Push processing stops (default: 0.0, or 2.0 for tri-x-1600/caffenol)"
        );
        println!(
            "  --eberhard <float>           Eberhard / Mackie line acutance halo strength (default: 1.0)"
        );
        println!(
            "  --no-scanner-otf             Disable optical scanner lens MTF aperture simulation"
        );
        println!("  --color                      Simulate color film dye-clouds (RGB)");
        println!("  --exposure <float>           Exposure multiplier (default: 1.0)");
        println!(
            "  --grain-size <pixels>        Crystal grain diameter in pixels (default: 2.4 for 35mm Tri-X)"
        );
        println!("  --grain-strength <float>     Grain contrast/intensity (default: 1.0)");
        println!("  --scale <float>              Global grain scale multiplier (default: 1.0)");
        println!(
            "  --auto-scale                 Auto-scale grain size & strength relative to 24MP 35mm scan"
        );
        println!(
            "  --reference-res <pixels>     Reference scan resolution long edge for auto-scale (default: 6000)"
        );
        println!(
            "  --supersample <1-4>          Optical aperture supersampling factor (default: 1)"
        );
        println!("  --tile-size <pixels>         Processing tile dimension (default: 320)");
        println!("  --width <pixels>             Target output width (default: match input)");
        println!("  --height <pixels>            Target output height (default: match input)");
        println!(
            "  --gpu                        Accelerate optical scanning with GPU compute shaders (wgpu)"
        );
        return;
    }

    let input_path = &args[0];
    let mut output_path = if args.len() > 1 && !args[1].starts_with('-') {
        args[1].clone()
    } else {
        "film_grained_output.png".to_string()
    };

    let mut preset = "tri-x".to_string();
    let mut format_arg: Option<String> = None;
    let mut color = false;
    let mut exposure_mult = 1.0;
    let mut grain_size_arg: Option<f64> = None;
    let mut grain_strength_arg: Option<f64> = None;
    let mut scale = 1.0;
    let mut auto_scale = false;
    let mut reference_res = 6000;
    let mut supersample = 1;
    let mut tile_size = 320;
    let mut custom_width: Option<u32> = None;
    let mut custom_height: Option<u32> = None;
    let mut gpu = false;
    let mut push_arg: Option<f64> = None;
    let mut eberhard_arg: Option<f64> = None;
    let mut scanner_otf = true;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-o" | "--output" if i + 1 < args.len() => {
                output_path = args[i + 1].clone();
                i += 2;
            }
            "--preset" if i + 1 < args.len() => {
                preset = args[i + 1].clone();
                i += 2;
            }
            "--format" if i + 1 < args.len() => {
                format_arg = Some(args[i + 1].clone());
                i += 2;
            }
            "--push" if i + 1 < args.len() => {
                if let Ok(v) = args[i + 1].parse::<f64>() {
                    push_arg = Some(v.max(0.0));
                }
                i += 2;
            }
            "--eberhard" if i + 1 < args.len() => {
                if let Ok(v) = args[i + 1].parse::<f64>() {
                    eberhard_arg = Some(v.max(0.0));
                }
                i += 2;
            }
            "--no-scanner-otf" => {
                scanner_otf = false;
                i += 1;
            }
            "--color" => {
                color = true;
                i += 1;
            }
            "--gpu" => {
                gpu = true;
                i += 1;
            }
            "--exposure" if i + 1 < args.len() => {
                if let Ok(v) = args[i + 1].parse::<f64>() {
                    exposure_mult = v;
                }
                i += 2;
            }
            "--grain-size" if i + 1 < args.len() => {
                if let Ok(v) = args[i + 1].parse::<f64>() {
                    grain_size_arg = Some(v);
                }
                i += 2;
            }
            "--grain-strength" if i + 1 < args.len() => {
                if let Ok(v) = args[i + 1].parse::<f64>() {
                    grain_strength_arg = Some(v);
                }
                i += 2;
            }
            "--scale" if i + 1 < args.len() => {
                if let Ok(v) = args[i + 1].parse::<f64>() {
                    scale = v;
                }
                i += 2;
            }
            "--auto-scale" => {
                auto_scale = true;
                i += 1;
            }
            "--reference-res" if i + 1 < args.len() => {
                if let Ok(v) = args[i + 1].parse::<u32>() {
                    reference_res = v;
                }
                i += 2;
            }
            "--supersample" if i + 1 < args.len() => {
                if let Ok(v) = args[i + 1].parse::<u32>() {
                    supersample = v.clamp(1, 4);
                }
                i += 2;
            }
            "--tile-size" if i + 1 < args.len() => {
                if let Ok(v) = args[i + 1].parse::<u32>() {
                    tile_size = v;
                }
                i += 2;
            }
            "--width" if i + 1 < args.len() => {
                if let Ok(v) = args[i + 1].parse::<u32>() {
                    custom_width = Some(v);
                }
                i += 2;
            }
            "--height" if i + 1 < args.len() => {
                if let Ok(v) = args[i + 1].parse::<u32>() {
                    custom_height = Some(v);
                }
                i += 2;
            }
            _ => {
                i += 1;
            }
        }
    }

    let is_push_preset =
        preset.to_lowercase().contains("1600") || preset.to_lowercase() == "caffenol";
    let default_push = if is_push_preset { 2.0 } else { 0.0 };
    let push = push_arg.unwrap_or(default_push);
    let default_strength = if is_push_preset || push >= 1.5 {
        1.25
    } else {
        1.0
    };
    let grain_strength = grain_strength_arg.unwrap_or(default_strength);
    let default_eberhard = 0.0;
    let eberhard = eberhard_arg.unwrap_or(default_eberhard);

    let format_scale = match format_arg
        .as_deref()
        .unwrap_or("35mm")
        .to_lowercase()
        .as_str()
    {
        "120" | "medium" => 0.55,
        "4x5" | "large" => 0.28,
        _ => 1.0, // default 35mm roll film
    };

    let base_preset_grain = match preset.to_lowercase().as_str() {
        "t-max" | "tmax" => 1.1,
        "hp5" | "hp5-plus" => 1.8,
        "tri-x-1600" | "trix-1600" | "caffenol" => 2.2,
        _ => {
            if push >= 1.5 {
                2.2
            } else {
                2.4 // Kodak Tri-X 400
            }
        }
    };

    let grain_size = match grain_size_arg {
        Some(v) => {
            if v < 0.05 {
                println!(
                    "[Notice] Grain size {:.2}px clamped to 0.05px minimum limit",
                    v
                );
                0.05
            } else {
                v
            }
        }
        None => base_preset_grain * format_scale,
    };

    let out_dims = match (custom_width, custom_height) {
        (Some(w), Some(h)) => Some((w, h)),
        (Some(w), None) => Some((w, 0)),
        (None, Some(h)) => Some((0, h)),
        (None, None) => None,
    };

    println!("\n[Input]  {}", input_path);
    println!("[Output] {}", output_path);
    println!(
        "[Preset] {} ({})",
        preset,
        if color {
            "Color Multi-Layer"
        } else {
            "B&W Panchromatic"
        }
    );
    if gpu {
        if let Some(engine) = fgrain::get_gpu_engine() {
            println!("[Acceleration] GPU ({})", engine.adapter_name());
        } else {
            println!(
                "[Acceleration] GPU requested but adapter unavailable, falling back to CPU Rayon"
            );
        }
    } else {
        println!("[Acceleration] CPU Multi-threaded (Rayon)");
    }
    println!(
        "[Format] {} | [Grain Size] {:.2}px | [Grain Strength] {:.2} | [Scale] {:.2}x",
        format_arg.as_deref().unwrap_or("35mm"),
        grain_size,
        grain_strength,
        scale
    );
    if push > 0.0 {
        println!(
            "[Push Processing] +{:.1} stops | [Eberhard Adjacency] {:.2}x | [Scanner OTF] {}",
            push,
            eberhard,
            if scanner_otf {
                "Active (Epson V850 MTF)"
            } else {
                "Disabled"
            }
        );
    } else {
        println!(
            "[Eberhard Adjacency] {:.2}x | [Scanner OTF] {}",
            eberhard,
            if scanner_otf {
                "Active (Epson V850 MTF)"
            } else {
                "Disabled"
            }
        );
    }
    if auto_scale {
        println!(
            "[Auto-Scale] Active (relative to {}px 35mm scan)",
            reference_res
        );
    }
    if supersample > 1 {
        println!(
            "[Supersampling] {}x optical scanner aperture integration",
            supersample
        );
    }
    println!("[Exposure Scale] {:.2}x", exposure_mult);

    let options = fgrain::FilmRenderOptions {
        preset,
        grain_size_px: grain_size,
        grain_strength,
        exposure: exposure_mult,
        color,
        output_resolution: out_dims,
        tile_size,
        seed: 42,
        gpu,
        auto_scale,
        scale,
        reference_res,
        supersample,
        push,
        eberhard,
        scanner_otf,
    };

    println!("\nProcessing image tiles...");
    match process_image_file(input_path, &output_path, &options) {
        Ok(report) => {
            println!("\nRender complete:");
            println!(
                "   Input Resolution:  {}x{}",
                report.input_resolution.0, report.input_resolution.1
            );
            println!(
                "   Output Resolution: {}x{}",
                report.output_resolution.0, report.output_resolution.1
            );
            println!("   Total Tiles:       {}", report.total_tiles);
            println!("   Total Crystals:    {}", report.total_crystals);
            println!(
                "   Developed Silver:  {} ({:.1}%)",
                report.developed_crystals,
                (report.developed_crystals as f64 / report.total_crystals.max(1) as f64) * 100.0
            );
            println!("   Silver Clumps:     {}", report.total_clumps);
            println!("   Processing Time:   {:.2}s", report.elapsed_seconds);
            println!("\nSaved output to: {}", output_path);
        }
        Err(e) => {
            eprintln!("\nError processing image {}: {}", input_path, e);
        }
    }
}

/// Runs a continuous exposure gradient demonstrating the transition from
/// granular midtones to dense clumping in highlights.
fn run_gradient_demo() {
    println!("\n[Synthesizing Emulsion] Kodak Tri-X 400...");
    let patch_w_um = 60.0;
    let patch_h_um = 60.0;
    let out_res = 384;

    let total_start = Instant::now();
    let config = EmulsionConfig::tri_x_400(patch_w_um, patch_h_um);

    let synth_start = Instant::now();
    let mut emulsion = VirtualEmulsionLayer::synthesize(config, 42);
    println!(
        "      Synthesized {} crystals across {:.1}x{:.1}x{:.1} um volume in {:.2?}",
        emulsion.crystals.len(),
        patch_w_um,
        patch_h_um,
        emulsion.config.dimensions.z,
        synth_start.elapsed()
    );

    println!("\n[Photon Transport] Gradient exposure...");
    let exp_config = ExposureConfig {
        photons_per_sq_um: 6.0,
        illumination_angle_rad: 0.12,
        crystal_absorption_coeff: 1.3,
        max_bounces: 48,
    };

    let exp_start = Instant::now();
    // Smooth horizontal exposure ramp from shadow (0.02) to specular highlight (1.0)
    expose_emulsion(
        &emulsion,
        |x, _y| {
            let t = (x / patch_w_um).clamp(0.0, 1.0);
            0.02 + 0.98 * t.powf(1.4)
        },
        &exp_config,
        1001,
    );
    println!(
        "      Photon transport completed in {:.2?}",
        exp_start.elapsed()
    );

    println!("\n[Chemical Development] Latent image reduction and clumping...");
    let dev_config = DevelopmentConfig {
        threshold_photons: 4,
        filament_expansion: 1.35,
        clumping_proximity_factor: 1.25,
        developer_contrast: 0.85,
    };

    let dev_start = Instant::now();
    let report = develop_emulsion(&mut emulsion.crystals, &emulsion.bvh, &dev_config);
    println!(
        "      Developed: {} / {} crystals ({:.1}%)",
        report.developed_crystals,
        report.total_crystals,
        (report.developed_crystals as f64 / report.total_crystals as f64) * 100.0
    );
    println!(
        "      Clumps: {} | Max clump size: {} grains | Avg clump: {:.2} grains",
        report.total_clumps, report.max_clump_size, report.avg_clump_size
    );
    println!(
        "      Chemical development completed in {:.2?}",
        dev_start.elapsed()
    );

    println!(
        "\n[Optical Scan] Transmittance scan ({}x{} px)...",
        out_res, out_res
    );
    let scan_config = ScanConfig {
        width_px: out_res,
        height_px: out_res,
        samples_per_pixel: 4,
        mode: ScanMode::PositivePrint,
        silver_opacity: 0.40,
    };

    let scan_start = Instant::now();
    let (_buffer, image) = scan_emulsion(&emulsion, &scan_config, 2002);
    println!(
        "      Optical scanning completed in {:.2?}",
        scan_start.elapsed()
    );

    let out_filename = "film_grain_gradient.png";
    image
        .save(out_filename)
        .expect("Failed to save output image");
    println!("\nRender complete. Saved output to: {}", out_filename);
    println!("Total runtime: {:.2?}", total_start.elapsed());
}

/// Runs a halation demonstration: intense central highlight with anti-halation backing reflections.
fn run_halation_demo() {
    println!("\n[Halation Demo] Sub-surface scattering and anti-halation reflection...");
    let patch_size = 50.0;
    let out_res = 256;

    let config = EmulsionConfig::tri_x_400(patch_size, patch_size);
    let mut emulsion = VirtualEmulsionLayer::synthesize(config, 777);

    let exp_config = ExposureConfig {
        photons_per_sq_um: 8.0,
        illumination_angle_rad: 0.15,
        crystal_absorption_coeff: 1.0,
        max_bounces: 64,
    };

    // Intense point highlight at center
    let center_x = patch_size * 0.5;
    let center_y = patch_size * 0.5;
    expose_emulsion(
        &emulsion,
        |x, y| {
            let dx = x - center_x;
            let dy = y - center_y;
            let dist = (dx * dx + dy * dy).sqrt();
            if dist < 4.0 {
                1.2 // Intense specular core
            } else {
                0.04 // Dim background
            }
        },
        &exp_config,
        888,
    );

    let dev_config = DevelopmentConfig {
        threshold_photons: 4,
        filament_expansion: 1.35,
        clumping_proximity_factor: 1.25,
        developer_contrast: 0.85,
    };
    let report = develop_emulsion(&mut emulsion.crystals, &emulsion.bvh, &dev_config);
    println!(
        "      Developed {} crystals into {} clumps (max clump: {})",
        report.developed_crystals, report.total_clumps, report.max_clump_size
    );

    let scan_config = ScanConfig {
        width_px: out_res,
        height_px: out_res,
        samples_per_pixel: 4,
        mode: ScanMode::PositivePrint,
        silver_opacity: 0.40,
    };

    let (_buffer, image) = scan_emulsion(&emulsion, &scan_config, 999);
    let out_filename = "film_grain_halation.png";
    image
        .save(out_filename)
        .expect("Failed to save halation image");
    println!("Saved halation test output to: {}", out_filename);
}

/// Runs a preset comparison demo (e.g. tri-x vs t-max).
fn run_preset_demo(preset_name: &str) {
    let patch_size = 50.0;
    let out_res = 256;

    let config = if preset_name == "t-max" {
        EmulsionConfig::t_max_100(patch_size, patch_size)
    } else {
        EmulsionConfig::tri_x_400(patch_size, patch_size)
    };

    println!("\n[Preset Demo: {}] Synthesizing emulsion...", preset_name);
    let mut emulsion = VirtualEmulsionLayer::synthesize(config, 555);

    let exp_config = ExposureConfig {
        photons_per_sq_um: 6.0,
        illumination_angle_rad: 0.12,
        crystal_absorption_coeff: 1.3,
        max_bounces: 48,
    };
    expose_emulsion(&emulsion, |_x, _y| 0.45, &exp_config, 666);

    let dev_config = DevelopmentConfig {
        threshold_photons: 4,
        filament_expansion: 1.35,
        clumping_proximity_factor: 1.25,
        developer_contrast: 0.85,
    };
    let report = develop_emulsion(&mut emulsion.crystals, &emulsion.bvh, &dev_config);
    println!(
        "      Developed {} crystals in {} clumps",
        report.developed_crystals, report.total_clumps
    );

    let scan_config = ScanConfig {
        width_px: out_res,
        height_px: out_res,
        samples_per_pixel: 4,
        mode: ScanMode::PositivePrint,
        silver_opacity: 0.40,
    };

    let (_buffer, image) = scan_emulsion(&emulsion, &scan_config, 777);
    let out_filename = format!("film_grain_{}.png", preset_name);
    image
        .save(&out_filename)
        .expect("Failed to save preset image");
    println!("Saved preset output to: {}", out_filename);
}
