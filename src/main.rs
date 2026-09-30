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
    let args: Vec<String> = env::args().collect();
    if let Err(err) = run_cli(&args) {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}

/// Routes CLI arguments to the appropriate command handler.
pub fn run_cli(args: &[String]) -> Result<(), String> {
    if args.len() <= 1 {
        print_main_help();
        return Ok(());
    }

    let first = &args[1];
    match first.as_str() {
        "-h" | "--help" => {
            if args.len() > 2 {
                match args[2].as_str() {
                    "apply" | "image" => print_apply_help(),
                    "gradient" => print_gradient_help(),
                    "tri-x" => print_tri_x_help(),
                    "t-max" => print_t_max_help(),
                    "halation" => print_halation_help(),
                    "demo" => print_demo_help(),
                    _ => print_main_help(),
                }
            } else {
                print_main_help();
            }
            Ok(())
        }
        "help" => {
            if args.len() > 2 {
                match args[2].as_str() {
                    "apply" | "image" => print_apply_help(),
                    "gradient" => print_gradient_help(),
                    "tri-x" => print_tri_x_help(),
                    "t-max" => print_t_max_help(),
                    "halation" => print_halation_help(),
                    "demo" => print_demo_help(),
                    other => {
                        return Err(format!(
                            "unknown help topic '{other}'. For a list of commands, try 'fgrain --help'."
                        ));
                    }
                }
            } else {
                print_main_help();
            }
            Ok(())
        }
        "apply" | "image" => run_image_apply(&args[2..]),
        "gradient" => run_gradient_demo_cli(&args[2..]),
        "tri-x" => run_preset_demo_cli("tri-x", &args[2..]),
        "t-max" => run_preset_demo_cli("t-max", &args[2..]),
        "halation" => run_halation_demo_cli(&args[2..]),
        "demo" => run_demo_cli(&args[2..]),
        other => {
            if is_image_file(other) {
                run_image_apply(&args[1..])
            } else if other.starts_with('-') {
                Err(format!(
                    "unrecognized option '{other}'. For more information, try '--help'."
                ))
            } else {
                Err(format!(
                    "unrecognized command '{other}'. For more information, try '--help'."
                ))
            }
        }
    }
}

/// Checks if a file path has an image file extension supported by fgrain.
pub fn is_image_file(path: &str) -> bool {
    let lower = path.to_lowercase();
    lower.ends_with(".png")
        || lower.ends_with(".jpg")
        || lower.ends_with(".jpeg")
        || lower.ends_with(".webp")
}

/// Prints top-level CLI help.
pub fn print_main_help() {
    println!("fgrain: Physically based film grain simulation engine");
    println!();
    println!("Usage:");
    println!("  fgrain <command> [arguments] [options]");
    println!("  fgrain <input_image> [output_image] [options]");
    println!();
    println!("Commands:");
    println!("  apply <input> [output]    Apply physically based film grain to an image");
    println!("  gradient [output]         Render continuous exposure ramp demonstration");
    println!("  tri-x [output]            Render Kodak Tri-X 400 preset demonstration");
    println!(
        "  t-max [output]            Render Kodak T-Max 100 tabular grain preset demonstration"
    );
    println!("  halation [output]         Render point-source halation demonstration");
    println!("  demo <name> [output]      Run a demonstration (gradient, tri-x, t-max, halation)");
    println!("  help [command]            Print this message or help for a given command");
    println!();
    println!("Options:");
    println!("  -h, --help                Print help information");
    println!();
    println!("Run 'fgrain apply --help' for image processing options.");
}

/// Prints help for the `apply` command.
pub fn print_apply_help() {
    println!("fgrain apply: Apply physically based film grain to an input image");
    println!();
    println!("Usage:");
    println!("  fgrain apply <input_image> [output_image] [options]");
    println!("  fgrain <input_image> [output_image] [options]");
    println!();
    println!("Supported formats:");
    println!("  PNG, JPEG, WebP (.png, .jpg, .jpeg, .webp)");
    println!();
    println!("Arguments:");
    println!("  <input_image>                 Path to input image file");
    println!(
        "  [output_image]                Path to output image file (default: film_grained_output.png)"
    );
    println!();
    println!("Options:");
    println!("  -o, --output <path>           Output image path");
    println!(
        "  -p, --preset <stock>          Emulsion stock: tri-x, tri-x-1600, caffenol, t-max, hp5 (default: tri-x)"
    );
    println!("  -f, --format <format>         Negative format: 35mm, 120, 4x5 (default: 35mm)");
    println!(
        "      --push <stops>            Push processing stops: 0.0 to 3.0 (default: 0.0, or 2.0 for tri-x-1600/caffenol)"
    );
    println!(
        "      --eberhard <float>        Eberhard / Mackie line acutance halo strength (default: 0.0)"
    );
    println!(
        "      --no-scanner-otf          Disable optical scanner lens MTF aperture simulation"
    );
    println!("  -c, --color                   Simulate color film dye-clouds (RGB)");
    println!("  -e, --exposure <float>        Exposure multiplier (default: 1.0)");
    println!(
        "  -s, --grain-size <pixels>     Crystal grain diameter in pixels (default: 2.4 for 35mm Tri-X)"
    );
    println!("      --grain-strength <float>  Grain contrast/intensity multiplier (default: 1.0)");
    println!("      --scale <float>           Global grain scale multiplier (default: 1.0)");
    println!(
        "      --auto-scale              Auto-scale grain size & strength relative to 24MP 35mm scan"
    );
    println!(
        "      --reference-res <pixels>  Reference scan resolution long edge for auto-scale (default: 6000)"
    );
    println!("      --supersample <1-4>       Optical aperture supersampling factor (default: 1)");
    println!("      --tile-size <pixels>      Processing tile dimension (default: 320)");
    println!("  -w, --width <pixels>          Target output width (default: match input)");
    println!("      --height <pixels>         Target output height (default: match input)");
    println!(
        "      --gpu                     Accelerate optical scanning with GPU compute shaders (wgpu)"
    );
    println!("  -h, --help                    Print help information");
}

/// Prints help for demo commands.
pub fn print_demo_help() {
    println!("fgrain demo: Run physical film grain simulation demonstrations");
    println!();
    println!("Usage:");
    println!("  fgrain demo <gradient|tri-x|t-max|halation> [output_image] [options]");
    println!("  fgrain <gradient|tri-x|t-max|halation> [output_image] [options]");
    println!();
    println!("Demonstrations:");
    println!("  gradient    Continuous exposure ramp from shadow to highlight");
    println!("  tri-x       Kodak Tri-X 400 cubic grain emulsion demo");
    println!("  t-max       Kodak T-Max 100 tabular grain emulsion demo");
    println!("  halation    Sub-surface scattering and anti-halation reflection demo");
    println!();
    println!("Options:");
    println!("  -o, --output <path>  Output image path");
    println!("  -h, --help           Print help information");
}

/// Prints help for the `gradient` demo.
pub fn print_gradient_help() {
    println!("fgrain gradient: Render continuous exposure ramp demonstration");
    println!();
    println!("Usage:");
    println!("  fgrain gradient [output_image] [options]");
    println!();
    println!("Arguments:");
    println!("  [output_image]       Output filename (default: film_grain_gradient.png)");
    println!();
    println!("Options:");
    println!("  -o, --output <path>  Output image path");
    println!("  -h, --help           Print help information");
}

/// Prints help for the `tri-x` demo.
pub fn print_tri_x_help() {
    println!("fgrain tri-x: Render Kodak Tri-X 400 preset demonstration");
    println!();
    println!("Usage:");
    println!("  fgrain tri-x [output_image] [options]");
    println!();
    println!("Arguments:");
    println!("  [output_image]       Output filename (default: film_grain_tri-x.png)");
    println!();
    println!("Options:");
    println!("  -o, --output <path>  Output image path");
    println!("  -h, --help           Print help information");
}

/// Prints help for the `t-max` demo.
pub fn print_t_max_help() {
    println!("fgrain t-max: Render Kodak T-Max 100 tabular grain preset demonstration");
    println!();
    println!("Usage:");
    println!("  fgrain t-max [output_image] [options]");
    println!();
    println!("Arguments:");
    println!("  [output_image]       Output filename (default: film_grain_t-max.png)");
    println!();
    println!("Options:");
    println!("  -o, --output <path>  Output image path");
    println!("  -h, --help           Print help information");
}

/// Prints help for the `halation` demo.
pub fn print_halation_help() {
    println!("fgrain halation: Render point-source halation demonstration");
    println!();
    println!("Usage:");
    println!("  fgrain halation [output_image] [options]");
    println!();
    println!("Arguments:");
    println!("  [output_image]       Output filename (default: film_grain_halation.png)");
    println!();
    println!("Options:");
    println!("  -o, --output <path>  Output image path");
    println!("  -h, --help           Print help information");
}

/// Applies the physically based film grain simulation to an arbitrary input image file.
fn run_image_apply(args: &[String]) -> Result<(), String> {
    if args.is_empty() || args.iter().any(|a| a == "-h" || a == "--help") {
        print_apply_help();
        if args.is_empty() {
            return Err("missing required argument <input_image>\n\nFor more information, try 'fgrain apply --help'.".to_string());
        }
        return Ok(());
    }

    let mut input_path: Option<String> = None;
    let mut output_path: Option<String> = None;
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

    let mut i = 0;
    while i < args.len() {
        let arg = &args[i];
        match arg.as_str() {
            "-o" | "--output" => {
                i += 1;
                if i >= args.len() {
                    return Err(format!("option '{arg}' requires an argument"));
                }
                output_path = Some(args[i].clone());
            }
            "-p" | "--preset" => {
                i += 1;
                if i >= args.len() {
                    return Err(format!("option '{arg}' requires an argument"));
                }
                preset = args[i].clone();
            }
            "-f" | "--format" => {
                i += 1;
                if i >= args.len() {
                    return Err(format!("option '{arg}' requires an argument"));
                }
                format_arg = Some(args[i].clone());
            }
            "--push" => {
                i += 1;
                if i >= args.len() {
                    return Err(format!("option '{arg}' requires an argument"));
                }
                let val: f64 = args[i].parse().map_err(|_| {
                    format!(
                        "invalid value '{}' for '--push': expected a positive number",
                        args[i]
                    )
                })?;
                push_arg = Some(val.max(0.0));
            }
            "--eberhard" => {
                i += 1;
                if i >= args.len() {
                    return Err(format!("option '{arg}' requires an argument"));
                }
                let val: f64 = args[i].parse().map_err(|_| {
                    format!(
                        "invalid value '{}' for '--eberhard': expected a number",
                        args[i]
                    )
                })?;
                eberhard_arg = Some(val.max(0.0));
            }
            "--no-scanner-otf" => {
                scanner_otf = false;
            }
            "-c" | "--color" => {
                color = true;
            }
            "--gpu" => {
                gpu = true;
            }
            "-e" | "--exposure" => {
                i += 1;
                if i >= args.len() {
                    return Err(format!("option '{arg}' requires an argument"));
                }
                let val: f64 = args[i].parse().map_err(|_| {
                    format!("invalid value '{}' for '{arg}': expected a number", args[i])
                })?;
                exposure_mult = val;
            }
            "-s" | "--grain-size" => {
                i += 1;
                if i >= args.len() {
                    return Err(format!("option '{arg}' requires an argument"));
                }
                let val: f64 = args[i].parse().map_err(|_| {
                    format!("invalid value '{}' for '{arg}': expected a number", args[i])
                })?;
                grain_size_arg = Some(val);
            }
            "--grain-strength" => {
                i += 1;
                if i >= args.len() {
                    return Err(format!("option '{arg}' requires an argument"));
                }
                let val: f64 = args[i].parse().map_err(|_| {
                    format!("invalid value '{}' for '{arg}': expected a number", args[i])
                })?;
                grain_strength_arg = Some(val);
            }
            "--scale" => {
                i += 1;
                if i >= args.len() {
                    return Err(format!("option '{arg}' requires an argument"));
                }
                let val: f64 = args[i].parse().map_err(|_| {
                    format!("invalid value '{}' for '{arg}': expected a number", args[i])
                })?;
                scale = val;
            }
            "--auto-scale" => {
                auto_scale = true;
            }
            "--reference-res" => {
                i += 1;
                if i >= args.len() {
                    return Err(format!("option '{arg}' requires an argument"));
                }
                let val: u32 = args[i].parse().map_err(|_| {
                    format!(
                        "invalid value '{}' for '{arg}': expected an integer",
                        args[i]
                    )
                })?;
                reference_res = val;
            }
            "--supersample" => {
                i += 1;
                if i >= args.len() {
                    return Err(format!("option '{arg}' requires an argument"));
                }
                let val: u32 = args[i].parse().map_err(|_| {
                    format!(
                        "invalid value '{}' for '{arg}': expected an integer (1-4)",
                        args[i]
                    )
                })?;
                supersample = val.clamp(1, 4);
            }
            "--tile-size" => {
                i += 1;
                if i >= args.len() {
                    return Err(format!("option '{arg}' requires an argument"));
                }
                let val: u32 = args[i].parse().map_err(|_| {
                    format!(
                        "invalid value '{}' for '{arg}': expected an integer",
                        args[i]
                    )
                })?;
                tile_size = val;
            }
            "-w" | "--width" => {
                i += 1;
                if i >= args.len() {
                    return Err(format!("option '{arg}' requires an argument"));
                }
                let val: u32 = args[i].parse().map_err(|_| {
                    format!(
                        "invalid value '{}' for '{arg}': expected an integer",
                        args[i]
                    )
                })?;
                custom_width = Some(val);
            }
            "--height" => {
                i += 1;
                if i >= args.len() {
                    return Err(format!("option '{arg}' requires an argument"));
                }
                let val: u32 = args[i].parse().map_err(|_| {
                    format!(
                        "invalid value '{}' for '{arg}': expected an integer",
                        args[i]
                    )
                })?;
                custom_height = Some(val);
            }
            other if other.starts_with('-') => {
                return Err(format!(
                    "unrecognized option '{other}'. For more information, try 'fgrain apply --help'."
                ));
            }
            positional => {
                if input_path.is_none() {
                    input_path = Some(positional.to_string());
                } else if output_path.is_none() {
                    output_path = Some(positional.to_string());
                } else {
                    return Err(format!(
                        "unexpected positional argument '{positional}'. For more information, try 'fgrain apply --help'."
                    ));
                }
            }
        }
        i += 1;
    }

    let input_path = match input_path {
        Some(p) => p,
        None => {
            return Err("missing required argument <input_image>\n\nUsage: fgrain apply <input_image> [output_image] [options]\nFor more information, try 'fgrain apply --help'.".to_string());
        }
    };
    let output_path = output_path.unwrap_or_else(|| "film_grained_output.png".to_string());

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
                if v > 4.5 {
                    println!(
                        "[Notice] Grain size {:.2}px exceeds standard 35mm physical scale (~0.8-2.5px); rendering as extreme macro / sub-miniature crop",
                        v
                    );
                }
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

    println!("[Input]  {}", input_path);
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
    match process_image_file(&input_path, &output_path, &options) {
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
            Ok(())
        }
        Err(e) => Err(format!("failed to process image '{input_path}': {e}")),
    }
}

/// Runs demo command dispatch.
fn run_demo_cli(args: &[String]) -> Result<(), String> {
    if args.is_empty() || args.iter().any(|a| a == "-h" || a == "--help") {
        print_demo_help();
        return Ok(());
    }

    let demo_name = &args[0];
    let rest = &args[1..];
    match demo_name.as_str() {
        "gradient" => run_gradient_demo_cli(rest),
        "tri-x" => run_preset_demo_cli("tri-x", rest),
        "t-max" => run_preset_demo_cli("t-max", rest),
        "halation" => run_halation_demo_cli(rest),
        other => Err(format!(
            "unknown demo '{other}'. Available demos: gradient, tri-x, t-max, halation.\nFor more information, try 'fgrain demo --help'."
        )),
    }
}

/// CLI runner for gradient demo.
fn run_gradient_demo_cli(args: &[String]) -> Result<(), String> {
    if args.iter().any(|a| a == "-h" || a == "--help") {
        print_gradient_help();
        return Ok(());
    }

    let mut output_path: Option<String> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-o" | "--output" => {
                i += 1;
                if i >= args.len() {
                    return Err(format!("option '{}' requires an argument", args[i - 1]));
                }
                output_path = Some(args[i].clone());
            }
            other if other.starts_with('-') => {
                return Err(format!(
                    "unrecognized option '{other}' for 'gradient'. For more information, try 'fgrain gradient --help'."
                ));
            }
            positional => {
                if output_path.is_none() {
                    output_path = Some(positional.to_string());
                } else {
                    return Err(format!(
                        "unexpected positional argument '{positional}'. For more information, try 'fgrain gradient --help'."
                    ));
                }
            }
        }
        i += 1;
    }

    run_gradient_demo(output_path.as_deref())
}

/// CLI runner for preset demos (tri-x, t-max).
fn run_preset_demo_cli(preset_name: &str, args: &[String]) -> Result<(), String> {
    if args.iter().any(|a| a == "-h" || a == "--help") {
        match preset_name {
            "t-max" => print_t_max_help(),
            _ => print_tri_x_help(),
        }
        return Ok(());
    }

    let mut output_path: Option<String> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-o" | "--output" => {
                i += 1;
                if i >= args.len() {
                    return Err(format!("option '{}' requires an argument", args[i - 1]));
                }
                output_path = Some(args[i].clone());
            }
            other if other.starts_with('-') => {
                return Err(format!(
                    "unrecognized option '{other}' for '{preset_name}'. For more information, try 'fgrain {preset_name} --help'."
                ));
            }
            positional => {
                if output_path.is_none() {
                    output_path = Some(positional.to_string());
                } else {
                    return Err(format!(
                        "unexpected positional argument '{positional}'. For more information, try 'fgrain {preset_name} --help'."
                    ));
                }
            }
        }
        i += 1;
    }

    run_preset_demo(preset_name, output_path.as_deref())
}

/// CLI runner for halation demo.
fn run_halation_demo_cli(args: &[String]) -> Result<(), String> {
    if args.iter().any(|a| a == "-h" || a == "--help") {
        print_halation_help();
        return Ok(());
    }

    let mut output_path: Option<String> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-o" | "--output" => {
                i += 1;
                if i >= args.len() {
                    return Err(format!("option '{}' requires an argument", args[i - 1]));
                }
                output_path = Some(args[i].clone());
            }
            other if other.starts_with('-') => {
                return Err(format!(
                    "unrecognized option '{other}' for 'halation'. For more information, try 'fgrain halation --help'."
                ));
            }
            positional => {
                if output_path.is_none() {
                    output_path = Some(positional.to_string());
                } else {
                    return Err(format!(
                        "unexpected positional argument '{positional}'. For more information, try 'fgrain halation --help'."
                    ));
                }
            }
        }
        i += 1;
    }

    run_halation_demo(output_path.as_deref())
}

/// Runs a continuous exposure gradient demonstrating the transition from
/// granular midtones to dense clumping in highlights.
fn run_gradient_demo(output_path: Option<&str>) -> Result<(), String> {
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
        (report.developed_crystals as f64 / report.total_crystals.max(1) as f64) * 100.0
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

    let out_filename = output_path.unwrap_or("film_grain_gradient.png");
    image
        .save(out_filename)
        .map_err(|e| format!("failed to save output image to '{out_filename}': {e}"))?;
    println!("\nRender complete. Saved output to: {}", out_filename);
    println!("Total runtime: {:.2?}", total_start.elapsed());
    Ok(())
}

/// Runs a halation demonstration: intense central highlight with anti-halation backing reflections.
fn run_halation_demo(output_path: Option<&str>) -> Result<(), String> {
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
    let out_filename = output_path.unwrap_or("film_grain_halation.png");
    image
        .save(out_filename)
        .map_err(|e| format!("failed to save halation image to '{out_filename}': {e}"))?;
    println!("Saved halation test output to: {}", out_filename);
    Ok(())
}

/// Runs a preset comparison demo (e.g. tri-x vs t-max).
fn run_preset_demo(preset_name: &str, output_path: Option<&str>) -> Result<(), String> {
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
    let default_name = format!("film_grain_{}.png", preset_name);
    let out_filename = output_path.unwrap_or(&default_name);
    image
        .save(out_filename)
        .map_err(|e| format!("failed to save preset image to '{out_filename}': {e}"))?;
    println!("Saved preset output to: {}", out_filename);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_help_flags() {
        assert!(run_cli(&["fgrain".into()]).is_ok());
        assert!(run_cli(&["fgrain".into(), "-h".into()]).is_ok());
        assert!(run_cli(&["fgrain".into(), "--help".into()]).is_ok());
        assert!(run_cli(&["fgrain".into(), "help".into()]).is_ok());
        assert!(run_cli(&["fgrain".into(), "help".into(), "apply".into()]).is_ok());
        assert!(run_cli(&["fgrain".into(), "help".into(), "gradient".into()]).is_ok());
        assert!(run_cli(&["fgrain".into(), "help".into(), "tri-x".into()]).is_ok());
        assert!(run_cli(&["fgrain".into(), "help".into(), "t-max".into()]).is_ok());
        assert!(run_cli(&["fgrain".into(), "help".into(), "halation".into()]).is_ok());
        assert!(run_cli(&["fgrain".into(), "help".into(), "demo".into()]).is_ok());
        assert!(run_cli(&["fgrain".into(), "apply".into(), "-h".into()]).is_ok());
        assert!(run_cli(&["fgrain".into(), "apply".into(), "--help".into()]).is_ok());
        assert!(run_cli(&["fgrain".into(), "gradient".into(), "-h".into()]).is_ok());
        assert!(run_cli(&["fgrain".into(), "tri-x".into(), "-h".into()]).is_ok());
        assert!(run_cli(&["fgrain".into(), "t-max".into(), "-h".into()]).is_ok());
        assert!(run_cli(&["fgrain".into(), "halation".into(), "-h".into()]).is_ok());
        assert!(run_cli(&["fgrain".into(), "demo".into(), "-h".into()]).is_ok());
        assert!(run_cli(&["fgrain".into(), "test.png".into(), "-h".into()]).is_ok());
    }

    #[test]
    fn test_invalid_commands_and_options() {
        let err = run_cli(&["fgrain".into(), "unknown_command".into()]).unwrap_err();
        assert!(err.contains("unrecognized command 'unknown_command'"));

        let err = run_cli(&["fgrain".into(), "--unknown_opt".into()]).unwrap_err();
        assert!(err.contains("unrecognized option '--unknown_opt'"));

        let err = run_cli(&["fgrain".into(), "help".into(), "unknown_topic".into()]).unwrap_err();
        assert!(err.contains("unknown help topic 'unknown_topic'"));

        let err = run_cli(&["fgrain".into(), "apply".into()]).unwrap_err();
        assert!(err.contains("missing required argument"));

        let err = run_cli(&[
            "fgrain".into(),
            "apply".into(),
            "test.png".into(),
            "--unknown_flag".into(),
        ])
        .unwrap_err();
        assert!(err.contains("unrecognized option '--unknown_flag'"));

        let err = run_cli(&[
            "fgrain".into(),
            "apply".into(),
            "test.png".into(),
            "--preset".into(),
        ])
        .unwrap_err();
        assert!(err.contains("option '--preset' requires an argument"));

        let err = run_cli(&[
            "fgrain".into(),
            "apply".into(),
            "test.png".into(),
            "--push".into(),
            "not_a_number".into(),
        ])
        .unwrap_err();
        assert!(err.contains("invalid value 'not_a_number' for '--push'"));
    }

    #[test]
    fn test_is_image_file() {
        assert!(is_image_file("photo.PNG"));
        assert!(is_image_file("scan.jpg"));
        assert!(is_image_file("sample.jpeg"));
        assert!(is_image_file("input.webp"));
        assert!(!is_image_file("gradient"));
        assert!(!is_image_file("apply"));
        assert!(!is_image_file("tri-x"));
    }
}
