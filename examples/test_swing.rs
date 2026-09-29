use fgrain::*;
use image::{GrayImage, Luma};
use std::time::Instant;

fn main() {
    let input_path = if std::path::Path::new("test_assets/crop_test_input.png").exists() {
        "test_assets/crop_test_input.png"
    } else {
        "crop_test_input.png"
    };
    println!("Testing on highlight region (swing post & lamp)...");
    let start = Instant::now();

    let input_img = image::open(input_path).unwrap().to_luma8();

    // Crop around top-right corner where the white swing post is located (x: 700..1024, y: 0..400)
    let swing_crop = image::imageops::crop_imm(&input_img, 700, 0, 324, 400).to_image();
    let swing_in_path = if std::path::Path::new("test_assets").exists() {
        "test_assets/swing_input.png"
    } else {
        "swing_input.png"
    };
    swing_crop.save(swing_in_path).unwrap();

    let w = 324;
    let h = 400;

    let pixel_size_um = 0.40; // ~2.5 pixels per crystal
    let patch_w_um = (w as f64) * pixel_size_um;
    let patch_h_um = (h as f64) * pixel_size_um;

    let config = EmulsionConfig::tri_x_400(patch_w_um, patch_h_um);
    let mut emulsion = VirtualEmulsionLayer::synthesize(config, 1234);
    println!(
        "Synthesized {} crystals in {:.1}x{:.1} um",
        emulsion.crystals.len(),
        patch_w_um,
        patch_h_um
    );

    let exp_config = ExposureConfig {
        photons_per_sq_um: 140.0,
        illumination_angle_rad: 0.12,
        crystal_absorption_coeff: 1.3,
        max_bounces: 48,
    };

    expose_emulsion(
        &emulsion,
        |x, y| {
            let u = (x / patch_w_um).clamp(0.0, 1.0);
            let v = (y / patch_h_um).clamp(0.0, 1.0);
            let px = (u * (w - 1) as f64) as u32;
            let py = (v * (h - 1) as f64) as u32;
            let val = swing_crop.get_pixel(px, py).0[0] as f64 / 255.0;
            val.powf(2.0) * 1.5
        },
        &exp_config,
        5678,
    );

    let dev_config = DevelopmentConfig {
        threshold_photons: 3,
        filament_expansion: 1.8,
        clumping_proximity_factor: 1.25,
        developer_contrast: 0.8,
    };

    let dev_report = develop_emulsion(&mut emulsion.crystals, &emulsion.bvh, &dev_config);
    println!(
        "Developed {} crystals in {} clumps (max clump: {})",
        dev_report.developed_crystals, dev_report.total_clumps, dev_report.max_clump_size
    );

    let scan_config = ScanConfig {
        width_px: w,
        height_px: h,
        samples_per_pixel: 16,
        mode: ScanMode::NegativeTransmittance,
        silver_opacity: 2.5,
    };

    let (transmittance_buf, _raw_img) = scan_emulsion(&emulsion, &scan_config, 9012);

    let mean_t: f64 =
        transmittance_buf.iter().map(|&v| v as f64).sum::<f64>() / (transmittance_buf.len() as f64);
    println!("Mean transmittance across patch: {:.3}", mean_t);

    let mut output_img = GrayImage::new(w, h);
    let grain_strength = 0.28;

    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) as usize;
            let base_val = swing_crop.get_pixel(x, y).0[0] as f64 / 255.0;
            let t = transmittance_buf[idx] as f64;

            // Unbiased zero-mean grain modulation
            let delta_t = (t - mean_t) * grain_strength;

            // Weight grain by tone: midtones have highest visibility, shadows have subtle grain, highlights have clumping
            let grain_weight = (1.0 - (base_val - 0.55).abs() * 1.4).max(0.2);
            let final_val = (base_val + delta_t * grain_weight).clamp(0.0, 1.0);

            let byte_val = (final_val * 255.0).round() as u8;
            output_img.put_pixel(x, y, Luma([byte_val]));
        }
    }

    let out_path = if std::path::Path::new("test_assets").exists() {
        "test_assets/swing_grain_output.png"
    } else {
        "swing_grain_output.png"
    };
    output_img.save(out_path).unwrap();
    println!("Saved {} in {:.2?}", out_path, start.elapsed());
}
