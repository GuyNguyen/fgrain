use fgrain::*;
use image::{GrayImage, Luma};
use std::time::Instant;

fn main() {
    let input_path = if std::path::Path::new("test_assets/crop_test_input.png").exists() {
        "test_assets/crop_test_input.png"
    } else {
        "crop_test_input.png"
    };
    println!(
        "Testing calibrated film grain rendering on {}...",
        input_path
    );
    let start = Instant::now();

    let input_img = image::open(input_path).unwrap().to_luma8();
    let (w, h) = input_img.dimensions();
    println!("Input size: {}x{}", w, h);

    // Let's test a 256x256 test sub-crop
    let test_crop = image::imageops::crop_imm(&input_img, 256, 256, 256, 256).to_image();
    let subcrop_path = if std::path::Path::new("test_assets").exists() {
        "test_assets/subcrop_input.png"
    } else {
        "subcrop_input.png"
    };
    test_crop.save(subcrop_path).unwrap();

    // Calibration:
    // Target grain size: ~2.5 pixels diameter per crystal
    // With mean crystal diameter ~1.0 um:
    // pixel size = 1.0 um / 2.5 = 0.4 um per pixel.
    let pixel_size_um = 0.40;
    let patch_w_um = 256.0 * pixel_size_um; // ~102.4 um
    let patch_h_um = 256.0 * pixel_size_um;

    let config = EmulsionConfig::tri_x_400(patch_w_um, patch_h_um);
    let mut emulsion = VirtualEmulsionLayer::synthesize(config, 42);
    println!(
        "Synthesized {} crystals in {:.1}x{:.1} um",
        emulsion.crystals.len(),
        patch_w_um,
        patch_h_um
    );

    // Expose using the input image
    let exp_config = ExposureConfig {
        photons_per_sq_um: 120.0,
        illumination_angle_rad: 0.12,
        crystal_absorption_coeff: 1.3,
        max_bounces: 48,
    };

    expose_emulsion(
        &emulsion,
        |x, y| {
            let u = (x / patch_w_um).clamp(0.0, 1.0);
            let v = (y / patch_h_um).clamp(0.0, 1.0);
            let px = (u * 255.0) as u32;
            let py = (v * 255.0) as u32;
            let val = test_crop.get_pixel(px.min(255), py.min(255)).0[0] as f64 / 255.0;
            // Gamma to linear scene radiance
            val.powf(2.0) * 1.5
        },
        &exp_config,
        1001,
    );

    let dev_config = DevelopmentConfig {
        threshold_photons: 3,
        filament_expansion: 1.8,
        clumping_proximity_factor: 1.25,
        developer_contrast: 0.8,
    };

    let dev_report = develop_emulsion(&mut emulsion.crystals, &emulsion.bvh, &dev_config);
    println!(
        "Developed {} crystals in {} clumps",
        dev_report.developed_crystals, dev_report.total_clumps
    );

    // Scan transmittance
    let scan_config = ScanConfig {
        width_px: 256,
        height_px: 256,
        samples_per_pixel: 16,
        mode: ScanMode::NegativeTransmittance,
        silver_opacity: 2.5, // Realistic semi-dense silver extinction
    };

    let (transmittance_buf, _raw_img) = scan_emulsion(&emulsion, &scan_config, 2002);

    // Combine scene tones with stochastic 3D grain transmittance
    let mut output_img = GrayImage::new(256, 256);
    let grain_strength = 0.35; // Contrast of film grain texture

    for y in 0..256 {
        for x in 0..256 {
            let idx = (y * 256 + x) as usize;
            let base_val = test_crop.get_pixel(x, y).0[0] as f64 / 255.0;

            // Optical transmittance: developed silver absorbs light (lower T)
            let t = transmittance_buf[idx] as f64;

            // Film characteristic curve (Hurter-Driffield S-curve) applied to scene
            let gamma = 1.15;
            let film_tone = base_val.powf(gamma);

            // Grain variation: difference from local mean transmittance
            // In highlights/midtones, silver grains create subtle absorption dips and organic clumps
            let grain_fluctuation = (t - 0.5) * grain_strength;

            // Modulate: grain is most visible in midtones, subtle in deep shadows, clumpy in highlights
            let midtone_weight = (1.0 - (2.0 * film_tone - 1.0).powi(2)).max(0.15);
            let final_val = (film_tone + grain_fluctuation * midtone_weight).clamp(0.0, 1.0);

            let byte_val = (final_val * 255.0).round() as u8;
            output_img.put_pixel(x, y, Luma([byte_val]));
        }
    }

    let out_path = if std::path::Path::new("test_assets").exists() {
        "test_assets/subcrop_grain_output.png"
    } else {
        "subcrop_grain_output.png"
    };
    output_img.save(out_path).unwrap();
    println!("Saved {} in {:.2?}", out_path, start.elapsed());
}
