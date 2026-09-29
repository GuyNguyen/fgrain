//! Resolution-independent optical transmittance and film scanning evaluation.
//!
//! Evaluates the fraction of transmitted light passing through the 3D mesh of developed
//! metallic silver grains, simulating an optical densitometer or drum scanner at arbitrary output resolution.

use crate::bvh::BVH;
use crate::crystal::HalideCrystal;
use crate::emulsion::VirtualEmulsionLayer;
use image::{GrayImage, Luma};
use nalgebra::{Point3, Vector3};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

/// Output representation mode for the scanned film.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScanMode {
    /// Photographic negative: developed silver blocks light, appearing dark on the negative.
    /// Pixel value corresponds directly to optical transmittance $T \in [0, 1]$.
    NegativeTransmittance,

    /// Positive print / digital inverted scan: high transmittance appears dark, dense silver appears bright.
    /// Pixel value corresponds to developed optical density / absorbed light ($1 - T$).
    PositivePrint,

    /// Diffuse optical density ($D = -\log_{10}(T)$), scaled for visualization.
    OpticalDensity,
}

/// Configuration for resolution-independent optical scanning.
#[derive(Debug, Clone)]
pub struct ScanConfig {
    /// Output image pixel width.
    pub width_px: u32,

    /// Output image pixel height.
    pub height_px: u32,

    /// Number of stratified Monte Carlo rays traced per pixel for anti-aliasing.
    pub samples_per_pixel: u32,

    /// Scanner rendering mode.
    pub mode: ScanMode,

    /// Optical attenuation factor for metallic silver filaments ($\mu m^{-1}$).
    /// Real metallic silver filaments have extremely high opacity ($\approx 10.0 - 50.0 \mu m^{-1}$).
    pub silver_opacity: f64,
}

impl Default for ScanConfig {
    fn default() -> Self {
        Self {
            width_px: 512,
            height_px: 512,
            samples_per_pixel: 16,
            mode: ScanMode::PositivePrint,
            silver_opacity: 25.0,
        }
    }
}

/// Evaluates resolution-independent optical transmittance of a developed emulsion layer.
///
/// Traces parallel scanner backlight rays across the 3D developed crystal volume using Rayon.
pub fn scan_emulsion(
    emulsion: &VirtualEmulsionLayer,
    config: &ScanConfig,
    seed: u64,
) -> (Vec<f32>, GrayImage) {
    let w_px = config.width_px;
    let h_px = config.height_px;
    let spp = config.samples_per_pixel.max(1);

    let film_w = emulsion.config.dimensions.x;
    let film_h = emulsion.config.dimensions.y;
    let film_d = emulsion.config.dimensions.z;

    let pixel_w = film_w / (w_px as f64);
    let pixel_h = film_h / (h_px as f64);

    // Build BVH over developed silver crystals only
    let developed_crystals: Vec<HalideCrystal> = emulsion
        .crystals
        .iter()
        .filter(|c| c.developed)
        .cloned()
        .collect();

    let scan_bvh = BVH::build(&developed_crystals);

    // Stratified grid dimension per pixel (e.g. 4x4 for spp=16)
    let grid_dim = (spp as f64).sqrt().round() as u32;
    let actual_spp = (grid_dim * grid_dim).max(1);

    let buffer: Vec<f32> = (0..h_px)
        .into_par_iter()
        .flat_map(|py| {
            let mut row_pixels = Vec::with_capacity(w_px as usize);

            for px in 0..w_px {
                // Per-pixel seed mixing
                let pixel_seed = seed
                    .wrapping_add((px as u64).wrapping_mul(0x9E3779B97F4A7C15))
                    .wrapping_add((py as u64).wrapping_mul(0xC6A4A7935BD1E995));
                let mut rng = StdRng::seed_from_u64(pixel_seed);

                let mut accumulated_transmittance = 0.0;

                // Stratified sub-pixel sampling
                for sy in 0..grid_dim {
                    for sx in 0..grid_dim {
                        let sub_x = (sx as f64 + rng.gen_range(0.0..1.0)) / (grid_dim as f64);
                        let sub_y = (sy as f64 + rng.gen_range(0.0..1.0)) / (grid_dim as f64);

                        let film_x = (px as f64 + sub_x) * pixel_w;
                        let film_y = (py as f64 + sub_y) * pixel_h;

                        // Backlight ray travels in -Z direction through the film
                        let ray_origin = Point3::new(film_x, film_y, film_d + 0.1);
                        let ray_dir = Vector3::new(0.0, 0.0, -1.0);

                        let mut total_optical_path = 0.0;

                        if !developed_crystals.is_empty() {
                            scan_bvh.intersect_ray_all(
                                &ray_origin,
                                &ray_dir,
                                &developed_crystals,
                                |_idx, t_enter, t_exit| {
                                    let path_in_silver = (t_exit - t_enter).max(0.0);
                                    total_optical_path += path_in_silver;
                                },
                            );
                        }

                        // Beer-Lambert attenuation: T = exp(-sigma * path)
                        let transmittance = (-config.silver_opacity * total_optical_path).exp();
                        accumulated_transmittance += transmittance;
                    }
                }

                let mean_transmittance = (accumulated_transmittance / f64::from(actual_spp)) as f32;

                let output_val = match config.mode {
                    ScanMode::NegativeTransmittance => mean_transmittance,
                    ScanMode::PositivePrint => 1.0 - mean_transmittance,
                    ScanMode::OpticalDensity => {
                        let t = mean_transmittance.max(1e-4);
                        let od = -t.log10();
                        (od / 3.0).clamp(0.0, 1.0)
                    }
                };

                row_pixels.push(output_val);
            }

            row_pixels
        })
        .collect();

    // Convert normalized buffer to 8-bit GrayImage
    let mut image = GrayImage::new(w_px, h_px);
    for (i, &val) in buffer.iter().enumerate() {
        let px = (i % w_px as usize) as u32;
        let py = (i / w_px as usize) as u32;
        let byte_val = (val.clamp(0.0, 1.0) * 255.0).round() as u8;
        image.put_pixel(px, py, Luma([byte_val]));
    }

    (buffer, image)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crystal::Morphology;
    use crate::emulsion::EmulsionConfig;
    use nalgebra::UnitQuaternion;

    #[test]
    fn test_transmittance_evaluation() {
        let config = EmulsionConfig::t_max_100(10.0, 10.0);
        let mut emulsion = VirtualEmulsionLayer::synthesize(config, 123);

        // Mark only one crystal as developed
        for c in emulsion.crystals.iter_mut() {
            c.developed = false;
        }
        emulsion.crystals[0].developed = true;
        emulsion.crystals[0].position = Point3::new(5.0, 5.0, 4.0);
        emulsion.crystals[0].dimensions = Vector3::new(2.0, 2.0, 1.0);
        emulsion.crystals[0].orientation = UnitQuaternion::identity();
        emulsion.crystals[0].morphology = Morphology::Cubic;
        emulsion.crystals[0].filament_expansion = 1.0;

        let scan_cfg = ScanConfig {
            width_px: 16,
            height_px: 16,
            samples_per_pixel: 4,
            mode: ScanMode::NegativeTransmittance,
            silver_opacity: 20.0,
        };

        let (buffer, _img) = scan_emulsion(&emulsion, &scan_cfg, 42);

        // Center pixel (around x=5, y=5) should have lower transmittance due to developed grain
        let center_idx = 8 * 16 + 8;
        let corner_idx = 0;

        assert!(
            buffer[center_idx] < buffer[corner_idx],
            "Developed grain must reduce optical transmittance: center {} vs corner {}",
            buffer[center_idx],
            buffer[corner_idx]
        );
        assert!(
            buffer[corner_idx] > 0.95,
            "Clear gelatin in corner must have high transmittance"
        );
    }
}
