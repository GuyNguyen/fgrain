//! 3D film grain simulation engine modeling silver halide photographic emulsions.
//!
//! # Pipeline Stages
//!
//! 1. **Emulsion Synthesis ([`emulsion`]):** Generates 3D crystal volumes using Poisson-disk sampling.
//! 2. **Photon Transport ([`transport`]):** Traces incident light through gelatin with Henyey-Greenstein scattering.
//! 3. **Chemical Development ([`development`]):** Evaluates latent image centers against exposure thresholds.
//! 4. **Optical Scanning ([`transmittance`]):** Calculates transmittance by integrating optical paths through developed silver.

pub mod bvh;
pub mod crystal;
pub mod development;
pub mod emulsion;
pub mod error;
pub mod gpu;
pub mod image_pipeline;
pub mod optics;
pub mod poisson;
pub mod transmittance;
pub mod transport;

pub use bvh::BVH;
pub use crystal::{HalideCrystal, Morphology};
pub use development::{DevelopmentConfig, DevelopmentReport, develop_emulsion};
pub use emulsion::{EmulsionConfig, VirtualEmulsionLayer};
pub use error::{GpuError, PipelineError};
pub use gpu::{GpuEngine, get_gpu_engine};
pub use image_pipeline::{
    FilmRenderOptions, ImageProcessReport, process_image_file, render_film_black_and_white_tiled,
    render_film_color_tiled,
};
pub use optics::{GelatinOptics, PhotonPacket};
pub use transmittance::{ScanConfig, ScanMode, scan_emulsion};
pub use transport::{ExposureConfig, expose_emulsion};

/// Configures and executes an end-to-end film simulation.
#[derive(Debug, Clone)]
pub struct FilmSimulation {
    /// Virtual emulsion configuration.
    pub emulsion_config: EmulsionConfig,
    /// Monte Carlo exposure parameters.
    pub exposure_config: ExposureConfig,
    /// Chemical development parameters.
    pub development_config: DevelopmentConfig,
    /// Densitometer / scanner rendering parameters.
    pub scan_config: ScanConfig,
}

impl FilmSimulation {
    /// Creates a default Kodak Tri-X 400 film simulation pipeline for a given film patch size.
    #[must_use]
    pub fn tri_x_400(width_um: f64, height_um: f64, width_px: u32, height_px: u32) -> Self {
        Self {
            emulsion_config: EmulsionConfig::tri_x_400(width_um, height_um),
            exposure_config: ExposureConfig::default(),
            development_config: DevelopmentConfig::default(),
            scan_config: ScanConfig {
                width_px,
                height_px,
                samples_per_pixel: 16,
                mode: ScanMode::PositivePrint,
                silver_opacity: 25.0,
            },
        }
    }

    /// Creates a default Kodak T-Max 100 film simulation pipeline with tabular T-Grains.
    #[must_use]
    pub fn t_max_100(width_um: f64, height_um: f64, width_px: u32, height_px: u32) -> Self {
        Self {
            emulsion_config: EmulsionConfig::t_max_100(width_um, height_um),
            exposure_config: ExposureConfig::default(),
            development_config: DevelopmentConfig::default(),
            scan_config: ScanConfig {
                width_px,
                height_px,
                samples_per_pixel: 16,
                mode: ScanMode::PositivePrint,
                silver_opacity: 25.0,
            },
        }
    }

    /// Runs the complete physical simulation pipeline.
    ///
    /// # Arguments
    /// * `exposure_fn` - Continuous spatial exposure distribution $E(x, y)$ over the film patch.
    /// * `seed` - Base random seed.
    pub fn run<F>(&self, exposure_fn: F, seed: u64) -> (DevelopmentReport, image::GrayImage)
    where
        F: Fn(f64, f64) -> f64 + Sync + Send,
    {
        let mut emulsion = VirtualEmulsionLayer::synthesize(self.emulsion_config.clone(), seed);

        expose_emulsion(
            &emulsion,
            exposure_fn,
            &self.exposure_config,
            seed.wrapping_add(1),
        );

        let report = develop_emulsion(
            &mut emulsion.crystals,
            &emulsion.bvh,
            &self.development_config,
        );

        let (_buffer, image) = scan_emulsion(&emulsion, &self.scan_config, seed.wrapping_add(2));

        (report, image)
    }
}
