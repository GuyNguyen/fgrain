//! Emulsion layer synthesis and film stock presets.

use crate::bvh::BVH;
use crate::crystal::{HalideCrystal, Morphology};
use crate::optics::GelatinOptics;
use crate::poisson::sample_poisson_3d;
use nalgebra::{Unit, UnitQuaternion, Vector3};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, LogNormal, Normal};

/// Physical configuration and synthesis parameters for a virtual emulsion layer.
#[derive(Debug, Clone)]
pub struct EmulsionConfig {
    /// Physical volume dimensions of the film patch (width, height, depth in micrometers).
    pub dimensions: Vector3<f64>,

    /// Minimum separation distance between crystal centroids (micrometers).
    pub r_min: f64,

    /// Mean crystal semi-major radius (micrometers).
    pub mean_crystal_radius: f64,

    /// Standard deviation parameter for log-normal grain size distribution.
    pub size_variance: f64,

    /// Crystal habit and morphology.
    pub morphology: Morphology,

    /// Aspect ratio for tabular grains (diameter / thickness). Typical T-grains have aspect ratio 5:1 to 10:1.
    pub tabular_aspect_ratio: f64,

    /// Optical properties of the gelatin matrix.
    pub optics: GelatinOptics,

    /// Base quantum sensitivity of crystals.
    pub base_sensitivity: f64,
}

impl EmulsionConfig {
    /// Preset configuration simulating Kodak Tri-X 400 (ISO 400 traditional emulsion).
    ///
    /// Characterized by thick emulsion layer (~14 um), polydisperse cubic crystals,
    /// and prominent classic grain clumping.
    #[must_use]
    pub fn tri_x_400(width: f64, height: f64) -> Self {
        Self {
            dimensions: Vector3::new(width, height, 14.0),
            r_min: 1.2,
            mean_crystal_radius: 0.55,
            size_variance: 0.40,
            morphology: Morphology::Cubic,
            tabular_aspect_ratio: 1.0,
            optics: GelatinOptics {
                refractive_index: 1.534,
                absorption_coeff: 0.003,
                scattering_coeff: 0.050,
                anisotropy_g: 0.80,
                antihalation_reflectance: 0.14,
                antihalation_density: 0.80,
            },
            base_sensitivity: 1.2,
        }
    }

    /// Preset configuration simulating Kodak Tri-X 400 pushed to EI 1600 (pushed processing / Caffenol).
    ///
    /// Characterized by thick emulsion layer (~15 um), swollen cubic crystal colonies,
    /// increased size dispersion, and enhanced base quantum sensitivity.
    #[must_use]
    pub fn tri_x_1600(width: f64, height: f64) -> Self {
        Self {
            dimensions: Vector3::new(width, height, 14.5),
            r_min: 1.1,
            mean_crystal_radius: 0.52,
            size_variance: 0.38,
            morphology: Morphology::Cubic,
            tabular_aspect_ratio: 1.0,
            optics: GelatinOptics {
                refractive_index: 1.534,
                absorption_coeff: 0.0032,
                scattering_coeff: 0.052,
                anisotropy_g: 0.80,
                antihalation_reflectance: 0.14,
                antihalation_density: 0.80,
            },
            base_sensitivity: 1.35,
        }
    }

    /// Preset configuration simulating Kodak T-Max 100 (tabular T-Grain technology).
    ///
    /// Characterized by a thinner emulsion (~8 um), flat hexagonal tabular crystals
    /// with high surface-to-volume ratio, aligned parallel to the film plane.
    #[must_use]
    pub fn t_max_100(width: f64, height: f64) -> Self {
        Self {
            dimensions: Vector3::new(width, height, 8.0),
            r_min: 0.9,
            mean_crystal_radius: 0.40,
            size_variance: 0.20,
            morphology: Morphology::Tabular,
            tabular_aspect_ratio: 6.5,
            optics: GelatinOptics {
                refractive_index: 1.534,
                absorption_coeff: 0.002,
                scattering_coeff: 0.035,
                anisotropy_g: 0.85,
                antihalation_reflectance: 0.08,
                antihalation_density: 1.10, // Efficient anti-halation backing
            },
            base_sensitivity: 0.8,
        }
    }

    /// Preset configuration simulating Ilford HP5 Plus (ISO 400 traditional cubic/octahedral grain).
    #[must_use]
    pub fn hp5_plus(width: f64, height: f64) -> Self {
        Self {
            dimensions: Vector3::new(width, height, 12.5),
            r_min: 1.1,
            mean_crystal_radius: 0.50,
            size_variance: 0.30,
            morphology: Morphology::Octahedral,
            tabular_aspect_ratio: 1.0,
            optics: GelatinOptics {
                refractive_index: 1.534,
                absorption_coeff: 0.0025,
                scattering_coeff: 0.045,
                anisotropy_g: 0.82,
                antihalation_reflectance: 0.12,
                antihalation_density: 0.85,
            },
            base_sensitivity: 1.1,
        }
    }
}

/// A 3D photographic emulsion layer containing silver halide crystals and gelatin optics.
#[derive(Debug, Clone)]
pub struct VirtualEmulsionLayer {
    /// Configuration used to synthesize this emulsion.
    pub config: EmulsionConfig,

    /// Contiguous buffer of silver halide crystals in memory.
    pub crystals: Vec<HalideCrystal>,

    /// Acceleration structure for ray-crystal queries.
    pub bvh: BVH,
}

impl VirtualEmulsionLayer {
    /// Synthesizes a new virtual emulsion layer using 3D Poisson-disk sampling.
    #[must_use]
    pub fn synthesize(config: EmulsionConfig, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);

        // Poisson-disk sampling for spatial distribution
        let points = sample_poisson_3d(&config.dimensions, config.r_min, 30, &mut rng);

        // Log-normal crystal size distribution
        let log_normal = LogNormal::new(config.mean_crystal_radius.ln(), config.size_variance)
            .unwrap_or_else(|_| LogNormal::new(0.0, 0.2).unwrap());
        let sensitivity_dist = Normal::new(config.base_sensitivity, 0.15).unwrap();

        let mut crystals = Vec::with_capacity(points.len());

        for (id, pos) in points.into_iter().enumerate() {
            let radius = log_normal.sample(&mut rng).clamp(0.15, config.r_min * 0.48);
            let sensitivity = sensitivity_dist.sample(&mut rng).max(0.1);

            let (dimensions, orientation) = match config.morphology {
                Morphology::Cubic | Morphology::Octahedral => {
                    let u1: f64 = rng.gen_range(0.0..1.0);
                    let u2: f64 = rng.gen_range(0.0..1.0);
                    let u3: f64 = rng.gen_range(0.0..1.0);

                    let q = UnitQuaternion::from_euler_angles(
                        u1 * std::f64::consts::TAU,
                        u2 * std::f64::consts::PI,
                        u3 * std::f64::consts::TAU,
                    );
                    (Vector3::new(radius, radius, radius), q)
                }
                Morphology::Tabular => {
                    // Tabular grains are thin plates oriented mostly parallel to the film plane (XY)
                    let half_thickness = (radius / config.tabular_aspect_ratio).max(0.05);

                    // Random tilt wobble (~10 degrees in radians)
                    let tilt_angle: f64 = rng.gen_range(-0.17..0.17);
                    let tilt_axis =
                        Vector3::new(rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0), 0.0)
                            .normalize();
                    let q = UnitQuaternion::from_axis_angle(
                        &Unit::new_normalize(tilt_axis),
                        tilt_angle,
                    );

                    (Vector3::new(radius, radius, half_thickness), q)
                }
            };

            crystals.push(HalideCrystal::new(
                id,
                pos,
                dimensions,
                orientation,
                config.morphology,
                sensitivity,
            ));
        }

        let bvh = BVH::build(&crystals);

        Self {
            config,
            crystals,
            bvh,
        }
    }

    /// Rebuilds the BVH after post-development filament expansion.
    pub fn rebuild_bvh(&mut self) {
        self.bvh = BVH::build(&self.crystals);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_emulsion_synthesis() {
        let config = EmulsionConfig::t_max_100(25.0, 25.0);
        let emulsion = VirtualEmulsionLayer::synthesize(config, 42);

        assert!(!emulsion.crystals.is_empty(), "Should synthesize crystals");
        assert!(!emulsion.bvh.nodes.is_empty(), "Should construct BVH");

        // Verify all crystals lie strictly inside the emulsion volume bounds
        for crystal in &emulsion.crystals {
            assert!(
                crystal.position.x >= 0.0 && crystal.position.x <= emulsion.config.dimensions.x
            );
            assert!(
                crystal.position.y >= 0.0 && crystal.position.y <= emulsion.config.dimensions.y
            );
            assert!(
                crystal.position.z >= 0.0 && crystal.position.z <= emulsion.config.dimensions.z
            );
        }
    }
}
