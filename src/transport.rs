//! Monte Carlo photon transport and ray-crystal exposure simulation.
//!
//! Uses Rayon parallel iterators to trace photon packets through the 3D emulsion volume,
//! resolving gelatin scattering, silver halide absorption, and anti-halation reflections.

use crate::emulsion::VirtualEmulsionLayer;
use crate::optics::PhotonPacket;
use nalgebra::{Point3, Vector3};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

/// Configuration parameters for an exposure pass.
#[derive(Debug, Clone)]
pub struct ExposureConfig {
    /// Total number of Monte Carlo photon packets traced per square micrometer at unit exposure (1.0).
    pub photons_per_sq_um: f64,

    /// Lens illumination half-angle cone (in radians; 0.0 represents collimated light normal to the surface).
    pub illumination_angle_rad: f64,

    /// Silver halide crystal intrinsic absorption coefficient $\alpha_{AgBr}$ ($\mu m^{-1}$).
    /// For blue/UV light, silver bromide has high absorption ($\sim 0.8 - 1.5 \mu m^{-1}$).
    pub crystal_absorption_coeff: f64,

    /// Maximum scattering bounces allowed per photon packet before Russian roulette termination.
    pub max_bounces: usize,
}

impl Default for ExposureConfig {
    fn default() -> Self {
        Self {
            photons_per_sq_um: 50.0,
            illumination_angle_rad: 0.15, // ~8.5 degrees cone
            crystal_absorption_coeff: 1.2,
            max_bounces: 64,
        }
    }
}

/// Simulates the exposure of a virtual emulsion layer using Monte Carlo photon transport.
///
/// # Arguments
/// * `emulsion` - The virtual emulsion containing crystals and optical parameters.
/// * `exposure_fn` - Spatial irradiance function $E(x, y) \in [0, \infty)$ giving relative intensity at film coordinate $(x, y)$.
/// * `config` - Exposure and photon transport settings.
/// * `seed` - Base random seed.
pub fn expose_emulsion<F>(
    emulsion: &VirtualEmulsionLayer,
    exposure_fn: F,
    config: &ExposureConfig,
    seed: u64,
) where
    F: Fn(f64, f64) -> f64 + Sync + Send,
{
    let dims = &emulsion.config.dimensions;
    let area = dims.x * dims.y;
    let base_photon_count = (area * config.photons_per_sq_um).round() as usize;

    if base_photon_count == 0 {
        return;
    }

    // Number of parallel batch chunks to distribute across CPU cores via Rayon
    let num_threads = rayon::current_num_threads().max(1);
    let chunks = num_threads * 8;
    let photons_per_chunk = base_photon_count.div_ceil(chunks);

    (0..chunks).into_par_iter().for_each(|chunk_idx| {
        let mut rng = StdRng::seed_from_u64(seed.wrapping_add(chunk_idx as u64 * 1013904223));

        for _ in 0..photons_per_chunk {
            let x = rng.gen_range(0.0..dims.x);
            let y = rng.gen_range(0.0..dims.y);

            let local_irradiance = exposure_fn(x, y).max(0.0);
            if local_irradiance <= 0.0 {
                continue;
            }

            let energy_weight = local_irradiance;

            // Sample incident direction within illumination cone
            let theta = rng.gen_range(0.0..config.illumination_angle_rad);
            let phi = rng.gen_range(0.0..std::f64::consts::TAU);
            let sin_t = theta.sin();
            let dir = Vector3::new(sin_t * phi.cos(), sin_t * phi.sin(), theta.cos()).normalize();

            // Air-gelatin interface refraction and Fresnel reflection
            let cos_theta_i = dir.z;
            let fresnel_r = emulsion.config.optics.fresnel_air_gelatin(cos_theta_i);
            if rng.gen_bool(fresnel_r) {
                continue;
            }

            let mut photon = PhotonPacket::new(Point3::new(x, y, 0.001), dir);
            photon.energy = energy_weight;

            // Transport loop within gelatin volume
            let mut bounces = 0;
            while photon.alive && bounces < config.max_bounces {
                bounces += 1;

                let free_path = emulsion.config.optics.sample_free_path(&mut rng);
                let step_dist = free_path.min(dims.z * 1.5);

                // Intersect ray with crystals along current path segment
                emulsion.bvh.intersect_ray_all(
                    &photon.position,
                    &photon.direction,
                    &emulsion.crystals,
                    |crystal_idx, t_enter, t_exit| {
                        if t_enter < step_dist {
                            let path_in_crystal = (t_exit.min(step_dist) - t_enter).max(0.0);
                            if path_in_crystal > 0.0 {
                                // Beer-Lambert absorption in crystal: P = 1 - e^(-alpha * path)
                                let p_abs = 1.0
                                    - (-config.crystal_absorption_coeff * path_in_crystal).exp();
                                let absorbed_energy = photon.energy
                                    * p_abs
                                    * emulsion.crystals[crystal_idx].sensitivity;

                                // Poisson quantum conversion to discrete latent photons
                                let expected_photons = absorbed_energy * 5.0;
                                let whole = expected_photons.floor() as u32;
                                let frac = expected_photons - f64::from(whole);
                                let extra = u32::from(rng.gen_bool(frac.clamp(0.0, 1.0)));
                                let count = whole + extra;

                                if count > 0 {
                                    emulsion.crystals[crystal_idx].record_absorption(count);
                                }

                                photon.energy *= (1.0 - p_abs * 0.9).max(0.05);
                            }
                        }
                    },
                );

                let new_pos = photon.position + photon.direction * step_dist;

                // Check lateral boundaries (x, y): periodic wrap-around or absorption
                if new_pos.x < 0.0 || new_pos.x >= dims.x || new_pos.y < 0.0 || new_pos.y >= dims.y
                {
                    photon.alive = false;
                    break;
                }

                // Check top boundary (z <= 0): backscattered out of the film
                if new_pos.z <= 0.0 {
                    photon.alive = false;
                    break;
                }

                // Check bottom boundary (z >= dims.z): anti-halation layer reflection
                if new_pos.z >= dims.z {
                    photon.position = Point3::new(new_pos.x, new_pos.y, dims.z - 0.001);
                    emulsion
                        .config
                        .optics
                        .handle_antihalation_reflection(&mut photon, &mut rng);
                    continue;
                }

                photon.position = new_pos;

                // Bulk gelatin scattering: sample Henyey-Greenstein direction
                photon.direction = emulsion
                    .config
                    .optics
                    .sample_henyey_greenstein(&photon.direction, &mut rng);

                // Gelatin bulk absorption
                photon.energy *= emulsion.config.optics.albedo();

                // Russian roulette termination
                if photon.energy < 1e-3 {
                    if rng.gen_bool(0.1) {
                        photon.energy /= 0.1;
                    } else {
                        photon.alive = false;
                        break;
                    }
                }
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::emulsion::EmulsionConfig;

    #[test]
    fn test_exposure_latent_image_formation() {
        let config = EmulsionConfig::tri_x_400(20.0, 20.0);
        let emulsion = VirtualEmulsionLayer::synthesize(config, 100);

        let exp_config = ExposureConfig {
            photons_per_sq_um: 100.0,
            illumination_angle_rad: 0.1,
            crystal_absorption_coeff: 1.5,
            max_bounces: 32,
        };

        // Expose left half of film (x <= 10.0) with high intensity, right half with 0
        expose_emulsion(
            &emulsion,
            |x, _y| if x <= 10.0 { 1.0 } else { 0.0 },
            &exp_config,
            42,
        );

        let mut left_absorbed = 0;
        let mut right_absorbed = 0;

        for c in &emulsion.crystals {
            let absorbed = c.absorbed_photons();
            if c.position.x <= 9.0 {
                left_absorbed += absorbed;
            } else if c.position.x >= 11.0 {
                right_absorbed += absorbed;
            }
        }

        assert!(
            left_absorbed > 0,
            "Exposed side should have absorbed latent photons"
        );
        assert!(
            left_absorbed > right_absorbed * 5,
            "Exposed side ({}) should have substantially more absorbed photons than unexposed side ({})",
            left_absorbed,
            right_absorbed
        );
    }
}
