//! Optical physics, subsurface scattering, and anti-halation backing models.
//!
//! Simulates photon transport within photographic gelatin, including Henyey-Greenstein
//! volume scattering, Snell-Descartes boundary refraction, Fresnel reflections, and
//! anti-halation undercoating (AHU) reflection dynamics.

use nalgebra::{Point3, Vector3};
use rand::Rng;

/// Represents a Monte Carlo photon packet undergoing stochastic transport.
#[derive(Debug, Clone)]
pub struct PhotonPacket {
    /// Current 3D position in the emulsion (micrometers).
    pub position: Point3<f64>,

    /// Normalized unit direction vector of propagation.
    pub direction: Vector3<f64>,

    /// Dynamic photon energy / statistical weight ($E \in [0, 1]$).
    pub energy: f64,

    /// Whether this photon packet remains active in the transport loop.
    pub alive: bool,
}

impl PhotonPacket {
    /// Creates a new photon packet with initial energy 1.0.
    #[must_use]
    pub fn new(position: Point3<f64>, direction: Vector3<f64>) -> Self {
        Self {
            position,
            direction: direction.normalize(),
            energy: 1.0,
            alive: true,
        }
    }
}

/// Optical properties of the gelatin matrix emulsion layer.
#[derive(Debug, Clone)]
pub struct GelatinOptics {
    /// Refractive index of photographic gelatin ($n \approx 1.53$).
    pub refractive_index: f64,

    /// Gelatin linear absorption coefficient $\mu_a$ ($\mu m^{-1}$).
    pub absorption_coeff: f64,

    /// Gelatin linear scattering coefficient $\mu_s$ ($\mu m^{-1}$).
    pub scattering_coeff: f64,

    /// Henyey-Greenstein scattering anisotropy parameter ($g \in (-1, 1)$, typical $g \approx 0.8$).
    pub anisotropy_g: f64,

    /// Anti-halation backing reflectance ($R_{AHU} \in [0, 1]$).
    /// Models the fraction of light reflected back into the emulsion from the film base.
    pub antihalation_reflectance: f64,

    /// Optical density of the anti-halation dye layer ($OD = -\log_{10}(T)$).
    pub antihalation_density: f64,
}

impl Default for GelatinOptics {
    fn default() -> Self {
        Self {
            refractive_index: 1.534,
            absorption_coeff: 0.002, // Low intrinsic absorption in visible light
            scattering_coeff: 0.045, // Medium-high turbid forward scattering
            anisotropy_g: 0.82,      // Strongly forward-peaked Mie/Rayleigh scattering
            antihalation_reflectance: 0.12, // Base Fresnel + backing reflection
            antihalation_density: 0.85, // Anti-halation dye attenuation
        }
    }
}

impl GelatinOptics {
    /// Total extinction coefficient $\mu_t = \mu_a + \mu_s$.
    #[inline]
    #[must_use]
    pub fn extinction_coeff(&self) -> f64 {
        self.absorption_coeff + self.scattering_coeff
    }

    /// Single-scattering albedo $a = \mu_s / \mu_t$.
    #[inline]
    #[must_use]
    pub fn albedo(&self) -> f64 {
        let mu_t = self.extinction_coeff();
        if mu_t > 0.0 {
            self.scattering_coeff / mu_t
        } else {
            0.0
        }
    }

    /// Samples a random free flight distance $s$ before a scattering or absorption event.
    ///
    /// $s = -\frac{\ln(1 - \xi)}{\mu_t}$
    #[inline]
    #[must_use]
    pub fn sample_free_path<R: Rng>(&self, rng: &mut R) -> f64 {
        let mu_t = self.extinction_coeff();
        if mu_t <= 0.0 {
            return 1e6;
        }
        let xi: f64 = rng.gen_range(0.0..1.0);
        -(1.0 - xi).max(f64::MIN_POSITIVE).ln() / mu_t
    }

    /// Samples a new scattering direction according to the Henyey-Greenstein phase function.
    ///
    /// $p(\cos\theta) = \frac{1}{4\pi} \frac{1 - g^2}{(1 + g^2 - 2g\cos\theta)^{3/2}}$
    #[must_use]
    pub fn sample_henyey_greenstein<R: Rng>(
        &self,
        current_dir: &Vector3<f64>,
        rng: &mut R,
    ) -> Vector3<f64> {
        let g = self.anisotropy_g;
        let xi1: f64 = rng.gen_range(0.0..1.0);
        let xi2: f64 = rng.gen_range(0.0..1.0);

        let cos_theta = if g.abs() < 1e-4 {
            2.0 * xi1 - 1.0
        } else {
            let term = (1.0 - g * g) / (1.0 - g + 2.0 * g * xi1);
            (1.0 + g * g - term * term) / (2.0 * g)
        };

        let cos_theta = cos_theta.clamp(-1.0, 1.0);
        let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
        let phi = 2.0 * std::f64::consts::PI * xi2;

        // Build orthonormal coordinate frame around current_dir
        let w = current_dir.normalize();
        let u = if w.x.abs() > 0.1 {
            Vector3::new(0.0, 1.0, 0.0).cross(&w).normalize()
        } else {
            Vector3::new(1.0, 0.0, 0.0).cross(&w).normalize()
        };
        let v = w.cross(&u);

        (u * (sin_theta * phi.cos()) + v * (sin_theta * phi.sin()) + w * cos_theta).normalize()
    }

    /// Evaluates Fresnel reflectance at the air-gelatin interface ($n_1 = 1.0 \to n_2 = 1.53$) using Schlick's approximation.
    #[must_use]
    pub fn fresnel_air_gelatin(&self, cos_theta_i: f64) -> f64 {
        let n1 = 1.0;
        let n2 = self.refractive_index;

        let r0 = ((n1 - n2) / (n1 + n2)).powi(2);
        let cos_i = cos_theta_i.abs().clamp(0.0, 1.0);
        r0 + (1.0 - r0) * (1.0 - cos_i).powi(5)
    }

    /// Evaluates anti-halation reflection at the bottom film base interface ($z = depth$).
    ///
    /// Returns the reflected photon packet with attenuated energy and redirected direction,
    /// or marks the photon as absorbed if energy falls below transmission threshold.
    pub fn handle_antihalation_reflection<R: Rng>(&self, photon: &mut PhotonPacket, rng: &mut R) {
        // Two-way optical density attenuation through anti-halation backing layer:
        // Transmittance T = 10^(-2 * OD)
        let dye_transmittance = 10.0_f64.powf(-2.0 * self.antihalation_density);
        let effective_reflectance = self.antihalation_reflectance * dye_transmittance;

        // Attenuate photon energy
        photon.energy *= effective_reflectance;

        if photon.energy < 1e-4 {
            // Russian roulette for negligible energy
            if rng.gen_bool(0.1) {
                photon.energy /= 0.1;
            } else {
                photon.alive = false;
                return;
            }
        }

        // Reflect direction back upward into the emulsion (z component negated or diffuse reflection)
        let diffuse: bool = rng.gen_bool(0.4); // Mix of specular base reflection and diffuse backing
        if diffuse {
            // Cosine-weighted hemisphere reflection pointing upwards (-Z)
            let u1: f64 = rng.gen_range(0.0..1.0);
            let u2: f64 = rng.gen_range(0.0..1.0);
            let r = u1.sqrt();
            let phi = 2.0 * std::f64::consts::PI * u2;
            let x = r * phi.cos();
            let y = r * phi.sin();
            let z = -(1.0 - u1).max(0.0).sqrt();
            photon.direction = Vector3::new(x, y, z).normalize();
        } else {
            // Specular reflection: invert z component
            photon.direction.z = -photon.direction.z.abs();
            photon.direction = photon.direction.normalize();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn test_free_path_distribution() {
        let optics = GelatinOptics::default();
        let mut rng = StdRng::seed_from_u64(123);

        let mut sum_dist = 0.0;
        let n = 10_000;
        for _ in 0..n {
            sum_dist += optics.sample_free_path(&mut rng);
        }

        let mean_free_path = sum_dist / (n as f64);
        let expected_mfp = 1.0 / optics.extinction_coeff();
        assert!(
            (mean_free_path - expected_mfp).abs() / expected_mfp < 0.05,
            "Mean free path {} should match theoretical 1/mu_t {}",
            mean_free_path,
            expected_mfp
        );
    }

    #[test]
    fn test_antihalation_energy_attenuation() {
        let optics = GelatinOptics::default();
        let mut rng = StdRng::seed_from_u64(999);

        let mut photon =
            PhotonPacket::new(Point3::new(0.0, 0.0, 10.0), Vector3::new(0.0, 0.0, 1.0));
        let initial_energy = photon.energy;

        optics.handle_antihalation_reflection(&mut photon, &mut rng);

        assert!(
            photon.energy < initial_energy,
            "Anti-halation layer must attenuate reflected photon energy"
        );
        assert!(
            photon.direction.z < 0.0,
            "Reflected photon must travel upwards (-Z) into the emulsion"
        );
    }
}
