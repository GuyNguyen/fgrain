//! Silver halide crystal structures and morphology models.

use nalgebra::{Point3, UnitQuaternion, Vector3};
use std::sync::atomic::{AtomicU32, Ordering};

/// Crystal habit of silver halide grains.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Morphology {
    /// Cubic habit with isotropic (100) faces, typical of traditional emulsions like Kodak Tri-X.
    Cubic,

    /// Tabular plate-like crystals with high aspect ratio oriented parallel to the film plane (e.g. T-Max).
    Tabular,

    /// Octahedral habit with (111) faces, typical of ammoniacal emulsions.
    Octahedral,
}

/// An individual silver halide crystal in the gelatin emulsion.
#[derive(Debug)]
pub struct HalideCrystal {
    /// Crystal index.
    pub id: usize,

    /// Centroid position in micrometers.
    pub position: Point3<f64>,

    /// Semi-axes (half-extents) in local crystal coordinates (micrometers).
    pub dimensions: Vector3<f64>,

    /// Orientation relative to the emulsion layer.
    pub orientation: UnitQuaternion<f64>,

    /// Crystal morphology habit.
    pub morphology: Morphology,

    /// Quantum sensitivity scale factor.
    pub sensitivity: f64,

    /// Number of absorbed photons recorded during exposure.
    pub latent_photons: AtomicU32,

    /// Whether this crystal is developed into metallic silver.
    pub developed: bool,

    /// Clump identifier if merged with adjacent grains during development.
    pub clump_id: Option<usize>,

    /// Filament expansion factor after chemical reduction.
    pub filament_expansion: f64,
}

impl Clone for HalideCrystal {
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            position: self.position,
            dimensions: self.dimensions,
            orientation: self.orientation,
            morphology: self.morphology,
            sensitivity: self.sensitivity,
            latent_photons: AtomicU32::new(self.latent_photons.load(Ordering::Relaxed)),
            developed: self.developed,
            clump_id: self.clump_id,
            filament_expansion: self.filament_expansion,
        }
    }
}

impl HalideCrystal {
    /// Creates a new silver halide crystal with zero initial latent photons.
    ///
    /// # Arguments
    /// * `id` - Unique crystal identifier
    /// * `position` - 3D coordinates within the emulsion (micrometers)
    /// * `dimensions` - Semi-axes along principal dimensions
    /// * `orientation` - 3D orientation quaternion
    /// * `morphology` - Crystal habit
    /// * `sensitivity` - Sensitivity multiplier (typically around 1.0)
    pub fn new(
        id: usize,
        position: Point3<f64>,
        dimensions: Vector3<f64>,
        orientation: UnitQuaternion<f64>,
        morphology: Morphology,
        sensitivity: f64,
    ) -> Self {
        Self {
            id,
            position,
            dimensions,
            orientation,
            morphology,
            sensitivity,
            latent_photons: AtomicU32::new(0),
            developed: false,
            clump_id: None,
            filament_expansion: 1.0,
        }
    }

    /// Computes the axis-aligned bounding box (AABB) for this crystal in world coordinates.
    ///
    /// Accounts for crystal dimensions, morphology, and post-development filament expansion.
    #[must_use]
    pub fn aabb(&self) -> (Point3<f64>, Point3<f64>) {
        let expansion = if self.developed {
            self.filament_expansion
        } else {
            1.0
        };

        // For transformed ellipsoid / oriented box, maximum radius along any axis:
        let max_dim = self
            .dimensions
            .x
            .max(self.dimensions.y)
            .max(self.dimensions.z)
            * expansion;

        let min_pt = Point3::new(
            self.position.x - max_dim,
            self.position.y - max_dim,
            self.position.z - max_dim,
        );
        let max_pt = Point3::new(
            self.position.x + max_dim,
            self.position.y + max_dim,
            self.position.z + max_dim,
        );

        (min_pt, max_pt)
    }

    /// Tests for ray intersection against this crystal in world space.
    ///
    /// Returns `Some((t_enter, t_exit))` if the ray intersects the crystal volume,
    /// where `t_enter` and `t_exit` are ray parametric distances.
    #[must_use]
    pub fn intersect_ray(
        &self,
        ray_origin: &Point3<f64>,
        ray_dir: &Vector3<f64>,
    ) -> Option<(f64, f64)> {
        let expansion = if self.developed {
            self.filament_expansion
        } else {
            1.0
        };

        // Transform ray into crystal local coordinate frame:
        let local_origin = self
            .orientation
            .inverse_transform_point(&(Point3::from(ray_origin.coords - self.position.coords)));
        let local_dir = self.orientation.inverse_transform_vector(ray_dir);

        let dims = self.dimensions * expansion;

        match self.morphology {
            Morphology::Cubic => {
                // Oriented bounding box / slab intersection in local space
                let inv_d = Vector3::new(1.0 / local_dir.x, 1.0 / local_dir.y, 1.0 / local_dir.z);

                let t1 = (-dims.x - local_origin.x) * inv_d.x;
                let t2 = (dims.x - local_origin.x) * inv_d.x;
                let t3 = (-dims.y - local_origin.y) * inv_d.y;
                let t4 = (dims.y - local_origin.y) * inv_d.y;
                let t5 = (-dims.z - local_origin.z) * inv_d.z;
                let t6 = (dims.z - local_origin.z) * inv_d.z;

                let t_min = t1.min(t2).max(t3.min(t4)).max(t5.min(t6));
                let t_max = t1.max(t2).min(t3.max(t4)).min(t5.max(t6));

                if t_max >= t_min && t_max > 0.0 {
                    Some((t_min.max(0.0), t_max))
                } else {
                    None
                }
            }
            Morphology::Tabular | Morphology::Octahedral => {
                // Approximate tabular/octahedral grains with an oriented ellipsoid
                // x^2/a^2 + y^2/b^2 + z^2/c^2 <= 1
                let p = Vector3::new(
                    local_origin.x / dims.x,
                    local_origin.y / dims.y,
                    local_origin.z / dims.z,
                );
                let d = Vector3::new(
                    local_dir.x / dims.x,
                    local_dir.y / dims.y,
                    local_dir.z / dims.z,
                );

                let a = d.dot(&d);
                let b = 2.0 * p.dot(&d);
                let c = p.dot(&p) - 1.0;

                let discriminant = b * b - 4.0 * a * c;
                if discriminant < 0.0 {
                    return None;
                }

                let sqrt_disc = discriminant.sqrt();
                let t0 = (-b - sqrt_disc) / (2.0 * a);
                let t1 = (-b + sqrt_disc) / (2.0 * a);

                let t_enter = t0.min(t1);
                let t_exit = t0.max(t1);

                if t_exit > 0.0 {
                    Some((t_enter.max(0.0), t_exit))
                } else {
                    None
                }
            }
        }
    }

    /// Records the absorption of photon energy, incrementing the latent image counter.
    #[inline]
    pub fn record_absorption(&self, count: u32) {
        self.latent_photons.fetch_add(count, Ordering::Relaxed);
    }

    /// Returns the accumulated absorbed photon count.
    #[inline]
    #[must_use]
    pub fn absorbed_photons(&self) -> u32 {
        self.latent_photons.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cubic_crystal_intersection() {
        let crystal = HalideCrystal::new(
            0,
            Point3::new(10.0, 10.0, 10.0),
            Vector3::new(1.0, 1.0, 1.0),
            UnitQuaternion::identity(),
            Morphology::Cubic,
            1.0,
        );

        // Ray passing directly through the center along Z
        let ray_orig = Point3::new(10.0, 10.0, 0.0);
        let ray_dir = Vector3::new(0.0, 0.0, 1.0);

        let hit = crystal.intersect_ray(&ray_orig, &ray_dir);
        assert!(hit.is_some(), "Ray should hit cubic crystal");
        let (t0, t1) = hit.unwrap();
        assert!((t0 - 9.0).abs() < 1e-4);
        assert!((t1 - 11.0).abs() < 1e-4);

        // Ray missing the crystal
        let ray_miss = Point3::new(20.0, 10.0, 0.0);
        assert!(crystal.intersect_ray(&ray_miss, &ray_dir).is_none());
    }

    #[test]
    fn test_tabular_crystal_intersection() {
        let crystal = HalideCrystal::new(
            1,
            Point3::new(0.0, 0.0, 5.0),
            Vector3::new(2.0, 2.0, 0.2), // Flat tablet: 2um radius, 0.2um half-thickness
            UnitQuaternion::identity(),
            Morphology::Tabular,
            1.0,
        );

        // Ray passing through center
        let ray_orig = Point3::new(0.0, 0.0, 0.0);
        let ray_dir = Vector3::new(0.0, 0.0, 1.0);
        let hit = crystal.intersect_ray(&ray_orig, &ray_dir);
        assert!(hit.is_some(), "Ray should hit tabular crystal");
        let (t0, t1) = hit.unwrap();
        assert!((t0 - 4.8).abs() < 1e-4);
        assert!((t1 - 5.2).abs() < 1e-4);
    }
}
