//! Chemical development and grain clumping model.
//!
//! Models reduction of exposed crystals into metallic silver filaments
//! and proximity-based grain clumping in dense highlight regions.

use crate::bvh::BVH;
use crate::crystal::HalideCrystal;
use std::collections::VecDeque;

/// Configuration for the photographic chemical development pass.
#[derive(Debug, Clone)]
pub struct DevelopmentConfig {
    /// Minimum number of absorbed photons required to form a developable latent image center
    /// (the Gurney-Mott critical speck size, typically 3 to 4 photons).
    pub threshold_photons: u32,

    /// Volume expansion factor of metallic silver filaments relative to unreduced crystal size.
    pub filament_expansion: f64,

    /// Proximity distance multiplier for catalytic clumping.
    /// When neighboring developing grains touch or come within this proximity, their filament
    /// meshes merge into contiguous clumps.
    pub clumping_proximity_factor: f64,

    /// Chemical developer contrast index ($\gamma$). Higher values yield steeper threshold curves.
    pub developer_contrast: f64,
}

impl Default for DevelopmentConfig {
    fn default() -> Self {
        Self {
            threshold_photons: 3,
            filament_expansion: 1.8,
            clumping_proximity_factor: 1.25,
            developer_contrast: 0.75,
        }
    }
}

/// Statistics and diagnostic metrics from the chemical development pass.
#[derive(Debug, Clone, Default)]
pub struct DevelopmentReport {
    /// Total number of crystals in the emulsion.
    pub total_crystals: usize,
    /// Number of crystals successfully reduced to metallic silver.
    pub developed_crystals: usize,
    /// Number of undeveloped crystals cleared by the fixing bath.
    pub fixed_crystals: usize,
    /// Number of distinct silver clumps formed.
    pub total_clumps: usize,
    /// Maximum number of crystals participating in a single highlight clump.
    pub max_clump_size: usize,
    /// Average number of crystals per clump.
    pub avg_clump_size: f64,
}

/// Evaluates latent energy against threshold photons, applies filament expansion,
/// and groups contiguous developed grains into clumps.
pub fn develop_emulsion(
    crystals: &mut [HalideCrystal],
    bvh: &BVH,
    config: &DevelopmentConfig,
) -> DevelopmentReport {
    let total_crystals = crystals.len();
    if total_crystals == 0 {
        return DevelopmentReport::default();
    }

    // Evaluate latent image threshold for each crystal
    let mut developed_count = 0;
    for crystal in crystals.iter_mut() {
        let absorbed = crystal.absorbed_photons();
        if absorbed >= config.threshold_photons {
            crystal.developed = true;
            crystal.filament_expansion = config.filament_expansion;
            developed_count += 1;
        } else {
            // Undeveloped crystals are dissolved during fixing
            crystal.developed = false;
            crystal.filament_expansion = 0.0;
        }
    }

    let fixed_crystals = total_crystals - developed_count;

    // Group contiguous developed grains into clumps via connected components
    let mut visited = vec![false; total_crystals];
    let mut clump_id_counter = 0;
    let mut max_clump_size = 0;
    let mut total_clumped_crystals = 0;

    // Maximum base crystal dimension across the emulsion to bound BVH radius queries
    let max_base_dim = crystals
        .iter()
        .map(|c| c.dimensions.x.max(c.dimensions.y).max(c.dimensions.z))
        .fold(0.0f64, f64::max);

    let mut queue = VecDeque::new();
    let mut neighbors = Vec::new();
    let mut clump_members = Vec::new();

    for i in 0..total_crystals {
        if !crystals[i].developed || visited[i] {
            continue;
        }

        let current_clump_id = clump_id_counter;
        clump_id_counter += 1;

        visited[i] = true;
        crystals[i].clump_id = Some(current_clump_id);
        queue.push_back(i);
        clump_members.clear();
        clump_members.push(i);

        while let Some(curr_idx) = queue.pop_front() {
            let curr_pos = crystals[curr_idx].position;
            let r_curr = crystals[curr_idx]
                .dimensions
                .x
                .max(crystals[curr_idx].dimensions.y)
                .max(crystals[curr_idx].dimensions.z)
                * crystals[curr_idx].filament_expansion;

            // Search radius in BVH accounts for both current crystal and largest possible neighbor
            let bvh_search_radius = (r_curr + max_base_dim * config.filament_expansion)
                * config.clumping_proximity_factor;

            neighbors.clear();
            bvh.query_radius(&curr_pos, bvh_search_radius, crystals, &mut neighbors);

            for &neighbor_idx in &neighbors {
                if neighbor_idx == curr_idx
                    || visited[neighbor_idx]
                    || !crystals[neighbor_idx].developed
                {
                    continue;
                }

                let r_neighbor = crystals[neighbor_idx]
                    .dimensions
                    .x
                    .max(crystals[neighbor_idx].dimensions.y)
                    .max(crystals[neighbor_idx].dimensions.z)
                    * crystals[neighbor_idx].filament_expansion;
                let touch_dist = (r_curr + r_neighbor) * config.clumping_proximity_factor;
                let diff = crystals[neighbor_idx].position - curr_pos;

                if diff.dot(&diff) <= touch_dist * touch_dist {
                    visited[neighbor_idx] = true;
                    crystals[neighbor_idx].clump_id = Some(current_clump_id);
                    queue.push_back(neighbor_idx);
                    clump_members.push(neighbor_idx);
                }
            }
        }

        let current_clump_size = clump_members.len();
        if current_clump_size > max_clump_size {
            max_clump_size = current_clump_size;
        }
        total_clumped_crystals += current_clump_size;

        // Catalytic infectious development: touching metallic silver filaments
        // sinter together, slightly expanding effective optical filament volume
        if current_clump_size > 1 {
            let clump_boost = 1.15 + 0.18 * ((current_clump_size.min(6) - 1) as f64);
            for &idx in &clump_members {
                crystals[idx].filament_expansion *= clump_boost;
            }
        }
    }

    let avg_clump_size = if clump_id_counter > 0 {
        total_clumped_crystals as f64 / clump_id_counter as f64
    } else {
        0.0
    };

    DevelopmentReport {
        total_crystals,
        developed_crystals: developed_count,
        fixed_crystals,
        total_clumps: clump_id_counter,
        max_clump_size,
        avg_clump_size,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crystal::Morphology;
    use nalgebra::{Point3, UnitQuaternion, Vector3};

    #[test]
    fn test_boolean_thresholding_and_clumping() {
        let mut crystals = Vec::new();

        // 3 closely spaced crystals in a cluster (simulating a highlight)
        for i in 0..3 {
            let c = HalideCrystal::new(
                i,
                Point3::new(i as f64 * 0.8, 0.0, 5.0),
                Vector3::new(0.5, 0.5, 0.5),
                UnitQuaternion::identity(),
                Morphology::Cubic,
                1.0,
            );
            // High exposure: 10 photons absorbed
            c.record_absorption(10);
            crystals.push(c);
        }

        // 1 isolated crystal in midtone
        let c_iso = HalideCrystal::new(
            3,
            Point3::new(10.0, 0.0, 5.0),
            Vector3::new(0.5, 0.5, 0.5),
            UnitQuaternion::identity(),
            Morphology::Cubic,
            1.0,
        );
        c_iso.record_absorption(4);
        crystals.push(c_iso);

        // 1 unexposed crystal below threshold
        let c_unexp = HalideCrystal::new(
            4,
            Point3::new(20.0, 0.0, 5.0),
            Vector3::new(0.5, 0.5, 0.5),
            UnitQuaternion::identity(),
            Morphology::Cubic,
            1.0,
        );
        c_unexp.record_absorption(1); // Below threshold of 3
        crystals.push(c_unexp);

        let bvh = BVH::build(&crystals);
        let config = DevelopmentConfig {
            threshold_photons: 3,
            filament_expansion: 2.0,
            clumping_proximity_factor: 1.5,
            developer_contrast: 0.75,
        };

        let report = develop_emulsion(&mut crystals, &bvh, &config);

        assert_eq!(report.developed_crystals, 4);
        assert_eq!(report.fixed_crystals, 1);
        assert!(
            !crystals[4].developed,
            "Crystal 4 should be fixed/dissolved"
        );

        // The 3 clustered crystals should merge into 1 clump
        assert_eq!(crystals[0].clump_id, crystals[1].clump_id);
        assert_eq!(crystals[1].clump_id, crystals[2].clump_id);

        // The isolated crystal should be in a separate clump
        assert_ne!(crystals[3].clump_id, crystals[0].clump_id);

        assert_eq!(report.total_clumps, 2);
        assert_eq!(report.max_clump_size, 3);
    }
}
