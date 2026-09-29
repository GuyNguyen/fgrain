//! 3D Poisson-disk sampling for crystal placement.

use nalgebra::{Point3, Vector3};
use rand::Rng;

/// Generates 3D points inside the volume `[0, width] x [0, height] x [0, depth]`
/// using Bridson's 3D Poisson-disk sampling algorithm.
///
/// # Arguments
/// * `bounds` - Extents of the volume along X, Y, Z (in micrometers)
/// * `r_min` - Minimum Euclidean distance between any two crystal centroids (micrometers)
/// * `k_samples` - Candidate samples per active point (typically 30)
/// * `rng` - Random number generator
pub fn sample_poisson_3d<R: Rng>(
    bounds: &Vector3<f64>,
    r_min: f64,
    k_samples: usize,
    rng: &mut R,
) -> Vec<Point3<f64>> {
    if bounds.x <= 0.0 || bounds.y <= 0.0 || bounds.z <= 0.0 || r_min <= 0.0 {
        return Vec::new();
    }

    // Grid cell size <= r_min / sqrt(3) ensures at most one point per cell.
    let cell_size = r_min / 3.0_f64.sqrt();
    let grid_w = (bounds.x / cell_size).ceil() as usize;
    let grid_h = (bounds.y / cell_size).ceil() as usize;
    let grid_d = (bounds.z / cell_size).ceil() as usize;

    let total_cells = grid_w * grid_h * grid_d;
    let mut grid: Vec<Option<usize>> = vec![None; total_cells];

    let grid_index =
        |gx: usize, gy: usize, gz: usize| -> usize { gx + gy * grid_w + gz * grid_w * grid_h };

    let point_to_grid = |pt: &Point3<f64>| -> (usize, usize, usize) {
        let gx = ((pt.x / cell_size).floor() as isize).clamp(0, grid_w as isize - 1) as usize;
        let gy = ((pt.y / cell_size).floor() as isize).clamp(0, grid_h as isize - 1) as usize;
        let gz = ((pt.z / cell_size).floor() as isize).clamp(0, grid_d as isize - 1) as usize;
        (gx, gy, gz)
    };

    let mut points: Vec<Point3<f64>> = Vec::new();
    let mut active_indices: Vec<usize> = Vec::new();

    let first_pt = Point3::new(
        rng.gen_range(0.0..bounds.x),
        rng.gen_range(0.0..bounds.y),
        rng.gen_range(0.0..bounds.z),
    );

    let (gx, gy, gz) = point_to_grid(&first_pt);
    grid[grid_index(gx, gy, gz)] = Some(0);
    points.push(first_pt);
    active_indices.push(0);

    let r_min_sq = r_min * r_min;

    while !active_indices.is_empty() {
        let active_pick = rng.gen_range(0..active_indices.len());
        let point_idx = active_indices[active_pick];
        let base_pt = points[point_idx];

        let mut found = false;

        for _ in 0..k_samples {
            let u: f64 = rng.gen_range(0.0..1.0);
            let cos_theta: f64 = rng.gen_range(-1.0..1.0);
            let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
            let phi: f64 = rng.gen_range(0.0..std::f64::consts::TAU);

            // Inverse transform sampling for uniform volume distribution in spherical shell [r_min, 2 * r_min]:
            // CDF F(r) = (r^3 - r_min^3) / ((2*r_min)^3 - r_min^3) = (r^3 - r_min^3) / (7 * r_min^3)
            let radius = (7.0 * u + 1.0).cbrt() * r_min;

            let candidate = Point3::new(
                base_pt.x + radius * sin_theta * phi.cos(),
                base_pt.y + radius * sin_theta * phi.sin(),
                base_pt.z + radius * cos_theta,
            );

            if candidate.x < 0.0
                || candidate.x >= bounds.x
                || candidate.y < 0.0
                || candidate.y >= bounds.y
                || candidate.z < 0.0
                || candidate.z >= bounds.z
            {
                continue;
            }

            // Check distance to neighbors in adjacent grid cells (within 2 cells)
            let (cgx, cgy, cgz) = point_to_grid(&candidate);
            let min_x = cgx.saturating_sub(2);
            let max_x = (cgx + 2).min(grid_w - 1);
            let min_y = cgy.saturating_sub(2);
            let max_y = (cgy + 2).min(grid_h - 1);
            let min_z = cgz.saturating_sub(2);
            let max_z = (cgz + 2).min(grid_d - 1);

            let mut too_close = false;
            'check: for z in min_z..=max_z {
                for y in min_y..=max_y {
                    for x in min_x..=max_x {
                        if let Some(existing_idx) = grid[grid_index(x, y, z)] {
                            let diff = points[existing_idx] - candidate;
                            if diff.dot(&diff) < r_min_sq {
                                too_close = true;
                                break 'check;
                            }
                        }
                    }
                }
            }

            if !too_close {
                let new_idx = points.len();
                grid[grid_index(cgx, cgy, cgz)] = Some(new_idx);
                points.push(candidate);
                active_indices.push(new_idx);
                found = true;
                break;
            }
        }

        if !found {
            active_indices.swap_remove(active_pick);
        }
    }

    points
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn test_poisson_min_distance() {
        let bounds = Vector3::new(20.0, 20.0, 10.0);
        let r_min = 2.0;
        let mut rng = StdRng::seed_from_u64(42);

        let points = sample_poisson_3d(&bounds, r_min, 30, &mut rng);
        assert!(!points.is_empty(), "Should generate at least one point");

        let r_min_sq = (r_min - 1e-6).powi(2);

        // Verify pairwise distances between all points
        for i in 0..points.len() {
            for j in (i + 1)..points.len() {
                let dist_sq = (points[i] - points[j]).norm_squared();
                assert!(
                    dist_sq >= r_min_sq,
                    "Points {} and {} violated minimum distance: {} < {}",
                    i,
                    j,
                    dist_sq.sqrt(),
                    r_min
                );
            }
        }
    }
}
