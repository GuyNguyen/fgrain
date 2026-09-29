//! Linear Bounding Volume Hierarchy (BVH) for ray-crystal intersection and radius queries.

use crate::crystal::HalideCrystal;
use nalgebra::{Point3, Vector3};

/// Axis-Aligned Bounding Box (AABB) in 3D Euclidean space.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AABB {
    /// Minimum corner coordinates.
    pub min: Point3<f64>,
    /// Maximum corner coordinates.
    pub max: Point3<f64>,
}

impl Default for AABB {
    fn default() -> Self {
        Self::empty()
    }
}

impl AABB {
    /// Creates an empty bounding box with inverted infinite boundaries.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            min: Point3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY),
            max: Point3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY),
        }
    }

    /// Creates a new AABB from min and max points.
    #[must_use]
    pub fn new(min: Point3<f64>, max: Point3<f64>) -> Self {
        Self { min, max }
    }

    /// Expands this AABB to enclose another AABB.
    #[must_use]
    pub fn union(&self, other: &Self) -> Self {
        Self {
            min: Point3::new(
                self.min.x.min(other.min.x),
                self.min.y.min(other.min.y),
                self.min.z.min(other.min.z),
            ),
            max: Point3::new(
                self.max.x.max(other.max.x),
                self.max.y.max(other.max.y),
                self.max.z.max(other.max.z),
            ),
        }
    }

    /// Expands this AABB to enclose a 3D point.
    #[must_use]
    pub fn union_point(&self, point: &Point3<f64>) -> Self {
        Self {
            min: Point3::new(
                self.min.x.min(point.x),
                self.min.y.min(point.y),
                self.min.z.min(point.z),
            ),
            max: Point3::new(
                self.max.x.max(point.x),
                self.max.y.max(point.y),
                self.max.z.max(point.z),
            ),
        }
    }

    /// Computes the geometric centroid of this bounding box.
    #[must_use]
    pub fn centroid(&self) -> Point3<f64> {
        Point3::new(
            0.5 * (self.min.x + self.max.x),
            0.5 * (self.min.y + self.max.y),
            0.5 * (self.min.z + self.max.z),
        )
    }

    /// Computes the surface area of this bounding box.
    #[must_use]
    pub fn surface_area(&self) -> f64 {
        let extent = self.max - self.min;
        if extent.x < 0.0 || extent.y < 0.0 || extent.z < 0.0 {
            return 0.0;
        }
        2.0 * (extent.x * extent.y + extent.y * extent.z + extent.z * extent.x)
    }

    /// Evaluates ray intersection using the Kay-Kajiya slab method.
    ///
    /// Returns `Some((t_near, t_far))` if the ray intersects the box volume.
    #[inline]
    #[must_use]
    pub fn intersect_ray(
        &self,
        ray_origin: &Point3<f64>,
        ray_inv_dir: &Vector3<f64>,
    ) -> Option<(f64, f64)> {
        let t1x = (self.min.x - ray_origin.x) * ray_inv_dir.x;
        let t2x = (self.max.x - ray_origin.x) * ray_inv_dir.x;
        let t_min_x = t1x.min(t2x);
        let t_max_x = t1x.max(t2x);

        let t1y = (self.min.y - ray_origin.y) * ray_inv_dir.y;
        let t2y = (self.max.y - ray_origin.y) * ray_inv_dir.y;
        let t_min_y = t1y.min(t2y);
        let t_max_y = t1y.max(t2y);

        let t1z = (self.min.z - ray_origin.z) * ray_inv_dir.z;
        let t2z = (self.max.z - ray_origin.z) * ray_inv_dir.z;
        let t_min_z = t1z.min(t2z);
        let t_max_z = t1z.max(t2z);

        let t_enter = t_min_x.max(t_min_y).max(t_min_z);
        let t_exit = t_max_x.min(t_max_y).min(t_max_z);

        if t_exit >= t_enter && t_exit > 0.0 {
            Some((t_enter.max(0.0), t_exit))
        } else {
            None
        }
    }
}

/// A node in the flattened linear BVH. Fits within 64 bytes (1 cache line).
#[derive(Debug, Clone)]
pub struct BVHNode {
    /// Bounding volume enclosing all primitives in this subtree. (48 bytes)
    pub aabb: AABB,
    /// Left child node index (0 for leaf nodes).
    pub left_child: u32,
    /// Right child node index (0 for leaf nodes).
    pub right_child: u32,
    /// Offset into the reordered crystal index buffer (leaf nodes only).
    pub offset: u32,
    /// Number of primitives in this leaf node (0 for interior nodes).
    pub count: u32,
}

/// A Bounding Volume Hierarchy accelerating spatial and ray queries over crystals.
#[derive(Debug, Clone)]
pub struct BVH {
    /// Flat array of BVH nodes in contiguous memory.
    pub nodes: Vec<BVHNode>,
    /// Reordered crystal indices referenced by leaf nodes.
    pub indices: Vec<usize>,
}

impl BVH {
    /// Maximum number of primitives permitted in a single leaf node.
    pub const MAX_LEAF_PRIMITIVES: usize = 4;

    /// Builds a BVH over a slice of silver halide crystals using spatial median partitioning.
    #[must_use]
    pub fn build(crystals: &[HalideCrystal]) -> Self {
        if crystals.is_empty() {
            return Self {
                nodes: Vec::new(),
                indices: Vec::new(),
            };
        }

        let mut indices: Vec<usize> = (0..crystals.len()).collect();
        let mut nodes: Vec<BVHNode> = Vec::with_capacity(crystals.len() * 2);

        // Precompute AABBs and centroids for all crystals
        let aabbs: Vec<AABB> = crystals
            .iter()
            .map(|c| {
                let (min, max) = c.aabb();
                AABB::new(min, max)
            })
            .collect();
        let centroids: Vec<Point3<f64>> = aabbs.iter().map(|b| b.centroid()).collect();

        // Recursively subdivide
        Self::subdivide(
            &mut nodes,
            &mut indices,
            &aabbs,
            &centroids,
            0,
            crystals.len(),
        );

        Self { nodes, indices }
    }

    fn subdivide(
        nodes: &mut Vec<BVHNode>,
        indices: &mut [usize],
        aabbs: &[AABB],
        centroids: &[Point3<f64>],
        start: usize,
        count: usize,
    ) -> u32 {
        let node_idx = nodes.len() as u32;

        // Compute enclosing bounding box for this range
        let mut box_all = AABB::empty();
        let mut centroid_box = AABB::empty();
        for &idx in &indices[start..start + count] {
            box_all = box_all.union(&aabbs[idx]);
            centroid_box = centroid_box.union_point(&centroids[idx]);
        }

        // If primitive count is small enough or centroids are degenerate, create a leaf
        if count <= Self::MAX_LEAF_PRIMITIVES {
            nodes.push(BVHNode {
                aabb: box_all,
                left_child: 0,
                right_child: 0,
                offset: start as u32,
                count: count as u32,
            });
            return node_idx;
        }

        // Choose split axis along largest centroid extent
        let extent = centroid_box.max - centroid_box.min;
        let axis = if extent.x >= extent.y && extent.x >= extent.z {
            0
        } else if extent.y >= extent.z {
            1
        } else {
            2
        };

        if extent[axis] <= 1e-7 {
            // Degenerate spatial extent: force leaf
            nodes.push(BVHNode {
                aabb: box_all,
                left_child: 0,
                right_child: 0,
                offset: start as u32,
                count: count as u32,
            });
            return node_idx;
        }

        // Spatial median partition
        let mid_point = 0.5 * (centroid_box.min[axis] + centroid_box.max[axis]);
        let mut i = start;
        let mut j = start + count - 1;

        while i <= j {
            let idx = indices[i];
            if centroids[idx][axis] < mid_point {
                i += 1;
            } else {
                indices.swap(i, j);
                if j == 0 {
                    break;
                }
                j -= 1;
            }
        }

        let left_count = i - start;
        let split_mid = if left_count == 0 || left_count == count {
            count / 2
        } else {
            left_count
        };

        // Reserve space for this interior node
        nodes.push(BVHNode {
            aabb: box_all,
            left_child: 0,
            right_child: 0,
            offset: 0,
            count: 0,
        });

        // Build left child
        let left_child = Self::subdivide(nodes, indices, aabbs, centroids, start, split_mid);

        // Build right child
        let right_child = Self::subdivide(
            nodes,
            indices,
            aabbs,
            centroids,
            start + split_mid,
            count - split_mid,
        );

        // Update interior node with explicit child indices
        nodes[node_idx as usize].left_child = left_child;
        nodes[node_idx as usize].right_child = right_child;

        node_idx
    }

    /// Traverses the BVH to find all crystals intersecting the specified ray.
    ///
    /// For each hit, invokes `callback(crystal_idx, t_enter, t_exit)`.
    pub fn intersect_ray_all<F>(
        &self,
        ray_origin: &Point3<f64>,
        ray_dir: &Vector3<f64>,
        crystals: &[HalideCrystal],
        mut callback: F,
    ) where
        F: FnMut(usize, f64, f64),
    {
        if self.nodes.is_empty() {
            return;
        }

        let inv_dir = Vector3::new(1.0 / ray_dir.x, 1.0 / ray_dir.y, 1.0 / ray_dir.z);

        // Fixed-size stack avoiding heap allocations during critical ray loop
        let mut stack: [u32; 64] = [0; 64];
        let mut stack_ptr = 0;
        stack[stack_ptr] = 0;
        stack_ptr += 1;

        while stack_ptr > 0 {
            stack_ptr -= 1;
            let node_idx = stack[stack_ptr] as usize;
            let node = &self.nodes[node_idx];

            if node.aabb.intersect_ray(ray_origin, &inv_dir).is_some() {
                if node.count > 0 {
                    // Leaf node: test all primitives in this leaf
                    let start = node.offset as usize;
                    let end = start + node.count as usize;
                    for &crystal_idx in &self.indices[start..end] {
                        let crystal = &crystals[crystal_idx];
                        if let Some((t0, t1)) = crystal.intersect_ray(ray_origin, ray_dir) {
                            callback(crystal_idx, t0, t1);
                        }
                    }
                } else {
                    // Interior node: push both children
                    if stack_ptr + 2 < stack.len() {
                        stack[stack_ptr] = node.left_child;
                        stack[stack_ptr + 1] = node.right_child;
                        stack_ptr += 2;
                    }
                }
            }
        }
    }

    /// Queries all crystal indices within distance `radius` of `center`.
    pub fn query_radius(
        &self,
        center: &Point3<f64>,
        radius: f64,
        crystals: &[HalideCrystal],
        results: &mut Vec<usize>,
    ) {
        if self.nodes.is_empty() {
            return;
        }

        let radius_sq = radius * radius;
        let query_box = AABB::new(
            Point3::new(center.x - radius, center.y - radius, center.z - radius),
            Point3::new(center.x + radius, center.y + radius, center.z + radius),
        );

        let mut stack: [u32; 64] = [0; 64];
        let mut stack_ptr = 0;
        stack[stack_ptr] = 0;
        stack_ptr += 1;

        while stack_ptr > 0 {
            stack_ptr -= 1;
            let node_idx = stack[stack_ptr] as usize;
            let node = &self.nodes[node_idx];

            // Test AABB overlap with query_box
            let overlaps = node.aabb.min.x <= query_box.max.x
                && node.aabb.max.x >= query_box.min.x
                && node.aabb.min.y <= query_box.max.y
                && node.aabb.max.y >= query_box.min.y
                && node.aabb.min.z <= query_box.max.z
                && node.aabb.max.z >= query_box.min.z;

            if overlaps {
                if node.count > 0 {
                    let start = node.offset as usize;
                    let end = start + node.count as usize;
                    for &c_idx in &self.indices[start..end] {
                        let diff = crystals[c_idx].position - center;
                        if diff.dot(&diff) <= radius_sq {
                            results.push(c_idx);
                        }
                    }
                } else if stack_ptr + 2 < stack.len() {
                    stack[stack_ptr] = node.left_child;
                    stack[stack_ptr + 1] = node.right_child;
                    stack_ptr += 2;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crystal::Morphology;
    use nalgebra::UnitQuaternion;

    #[test]
    fn test_bvh_construction_and_query() {
        let mut crystals = Vec::new();
        for i in 0..20 {
            crystals.push(HalideCrystal::new(
                i,
                Point3::new(i as f64 * 3.0, 0.0, 10.0),
                Vector3::new(1.0, 1.0, 1.0),
                UnitQuaternion::identity(),
                Morphology::Cubic,
                1.0,
            ));
        }

        let bvh = BVH::build(&crystals);
        assert!(!bvh.nodes.is_empty(), "BVH should have nodes");

        // Ray targeted at crystal #5 (x=15, y=0, z=10)
        let ray_orig = Point3::new(15.0, 0.0, 0.0);
        let ray_dir = Vector3::new(0.0, 0.0, 1.0);

        let mut hits = Vec::new();
        bvh.intersect_ray_all(&ray_orig, &ray_dir, &crystals, |idx, t0, t1| {
            hits.push((idx, t0, t1));
        });

        assert_eq!(hits.len(), 1, "Should hit exactly one crystal");
        assert_eq!(hits[0].0, 5, "Hit should be crystal index 5");
    }

    #[test]
    fn test_bvh_radius_query() {
        let mut crystals = Vec::new();
        for i in 0..10 {
            crystals.push(HalideCrystal::new(
                i,
                Point3::new(i as f64, 0.0, 0.0),
                Vector3::new(0.4, 0.4, 0.4),
                UnitQuaternion::identity(),
                Morphology::Cubic,
                1.0,
            ));
        }

        let bvh = BVH::build(&crystals);
        let mut results = Vec::new();
        bvh.query_radius(&Point3::new(5.0, 0.0, 0.0), 1.5, &crystals, &mut results);

        // Crystals at x=4, 5, 6 should be within radius 1.5
        results.sort();
        assert_eq!(results, vec![4, 5, 6]);
    }
}
