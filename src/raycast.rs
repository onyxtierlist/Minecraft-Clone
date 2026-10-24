//! Voxel traversal (Amanatides & Woo DDA) for block picking.

use glam::Vec3;

use crate::block::Block;
use crate::world::World;

pub struct Hit {
    /// The block that was hit.
    pub block_pos: (i32, i32, i32),
    /// Outward normal of the face that was hit (used for placement).
    pub normal: (i32, i32, i32),
}

/// March a ray through the voxel grid up to `max_dist`, stopping at the first
/// non-air block. Returns the hit plus the face normal so placement can offset.
pub fn raycast(world: &World, origin: Vec3, dir: Vec3, max_dist: f32) -> Option<Hit> {
    let dir = dir.normalize_or_zero();
    if dir == Vec3::ZERO {
        return None;
    }

    let mut x = origin.x.floor() as i32;
    let mut y = origin.y.floor() as i32;
    let mut z = origin.z.floor() as i32;

    let step = [
        if dir.x > 0.0 { 1 } else { -1 },
        if dir.y > 0.0 { 1 } else { -1 },
        if dir.z > 0.0 { 1 } else { -1 },
    ];

    let inv = [
        (1.0 / dir.x).abs(),
        (1.0 / dir.y).abs(),
        (1.0 / dir.z).abs(),
    ];

    let mut t_max = [
        intbound(origin.x, dir.x),
        intbound(origin.y, dir.y),
        intbound(origin.z, dir.z),
    ];
    let t_delta = [inv[0], inv[1], inv[2]];

    let mut normal = (0, 0, 0);
    let mut t = 0.0f32;

    while t <= max_dist {
        if !world.get(x, y, z).is_air() {
            return Some(Hit {
                block_pos: (x, y, z),
                normal,
            });
        }

        // Step along the axis with the nearest boundary.
        if t_max[0] < t_max[1] {
            if t_max[0] < t_max[2] {
                x += step[0];
                t = t_max[0];
                t_max[0] += t_delta[0];
                normal = (-step[0], 0, 0);
            } else {
                z += step[2];
                t = t_max[2];
                t_max[2] += t_delta[2];
                normal = (0, 0, -step[2]);
            }
        } else if t_max[1] < t_max[2] {
            y += step[1];
            t = t_max[1];
            t_max[1] += t_delta[1];
            normal = (0, -step[1], 0);
        } else {
            z += step[2];
            t = t_max[2];
            t_max[2] += t_delta[2];
            normal = (0, 0, -step[2]);
        }
    }

    // Never hit anything within range.
    None
}

/// Distance along `dir` from `s` to the next integer boundary.
fn intbound(s: f32, ds: f32) -> f32 {
    if ds == 0.0 {
        return f32::INFINITY;
    }
    if ds > 0.0 {
        // next integer above s, minus s
        let frac = s.ceil() - s;
        let frac = if frac == 0.0 { 1.0 } else { frac };
        frac / ds
    } else {
        let frac = s - s.floor();
        let frac = if frac == 0.0 { 1.0 } else { frac };
        frac / -ds
    }
}

/// Convenience: is the target cell free enough to place a block into?
pub fn can_place_at(world: &World, pos: (i32, i32, i32)) -> bool {
    matches!(world.get(pos.0, pos.1, pos.2), Block::Air | Block::Water)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::Block;
    use crate::chunk::ChunkPos;
    use crate::world::World;

    #[test]
    fn hits_block_straight_ahead() {
        let mut w = World::new(1);
        // Empty chunk so the only obstacle is the block we place.
        w.insert_air_chunk(ChunkPos::new(0, 0));
        w.set(5, 90, 3, Block::Stone);

        // Fire from (0.5, 90.5, 3.5) toward +X.
        let hit = raycast(&w, Vec3::new(0.5, 90.5, 3.5), Vec3::X, 32.0)
            .expect("should hit the stone");
        assert_eq!(hit.block_pos, (5, 90, 3));
        assert_eq!(hit.normal, (-1, 0, 0));
    }

    #[test]
    fn no_hit_when_clear() {
        let mut w = World::new(1);
        w.ensure_chunk(ChunkPos::new(0, 0));
        // Everything above y=40 is air in normal terrain; aim into the sky.
        let hit = raycast(&w, Vec3::new(0.5, 80.0, 0.5), Vec3::new(0.0, 1.0, 0.0), 16.0);
        assert!(hit.is_none());
    }

    #[test]
    fn vertical_hit_reports_top_normal() {
        let mut w = World::new(1);
        w.ensure_chunk(ChunkPos::new(0, 0));
        let surface = w.surface_y(8, 8);
        // Aim straight down from above the surface.
        let hit = raycast(&w, Vec3::new(8.5, surface as f32 + 3.0, 8.5), Vec3::NEG_Y, 8.0)
            .expect("should hit ground");
        assert_eq!(hit.normal, (0, 1, 0), "hitting from above -> +Y face");
    }
}
