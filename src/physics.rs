//! Player movement: gravity, jumping, creative flight, and AABB-vs-voxel collision.

use glam::Vec3;

use crate::world::World;

pub const HALF_WIDTH: f32 = 0.3;
pub const HEIGHT: f32 = 1.8;
pub const EYE_HEIGHT: f32 = 1.62;

/// Shrink the AABB max side by this much so cells only *touched* at the exact
/// boundary aren't counted as penetrated (otherwise pressing flush against a
/// wall misreads that wall as overlapping on other axes and climbs it).
const AABB_EPS: f32 = 1e-3;

const GRAVITY: f32 = 28.0;
const TERMINAL: f32 = 78.0;
const JUMP_SPEED: f32 = 8.6;
const WALK_SPEED: f32 = 4.6;
const SPRINT_SPEED: f32 = 7.2;
const FLY_SPEED: f32 = 12.0;

/// Per-frame movement intent (axes in -1..=1).
#[derive(Default, Clone, Copy)]
pub struct MoveInput {
    pub forward: f32,
    pub strafe: f32,
    pub jump: bool,
    pub sneak: bool,
    pub sprint: bool,
}

pub struct Player {
    /// Feet position (bottom-center of the AABB).
    pub pos: Vec3,
    pub vel: Vec3,
    pub on_ground: bool,
    pub fly: bool,
}

impl Player {
    pub fn new(pos: Vec3) -> Self {
        Self {
            pos,
            vel: Vec3::ZERO,
            on_ground: false,
            fly: false,
        }
    }

    pub fn eye(&self) -> Vec3 {
        self.pos + Vec3::new(0.0, EYE_HEIGHT, 0.0)
    }

    /// `cam_forward` is the full 3-D look vector; `cam_right` the strafe vector.
    pub fn update(
        &mut self,
        dt: f32,
        world: &World,
        input: &MoveInput,
        cam_forward: Vec3,
        cam_right: Vec3,
    ) {
        // Clamp huge frames (alt-tab, breakpoint) to avoid tunnelling through walls.
        let dt = dt.min(0.05);

        if self.fly {
            self.fly_step(dt, input, cam_forward, cam_right);
        } else {
            self.walk_step(dt, input, cam_forward, cam_right);
        }

        // Capture per-axis displacement *before* any axis is resolved, because
        // resolving one axis zeroes its velocity component.
        let dx = self.vel.x * dt;
        let dy = self.vel.y * dt;
        let dz = self.vel.z * dt;

        // Horizontal first, vertical last so landing is the final word on
        // `on_ground` for this frame.
        self.move_axis(world, 0, dx);
        self.move_axis(world, 2, dz);
        self.move_axis(world, 1, dy);
    }

    fn walk_step(&mut self, dt: f32, input: &MoveInput, cam_forward: Vec3, cam_right: Vec3) {
        // Horizontal intent, flattened so looking up/down doesn't slow you down.
        let mut dir = Vec3::ZERO;
        let flat_fwd = Vec3::new(cam_forward.x, 0.0, cam_forward.z);
        dir += flat_fwd.normalize_or_zero() * input.forward;
        dir += cam_right.normalize_or_zero() * input.strafe;
        dir = dir.normalize_or_zero();

        let speed = if input.sprint { SPRINT_SPEED } else { WALK_SPEED };
        self.vel.x = dir.x * speed;
        self.vel.z = dir.z * speed;

        self.vel.y -= GRAVITY * dt;
        if self.vel.y < -TERMINAL {
            self.vel.y = -TERMINAL;
        }
        if input.jump && self.on_ground {
            self.vel.y = JUMP_SPEED;
            self.on_ground = false;
        }
    }

    fn fly_step(&mut self, _dt: f32, input: &MoveInput, cam_forward: Vec3, cam_right: Vec3) {
        let speed = if input.sprint {
            FLY_SPEED * 2.0
        } else {
            FLY_SPEED
        };
        let mut v = Vec3::ZERO;
        v += cam_forward.normalize_or_zero() * input.forward;
        v += cam_right.normalize_or_zero() * input.strafe;
        if input.jump {
            v.y += 1.0;
        }
        if input.sneak {
            v.y -= 1.0;
        }
        self.vel = v.normalize_or_zero() * speed;
    }

    fn bounds_min(&self) -> Vec3 {
        self.pos + Vec3::new(-HALF_WIDTH, 0.0, -HALF_WIDTH)
    }

    fn bounds_max(&self) -> Vec3 {
        self.pos + Vec3::new(HALF_WIDTH, HEIGHT, HALF_WIDTH)
    }

    fn move_axis(&mut self, world: &World, axis: usize, delta: f32) {
        if delta == 0.0 {
            return;
        }

        // Distance from `pos[axis]` to the AABB face on each side.
        let (lower_ext, upper_ext) = if axis == 1 {
            (0.0, HEIGHT)
        } else {
            (-HALF_WIDTH, HALF_WIDTH)
        };

        // Move first, then push back out of any solid cell we entered.
        match axis {
            0 => self.pos.x += delta,
            1 => {
                self.pos.y += delta;
                self.on_ground = false;
            }
            _ => self.pos.z += delta,
        }

        let bmin = self.bounds_min();
        let bmax = self.bounds_max();

        // Half-open overlap test: [min, max) via `floor(min) ..= floor(max - eps)`.
        let x0 = bmin.x.floor() as i32;
        let x1 = (bmax.x - AABB_EPS).floor() as i32;
        let y0 = bmin.y.floor() as i32;
        let y1 = (bmax.y - AABB_EPS).floor() as i32;
        let z0 = bmin.z.floor() as i32;
        let z1 = (bmax.z - AABB_EPS).floor() as i32;

        let mut clamp_to: Option<f32> = None;

        for cy in y0..=y1 {
            for cx in x0..=x1 {
                for cz in z0..=z1 {
                    if !world.get(cx, cy, cz).is_solid() {
                        continue;
                    }
                    let cell = [cx, cy, cz][axis] as f32;
                    let cand = if delta > 0.0 {
                        cell - upper_ext
                    } else {
                        (cell + 1.0) - lower_ext
                    };
                    clamp_to = Some(match clamp_to {
                        None => cand,
                        Some(prev) => {
                            if delta > 0.0 {
                                prev.min(cand)
                            } else {
                                prev.max(cand)
                            }
                        }
                    });
                }
            }
        }

        if let Some(target) = clamp_to {
            match axis {
                0 => self.pos.x = target,
                1 => {
                    self.pos.y = target;
                    if delta < 0.0 {
                        self.on_ground = true;
                    }
                }
                _ => self.pos.z = target,
            }
            match axis {
                0 => self.vel.x = 0.0,
                1 => self.vel.y = 0.0,
                _ => self.vel.z = 0.0,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::Block;
    use crate::chunk::ChunkPos;

    fn flat_world() -> World {
        let mut w = World::new(1);
        for cz in -1..=1 {
            for cx in -1..=1 {
                w.insert_air_chunk(ChunkPos::new(cx, cz));
            }
        }
        // Solid floor at y=10.
        for z in -8..=8 {
            for x in -8..=8 {
                w.set(x, 10, z, Block::Stone);
            }
        }
        w
    }

    #[test]
    fn falls_and_lands_on_ground() {
        let w = flat_world();
        let mut p = Player::new(Vec3::new(0.5, 20.0, 0.5));
        let input = MoveInput::default();
        for _ in 0..600 {
            p.update(1.0 / 60.0, &w, &input, Vec3::Z, Vec3::X);
        }
        assert!(p.on_ground, "should have landed");
        // Feet rest on top of the y=10 block (top face at y=11).
        assert!((p.pos.y - 11.0).abs() < 1e-3, "y={}", p.pos.y);
    }

    #[test]
    fn wall_blocks_horizontal_motion() {
        let mut w = flat_world();
        for z in -4..=4 {
            for y in 11..=14 {
                w.set(5, y, z, Block::Stone);
            }
        }
        let mut p = Player::new(Vec3::new(0.5, 11.0, 0.5));
        let input = MoveInput {
            forward: 1.0,
            ..Default::default()
        };
        for _ in 0..300 {
            // Look along +X so "forward" pushes toward the wall.
            p.update(1.0 / 60.0, &w, &input, Vec3::X, Vec3::Z);
        }
        assert!(
            p.pos.x <= 5.0 - HALF_WIDTH + 1e-3,
            "player must not pass through the wall, x={}",
            p.pos.x
        );
    }

    #[test]
    fn does_not_fall_through_floor() {
        let w = flat_world();
        let mut p = Player::new(Vec3::new(0.5, 10.9, 0.5));
        let input = MoveInput::default();
        for _ in 0..120 {
            p.update(1.0 / 60.0, &w, &input, Vec3::Z, Vec3::X);
        }
        assert!(p.pos.y >= 11.0 - 1e-3, "fell through floor: {}", p.pos.y);
    }

    #[test]
    fn jump_leaves_ground_then_returns() {
        let w = flat_world();
        let mut p = Player::new(Vec3::new(0.5, 11.0, 0.5));
        let mut input = MoveInput::default();
        // Settle first.
        for _ in 0..60 {
            p.update(1.0 / 60.0, &w, &input, Vec3::Z, Vec3::X);
        }
        assert!(p.on_ground);
        input.jump = true;
        let start_y = p.pos.y;
        let mut reached = start_y;
        for _ in 0..60 {
            p.update(1.0 / 60.0, &w, &input, Vec3::Z, Vec3::X);
            reached = reached.max(p.pos.y);
        }
        assert!(reached > start_y + 1.0, "jump should gain height");
    }
}
