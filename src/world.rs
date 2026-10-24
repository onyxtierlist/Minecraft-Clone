//! Deterministic terrain generation and the block-access API over loaded chunks.

use std::collections::HashMap;

use crate::block::Block;
use crate::chunk::{Chunk, ChunkPos, CHUNK_SIZE, WORLD_HEIGHT};
use crate::noise::{fbm2, fbm3, hash2};

pub const SEA_LEVEL: i32 = 32;
const BASE_HEIGHT: f64 = 41.0;
// fbm concentrates near 0.5, so these need to be large to produce a real
// spread: valleys below sea level (lakes/oceans) and proper hills.
const CONTINENTAL_AMP: f64 = 120.0;
const HILL_AMP: f64 = 40.0;

const TREE_SEED: u64 = 0x7EE5_1A9C_3B21_D4F6;
const TREE_CHANCE: u64 = 2; // percent of eligible columns
const CANOPY: i32 = 3; // leaf stamp radius reach (keeps trees seamless across chunks)

pub struct World {
    pub seed: u64,
    chunks: HashMap<ChunkPos, Chunk>,
}

impl World {
    pub fn new(seed: u64) -> Self {
        Self {
            seed,
            chunks: HashMap::new(),
        }
    }

    pub fn chunk(&self, pos: ChunkPos) -> Option<&Chunk> {
        self.chunks.get(&pos)
    }

    pub fn chunk_mut(&mut self, pos: ChunkPos) -> Option<&mut Chunk> {
        self.chunks.get_mut(&pos)
    }

    pub fn ensure_chunk(&mut self, pos: ChunkPos) -> &mut Chunk {
        self.chunks.entry(pos).or_insert_with(|| generate_chunk(self.seed, pos))
    }

    /// Test-only: insert a chunk pre-filled with air so tests fully control the
    /// geometry instead of inheriting generated terrain.
    #[cfg(test)]
    pub fn insert_air_chunk(&mut self, pos: ChunkPos) {
        self.chunks.insert(pos, Chunk::filled(Block::Air));
    }

    /// Block at world coordinates; `Air` when outside vertical bounds or ungenerated.
    pub fn get(&self, wx: i32, wy: i32, wz: i32) -> Block {
        if wy < 0 || wy >= WORLD_HEIGHT as i32 {
            return Block::Air;
        }
        let pos = ChunkPos::from_world(wx, wz);
        match self.chunks.get(&pos) {
            Some(c) => {
                let lx = wx.rem_euclid(CHUNK_SIZE as i32) as usize;
                let lz = wz.rem_euclid(CHUNK_SIZE as i32) as usize;
                c.get(lx, wy as usize, lz)
            }
            None => Block::Air,
        }
    }

    /// Set a block and return every chunk whose mesh is now stale (self +
    /// horizontal neighbours if the edit touched a chunk border).
    pub fn set(&mut self, wx: i32, wy: i32, wz: i32, b: Block) -> Vec<ChunkPos> {
        let mut dirty = Vec::new();
        if wy < 0 || wy >= WORLD_HEIGHT as i32 {
            return dirty;
        }
        let pos = ChunkPos::from_world(wx, wz);
        let lx = wx.rem_euclid(CHUNK_SIZE as i32);
        let lz = wz.rem_euclid(CHUNK_SIZE as i32);

        let chunk = self.ensure_chunk(pos);
        chunk.set(lx as usize, wy as usize, lz as usize, b);

        dirty.push(pos);
        if lx == 0 {
            dirty.push(ChunkPos::new(pos.x - 1, pos.z));
        }
        if lx == CHUNK_SIZE as i32 - 1 {
            dirty.push(ChunkPos::new(pos.x + 1, pos.z));
        }
        if lz == 0 {
            dirty.push(ChunkPos::new(pos.x, pos.z - 1));
        }
        if lz == CHUNK_SIZE as i32 - 1 {
            dirty.push(ChunkPos::new(pos.x, pos.z + 1));
        }

        // Flag every affected chunk (that is loaded) as needing a remesh.
        for p in &dirty {
            if let Some(c) = self.chunks.get_mut(p) {
                c.mesh_dirty = true;
            }
        }
        dirty
    }

    /// Highest non-air block at (wx, wz); used to drop the player on the surface.
    pub fn surface_y(&self, wx: i32, wz: i32) -> i32 {
        for y in (0..WORLD_HEIGHT as i32).rev() {
            if !self.get(wx, y, wz).is_air() {
                return y;
            }
        }
        SEA_LEVEL
    }
}

/// Analytic terrain height at a world column.
pub fn surface_height(seed: u64, x: i32, z: i32) -> i32 {
    let xf = x as f64;
    let zf = z as f64;
    let cont = fbm2(seed, xf * 0.0035, zf * 0.0035, 4, 2.0, 0.5);
    let hills = fbm2(seed ^ 0x1234_5678_9ABC_DEF0, xf * 0.014, zf * 0.014, 4, 2.0, 0.5);
    let h = BASE_HEIGHT + (cont - 0.5) * CONTINENTAL_AMP + (hills - 0.5) * HILL_AMP;
    (h.round() as i32).clamp(2, WORLD_HEIGHT as i32 - 8)
}

/// True when a cave should carve out this underground cell.
fn is_cave(seed: u64, x: i32, y: i32, z: i32) -> bool {
    if y < 2 || y > surface_height(seed, x, z) - 2 {
        return false;
    }
    let n = fbm3(
        seed ^ 0xCAFE_F00D_DEAD_BEEF,
        x as f64 * 0.07,
        y as f64 * 0.09,
        z as f64 * 0.07,
        3,
        2.0,
        0.5,
    );
    n > 0.62
}

fn generate_chunk(seed: u64, pos: ChunkPos) -> Chunk {
    let mut c = Chunk::filled(Block::Air);
    let ox = pos.x * CHUNK_SIZE as i32;
    let oz = pos.z * CHUNK_SIZE as i32;

    for lz in 0..CHUNK_SIZE {
        for lx in 0..CHUNK_SIZE {
            let wx = ox + lx as i32;
            let wz = oz + lz as i32;
            let h = surface_height(seed, wx, wz);
            let beach = h <= SEA_LEVEL + 1;

            for y in 0..=h {
                if y >= WORLD_HEIGHT as i32 {
                    break;
                }
                let b = if y == 0 {
                    Block::Bedrock
                } else if is_cave(seed, wx, y, wz) {
                    Block::Air
                } else if y == h {
                    if beach {
                        Block::Sand
                    } else {
                        Block::Grass
                    }
                } else if y >= h - 3 {
                    if beach {
                        Block::Sand
                    } else {
                        Block::Dirt
                    }
                } else {
                    Block::Stone
                };
                c.set(lx, y as usize, lz, b);
            }

            // Standing water in low columns.
            if h < SEA_LEVEL {
                for y in (h + 1)..=SEA_LEVEL {
                    if y < WORLD_HEIGHT as i32 {
                        c.set(lx, y as usize, lz, Block::Water);
                    }
                }
            }
        }
    }

    stamp_trees(&mut c, seed, pos);
    c.mesh_dirty = true;
    c
}

/// Whether a tree root sits at this column.
fn tree_here(seed: u64, x: i32, z: i32) -> bool {
    let h = surface_height(seed, x, z);
    if h <= SEA_LEVEL + 1 || is_cave(seed, x, h, z) {
        return false;
    }
    hash2(TREE_SEED ^ seed, x, z) % 100 < TREE_CHANCE
}

fn stamp_trees(c: &mut Chunk, seed: u64, pos: ChunkPos) {
    let ox = pos.x * CHUNK_SIZE as i32;
    let oz = pos.z * CHUNK_SIZE as i32;

    // Roots up to CANOPY outside the chunk can drop leaves into it, keeping
    // trees seamless across chunk borders.
    let x0 = ox - CANOPY;
    let x1 = ox + CHUNK_SIZE as i32 + CANOPY;
    let z0 = oz - CANOPY;
    let z1 = oz + CHUNK_SIZE as i32 + CANOPY;

    for tz in z0..z1 {
        for tx in x0..x1 {
            if !tree_here(seed, tx, tz) {
                continue;
            }
            let base = surface_height(seed, tx, tz) + 1;
            let th = 4 + (hash2(TREE_SEED ^ 0xA5A5, tx, tz) % 3) as i32; // 4..=6
            let top = base + th - 1;

            // Trunk overwrites anything.
            for y in base..=top {
                put(c, ox, oz, tx, y, tz, Block::OakLog, false);
            }

            // Leaf canopy: two wide layers then a smaller cap.
            for dy in 0i32..=2 {
                let radius: i32 = if dy < 2 { 2 } else { 1 };
                let ly = top - 1 + dy;
                for dz in -radius..=radius {
                    for dx in -radius..=radius {
                        // Clip blocky corners on the top layer.
                        if dy == 2 && dx.abs() == radius && dz.abs() == radius {
                            continue;
                        }
                        put(c, ox, oz, tx + dx, ly, tz + dz, Block::OakLeaves, true);
                    }
                }
            }
        }
    }
}

/// Write a block at world coords into `c` if it falls inside this chunk.
fn put(c: &mut Chunk, ox: i32, oz: i32, wx: i32, wy: i32, wz: i32, b: Block, only_air: bool) {
    let lx = wx - ox;
    let lz = wz - oz;
    if !Chunk::in_bounds(lx, wy, lz) {
        return;
    }
    let (lx, lz) = (lx as usize, lz as usize);
    if only_air && !c.get(lx, wy as usize, lz).is_air() {
        return;
    }
    c.set(lx, wy as usize, lz, b);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generation_is_deterministic() {
        let a = generate_chunk(1234, ChunkPos::new(4, -7));
        let b = generate_chunk(1234, ChunkPos::new(4, -7));
        for i in 0..1000 {
            let idx = i * 37 % crate::chunk::CHUNK_VOLUME;
            assert_eq!(a.get(idx % 16, (idx / 16) % WORLD_HEIGHT, 0), b.get(idx % 16, (idx / 16) % WORLD_HEIGHT, 0));
        }
    }

    #[test]
    fn bedrock_floor_and_solid_ground() {
        let c = generate_chunk(99, ChunkPos::new(0, 0));
        assert_eq!(c.get(0, 0, 0), Block::Bedrock);
        let h = surface_height(99, 0, 0);
        assert!(c.get(0, h as usize, 0) == Block::Grass || c.get(0, h as usize, 0) == Block::Sand);
    }

    #[test]
    fn sea_level_is_some_water_somewhere() {
        // Terrain amplitude is tuned so valleys dip below sea level; verify
        // water actually generates for at least some seeds in a broad scan.
        let mut seeds_with_water = 0;
        for seed in 0..40u64 {
            let has = (-40..40).any(|z| {
                (-40..40).any(|x| surface_height(seed, x, z) < SEA_LEVEL)
            });
            if has {
                seeds_with_water += 1;
            }
        }
        assert!(
            seeds_with_water > 0,
            "water must generate for at least one seed"
        );
    }

    #[test]
    fn terrain_spans_below_and_above_sea_level() {
        let mut min_h = i32::MAX;
        let mut max_h = i32::MIN;
        for seed in 0..12u64 {
            for z in -40..40 {
                for x in -40..40 {
                    let h = surface_height(seed, x, z);
                    min_h = min_h.min(h);
                    max_h = max_h.max(h);
                }
            }
        }
        assert!(min_h < SEA_LEVEL, "need valleys below sea level (min={min_h})");
        assert!(max_h > SEA_LEVEL, "need land above sea level (max={max_h})");
        assert!(min_h >= 1 && max_h < WORLD_HEIGHT as i32, "heights in world bounds");
    }
}
