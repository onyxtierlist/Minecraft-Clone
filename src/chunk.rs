//! Chunk storage: a fixed 16 x 16 column of blocks, full world height.

use crate::block::Block;

pub const CHUNK_SIZE: usize = 16;
pub const WORLD_HEIGHT: usize = 96;
pub const CHUNK_VOLUME: usize = CHUNK_SIZE * CHUNK_SIZE * WORLD_HEIGHT;

/// Horizontal chunk coordinate (one chunk is a full-height column).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct ChunkPos {
    pub x: i32,
    pub z: i32,
}

impl ChunkPos {
    pub fn new(x: i32, z: i32) -> Self {
        Self { x, z }
    }

    #[inline]
    pub fn from_world(x: i32, z: i32) -> Self {
        Self {
            x: x.div_euclid(CHUNK_SIZE as i32),
            z: z.div_euclid(CHUNK_SIZE as i32),
        }
    }
}

pub struct Chunk {
    blocks: Vec<Block>,
    /// Set when block data changed but the mesh is stale.
    pub mesh_dirty: bool,
}

impl Chunk {
    pub fn filled(fill: Block) -> Self {
        Self {
            blocks: vec![fill; CHUNK_VOLUME],
            mesh_dirty: true,
        }
    }

    /// Flat index for local (x, y, z). x/z in 0..CHUNK_SIZE, y in 0..WORLD_HEIGHT.
    #[inline]
    pub fn index(x: usize, y: usize, z: usize) -> usize {
        x + CHUNK_SIZE * (z + CHUNK_SIZE * y)
    }

    #[inline]
    pub fn in_bounds(lx: i32, y: i32, lz: i32) -> bool {
        lx >= 0
            && lx < CHUNK_SIZE as i32
            && lz >= 0
            && lz < CHUNK_SIZE as i32
            && y >= 0
            && y < WORLD_HEIGHT as i32
    }

    #[inline]
    pub fn get(&self, lx: usize, y: usize, lz: usize) -> Block {
        debug_assert!(lx < CHUNK_SIZE && lz < CHUNK_SIZE && y < WORLD_HEIGHT);
        self.blocks[Self::index(lx, y, lz)]
    }

    #[inline]
    pub fn set(&mut self, lx: usize, y: usize, lz: usize, b: Block) {
        debug_assert!(lx < CHUNK_SIZE && lz < CHUNK_SIZE && y < WORLD_HEIGHT);
        self.blocks[Self::index(lx, y, lz)] = b;
    }
}
