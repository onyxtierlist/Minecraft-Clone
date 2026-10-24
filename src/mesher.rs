//! Converts voxel chunks into GPU-ready vertex/index buffers.
//!
//! Only faces touching a non-occluding neighbour are emitted, so interior
//! geometry costs nothing. Faces are wound counter-clockwise when viewed from
//! outside (cross product of the first triangle matches the face normal) so the
//! pipeline can back-face cull.

use bytemuck::{Pod, Zeroable};

use crate::block::Block;
use crate::chunk::{ChunkPos, CHUNK_SIZE, WORLD_HEIGHT};
use crate::world::World;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct ChunkVertex {
    pub pos: [f32; 3],
    pub uv: [f32; 2],
    /// x = texture-array layer, y = directional face shading.
    pub layer_light: [f32; 2],
}

#[derive(Default)]
pub struct MeshData {
    pub vertices: Vec<ChunkVertex>,
    pub indices: Vec<u32>,
}

impl MeshData {
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }
}

#[derive(Default)]
pub struct ChunkMeshes {
    pub opaque: MeshData,
    pub transparent: MeshData,
}

struct Face {
    normal: [i32; 3],
    corners: [[f32; 3]; 4],
    uv: [[f32; 2]; 4],
    light: f32,
    top: bool,
    bottom: bool,
}

// CCW from outside; normals derived from (v1-v0) x (v2-v0).
const FACES: [Face; 6] = [
    // +X
    Face {
        normal: [1, 0, 0],
        corners: [[1.0, 0.0, 1.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0]],
        uv: [[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]],
        light: 0.75,
        top: false,
        bottom: false,
    },
    // -X
    Face {
        normal: [-1, 0, 0],
        corners: [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 1.0, 0.0]],
        uv: [[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]],
        light: 0.75,
        top: false,
        bottom: false,
    },
    // +Y (top)
    Face {
        normal: [0, 1, 0],
        corners: [[0.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        uv: [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        light: 1.0,
        top: true,
        bottom: false,
    },
    // -Y (bottom)
    Face {
        normal: [0, -1, 0],
        corners: [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
        uv: [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        light: 0.5,
        top: false,
        bottom: true,
    },
    // +Z
    Face {
        normal: [0, 0, 1],
        corners: [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]],
        uv: [[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]],
        light: 0.65,
        top: false,
        bottom: false,
    },
    // -Z
    Face {
        normal: [0, 0, -1],
        corners: [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
        uv: [[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]],
        light: 0.65,
        top: false,
        bottom: false,
    },
];

fn tile_for(b: Block, face: &Face) -> u16 {
    if face.top {
        b.top_tile()
    } else if face.bottom {
        b.bottom_tile()
    } else {
        b.side_tile()
    }
}

fn face_visible(b: Block, neighbor: Block) -> bool {
    if b.is_opaque() {
        // Emit unless the neighbour fully hides this face.
        !neighbor.is_opaque()
    } else {
        // Water/glass: only against empty space or a *different* transparent
        // block; never against opaque blocks or identical ones.
        neighbor.is_air() || (neighbor.is_transparent() && neighbor != b)
    }
}

/// Mesh one chunk. Neighbouring chunks (including the four horizontal
/// neighbours) must already be generated so border faces resolve correctly.
pub fn mesh_chunk(world: &World, pos: ChunkPos) -> ChunkMeshes {
    let Some(chunk) = world.chunk(pos) else {
        return ChunkMeshes::default();
    };

    let ox = pos.x * CHUNK_SIZE as i32;
    let oz = pos.z * CHUNK_SIZE as i32;

    let mut opaque = MeshData::default();
    let mut transparent = MeshData::default();

    for lz in 0..CHUNK_SIZE {
        for lx in 0..CHUNK_SIZE {
            let wx = ox + lx as i32;
            let wz = oz + lz as i32;
            for y in 0..WORLD_HEIGHT {
                let b = chunk.get(lx, y, lz);
                if b.is_air() {
                    continue;
                }
                let wy = y as i32;
                let target = if b.is_transparent() {
                    &mut transparent
                } else {
                    &mut opaque
                };

                for face in FACES.iter() {
                    let nx = wx + face.normal[0];
                    let ny = wy + face.normal[1];
                    let nz = wz + face.normal[2];
                    let neighbor = world.get(nx, ny, nz);
                    if !face_visible(b, neighbor) {
                        continue;
                    }

                    let layer = tile_for(b, face) as f32;
                    let base = target.vertices.len() as u32;
                    for (ci, corner) in face.corners.iter().enumerate() {
                        target.vertices.push(ChunkVertex {
                            pos: [wx as f32 + corner[0], wy as f32 + corner[1], wz as f32 + corner[2]],
                            uv: face.uv[ci],
                            layer_light: [layer, face.light],
                        });
                    }
                    // two CCW triangles
                    target.indices.extend_from_slice(&[
                        base,
                        base + 1,
                        base + 2,
                        base + 2,
                        base + 3,
                        base,
                    ]);
                }
            }
        }
    }

    ChunkMeshes { opaque, transparent }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::Block;

    #[test]
    fn isolated_block_emits_six_faces() {
        let mut world = World::new(5);
        // A genuinely empty chunk with a single stone block.
        world.insert_air_chunk(ChunkPos::new(0, 0));
        world.set(5, 50, 5, Block::Stone);
        let m = mesh_chunk(&world, ChunkPos::new(0, 0));
        assert_eq!(m.opaque.indices.len() / 6, 6, "one block -> six quads");
        assert_eq!(m.opaque.vertices.len(), 24);
        assert!(m.transparent.is_empty());
    }

    #[test]
    fn buried_block_emits_nothing() {
        let mut world = World::new(5);
        world.insert_air_chunk(ChunkPos::new(0, 0));
        // Surround a cell entirely with stone.
        for z in 4..=6 {
            for y in 49..=51 {
                for x in 4..=6 {
                    world.set(x, y, z, Block::Stone);
                }
            }
        }
        let m = mesh_chunk(&world, ChunkPos::new(0, 0));
        // 3x3x3 = 27 blocks; center is fully hidden. Outer shell quads = 6*9*2 - shared…
        // Just assert the center contributed nothing by checking no vertex sits at the center face plane.
        let center_hidden = !m.opaque.vertices.iter().any(|v| {
            (v.pos[0] - 5.0).abs() < 1e-4
                && (v.pos[1] - 50.0).abs() < 1e-4
                && (v.pos[2] - 5.0).abs() < 1e-4
        });
        assert!(center_hidden, "fully enclosed faces must be culled");
    }

    #[test]
    fn winding_matches_normals() {
        // Verify CCW winding: (v1-v0) x (v2-v0) points along the face normal.
        for face in FACES.iter() {
            let v0 = face.corners[0];
            let v1 = face.corners[1];
            let v2 = face.corners[2];
            let e1 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
            let e2 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];
            let n = [
                e1[1] * e2[2] - e1[2] * e2[1],
                e1[2] * e2[0] - e1[0] * e2[2],
                e1[0] * e2[1] - e1[1] * e2[0],
            ];
            assert_eq!(
                n,
                [
                    face.normal[0] as f32,
                    face.normal[1] as f32,
                    face.normal[2] as f32
                ],
                "winding/normal mismatch"
            );
        }
    }
}
