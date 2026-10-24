//! Block types and their rendering/collision properties.

/// Index of a layer inside the texture-array atlas.
/// These are generated procedurally at startup (see `renderer::build_atlas`).
pub mod tile {
    pub const GRASS_TOP: u16 = 0;
    pub const GRASS_SIDE: u16 = 1;
    pub const DIRT: u16 = 2;
    pub const STONE: u16 = 3;
    pub const SAND: u16 = 4;
    pub const WATER: u16 = 5;
    pub const LOG_SIDE: u16 = 6;
    pub const LOG_TOP: u16 = 7;
    pub const LEAVES: u16 = 8;
    pub const PLANKS: u16 = 9;
    pub const COBBLE: u16 = 10;
    pub const BEDROCK: u16 = 11;
    pub const GLASS: u16 = 12;
    pub const WHITE: u16 = 13;
    pub const DARK: u16 = 14;

    pub const COUNT: u16 = 15;
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum Block {
    Air,
    Grass,
    Dirt,
    Stone,
    Sand,
    Water,
    OakLog,
    OakLeaves,
    Planks,
    Cobblestone,
    Glass,
    Bedrock,
}

/// The blocks available on the hotbar, in slot order.
pub const HOTBAR: [Block; 9] = [
    Block::Grass,
    Block::Dirt,
    Block::Stone,
    Block::Sand,
    Block::OakLog,
    Block::OakLeaves,
    Block::Planks,
    Block::Cobblestone,
    Block::Glass,
];

impl Block {
    #[inline]
    pub fn is_air(self) -> bool {
        matches!(self, Block::Air)
    }

    /// Blocks player movement / has a collision box.
    #[inline]
    pub fn is_solid(self) -> bool {
        !matches!(self, Block::Air | Block::Water)
    }

    /// Fully occludes neighbouring faces (so we can skip emitting them).
    #[inline]
    pub fn is_opaque(self) -> bool {
        !matches!(self, Block::Air | Block::Water | Block::Glass)
    }

    /// Rendered in the transparent pass (sorted after opaque, alpha blended).
    #[inline]
    pub fn is_transparent(self) -> bool {
        matches!(self, Block::Water | Block::Glass)
    }

    #[inline]
    pub fn top_tile(self) -> u16 {
        match self {
            Block::Grass => tile::GRASS_TOP,
            Block::Dirt => tile::DIRT,
            Block::Stone => tile::STONE,
            Block::Sand => tile::SAND,
            Block::Water => tile::WATER,
            Block::OakLog => tile::LOG_TOP,
            Block::OakLeaves => tile::LEAVES,
            Block::Planks => tile::PLANKS,
            Block::Cobblestone => tile::COBBLE,
            Block::Glass => tile::GLASS,
            Block::Bedrock => tile::BEDROCK,
            Block::Air => tile::WHITE,
        }
    }

    #[inline]
    pub fn bottom_tile(self) -> u16 {
        match self {
            Block::Grass => tile::DIRT,
            Block::OakLog => tile::LOG_TOP,
            other => other.top_tile(),
        }
    }

    #[inline]
    pub fn side_tile(self) -> u16 {
        match self {
            Block::Grass => tile::GRASS_SIDE,
            Block::OakLog => tile::LOG_SIDE,
            other => other.top_tile(),
        }
    }

    /// Icon tile shown in the hotbar for this block.
    #[inline]
    pub fn icon_tile(self) -> u16 {
        match self {
            Block::Grass => tile::GRASS_SIDE,
            other => other.top_tile(),
        }
    }
}
