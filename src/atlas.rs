//! Procedural texture atlas: paints every tile in code so the project ships no
//! binary image assets. Each tile is `ATLAS_TILE` square, stored as one layer of
//! a `texture_2d_array`.

use crate::block::tile;
use crate::renderer::ATLAS_TILE;

const T: usize = ATLAS_TILE as usize;

#[inline]
fn hash(x: i32, y: i32, salt: u32) -> u32 {
    let mut h = (x as u32)
        .wrapping_mul(0x9E37_79B1)
        ^ (y as u32).wrapping_mul(0x85EB_CA77)
        ^ salt.wrapping_mul(0xC2B2_AE3D);
    h ^= h >> 15;
    h = h.wrapping_mul(0x2545_F491);
    h ^= h >> 13;
    h
}

/// Signed value used to vary a colour. `amp >= 0` yields `-amp..=amp`
/// (symmetric); `amp < 0` yields `amp..=0` (darken-only), so callers can pass a
/// negative amplitude to carve mortar lines, bark grooves, etc.
#[inline]
fn noise(x: i32, y: i32, salt: u32, amp: i32) -> i32 {
    if amp >= 0 {
        (hash(x, y, salt) % (amp as u32 * 2 + 1)) as i32 - amp
    } else {
        let a = amp.unsigned_abs();
        (hash(x, y, salt) % (a + 1)) as i32 + amp
    }
}

#[inline]
fn clamp8(v: i32) -> u8 {
    v.clamp(0, 255) as u8
}

struct Tile {
    buf: Vec<u8>,
}

impl Tile {
    fn new() -> Self {
        Self {
            buf: vec![0u8; T * T * 4],
        }
    }

    #[inline]
    fn set(&mut self, x: usize, y: usize, r: u8, g: u8, b: u8, a: u8) {
        let i = (y * T + x) * 4;
        self.buf[i] = r;
        self.buf[i + 1] = g;
        self.buf[i + 2] = b;
        self.buf[i + 3] = a;
    }

    #[inline]
    fn get(&self, x: usize, y: usize) -> [u8; 4] {
        let i = (y * T + x) * 4;
        [self.buf[i], self.buf[i + 1], self.buf[i + 2], self.buf[i + 3]]
    }

    #[inline]
    fn jitter(&mut self, x: usize, y: usize, salt: u32, amp: i32) {
        let [r, g, b, a] = self.get(x, y);
        let d = noise(x as i32, y as i32, salt, amp);
        self.set(x, y, clamp8(r as i32 + d), clamp8(g as i32 + d), clamp8(b as i32 + d), a);
    }
}

fn speckle(t: &mut Tile, base: [u8; 3], salt: u32, amp: i32) {
    for y in 0..T {
        for x in 0..T {
            let d = noise(x as i32, y as i32, salt, amp);
            t.set(
                x,
                y,
                clamp8(base[0] as i32 + d),
                clamp8(base[1] as i32 + d),
                clamp8(base[2] as i32 + d),
                255,
            );
        }
    }
}

fn paint(layer: u16) -> Vec<u8> {
    let mut t = Tile::new();
    match layer {
        tile::GRASS_TOP => {
            speckle(&mut t, [98, 158, 62], 11, 16);
            // scattered darker blades
            for y in 0..T {
                for x in 0..T {
                    if hash(x as i32, y as i32, 3) % 7 == 0 {
                        t.jitter(x, y, 5, 18);
                    }
                }
            }
        }
        tile::GRASS_SIDE => {
            speckle(&mut t, [134, 96, 67], 21, 12);
            // grass fringe hanging over the top edge, jagged
            for x in 0..T {
                let depth = 3 + (hash(x as i32, 0, 7) % 2) as usize;
                for y in 0..depth {
                    let d = noise(x as i32, y as i32, 9, 14);
                    t.set(
                        x,
                        y,
                        clamp8(98 + d),
                        clamp8(158 + d),
                        clamp8(62 + d),
                        255,
                    );
                }
            }
        }
        tile::DIRT => speckle(&mut t, [134, 96, 67], 31, 13),
        tile::STONE => {
            speckle(&mut t, [126, 126, 126], 41, 14);
            for y in 0..T {
                for x in 0..T {
                    if hash(x as i32, y as i32, 43) % 9 == 0 {
                        t.jitter(x, y, 45, -22);
                    }
                }
            }
        }
        tile::SAND => speckle(&mut t, [219, 207, 163], 51, 8),
        tile::WATER => {
            speckle(&mut t, [52, 92, 180], 61, 6);
            for y in 0..T {
                for x in 0..T {
                    let a = if (x + y) % 4 == 0 { 150 } else { 175 };
                    let [r, g, b, _] = t.get(x, y);
                    t.set(x, y, r, g, b, a);
                }
            }
        }
        tile::LOG_SIDE => {
            speckle(&mut t, [104, 82, 52], 71, 8);
            for x in 0..T {
                if x % 4 == 0 {
                    for y in 0..T {
                        t.jitter(x, y, 73, -26);
                    }
                }
            }
        }
        tile::LOG_TOP => {
            for y in 0..T {
                for x in 0..T {
                    let dx = x as f32 - (T as f32 - 1.0) / 2.0;
                    let dy = y as f32 - (T as f32 - 1.0) / 2.0;
                    let r = (dx * dx + dy * dy).sqrt() as i32;
                    let ring = if r % 3 == 0 { -18 } else { 6 };
                    let d = noise(x as i32, y as i32, 81, 6) + ring;
                    t.set(
                        x,
                        y,
                        clamp8(160 + d),
                        clamp8(130 + d),
                        clamp8(84 + d),
                        255,
                    );
                }
            }
        }
        tile::LEAVES => {
            for y in 0..T {
                for x in 0..T {
                    let h = hash(x as i32, y as i32, 91);
                    if h % 7 == 0 {
                        t.set(x, y, 0, 0, 0, 0); // cut-out hole
                    } else {
                        let d = noise(x as i32, y as i32, 93, 20);
                        t.set(
                            x,
                            y,
                            clamp8(58 + d),
                            clamp8(132 + d),
                            clamp8(46 + d),
                            255,
                        );
                    }
                }
            }
        }
        tile::PLANKS => {
            speckle(&mut t, [150, 120, 80], 101, 7);
            for y in 0..T {
                for x in 0..T {
                    if y % 4 == 3 {
                        t.jitter(x, y, 103, -40);
                    }
                    if x % 5 == 0 && y % 4 != 3 {
                        t.jitter(x, y, 107, -14);
                    }
                }
            }
        }
        tile::COBBLE => {
            speckle(&mut t, [130, 130, 130], 111, 16);
            for y in 0..T {
                for x in 0..T {
                    let inx = x % 5;
                    let iny = y % 5;
                    if inx == 0 || iny == 0 {
                        t.jitter(x, y, 113, -44); // mortar
                    }
                }
            }
        }
        tile::BEDROCK => {
            speckle(&mut t, [74, 74, 74], 121, 22);
            for y in 0..T {
                for x in 0..T {
                    if hash(x as i32, y as i32, 123) % 5 == 0 {
                        t.jitter(x, y, 125, -30);
                    }
                }
            }
        }
        tile::GLASS => {
            for y in 0..T {
                for x in 0..T {
                    let border = x == 0 || y == 0 || x == T - 1 || y == T - 1;
                    if border {
                        t.set(x, y, 226, 236, 244, 230);
                    } else {
                        let shine = (x == 2 && y > 1 && y < 6) || (y == 2 && x > 1 && x < 6);
                        if shine {
                            t.set(x, y, 245, 250, 255, 120);
                        } else {
                            t.set(x, y, 200, 220, 232, 40);
                        }
                    }
                }
            }
        }
        tile::WHITE => {
            for y in 0..T {
                for x in 0..T {
                    t.set(x, y, 255, 255, 255, 255);
                }
            }
        }
        tile::DARK => {
            for y in 0..T {
                for x in 0..T {
                    t.set(x, y, 26, 26, 26, 205);
                }
            }
        }
        _ => unreachable!("unpainted atlas layer"),
    }
    t.buf
}

/// Build the full atlas pixel data: `ATLAS_LAYERS` layers of RGBA8.
pub fn build_atlas() -> Vec<u8> {
    let mut data = vec![0u8; (ATLAS_TILE * ATLAS_TILE * 4 * crate::block::tile::COUNT as u32) as usize];
    let layer_size = (ATLAS_TILE * ATLAS_TILE * 4) as usize;
    for layer in 0..crate::block::tile::COUNT {
        let pixels = paint(layer);
        let start = layer as usize * layer_size;
        data[start..start + layer_size].copy_from_slice(&pixels);
    }
    data
}
