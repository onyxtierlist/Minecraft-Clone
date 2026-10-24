//! Deterministic, cross-platform value noise.
//!
//! No external dependencies and no floating-point accumulation that could
//! diverge between platforms: everything is built from wrapping integer hashes,
//! so the same seed always produces the identical world on every machine.

#[inline]
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[inline]
pub fn hash2(seed: u64, x: i32, y: i32) -> u64 {
    let mut h = seed;
    h ^= (x as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    h = splitmix64(h);
    h ^= (y as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    splitmix64(h)
}

#[inline]
pub fn hash3(seed: u64, x: i32, y: i32, z: i32) -> u64 {
    let mut h = seed;
    h ^= (x as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    h = splitmix64(h);
    h ^= (y as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    h = splitmix64(h);
    h ^= (z as u64).wrapping_mul(0x1656_67B1_9E37_79F9);
    splitmix64(h)
}

/// Integer hash mapped uniformly into `[0, 1)`.
#[inline]
fn hash_to_unit(h: u64) -> f64 {
    // top 53 bits -> exact double in [0,1)
    ((h >> 11) as f64) / ((1u64 << 53) as f64)
}

#[inline]
fn fade(t: f64) -> f64 {
    t * t * (3.0 - 2.0 * t)
}

/// Value at integer lattice point in `[0, 1)`.
#[inline]
pub fn value2(seed: u64, x: i32, y: i32) -> f64 {
    hash_to_unit(hash2(seed, x, y))
}

/// Smooth 2-D value noise in `[0, 1)`.
pub fn noise2(seed: u64, xf: f64, yf: f64) -> f64 {
    let x0 = xf.floor() as i32;
    let y0 = yf.floor() as i32;
    let fx = xf - x0 as f64;
    let fy = yf - y0 as f64;
    let u = fade(fx);
    let v = fade(fy);

    let a = value2(seed, x0, y0);
    let b = value2(seed, x0 + 1, y0);
    let c = value2(seed, x0, y0 + 1);
    let d = value2(seed, x0 + 1, y0 + 1);

    let ab = a + (b - a) * u;
    let cd = c + (d - c) * u;
    ab + (cd - ab) * v
}

/// Smooth 3-D value noise in `[0, 1)` (used for cave carving).
pub fn noise3(seed: u64, xf: f64, yf: f64, zf: f64) -> f64 {
    let x0 = xf.floor() as i32;
    let y0 = yf.floor() as i32;
    let z0 = zf.floor() as i32;
    let fx = fade(xf - x0 as f64);
    let fy = fade(yf - y0 as f64);
    let fz = fade(zf - z0 as f64);

    let c000 = hash_to_unit(hash3(seed, x0, y0, z0));
    let c100 = hash_to_unit(hash3(seed, x0 + 1, y0, z0));
    let c010 = hash_to_unit(hash3(seed, x0, y0 + 1, z0));
    let c110 = hash_to_unit(hash3(seed, x0 + 1, y0 + 1, z0));
    let c001 = hash_to_unit(hash3(seed, x0, y0, z0 + 1));
    let c101 = hash_to_unit(hash3(seed, x0 + 1, y0, z0 + 1));
    let c011 = hash_to_unit(hash3(seed, x0, y0 + 1, z0 + 1));
    let c111 = hash_to_unit(hash3(seed, x0 + 1, y0 + 1, z0 + 1));

    let x00 = c000 + (c100 - c000) * fx;
    let x10 = c010 + (c110 - c010) * fx;
    let x01 = c001 + (c101 - c001) * fx;
    let x11 = c011 + (c111 - c011) * fx;
    let y0i = x00 + (x10 - x00) * fy;
    let y1i = x01 + (x11 - x01) * fy;
    y0i + (y1i - y0i) * fz
}

/// Fractal Brownian motion over `noise2`, normalized back to `[0, 1]`.
pub fn fbm2(seed: u64, x: f64, y: f64, octaves: u32, lacunarity: f64, gain: f64) -> f64 {
    let mut amp = 1.0f64;
    let mut freq = 1.0f64;
    let mut sum = 0.0f64;
    let mut norm = 0.0f64;
    for i in 0..octaves {
        sum += amp * noise2(seed.wrapping_add(i as u64 * 0x9E37), x * freq, y * freq);
        norm += amp;
        amp *= gain;
        freq *= lacunarity;
    }
    sum / norm
}

/// Fractal Brownian motion over `noise3`, normalized to `[0, 1]`.
pub fn fbm3(seed: u64, x: f64, y: f64, z: f64, octaves: u32, lacunarity: f64, gain: f64) -> f64 {
    let mut amp = 1.0f64;
    let mut freq = 1.0f64;
    let mut sum = 0.0f64;
    let mut norm = 0.0f64;
    for i in 0..octaves {
        sum += amp * noise3(
            seed.wrapping_add(i as u64 * 0x85EB),
            x * freq,
            y * freq,
            z * freq,
        );
        norm += amp;
        amp *= gain;
        freq *= lacunarity;
    }
    sum / norm
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic() {
        assert_eq!(noise2(42, 3.5, -7.25), noise2(42, 3.5, -7.25));
        assert_ne!(noise2(42, 3.5, -7.25), noise2(43, 3.5, -7.25));
    }

    #[test]
    fn in_range() {
        for i in 0..500 {
            let x = i as f64 * 0.37 - 50.0;
            let v = fbm2(7, x, x * 0.5, 5, 2.0, 0.5);
            assert!((0.0..=1.0).contains(&v), "out of range: {v}");
            let n = noise3(7, x, x * 1.3, x * 0.7);
            assert!((0.0..1.0).contains(&n), "noise3 out of range: {n}");
        }
    }
}
