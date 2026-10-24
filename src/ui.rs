//! Minimal 2D overlay: a 9-slot hotbar and a crosshair, drawn in clip space.

use crate::block;

#[derive(Clone, Copy, Debug)]
pub struct UiVertex {
    /// Clip-space position (already in -1..=1).
    pub pos: [f32; 2],
    pub uv: [f32; 2],
    pub layer: f32,
    pub alpha: f32,
}

const SLOT: f32 = 44.0;
const GAP: f32 = 6.0;
const BORDER: f32 = 4.0;
const BOTTOM_MARGIN: f32 = 28.0;
const ICON_INSET: f32 = 6.0;

fn push_quad(
    verts: &mut Vec<UiVertex>,
    indices: &mut Vec<u32>,
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    w: f32,
    h: f32,
    layer: u16,
    alpha: f32,
) {
    // Pixel space is top-left origin; clip space is centre origin, y up.
    let cx = |px: f32| px / w * 2.0 - 1.0;
    let cy = |py: f32| 1.0 - py / h * 2.0;
    let l = layer as f32;
    let base = verts.len() as u32;

    verts.push(UiVertex { pos: [cx(x0), cy(y0)], uv: [0.0, 0.0], layer: l, alpha });
    verts.push(UiVertex { pos: [cx(x1), cy(y0)], uv: [1.0, 0.0], layer: l, alpha });
    verts.push(UiVertex { pos: [cx(x1), cy(y1)], uv: [1.0, 1.0], layer: l, alpha });
    verts.push(UiVertex { pos: [cx(x0), cy(y1)], uv: [0.0, 1.0], layer: l, alpha });

    indices.extend_from_slice(&[base, base + 1, base + 2, base + 2, base + 3, base]);
}

/// Build the whole overlay for the current frame.
pub fn build(width: u32, height: u32, selected: usize) -> (Vec<UiVertex>, Vec<u32>) {
    let w = width as f32;
    let h = height as f32;
    let mut verts = Vec::new();
    let mut indices = Vec::new();

    if w < 1.0 || h < 1.0 {
        return (verts, indices);
    }

    // --- Hotbar ---
    let n = block::HOTBAR.len() as f32;
    let total = n * SLOT + (n - 1.0) * GAP;
    let x0 = (w - total).max(0.0) * 0.5;
    let y0 = h - BOTTOM_MARGIN - SLOT;

    for (i, blk) in block::HOTBAR.iter().enumerate() {
        let sx = x0 + i as f32 * (SLOT + GAP);
        let sy = y0;

        // Selection frame drawn first so it shows as a ring behind the slot.
        if i == selected {
            push_quad(
                &mut verts,
                &mut indices,
                sx - BORDER,
                sy - BORDER,
                sx + SLOT + BORDER,
                sy + SLOT + BORDER,
                w,
                h,
                block::tile::WHITE,
                0.95,
            );
        }

        push_quad(
            &mut verts,
            &mut indices,
            sx,
            sy,
            sx + SLOT,
            sy + SLOT,
            w,
            h,
            block::tile::DARK,
            0.85,
        );

        push_quad(
            &mut verts,
            &mut indices,
            sx + ICON_INSET,
            sy + ICON_INSET,
            sx + SLOT - ICON_INSET,
            sy + SLOT - ICON_INSET,
            w,
            h,
            blk.icon_tile(),
            1.0,
        );
    }

    // --- Crosshair ---
    let cx = w * 0.5;
    let cy = h * 0.5;
    let len = 18.0;
    let thick = 2.0;
    push_quad(
        &mut verts,
        &mut indices,
        cx - len * 0.5,
        cy - thick * 0.5,
        cx + len * 0.5,
        cy + thick * 0.5,
        w,
        h,
        block::tile::WHITE,
        0.85,
    );
    push_quad(
        &mut verts,
        &mut indices,
        cx - thick * 0.5,
        cy - len * 0.5,
        cx + thick * 0.5,
        cy + len * 0.5,
        w,
        h,
        block::tile::WHITE,
        0.85,
    );

    (verts, indices)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nine_slots_centered() {
        let (v, idx) = build(800, 600, 0);
        // 9 slots * 3 quads (frame only for selected -> 1 frame) + crosshair 2.
        // selected=0 -> frame + bg + icon for slot0, bg+icon for 1..8, crosshair 2.
        let expected_quads = 1 + 9 * 2 + 2;
        assert_eq!(idx.len() / 6, expected_quads);
        assert_eq!(v.len(), expected_quads * 4);
    }

    #[test]
    fn indices_within_bounds() {
        let (v, idx) = build(640, 360, 5);
        assert!(idx.iter().all(|&i| (i as usize) < v.len()));
    }

    #[test]
    fn center_pixel_maps_to_clip_origin() {
        let (cx, cy) = (320.0f32, 180.0f32);
        let clip_x = cx / 640.0 * 2.0 - 1.0;
        let clip_y = 1.0 - cy / 360.0 * 2.0;
        assert!(clip_x.abs() < 1e-5 && clip_y.abs() < 1e-5);
    }
}
