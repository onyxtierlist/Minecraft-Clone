mod atlas;
mod block;
mod camera;
mod chunk;
mod mesher;
mod noise;
mod physics;
mod raycast;
mod renderer;
mod ui;
mod world;

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

use glam::Vec3;
use winit::application::ApplicationHandler;
use winit::event::{
    DeviceEvent, DeviceId, ElementState, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent,
};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{CursorGrabMode, Window, WindowAttributes, WindowId};

use block::Block;
use camera::Camera;
use chunk::ChunkPos;
use mesher::mesh_chunk;
use physics::{MoveInput, Player, HALF_WIDTH, HEIGHT};
use renderer::{GpuMesh, GpuUiVertex, Renderer, Uniforms};
use world::World;

const RENDER_DISTANCE: i32 = 5; // chunks
const SHOT_RADIUS: i32 = 4;
const SENSITIVITY: f32 = 0.0024;
const REACH: f32 = 6.0;

fn main() {
    env_logger::init();
    let args: Vec<String> = std::env::args().collect();

    if let Some(i) = args.iter().position(|a| a == "--help" || a == "-h") {
        let _ = i;
        println!(
            "rustcraft — a Minecraft-like voxel engine (Rust + wgpu → Vulkan/Metal)\n\n\
             Usage:\n  rustcraft                         play in a window\n  \
             rustcraft --screenshot out.png [w] [h]   render one frame headlessly"
        );
        return;
    }

    if let Some(i) = args.iter().position(|a| a == "--screenshot") {
        let path = args.get(i + 1).cloned().unwrap_or_else(|| "screenshot.png".into());
        let w = args
            .get(i + 2)
            .and_then(|s| s.parse().ok())
            .unwrap_or(640);
        let h = args
            .get(i + 3)
            .and_then(|s| s.parse().ok())
            .unwrap_or(360);
        run_screenshot(&path, w, h, 1);
        return;
    }

    run_windowed();
}

// ---------------------------------------------------------------------------
// Headless screenshot (used to verify rendering without a display)
// ---------------------------------------------------------------------------

fn run_screenshot(path: &str, w: u32, h: u32, seed: u64) {
    let mut renderer = pollster::block_on(Renderer::new(None, w, h));
    let mut game = Game::new(seed);

    let center = ChunkPos::new(0, 0);
    game.stream(&renderer, center, SHOT_RADIUS, f32::INFINITY);

    let surface = game.world.surface_y(8, 8);
    let water = find_water(&game);
    // Hover just above and behind the nearest water column so the shot is
    // guaranteed to show the transparent water surface.
    let eye = match water {
        Some((wx, wz)) => Vec3::new(
            wx as f32 + 0.5,
            world::SEA_LEVEL as f32 + 16.0,
            wz as f32 + 0.5 - 9.0,
        ),
        None => Vec3::new(8.5, surface as f32 + 22.0, 8.5),
    };
    let mut camera = Camera::new(eye);
    match water.map(|(wx, wz)| {
        Vec3::new(wx as f32 + 0.5, world::SEA_LEVEL as f32, wz as f32 + 0.5)
    }) {
        Some(a) => {
            let dir = (a - eye).normalize_or_zero();
            camera.pitch = dir.y.asin().clamp(-1.4, 1.4);
            camera.yaw = dir.z.atan2(dir.x);
        }
        None => {
            camera.yaw = 0.7;
            camera.pitch = -0.45;
        }
    }

    let u = build_uniforms(&camera, w, h);
    let eye = camera.eye;
    let (op, tr) = game.draw_lists(eye);
    let (ui_v, ui_i) = ui::build(w, h, 0);
    let ui_v: Vec<GpuUiVertex> = ui_v.into_iter().map(|v| v.into()).collect();

    renderer
        .render(&u, &op, &tr, &ui_v, &ui_i)
        .expect("headless render");

    let (data, rw, rh) = renderer.read_back();
    image::save_buffer(path, &data, rw, rh, image::ColorType::Rgba8)
        .expect("write screenshot");
    println!("wrote {path} ({rw}x{rh})");
}

fn build_uniforms(camera: &Camera, w: u32, h: u32) -> Uniforms {
    let aspect = (w as f32 / h as f32).max(0.1);
    let mut u = Uniforms::default();
    u.view_proj = camera.view_proj(aspect).to_cols_array_2d();
    let eye = camera.eye;
    u.cam_pos = [eye.x, eye.y, eye.z, 0.0];
    let fog_end = (RENDER_DISTANCE as f32 + 1.0) * 16.0;
    u.params = [fog_end * 0.55, fog_end, 0.0, 0.0];
    u
}/// Scan loaded chunks for the sea-level water column nearest the origin (used
/// to aim the verification screenshot at the transparent-water path).
fn find_water(game: &Game) -> Option<(i32, i32)> {
    let size = chunk::CHUNK_SIZE as i32;
    let mut best: Option<(i64, i32, i32)> = None;
    for pos in game.meshes.keys() {
        let ox = pos.x * size;
        let oz = pos.z * size;
        for dz in 0..size {
            for dx in 0..size {
                let (wx, wz) = (ox + dx, oz + dz);
                if game.world.get(wx, world::SEA_LEVEL, wz) == Block::Water {
                    let d2 = wx as i64 * wx as i64 + wz as i64 * wz as i64;
                    if best.map_or(true, |(bd, _, _)| d2 < bd) {
                        best = Some((d2, wx, wz));
                    }
                }
            }
        }
    }
    best.map(|(_, x, z)| (x, z))
}

/// Nearest-to-origin column that sits comfortably above sea level (dry land).
fn find_dry_column(seed: u64) -> (i32, i32) {
    let mut best = (8i32, 8i32);
    let mut best_d2 = i32::MAX;
    for z in -48..48 {
        for x in -48..48 {
            if world::surface_height(seed, x, z) > world::SEA_LEVEL + 2 {
                let d2 = x * x + z * z;
                if d2 < best_d2 {
                    best_d2 = d2;
                    best = (x, z);
                }
            }
        }
    }
    best
}

// ---------------------------------------------------------------------------
// World + chunk streaming
// ---------------------------------------------------------------------------

struct ChunkGpu {
    opaque: Option<GpuMesh>,
    transparent: Option<GpuMesh>,
}

struct Game {
    world: World,
    meshes: HashMap<ChunkPos, ChunkGpu>,
}

impl Game {
    fn new(seed: u64) -> Self {
        Self {
            world: World::new(seed),
            meshes: HashMap::new(),
        }
    }

    /// Generate the chunk and its horizontal neighbours (so border faces
    /// resolve), mesh it, and store the GPU buffers.
    fn ensure_and_mesh(&mut self, renderer: &Renderer, pos: ChunkPos) {
        for (dx, dz) in [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)] {
            self.world
                .ensure_chunk(ChunkPos::new(pos.x + dx, pos.z + dz));
        }
        let m = mesh_chunk(&self.world, pos);
        let opaque = renderer.upload_mesh(&m.opaque);
        let transparent = renderer.upload_mesh(&m.transparent);
        self.meshes.insert(pos, ChunkGpu { opaque, transparent });
        if let Some(c) = self.world.chunk_mut(pos) {
            c.mesh_dirty = false;
        }
    }

    /// Build any missing/dirty chunks within `radius` of `center`, nearest
    /// first, stopping once `budget_ms` has elapsed (unbounded for headless).
    fn stream(&mut self, renderer: &Renderer, center: ChunkPos, radius: i32, budget_ms: f32) {
        let start = Instant::now();
        let mut wanted: Vec<(i32, ChunkPos)> = Vec::new();
        for dz in -radius..=radius {
            for dx in -radius..=radius {
                let p = ChunkPos::new(center.x + dx, center.z + dz);
                let missing = self.meshes.get(&p).is_none();
                let dirty = self
                    .world
                    .chunk(p)
                    .map_or(false, |c| c.mesh_dirty);
                if missing || dirty {
                    wanted.push((dx * dx + dz * dz, p));
                }
            }
        }
        wanted.sort_by_key(|(d, _)| *d);
        for (_, p) in wanted {
            self.ensure_and_mesh(renderer, p);
            if start.elapsed().as_secs_f32() * 1000.0 > budget_ms {
                break;
            }
        }
    }

    fn edit(&mut self, renderer: &Renderer, wx: i32, wy: i32, wz: i32, b: Block) {
        let dirty = self.world.set(wx, wy, wz, b);
        for p in dirty {
            if self.world.chunk(p).is_some() {
                self.ensure_and_mesh(renderer, p);
            }
        }
    }

    /// Opaque meshes front-to-back, transparent back-to-front.
    fn draw_lists(&self, eye: Vec3) -> (Vec<&GpuMesh>, Vec<&GpuMesh>) {
        let mut opaque: Vec<(f32, &GpuMesh)> = Vec::new();
        let mut trans: Vec<(f32, &GpuMesh)> = Vec::new();
        for (pos, cg) in &self.meshes {
            let cx = (pos.x * 16 + 8) as f32;
            let cz = (pos.z * 16 + 8) as f32;
            let d = (eye.x - cx).powi(2) + (eye.z - cz).powi(2);
            if let Some(m) = &cg.opaque {
                opaque.push((d, m));
            }
            if let Some(m) = &cg.transparent {
                trans.push((d, m));
            }
        }
        opaque.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        trans.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        (
            opaque.into_iter().map(|(_, m)| m).collect(),
            trans.into_iter().map(|(_, m)| m).collect(),
        )
    }
}

// ---------------------------------------------------------------------------
// Windowed application
// ---------------------------------------------------------------------------

struct App {
    seed: u64,
    window: Option<Arc<Window>>,
    renderer: Option<Renderer>,
    game: Game,
    camera: Camera,
    player: Player,
    selected: usize,
    keys: HashSet<KeyCode>,
    grabbed: bool,
    mouse_delta: (f32, f32),
    last_time: Option<Instant>,
}

impl App {
    fn new(seed: u64) -> Self {
        Self {
            seed,
            window: None,
            renderer: None,
            game: Game::new(seed),
            camera: Camera::new(Vec3::new(8.5, 70.0, 8.5)),
            player: Player::new(Vec3::new(8.5, 70.0, 8.5)),
            selected: 0,
            keys: HashSet::new(),
            grabbed: false,
            mouse_delta: (0.0, 0.0),
            last_time: None,
        }
    }

    fn move_input(&self) -> MoveInput {
        let mut f = 0.0;
        let mut s = 0.0;
        if self.keys.contains(&KeyCode::KeyW) {
            f += 1.0;
        }
        if self.keys.contains(&KeyCode::KeyS) {
            f -= 1.0;
        }
        if self.keys.contains(&KeyCode::KeyD) {
            s += 1.0;
        }
        if self.keys.contains(&KeyCode::KeyA) {
            s -= 1.0;
        }
        MoveInput {
            forward: f,
            strafe: s,
            jump: self.keys.contains(&KeyCode::Space),
            sneak: self.keys.contains(&KeyCode::ShiftLeft),
            sprint: self.keys.contains(&KeyCode::ControlLeft),
        }
    }

    fn spawn(&mut self) {
        // Pick the column nearest the origin that is comfortably above sea level
        // (so we never spawn under water on an ocean seed), then load its chunk
        // plus neighbours so physics has solid ground.
        let (sx, sz) = find_dry_column(self.seed);
        let chunk = ChunkPos::from_world(sx, sz);
        if let Some(r) = &self.renderer {
            for (dx, dz) in [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)] {
                let cp = ChunkPos::new(chunk.x + dx, chunk.z + dz);
                self.game.ensure_and_mesh(r, cp);
            }
        }
        let sy = self.game.world.surface_y(sx, sz);
        let feet = Vec3::new(sx as f32 + 0.5, sy as f32 + 1.0, sz as f32 + 0.5);
        self.player = Player::new(feet);
        self.camera = Camera::new(self.player.eye());
    }

    fn grab_cursor(&mut self) {
        if let Some(w) = &self.window {
            let _ = w
                .set_cursor_grab(CursorGrabMode::Locked)
                .or_else(|_| w.set_cursor_grab(CursorGrabMode::Confined));
            w.set_cursor_visible(false);
        }
        self.grabbed = true;
    }

    fn release_cursor(&mut self) {
        if let Some(w) = &self.window {
            let _ = w.set_cursor_grab(CursorGrabMode::None);
            w.set_cursor_visible(true);
        }
        self.grabbed = false;
    }

    fn break_block(&mut self) {
        let eye = self.camera.eye;
        let dir = self.camera.forward();
        let hit = raycast::raycast(&self.game.world, eye, dir, REACH);
        if let Some(h) = hit {
            let (x, y, z) = h.block_pos;
            if self.game.world.get(x, y, z) == Block::Bedrock {
                return;
            }
            if let Some(r) = &self.renderer {
                self.game.edit(r, x, y, z, Block::Air);
            }
        }
    }

    fn place_block(&mut self) {
        let eye = self.camera.eye;
        let dir = self.camera.forward();
        let hit = raycast::raycast(&self.game.world, eye, dir, REACH);
        if let Some(h) = hit {
            let (bx, by, bz) = h.block_pos;
            let (nx, ny, nz) = h.normal;
            let (px, py, pz) = (bx + nx, by + ny, bz + nz);
            if !raycast::can_place_at(&self.game.world, (px, py, pz)) {
                return;
            }
            if self.player_overlaps_block(px, py, pz) {
                return;
            }
            let blk = block::HOTBAR[self.selected];
            if let Some(r) = &self.renderer {
                self.game.edit(r, px, py, pz, blk);
            }
        }
    }

    fn player_overlaps_block(&self, bx: i32, by: i32, bz: i32) -> bool {
        let p = &self.player;
        let min = p.pos + Vec3::new(-HALF_WIDTH, 0.0, -HALF_WIDTH);
        let max = p.pos + Vec3::new(HALF_WIDTH, HEIGHT, HALF_WIDTH);
        let x = bx as f32;
        let y = by as f32;
        let z = bz as f32;
        // Cell [x,x+1] vs player AABB [min,max] — strict overlap test.
        x < max.x
            && x + 1.0 > min.x
            && y < max.y
            && y + 1.0 > min.y
            && z < max.z
            && z + 1.0 > min.z
    }

    fn on_key_pressed(&mut self, code: KeyCode, event_loop: &ActiveEventLoop) {
        match code {
            KeyCode::KeyF => self.player.fly = !self.player.fly,
            KeyCode::Escape => {
                if self.grabbed {
                    self.release_cursor();
                } else {
                    event_loop.exit();
                }
            }
            KeyCode::Digit1 => self.selected = 0,
            KeyCode::Digit2 => self.selected = 1,
            KeyCode::Digit3 => self.selected = 2,
            KeyCode::Digit4 => self.selected = 3,
            KeyCode::Digit5 => self.selected = 4,
            KeyCode::Digit6 => self.selected = 5,
            KeyCode::Digit7 => self.selected = 6,
            KeyCode::Digit8 => self.selected = 7,
            KeyCode::Digit9 => self.selected = 8,
            _ => {}
        }
    }

    fn render_frame(&mut self) {
        // Split borrows: `game` (immutable) vs `renderer` (mutable).
        let renderer = match &mut self.renderer {
            Some(r) => r,
            None => return,
        };
        let (w, h) = renderer.size();
        if w == 0 || h == 0 {
            return;
        }

        let u = build_uniforms(&self.camera, w, h);
        let eye = self.camera.eye;
        let (op, tr) = self.game.draw_lists(eye);
        let (ui_v, ui_i) = ui::build(w, h, self.selected);
        let ui_v: Vec<GpuUiVertex> = ui_v.into_iter().map(|v| v.into()).collect();

        match renderer.render(&u, &op, &tr, &ui_v, &ui_i) {
            Ok(()) => {}
            Err(wgpu::SurfaceError::Lost) | Err(wgpu::SurfaceError::Outdated) => {
                renderer.reconfigure();
            }
            Err(wgpu::SurfaceError::OutOfMemory) => {
                log::error!("GPU out of memory");
            }
            Err(wgpu::SurfaceError::Timeout) => {}
            Err(wgpu::SurfaceError::Other) => {
                log::warn!("surface error: Other");
            }
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        event_loop.set_control_flow(ControlFlow::Poll);

        let attrs = WindowAttributes::default()
            .with_title("rustcraft — Rust + wgpu voxel engine")
            .with_inner_size(winit::dpi::PhysicalSize::new(1280u32, 720u32));
        let window = Arc::new(event_loop.create_window(attrs).expect("create window"));
        let size = window.inner_size();

        let renderer = pollster::block_on(Renderer::new(
            Some(window.clone()),
            size.width,
            size.height,
        ));

        self.window = Some(window);
        self.renderer = Some(renderer);
        self.spawn();
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(sz) => {
                if let Some(r) = &mut self.renderer {
                    r.resize(sz.width, sz.height);
                }
            }
            WindowEvent::Focused(false) => self.release_cursor(),
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key,
                        state,
                        ..
                    },
                ..
            } => {
                if let PhysicalKey::Code(code) = physical_key {
                    match state {
                        ElementState::Pressed => {
                            if self.keys.insert(code) {
                                self.on_key_pressed(code, event_loop);
                            }
                        }
                        ElementState::Released => {
                            self.keys.remove(&code);
                        }
                    }
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                if state == ElementState::Pressed {
                    if !self.grabbed {
                        self.grab_cursor();
                    } else {
                        match button {
                            MouseButton::Left => self.break_block(),
                            MouseButton::Right => self.place_block(),
                            _ => {}
                        }
                    }
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let dir = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => p.y.signum() as f32,
                };
                if dir != 0.0 {
                    let n = block::HOTBAR.len() as i32;
                    let mut sel = self.selected as i32;
                    if dir > 0.0 {
                        sel = (sel + 1) % n;
                    } else {
                        sel = (sel - 1).rem_euclid(n);
                    }
                    self.selected = sel as usize;
                }
            }
            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let dt = self
                    .last_time
                    .take()
                    .map(|t| now.duration_since(t).as_secs_f32())
                    .unwrap_or(1.0 / 60.0);
                self.last_time = Some(now);

                // Camera look from accumulated raw mouse motion.
                if self.grabbed && (self.mouse_delta.0 != 0.0 || self.mouse_delta.1 != 0.0) {
                    let (dx, dy) = self.mouse_delta;
                    self.mouse_delta = (0.0, 0.0);
                    self.camera.apply_mouse_delta(dx, dy, SENSITIVITY);
                }

                // Physics.
                let input = self.move_input();
                let fwd = self.camera.forward();
                let right = self.camera.right();
                self.player.update(dt, &self.game.world, &input, fwd, right);
                self.camera.eye = self.player.eye();

                // Stream chunks around the player (time-budgeted).
                let pchunk = ChunkPos::from_world(
                    self.player.pos.x.floor() as i32,
                    self.player.pos.z.floor() as i32,
                );
                if let Some(r) = &self.renderer {
                    self.game.stream(r, pchunk, RENDER_DISTANCE, 6.0);
                }

                self.render_frame();
            }
            _ => {}
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: DeviceId,
        event: DeviceEvent,
    ) {
        if let DeviceEvent::MouseMotion { delta } = event {
            if self.grabbed {
                self.mouse_delta.0 += delta.0 as f32;
                self.mouse_delta.1 += delta.1 as f32;
            }
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(w) = &self.window {
            w.request_redraw();
        }
    }
}

fn run_windowed() {
    let event_loop = EventLoop::new().expect("create event loop");
    let mut app = App::new(42);
    event_loop.run_app(&mut app).expect("run event loop");
}
