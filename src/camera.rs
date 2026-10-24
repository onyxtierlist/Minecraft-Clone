//! First-person camera and view-projection matrices.

use glam::{Mat4, Vec3};

pub const DEFAULT_FOV_Y: f32 = 70.0f32.to_radians();
pub const NEAR: f32 = 0.1;
pub const FAR: f32 = 1000.0;

pub struct Camera {
    pub eye: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub fov_y: f32,
}

impl Camera {
    pub fn new(eye: Vec3) -> Self {
        Self {
            eye,
            yaw: 0.0,
            pitch: -0.15,
            fov_y: DEFAULT_FOV_Y,
        }
    }

    /// Unit forward vector from yaw/pitch.
    pub fn forward(&self) -> Vec3 {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        Vec3::new(cy * cp, sp, sy * cp)
    }

    pub fn right(&self) -> Vec3 {
        // Horizontal right, independent of pitch (keeps strafing flat).
        let (sy, cy) = self.yaw.sin_cos();
        Vec3::new(sy, 0.0, -cy)
    }

    pub fn apply_mouse_delta(&mut self, dx: f32, dy: f32, sensitivity: f32) {
        self.yaw += dx * sensitivity;
        self.pitch -= dy * sensitivity; // screen +y is down; invert for natural look
        let limit = 89.0f32.to_radians();
        self.pitch = self.pitch.clamp(-limit, limit);
        // Keep yaw in a sane range to avoid float drift over long sessions.
        self.yaw = self.yaw.rem_euclid(std::f32::consts::TAU);
    }

    /// Right-handed perspective with [0,1] depth (wgpu/Vulkan/Metal clip space).
    pub fn projection(&self, aspect: f32) -> Mat4 {
        Mat4::perspective_rh(self.fov_y, aspect, NEAR, FAR)
    }

    pub fn view(&self) -> Mat4 {
        Mat4::look_at_rh(self.eye, self.eye + self.forward(), Vec3::Y)
    }

    pub fn view_proj(&self, aspect: f32) -> Mat4 {
        self.projection(aspect) * self.view()
    }
}
