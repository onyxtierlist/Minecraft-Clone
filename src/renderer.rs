//! wgpu renderer: surface/offscreen target, procedural texture atlas, WGSL
//! shader, and the opaque / transparent / UI render pipelines.

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;
use wgpu::TextureUsages;

use crate::atlas::build_atlas;
use crate::block::tile;
use crate::mesher::MeshData;
use crate::ui::UiVertex;

pub const ATLAS_TILE: u32 = 16;
const ATLAS_LAYERS: u32 = tile::COUNT as u32;

/// GPU-facing per-vertex data for the UI (already in clip space).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct GpuUiVertex {
    pub pos: [f32; 2],
    pub uv: [f32; 2],
    pub layer: f32,
    pub alpha: f32,
}

impl From<UiVertex> for GpuUiVertex {
    fn from(v: UiVertex) -> Self {
        Self {
            pos: v.pos,
            uv: v.uv,
            layer: v.layer,
            alpha: v.alpha,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct Uniforms {
    pub view_proj: [[f32; 4]; 4],
    /// xyz = camera world position
    pub cam_pos: [f32; 4],
    /// x = fog start, y = fog end, z = time, w = unused
    pub params: [f32; 4],
    /// rgb = sky/fog colour
    pub sky: [f32; 4],
}

impl Default for Uniforms {
    fn default() -> Self {
        Self {
            view_proj: glam::Mat4::IDENTITY.to_cols_array_2d(),
            cam_pos: [0.0; 4],
            params: [70.0, 260.0, 0.0, 0.0],
            sky: [0.55, 0.78, 0.95, 1.0],
        }
    }
}

/// A chunk's GPU geometry (world-space vertices; no per-chunk transform).
pub struct GpuMesh {
    pub vbuf: wgpu::Buffer,
    pub ibuf: wgpu::Buffer,
    pub count: u32,
}

pub struct Renderer {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    format: wgpu::TextureFormat,

    surface: Option<wgpu::Surface<'static>>,
    surface_config: Option<wgpu::SurfaceConfiguration>,
    /// (texture, view) for headless rendering.
    offscreen: Option<(wgpu::Texture, wgpu::TextureView)>,

    depth: wgpu::TextureView,
    size: (u32, u32),

    uniform_buf: wgpu::Buffer,
    bind_group: wgpu::BindGroup,

    chunk_pipeline: wgpu::RenderPipeline,
    transparent_pipeline: wgpu::RenderPipeline,
    ui_pipeline: wgpu::RenderPipeline,

    ui_vbuf: wgpu::Buffer,
    ui_ibuf: wgpu::Buffer,
    ui_vcap: usize,
    ui_icap: usize,
}

impl Renderer {
    /// `window = None` builds a headless renderer targeting an offscreen
    /// texture (screenshot verification); `Some(window)` targets a swapchain.
    pub async fn new(window: Option<Arc<winit::window::Window>>, width: u32, height: u32) -> Self {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());

        let surface = match &window {
            Some(w) => Some(instance.create_surface(w.clone()).expect("create surface")),
            None => None,
        };

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: surface.as_ref(),
                force_fallback_adapter: window.is_none(),
            })
            .await
            .expect("no suitable GPU adapter (Vulkan/Metal/DX12/GL)");

        log::info!(
            "adapter: {} ({:?})",
            adapter.get_info().name,
            adapter.get_info().device_type
        );

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::default(),
                    trace: Default::default(),
                },
            )
            .await
            .expect("request_device");

        let format = match &surface {
            Some(s) => {
                let caps = s.get_capabilities(&adapter);
                caps.formats
                    .iter()
                    .copied()
                    .find(|f| f.is_srgb())
                    .unwrap_or(caps.formats[0])
            }
            None => wgpu::TextureFormat::Rgba8UnormSrgb,
        };

        // --- Texture atlas (procedural; no asset files) ---
        let atlas_pixels = build_atlas();
        let atlas_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("atlas"),
            size: wgpu::Extent3d {
                width: ATLAS_TILE,
                height: ATLAS_TILE,
                depth_or_array_layers: ATLAS_LAYERS,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &atlas_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &atlas_pixels,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(ATLAS_TILE * 4),
                rows_per_image: Some(ATLAS_TILE),
            },
            wgpu::Extent3d {
                width: ATLAS_TILE,
                height: ATLAS_TILE,
                depth_or_array_layers: ATLAS_LAYERS,
            },
        );
        let atlas_view = atlas_tex.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("atlas_sampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // --- Uniform + texture bind group (shared by all pipelines) ---
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("uniforms"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("main_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("main_bg"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&atlas_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("main_shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("main_pl"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let chunk_vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<crate::mesher::ChunkVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: 12,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: 20,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x2,
                },
            ],
        };

        let chunk_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("chunk_opaque"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[chunk_vertex_layout.clone()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let transparent_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("chunk_transparent"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[chunk_vertex_layout.clone()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let ui_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("ui"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_ui"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<GpuUiVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute {
                            offset: 0,
                            shader_location: 0,
                            format: wgpu::VertexFormat::Float32x2,
                        },
                        wgpu::VertexAttribute {
                            offset: 8,
                            shader_location: 1,
                            format: wgpu::VertexFormat::Float32x2,
                        },
                        wgpu::VertexAttribute {
                            offset: 16,
                            shader_location: 2,
                            format: wgpu::VertexFormat::Float32,
                        },
                        wgpu::VertexAttribute {
                            offset: 20,
                            shader_location: 3,
                            format: wgpu::VertexFormat::Float32,
                        },
                    ],
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_ui"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // --- Surface / offscreen target ---
        let surface_config = match &surface {
            Some(s) => {
                let config = wgpu::SurfaceConfiguration {
                    usage: TextureUsages::RENDER_ATTACHMENT,
                    format,
                    width: width.max(1),
                    height: height.max(1),
                    present_mode: wgpu::PresentMode::Fifo,
                    desired_maximum_frame_latency: 2,
                    alpha_mode: wgpu::CompositeAlphaMode::Auto,
                    view_formats: vec![],
                };
                s.configure(&device, &config);
                Some(config)
            }
            None => None,
        };
        let offscreen = if surface.is_none() {
            Some(create_offscreen(&device, format, width, height))
        } else {
            None
        };

        let size = (width.max(1), height.max(1));
        let depth = create_depth_view(&device, size.0, size.1);

        let ui_vbuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ui_vbuf"),
            size: 8192,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let ui_ibuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ui_ibuf"),
            size: 4096,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            device,
            queue,
            format,
            surface,
            surface_config,
            offscreen,
            depth,
            size,
            uniform_buf,
            bind_group,
            chunk_pipeline,
            transparent_pipeline,
            ui_pipeline,
            ui_vbuf,
            ui_ibuf,
            ui_vcap: 8192,
            ui_icap: 4096,
        }
    }

    pub fn size(&self) -> (u32, u32) {
        self.size
    }

    /// Re-apply the surface configuration (after Lost/Outdated errors).
    pub fn reconfigure(&mut self) {
        if let (Some(surface), Some(config)) = (&self.surface, &self.surface_config) {
            surface.configure(&self.device, config);
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        let width = width.max(1);
        let height = height.max(1);
        if (width, height) == self.size {
            return;
        }
        self.size = (width, height);
        if let (Some(surface), Some(config)) = (&self.surface, &mut self.surface_config) {
            config.width = width;
            config.height = height;
            surface.configure(&self.device, config);
        }
        if self.surface.is_none() {
            self.offscreen = Some(create_offscreen(&self.device, self.format, width, height));
        }
        self.depth = create_depth_view(&self.device, width, height);
    }

    fn upload_ui(&mut self, vbytes: usize, ibytes: usize) {
        if vbytes > self.ui_vcap {
            self.ui_vcap = vbytes.next_power_of_two().max(8192);
            self.ui_vbuf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("ui_vbuf"),
                size: self.ui_vcap as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }
        if ibytes > self.ui_icap {
            self.ui_icap = ibytes.next_power_of_two().max(4096);
            self.ui_ibuf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("ui_ibuf"),
                size: self.ui_icap as u64,
                usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }
    }

    pub fn upload_mesh(&self, data: &MeshData) -> Option<GpuMesh> {
        if data.is_empty() {
            return None;
        }
        let vbuf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("chunk_vbuf"),
            contents: bytemuck::cast_slice(&data.vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let ibuf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("chunk_ibuf"),
            contents: bytemuck::cast_slice(&data.indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        Some(GpuMesh {
            vbuf,
            ibuf,
            count: data.indices.len() as u32,
        })
    }

    /// Render one frame. Returns `SurfaceError` only for windowed rendering.
    pub fn render(
        &mut self,
        u: &Uniforms,
        opaque: &[&GpuMesh],
        transparent: &[&GpuMesh],
        ui_vertices: &[GpuUiVertex],
        ui_indices: &[u32],
    ) -> Result<(), wgpu::SurfaceError> {
        self.queue
            .write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(u));

        let mut surface_frame: Option<wgpu::SurfaceTexture> = None;
        let target: wgpu::TextureView = match &self.surface {
            Some(surface) => {
                let frame = match surface.get_current_texture() {
                    Ok(f) => f,
                    Err(e) => return Err(e),
                };
                let view = frame
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());
                surface_frame = Some(frame);
                view
            }
            None => self
                .offscreen
                .as_ref()
                .expect("offscreen target required in headless mode")
                .1
                .clone(),
        };

        let clear = wgpu::Color {
            r: u.sky[0] as f64,
            g: u.sky[1] as f64,
            b: u.sky[2] as f64,
            a: 1.0,
        };

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frame"),
            });

        // World pass: opaque then transparent, with depth testing.
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("world"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(clear),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            pass.set_bind_group(0, &self.bind_group, &[]);

            pass.set_pipeline(&self.chunk_pipeline);
            for m in opaque {
                if m.count == 0 {
                    continue;
                }
                pass.set_vertex_buffer(0, m.vbuf.slice(..));
                pass.set_index_buffer(m.ibuf.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..m.count, 0, 0..1);
            }

            pass.set_pipeline(&self.transparent_pipeline);
            for m in transparent {
                if m.count == 0 {
                    continue;
                }
                pass.set_vertex_buffer(0, m.vbuf.slice(..));
                pass.set_index_buffer(m.ibuf.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..m.count, 0, 0..1);
            }
        }

        // UI pass: no depth, drawn over the world.
        if !ui_indices.is_empty() {
            self.upload_ui(
                std::mem::size_of_val(ui_vertices),
                std::mem::size_of_val(ui_indices),
            );
            self.queue
                .write_buffer(&self.ui_vbuf, 0, bytemuck::cast_slice(ui_vertices));
            self.queue
                .write_buffer(&self.ui_ibuf, 0, bytemuck::cast_slice(ui_indices));

            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("ui"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.ui_pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.set_vertex_buffer(0, self.ui_vbuf.slice(..));
            pass.set_index_buffer(self.ui_ibuf.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..ui_indices.len() as u32, 0, 0..1);
        }

        self.queue.submit([encoder.finish()]);

        if let Some(frame) = surface_frame {
            frame.present();
        }
        Ok(())
    }

    /// Read back the headless target as raw sRGB-encoded RGBA8 bytes.
    pub fn read_back(&self) -> (Vec<u8>, u32, u32) {
        let (w, h) = self.size;
        let bytes_per_row = w * 4;
        // Index 0 is the Texture (needed for the GPU->buffer copy).
        let (tex, _) = self
            .offscreen
            .as_ref()
            .expect("read_back is only valid for headless rendering");

        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: (bytes_per_row * h) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("readback"),
            });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit([encoder.finish()]);

        let slice = buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });

        // Drive the device until the mapping resolves (no event loop headlessly).
        let mapped = loop {
            if let Ok(res) = rx.try_recv() {
                break res.expect("buffer map failed");
            }
            let _ = self.device.poll(wgpu::PollType::Wait);
        };
        let _ = mapped;

        let data = slice.get_mapped_range().to_vec();
        buffer.unmap();
        (data, w, h)
    }
}

fn create_offscreen(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    width: u32,
    height: u32,
) -> (wgpu::Texture, wgpu::TextureView) {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("offscreen"),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    (tex, view)
}

fn create_depth_view(device: &wgpu::Device, width: u32, height: u32) -> wgpu::TextureView {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("depth"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    tex.create_view(&wgpu::TextureViewDescriptor::default())
}

const SHADER: &str = r#"
struct Uniforms {
    view_proj: mat4x4<f32>,
    cam_pos:   vec4<f32>,
    params:    vec4<f32>, // x=fog start, y=fog end, z=time
    sky:       vec4<f32>,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var t_atlas: texture_2d_array<f32>;
@group(0) @binding(2) var s_atlas: sampler;

// ---- World (voxel) shaders ----
struct VsIn {
    @location(0) pos: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) layer_light: vec2<f32>,
};

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) layer: i32,
    @location(2) light: f32,
    @location(3) dist: f32,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    var out: VsOut;
    out.clip = uniforms.view_proj * vec4<f32>(in.pos, 1.0);
    out.uv = in.uv;
    out.layer = i32(in.layer_light.x);
    out.light = in.layer_light.y;
    out.dist = distance(in.pos, uniforms.cam_pos.xyz);
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let tex = textureSample(t_atlas, s_atlas, in.uv, in.layer);
    if (tex.a < 0.35) { discard; }              // cut-out (leaves, glass centre)
    var col = tex.rgb * in.light;
    let fog = smoothstep(uniforms.params.x, uniforms.params.y, in.dist);
    col = mix(col, uniforms.sky.rgb, fog);
    return vec4<f32>(col, tex.a);
}

// ---- 2D overlay (hotbar + crosshair) ----
struct UiVsIn {
    @location(0) pos: vec2<f32>,   // already in clip space
    @location(1) uv: vec2<f32>,
    @location(2) layer: f32,
    @location(3) alpha: f32,
};

struct UiVsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) layer: i32,
    @location(2) alpha: f32,
};

@vertex
fn vs_ui(in: UiVsIn) -> UiVsOut {
    var out: UiVsOut;
    out.clip = vec4<f32>(in.pos, 0.0, 1.0);
    out.uv = in.uv;
    out.layer = i32(in.layer);
    out.alpha = in.alpha;
    return out;
}

@fragment
fn fs_ui(in: UiVsOut) -> @location(0) vec4<f32> {
    let tex = textureSample(t_atlas, s_atlas, in.uv, in.layer);
    return vec4<f32>(tex.rgb, tex.a * in.alpha);
}
"#;
