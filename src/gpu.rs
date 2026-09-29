//! GPU compute backend for emulsion synthesis, development, and optical scanning using wgpu.

use crate::crystal::{HalideCrystal, Morphology};
use crate::error::GpuError;
use pollster::FutureExt;
use std::sync::OnceLock;

/// Global singleton instance of the GPU engine, initialized on first use if --gpu is specified.
static GLOBAL_GPU_ENGINE: OnceLock<Result<GpuEngine, GpuError>> = OnceLock::new();

/// Returns a reference to the global GPU engine if available and initialized.
pub fn get_gpu_engine() -> Option<&'static GpuEngine> {
    GLOBAL_GPU_ENGINE.get_or_init(GpuEngine::new).as_ref().ok()
}

/// Parameters describing a single tile domain for end-to-end GPU film rendering.
#[derive(Debug, Clone, Copy)]
pub struct GpuTileParams {
    pub film_w: f64,
    pub film_h: f64,
    pub film_d: f64,
    pub width_px: u32,
    pub height_px: u32,
    pub samples_per_pixel: u32,
    pub silver_opacity: f64,
    pub seed: u64,
    pub r_min: f64,
    pub mean_crystal_radius: f64,
    pub size_variance: f64,
    pub morphology: u32, // 0 = Cubic, 1 = Tabular, 2 = Octahedral
    pub tabular_aspect_ratio: f64,
    pub base_sensitivity: f64,
    pub exposure: f64,
    pub threshold_photons: u32,
    pub filament_expansion: f64,
    pub clumping_proximity_factor: f64,
}

impl Default for GpuTileParams {
    fn default() -> Self {
        Self {
            film_w: 400.0,
            film_h: 400.0,
            film_d: 14.0,
            width_px: 320,
            height_px: 320,
            samples_per_pixel: 4,
            silver_opacity: 0.40,
            seed: 42,
            r_min: 1.2,
            mean_crystal_radius: 0.55,
            size_variance: 0.40,
            morphology: 0,
            tabular_aspect_ratio: 1.0,
            base_sensitivity: 1.2,
            exposure: 1.0,
            threshold_photons: 4,
            filament_expansion: 1.35,
            clumping_proximity_factor: 1.25,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PoissonUniforms {
    pub grid_w: u32,
    pub grid_h: u32,
    pub grid_d: u32,
    pub phase: u32,

    pub cell_size: f32,
    pub r_min: f32,
    pub mean_radius: f32,
    pub size_variance: f32,

    pub film_w: f32,
    pub film_h: f32,
    pub film_d: f32,
    pub seed: u32,

    pub morphology: u32,
    pub tabular_aspect_ratio: f32,
    pub base_sensitivity: f32,
    pub pad: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DevelopUniforms {
    pub grid_w: u32,
    pub grid_h: u32,
    pub grid_d: u32,
    pub threshold_photons: u32,

    pub width_px: u32,
    pub height_px: u32,
    pub seed: u32,
    pub stage: u32,

    pub film_w: f32,
    pub film_h: f32,
    pub film_d: f32,
    pub cell_size: f32,

    pub exposure: f32,
    pub filament_expansion: f32,
    pub clumping_factor: f32,
    pub base_sensitivity: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ScanUniforms {
    pub width_px: u32,
    pub height_px: u32,
    pub samples_per_pixel: u32,
    pub grid_w: u32,

    pub grid_h: u32,
    pub grid_d: u32,
    pub seed: u32,
    pub pad0: u32,

    pub film_w: f32,
    pub film_h: f32,
    pub film_d: f32,
    pub cell_size: f32,

    pub silver_opacity: f32,
    pub pad1: f32,
    pub pad2: f32,
    pub pad3: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuGridCell {
    pub pos: [f32; 4],
    pub dims_exp: [f32; 4],
    pub rot: [f32; 4],
    pub flags: [u32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct GaussianUniforms {
    width_px: u32,
    height_px: u32,
    direction: u32,
    pad: u32,
}

/// GPU Compute Engine executing 3D Poisson-disk sampling, chemical development, optical scanning, and filtering.
pub struct GpuEngine {
    device: wgpu::Device,
    queue: wgpu::Queue,
    clear_pipeline: wgpu::ComputePipeline,
    poisson_gen_pipeline: wgpu::ComputePipeline,
    poisson_resolve_pipeline: wgpu::ComputePipeline,
    poisson_bind_group_layout: wgpu::BindGroupLayout,
    develop_pipeline: wgpu::ComputePipeline,
    develop_bind_group_layout: wgpu::BindGroupLayout,
    scan_pipeline: wgpu::ComputePipeline,
    scan_bind_group_layout: wgpu::BindGroupLayout,
    gauss_pipeline: wgpu::ComputePipeline,
    gauss_bind_group_layout: wgpu::BindGroupLayout,
    adapter_name: String,
    dispatch_lock: std::sync::Mutex<()>,
}

impl GpuEngine {
    /// Initializes wgpu, selecting the high-performance GPU adapter and compiling compute shaders.
    pub fn new() -> Result<Self, GpuError> {
        let instance = wgpu::Instance::default();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
                ..Default::default()
            })
            .block_on()
            .map_err(|e| GpuError::AdapterNotFound(format!("{e:?}")))?;

        let adapter_info = adapter.get_info();
        let adapter_name = format!("{} ({:?})", adapter_info.name, adapter_info.backend);

        let limits = adapter.limits();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("fgrain-gpu-device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits {
                    max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size,
                    max_buffer_size: limits.max_buffer_size,
                    max_storage_buffers_per_shader_stage: limits
                        .max_storage_buffers_per_shader_stage,
                    max_compute_workgroups_per_dimension: limits
                        .max_compute_workgroups_per_dimension,
                    ..wgpu::Limits::default()
                },
                memory_hints: wgpu::MemoryHints::Performance,
                ..Default::default()
            })
            .block_on()
            .map_err(|e| GpuError::DeviceRequestFailed(format!("{e:?}")))?;

        // Poisson pipeline (grid clear, candidate generation, conflict resolution)
        let poisson_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("poisson-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/poisson.wgsl").into()),
        });

        let poisson_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("poisson-bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let poisson_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("poisson-pipeline-layout"),
                bind_group_layouts: &[Some(&poisson_bind_group_layout)],
                ..Default::default()
            });

        let clear_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("clear-compute-pipeline"),
            layout: Some(&poisson_pipeline_layout),
            module: &poisson_shader,
            entry_point: Some("clear_grid"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let poisson_gen_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("poisson-gen-pipeline"),
                layout: Some(&poisson_pipeline_layout),
                module: &poisson_shader,
                entry_point: Some("generate_candidates"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        let poisson_resolve_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("poisson-resolve-pipeline"),
                layout: Some(&poisson_pipeline_layout),
                module: &poisson_shader,
                entry_point: Some("resolve_conflicts"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        // Exposure and development pipeline
        let develop_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("develop-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/develop.wgsl").into()),
        });

        let develop_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("develop-bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let develop_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("develop-pipeline-layout"),
                bind_group_layouts: &[Some(&develop_bind_group_layout)],
                ..Default::default()
            });

        let develop_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("develop-compute-pipeline"),
            layout: Some(&develop_pipeline_layout),
            module: &develop_shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        // Transmittance scan pipeline
        let scan_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("scan-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/scan.wgsl").into()),
        });

        let scan_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("scan-bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let scan_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("scan-pipeline-layout"),
            bind_group_layouts: &[Some(&scan_bind_group_layout)],
            ..Default::default()
        });

        let scan_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("scan-compute-pipeline"),
            layout: Some(&scan_pipeline_layout),
            module: &scan_shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        // Gaussian smoothing filter pipeline
        let gauss_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gaussian-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/gaussian.wgsl").into()),
        });

        let gauss_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("gauss-bind-group-layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let gauss_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("gauss-pipeline-layout"),
                bind_group_layouts: &[Some(&gauss_bind_group_layout)],
                ..Default::default()
            });

        let gauss_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("gauss-compute-pipeline"),
            layout: Some(&gauss_pipeline_layout),
            module: &gauss_shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Ok(Self {
            device,
            queue,
            clear_pipeline,
            poisson_gen_pipeline,
            poisson_resolve_pipeline,
            poisson_bind_group_layout,
            develop_pipeline,
            develop_bind_group_layout,
            scan_pipeline,
            scan_bind_group_layout,
            gauss_pipeline,
            gauss_bind_group_layout,
            adapter_name,
            dispatch_lock: std::sync::Mutex::new(()),
        })
    }

    /// Returns human-readable adapter name and backend (e.g. "NVIDIA GeForce RTX 4070 Ti SUPER (Vulkan)").
    pub fn adapter_name(&self) -> &str {
        &self.adapter_name
    }

    /// Fully renders a tile on the GPU: 3D Poisson synthesis, exposure & development, scanning, and filtering.
    ///
    /// The entire physical simulation executes on the GPU without intermediate CPU memory allocations.
    pub fn render_tile_all_gpu(
        &self,
        tile_img: &[f32],
        params: &GpuTileParams,
    ) -> Result<Vec<f32>, GpuError> {
        let width_px = params.width_px;
        let height_px = params.height_px;
        let num_pixels = (width_px * height_px) as usize;
        if num_pixels == 0 {
            return Ok(Vec::new());
        }

        let _lock = self.dispatch_lock.lock().unwrap_or_else(|e| e.into_inner());

        // Grid dimensions and bounds
        let cell_size = (params.r_min.max(1.0)) as f32;
        let max_grid_dim = 512u32;
        let raw_grid_w = (params.film_w as f32 / cell_size).ceil() as u32;
        let raw_grid_h = (params.film_h as f32 / cell_size).ceil() as u32;

        let (cell_size, grid_w, grid_h) = if raw_grid_w > max_grid_dim || raw_grid_h > max_grid_dim
        {
            let scale_factor = (raw_grid_w.max(raw_grid_h) as f32) / (max_grid_dim as f32);
            let adjusted_cell_size = cell_size * scale_factor;
            let gw =
                ((params.film_w as f32 / adjusted_cell_size).ceil() as u32).clamp(1, max_grid_dim);
            let gh =
                ((params.film_h as f32 / adjusted_cell_size).ceil() as u32).clamp(1, max_grid_dim);
            (adjusted_cell_size, gw, gh)
        } else {
            (cell_size, raw_grid_w.max(1), raw_grid_h.max(1))
        };
        let grid_d = ((params.film_d as f32 / cell_size).ceil() as u32).clamp(1, 32);
        let total_cells = (grid_w * grid_h * grid_d) as usize;

        use wgpu::util::DeviceExt;

        // Storage buffers
        let cells_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu-cells-buffer"),
            size: (total_cells * std::mem::size_of::<GpuGridCell>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let input_img_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("gpu-input-img-buffer"),
                contents: bytemuck::cast_slice(tile_img),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let pixel_bytes = (num_pixels * std::mem::size_of::<f32>()) as u64;

        let density_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu-density-buffer"),
            size: pixel_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let gauss_temp_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu-gauss-temp-buffer"),
            size: pixel_bytes,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let out_delta_d_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu-out-delta-d-buffer"),
            size: pixel_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let readback_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu-readback-buffer"),
            size: pixel_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Poisson uniforms and bind group
        let base_poisson_uniforms = PoissonUniforms {
            grid_w,
            grid_h,
            grid_d,
            phase: 0,
            cell_size,
            r_min: params.r_min as f32,
            mean_radius: params.mean_crystal_radius as f32,
            size_variance: params.size_variance as f32,
            film_w: params.film_w as f32,
            film_h: params.film_h as f32,
            film_d: params.film_d as f32,
            seed: (params.seed & 0xFFFFFFFF) as u32,
            morphology: params.morphology,
            tabular_aspect_ratio: params.tabular_aspect_ratio as f32,
            base_sensitivity: params.base_sensitivity as f32,
            pad: 0,
        };

        let poisson_ubuf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("poisson-ubuf"),
                contents: bytemuck::bytes_of(&base_poisson_uniforms),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let poisson_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("poisson-bg"),
            layout: &self.poisson_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: poisson_ubuf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: cells_buffer.as_entire_binding(),
                },
            ],
        });

        // Development uniforms and bind groups (pass 0: exposure, pass 1: clumping)
        let dev_p0_uniforms = DevelopUniforms {
            grid_w,
            grid_h,
            grid_d,
            threshold_photons: params.threshold_photons,
            width_px,
            height_px,
            seed: (params.seed & 0xFFFFFFFF) as u32,
            stage: 0,
            film_w: params.film_w as f32,
            film_h: params.film_h as f32,
            film_d: params.film_d as f32,
            cell_size,
            exposure: params.exposure as f32,
            filament_expansion: params.filament_expansion as f32,
            clumping_factor: params.clumping_proximity_factor as f32,
            base_sensitivity: params.base_sensitivity as f32,
        };
        let dev_p0_ubuf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("develop-p0-ubuf"),
                contents: bytemuck::bytes_of(&dev_p0_uniforms),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let dev_p0_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("develop-p0-bg"),
            layout: &self.develop_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: dev_p0_ubuf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: input_img_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: cells_buffer.as_entire_binding(),
                },
            ],
        });

        let mut dev_p1_uniforms = dev_p0_uniforms;
        dev_p1_uniforms.stage = 1;
        let dev_p1_ubuf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("develop-p1-ubuf"),
                contents: bytemuck::bytes_of(&dev_p1_uniforms),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let dev_p1_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("develop-p1-bg"),
            layout: &self.develop_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: dev_p1_ubuf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: input_img_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: cells_buffer.as_entire_binding(),
                },
            ],
        });

        // Scan uniforms and bind group
        let scan_uniforms = ScanUniforms {
            width_px,
            height_px,
            samples_per_pixel: params.samples_per_pixel,
            grid_w,
            grid_h,
            grid_d,
            seed: (params.seed.wrapping_add(1013904223) & 0xFFFFFFFF) as u32,
            pad0: 0,
            film_w: params.film_w as f32,
            film_h: params.film_h as f32,
            film_d: params.film_d as f32,
            cell_size,
            silver_opacity: params.silver_opacity as f32,
            pad1: 0.0,
            pad2: 0.0,
            pad3: 0.0,
        };
        let scan_ubuf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("scan-ubuf"),
                contents: bytemuck::bytes_of(&scan_uniforms),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let scan_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scan-bg"),
            layout: &self.scan_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: scan_ubuf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: cells_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: density_buffer.as_entire_binding(),
                },
            ],
        });

        // Gaussian filter uniforms and bind groups
        let gauss_h_uniforms = GaussianUniforms {
            width_px,
            height_px,
            direction: 0,
            pad: 0,
        };
        let gauss_h_ubuf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("gauss-h-ubuf"),
                contents: bytemuck::bytes_of(&gauss_h_uniforms),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let gauss_h_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gauss-h-bg"),
            layout: &self.gauss_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: gauss_h_ubuf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: density_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: gauss_temp_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: density_buffer.as_entire_binding(),
                },
            ],
        });

        let gauss_v_uniforms = GaussianUniforms {
            width_px,
            height_px,
            direction: 1,
            pad: 0,
        };
        let gauss_v_ubuf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("gauss-v-ubuf"),
                contents: bytemuck::bytes_of(&gauss_v_uniforms),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let gauss_v_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gauss-v-bg"),
            layout: &self.gauss_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: gauss_v_ubuf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: gauss_temp_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out_delta_d_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: density_buffer.as_entire_binding(),
                },
            ],
        });

        // Dispatch compute passes
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("fgrain-all-gpu-encoder"),
            });

        let pw_x = grid_w.div_ceil(8);
        let pw_y = grid_h.div_ceil(8);
        let pw_z = grid_d.div_ceil(4);

        // Grid clear pass
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("clear-grid-pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.clear_pipeline);
            cpass.set_bind_group(0, &poisson_bg, &[]);
            cpass.dispatch_workgroups(pw_x, pw_y, pw_z);
        }

        // Candidate generation pass
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("poisson-gen-pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.poisson_gen_pipeline);
            cpass.set_bind_group(0, &poisson_bg, &[]);
            cpass.dispatch_workgroups(pw_x, pw_y, pw_z);
        }

        // Matérn conflict resolution pass
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("poisson-resolve-pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.poisson_resolve_pipeline);
            cpass.set_bind_group(0, &poisson_bg, &[]);
            cpass.dispatch_workgroups(pw_x, pw_y, pw_z);
        }

        // Latent image exposure pass
        let dev_wg_x = grid_w.div_ceil(8);
        let dev_wg_y = grid_h.div_ceil(8);
        let dev_wg_z = grid_d.div_ceil(4);
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("develop-pass-0"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.develop_pipeline);
            cpass.set_bind_group(0, &dev_p0_bg, &[]);
            cpass.dispatch_workgroups(dev_wg_x, dev_wg_y, dev_wg_z);
        }

        // Clumping development pass
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("develop-pass-1-clumping"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.develop_pipeline);
            cpass.set_bind_group(0, &dev_p1_bg, &[]);
            cpass.dispatch_workgroups(dev_wg_x, dev_wg_y, dev_wg_z);
        }

        // Optical transmittance ray scan pass
        let scan_wg_x = width_px.div_ceil(16);
        let scan_wg_y = height_px.div_ceil(16);
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("scan-pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.scan_pipeline);
            cpass.set_bind_group(0, &scan_bg, &[]);
            cpass.dispatch_workgroups(scan_wg_x, scan_wg_y, 1);
        }

        // Horizontal Gaussian blur pass
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gauss-h-pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.gauss_pipeline);
            cpass.set_bind_group(0, &gauss_h_bg, &[]);
            cpass.dispatch_workgroups(scan_wg_x, scan_wg_y, 1);
        }

        // Vertical Gaussian blur pass
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gauss-v-pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.gauss_pipeline);
            cpass.set_bind_group(0, &gauss_v_bg, &[]);
            cpass.dispatch_workgroups(scan_wg_x, scan_wg_y, 1);
        }

        // Copy density buffer to readback staging buffer
        encoder.copy_buffer_to_buffer(&density_buffer, 0, &readback_buffer, 0, pixel_bytes);
        self.queue.submit(Some(encoder.finish()));

        // Read back mapped results to CPU buffer
        let slice = readback_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });

        let _ = self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
        rx.recv()
            .map_err(|e| GpuError::ChannelError(format!("{e:?}")))?
            .map_err(|e| GpuError::BufferMapError(format!("{e:?}")))?;

        let mapped_view = slice
            .get_mapped_range()
            .map_err(|e| GpuError::BufferRangeError(format!("{e:?}")))?;
        let d_neg: Vec<f32> = bytemuck::cast_slice(&mapped_view).to_vec();
        drop(mapped_view);
        readback_buffer.unmap();

        Ok(d_neg)
    }

    /// Compatibility method for optical scanning of pre-existing crystals on the GPU.
    pub fn render_tile_delta_d(
        &self,
        developed_crystals: &[&HalideCrystal],
        params: &GpuTileParams,
    ) -> Result<Vec<f32>, GpuError> {
        let width_px = params.width_px;
        let height_px = params.height_px;
        let num_pixels = (width_px * height_px) as usize;
        if num_pixels == 0 {
            return Ok(Vec::new());
        }

        let _lock = self.dispatch_lock.lock().unwrap_or_else(|e| e.into_inner());

        let cell_size = (params.r_min.max(1.0)) as f32;
        let grid_w = ((params.film_w as f32 / cell_size).ceil() as u32).max(1);
        let grid_h = ((params.film_h as f32 / cell_size).ceil() as u32).max(1);
        let grid_d = ((params.film_d as f32 / cell_size).ceil() as u32).max(1);
        let total_cells = (grid_w * grid_h * grid_d) as usize;

        let mut cells = vec![
            GpuGridCell {
                pos: [0.0; 4],
                dims_exp: [0.0; 4],
                rot: [0.0, 0.0, 0.0, 1.0],
                flags: [0; 4],
            };
            total_cells
        ];

        for &c in developed_crystals {
            let gx = ((c.position.x as f32 / cell_size).floor() as i32).clamp(0, grid_w as i32 - 1)
                as u32;
            let gy = ((c.position.y as f32 / cell_size).floor() as i32).clamp(0, grid_h as i32 - 1)
                as u32;
            let gz = ((c.position.z as f32 / cell_size).floor() as i32).clamp(0, grid_d as i32 - 1)
                as u32;
            let cell_idx = (gz * (grid_w * grid_h) + gy * grid_w + gx) as usize;

            let morph = match c.morphology {
                Morphology::Cubic => 0u32,
                Morphology::Tabular => 1u32,
                Morphology::Octahedral => 2u32,
            };
            cells[cell_idx] = GpuGridCell {
                pos: [
                    c.position.x as f32,
                    c.position.y as f32,
                    c.position.z as f32,
                    0.0,
                ],
                dims_exp: [
                    c.dimensions.x as f32,
                    c.dimensions.y as f32,
                    c.dimensions.z as f32,
                    c.filament_expansion as f32,
                ],
                rot: [
                    c.orientation.i as f32,
                    c.orientation.j as f32,
                    c.orientation.k as f32,
                    c.orientation.w as f32,
                ],
                flags: [1, morph, 1, 0],
            };
        }

        use wgpu::util::DeviceExt;

        let cells_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("gpu-cells-buffer-compat"),
                contents: bytemuck::cast_slice(&cells),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let pixel_bytes = (num_pixels * std::mem::size_of::<f32>()) as u64;

        let density_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu-density-buffer-compat"),
            size: pixel_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let gauss_temp_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu-gauss-temp-buffer-compat"),
            size: pixel_bytes,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let out_delta_d_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu-out-delta-d-buffer-compat"),
            size: pixel_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let readback_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu-readback-buffer-compat"),
            size: pixel_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let scan_uniforms = ScanUniforms {
            width_px,
            height_px,
            samples_per_pixel: params.samples_per_pixel,
            grid_w,
            grid_h,
            grid_d,
            seed: (params.seed.wrapping_add(1013904223) & 0xFFFFFFFF) as u32,
            pad0: 0,
            film_w: params.film_w as f32,
            film_h: params.film_h as f32,
            film_d: params.film_d as f32,
            cell_size,
            silver_opacity: params.silver_opacity as f32,
            pad1: 0.0,
            pad2: 0.0,
            pad3: 0.0,
        };
        let scan_ubuf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("scan-ubuf-compat"),
                contents: bytemuck::bytes_of(&scan_uniforms),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let scan_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scan-bg-compat"),
            layout: &self.scan_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: scan_ubuf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: cells_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: density_buffer.as_entire_binding(),
                },
            ],
        });

        let gauss_h_uniforms = GaussianUniforms {
            width_px,
            height_px,
            direction: 0,
            pad: 0,
        };
        let gauss_h_ubuf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("gauss-h-ubuf-compat"),
                contents: bytemuck::bytes_of(&gauss_h_uniforms),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let gauss_h_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gauss-h-bg-compat"),
            layout: &self.gauss_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: gauss_h_ubuf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: density_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: gauss_temp_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: density_buffer.as_entire_binding(),
                },
            ],
        });

        let gauss_v_uniforms = GaussianUniforms {
            width_px,
            height_px,
            direction: 1,
            pad: 0,
        };
        let gauss_v_ubuf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("gauss-v-ubuf-compat"),
                contents: bytemuck::bytes_of(&gauss_v_uniforms),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let gauss_v_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gauss-v-bg-compat"),
            layout: &self.gauss_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: gauss_v_ubuf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: gauss_temp_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out_delta_d_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: density_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("fgrain-compat-encoder"),
            });

        let scan_wg_x = width_px.div_ceil(16);
        let scan_wg_y = height_px.div_ceil(16);

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("scan-pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.scan_pipeline);
            cpass.set_bind_group(0, &scan_bg, &[]);
            cpass.dispatch_workgroups(scan_wg_x, scan_wg_y, 1);
        }

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gauss-h-pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.gauss_pipeline);
            cpass.set_bind_group(0, &gauss_h_bg, &[]);
            cpass.dispatch_workgroups(scan_wg_x, scan_wg_y, 1);
        }

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gauss-v-pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.gauss_pipeline);
            cpass.set_bind_group(0, &gauss_v_bg, &[]);
            cpass.dispatch_workgroups(scan_wg_x, scan_wg_y, 1);
        }

        encoder.copy_buffer_to_buffer(&out_delta_d_buffer, 0, &readback_buffer, 0, pixel_bytes);
        self.queue.submit(Some(encoder.finish()));

        let slice = readback_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });

        let _ = self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
        rx.recv()
            .map_err(|e| GpuError::ChannelError(format!("{e:?}")))?
            .map_err(|e| GpuError::BufferMapError(format!("{e:?}")))?;

        let mapped_view = slice
            .get_mapped_range()
            .map_err(|e| GpuError::BufferRangeError(format!("{e:?}")))?;
        let delta_d: Vec<f32> = bytemuck::cast_slice(&mapped_view).to_vec();
        drop(mapped_view);
        readback_buffer.unmap();

        Ok(delta_d)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Point3, UnitQuaternion, Vector3};

    #[test]
    fn test_gpu_engine_initialization_and_render() {
        let engine = match GpuEngine::new() {
            Ok(e) => e,
            Err(e) => {
                println!("Skipping GPU test (no GPU adapter available): {}", e);
                return;
            }
        };

        println!("Running GPU test on: {}", engine.adapter_name());

        let mut crystal = HalideCrystal::new(
            0,
            Point3::new(5.0, 5.0, 5.0),
            Vector3::new(1.0, 1.0, 1.0),
            UnitQuaternion::identity(),
            Morphology::Cubic,
            1.0,
        );
        crystal.developed = true;
        crystal.filament_expansion = 1.35;

        let params = GpuTileParams {
            film_w: 10.0,
            film_h: 10.0,
            film_d: 10.0,
            width_px: 16,
            height_px: 16,
            samples_per_pixel: 4,
            silver_opacity: 20.0,
            ..GpuTileParams::default()
        };

        let delta_d = engine
            .render_tile_delta_d(&[&crystal], &params)
            .expect("GPU rendering failed");

        assert_eq!(delta_d.len(), 256);
        let has_variation = delta_d.iter().any(|&v| v.abs() > 1e-4);
        assert!(has_variation, "GPU delta_d should have optical variation");
    }

    #[test]
    fn test_gpu_engine_all_gpu_render() {
        let engine = match GpuEngine::new() {
            Ok(e) => e,
            Err(e) => {
                println!("Skipping GPU test (no GPU adapter available): {}", e);
                return;
            }
        };

        for exp_val in [0.0f32, 0.05, 0.18, 0.5, 1.0] {
            let tile_img = vec![exp_val; 16 * 16];
            let params = GpuTileParams {
                film_w: 15.0,
                film_h: 15.0,
                film_d: 10.0,
                width_px: 16,
                height_px: 16,
                r_min: 1.2,
                silver_opacity: 0.32,
                ..GpuTileParams::default()
            };

            let d_neg = engine
                .render_tile_all_gpu(&tile_img, &params)
                .expect("All-GPU tile render failed");

            assert_eq!(d_neg.len(), 256);
            let min_v = d_neg.iter().cloned().fold(f32::INFINITY, f32::min);
            let max_v = d_neg.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mean = d_neg.iter().sum::<f32>() / d_neg.len() as f32;
            let std = (d_neg.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / d_neg.len() as f32)
                .sqrt();
            println!(
                "Exp {}: min={:.4}, max={:.4}, mean={:.4}, std={:.4}",
                exp_val, min_v, max_v, mean, std
            );
        }
    }
}
