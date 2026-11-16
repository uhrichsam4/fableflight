//! AMD FidelityFX Super Resolution 3 (FSR3) upscaler, for the Graphics
//! setting Upscaler: FSR (render.rs `Renderer::fsr`).
//!
//! Vendored from wgpu-ffx (https://github.com/cwfitzgerald/wgpu-ffx, MIT,
//! commit 6fe7064, see LICENSE here; the shaders are AMD's FidelityFX SDK
//! 1.1.4 GLSL, MIT, compiled to SPIR-V by that project). Changes from it:
//! - wgpu 30, and the shaders always go through naga (SPIR-V in, Metal out):
//!   no passthrough.
//! - Only the one shader permutation this game uses (HDR colour, reversed
//!   infinite depth, render-resolution motion vectors without jitter, the
//!   reprojection's Lanczos from its table (Apple's GPUs run 32-wide, where
//!   AMD picks it), full precision) is kept, in spv/, and patched as it is
//!   loaded so naga and Metal take it (`spirv::patch`).
//! - naga's loop bounding is off for these shaders (it cost a fifth of the
//!   accumulate pass); its bounds checks stay.
//! - The SPD mip chain is Rgba16Float, not Rg16Float: Metal has no
//!   read-write Rg16Float storage.
//! - 32-bit float inputs are bound as non-filterable (no FLOAT32_FILTERABLE).
//! - No debug view, no generate-reactive pass (the renderer makes its own
//!   reactive mask), no buffer-clear compute pass (with reversed depth both
//!   buffers clear to zero, `clear_buffer`), no error scope, and no
//!   bitflags / thiserror / half dependencies.
//! - The compute pass can be timed (--gpu-profile).
//!
//! Needs Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES (R8 / R16Float
//! read-write storage), TEXTURE_FORMAT_16BIT_NORM (the Lanczos table) and
//! CLEAR_TEXTURE, and 8 storage textures per stage (`required`).

mod constants;
mod jitter;
mod lanczos2;
mod pass;
mod rcas;
mod resources;
pub(crate) mod spirv;
mod spd;
mod validation;

use std::mem;

use wgpu::util::DeviceExt as _;

use crate::fsr::{
    constants::FsrConstants,
    pass::ResourceAccess,
    resources::{FsrResourceName, OwnedBindingResource},
};

pub use jitter::get_jitter_phase_count;
pub use validation::FsrDispatchError;

/// What the device must have for FSR, and the limits to ask for.
pub fn required() -> (wgpu::Features, u32) {
    (wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES | wgpu::Features::TEXTURE_FORMAT_16BIT_NORM | wgpu::Features::CLEAR_TEXTURE, 8)
}

/// The SPIR-V of each pass (the one permutation, see the module's notes).
pub(crate) struct Shaders {
    pub accumulate: &'static [u8],
    pub accumulate_sharpen: &'static [u8],
    pub luma_instability: &'static [u8],
    pub luma_pyramid: &'static [u8],
    pub prepare_inputs: &'static [u8],
    pub prepare_reactivity: &'static [u8],
    pub rcas: &'static [u8],
    pub shading_change: &'static [u8],
    pub shading_change_pyramid: &'static [u8],
}

pub(crate) const SHADERS: Shaders = Shaders {
    accumulate: include_bytes!("spv/accumulate_pass.spv"),
    accumulate_sharpen: include_bytes!("spv/accumulate_sharpen_pass.spv"),
    luma_instability: include_bytes!("spv/luma_instability_pass.spv"),
    luma_pyramid: include_bytes!("spv/luma_pyramid_pass.spv"),
    prepare_inputs: include_bytes!("spv/prepare_inputs_pass.spv"),
    prepare_reactivity: include_bytes!("spv/prepare_reactivity_pass.spv"),
    rcas: include_bytes!("spv/rcas_pass.spv"),
    shading_change: include_bytes!("spv/shading_change_pass.spv"),
    shading_change_pyramid: include_bytes!("spv/shading_change_pyramid_pass.spv"),
};

/// The main FSR3 upscaler context: the compiled pipelines.
pub struct FsrContext {
    device: wgpu::Device,

    pass_prepare_inputs: pass::FsrPass,
    pass_prepare_reactivity: pass::FsrPass,
    pass_shading_change: pass::FsrPass,
    pass_accumulate: pass::FsrPass,
    pass_accumulate_sharpen: pass::FsrPass,
    pass_rcas: pass::FsrPass,
    pass_luma_pyramid: pass::FsrPass,
    pass_shading_change_pyramid: pass::FsrPass,
    pass_luma_instability: pass::FsrPass,

    flags: FsrContextFlags,
}

/// Per-resolution state: all internal GPU textures and the temporal history.
pub struct FsrView {
    constants: constants::Constants,
    resources: resources::FsrResources,

    max_render_size: [u32; 2],
    max_upscale_size: [u32; 2],

    first_execution: bool,
    pre_exposure: f32,
    previous_frame_pre_exposure: f32,
    frame_kind: FrameKind,
}

impl FsrContext {
    /// Compiles every pass's pipeline. The flags must be those of the shipped
    /// permutation: HIGH_DYNAMIC_RANGE | DEPTH_INVERTED (| DEPTH_INFINITE).
    pub fn new(device: &wgpu::Device, flags: FsrContextFlags) -> Self {
        assert!(
            flags.contains(FsrContextFlags::HIGH_DYNAMIC_RANGE) && flags.contains(FsrContextFlags::DEPTH_INVERTED)
                && !flags.contains(FsrContextFlags::DISPLAY_RESOLUTION_MOTION_VECTORS) && !flags.contains(FsrContextFlags::MOTION_VECTORS_JITTER_CANCELLATION),
            "only the HDR, reversed-depth, render-resolution unjittered motion vector shaders are vendored"
        );
        let shaders = &SHADERS;
        let p = |kind| pass::FsrPass::new(device, kind, flags, shaders);
        Self {
            device: device.clone(),
            pass_prepare_inputs: p(pass::FsrPassKind::PrepareInputs),
            pass_prepare_reactivity: p(pass::FsrPassKind::PrepareReactivity),
            pass_shading_change: p(pass::FsrPassKind::ShadingChange),
            pass_accumulate: p(pass::FsrPassKind::Accumulate),
            pass_accumulate_sharpen: p(pass::FsrPassKind::AccumulateSharpen),
            pass_rcas: p(pass::FsrPassKind::Rcas),
            pass_luma_pyramid: p(pass::FsrPassKind::LumaPyramid),
            pass_shading_change_pyramid: p(pass::FsrPassKind::ShadingChangePyramid),
            pass_luma_instability: p(pass::FsrPassKind::LumaInstability),
            flags,
        }
    }

    /// Allocate internal GPU resources for a given maximum resolution pair.
    pub fn create_view(&self, queue: &wgpu::Queue, max_render_size: [u32; 2], max_upscale_size: [u32; 2]) -> FsrView {
        FsrView::validate_sizes(max_render_size, max_upscale_size);
        FsrView::new(&self.device, queue, max_render_size, max_upscale_size)
    }

    fn setup_device_depth_to_view_space_depth_params(flags: FsrContextFlags, fsrc: &mut FsrConstants, info: &FsrDispatchInfo) {
        let b_inverted = flags.contains(FsrContextFlags::DEPTH_INVERTED);
        let b_infinite = flags.contains(FsrContextFlags::DEPTH_INFINITE);

        let mut f_min = f32::min(info.camera_near, info.camera_far);
        let mut f_max = f32::max(info.camera_near, info.camera_far);

        if b_inverted {
            mem::swap(&mut f_min, &mut f_max);
        }

        // a 0 0 0   x
        // 0 b 0 0   y
        // 0 0 c d   z
        // 0 0 e 0   1

        let f_q = f_max / (f_min - f_max);
        let d = -1.0f32;

        let matrix_elem_c = [
            [
                f_q,                    // non reversed, non infinite
                -1.0f32 - f32::EPSILON, // non reversed, infinite
            ],
            [
                f_q,                   // reversed, non infinite
                0.0f32 + f32::EPSILON, // reversed, infinite
            ],
        ];

        let matrix_elem_e = [
            [
                f_q * f_min,           // non reversed, non infinite
                -f_min - f32::EPSILON, // non reversed, infinite
            ],
            [
                f_q * f_min, // reversed, non infinite
                f_max,       // reversed, infinite
            ],
        ];

        fsrc.device_to_view_depth[0] = d * matrix_elem_c[b_inverted as usize][b_infinite as usize];
        fsrc.device_to_view_depth[1] = matrix_elem_e[b_inverted as usize][b_infinite as usize];

        let aspect = info.render_size[0] as f32 / info.render_size[1] as f32;
        let cot_half_fov_y = f32::cos(0.5f32 * info.camera_fov_y) / f32::sin(0.5f32 * info.camera_fov_y);
        let a = cot_half_fov_y / aspect;
        let b = cot_half_fov_y;

        fsrc.device_to_view_depth[2] = f32::recip(a);
        fsrc.device_to_view_depth[3] = f32::recip(b);
    }

    /// Record the FSR3 upscaling compute passes into `encoder`: one compute
    /// pass, or with --gpu-profile one per pass, each timed.
    pub fn dispatch(&self, view: &mut FsrView, encoder: &mut wgpu::CommandEncoder, info: &FsrDispatchInfo, prof: Option<&crate::gpuprof::GpuProf>) -> Result<(), FsrDispatchError> {
        validation::check_dispatch(info, self.flags, view.max_render_size, view.max_upscale_size)?;

        let reset_accumulation = info.reset_history || view.first_execution;
        view.first_execution = false;

        let fsrc = &mut view.constants.fsr;

        fsrc.previous_frame_jitter_offset = fsrc.jitter_offset;
        fsrc.previous_frame_upscale_size = fsrc.upscale_size;
        fsrc.previous_frame_render_size = fsrc.render_size;

        fsrc.jitter_offset = info.jitter_offset;
        fsrc.upscale_size = info.upscale_size;
        fsrc.render_size = info.render_size;
        fsrc.downscale_factor = [info.render_size[0] as f32 / info.upscale_size[0] as f32, info.render_size[1] as f32 / info.upscale_size[1] as f32];

        // compute the horizontal FOV for the shader from the vertical one.
        let aspect_ratio = info.render_size[0] as f32 / info.render_size[1] as f32;
        let camera_angle_horizontal = f32::atan(f32::tan(info.camera_fov_y * 0.5) * aspect_ratio) * 2.0;
        fsrc.tan_half_fov = f32::tan(camera_angle_horizontal * 0.5);
        fsrc.view_space_to_meters_factor = info.view_space_to_meters_factor;

        Self::setup_device_depth_to_view_space_depth_params(self.flags, fsrc, info);

        // calculate pre-exposure relevant factors
        view.previous_frame_pre_exposure = view.pre_exposure;
        view.pre_exposure = info.pre_exposure;

        if view.previous_frame_pre_exposure > 0.0 {
            fsrc.delta_pre_exposure = view.pre_exposure / view.previous_frame_pre_exposure;
        } else {
            fsrc.delta_pre_exposure = 1.0;
        }

        // motion vector scale (motion vectors at render resolution)
        fsrc.motion_vector_scale = std::array::from_fn(|i| info.motion_vector_scale[i] / info.render_size[i] as f32);

        let jitter_phase_count = jitter::get_jitter_phase_count(info.render_size[0] as i32, info.upscale_size[0] as i32);

        if reset_accumulation || fsrc.jitter_phase_count == 0.0 {
            fsrc.jitter_phase_count = jitter_phase_count as f32;
        } else {
            let jitter_phase_count_delta = jitter_phase_count - fsrc.jitter_phase_count as i32;
            match jitter_phase_count_delta.cmp(&0) {
                std::cmp::Ordering::Greater => {
                    fsrc.jitter_phase_count += 1.0;
                }
                std::cmp::Ordering::Less => {
                    fsrc.jitter_phase_count -= 1.0;
                }
                std::cmp::Ordering::Equal => {}
            }
        }

        // convert delta time to seconds and clamp to [0, 1].
        fsrc.delta_time = f32::clamp(info.frame_time_delta * 0.001, 0.0, 1.0);

        if reset_accumulation {
            fsrc.frame_index = 0.0;
        } else {
            fsrc.frame_index += 1.0;
        }

        let thread_group_work_region_dim = 8;
        let workgroups_src_x = fsrc.render_size[0].div_ceil(thread_group_work_region_dim);
        let workgroups_src_y = fsrc.render_size[1].div_ceil(thread_group_work_region_dim);
        let workgroups_dst_x = fsrc.upscale_size[0].div_ceil(thread_group_work_region_dim);
        let workgroups_dst_y = fsrc.upscale_size[1].div_ceil(thread_group_work_region_dim);
        let workgroups_shading_change_x = (fsrc.render_size[0] / 2).div_ceil(thread_group_work_region_dim);
        let workgroups_shading_change_y = (fsrc.render_size[1] / 2).div_ceil(thread_group_work_region_dim);
        let spd_thread_group_work_region_dim = 64;
        let workgroups_spd_x = info.render_size[0].div_ceil(spd_thread_group_work_region_dim);
        let workgroups_spd_y = info.render_size[1].div_ceil(spd_thread_group_work_region_dim);

        if reset_accumulation {
            // Clear what this frame reads as the previous accumulation, and the SPD mips.
            for name in [FsrResourceName::AccumulationPrevious, FsrResourceName::SpdMips] {
                let access = ResourceAccess { name, access_type: resources::AccessType::Srv, desc: None };
                let OwnedBindingResource::View(v) = view.resources.to_view(info, access, view.frame_kind) else { unreachable!() };
                encoder.clear_texture(v.texture(), &wgpu::ImageSubresourceRange::default());
            }

            let mut clear_values_exposure = vec![-1.0f32, 1.0, 0.0, 0.0];
            clear_values_exposure.resize(64, 0.0);
            let staging_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("exposure_staging_buffer"),
                contents: bytemuck::cast_slice(&clear_values_exposure),
                usage: wgpu::BufferUsages::COPY_SRC,
            });

            encoder.copy_buffer_to_texture(
                wgpu::TexelCopyBufferInfo { buffer: &staging_buffer, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(256), rows_per_image: Some(1) } },
                wgpu::TexelCopyTextureInfo { texture: &view.resources.frame_info, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            );
        }

        // Reversed depth: the reconstructed previous depth (a max store) and
        // the SPD's atomic counter both start from zero.
        encoder.clear_buffer(&info.reconstructed_previous_depth, 0, None);
        encoder.clear_buffer(&view.resources.spd_atomic_counter, 0, None);

        view.constants.spd = constants::SpdConstants::new(spd::RectInput::new(info.render_size[0], info.render_size[1]));

        let sharpness_remapped = (-2.0 * info.sharpness) + 2.0;
        view.constants.rcas = rcas::populate_rcas_constants(sharpness_remapped);

        let constants_staging_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("FsrContext::constants_staging_buffer"),
            contents: bytemuck::bytes_of(&view.constants),
            usage: wgpu::BufferUsages::COPY_SRC,
        });

        encoder.copy_buffer_to_buffer(&constants_staging_buffer, 0, &view.resources.constant_buffer, 0, mem::size_of::<constants::Constants>() as wgpu::BufferAddress);

        encoder.clear_texture(&view.resources.spd_mips, &wgpu::ImageSubresourceRange::default());

        let passes = [
            (&self.pass_prepare_inputs, workgroups_src_x, workgroups_src_y),
            (&self.pass_luma_pyramid, workgroups_spd_x, workgroups_spd_y),
            (&self.pass_shading_change_pyramid, workgroups_spd_x, workgroups_spd_y),
            (&self.pass_shading_change, workgroups_shading_change_x, workgroups_shading_change_y),
            (&self.pass_prepare_reactivity, workgroups_src_x, workgroups_src_y),
            (&self.pass_luma_instability, workgroups_src_x, workgroups_src_y),
            (if info.enable_sharpening { &self.pass_accumulate_sharpen } else { &self.pass_accumulate }, workgroups_dst_x, workgroups_dst_y),
        ];
        let thread_group_work_region_dim_rcas = 16;
        let rcas = info.enable_sharpening.then_some((&self.pass_rcas, info.upscale_size[0].div_ceil(thread_group_work_region_dim_rcas), info.upscale_size[1].div_ceil(thread_group_work_region_dim_rcas)));
        let all = passes.into_iter().chain(rcas);
        match prof {
            None => {
                let mut cp = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("FSR3"), timestamp_writes: None });
                for (p, x, y) in all {
                    p.dispatch(&self.device, &mut cp, &view.resources, info, self.flags, view.frame_kind, x, y);
                }
            }
            Some(prof) => {
                for (p, x, y) in all {
                    let mut cp = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some(p.label()), timestamp_writes: prof.compute(p.label()) });
                    p.dispatch(&self.device, &mut cp, &view.resources, info, self.flags, view.frame_kind, x, y);
                }
            }
        }

        view.frame_kind.advance();

        Ok(())
    }
}

impl FsrView {
    fn validate_sizes(max_render_size: [u32; 2], max_upscale_size: [u32; 2]) {
        assert!(max_render_size[0] >= 2 && max_render_size[1] >= 2, "max_render_size must be at least [2, 2], got {max_render_size:?}");
        assert!(max_upscale_size[0] >= 1 && max_upscale_size[1] >= 1, "max_upscale_size must be at least [1, 1], got {max_upscale_size:?}");
    }

    fn new(device: &wgpu::Device, queue: &wgpu::Queue, max_render_size: [u32; 2], max_upscale_size: [u32; 2]) -> Self {
        Self {
            constants: constants::Constants {
                fsr: FsrConstants {
                    max_render_size,
                    max_upscale_size,
                    velocity_factor: 1.0,
                    reactiveness_scale: 1.0,
                    shading_change_scale: 1.0,
                    accumulation_added_per_frame: 1.0 / 3.0,
                    min_disocclusion_accumulation: -1.0 / 3.0,
                    ..Default::default()
                },
                ..Default::default()
            },
            resources: resources::FsrResources::new(device, queue, max_render_size, max_upscale_size),
            max_render_size,
            max_upscale_size,
            first_execution: true,
            pre_exposure: 1.0,
            previous_frame_pre_exposure: 1.0,
            frame_kind: FrameKind::Even,
        }
    }

    /// The maximum render resolution this view was allocated for.
    pub fn max_render_size(&self) -> [u32; 2] {
        self.max_render_size
    }

    /// The maximum upscale resolution this view was allocated for.
    pub fn max_upscale_size(&self) -> [u32; 2] {
        self.max_upscale_size
    }
}

/// Configuration options for the FSR context (as the SDK's flags).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FsrContextFlags(u32);

impl FsrContextFlags {
    /// The input colour is high dynamic range.
    pub const HIGH_DYNAMIC_RANGE: Self = Self(1 << 0);
    /// The motion vectors are at display resolution.
    pub const DISPLAY_RESOLUTION_MOTION_VECTORS: Self = Self(1 << 1);
    /// The motion vectors have the jitter in them.
    pub const MOTION_VECTORS_JITTER_CANCELLATION: Self = Self(1 << 2);
    /// The depth is reversed [1..0].
    pub const DEPTH_INVERTED: Self = Self(1 << 3);
    /// The depth has an infinite far plane.
    pub const DEPTH_INFINITE: Self = Self(1 << 4);
    /// Expose the colour automatically.
    pub const AUTO_EXPOSURE: Self = Self(1 << 5);

    pub fn contains(self, o: Self) -> bool {
        self.0 & o.0 == o.0
    }
}

impl std::ops::BitOr for FsrContextFlags {
    type Output = Self;
    fn bitor(self, o: Self) -> Self {
        Self(self.0 | o.0)
    }
}

/// Per-frame dispatch parameters for the FSR3 upscaler.
pub struct FsrDispatchInfo {
    /// The colour for the current frame (render resolution, Rgba16Float).
    pub color: wgpu::Texture,
    /// 32-bit depth for the current frame (render resolution).
    pub depth: wgpu::Texture,
    /// Motion vectors, Rg16Float at render resolution: where each pixel was
    /// last frame, less where it is (scaled by `motion_vector_scale`).
    pub motion_vectors: wgpu::Texture,
    /// An optional 1x1 exposure (R32Float).
    pub exposure: Option<wgpu::Texture>,
    /// An optional reactive mask (R8Unorm, render resolution).
    pub reactive_mask: Option<wgpu::Texture>,
    /// An optional transparency-and-composition mask (R8Unorm, render resolution).
    pub transparency_and_composition: Option<wgpu::Texture>,
    /// R32Float at render resolution, STORAGE_BINDING | TEXTURE_BINDING.
    pub dilated_depth: wgpu::Texture,
    /// Rg16Float at render resolution, STORAGE_BINDING | TEXTURE_BINDING.
    pub dilated_motion_vectors: wgpu::Texture,
    /// render_width * render_height * 4 bytes, STORAGE | COPY_DST.
    pub reconstructed_previous_depth: wgpu::Buffer,
    /// The output (upscale resolution, Rgba16Float, STORAGE_BINDING).
    pub output: wgpu::Texture,

    /// The subpixel jitter offset applied to the camera, in render pixels.
    pub jitter_offset: [f32; 2],
    /// The scale factor to apply to motion vectors.
    pub motion_vector_scale: [f32; 2],

    /// The resolution that was used for rendering the input resources.
    pub render_size: [u32; 2],
    /// The resolution that the upscaler will output.
    pub upscale_size: [u32; 2],

    /// Enable an additional sharpening pass (RCAS).
    pub enable_sharpening: bool,
    /// 0 (no additional sharpness) to 1 (most).
    pub sharpness: f32,
    /// The time elapsed since the last frame, in milliseconds.
    pub frame_time_delta: f32,
    /// The pre exposure value (must be > 0.0f)
    pub pre_exposure: f32,
    /// The camera moved discontinuously: start the history over.
    pub reset_history: bool,
    /// The distance to the near plane of the camera (f32::MAX with reversed infinite depth).
    pub camera_near: f32,
    /// The distance to the far plane of the camera (the near plane's with reversed infinite depth).
    pub camera_far: f32,
    /// The camera's vertical field of view, in radians.
    pub camera_fov_y: f32,
    /// The scale factor to convert view space units to meters
    pub view_space_to_meters_factor: f32,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
enum FrameKind {
    #[default]
    Even,
    Odd,
}

impl FrameKind {
    fn advance(&mut self) {
        *self = match self {
            FrameKind::Even => FrameKind::Odd,
            FrameKind::Odd => FrameKind::Even,
        }
    }
}
