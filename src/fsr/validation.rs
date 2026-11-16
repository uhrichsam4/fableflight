//! Validation for FSR dispatch parameters.
//!
//! All checks are run before any GPU work is recorded, so a validation
//! failure will never leave the command encoder in a partially-recorded state.

use crate::fsr::{FsrContextFlags, FsrDispatchInfo};

/// Errors that can occur during FSR dispatch validation.
/// (Their fields are read by Debug only, for the log.)
#[allow(dead_code)]
#[derive(Debug)]
pub enum FsrDispatchError {
    ExposureWithAutoExposureFlag,

    JitterOffsetOutOfRange { x: f32, y: f32 },

    MotionVectorScaleTooLarge {
        x: f32,
        y: f32,
        max_width: u32,
        max_height: u32,
    },

    MotionVectorScaleZero { x: f32, y: f32 },

    RenderSizeTooLarge {
        width: u32,
        height: u32,
        max_width: u32,
        max_height: u32,
    },

    RenderSizeZero { width: u32, height: u32 },

    UpscaleSizeTooLarge {
        width: u32,
        height: u32,
        max_width: u32,
        max_height: u32,
    },

    UpscaleSizeZero { width: u32, height: u32 },

    SharpnessOutOfRange(f32),

    FrameTimeDeltaTooLow(f32),

    PreExposureNotPositive(f32),

    InvertedDepthNearLessThanFar { camera_near: f32, camera_far: f32 },

    InvertedInfiniteDepthNearNotMax { camera_near: f32 },

    NormalDepthNearGreaterThanFar { camera_near: f32, camera_far: f32 },

    InfiniteDepthFarNotMax { camera_far: f32 },

    CameraFovTooLow(f32),

    CameraFovTooHigh(f32),
}

impl std::fmt::Display for FsrDispatchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "FSR dispatch: {self:?}")
    }
}

impl std::error::Error for FsrDispatchError {}

/// Validate dispatch parameters for correctness.
///
/// This performs comprehensive validation of all dispatch parameters to ensure they are
/// within expected ranges and consistent with the context configuration.
pub fn check_dispatch(
    info: &FsrDispatchInfo,
    flags: FsrContextFlags,
    max_render_size: [u32; 2],
    max_upscale_size: [u32; 2],
) -> Result<(), FsrDispatchError> {
    // Check exposure configuration
    if info.exposure.is_some() && flags.contains(FsrContextFlags::AUTO_EXPOSURE) {
        return Err(FsrDispatchError::ExposureWithAutoExposureFlag);
    }

    // Check jitter offset range
    if info.jitter_offset[0].abs() > 1.0 || info.jitter_offset[1].abs() > 1.0 {
        return Err(FsrDispatchError::JitterOffsetOutOfRange {
            x: info.jitter_offset[0],
            y: info.jitter_offset[1],
        });
    }

    // Check motion vector scale — display-res MVs are scaled against upscale size
    let mv_max_size = if flags.contains(FsrContextFlags::DISPLAY_RESOLUTION_MOTION_VECTORS) {
        max_upscale_size
    } else {
        max_render_size
    };
    if info.motion_vector_scale[0] > mv_max_size[0] as f32
        || info.motion_vector_scale[1] > mv_max_size[1] as f32
    {
        return Err(FsrDispatchError::MotionVectorScaleTooLarge {
            x: info.motion_vector_scale[0],
            y: info.motion_vector_scale[1],
            max_width: mv_max_size[0],
            max_height: mv_max_size[1],
        });
    }
    if info.motion_vector_scale[0] == 0.0 || info.motion_vector_scale[1] == 0.0 {
        return Err(FsrDispatchError::MotionVectorScaleZero {
            x: info.motion_vector_scale[0],
            y: info.motion_vector_scale[1],
        });
    }

    // Check render size
    if info.render_size[0] > max_render_size[0] || info.render_size[1] > max_render_size[1] {
        return Err(FsrDispatchError::RenderSizeTooLarge {
            width: info.render_size[0],
            height: info.render_size[1],
            max_width: max_render_size[0],
            max_height: max_render_size[1],
        });
    }
    if info.render_size[0] == 0 || info.render_size[1] == 0 {
        return Err(FsrDispatchError::RenderSizeZero {
            width: info.render_size[0],
            height: info.render_size[1],
        });
    }

    // Check upscale size
    if info.upscale_size[0] > max_upscale_size[0] || info.upscale_size[1] > max_upscale_size[1] {
        return Err(FsrDispatchError::UpscaleSizeTooLarge {
            width: info.upscale_size[0],
            height: info.upscale_size[1],
            max_width: max_upscale_size[0],
            max_height: max_upscale_size[1],
        });
    }
    if info.upscale_size[0] == 0 || info.upscale_size[1] == 0 {
        return Err(FsrDispatchError::UpscaleSizeZero {
            width: info.upscale_size[0],
            height: info.upscale_size[1],
        });
    }

    // Check sharpness range
    if info.sharpness < 0.0 || info.sharpness > 1.0 {
        return Err(FsrDispatchError::SharpnessOutOfRange(info.sharpness));
    }

    // Check frame time delta
    if info.frame_time_delta < 1.0 {
        return Err(FsrDispatchError::FrameTimeDeltaTooLow(
            info.frame_time_delta,
        ));
    }

    // Check pre-exposure
    if info.pre_exposure <= 0.0 {
        return Err(FsrDispatchError::PreExposureNotPositive(info.pre_exposure));
    }

    // Check depth configuration
    let infinite_depth = flags.contains(FsrContextFlags::DEPTH_INFINITE);
    let inverted_depth = flags.contains(FsrContextFlags::DEPTH_INVERTED);

    if inverted_depth {
        if info.camera_near < info.camera_far {
            return Err(FsrDispatchError::InvertedDepthNearLessThanFar {
                camera_near: info.camera_near,
                camera_far: info.camera_far,
            });
        }
        if infinite_depth && (info.camera_near != f32::MAX && info.camera_near != f32::INFINITY) {
            return Err(FsrDispatchError::InvertedInfiniteDepthNearNotMax {
                camera_near: info.camera_near,
            });
        }
    } else {
        if info.camera_near > info.camera_far {
            return Err(FsrDispatchError::NormalDepthNearGreaterThanFar {
                camera_near: info.camera_near,
                camera_far: info.camera_far,
            });
        }
        if infinite_depth && (info.camera_far != f32::MAX && info.camera_far != f32::INFINITY) {
            return Err(FsrDispatchError::InfiniteDepthFarNotMax {
                camera_far: info.camera_far,
            });
        }
    }

    // Check camera FOV
    if info.camera_fov_y <= 0.0 {
        return Err(FsrDispatchError::CameraFovTooLow(info.camera_fov_y));
    }
    if info.camera_fov_y > std::f32::consts::PI {
        return Err(FsrDispatchError::CameraFovTooHigh(info.camera_fov_y));
    }

    Ok(())
}
