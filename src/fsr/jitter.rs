//! FSR3's jitter: sub-pixel offsets from a Halton sequence (bases 2 and 3),
//! applied to the projection (main.rs makes them).

/// Return the number of jitter phases for the given render and display widths.
///
/// The phase count scales quadratically with the upscale ratio
/// (`display_width / render_width`), starting from a base of 8 at 1:1.
pub fn get_jitter_phase_count(render_width: i32, display_width: i32) -> i32 {
    const BASE_PHASE_COUNT: f32 = 8.0;

    (BASE_PHASE_COUNT * f32::powf(display_width as f32 / render_width as f32, 2.0)) as i32
}
