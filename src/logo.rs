//! The "SAMIAM'S FLIGHT SIM" title over the start screen: no box behind it, drawn
//! as the inverse of whatever colour is under each pixel (see `fs_blit` in
//! post.wgsl), so it reads against space, the planet and its glow alike.
//! This rasterises its coverage mask, one byte per screen pixel.
use ab_glyph::{Font, FontRef, PxScale, ScaleFont, point};

pub const TEXT: &str = "SAMIAM'S FLIGHT SIM";

/// A bold system sans for the title (the UI borrows the system's fonts too).
pub fn find_font() -> Option<Vec<u8>> {
    let fc = std::process::Command::new("fc-match").args(["-f", "%{file}", "sans:weight=bold"]).output().ok().and_then(|o| String::from_utf8(o.stdout).ok());
    let fallbacks = ["/usr/share/fonts/google-noto/NotoSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf", "C:/Windows/Fonts/segoeuib.ttf"];
    fc.iter().map(|s| s.trim().to_string()).chain(fallbacks.iter().map(|s| s.to_string())).filter(|p| p.ends_with(".ttf") || p.ends_with(".otf")).find_map(|p| std::fs::read(p).ok())
}

/// The title's coverage (0..255) on a `w` x `h` screen with UI scale
/// `scale`: centred, near the top, widely letter-spaced.
pub fn mask(font: &[u8], w: u32, h: u32, scale: f32) -> Vec<u8> {
    let mut out = vec![0u8; (w * h) as usize];
    let Ok(font) = FontRef::try_from_slice(font) else { return out };
    // Size the font so its capitals stand `cap` pixels tall.
    // (a PxScale is the font's ascent-to-descent height in pixels)
    let cap_units = font.outline(font.glyph_id('H')).map(|o| (o.bounds.max.y - o.bounds.min.y).abs()).unwrap_or(0.7 * font.height_unscaled());
    let layout = |cap: f32| {
        let px = PxScale::from(cap * font.height_unscaled() / cap_units.max(1.0));
        let sf = font.as_scaled(px);
        let track = 0.2 * cap;
        let mut x = 0.0;
        let mut prev = None;
        let mut glyphs = Vec::new();
        for ch in TEXT.chars() {
            let id = font.glyph_id(ch);
            if let Some(p) = prev {
                x += sf.kern(p, id);
            }
            glyphs.push((id, x));
            x += sf.h_advance(id) + track;
            prev = Some(id);
        }
        (px, glyphs, x - track)
    };
    // No wider than half the screen (the old twelve-letter title's width).
    let mut cap = 27.0 * scale;
    let (_, _, w0) = layout(cap);
    let limit = w as f32 * 0.5;
    if w0 > limit {
        cap *= limit / w0;
    }
    let (px, glyphs, width) = layout(cap);
    let x0 = (w as f32 - width) * 0.5;
    // Capitals sit between `top` and `top + cap`.
    // (above the start screen's panels, which begin some 45 points down)
    let top = 9.0 * scale;
    let base = top + cap;
    for (id, gx) in glyphs {
        let g = id.with_scale_and_position(px, point(x0 + gx, base));
        if let Some(o) = font.outline_glyph(g) {
            let b = o.px_bounds();
            o.draw(|ix, iy, c| {
                let (px, py) = (b.min.x as i32 + ix as i32, b.min.y as i32 + iy as i32);
                if px >= 0 && py >= 0 && (px as u32) < w && (py as u32) < h {
                    let i = (py as u32 * w + px as u32) as usize;
                    out[i] = out[i].max((c.clamp(0.0, 1.0) * 255.0).round() as u8);
                }
            });
        }
    }
    out
}
