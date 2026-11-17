//! wgpu renderer: G-buffer pass, shadow map, deferred sky/water resolve,
//! ambient occlusion (GTAO), volumetric clouds, GPU particles, temporal
//! anti-aliasing, bloom and tonemap.
use bytemuck::{Pod, Zeroable};
use glam::Mat4;
use std::sync::Arc;
use wgpu::util::DeviceExt;

pub const HDR: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
pub const DEPTH: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
/// FSR's motion vectors (`FsrTargets`).
const MOTION: wgpu::TextureFormat = wgpu::TextureFormat::Rg16Float;
/// Distance (km) to the clouds a texel sees (reprojects the cloud history) and to the
/// scene surface it marched to (depth-aware filtering and upsampling).
const CLOUD_DIST: wgpu::TextureFormat = wgpu::TextureFormat::Rg16Float;
/// Share of each pixel's light that is ambient (sky and bounce), and the
/// ambient occlusion that darkens it.
pub(crate) const AMB: wgpu::TextureFormat = wgpu::TextureFormat::R8Unorm;
/// Side of the shadow map's layers (the Graphics settings' Low shadows halve it: `set_shadow_res`).
pub const SHADOW_RES: u32 = 4096;
pub const CASCADES: usize = 3;
pub(crate) const MSAA: u32 = 4;
/// Render scales under this upscale temporally (`Renderer::taau`).
pub const TAAU_BELOW: f32 = 0.7;
use crate::gpuprof::skip as sk;
const MAX_INST: usize = 16384;
const MAX_FALL: usize = 4096;
/// trees and undergrowth close enough to the camera for their leaf cards
pub const MAX_NEAR: usize = 20000;
const BLOOM_MIPS: u32 = 6;
pub const MAX_PARTICLES: u32 = 1 << 16;
/// Smoke volume around the effects anchor: cells along east, north, up; cell
/// size in metres; height of the bottom of the grid above the anchor.
pub const VOL_N: [u32; 3] = [40, 40, 48];
pub const VOL_CELL: f32 = 8.0;
pub const VOL_ZMIN: f32 = -24.0;
const VOL_CELLS: u64 = (VOL_N[0] * VOL_N[1] * VOL_N[2]) as u64;
const UB_ALIGN: u64 = 256;
const MAX_TILES: u64 = 1024;
const MAX_BUILDS: u64 = 512; // detail tiles; one extra slot is kept for loose trees
const MAX_OBJS: u64 = 256;
/// airfields drawn at once, and lights on moving things (aircraft)
pub const MAX_FIELDS: usize = 8;
pub const MAX_DYN_LIGHTS: usize = 8192;
/// Shadow map layer that only the aircraft casts into, a few metres across.
pub const PLANE_LAYER: usize = CASCADES;
/// Resolution of the part of that layer it uses (`Renderer::plane_shadow_res`):
/// all of it, or a quarter of it on an integrated GPU, where every texel the
/// aircraft covers runs its window and gear-well cut-outs (a few mm a texel either way).
const PLANE_SHADOW_RES: u32 = SHADOW_RES;
const PLANE_SHADOW_RES_INTEGRATED: u32 = SHADOW_RES / 2;
/// Photorealistic 3D tiles drawn at once, and the side of their coverage mask
/// (texels; it spans 2 * tiles3d::MASK_HALF metres).
const MAX_PHOTO: u64 = 4096;
const PHOTO_MASK: u32 = 4096;
/// r: the 3D tiles cover this texel; g: height of our water surface (m), or
/// -1e4; b: light from the street lamps on our roads there (night only);
/// a: height of those roads (m), or -1e4
const PHOTO_MASK_FMT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// Slot of the coverage mask's projection among the cascade matrices.
pub const MASK_SLOT: usize = CASCADES + 1;
/// Slot of the shore mask's projection (shore.rs).
pub const SHORE_SLOT: usize = CASCADES + 2;
/// Slot of the shore's wide mask's projection (shore.rs).
pub const SHORE_WIDE_SLOT: usize = CASCADES + 3;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Globals {
    pub view_proj: [[f32; 4]; 4],
    pub inv_view_proj: [[f32; 4]; 4],
    pub shadow_vp: [[[f32; 4]; 4]; CASCADES],
    pub planet: [f32; 4],
    pub sun_dir: [f32; 4],
    pub east: [f32; 4],
    pub north: [f32; 4],
    pub up: [f32; 4],
    pub screen: [f32; 4],
    pub time: [f32; 4],
    pub cloud: [f32; 4],
    pub light_pos: [f32; 4],
    pub light_col: [f32; 4],
    /// explosion volume: corner of the grid relative to camera, w = cell size
    pub vol_o: [f32; 4],
    /// explosion volume up axis, w = 1 while the volume holds live smoke
    pub vol_u: [f32; 4],
    pub misc: [f32; 4],
    pub csm: [f32; 4],
    pub scorch: [[f32; 4]; 8],
    pub crater: [[f32; 4]; 8],
    /// last frame's view-projection (without its sub-pixel offset), relative to last frame's camera
    pub prev_vp: [[f32; 4]; 4],
    /// camera movement since last frame; w = 1 when the cloud history is usable
    pub cam_delta: [f32; 4],
    /// xyz direction to the moon, w its illuminance
    pub moon: [f32; 4],
    /// explosion volume east and north axes
    pub vol_e: [f32; 4],
    pub vol_n: [f32; 4],
    /// the aircraft's own shadow map
    pub pshadow_vp: [[f32; 4]; 4],
    /// texel size (m), depth bias, share of the layer used, w = 1 when in use
    pub pshadow: [f32; 4],
    /// temporal anti-aliasing: xy this frame's sub-pixel offset in uv, z = 1
    /// when it is on, w = 1 when the aircraft is drawn
    pub taa: [f32; 4],
    /// camera-relative position to the aircraft's body frame
    pub plane_inv: [[f32; 4]; 4],
    /// body frame to last frame's clip space
    pub plane_prev: [[f32; 4]; 4],
    /// centre and half size of the aircraft in its body frame
    pub plane_box: [[f32; 4]; 2],
    /// xyz how far the wind has carried the clouds (planet frame, m), w haze (1 = clear day)
    pub weather: [f32; 4],
    /// xyz direction the wind aloft blows towards (planet frame), w number of storm cells
    pub storm: [f32; 4],
    /// storm cells, two each: centre (km, cloud space) and tower radius (km); maturity
    pub storm_cells: [[f32; 4]; 2 * crate::weather::MAX_STORMS],
    /// ground material layers: metres per repeat, metres of relief, scale of
    /// the second sample, photo contrast (0 = procedural); set by the renderer
    pub det: [[f32; 4]; 16],
    /// photorealistic 3D tiles: camera-relative to coverage mask projection
    pub photo_vp: [[f32; 4]; 4],
    /// zone radius (m, 0 = off), dissolve band (m), mask valid
    pub photo: [f32; 4],
    /// zone centre (on the ground) relative to the camera
    pub photo_c: [f32; 4],
    /// the sun the 3D tiles' photographs were taken under (unit), w how sure
    /// that is (0..1), or -1 to light them flat as photographs
    pub photo_sun: [f32; 4],
    /// destroyed buildings cut out of the 3D tiles (tiles3d::PhotoFrame)
    pub cut: [[f32; 4]; 3 * crate::tiles3d::MAX_CUTS],
    pub cut_v: [[f32; 4]; crate::tiles3d::MAX_CUT_V / 2],
    /// cloud shadow map (`CloudShadow`, set by the renderer): the one to look up
    /// (`CloudShadow`, common.wgsl `cloud_sun`): the one lookups read, the one being drawn
    pub cs: [[f32; 4]; 2 * CS_MAP],
    /// mid-level cloud: amount 0..1, height (m)
    pub alto: [f32; 4],
    /// the ocean (ocean.rs `Ocean::globals`): offset of its coordinates and
    /// the wind direction; wind, wave height, whitecap cover; the ripples'
    /// slope covariance; the slope covariance each cascade's mips lose
    pub ocean: [[f32; 4]; crate::ocean::OCEAN_G],
    /// camera-relative to the shore distance field (shore.rs)
    pub shore_vp: [[f32; 4]; 4],
    /// field valid, metres per texel, farthest distance kept (m), side of the square (m)
    pub shore: [f32; 4],
    /// churned white water where a crash went into it, two each: centre
    /// (camera-relative) + radius (m, 0 = none); strength 0..1, age (s), seed
    pub foam: [[f32; 4]; 16],
    /// the start screen: x = 1 while it shows (deep-space sky, no clouds);
    /// set by the renderer: y the cloud quality (0..=2, `Renderer::clouds`),
    /// z = 1 with bloom off, w the shadow filter (common.wgsl `Globals.start`)
    pub start: [f32; 4],
    /// falling snow: xyz the direction the flakes stream from (camera
    /// motion through them), w metres moved through them (wrapped)
    pub snow: [f32; 4],
    /// the boat's wake (boat.rs `Boat::wake_rows`): count, then two rows a
    /// track point, newest first
    pub wake: [[f32; 4]; WAKE_ROWS],
}
/// Rows of `Globals::wake`.
pub const WAKE_ROWS: usize = 1 + 2 * 24;

/// The cloud shadow map: the sun's view of the cloud layer (common.wgsl
/// `cloud_sun`). Each texel is a ray along the sun from a point on the cloud
/// base; it holds the cloud's optical depth in three height bands and how
/// closed the deck is there. Two cascades around the camera, in three slots of
/// a six-layer array: the back map is redrawn a slice of rows per frame, while
/// what lookups read (a separate two-layer texture, csmix.wgsl) cross-fades
/// from the previous map to the front one over the same cycle, so the shadows
/// glide instead of stepping when a map is swapped in. Each map keeps the sun
/// it was drawn for (aimed ahead at where the sun will be while it shows) and
/// the wind drift of the clouds then, which lookups make up for. Map centres
/// snap to whole outer texels of a fixed tangent frame, so one map's texels sit
/// exactly where the last one's did: no crawling, and the cross-fade is a plain
/// texel-by-texel blend.
const CS_RES: u32 = 1024;
const CS_SLICES: u32 = 16;
/// The same on an integrated GPU, where the rays cost a few ms a frame: the
/// map then takes ~2 s to redraw, in which the clouds drift a fraction of a texel.
const CS_SLICES_INTEGRATED: u32 = 64;
const CS_FMT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// half sizes of the two cascades, metres
const CS_INNER: f64 = 40000.0;
const CS_OUTER: f64 = 200000.0;
/// vec4s of Globals.cs per map (common.wgsl `cs_map`)
pub const CS_MAP: usize = 6;
/// how far the mid-level layer reaches above its height (clouds.wgsl ALTO_TOP)
const ALTO_TOP: f32 = 700.0;
type CsMap = [[f32; 4]; CS_MAP];
struct CloudShadow {
    p_in: wgpu::RenderPipeline,
    p_out: wgpu::RenderPipeline,
    layers: Vec<wgpu::TextureView>,
    /// the cross-fade into what lookups read
    p_mix: [wgpu::RenderPipeline; 2],
    bg_mix: wgpu::BindGroup,
    mix_ub: wgpu::Buffer,
    mix_layers: Vec<wgpu::TextureView>,
    prev: CsMap,
    front: CsMap,
    back: CsMap,
    /// slots (pairs of layers) of the previous, front and back maps
    slot: [u32; 3],
    slice: u32,
    /// how many frames a redraw is spread over
    slices: u32,
    ok: bool,
    /// the tangent frame map centres snap in: centre, east, north
    anchor: [glam::DVec3; 3],
    /// last frame's sun, and how far it moves per frame (smoothed)
    sun: glam::DVec3,
    sun_v: glam::DVec3,
    /// this frame's cross-fade: layers, usable, shifts; weight
    mix: ([i32; 8], f32),
}

impl CloudShadow {
    /// Where the map should be now: around the point of the cloud base the sun
    /// shines down through onto the ground below the camera. The sun it is drawn
    /// for is `sun` (unit).
    fn aim(&mut self, g: &Globals, sun: glam::DVec3, reanchor: bool) -> CsMap {
        let v = |a: &[f32; 4]| glam::DVec3::new(a[0] as f64, a[1] as f64, a[2] as f64);
        let up = (-v(&g.planet)).normalize();
        // The bands reach up to the mid-level layer when there is one.
        let (base, mut top) = (g.cloud[1], g.cloud[3].max(g.cloud[1] + 100.0));
        if g.alto[0] > 0.0 {
            top = top.max(g.alto[1] + ALTO_TOP);
        }
        let mu = up.dot(sun);
        // Where the sun's ray from the ground below crosses the cloud base (far off for a low sun).
        let rb = g.planet[3] as f64 + base as f64;
        let gp = up * g.planet[3] as f64;
        let b = gp.dot(sun);
        let t = -b + (b * b - gp.length_squared() + rb * rb).max(0.0).sqrt();
        let c = if mu > -0.05 { (gp + sun * t.min(150000.0)).normalize() } else { up };
        // Snap the centre to whole outer texels (5 inner ones) in the anchor's
        // gnomonic plane; a new anchor once the centre has wandered far from it.
        let [ca, ea, na] = self.anchor;
        let gno = |c: glam::DVec3| glam::DVec2::new(c.dot(ea), c.dot(na)) / c.dot(ca).max(0.5) * rb;
        if reanchor || ca.dot(c) < 0.9 || gno(c).length() > 100000.0 {
            let east = v(&g.east);
            let e = (east - c * east.dot(c)).normalize();
            self.anchor = [c, e, c.cross(e)];
        }
        let [ca, ea, na] = self.anchor;
        let step = 2.0 * CS_OUTER / CS_RES as f64;
        let o = (gno(c) / step).round() * step;
        let f = |a: glam::DVec3, w: f64| [a.x as f32, a.y as f32, a.z as f32, w as f32];
        [f(ca, CS_INNER), f(ea, CS_OUTER), f(na, 0.0), [base, top, o.x as f32, o.y as f32], f(sun, 1.0), [g.weather[0], g.weather[1], g.weather[2], 0.0]]
    }

    /// Fill in this frame's maps; true when the back map is to be drawn whole.
    fn prepare(&mut self, g: &mut Globals) -> bool {
        let v = |a: &[f32; 4]| glam::DVec3::new(a[0] as f64, a[1] as f64, a[2] as f64);
        // The sun's motion per frame (fast when time runs faster); a jump of the
        // clock (or the first frame) is not motion.
        let sun = v(&g.sun_dir).normalize();
        let ds = sun - self.sun;
        let sun_jump = !self.ok || ds.length() > 0.015;
        self.sun_v = if sun_jump { glam::DVec3::ZERO } else { self.sun_v * 0.9 + ds * 0.1 };
        self.sun = sun;
        // Aim each map at the sun of the frame it shows at most: two cycles on,
        // when the lookups have cross-faded fully onto it.
        let ahead = (sun + self.sun_v * (2 * CS_SLICES) as f64).normalize();
        // A jump (new flight, teleport, a new cloud base, a clock set by hand) redraws it all at once.
        let want = self.aim(g, ahead, !self.ok);
        let cen = |m: &CsMap| (v(&m[0]) + (v(&m[1]) * m[3][2] as f64 + v(&m[2]) * m[3][3] as f64) / 6.4e6).normalize();
        let jumped = cen(&self.front).dot(cen(&want)) < (0.3 * CS_INNER / 6.4e6).cos()
            || (self.front[3][0] - want[3][0]).abs() > 50.0
            || (self.front[3][1] - want[3][1]).abs() > 200.0
            || sun_jump;
        let full = !self.ok || jumped;
        if full || self.slice == 0 {
            self.back = want;
        }
        if full {
            // Nothing to fade from: every map is the one drawn now.
            let s = (self.slot[1] + 1) % 3;
            self.slot = [s, s, (s + 1) % 3];
            self.slice = 0;
        }
        let layer = |s: u32| (s * 2) as f32;
        self.back[2][3] = layer(if full { self.slot[0] } else { self.slot[2] });
        let (prev, front) = if full { (self.back, self.back) } else { (self.prev, self.front) };
        // How far the lookups have faded from the previous map to the front one;
        // not at all across a change of tangent frame (texels no longer line up).
        let same = prev[..2] == front[..2] && prev[2][..3] == front[2][..3];
        let a = if full || !same { 1.0 } else { (self.slice + 1) as f32 / CS_SLICES as f32 };
        let lerp = |i: usize| -> [f32; 4] { std::array::from_fn(|j| prev[i][j] + (front[i][j] - prev[i][j]) * a) };
        // Lookups use the front map's grid, with the sun and drift in between.
        let mut look = front;
        let s = v(&lerp(4)).normalize();
        look[4] = [s.x as f32, s.y as f32, s.z as f32, 1.0];
        look[5] = lerp(5);
        g.cs[..CS_MAP].copy_from_slice(&look);
        g.cs[CS_MAP..].copy_from_slice(&self.back);
        // Front-to-previous texel shift of each cascade (whole texels: centres snap to outer ones).
        let sh = |ext: f64, k: usize| ((front[3][2 + k] - prev[3][2 + k]) as f64 * CS_RES as f64 / (2.0 * ext)).round() as i32;
        let lay = |s: u32| (s * 2) as i32;
        let (lp, lf) = if full { (lay(self.slot[0]), lay(self.slot[0])) } else { (lay(self.slot[0]), lay(self.slot[1])) };
        let p: [i32; 8] = [lp, lf, (a < 1.0) as i32, 0, sh(CS_INNER, 0), sh(CS_INNER, 1), sh(CS_OUTER, 0), sh(CS_OUTER, 1)];
        self.mix = (p, a);
        full
    }

    /// The cross-fade's parameters (csmix.wgsl `Mix`) for this frame.
    fn mix_bytes(&self) -> [u8; 48] {
        let mut b = [0u8; 48];
        b[..32].copy_from_slice(bytemuck::cast_slice(&self.mix.0));
        b[32..36].copy_from_slice(&self.mix.1.to_ne_bytes());
        b
    }

    /// Draw (a slice of) the back map, then make it the front once it is complete.
    fn draw(&mut self, enc: &mut wgpu::CommandEncoder, bg0: &wgpu::BindGroup, full: bool) {
        let back_layer = self.back[2][3] as usize;
        let rows = CS_RES / self.slices;
        for (k, p) in [&self.p_in, &self.p_out].into_iter().enumerate() {
            let mut rp = enc.begin_render_pass(&pass_desc(&[Some(color_att(&self.layers[back_layer + k], false))], None));
            rp.set_pipeline(p);
            rp.set_bind_group(0, bg0, &[]);
            if !full {
                rp.set_scissor_rect(0, self.slice * rows, CS_RES, rows);
            }
            rp.draw(0..3, 0..1);
        }
        // What lookups read this frame, from the maps as they were before this
        // frame's slice (the back map's slot is neither of them).
        for (k, p) in self.p_mix.iter().enumerate() {
            let mut rp = enc.begin_render_pass(&pass_desc(&[Some(color_att(&self.mix_layers[k], false))], None));
            rp.set_pipeline(p);
            rp.set_bind_group(0, &self.bg_mix, &[]);
            rp.draw(0..3, 0..1);
        }
        if full {
            self.prev = self.back;
            self.front = self.back;
            self.ok = true;
            return;
        }
        self.slice += 1;
        if self.slice >= self.slices {
            // previous <- front <- back; the slot neither of them holds is drawn next.
            self.prev = self.front;
            self.front = self.back;
            let (p, f) = (self.slot[1], self.slot[2]);
            self.slot = [p, f, (0..3).find(|&s| s != p && s != f).unwrap()];
            self.slice = 0;
        }
    }
}

/// Pipelines, bind groups and targets for drawing the atmosphere's
/// transmittance and multiple-scattering tables.
struct LutBake {
    p_trans: wgpu::RenderPipeline,
    p_ms: wgpu::RenderPipeline,
    bg_trans: wgpu::BindGroup,
    bg_ms: wgpu::BindGroup,
    trans: wgpu::TextureView,
    ms: wgpu::TextureView,
}

impl LutBake {
    /// Uses whatever is in the globals buffer: only the planet radius and the haze matter.
    fn draw(&self, enc: &mut wgpu::CommandEncoder) {
        for (p, bg, target) in [(&self.p_trans, &self.bg_trans, &self.trans), (&self.p_ms, &self.bg_ms, &self.ms)] {
            let mut rp = enc.begin_render_pass(&pass_desc(&[Some(color_att(target, true))], None));
            rp.set_pipeline(p);
            rp.set_bind_group(0, bg, &[]);
            rp.draw(0..3, 0..1);
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Default)]
pub struct Particle {
    pub pos: [f32; 3],
    pub age: f32,
    pub vel: [f32; 3],
    pub life: f32,
    pub size0: f32,
    pub size1: f32,
    pub temp: f32,
    pub kind: f32,
    pub drag: f32,
    pub buoy: f32,
    pub seed: f32,
    pub alpha: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Default)]
pub struct PSim {
    pub anchor: [f32; 4],
    pub upv: [f32; 4],
    pub cam_r: [f32; 4],
    pub cam_u: [f32; 4],
    pub wind: [f32; 4],
    /// volume grid: east axis (w cell size), north axis (w lowest altitude)
    pub ve: [f32; 4],
    pub vn: [f32; 4],
    /// xyz: move every live particle by this much, once (the frame was re-anchored)
    pub shift: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct MeshVertex {
    pub pos: [f32; 3],
    pub nrm: [f32; 3],
    pub col: [f32; 4],
}

/// A vertex of the aircraft (or its wreckage).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Default, Debug)]
pub struct PartVertex {
    /// body-frame position at rest
    pub pos: [f32; 3],
    pub nrm: [f32; 3],
    /// linear albedo, a = baked ambient occlusion
    pub col: [f32; 4],
    /// surface coordinates in metres (seams, rivets, decals) or a local [-1, 1] frame
    pub uv: [f32; 2],
    /// material, surface pattern, exposure to wear, material-specific extra
    pub mat: [f32; 4],
}

/// A vertex of a tree or undergrowth model.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct TreeVertex {
    pub pos: [f32; 3],
    pub nrm: [f32; 3],
    /// rgb; a = 0 bark, 1 foliage, or on a leaf card the number of leaves along it
    pub col: [f32; 4],
    /// leaf cards: along, across, seed, style. Solids: w = kind of bark
    pub uv: [f32; 4],
    /// leaf cards: aspect and two shape parameters. Solids: how far the vertex
    /// draws in once the cards are out. w = openness to the sky
    pub aux: [f32; 4],
}

/// One tree: where, how big, which way it faces, and how far it has fallen.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct TreeInst {
    /// xyz position (relative to its tile centre), w height scale (0 = gone)
    pub a: [f32; 4],
    /// xy fall direction (east, north) times fall angle, z yaw, w width scale
    pub b: [f32; 4],
    /// colour multiplier, 127 = neutral; a = 255, or for undergrowth the distance (m) it is drawn to
    pub tint: [u8; 4],
}

/// One instanced solid: rows of a 3x4 camera-relative model matrix and a tint
/// (tint.a = 1 selects the matte rubble material).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Inst {
    pub m: [[f32; 4]; 3],
    pub tint: [f32; 4],
    /// rubble: a piece of facade's wall coordinates and style (inst.wgsl), else 0
    pub ext: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct RoadVertex {
    pub pos: [f32; 3],
    /// across, along, half width, distance to the nearest end (all metres)
    pub q: [f32; 4],
    pub kind: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct ObjU {
    model: [[f32; 4]; 4],
    tint: [f32; 4],
    glow: [f32; 4],
    params: [[f32; 4]; 6],
    tear: [[f32; 4]; 2],
    dmg: [f32; 4],
    dmg2: [f32; 4],
}

/// An airfield's pavement and lights on the GPU.
pub struct FieldGpu {
    vbuf: wgpu::Buffer,
    ibuf: wgpu::Buffer,
    n_idx: u32,
    lights: Option<(wgpu::Buffer, u32)>,
}

pub struct TileGpu {
    vbuf: wgpu::Buffer,
    bind: wgpu::BindGroup,
}
/// A photorealistic 3D tile (tiles3d.rs) on the GPU.
pub struct PhotoGpu {
    vbuf: wgpu::Buffer,
    ibuf: wgpu::Buffer,
    n_idx: u32,
    bind: wgpu::BindGroup,
    /// its texture, for the pieces of a destroyed building (skin.rs)
    pub view: wgpu::TextureView,
    /// radius about the tile's origin (m), to leave it out of cascades it misses
    rad: f32,
}

/// A vertex of the 3D tiles' mesh carried by a piece of a destroyed
/// building (skin.rs): on the piece's own axes, with the slot of the piece.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Default)]
pub struct SkinVertex {
    pub pos: [f32; 3],
    pub uv: [f32; 2],
    pub nrm: [i8; 4],
    pub slot: u32,
}
/// Slots of the pieces one destroyed building's mesh rides on, at most, and
/// over all buildings drawn at once (64 bytes each, photo.wgsl `Slot`).
pub const SKIN_SLOTS: usize = 16384;
const SKIN_SLOT_BUF: usize = 4 * SKIN_SLOTS;
/// The mesh of a destroyed building from one tile, on the GPU.
pub struct SkinGpu {
    vbuf: wgpu::Buffer,
    bind: wgpu::BindGroup,
    n: u32,
}
pub struct BuildGpu {
    vbuf: wgpu::Buffer,
    pub ibuf: wgpu::Buffer,
    n_idx: u32,
    n_main: u32,
    n_det: u32,
    n_bld: u32,
    n_far: u32,
    n_far_tall: u32,
    cls: [[u32; 4]; 4],
    bind: wgpu::BindGroup,
    road: Option<(wgpu::Buffer, wgpu::Buffer, u32)>,
    pub trees: Option<wgpu::Buffer>,
}
impl BuildGpu {
    /// The geometry buffers, in the order of `building_payload`.
    pub fn targets(&self) -> [Option<&wgpu::Buffer>; 5] {
        [Some(&self.vbuf), Some(&self.ibuf), self.road.as_ref().map(|r| &r.0), self.road.as_ref().map(|r| &r.1), self.trees.as_ref()]
    }
}
/// The bytes each of `BuildGpu::targets` is filled with.
pub fn building_payload(b: &crate::buildings::BuildingData) -> [&[u8]; 5] {
    [bytemuck::cast_slice(&b.verts), bytemuck::cast_slice(&b.indices), bytemuck::cast_slice(&b.road_verts), bytemuck::cast_slice(&b.road_idx), bytemuck::cast_slice(&b.trees)]
}
/// Bytes of a terrain tile's upload, for the per-frame upload budget.
pub fn tile_bytes(t: &crate::sources::TileData) -> usize {
    std::mem::size_of_val(&t.verts[..]) + t.imagery.iter().chain(&t.mask).map(|m| m.len()).sum::<usize>()
}
/// One model drawn for a run of the frame's near instances.
pub struct NearDraw {
    pub mesh: u32,
    pub first: u32,
    pub count: u32,
    /// leaf cards rather than a solid model
    pub cards: bool,
}
pub struct TreeDraw {
    pub mesh: u32,
    pub first: u32,
    pub count: u32,
    pub view: bool,
    /// bit per shadow cascade this batch casts into
    pub casc: u8,
    /// some of its trees may be close enough to have their crowns cut
    pub close: bool,
    /// drawn as impostors (`mesh` is then the model)
    pub imp: bool,
    /// every tree in the batch drawn with `mesh`, whatever its distance (shadows)
    pub fixed: bool,
}
/// Everything drawn for one detail tile this frame.
pub struct DetailDraw<'a> {
    pub gpu: &'a BuildGpu,
    /// tile offset, roof atlas mapping, and how far each level of detail has
    /// faded in: middle, finest, the whole tile
    pub u: [f32; 16],
    pub view: bool,
    /// close enough for roads and the middle level of detail
    pub near: bool,
    /// buildings: 0 all close enough for their full geometry, 2 all far enough
    /// for their stand-ins, 1 some of each (the shader dissolves between them)
    pub far: u8,
    /// the furthest any of it stands from the camera (m)
    pub reach_m: f32,
    /// how many height classes (buildings::HCLASS, tallest first) are close
    /// enough for their full geometry, their middle and their finest level
    pub reach: [u8; 3],
    pub casc: u8,
    pub trees: Vec<TreeDraw>,
    pub close: Vec<NearDraw>,
}
struct MeshLib {
    vbuf: wgpu::Buffer,
    ibuf: wgpu::Buffer,
    ranges: Vec<(u32, u32, i32)>,
}
pub struct MeshGpu {
    vbuf: wgpu::Buffer,
    ibuf: wgpu::Buffer,
    n_idx: u32,
}
pub struct DrawObj {
    pub mesh: usize,
    pub model: Mat4,
    pub tint: [f32; 4],
    pub glow: [f32; 4],
    /// instrument readings and light states (see plane.wgsl)
    pub params: [[f32; 4]; 6],
    /// bit per shadow layer it casts into (bit `PLANE_LAYER` = the aircraft's own)
    pub cast: u8,
    /// drawn in the transparent pass
    pub glass: bool,
    /// has openings cut in it, or a break (needs the discarding pipelines)
    pub cut: bool,
    /// up to two breaks through the body frame: plane normal xyz, offset (see damage.rs)
    pub tear: [[f32; 4]; 2],
    /// mode of each break, scrapes 0..1, raggedness of the edge (m)
    pub dmg: [f32; 4],
    /// oil film on the windshield, crack centre z and y, crack size
    pub dmg2: [f32; 4],
}

impl DrawObj {
    pub const DEFAULT: DrawObj = DrawObj { mesh: 0, model: Mat4::IDENTITY, tint: [0.0; 4], glow: [0.0; 4], params: [[0.0; 4]; 6], cast: 0, glass: false, cut: false, tear: [[0.0; 4]; 2], dmg: [0.0; 4], dmg2: [0.0; 4] };
}

pub struct Frame<'a> {
    pub g: Globals,
    /// terrain tile and its uniform: offset, info, tessellated cell block (see terrain.wgsl)
    pub tiles: Vec<(&'a TileGpu, [f32; 12])>,
    pub details: Vec<DetailDraw<'a>>,
    /// photorealistic 3D tiles: offset from the camera and geometric error,
    /// then their level-of-detail crossfade (photo.wgsl `PhotoTile`)
    pub photo: Vec<(&'a PhotoGpu, [[f32; 4]; 2])>,
    /// destroyed buildings' meshes and where their slots start in `skin_slots`
    pub skins: Vec<(&'a SkinGpu, u32)>,
    /// per slot: rotation rows with the camera-relative offset in w, tint (w = 0: gone)
    pub skin_slots: Vec<[[f32; 4]; 4]>,
    /// airfields and their centres relative to the camera
    pub fields: Vec<(&'a FieldGpu, [f32; 3])>,
    /// lights on moving things, camera-relative
    pub lights: Vec<crate::airfield::Light>,
    pub objs: Vec<DrawObj>,
    /// (mesh, instance) for cars, rubble, rocks
    pub insts: Vec<(u32, Inst)>,
    /// (tree mesh, instance) for trees that have been knocked loose, camera-relative
    pub fall: Vec<(u32, TreeInst)>,
    /// trees (for their leaf cards) and undergrowth near the camera, relative to their tiles
    pub near: Vec<TreeInst>,
    /// shadow cascades, the aircraft's shadow, the 3D tiles' coverage mask
    pub casc_vp: [[[f32; 4]; 4]; CASCADES + 4],
    /// the shore field's part this frame (shore.rs), and for drawing its
    /// mask: the new square's centre (camera-relative) and the terrain tiles
    /// near it that are out of view (their uniforms follow the view tiles')
    pub shore: Option<crate::shore::Step>,
    pub shore_c: [f32; 3],
    pub shore_tiles: Vec<(&'a TileGpu, [f32; 12])>,
    pub psim: PSim,
    pub sim_steps: bool,
    /// the crash's fire simulation, while there is one
    pub fluid: Option<crate::fluid::FluidFrame>,
    /// this frame's ocean spectrum (ocean.rs)
    pub ocean: crate::ocean::OceanU,
    pub egui: Option<(Vec<egui::ClippedPrimitive>, egui::TexturesDelta, f32)>,
    pub shot: Option<std::path::PathBuf>,
    /// the start screen's title, drawn at this UI scale (None: not shown)
    pub logo: Option<f32>,
    /// bit per sun shadow cascade to draw this frame; the others keep what
    /// they hold (drawn on an earlier frame, with the matrix `g.shadow_vp` has for them)
    pub casc_draw: u8,
    /// the camera's vertical field of view (radians) and near plane (m), for FSR
    pub cam: [f32; 2],
}

/// FSR's inputs that the scene does not already have (`Renderer::fsr`), at the scene's resolution.
struct FsrTargets {
    /// where each pixel's surface was last frame, less where it is (taa.wgsl fs_motion)
    motion: wgpu::Texture,
    /// how much the particles and the fire changed each pixel (taa.wgsl fs_reactive)
    reactive: wgpu::Texture,
    /// `hdr` before them, for the reactive mask
    opaque: wgpu::Texture,
    bg_reactive: wgpu::BindGroup,
    /// FSR's own: its dilated depth and motion, and last frame's depth as it reconstructs it
    dilated_depth: wgpu::Texture,
    dilated_motion: wgpu::Texture,
    prev_depth: wgpu::Buffer,
}

/// FSR 3's upscaler (fsr/): its pipelines, made when it is first picked, its
/// history (made for the largest scene size used since, and the window's).
struct Fsr {
    ctx: crate::fsr::FsrContext,
    view: Option<crate::fsr::FsrView>,
    /// 1x1: the exposure the frame will be shown at (FSR judges colours by it)
    exposure: wgpu::Texture,
    /// false: the next frame starts the history over
    hist_ok: bool,
    last: std::time::Instant,
}

/// How much FSR sharpens its result (RCAS, 0..1).
const FSR_SHARPNESS: f32 = 0.8;

struct Targets {
    color_ms: wgpu::TextureView,
    aux_ms: wgpu::TextureView,
    amb_ms: wgpu::TextureView,
    depth_ms: wgpu::TextureView,
    ao_raw: wgpu::TextureView,
    ao: wgpu::TextureView,
    bg_gtao: wgpu::BindGroup,
    bg_ao_blur: wgpu::BindGroup,
    /// (None with MSAA off: the G-buffer's depth is `depth` itself)
    bg_dres: Option<wgpu::BindGroup>,
    color: wgpu::TextureView,
    aux: wgpu::TextureView,
    amb: wgpu::TextureView,
    depth: wgpu::TextureView,
    depth_tex: wgpu::Texture,
    hdr: wgpu::TextureView,
    hdr_tex: wgpu::Texture,
    /// `hdr` as it was before the glass: what the windows refract
    scene_tex: wgpu::Texture,
    bg_glass: wgpu::BindGroup,
    cloud: wgpu::TextureView,
    cloud_dist: wgpu::TextureView,
    cloud_hist: [wgpu::TextureView; 2],
    /// anti-aliased frames: this one and the one before, taking turns
    taa: [wgpu::TextureView; 2],
    /// (with FSR the first is its output, at the window's resolution)
    taa_tex: [wgpu::Texture; 2],
    bg_taa: [wgpu::BindGroup; 2],
    /// (None but with FSR)
    fsr: Option<FsrTargets>,
    bloom: Vec<wgpu::TextureView>,
    ldr_tex: wgpu::Texture,
    /// the start screen's title, one coverage byte per pixel (see logo.rs)
    logo_tex: wgpu::Texture,
    bg_logo: wgpu::BindGroup,
    ldr: wgpu::TextureView,
    bg_composite: wgpu::BindGroup,
    bg_cloud: wgpu::BindGroup,
    bg_cloud_temporal: [wgpu::BindGroup; 2],
    bg_cloud_apply: [wgpu::BindGroup; 2],
    bg_particles: wgpu::BindGroup,
    /// Bloom and the tonemap read the frame from `hdr` (index 0) or from either `taa` texture (1, 2).
    bg_down0: [wgpu::BindGroup; 3],
    bg_down: Vec<wgpu::BindGroup>,
    bg_up: Vec<wgpu::BindGroup>,
    bg_final: [wgpu::BindGroup; 3],
    bg_blit: wgpu::BindGroup,
}

pub struct Renderer {
    ground_mats: GroundMats,
    pub device: wgpu::Device,
    /// --fps-log: seconds spent this period in the game's own update, waiting
    /// for a frame to draw into (the GPU catching up), encoding, and submitting
    pub stats: [f64; 4],
    /// --headless: draw every frame in full but never show it
    pub headless: bool,
    /// The 3D scene's resolution as a fraction of the window's (--render-scale);
    /// the UI is always drawn at the window's.
    scale: f32,
    /// a GPU that shares the CPU's memory and power (Apple silicon, laptop
    /// chips): the defaults trade a little for speed there
    pub integrated: bool,
    /// --gpu-profile: GPU time per pass (gpuprof.rs)
    prof: Option<crate::gpuprof::GpuProf>,
    /// the Graphics setting Upscaler is FSR (`fsr_active`), and whether this
    /// GPU can run it (fsr::required)
    fsr_on: bool,
    fsr_ok: bool,
    fsr: Option<Fsr>,
    p_motion: wgpu::RenderPipeline,
    p_reactive: wgpu::RenderPipeline,
    pub queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    pub config: wgpu::SurfaceConfiguration,
    globals_ub: wgpu::Buffer,
    bg0: wgpu::BindGroup,
    bg0_pre: wgpu::BindGroup,
    /// for the shadow cascades: as `bg0_pre`, with the 3D tiles' coverage mask
    bg0_shadow: wgpu::BindGroup,
    /// `bg0` for another shadow map (`set_shadow_res`)
    bg0_with: Box<dyn Fn(&wgpu::TextureView) -> wgpu::BindGroup>,
    lut_sky: wgpu::TextureView,
    shadow: Vec<wgpu::TextureView>,
    shadow_res: u32,
    /// Graphics settings: cloud quality 0..=2 (2 = half-resolution march, the
    /// others a quarter-resolution, coarser one: clouds.wgsl `cloud_div`),
    /// ambient occlusion, bloom
    clouds: u8,
    pub ao: bool,
    pub bloom: bool,
    /// Low and Medium shadows: only read under FABLE_OPT_OFF=shadow_taps (the
    /// old cheaper filter; otherwise every preset takes the exact one, pcf9)
    pub shadow_fast: bool,
    /// the G-buffer multisampled (MSAA samples) or not, as every preset but
    /// Low (the temporal AA alone smooths edges then): the main view's
    /// pipelines come in both kinds, [0] multisampled and [1] not (`mi`)
    msaa: bool,
    casc_ub: wgpu::Buffer,
    bg_casc: wgpu::BindGroup,
    bg_tileu: wgpu::BindGroup,
    l_dres: wgpu::BindGroupLayout,
    l_ao: wgpu::BindGroupLayout,
    p_gtao: wgpu::RenderPipeline,
    p_ao_blur: wgpu::RenderPipeline,
    p_dres: wgpu::RenderPipeline,
    p_road: [wgpu::RenderPipeline; 2],
    p_field: [wgpu::RenderPipeline; 2],
    p_lights: wgpu::RenderPipeline,
    dyn_lights: wgpu::Buffer,
    /// loose trees; `p_tree_near` draws the rest short of IMP_DIST (with the
    /// outlines of their crowns cut into clumps), `imp` the ones beyond
    /// [with the 3D tiles zone's discard, without (common.wgsl PHOTO)][MSAA, not]
    p_tree: [[wgpu::RenderPipeline; 2]; 2],
    p_tree_near: [wgpu::RenderPipeline; 2],
    imp: crate::impostor::Impostors,
    p_tree_shadow: wgpu::RenderPipeline,
    p_tree_shadow_fixed: wgpu::RenderPipeline,
    p_card: [wgpu::RenderPipeline; 2],
    near_buf: wgpu::Buffer,
    /// [with the 3D tiles zone's discard, without (common.wgsl PHOTO)][MSAA, not]
    p_inst: [[wgpu::RenderPipeline; 2]; 2],
    p_inst_shadow: wgpu::RenderPipeline,
    inst_buf: wgpu::Buffer,
    fall_buf: wgpu::Buffer,
    tree_lib: Option<MeshLib>,
    inst_lib: Option<MeshLib>,
    l_tile: wgpu::BindGroupLayout,
    l_build: wgpu::BindGroupLayout,
    l_composite: wgpu::BindGroupLayout,
    /// the composite's inputs without their multisampled versions (MSAA off)
    l_composite1: wgpu::BindGroupLayout,
    l_cloud: wgpu::BindGroupLayout,
    l_cloud_apply: wgpu::BindGroupLayout,
    l_cloud_temporal: wgpu::BindGroupLayout,
    l_post: wgpu::BindGroupLayout,
    l_pdraw: wgpu::BindGroupLayout,
    /// the glass pass: a copy of the lit scene behind it (to refract) and the depth
    l_glass: wgpu::BindGroupLayout,
    tile_ub: wgpu::Buffer,
    l_photo: wgpu::BindGroupLayout,
    photo_ub: wgpu::Buffer,
    p_photo: [wgpu::RenderPipeline; 2],
    p_photo_mask: wgpu::RenderPipeline,
    p_photo_shadow: wgpu::RenderPipeline,
    p_photo_shadow_fast: wgpu::RenderPipeline,
    l_skin: wgpu::BindGroupLayout,
    p_skin: [wgpu::RenderPipeline; 2],
    skin_slots: wgpu::Buffer,
    p_water_top: wgpu::RenderPipeline,
    /// where any ground is, into the shore mask (shore.rs)
    p_shore_cover: wgpu::RenderPipeline,
    p_lamp_top: wgpu::RenderPipeline,
    photo_mask: wgpu::TextureView,
    /// the mask holds something that has to be cleared when no tiles are drawn
    photo_mask_dirty: bool,
    build_ub: wgpu::Buffer,
    obj_ub: wgpu::Buffer,
    bg_obj: wgpu::BindGroup,
    tile_ibuf: wgpu::Buffer,
    tile_nidx: u32,
    p_sky: wgpu::RenderPipeline,
    /// [with the 3D tiles zone's discard, without (common.wgsl PHOTO)][MSAA, not]
    p_terrain: [[wgpu::RenderPipeline; 2]; 2],
    p_terrain_tess: [[wgpu::RenderPipeline; 2]; 2],
    /// a tile's water as its own fine grid (Water High), and the grids' index
    /// buffers by `WATER_GRIDS`
    p_water: [wgpu::RenderPipeline; 2],
    water_ibuf: [(wgpu::Buffer, u32); 2],
    p_terrain_cells: [[wgpu::RenderPipeline; 2]; 2],
    /// index buffers of a tessellated cell patch and of a single cell quad
    tess_ibuf: wgpu::Buffer,
    tess_nidx: u32,
    cell_ibuf: wgpu::Buffer,
    p_build: [wgpu::RenderPipeline; 2],
    /// p_build without the dissolves' discards (buildings.wgsl DISSOLVE)
    p_build_solid: [wgpu::RenderPipeline; 2],
    p_build_shadow: wgpu::RenderPipeline,
    p_obj: [wgpu::RenderPipeline; 2],
    p_obj_shadow: wgpu::RenderPipeline,
    p_obj_solid: [wgpu::RenderPipeline; 2],
    p_obj_shadow_solid: wgpu::RenderPipeline,
    p_glass: wgpu::RenderPipeline,
    p_composite: [wgpu::RenderPipeline; 2],
    /// The composite split in two (FABLE_OPT_OFF=comp_split for one pass):
    /// land and sky with the water shading compiled out, then the water over
    /// it (composite.wgsl WATER). Per MSAA as p_composite.
    p_comp_land: [wgpu::RenderPipeline; 2],
    p_comp_water: [wgpu::RenderPipeline; 2],
    /// and with MSAA, the pixels that mix water and not, sample by sample
    p_comp_edge: wgpu::RenderPipeline,
    p_cloud: wgpu::RenderPipeline,
    p_cloud_apply: wgpu::RenderPipeline,
    p_cloud_temporal: wgpu::RenderPipeline,
    p_particles: wgpu::RenderPipeline,
    p_taa: wgpu::RenderPipeline,
    /// temporal upscaling (`taau`): the history at the window's resolution
    p_taa_up: wgpu::RenderPipeline,
    p_final_taau: wgpu::RenderPipeline,
    p_psim: wgpu::ComputePipeline,
    p_vol: [wgpu::ComputePipeline; 5],
    bg_vol_a: wgpu::BindGroup,
    bg_vol_b: wgpu::BindGroup,
    vol_acc: wgpu::Buffer,
    pub fluid: crate::fluid::FluidGpu,
    ocean: crate::ocean::OceanGpu,
    shore: crate::shore::ShoreGpu,
    fluid_t: crate::fluid::FluidTargets,
    /// game time (s) until which some smoke may still be alive, and the time now
    smoke_until: f32,
    /// until when (game time) any particle may still be alive: past it the
    /// whole ring is dead, and neither simulated nor drawn
    parts_until: f32,
    now: f32,
    p_down: wgpu::RenderPipeline,
    p_up: wgpu::RenderPipeline,
    p_final: wgpu::RenderPipeline,
    p_final_taa: wgpu::RenderPipeline,
    p_final_taa_up: wgpu::RenderPipeline,
    p_blit: wgpu::RenderPipeline,
    /// the start screen's title, inverting what is under it
    p_logo: wgpu::RenderPipeline,
    psim_ub: wgpu::Buffer,
    pbuf: wgpu::Buffer,
    bg_psim: wgpu::BindGroup,
    phead: u32,
    black: wgpu::TextureView,
    targets: Targets,
    /// which cloud history texture this frame writes; false until it holds a valid frame
    cloud_flip: usize,
    cloud_hist_ok: bool,
    cshadow: CloudShadow,
    /// what the transmittance and multi-scatter tables need to be redrawn
    lut_bake: LutBake,
    /// the haze the tables were last drawn with
    lut_haze: f32,
    /// the same for the anti-aliased frame history
    taa_flip: usize,
    taa_hist_ok: bool,
    pub meshes: Vec<MeshGpu>,
    pub egui: egui_wgpu::Renderer,
    /// the font of the start screen's title, and the screen size and UI
    /// scale its mask was last drawn for
    pub logo_font: Option<Vec<u8>>,
    logo_key: Option<(u32, u32, u32)>,
}

pub(crate) fn tex_entry(binding: u32, dim: wgpu::TextureViewDimension, depth: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: ALL_STAGES,
        ty: wgpu::BindingType::Texture {
            sample_type: if depth { wgpu::TextureSampleType::Depth } else { wgpu::TextureSampleType::Float { filterable: true } },
            view_dimension: dim,
            multisampled: false,
        },
        count: None,
    }
}
/// Every stage the game has shaders for. Not `ShaderStages::all()`: that also
/// names the task and mesh stages, and wgpu 30's Metal backend then hands the
/// fragment and compute stages the wrong buffers and textures (it stores a
/// group's resources in one stage order and looks them up in another). The
/// particle step got the 128-byte uniform block in place of the particle
/// buffer and read and wrote megabytes past its end: GPU faults, blank frames.
const ALL_STAGES: wgpu::ShaderStages = wgpu::ShaderStages::VERTEX.union(wgpu::ShaderStages::FRAGMENT).union(wgpu::ShaderStages::COMPUTE);
pub(crate) fn ub_entry(binding: u32, dynamic: bool, size: u64) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: ALL_STAGES,
        ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: dynamic, min_binding_size: wgpu::BufferSize::new(size) },
        count: None,
    }
}
pub(crate) fn samp_entry(binding: u32, ty: wgpu::SamplerBindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry { binding, visibility: ALL_STAGES, ty: wgpu::BindingType::Sampler(ty), count: None }
}
pub(crate) fn tex_res(binding: u32, v: &wgpu::TextureView) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry { binding, resource: wgpu::BindingResource::TextureView(v) }
}
pub(crate) fn buf_res(binding: u32, b: &wgpu::Buffer, size: u64) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry { binding, resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: b, offset: 0, size: wgpu::BufferSize::new(size) }) }
}

pub(crate) fn make_tex(device: &wgpu::Device, label: &str, w: u32, h: u32, d: u32, mips: u32, format: wgpu::TextureFormat, usage: wgpu::TextureUsages) -> wgpu::Texture {
    make_tex_ms(device, label, w, h, d, mips, format, usage, 1)
}
#[allow(clippy::too_many_arguments)]
fn make_tex_ms(device: &wgpu::Device, label: &str, w: u32, h: u32, d: u32, mips: u32, format: wgpu::TextureFormat, usage: wgpu::TextureUsages, samples: u32) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d { width: w.max(1), height: h.max(1), depth_or_array_layers: d },
        mip_level_count: mips,
        sample_count: samples,
        dimension: if d > 1 { wgpu::TextureDimension::D3 } else { wgpu::TextureDimension::D2 },
        format,
        usage,
        view_formats: &[],
    })
}
pub(crate) fn view(t: &wgpu::Texture) -> wgpu::TextureView {
    t.create_view(&wgpu::TextureViewDescriptor::default())
}

/// Upload a full mip chain for a square texture.
fn upload_mips(device: &wgpu::Device, queue: &wgpu::Queue, label: &str, size: u32, format: wgpu::TextureFormat, bpp: u32, mips: &[Vec<u8>]) -> wgpu::TextureView {
    let t = make_tex(device, label, size, size, 1, mips.len() as u32, format, wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST);
    for (i, m) in mips.iter().enumerate() {
        let s = (size >> i).max(1);
        queue.write_texture(
            wgpu::TexelCopyTextureInfo { texture: &t, mip_level: i as u32, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            m,
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(s * bpp), rows_per_image: Some(s) },
            wgpu::Extent3d { width: s, height: s, depth_or_array_layers: 1 },
        );
    }
    view(&t)
}

/// The Hornet's paint (assets/fa18_paint.jpg, baked by assets/blender/fa18_paint.py):
/// three pictures side by side (the aircraft from above, from below and from
/// the left), each a layer of the array with its own mip chain. plane.wgsl
/// projects them onto the skin (`hornet_paint`).
fn paint_texture(device: &wgpu::Device, queue: &wgpu::Queue) -> wgpu::TextureView {
    let img = image::load_from_memory_with_format(include_bytes!("../assets/fa18_paint.jpg"), image::ImageFormat::Jpeg).expect("the Hornet's paint").to_rgba8();
    let (w, h) = (img.width() / 3, img.height());
    let mips = 32 - w.max(h).leading_zeros();
    let t = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("hornet paint"),
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 3 },
        mip_level_count: mips,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    for layer in 0..3u32 {
        let mut cur: Vec<u8> = Vec::with_capacity((w * h * 4) as usize);
        for y in 0..h {
            let row = ((y * img.width() + layer * w) * 4) as usize;
            cur.extend_from_slice(&img.as_raw()[row..row + (w * 4) as usize]);
        }
        let (mut cw, mut ch) = (w, h);
        for level in 0..mips {
            queue.write_texture(
                wgpu::TexelCopyTextureInfo { texture: &t, mip_level: level, origin: wgpu::Origin3d { x: 0, y: 0, z: layer }, aspect: wgpu::TextureAspect::All },
                &cur,
                wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(cw * 4), rows_per_image: Some(ch) },
                wgpu::Extent3d { width: cw, height: ch, depth_or_array_layers: 1 },
            );
            // The next level: each texel the mean of the four under it.
            let (nw, nh) = ((cw / 2).max(1), (ch / 2).max(1));
            let mut next = vec![0u8; (nw * nh * 4) as usize];
            for y in 0..nh {
                for x in 0..nw {
                    for c in 0..4 {
                        let s = |dx: u32, dy: u32| cur[(((y * 2 + dy).min(ch - 1) * cw + (x * 2 + dx).min(cw - 1)) * 4 + c) as usize] as u32;
                        next[((y * nw + x) * 4 + c) as usize] = ((s(0, 0) + s(1, 0) + s(0, 1) + s(1, 1) + 2) / 4) as u8;
                    }
                }
            }
            (cur, cw, ch) = (next, nw, nh);
        }
    }
    t.create_view(&wgpu::TextureViewDescriptor { dimension: Some(wgpu::TextureViewDimension::D2Array), ..Default::default() })
}

/// Small allocator blocks (host-visible ones are half these sizes). Every
/// upload goes through staging memory that wgpu takes from the 246 MiB
/// CPU-visible VRAM window (BAR1) on cards without resizable BAR. With the
/// default 64-128 MiB blocks a burst of tile uploads grew the staging pool
/// until BAR1 was nearly full; the NVIDIA driver then maps the new block
/// through a slow path (~9 MB/s), and as the allocator keeps its last block
/// for good, every later upload crawled: ~1 fps over cities. Small blocks
/// keep BAR1 use close to what is in flight (see also world::UPLOAD_BUDGET).
const MEMORY_HINTS: wgpu::MemoryHints = wgpu::MemoryHints::Manual { suballocated_device_memory_block_size: (32 << 20)..(64 << 20) };

/// Vertices per side of the fine water grids (terrain.wgsl `vs_water`): a
/// tile's uniform asks for one of these in `info.y` (world.rs `draw_lists`).
pub const WATER_GRIDS: [u32; 2] = [128, 256];
/// Triangles of a water grid of n x n quads (split along the same diagonal as
/// the terrain's cells), then the skirt around its border: the 4n border
/// vertices (bottom, right, top, left) repeated after the (n + 1)^2 grid.
fn water_indices(n: u32) -> Vec<u32> {
    let row = n + 1;
    let mut v = Vec::with_capacity((n * n * 6 + n * 24) as usize);
    for b in 0..n {
        for a in 0..n {
            let (p00, p10, p01, p11) = (b * row + a, b * row + a + 1, (b + 1) * row + a, (b + 1) * row + a + 1);
            v.extend([p00, p01, p10, p10, p01, p11]);
        }
    }
    let border = |r: u32| {
        let (side, o) = ((r / n) % 4, r % n);
        let (i, j) = match side {
            0 => (o, 0),
            1 => (n, o),
            2 => (n - o, n),
            _ => (0, n - o),
        };
        j * row + i
    };
    for r in 0..4 * n {
        let r1 = (r + 1) % (4 * n);
        let (a, b, sa, sb) = (border(r), border(r1), row * row + r, row * row + r1);
        v.extend([a, b, sa, sa, b, sb]);
    }
    v
}

/// Sub-cells per side of a tessellated terrain cell (TN in terrain.wgsl).
const TESS_N: u16 = 16;
/// Size and layer count of the ground material arrays (see detail.wgsl).
const DET_N: u32 = crate::materials::N;
const DET_LAYERS: u32 = 13;
/// Per-layer metres per repeat, metres of relief, scale of the second sample,
/// photo contrast (0 = procedural layer): G.det, see common.wgsl. Must match
/// SIZE / RANGE in detail.wgsl for the procedural layers. Layers 8 and up
/// (mid-scale scans) are blank until their scan arrives.
const DET_PROC: [[f32; 4]; 16] = [
    [1.0, 0.05, 1.0, 0.0],
    [2.0, 0.06, 4.0, 0.0],
    [4.0, 0.45, 4.0, 0.0],
    [2.0, 0.03, 3.0, 0.0],
    [2.0, 0.006, 1.0, 0.0],
    [2.0, 0.035, 1.0, 0.0],
    [1.0, 0.08, 1.0, 0.0],
    [2.0, 0.05, 1.0, 0.0],
    [20.0, 0.0, 1.0, 0.0],
    [20.0, 0.0, 1.0, 0.0],
    [20.0, 0.0, 1.0, 0.0],
    [20.0, 0.0, 1.0, 0.0],
    [20.0, 0.0, 1.0, 0.0],
    [20.0, 0.0, 1.0, 0.0],
    [20.0, 0.0, 1.0, 0.0],
    [20.0, 0.0, 1.0, 0.0],
];

/// The ground material library: albedo + height and normal + cavity arrays
/// with full mip chains. Generated on the GPU at start-up (detail.wgsl), then
/// layers are replaced by photo scans as they finish loading (materials.rs).
struct GroundMats {
    alb: wgpu::Texture,
    nrm: wgpu::Texture,
    p_down: wgpu::ComputePipeline,
    params: [[f32; 4]; 16],
    photos: Option<crossbeam_channel::Receiver<crate::materials::Layer>>,
}

fn det_level(t: &wgpu::Texture, m: u32) -> wgpu::TextureView {
    t.create_view(&wgpu::TextureViewDescriptor { dimension: Some(wgpu::TextureViewDimension::D2Array), base_mip_level: m, mip_level_count: Some(1), ..Default::default() })
}

fn det_run(d: &wgpu::Device, enc: &mut wgpu::CommandEncoder, p: &wgpu::ComputePipeline, entries: &[wgpu::BindGroupEntry], size: u32) {
    let bg = d.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &p.get_bind_group_layout(0), entries });
    let mut cp = enc.begin_compute_pass(&Default::default());
    cp.set_pipeline(p);
    cp.set_bind_group(0, &bg, &[]);
    cp.dispatch_workgroups(size.div_ceil(8), size.div_ceil(8), DET_LAYERS);
}

impl GroundMats {
    fn new(d: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let mips = DET_N.ilog2() + 1;
        let mk = |label: &str| {
            d.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d { width: DET_N, height: DET_N, depth_or_array_layers: DET_LAYERS },
                mip_level_count: mips,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            })
        };
        let (alb, nrm) = (mk("ground albedo"), mk("ground normal"));
        let m = d.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("detail"), source: wgpu::ShaderSource::Wgsl(include_str!("shaders/detail.wgsl").into()) });
        let pipe = |entry: &str| d.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some(entry), layout: None, module: &m, entry_point: Some(entry), compilation_options: Default::default(), cache: None });
        let (p_gen, p_nrm, p_down) = (pipe("gen"), pipe("nrm"), pipe("down"));
        let mut enc = d.create_command_encoder(&Default::default());
        let (a0, n0) = (det_level(&alb, 0), det_level(&nrm, 0));
        det_run(d, &mut enc, &p_gen, &[tex_res(0, &a0)], DET_N);
        det_run(d, &mut enc, &p_nrm, &[tex_res(1, &a0), tex_res(2, &n0)], DET_N);
        let g = Self { alb, nrm, p_down, params: DET_PROC, photos: None };
        g.mips(d, &mut enc);
        queue.submit([enc.finish()]);
        g
    }

    /// Rebuild every mip level from level 0 (both arrays, all layers).
    fn mips(&self, d: &wgpu::Device, enc: &mut wgpu::CommandEncoder) {
        for t in [&self.alb, &self.nrm] {
            for i in 1..t.mip_level_count() {
                let (src, dst) = (det_level(t, i - 1), det_level(t, i));
                det_run(d, enc, &self.p_down, &[tex_res(3, &src), tex_res(4, &dst)], (DET_N >> i).max(1));
            }
        }
    }

    /// Swap in any photo layers that have finished loading.
    fn poll(&mut self, d: &wgpu::Device, queue: &wgpu::Queue) {
        let Some(rx) = &self.photos else { return };
        let mut any = false;
        while let Ok(l) = rx.try_recv() {
            for (t, data) in [(&self.alb, &l.alb), (&self.nrm, &l.nrm)] {
                queue.write_texture(
                    wgpu::TexelCopyTextureInfo { texture: t, mip_level: 0, origin: wgpu::Origin3d { x: 0, y: 0, z: l.layer }, aspect: wgpu::TextureAspect::All },
                    data,
                    wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(DET_N * 4), rows_per_image: Some(DET_N) },
                    wgpu::Extent3d { width: DET_N, height: DET_N, depth_or_array_layers: 1 },
                );
            }
            self.params[l.layer as usize] = l.params;
            any = true;
        }
        if any {
            let mut enc = d.create_command_encoder(&Default::default());
            self.mips(d, &mut enc);
            queue.submit([enc.finish()]);
        }
    }

    fn views(&self) -> (wgpu::TextureView, wgpu::TextureView) {
        let all = |t: &wgpu::Texture| t.create_view(&wgpu::TextureViewDescriptor { dimension: Some(wgpu::TextureViewDimension::D2Array), ..Default::default() });
        (all(&self.alb), all(&self.nrm))
    }
}

/// Tiling wave texture: RG slope, B height (caustics), A an independent noise
/// channel. The water's waves themselves come from the FFT ocean (ocean.rs).
fn wave_texture() -> Vec<u8> {
    const N: usize = 256;
    let mut rng = 0x9e3779b97f4a7c15u64;
    let mut rnd = move || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        (rng >> 11) as f32 / (1u64 << 53) as f32
    };
    let field = |waves: &[(f32, f32, f32, f32)]| -> Vec<[f32; 3]> {
        let mut f = vec![[0f32; 3]; N * N];
        for y in 0..N {
            for x in 0..N {
                let (u, v) = (x as f32 / N as f32 * std::f32::consts::TAU, y as f32 / N as f32 * std::f32::consts::TAU);
                let mut o = [0f32; 3];
                for (kx, ky, a, ph) in waves {
                    let th = kx * u + ky * v + ph;
                    o[0] += a * th.sin();
                    o[1] += a * kx * th.cos();
                    o[2] += a * ky * th.cos();
                }
                f[y * N + x] = o;
            }
        }
        f
    };
    let mut wgen = |kmin: f32, kmax: f32, pow: f32, n: usize| -> Vec<(f32, f32, f32, f32)> {
        (0..n)
            .map(|_| {
                let k = kmin + (kmax - kmin) * rnd().powf(1.6);
                let a = rnd() * std::f32::consts::TAU;
                let (kx, ky) = ((k * a.cos()).round(), (k * a.sin()).round());
                let kk = (kx * kx + ky * ky).sqrt().max(1.0);
                (kx, ky, kk.powf(-pow), rnd() * std::f32::consts::TAU)
            })
            .collect()
    };
    let w1 = wgen(3.0, 26.0, 1.55, 64);
    let w2 = wgen(1.0, 18.0, 1.2, 48);
    let (f1, f2) = (field(&w1), field(&w2));
    let norm = |f: &[[f32; 3]], c: usize| f.iter().fold(0f32, |m, v| m.max(v[c].abs())).max(1e-6);
    let (hs, sx, sy, ns) = (norm(&f1, 0), norm(&f1, 1), norm(&f1, 2), norm(&f2, 0));
    let s = sx.max(sy);
    let mut out = vec![0u8; N * N * 4];
    for i in 0..N * N {
        out[i * 4] = ((f1[i][1] / s * 0.5 + 0.5) * 255.0) as u8;
        out[i * 4 + 1] = ((f1[i][2] / s * 0.5 + 0.5) * 255.0) as u8;
        out[i * 4 + 2] = ((f1[i][0] / hs * 0.5 + 0.5) * 255.0) as u8;
        out[i * 4 + 3] = ((f2[i][0] / ns * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0) as u8;
    }
    out
}

impl Renderer {
    pub async fn new(window: Arc<winit::window::Window>, vsync: bool) -> anyhow::Result<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let surface = instance.create_surface(window.clone())?;
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, force_fallback_adapter: false, compatible_surface: Some(&surface), ..Default::default() })
            .await?;
        let info = adapter.get_info();
        log::info!("GPU: {} ({:?})", info.name, info.device_type);
        let integrated = info.device_type == wgpu::DeviceType::IntegratedGpu;
        let profile = std::env::args().any(|a| a == "--gpu-profile") && adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY);
        if std::env::args().any(|a| a == "--gpu-profile") && !profile {
            log::warn!("--gpu-profile: this GPU has no timestamp queries");
        }
        // (FSR, the Graphics setting Upscaler, where the GPU has what it needs)
        let (fsr_feat, fsr_storage) = crate::fsr::required();
        let fsr_ok = adapter.features().contains(fsr_feat) && adapter.limits().max_storage_textures_per_shader_stage >= fsr_storage;
        if !fsr_ok {
            log::info!("FSR: not on this GPU (TAAU instead)");
        }
        let features = if profile { wgpu::Features::TIMESTAMP_QUERY } else { wgpu::Features::empty() } | if fsr_ok { fsr_feat } else { wgpu::Features::empty() };
        let limits = wgpu::Limits {
            max_texture_dimension_2d: 8192,
            max_sampled_textures_per_shader_stage: adapter.limits().max_sampled_textures_per_shader_stage.clamp(16, 32),
            max_storage_textures_per_shader_stage: if fsr_ok { fsr_storage } else { wgpu::Limits::default().max_storage_textures_per_shader_stage },
            ..Default::default()
        };
        let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor { label: None, required_features: features, required_limits: limits, memory_hints: MEMORY_HINTS, ..Default::default() }).await?;
        let size = window.inner_size();
        let mut config = surface.get_default_config(&adapter, size.width.max(1), size.height.max(1)).ok_or_else(|| anyhow::anyhow!("surface unsupported"))?;
        let caps = surface.get_capabilities(&adapter);
        config.format = caps.formats.iter().copied().find(|f| !f.is_srgb()).unwrap_or(caps.formats[0]);
        config.present_mode = if vsync { wgpu::PresentMode::AutoVsync } else { wgpu::PresentMode::AutoNoVsync };
        surface.configure(&device, &config);
        log::info!("surface {}x{}", config.width, config.height);

        let d = &device;
        let common = include_str!("shaders/common.wgsl");
        let module = |name: &str, src: &str| d.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some(name), source: wgpu::ShaderSource::Wgsl(format!("{common}\n{src}").into()) });
        let m_lut = module("lut", include_str!("shaders/lut.wgsl"));
        let m_terrain = module("terrain", include_str!("shaders/terrain.wgsl"));
        let m_photo = module("photo", include_str!("shaders/photo.wgsl"));
        let facade = include_str!("shaders/facade.wgsl");
        let m_build = module("buildings", &format!("{facade}\n{}", include_str!("shaders/buildings.wgsl")));
        let m_obj = module("plane", &format!("{}\n{}\n{}\n{}", include_str!("shaders/plane.wgsl"), crate::aircraft::b747::deck::strings_wgsl(), include_str!("shaders/deck747.wgsl"), include_str!("shaders/pit18.wgsl")));
        let sky = include_str!("shaders/sky.wgsl");
        let m_comp = module("composite", &format!("{sky}\n{}", include_str!("shaders/composite.wgsl")));
        let m_cloud = module("clouds", &format!("{sky}\n{}", include_str!("shaders/clouds.wgsl")));
        let m_post = module("post", include_str!("shaders/post.wgsl"));
        let m_psim = module("psim", include_str!("shaders/particles_sim.wgsl"));
        let m_pdraw = module("pdraw", include_str!("shaders/particles_draw.wgsl"));
        let m_road = module("roads", include_str!("shaders/roads.wgsl"));
        let m_field = module("airfield", include_str!("shaders/airfield.wgsl"));
        let m_lights = module("lights", include_str!("shaders/lights.wgsl"));
        let m_tree = module("trees", include_str!("shaders/trees.wgsl"));
        let m_inst = module("inst", &format!("{facade}\n{}", include_str!("shaders/inst.wgsl")));
        let m_vol = module("vol", include_str!("shaders/vol.wgsl"));
        let m_ao = module("ao", include_str!("shaders/ao.wgsl"));
        let m_taa = module("taa", include_str!("shaders/taa.wgsl"));
        let m_dres = d.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("resolve"), source: wgpu::ShaderSource::Wgsl(include_str!("shaders/resolve.wgsl").into()) });

        // ---- shared resources
        let sampler = |addr, aniso: u16, cmp| {
            d.create_sampler(&wgpu::SamplerDescriptor {
                address_mode_u: addr,
                address_mode_v: addr,
                address_mode_w: addr,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: if cmp { wgpu::MipmapFilterMode::Nearest } else { wgpu::MipmapFilterMode::Linear },
                anisotropy_clamp: aniso,
                compare: if cmp { Some(wgpu::CompareFunction::LessEqual) } else { None },
                ..Default::default()
            })
        };
        let s_clamp = sampler(wgpu::AddressMode::ClampToEdge, 1, false);
        let s_repeat = sampler(wgpu::AddressMode::Repeat, 1, false);
        let s_aniso = sampler(wgpu::AddressMode::ClampToEdge, 16, false);
        let s_shadow = sampler(wgpu::AddressMode::ClampToEdge, 1, true);
        let rt = wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING;
        // Apple's GPUs keep render targets in a compressed form, and (macOS 27.2,
        // M4) the transmittance table came back from it with blocks missing:
        // every light in the scene lost its red and blue. Marking the tables as
        // storage textures too keeps them uncompressed; they are tiny.
        let lut = rt | wgpu::TextureUsages::STORAGE_BINDING;
        let lut_trans = view(&make_tex(d, "trans", 256, 64, 1, 1, HDR, lut));
        let lut_ms = view(&make_tex(d, "ms", 32, 32, 1, 1, HDR, lut));
        let lut_sky = view(&make_tex(d, "sky", 256, 128, 1, 1, HDR, lut));
        let black = view(&make_tex(d, "black", 1, 1, 1, 1, HDR, rt));
        let cs_tex = d.create_texture(&wgpu::TextureDescriptor {
            label: Some("cloud shadow"),
            size: wgpu::Extent3d { width: CS_RES, height: CS_RES, depth_or_array_layers: 6 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: CS_FMT,
            usage: rt,
            view_formats: &[],
        });
        let cs_all = cs_tex.create_view(&wgpu::TextureViewDescriptor { dimension: Some(wgpu::TextureViewDimension::D2Array), ..Default::default() });
        let cs_layers: Vec<_> = (0..6)
            .map(|i| cs_tex.create_view(&wgpu::TextureViewDescriptor { dimension: Some(wgpu::TextureViewDimension::D2), base_array_layer: i, array_layer_count: Some(1), ..Default::default() }))
            .collect();
        // What lookups read: the two cascades cross-faded between the last two maps.
        let cs_mix_tex = d.create_texture(&wgpu::TextureDescriptor {
            label: Some("cloud shadow mix"),
            size: wgpu::Extent3d { width: CS_RES, height: CS_RES, depth_or_array_layers: 2 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: CS_FMT,
            usage: rt,
            view_formats: &[],
        });
        let cs_mix = cs_mix_tex.create_view(&wgpu::TextureViewDescriptor { dimension: Some(wgpu::TextureViewDimension::D2Array), ..Default::default() });
        let cs_mix_layers: Vec<_> = (0..2)
            .map(|i| cs_mix_tex.create_view(&wgpu::TextureViewDescriptor { dimension: Some(wgpu::TextureViewDimension::D2), base_array_layer: i, array_layer_count: Some(1), ..Default::default() }))
            .collect();
        // No cloud shadow (all zeros) for the passes that draw it or run before it.
        let no_cs = make_tex(d, "no cloud shadow", 1, 1, 1, 1, CS_FMT, rt).create_view(&wgpu::TextureViewDescriptor { dimension: Some(wgpu::TextureViewDimension::D2Array), ..Default::default() });
        let (shadow_all, shadow) = Self::shadow_map(d, SHADOW_RES);
        let arr = |t: &wgpu::Texture| t.create_view(&wgpu::TextureViewDescriptor { dimension: Some(wgpu::TextureViewDimension::D2Array), ..Default::default() });
        let dummy_depth = arr(&make_tex(d, "dummy depth", 1, 1, 1, 1, DEPTH, rt));
        let vol3 = |label: &str, n: [u32; 3]| {
            d.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d { width: n[0], height: n[1], depth_or_array_layers: n[2] },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D3,
                format: HDR,
                usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
            .create_view(&Default::default())
        };
        let vol_dens = vol3("smoke density", VOL_N);
        let vol_light = vol3("smoke light", VOL_N);
        let dummy_vol = vol3("dummy volume", [1, 1, 1]);
        let noise_tex = make_tex(d, "noise3d", 128, 128, 128, 1, wgpu::TextureFormat::Rgba8Unorm, wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING);
        let noise = view(&noise_tex);
        {
            let m = d.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("noise3d"), source: wgpu::ShaderSource::Wgsl(include_str!("shaders/noise3d.wgsl").into()) });
            let p = d.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some("noise3d"), layout: None, module: &m, entry_point: Some("main"), compilation_options: Default::default(), cache: None });
            let bg = d.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &p.get_bind_group_layout(0), entries: &[tex_res(0, &noise)] });
            let mut enc = d.create_command_encoder(&Default::default());
            {
                let mut cp = enc.begin_compute_pass(&Default::default());
                cp.set_pipeline(&p);
                cp.set_bind_group(0, &bg, &[]);
                cp.dispatch_workgroups(32, 32, 32);
            }
            queue.submit([enc.finish()]);
        }
        let ocean = crate::ocean::OceanGpu::new(d, &queue);
        let shore = crate::shore::ShoreGpu::new(d, &queue, PHOTO_MASK_FMT);
        let wave = upload_mips(d, &queue, "wave", 256, wgpu::TextureFormat::Rgba8Unorm, 4, &crate::sources::mip_chain(wave_texture(), 256, 4));
        let ground_mats = GroundMats::new(d, &queue);
        let (det_alb, det_nrm) = ground_mats.views();
        let s_detail = sampler(wgpu::AddressMode::Repeat, 8, false);

        let globals_ub = d.create_buffer(&wgpu::BufferDescriptor { label: Some("globals"), size: std::mem::size_of::<Globals>() as u64, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let d2 = wgpu::TextureViewDimension::D2;
        let l0 = d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("globals"),
            entries: &[
                ub_entry(0, false, std::mem::size_of::<Globals>() as u64),
                tex_entry(1, d2, false),
                tex_entry(2, d2, false),
                tex_entry(3, d2, false),
                tex_entry(4, wgpu::TextureViewDimension::D3, false),
                tex_entry(5, d2, false),
                samp_entry(6, wgpu::SamplerBindingType::Filtering),
                samp_entry(7, wgpu::SamplerBindingType::Filtering),
                tex_entry(8, wgpu::TextureViewDimension::D2Array, true),
                samp_entry(9, wgpu::SamplerBindingType::Comparison),
                samp_entry(10, wgpu::SamplerBindingType::Filtering),
                tex_entry(11, wgpu::TextureViewDimension::D3, false),
                tex_entry(12, wgpu::TextureViewDimension::D2Array, false),
                tex_entry(13, wgpu::TextureViewDimension::D2Array, false),
                samp_entry(14, wgpu::SamplerBindingType::Filtering),
                tex_entry(15, d2, false),
                tex_entry(16, wgpu::TextureViewDimension::D2Array, false),
                tex_entry(17, wgpu::TextureViewDimension::D2Array, false),
                tex_entry(18, d2, false),
            ],
        });
        // Coverage of the photorealistic 3D tiles, seen from above (tiles3d.rs).
        let photo_mask = view(&make_tex(d, "photo mask", PHOTO_MASK, PHOTO_MASK, 1, 1, PHOTO_MASK_FMT, rt));
        let no_mask = view(&make_tex(d, "no photo mask", 1, 1, 1, 1, PHOTO_MASK_FMT, rt));
        // (owns what it shares, so `bg0` can be made again for a new shadow map)
        let make_bg0 = {
            let (d, l0, globals_ub, noise, wave, det_alb, det_nrm, ocean_v, shore_v) = (d.clone(), l0.clone(), globals_ub.clone(), noise.clone(), wave.clone(), det_alb.clone(), det_nrm.clone(), ocean.view.clone(), shore.view.clone());
            let (s_clamp, s_repeat, s_shadow, s_aniso, s_detail) = (s_clamp.clone(), s_repeat.clone(), s_shadow.clone(), s_aniso.clone(), s_detail.clone());
            move |sky: &wgpu::TextureView, trans: &wgpu::TextureView, ms: &wgpu::TextureView, sh: &wgpu::TextureView, vl: &wgpu::TextureView, pm: &wgpu::TextureView, cs: &wgpu::TextureView| {
            d.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("globals"),
                layout: &l0,
                entries: &[
                    buf_res(0, &globals_ub, 0),
                    tex_res(1, trans),
                    tex_res(2, ms),
                    tex_res(3, sky),
                    tex_res(4, &noise),
                    tex_res(5, &wave),
                    wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::Sampler(&s_clamp) },
                    wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::Sampler(&s_repeat) },
                    tex_res(8, sh),
                    wgpu::BindGroupEntry { binding: 9, resource: wgpu::BindingResource::Sampler(&s_shadow) },
                    wgpu::BindGroupEntry { binding: 10, resource: wgpu::BindingResource::Sampler(&s_aniso) },
                    tex_res(11, vl),
                    tex_res(12, &det_alb),
                    tex_res(13, &det_nrm),
                    wgpu::BindGroupEntry { binding: 14, resource: wgpu::BindingResource::Sampler(&s_detail) },
                    tex_res(15, pm),
                    tex_res(16, cs),
                    tex_res(17, &ocean_v),
                    tex_res(18, &shore_v),
                ],
            })
        }};
        let bg0 = make_bg0(&lut_sky, &lut_trans, &lut_ms, &shadow_all, &vol_light, &photo_mask, &cs_mix);
        // Variant for passes that render into the LUTs / shadow map / smoke volume / photo mask / cloud shadow themselves.
        let bg0_pre = make_bg0(&black, &lut_trans, &lut_ms, &dummy_depth, &dummy_vol, &no_mask, &no_cs);
        let bg0_shadow = make_bg0(&black, &lut_trans, &lut_ms, &dummy_depth, &dummy_vol, &photo_mask, &no_cs);
        let bg0_boot = make_bg0(&black, &black, &black, &dummy_depth, &dummy_vol, &no_mask, &no_cs);
        let bg0_ms = make_bg0(&black, &lut_trans, &black, &dummy_depth, &dummy_vol, &no_mask, &no_cs);
        let bg0_with: Box<dyn Fn(&wgpu::TextureView) -> wgpu::BindGroup> = {
            let (sky, trans, ms, vl, pm, cs) = (lut_sky.clone(), lut_trans.clone(), lut_ms.clone(), vol_light.clone(), photo_mask.clone(), cs_mix.clone());
            Box::new(move |sh| make_bg0(&sky, &trans, &ms, sh, &vl, &pm, &cs))
        };

        // ---- layouts
        let layout = |label: &str, entries: &[wgpu::BindGroupLayoutEntry]| d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some(label), entries });
        let l_tile = layout(
            "tile",
            &[
                ub_entry(0, true, 48),
                tex_entry(1, d2, false),
                tex_entry(2, d2, false),
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::VERTEX, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        );
        let l_build = layout("build", &[ub_entry(0, true, 64), tex_entry(1, d2, false)]);
        let l_photo = layout("photo", &[ub_entry(0, true, 32), tex_entry(1, d2, false)]);
        let l_skin = layout(
            "skin",
            &[
                tex_entry(1, d2, false),
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::VERTEX, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: true, min_binding_size: wgpu::BufferSize::new(64) }, count: None },
            ],
        );
        let skin_slots = d.create_buffer(&wgpu::BufferDescriptor { label: Some("skin slots"), size: ((SKIN_SLOT_BUF + SKIN_SLOTS) * 64) as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let l_tileu = layout("tile offset", &[ub_entry(0, true, 48)]);
        let l_casc = layout("cascade", &[ub_entry(0, true, 64)]);
        let l_dres = layout(
            "depth resolve",
            &[wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Depth, view_dimension: d2, multisampled: true }, count: None }],
        );
        let l_obj = layout("obj", &[ub_entry(0, true, std::mem::size_of::<ObjU>() as u64), tex_entry(1, wgpu::TextureViewDimension::D2Array, false)]);
        let l_glass = layout("glass", &[tex_entry(1, d2, false), tex_entry(2, d2, true)]);
        let ms_entry = |binding, depth| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: if depth { wgpu::TextureSampleType::Depth } else { wgpu::TextureSampleType::Float { filterable: false } },
                view_dimension: d2,
                multisampled: true,
            },
            count: None,
        };
        let l_composite = layout("composite", &[
            tex_entry(0, d2, false), tex_entry(1, d2, false), tex_entry(2, d2, true), tex_entry(3, d2, false), tex_entry(4, d2, false),
            ms_entry(5, false), ms_entry(6, false), ms_entry(7, true), ms_entry(8, false),
        ]);
        let l_composite1 = layout("composite 1x", &[tex_entry(0, d2, false), tex_entry(1, d2, false), tex_entry(2, d2, true), tex_entry(3, d2, false), tex_entry(4, d2, false)]);
        let l_ao = layout("ao", &[tex_entry(0, d2, true), tex_entry(1, d2, false), tex_entry(2, d2, false)]);
        let l_cloud = layout("cloud", &[tex_entry(0, d2, true)]);
        let l_cloud_apply = layout("cloud apply", &[tex_entry(0, d2, true), tex_entry(1, d2, false), tex_entry(3, d2, false), tex_entry(5, d2, false)]);
        let l_cloud_temporal = layout("cloud temporal", &[tex_entry(2, d2, false), tex_entry(3, d2, false), tex_entry(4, d2, false)]);
        let l_post = layout("post", &[tex_entry(0, d2, false), tex_entry(1, d2, false), tex_entry(2, d2, true)]);
        let storage = |ro: bool| wgpu::BindGroupLayoutEntry {
            binding: 1,
            visibility: if ro { wgpu::ShaderStages::VERTEX } else { wgpu::ShaderStages::COMPUTE },
            ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: ro }, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        };
        let psim_size = std::mem::size_of::<PSim>() as u64;
        let l_psim = layout("psim", &[ub_entry(0, false, psim_size), storage(false)]);
        let l_pdraw = layout("pdraw", &[ub_entry(0, false, psim_size), storage(true), tex_entry(2, d2, true)]);
        let cbuf = |binding: u32, ro: bool| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: ro }, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        };
        let stex = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::StorageTexture { access: wgpu::StorageTextureAccess::WriteOnly, format: HDR, view_dimension: wgpu::TextureViewDimension::D3 },
            count: None,
        };
        let l_vol_a = layout("smoke splat", &[ub_entry(0, false, psim_size), cbuf(1, true), cbuf(2, false), cbuf(3, false), cbuf(4, false), stex(5)]);
        let l_vol_b = layout("smoke light", &[ub_entry(0, false, psim_size), tex_entry(6, wgpu::TextureViewDimension::D3, false), stex(7)]);

        let pl = |l1: &wgpu::BindGroupLayout| d.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(&l0), Some(l1)], immediate_size: 0 });
        let pl0 = d.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(&l0)], immediate_size: 0 });

        // ---- pipelines
        let full = |label: &str, layout: &wgpu::PipelineLayout, m: &wgpu::ShaderModule, fs: &str, format: wgpu::TextureFormat, blend: Option<wgpu::BlendState>| {
            d.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(layout),
                vertex: wgpu::VertexState { module: m, entry_point: Some("vs_full"), compilation_options: Default::default(), buffers: &[] },
                primitive: Default::default(),
                depth_stencil: None,
                multisample: Default::default(),
                fragment: Some(wgpu::FragmentState { module: m, entry_point: Some(fs), compilation_options: Default::default(), targets: &[Some(wgpu::ColorTargetState { format, blend, write_mask: wgpu::ColorWrites::ALL })] }),
                multiview_mask: None,
                cache: None,
            })
        };
        let p_trans = full("trans", &pl0, &m_lut, "fs_trans", HDR, None);
        let p_ms = full("ms", &pl0, &m_lut, "fs_ms", HDR, None);
        let p_sky = full("sky", &pl0, &m_lut, "fs_sky", HDR, None);
        let l_csmix = layout("cloud shadow mix", &[tex_entry(0, wgpu::TextureViewDimension::D2Array, false), ub_entry(1, false, 48)]);
        let mix_ub = d.create_buffer(&wgpu::BufferDescriptor { label: Some("cloud shadow mix"), size: 48, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let bg_mix = d.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cloud shadow mix"),
            layout: &l_csmix,
            entries: &[tex_res(0, &cs_all), wgpu::BindGroupEntry { binding: 1, resource: mix_ub.as_entire_binding() }],
        });
        let m_csmix = d.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("csmix"), source: wgpu::ShaderSource::Wgsl(include_str!("shaders/csmix.wgsl").into()) });
        let pl_csmix = d.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(&l_csmix)], immediate_size: 0 });
        let cshadow = CloudShadow {
            p_in: full("cloud shadow", &pl0, &m_cloud, "fs_cshadow_in", CS_FMT, None),
            p_out: full("cloud shadow", &pl0, &m_cloud, "fs_cshadow_out", CS_FMT, None),
            layers: cs_layers,
            p_mix: [full("cloud shadow mix", &pl_csmix, &m_csmix, "fs_in", CS_FMT, None), full("cloud shadow mix", &pl_csmix, &m_csmix, "fs_out", CS_FMT, None)],
            bg_mix,
            mix_ub,
            mix_layers: cs_mix_layers,
            prev: [[0.0; 4]; CS_MAP],
            front: [[0.0; 4]; CS_MAP],
            back: [[0.0; 4]; CS_MAP],
            slot: [0, 0, 1],
            slice: 0,
            slices: if integrated { CS_SLICES_INTEGRATED } else { CS_SLICES },
            ok: false,
            anchor: [glam::DVec3::Z, glam::DVec3::X, glam::DVec3::Y],
            sun: glam::DVec3::Z,
            sun_v: glam::DVec3::ZERO,
            mix: ([0; 8], 1.0),
        };
        let p_composite = [full("composite", &pl(&l_composite), &m_comp, "fs", HDR, None), full("composite 1x", &pl(&l_composite1), &m_comp, "fs1", HDR, None)];
        // The land pass: fs1 (one sample a pixel) with WATER false.
        let land = |label: &str, l: &wgpu::BindGroupLayout| {
            d.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&pl(l)),
                vertex: wgpu::VertexState { module: &m_comp, entry_point: Some("vs_full"), compilation_options: Default::default(), buffers: &[] },
                primitive: Default::default(),
                depth_stencil: None,
                multisample: Default::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &m_comp,
                    entry_point: Some("fs1"),
                    compilation_options: wgpu::PipelineCompilationOptions { constants: &[("WATER", 0.0)], ..Default::default() },
                    targets: &[Some(wgpu::ColorTargetState { format: HDR, blend: None, write_mask: wgpu::ColorWrites::ALL })],
                }),
                multiview_mask: None,
                cache: None,
            })
        };
        let p_comp_land = [land("composite land", &l_composite), land("composite land 1x", &l_composite1)];
        let p_comp_water = [full("composite water", &pl(&l_composite), &m_comp, "fs_water1", HDR, None), full("composite water 1x", &pl(&l_composite1), &m_comp, "fs_water1", HDR, None)];
        let p_comp_edge = full("composite edge", &pl(&l_composite), &m_comp, "fs_edge", HDR, None);
        let p_gtao = full("gtao", &pl(&l_ao), &m_ao, "fs_gtao", AMB, None);
        let p_ao_blur = full("ao blur", &pl(&l_ao), &m_ao, "fs_blur", AMB, None);
        let p_cloud = d.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("cloud"),
            layout: Some(&pl(&l_cloud)),
            vertex: wgpu::VertexState { module: &m_cloud, entry_point: Some("vs_full"), compilation_options: Default::default(), buffers: &[] },
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &m_cloud,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState { format: HDR, blend: None, write_mask: wgpu::ColorWrites::ALL }), Some(wgpu::ColorTargetState { format: CLOUD_DIST, blend: None, write_mask: wgpu::ColorWrites::ALL })],
            }),
            multiview_mask: None,
            cache: None,
        });
        let p_cloud_temporal = full("cloud temporal", &pl(&l_cloud_temporal), &m_cloud, "fs_temporal", HDR, None);
        let over = wgpu::BlendComponent { src_factor: wgpu::BlendFactor::One, dst_factor: wgpu::BlendFactor::SrcAlpha, operation: wgpu::BlendOperation::Add };
        let p_cloud_apply = full("cloud apply", &pl(&l_cloud_apply), &m_cloud, "fs_apply", HDR, Some(wgpu::BlendState { color: over, alpha: wgpu::BlendComponent { src_factor: wgpu::BlendFactor::Zero, dst_factor: wgpu::BlendFactor::One, operation: wgpu::BlendOperation::Add } }));
        let p_taa = full("taa", &pl(&l_post), &m_taa, "fs_taa", HDR, None);
        let p_taa_up = full("taa up", &pl(&l_post), &m_taa, "fs_taa_up", HDR, None);
        let p_motion = full("motion", &pl(&l_post), &m_taa, "fs_motion", MOTION, None);
        let p_reactive = full("reactive", &pl(&l_post), &m_taa, "fs_reactive", wgpu::TextureFormat::R8Unorm, None);
        let p_down = full("down", &pl(&l_post), &m_post, "fs_down", HDR, None);
        let add = wgpu::BlendComponent { src_factor: wgpu::BlendFactor::One, dst_factor: wgpu::BlendFactor::One, operation: wgpu::BlendOperation::Add };
        let p_up = full("up", &pl(&l_post), &m_post, "fs_up", HDR, Some(wgpu::BlendState { color: add, alpha: add }));
        let p_final = full("final", &pl(&l_post), &m_post, "fs_final", wgpu::TextureFormat::Rgba8Unorm, None);
        let p_final_taa = full("final taa", &pl(&l_post), &m_post, "fs_final_taa", wgpu::TextureFormat::Rgba8Unorm, None);
        let p_final_taa_up = full("final taa up", &pl(&l_post), &m_post, "fs_final_taa_up", wgpu::TextureFormat::Rgba8Unorm, None);
        let p_final_taau = full("final taau", &pl(&l_post), &m_post, "fs_final_taau", wgpu::TextureFormat::Rgba8Unorm, None);
        let invert = wgpu::BlendComponent { src_factor: wgpu::BlendFactor::OneMinusDst, dst_factor: wgpu::BlendFactor::OneMinusSrc, operation: wgpu::BlendOperation::Add };
        let p_logo = full("logo", &pl(&l_post), &m_post, "fs_logo", wgpu::TextureFormat::Rgba8Unorm, Some(wgpu::BlendState { color: invert, alpha: wgpu::BlendComponent { src_factor: wgpu::BlendFactor::Zero, dst_factor: wgpu::BlendFactor::One, operation: wgpu::BlendOperation::Add } }));
        let p_blit = full("blit", &pl(&l_post), &m_post, if config.format.is_srgb() { "fs_blit_srgb" } else { "fs_blit" }, config.format, None);
        let pm = wgpu::BlendComponent { src_factor: wgpu::BlendFactor::One, dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha, operation: wgpu::BlendOperation::Add };
        let p_particles = d.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("particles"),
            layout: Some(&pl(&l_pdraw)),
            vertex: wgpu::VertexState { module: &m_pdraw, entry_point: Some("vs"), compilation_options: Default::default(), buffers: &[] },
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState { module: &m_pdraw, entry_point: Some("fs"), compilation_options: Default::default(), targets: &[Some(wgpu::ColorTargetState { format: HDR, blend: Some(wgpu::BlendState { color: pm, alpha: pm }), write_mask: wgpu::ColorWrites::ALL })] }),
            multiview_mask: None,
            cache: None,
        });
        let p_psim = d.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some("psim"), layout: Some(&pl(&l_psim)), module: &m_psim, entry_point: Some("main"), compilation_options: Default::default(), cache: None });
        let (pl_va, pl_vb) = (pl(&l_vol_a), pl(&l_vol_b));
        let p_vol = [("splat", &pl_va), ("blur_x", &pl_va), ("blur_y", &pl_va), ("blur_z", &pl_va), ("light", &pl_vb)]
            .map(|(e, l)| d.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some(e), layout: Some(l), module: &m_vol, entry_point: Some(e), compilation_options: Default::default(), cache: None }));

        // Geometry pipelines. Mode 0: lit colour + aux into the multisampled G-buffer.
        // Mode 1: depth only into a shadow cascade. Mode 2: blended decal (roads).
        // Mode 3: as 0, with the leaf card entry points. Mode 4: as 1, with a
        // fragment stage that can cut holes (the aircraft's windows).
        // Mode 7: as 0, with the entry points for trees near the camera.
        // Mode 10: as 1, drawing each tree with the model it is given.
        // Modes 12 and 13: as 1 and 4, only the faces turned away from the sun
        // (the 3D tiles: their ground then casts nothing, sparing the fill of
        // a whole layer of ground per cascade, and walls cast from the back).
        // (consts: the fragment shader's pipeline-overridable constants)
        let geom_k = |label: &str, l1: Option<&wgpu::BindGroupLayout>, m: &wgpu::ShaderModule, buffers: &[Option<wgpu::VertexBufferLayout>], mode: u8, samples: u32, consts: &[(&str, f64)]| {
            let shadow = mode == 1 || mode == 4 || mode == 10 || mode == 12 || mode == 13;
            let groups = [Some(&l0), l1, Some(&l_casc)];
            let layout = d.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: if shadow { &groups[..] } else { &groups[..2] }, immediate_size: 0 });
            let alpha = wgpu::BlendComponent { src_factor: wgpu::BlendFactor::SrcAlpha, dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha, operation: wgpu::BlendOperation::Add };
            let keep = wgpu::BlendComponent { src_factor: wgpu::BlendFactor::Zero, dst_factor: wgpu::BlendFactor::One, operation: wgpu::BlendOperation::Add };
            let targets = [
                Some(wgpu::ColorTargetState { format: HDR, blend: if mode == 2 { Some(wgpu::BlendState { color: alpha, alpha: keep }) } else { None }, write_mask: wgpu::ColorWrites::ALL }),
                Some(wgpu::ColorTargetState { format: HDR, blend: None, write_mask: if mode == 2 { wgpu::ColorWrites::empty() } else { wgpu::ColorWrites::ALL } }),
                Some(wgpu::ColorTargetState { format: AMB, blend: None, write_mask: if mode == 2 { wgpu::ColorWrites::empty() } else { wgpu::ColorWrites::ALL } }),
            ];
            d.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                vertex: wgpu::VertexState { module: m, entry_point: Some(match mode {
                    10 => "vs_shadow_fixed",
                    _ if shadow => "vs_shadow",
                    3 => "vs_card",
                    7 => "vs_near",
                    11 => "vs_skin",
                    8 => "vs_tess",
                    9 => "vs_cells",
                    14 => "vs_water",
                    _ => "vs",
                }), compilation_options: Default::default(), buffers },
                primitive: wgpu::PrimitiveState { cull_mode: if mode == 12 || mode == 13 { Some(wgpu::Face::Front) } else { None }, ..Default::default() },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH,
                    depth_write_enabled: Some(mode != 2),
                    depth_compare: Some(match mode {
                        _ if shadow => wgpu::CompareFunction::LessEqual,
                        2 => wgpu::CompareFunction::GreaterEqual,
                        _ => wgpu::CompareFunction::Greater,
                    }),
                    stencil: Default::default(),
                    bias: if shadow { wgpu::DepthBiasState { constant: 2, slope_scale: 2.0, clamp: 0.0 } } else { Default::default() },
                }),
                multisample: wgpu::MultisampleState { count: if shadow { 1 } else { samples }, mask: !0, alpha_to_coverage_enabled: false },
                fragment: if mode == 4 || mode == 13 {
                    Some(wgpu::FragmentState { module: m, entry_point: Some("fs_shadow"), compilation_options: Default::default(), targets: &[] })
                } else if shadow {
                    None
                } else {
                    Some(wgpu::FragmentState { module: m, entry_point: Some(match mode { 3 => "fs_card", 5 => "fs_solid", 7 => "fs_cut", 11 => "fs_skin", 14 => "fs_water", _ => "fs" }), compilation_options: wgpu::PipelineCompilationOptions { constants: consts, ..Default::default() }, targets: &targets })
                },
                multiview_mask: None,
                cache: None,
            })
        };
        // The shadow pipelines, and the main view's in both kinds (`Renderer::msaa`).
        let geom_n = |label: &str, l1: Option<&wgpu::BindGroupLayout>, m: &wgpu::ShaderModule, buffers: &[Option<wgpu::VertexBufferLayout>], mode: u8, samples: u32| geom_k(label, l1, m, buffers, mode, samples, &[]);
        let geom = |label: &str, l1: Option<&wgpu::BindGroupLayout>, m: &wgpu::ShaderModule, buffers: &[Option<wgpu::VertexBufferLayout>], mode: u8| geom_n(label, l1, m, buffers, mode, 1);
        let geom2 = |label: &str, l1: Option<&wgpu::BindGroupLayout>, m: &wgpu::ShaderModule, buffers: &[Option<wgpu::VertexBufferLayout>], mode: u8| [geom_n(label, l1, m, buffers, mode, MSAA), geom_n(label, l1, m, buffers, mode, 1)];
        use wgpu::VertexFormat as VF;
        let vb = |stride: u64, instance: bool, attributes: &'static [wgpu::VertexAttribute]| {
            Some(wgpu::VertexBufferLayout { array_stride: stride, step_mode: if instance { wgpu::VertexStepMode::Instance } else { wgpu::VertexStepMode::Vertex }, attributes })
        };
        const A_TERRAIN: [wgpu::VertexAttribute; 5] = wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32, 2 => Float32x3, 3 => Float32, 4 => Float32x2];
        const A_BUILD: [wgpu::VertexAttribute; 5] = [
            wgpu::VertexAttribute { format: VF::Float32x3, offset: 0, shader_location: 0 },
            wgpu::VertexAttribute { format: VF::Float32x3, offset: 12, shader_location: 1 },
            wgpu::VertexAttribute { format: VF::Float32x2, offset: 24, shader_location: 2 },
            wgpu::VertexAttribute { format: VF::Unorm8x4, offset: 32, shader_location: 3 },
            wgpu::VertexAttribute { format: VF::Uint8x4, offset: 36, shader_location: 4 },
        ];
        const A_MESH: [wgpu::VertexAttribute; 3] = wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x4];
        const A_ROAD: [wgpu::VertexAttribute; 3] = wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x4, 2 => Float32];
        const A_TREEV: [wgpu::VertexAttribute; 5] = wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x4, 6 => Float32x4, 7 => Float32x4];
        const A_TREE: [wgpu::VertexAttribute; 3] = [
            wgpu::VertexAttribute { format: VF::Float32x4, offset: 0, shader_location: 3 },
            wgpu::VertexAttribute { format: VF::Float32x4, offset: 16, shader_location: 4 },
            wgpu::VertexAttribute { format: VF::Unorm8x4, offset: 32, shader_location: 5 },
        ];
        const A_INST: [wgpu::VertexAttribute; 5] = wgpu::vertex_attr_array![3 => Float32x4, 4 => Float32x4, 5 => Float32x4, 6 => Float32x4, 7 => Float32x4];
        let geom2k = |label: &str, l1: Option<&wgpu::BindGroupLayout>, m: &wgpu::ShaderModule, buffers: &[Option<wgpu::VertexBufferLayout>], mode: u8, consts: &[(&str, f64)]| [MSAA, 1].map(|n| geom_k(label, l1, m, buffers, mode, n, consts));
        let no_photo = [("PHOTO", 0.0)];
        let p_terrain = [geom2("terrain", Some(&l_tile), &m_terrain, &[vb(40, false, &A_TERRAIN)], 0), geom2k("terrain solid", Some(&l_tile), &m_terrain, &[vb(40, false, &A_TERRAIN)], 0, &no_photo)];
        // Mode 8: cells near the camera as tessellated patches; 9: the rest of such a tile.
        let p_terrain_tess = [geom2("terrain tess", Some(&l_tile), &m_terrain, &[], 8), geom2k("terrain tess solid", Some(&l_tile), &m_terrain, &[], 8, &no_photo)];
        let p_terrain_cells = [geom2("terrain cells", Some(&l_tile), &m_terrain, &[], 9), geom2k("terrain cells solid", Some(&l_tile), &m_terrain, &[], 9, &no_photo)];
        // Mode 14: a tile's water as its own fine grid (Water High, terrain.wgsl vs_water).
        let p_water = geom2("water grid", Some(&l_tile), &m_terrain, &[], 14);
        let p_build = geom2("build", Some(&l_build), &m_build, &[vb(40, false, &A_BUILD)], 0);
        let p_build_solid = geom2k("build solid", Some(&l_build), &m_build, &[vb(40, false, &A_BUILD)], 0, &[("DISSOLVE", 0.0)]);
        const A_PHOTO: [wgpu::VertexAttribute; 3] = [
            wgpu::VertexAttribute { format: VF::Float32x3, offset: 0, shader_location: 0 },
            wgpu::VertexAttribute { format: VF::Float32x2, offset: 12, shader_location: 1 },
            wgpu::VertexAttribute { format: VF::Snorm8x4, offset: 20, shader_location: 2 },
        ];
        let p_photo = geom2("photo", Some(&l_photo), &m_photo, &[vb(24, false, &A_PHOTO)], 0);
        // Mode 11: the 3D tiles' mesh of a destroyed building, riding on its pieces.
        const A_SKIN: [wgpu::VertexAttribute; 4] = [
            wgpu::VertexAttribute { format: VF::Float32x3, offset: 0, shader_location: 0 },
            wgpu::VertexAttribute { format: VF::Float32x2, offset: 12, shader_location: 1 },
            wgpu::VertexAttribute { format: VF::Snorm8x4, offset: 20, shader_location: 2 },
            wgpu::VertexAttribute { format: VF::Uint32, offset: 24, shader_location: 3 },
        ];
        let p_skin = geom2("skin", Some(&l_skin), &m_photo, &[vb(28, false, &A_SKIN)], 11);
        let p_photo_shadow = geom("photo shadow", Some(&l_photo), &m_photo, &[vb(24, false, &A_PHOTO)], 13);
        let p_photo_shadow_fast = geom("photo shadow fast", Some(&l_photo), &m_photo, &[vb(24, false, &A_PHOTO)], 12);
        // The coverage mask: the same tiles from above, 1 wherever they are.
        let p_photo_mask = d.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("photo mask"),
            layout: Some(&d.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(&l0), Some(&l_photo), Some(&l_casc)], immediate_size: 0 })),
            vertex: wgpu::VertexState { module: &m_photo, entry_point: Some("vs_mask"), compilation_options: Default::default(), buffers: &[vb(24, false, &A_PHOTO)] },
            primitive: wgpu::PrimitiveState { cull_mode: None, ..Default::default() },
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState { module: &m_photo, entry_point: Some("fs_mask"), compilation_options: Default::default(), targets: &[Some(wgpu::ColorTargetState { format: PHOTO_MASK_FMT, blend: None, write_mask: wgpu::ColorWrites::RED })] }),
            multiview_mask: None,
            cache: None,
        });
        // Our water into the same mask (green): where the mesh gives way to it.
        let p_water_top = d.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("water top"),
            layout: Some(&d.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(&l0), Some(&l_tile), Some(&l_casc)], immediate_size: 0 })),
            vertex: wgpu::VertexState { module: &m_terrain, entry_point: Some("vs_top"), compilation_options: Default::default(), buffers: &[vb(40, false, &A_TERRAIN)] },
            primitive: wgpu::PrimitiveState { cull_mode: None, ..Default::default() },
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState { module: &m_terrain, entry_point: Some("fs_top"), compilation_options: Default::default(), targets: &[Some(wgpu::ColorTargetState { format: PHOTO_MASK_FMT, blend: None, write_mask: wgpu::ColorWrites::GREEN })] }),
            multiview_mask: None,
            cache: None,
        });
        // Where there is ground at all, into the shore mask (red; our water
        // goes into its green with `p_water_top`).
        let p_shore_cover = d.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("shore cover"),
            layout: Some(&d.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(&l0), Some(&l_tile), Some(&l_casc)], immediate_size: 0 })),
            vertex: wgpu::VertexState { module: &m_terrain, entry_point: Some("vs_top"), compilation_options: Default::default(), buffers: &[vb(40, false, &A_TERRAIN)] },
            primitive: wgpu::PrimitiveState { cull_mode: None, ..Default::default() },
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState { module: &m_terrain, entry_point: Some("fs_cover"), compilation_options: Default::default(), targets: &[Some(wgpu::ColorTargetState { format: PHOTO_MASK_FMT, blend: None, write_mask: wgpu::ColorWrites::RED })] }),
            multiview_mask: None,
            cache: None,
        });
        // Our street lamps into the same mask (blue, with the road's height in
        // alpha): the light they would throw on the mesh at night.
        let max = wgpu::BlendComponent { src_factor: wgpu::BlendFactor::One, dst_factor: wgpu::BlendFactor::One, operation: wgpu::BlendOperation::Max };
        let p_lamp_top = d.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("lamp top"),
            layout: Some(&d.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(&l0), Some(&l_tileu), Some(&l_casc)], immediate_size: 0 })),
            vertex: wgpu::VertexState { module: &m_road, entry_point: Some("vs_top"), compilation_options: Default::default(), buffers: &[vb(32, false, &A_ROAD)] },
            primitive: wgpu::PrimitiveState { cull_mode: None, ..Default::default() },
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState { module: &m_road, entry_point: Some("fs_top"), compilation_options: Default::default(), targets: &[Some(wgpu::ColorTargetState { format: PHOTO_MASK_FMT, blend: Some(wgpu::BlendState { color: max, alpha: max }), write_mask: wgpu::ColorWrites::BLUE | wgpu::ColorWrites::ALPHA })] }),
            multiview_mask: None,
            cache: None,
        });
        let p_build_shadow = geom("build shadow", Some(&l_build), &m_build, &[vb(40, false, &A_BUILD)], 1);
        const A_PART: [wgpu::VertexAttribute; 5] = wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x4, 3 => Float32x2, 4 => Float32x4];
        let p_obj = geom2("obj", Some(&l_obj), &m_obj, &[vb(64, false, &A_PART)], 0);
        let p_obj_shadow = geom("obj shadow", Some(&l_obj), &m_obj, &[vb(64, false, &A_PART)], 4);
        let p_obj_solid = geom2("obj solid", Some(&l_obj), &m_obj, &[vb(64, false, &A_PART)], 5);
        let p_obj_shadow_solid = geom("obj shadow solid", Some(&l_obj), &m_obj, &[vb(64, false, &A_PART)], 1);
        // Glass and the propeller blur: blended over the finished, lit scene,
        // tested against the resolved depth.
        let p_glass = d.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("glass"),
            layout: Some(&d.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(&l0), Some(&l_obj), Some(&l_glass)], immediate_size: 0 })),
            vertex: wgpu::VertexState { module: &m_obj, entry_point: Some("vs"), compilation_options: Default::default(), buffers: &[vb(64, false, &A_PART)] },
            primitive: wgpu::PrimitiveState { cull_mode: None, ..Default::default() },
            depth_stencil: Some(wgpu::DepthStencilState { format: DEPTH, depth_write_enabled: Some(false), depth_compare: Some(wgpu::CompareFunction::Greater), stencil: Default::default(), bias: Default::default() }),
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &m_obj,
                entry_point: Some("fs_glass"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: HDR,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent { src_factor: wgpu::BlendFactor::One, dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha, operation: wgpu::BlendOperation::Add },
                        alpha: wgpu::BlendComponent { src_factor: wgpu::BlendFactor::Zero, dst_factor: wgpu::BlendFactor::One, operation: wgpu::BlendOperation::Add },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        let p_road = geom2("road", Some(&l_tileu), &m_road, &[vb(32, false, &A_ROAD)], 2);
        const A_AIR: [wgpu::VertexAttribute; 5] = wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x4, 2 => Float32x4, 3 => Float32, 4 => Float32x2];
        let p_field = geom2("airfield", Some(&l_tileu), &m_field, &[vb(56, false, &A_AIR)], 2);
        const A_LIGHT: [wgpu::VertexAttribute; 6] = wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32, 2 => Float32x3, 3 => Float32, 4 => Unorm8x4, 5 => Uint8x4];
        let p_lights = d.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("lights"),
            layout: Some(&d.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(&l0), Some(&l_tileu)], immediate_size: 0 })),
            vertex: wgpu::VertexState { module: &m_lights, entry_point: Some("vs"), compilation_options: Default::default(), buffers: &[vb(40, true, &A_LIGHT)] },
            primitive: Default::default(),
            depth_stencil: Some(wgpu::DepthStencilState { format: DEPTH, depth_write_enabled: Some(false), depth_compare: Some(wgpu::CompareFunction::Greater), stencil: Default::default(), bias: Default::default() }),
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &m_lights,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: HDR,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent { src_factor: wgpu::BlendFactor::One, dst_factor: wgpu::BlendFactor::One, operation: wgpu::BlendOperation::Add },
                        alpha: wgpu::BlendComponent { src_factor: wgpu::BlendFactor::Zero, dst_factor: wgpu::BlendFactor::One, operation: wgpu::BlendOperation::Add },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        let p_tree = [geom2("tree", Some(&l_tileu), &m_tree, &[vb(72, false, &A_TREEV), vb(36, true, &A_TREE)], 0), geom2k("tree solid", Some(&l_tileu), &m_tree, &[vb(72, false, &A_TREEV), vb(36, true, &A_TREE)], 0, &no_photo)];
        let imp = crate::impostor::Impostors::new(d, &l0, &l_tileu, &m_tree);
        let p_tree_near = geom2("trees near", Some(&l_tileu), &m_tree, &[vb(72, false, &A_TREEV), vb(36, true, &A_TREE)], 7);
        let p_tree_shadow = geom("tree shadow", Some(&l_tileu), &m_tree, &[vb(72, false, &A_TREEV), vb(36, true, &A_TREE)], 1);
        let p_tree_shadow_fixed = geom("tree shadow fixed", Some(&l_tileu), &m_tree, &[vb(72, false, &A_TREEV), vb(36, true, &A_TREE)], 10);
        let p_card = geom2("leaf cards", Some(&l_tileu), &m_tree, &[vb(72, false, &A_TREEV), vb(36, true, &A_TREE)], 3);
        let p_inst = [geom2("inst", None, &m_inst, &[vb(40, false, &A_MESH), vb(80, true, &A_INST)], 0), geom2k("inst solid", None, &m_inst, &[vb(40, false, &A_MESH), vb(80, true, &A_INST)], 0, &no_photo)];
        let p_inst_shadow = geom("inst shadow", None, &m_inst, &[vb(40, false, &A_MESH), vb(80, true, &A_INST)], 1);
        let p_dres = d.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("depth resolve"),
            layout: Some(&d.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(&l_dres)], immediate_size: 0 })),
            vertex: wgpu::VertexState { module: &m_dres, entry_point: Some("vs"), compilation_options: Default::default(), buffers: &[] },
            primitive: Default::default(),
            depth_stencil: Some(wgpu::DepthStencilState { format: DEPTH, depth_write_enabled: Some(true), depth_compare: Some(wgpu::CompareFunction::Always), stencil: Default::default(), bias: Default::default() }),
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState { module: &m_dres, entry_point: Some("fs"), compilation_options: Default::default(), targets: &[] }),
            multiview_mask: None,
            cache: None,
        });

        // ---- per-draw uniform pools
        let pool = |label: &str, n: u64| d.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: n * UB_ALIGN, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let (tile_ub, build_ub, obj_ub, casc_ub) = (pool("tiles", MAX_TILES), pool("builds", MAX_BUILDS + 1 + MAX_FIELDS as u64), pool("objs", MAX_OBJS), pool("cascades", CASCADES as u64 + 4));
        let photo_ub = pool("photo tiles", MAX_PHOTO);
        let bg_tileu = d.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &l_tileu, entries: &[buf_res(0, &build_ub, 48)] });
        let bg_casc = d.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &l_casc, entries: &[buf_res(0, &casc_ub, 64)] });
        let dynbuf = |label: &str, size: u64| d.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size, usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let (inst_buf, fall_buf) = (dynbuf("instances", MAX_INST as u64 * 80), dynbuf("loose trees", MAX_FALL as u64 * 36));
        let near_buf = dynbuf("near trees", MAX_NEAR as u64 * 36);
        let dyn_lights = dynbuf("aircraft lights", MAX_DYN_LIGHTS as u64 * 40);
        let paint = paint_texture(d, &queue);
        let bg_obj = d.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &l_obj, entries: &[buf_res(0, &obj_ub, std::mem::size_of::<ObjU>() as u64), tex_res(1, &paint)] });
        let ix = crate::sources::tile_indices();
        let tile_ibuf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("tile idx"), contents: bytemuck::cast_slice(&ix), usage: wgpu::BufferUsages::INDEX });
        let patch = |n: u16| {
            let mut v = Vec::new();
            for b in 0..n {
                for a in 0..n {
                    let (p00, p10, p01, p11) = (b * (n + 1) + a, b * (n + 1) + a + 1, (b + 1) * (n + 1) + a, (b + 1) * (n + 1) + a + 1);
                    v.extend([p00, p01, p10, p10, p01, p11]);
                }
            }
            v
        };
        let (tix, cix) = (patch(TESS_N), patch(1));
        let tess_ibuf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("tess idx"), contents: bytemuck::cast_slice(&tix), usage: wgpu::BufferUsages::INDEX });
        let cell_ibuf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("cell idx"), contents: bytemuck::cast_slice(&cix), usage: wgpu::BufferUsages::INDEX });
        let water_ibuf = WATER_GRIDS.map(|n| {
            let ix = water_indices(n);
            (d.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("water grid idx"), contents: bytemuck::cast_slice(&ix), usage: wgpu::BufferUsages::INDEX }), ix.len() as u32)
        });

        // ---- particles
        let psim_ub = d.create_buffer(&wgpu::BufferDescriptor { label: Some("psim"), size: psim_size, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let pbuf = d.create_buffer(&wgpu::BufferDescriptor { label: Some("particles"), size: MAX_PARTICLES as u64 * 64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let bg_psim = d.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &l_psim, entries: &[buf_res(0, &psim_ub, 0), buf_res(1, &pbuf, 0)] });
        let sbuf = |label: &str, size: u64| d.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let (vol_acc, vol_a, vol_b) = (sbuf("smoke splat", VOL_CELLS * 8), sbuf("smoke blur a", VOL_CELLS * 8), sbuf("smoke blur b", VOL_CELLS * 8));
        let bg_vol_a = d.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &l_vol_a,
            entries: &[buf_res(0, &psim_ub, 0), buf_res(1, &pbuf, 0), buf_res(2, &vol_acc, 0), buf_res(3, &vol_a, 0), buf_res(4, &vol_b, 0), tex_res(5, &vol_dens)],
        });
        let bg_vol_b = d.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &l_vol_b, entries: &[buf_res(0, &psim_ub, 0), tex_res(6, &vol_dens), tex_res(7, &vol_light)] });
        let fluid = crate::fluid::FluidGpu::new(d, &l0, common, &vol_acc);

        let egui = egui_wgpu::Renderer::new(d, wgpu::TextureFormat::Rgba8Unorm, egui_wgpu::RendererOptions::default());
        let targets = Self::make_targets(d, (config.width, config.height), (config.width, config.height), true, false, false, &l_dres, &l_ao, &l_composite, &l_composite1, &l_cloud, &l_cloud_apply, &l_cloud_temporal, &l_post, &l_pdraw, &l_glass, &psim_ub, &pbuf, &black, 2);
        let fluid_t = fluid.draw_targets(d, &targets.depth, config.width, config.height);
        // The transmittance and multi-scatter tables depend only on the planet
        // and the haze: bake them now, and again whenever the haze changes.
        let lut_bake = LutBake { p_trans, p_ms, bg_trans: bg0_boot, bg_ms: bg0_ms, trans: lut_trans, ms: lut_ms };
        {
            let mut g = Globals::zeroed();
            g.planet = [0.0, 0.0, 0.0, crate::geo::R as f32];
            g.weather[3] = 1.0;
            queue.write_buffer(&globals_ub, 0, bytemuck::bytes_of(&g));
            let mut enc = d.create_command_encoder(&Default::default());
            lut_bake.draw(&mut enc);
            queue.submit([enc.finish()]);
        }
        let prof = profile.then(|| crate::gpuprof::GpuProf::new(&device, &queue));
        Ok(Self {
            ground_mats,
            casc_ub, bg_casc, bg_tileu, l_dres, l_ao, p_gtao, p_ao_blur, p_dres, p_road, p_field, p_lights, dyn_lights, p_tree, p_tree_near, imp, p_tree_shadow, p_tree_shadow_fixed, p_card, near_buf, p_inst, p_inst_shadow, inst_buf, fall_buf, tree_lib: None, inst_lib: None,
            globals_ub, bg0, bg0_pre, bg0_shadow, bg0_with, lut_sky, shadow, shadow_res: SHADOW_RES, clouds: 2, ao: true, bloom: true, shadow_fast: false, msaa: true, l_tile, l_build, l_composite, l_composite1, l_cloud, l_cloud_apply, l_cloud_temporal, l_post, l_pdraw, l_glass,
            tile_ub, l_photo, photo_ub, p_photo, p_photo_mask, p_photo_shadow, p_photo_shadow_fast, l_skin, p_skin, skin_slots, p_water_top, p_lamp_top, photo_mask, photo_mask_dirty: true, build_ub, obj_ub, bg_obj, tile_ibuf, tile_nidx: ix.len() as u32,
            p_sky, p_terrain, p_terrain_tess, p_terrain_cells, p_water, water_ibuf, tess_ibuf, tess_nidx: tix.len() as u32, cell_ibuf, p_build, p_build_solid, p_build_shadow, p_obj, p_obj_shadow, p_obj_solid, p_obj_shadow_solid, p_glass, p_composite, p_comp_land, p_comp_water, p_comp_edge, p_cloud, p_cloud_apply, p_cloud_temporal, p_particles, p_taa, p_taa_up, p_final_taau, p_motion, p_reactive, p_psim, p_down, p_up, p_final, p_final_taa, p_final_taa_up, p_blit, p_logo,
            psim_ub, pbuf, bg_psim, phead: 0, p_vol, bg_vol_a, bg_vol_b, vol_acc, fluid, fluid_t, ocean, shore, p_shore_cover, smoke_until: 0.0, parts_until: 0.0, now: 0.0, black, targets, cloud_flip: 0, cloud_hist_ok: false, cshadow, lut_bake, lut_haze: 1.0, taa_flip: 0, taa_hist_ok: false, meshes: Vec::new(), egui, logo_font: crate::logo::find_font(), logo_key: None, surface, config, queue, device, stats: [0.0; 4], headless: false, scale: 1.0, integrated, prof, fsr_on: false, fsr_ok, fsr: None,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn make_targets(
        d: &wgpu::Device, (w, h): (u32, u32), native: (u32, u32), msaa: bool, taau: bool, fsr: bool, l_dres: &wgpu::BindGroupLayout, l_ao: &wgpu::BindGroupLayout, l_composite: &wgpu::BindGroupLayout, l_composite1: &wgpu::BindGroupLayout, l_cloud: &wgpu::BindGroupLayout, l_cloud_apply: &wgpu::BindGroupLayout, l_cloud_temporal: &wgpu::BindGroupLayout,
        l_post: &wgpu::BindGroupLayout, l_pdraw: &wgpu::BindGroupLayout, l_glass: &wgpu::BindGroupLayout, psim_ub: &wgpu::Buffer, pbuf: &wgpu::Buffer, black: &wgpu::TextureView, cloud_div: u32,
    ) -> Targets {
        let rt = wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING;
        let ms = |label: &str, f| view(&make_tex_ms(d, label, w, h, 1, 1, f, rt, MSAA));
        let amb = view(&make_tex(d, "amb", w, h, 1, 1, AMB, rt));
        let ao_raw = view(&make_tex(d, "ao raw", w, h, 1, 1, AMB, rt));
        let ao = view(&make_tex(d, "ao", w, h, 1, 1, AMB, rt));
        let color = view(&make_tex(d, "color", w, h, 1, 1, HDR, rt));
        let aux = view(&make_tex(d, "aux", w, h, 1, 1, HDR, rt));
        let depth_tex = make_tex(d, "depth", w, h, 1, 1, DEPTH, rt);
        let depth = view(&depth_tex);
        // MSAA off: no multisampled G-buffer, the plain one stands for it.
        let (color_ms, aux_ms, amb_ms, depth_ms) = if msaa {
            (ms("color ms", HDR), ms("aux ms", HDR), ms("amb ms", AMB), ms("depth ms", DEPTH))
        } else {
            (color.clone(), aux.clone(), amb.clone(), depth.clone())
        };
        let hdr_tex = make_tex(d, "hdr", w, h, 1, 1, HDR, rt | wgpu::TextureUsages::COPY_SRC);
        let hdr = view(&hdr_tex);
        let scene_tex = make_tex(d, "scene behind glass", w, h, 1, 1, HDR, wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST);
        // The clouds are marched at a half or a quarter of the scene's resolution.
        let (cw, ch) = ((w / cloud_div).max(1), (h / cloud_div).max(1));
        let cloud = view(&make_tex(d, "cloud", cw, ch, 1, 1, HDR, rt));
        let cloud_dist = view(&make_tex(d, "cloud dist", cw, ch, 1, 1, CLOUD_DIST, rt));
        let cloud_hist = [view(&make_tex(d, "cloud hist 0", cw, ch, 1, 1, HDR, rt)), view(&make_tex(d, "cloud hist 1", cw, ch, 1, 1, HDR, rt))];
        let bloom_tex = make_tex(d, "bloom", (w / 2).max(64), (h / 2).max(64), 1, BLOOM_MIPS, HDR, rt);
        let bloom: Vec<_> = (0..BLOOM_MIPS).map(|i| bloom_tex.create_view(&wgpu::TextureViewDescriptor { base_mip_level: i, mip_level_count: Some(1), ..Default::default() })).collect();
        // (at the window's resolution: the UI is drawn into it)
        let ldr_tex = make_tex(d, "ldr", native.0, native.1, 1, 1, wgpu::TextureFormat::Rgba8Unorm, rt | wgpu::TextureUsages::COPY_SRC);
        let ldr = view(&ldr_tex);
        let logo_tex = make_tex(d, "logo", native.0, native.1, 1, 1, wgpu::TextureFormat::R8Unorm, wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST);
        let logo = view(&logo_tex);
        // (temporal upscaling: at the window's resolution, `Renderer::taau`)
        // (FSR: the first is its output, also at the window's resolution)
        let (tw, th) = if taau || fsr { native } else { (w, h) };
        let tu = if fsr { rt | wgpu::TextureUsages::STORAGE_BINDING } else { rt };
        let taa_tex = [make_tex(d, "taa 0", tw, th, 1, 1, HDR, tu), make_tex(d, "taa 1", tw, th, 1, 1, HDR, tu)];
        let taa = [view(&taa_tex[0]), view(&taa_tex[1])];
        let bg = |l: &wgpu::BindGroupLayout, e: &[wgpu::BindGroupEntry]| d.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: l, entries: e });
        let post = |src: &wgpu::TextureView, bl: &wgpu::TextureView| bg(l_post, &[tex_res(0, src), tex_res(1, bl), tex_res(2, &depth)]);
        Targets {
            bg_dres: msaa.then(|| bg(l_dres, &[tex_res(0, &depth_ms)])),
            bg_composite: if msaa {
                bg(l_composite, &[
                    tex_res(0, &color), tex_res(1, &aux), tex_res(2, &depth), tex_res(3, &ao), tex_res(4, &amb),
                    tex_res(5, &color_ms), tex_res(6, &aux_ms), tex_res(7, &depth_ms), tex_res(8, &amb_ms),
                ])
            } else {
                bg(l_composite1, &[tex_res(0, &color), tex_res(1, &aux), tex_res(2, &depth), tex_res(3, &ao), tex_res(4, &amb)])
            },
            color_ms, aux_ms, amb_ms, depth_ms,
            // Each pass reads the other AO texture from the one it writes.
            bg_gtao: bg(l_ao, &[tex_res(0, &depth), tex_res(1, &aux), tex_res(2, &ao)]),
            bg_ao_blur: bg(l_ao, &[tex_res(0, &depth), tex_res(1, &aux), tex_res(2, &ao_raw)]),
            ao_raw, ao,
            bg_cloud: bg(l_cloud, &[tex_res(0, &depth)]),
            bg_cloud_temporal: [0, 1].map(|i| bg(l_cloud_temporal, &[tex_res(2, &cloud), tex_res(3, &cloud_dist), tex_res(4, &cloud_hist[1 - i])])),
            bg_cloud_apply: [0, 1].map(|i| bg(l_cloud_apply, &[tex_res(0, &depth), tex_res(1, &cloud_hist[i]), tex_res(3, &cloud_dist), tex_res(5, &color)])),
            bg_glass: bg(l_glass, &[tex_res(1, &view(&scene_tex)), tex_res(2, &depth)]),
            hdr_tex, scene_tex,
            bg_particles: bg(l_pdraw, &[buf_res(0, psim_ub, 0), buf_res(1, pbuf, 0), tex_res(2, &depth)]),
            bg_taa: [0, 1].map(|i| post(&hdr, &taa[1 - i])),
            bg_down0: [&hdr, &taa[0], &taa[1]].map(|src| post(src, black)),
            bg_down: (1..BLOOM_MIPS as usize).map(|i| post(&bloom[i - 1], black)).collect(),
            bg_up: (0..BLOOM_MIPS as usize - 1).map(|i| post(&bloom[i + 1], black)).collect(),
            bg_final: [&hdr, &taa[0], &taa[1]].map(|src| post(src, &bloom[0])),
            fsr: fsr.then(|| {
                let st = wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING;
                let opaque = make_tex(d, "fsr opaque", w, h, 1, 1, HDR, wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST);
                FsrTargets {
                    motion: make_tex(d, "fsr motion", w, h, 1, 1, MOTION, rt),
                    reactive: make_tex(d, "fsr reactive", w, h, 1, 1, wgpu::TextureFormat::R8Unorm, rt),
                    bg_reactive: post(&hdr, &view(&opaque)),
                    opaque,
                    dilated_depth: make_tex(d, "fsr dilated depth", w, h, 1, 1, wgpu::TextureFormat::R32Float, st),
                    dilated_motion: make_tex(d, "fsr dilated motion", w, h, 1, 1, MOTION, st),
                    prev_depth: d.create_buffer(&wgpu::BufferDescriptor { label: Some("fsr previous depth"), size: w as u64 * h as u64 * 4, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false }),
                }
            }),
            taa,
            taa_tex,
            bg_blit: post(&ldr, black),
            bg_logo: post(&logo, black),
            logo_tex,
            color, aux, amb, depth, depth_tex, hdr, cloud, cloud_dist, cloud_hist, bloom, ldr_tex, ldr,
        }
    }

    pub fn resize(&mut self, w: u32, h: u32) {
        if w == 0 || h == 0 || (w == self.config.width && h == self.config.height) {
            return;
        }
        self.config.width = w;
        self.config.height = h;
        self.surface.configure(&self.device, &self.config);
        log::info!("surface {w}x{h}");
        self.cshadow.ok = false;
        self.remake_targets();
    }

    /// Draw the 3D scene at `s` times the window's resolution (settings::DYN_MIN_SCALE..1: the lowest is dynamic resolution's).
    pub fn set_render_scale(&mut self, s: f32) {
        let s = s.clamp(crate::settings::DYN_MIN_SCALE, 1.0);
        if s != self.scale {
            self.scale = s;
            log::info!("render scale {s:.2}");
            self.remake_targets();
        }
    }

    /// The shadow map: all its layers (to sample) and each one (to draw into).
    fn shadow_map(d: &wgpu::Device, res: u32) -> (wgpu::TextureView, Vec<wgpu::TextureView>) {
        let t = d.create_texture(&wgpu::TextureDescriptor {
            label: Some("shadow"),
            size: wgpu::Extent3d { width: res, height: res, depth_or_array_layers: CASCADES as u32 + 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let all = t.create_view(&wgpu::TextureViewDescriptor { dimension: Some(wgpu::TextureViewDimension::D2Array), ..Default::default() });
        let layers = (0..CASCADES as u32 + 1)
            .map(|i| t.create_view(&wgpu::TextureViewDescriptor { dimension: Some(wgpu::TextureViewDimension::D2), base_array_layer: i, array_layer_count: Some(1), ..Default::default() }))
            .collect();
        (all, layers)
    }

    /// Side of the shadow map's layers: SHADOW_RES, or half of it for Low
    /// shadows. A new map starts empty: draw every cascade into it next frame.
    pub fn set_shadow_res(&mut self, res: u32) {
        if res != self.shadow_res {
            let (all, layers) = Self::shadow_map(&self.device, res);
            self.bg0 = (self.bg0_with)(&all);
            self.shadow = layers;
            self.shadow_res = res;
            // (the impostors were baked with the old one bound, which they do not read)
        }
    }

    pub fn shadow_res(&self) -> u32 {
        self.shadow_res
    }

    /// Cloud quality 0..=2 (Graphics settings): below High the march runs at
    /// a quarter of the scene's resolution.
    pub fn set_clouds(&mut self, q: u8) {
        let q = q.min(2);
        if q != self.clouds {
            let remake = Self::cloud_div(q) != Self::cloud_div(self.clouds);
            self.clouds = q;
            if remake {
                self.remake_targets();
            }
            self.cloud_hist_ok = false;
        }
    }

    /// Temporal upscaling (taa.wgsl fs_taa_up): for a scene well under the
    /// window's size (Medium and Low; High's 0.75 on an integrated GPU keeps
    /// the plain TAA and stretch). FABLE_OPT_OFF=taau for the stretch alone.
    pub fn taau(&self) -> bool {
        self.scale < TAAU_BELOW && crate::gpuprof::opt_on("taau") && !self.fsr_active()
    }

    /// FSR 3 (fsr/) in place of the TAA and the temporal upscaling, at any
    /// scale, while the temporal AA is on: the Graphics setting Upscaler,
    /// where the GPU can run it.
    pub fn fsr_active(&self) -> bool {
        self.fsr_on && self.fsr_ok
    }

    /// Whether this GPU can run FSR (the Graphics setting offers it).
    pub fn fsr_supported(&self) -> bool {
        self.fsr_ok
    }

    /// The Graphics setting Upscaler: FSR (true) or the game's own TAAU.
    pub fn set_upscaler(&mut self, fsr: bool) {
        if fsr == self.fsr_on {
            return;
        }
        self.fsr_on = fsr;
        // (off: its history goes, the pipelines stay for the next time)
        if let Some(s) = self.fsr.as_mut().filter(|_| !fsr) {
            s.view = None;
        }
        if self.fsr_active() && self.fsr.is_none() {
            let t = std::time::Instant::now();
            use crate::fsr::FsrContextFlags as F;
            let ctx = crate::fsr::FsrContext::new(&self.device, F::HIGH_DYNAMIC_RANGE | F::DEPTH_INVERTED | F::DEPTH_INFINITE);
            let exposure = make_tex(&self.device, "fsr exposure", 1, 1, 1, 1, wgpu::TextureFormat::R32Float, wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST);
            self.fsr = Some(Fsr { ctx, view: None, exposure, hist_ok: false, last: t });
            log::info!("FSR: pipelines made in {:.0} ms", t.elapsed().as_secs_f64() * 1e3);
        }
        self.remake_targets();
    }

    /// FSR: upscale this frame (`hdr`, at the scene's resolution) into the
    /// first `taa` texture (the window's).
    /// (A function of the parts it uses: `render` holds the rest borrowed.)
    #[allow(clippy::too_many_arguments)]
    fn fsr_dispatch(fsr: &mut Fsr, queue: &wgpu::Queue, prof: &Option<crate::gpuprof::GpuProf>, t: &Targets, (w, h): (u32, u32), up: [u32; 2], enc: &mut wgpu::CommandEncoder, g: &Globals, cam: [f32; 2], reactive: bool) {
        let Some(ft) = t.fsr.as_ref() else { return };
        // Its history is made for the largest scene yet (dynamic resolution
        // then works within it) and the window's size.
        if fsr.view.as_ref().is_none_or(|v| v.max_render_size()[0] < w || v.max_render_size()[1] < h || v.max_upscale_size() != up) {
            fsr.view = Some(fsr.ctx.create_view(queue, [w, h], up));
            log::info!("FSR: {w}x{h} to {}x{}", up[0], up[1]);
        }
        // Colours are judged as they will be shown (post.wgsl finish: G.time.z, AGX_EXPOSURE).
        queue.write_texture(fsr.exposure.as_image_copy(), bytemuck::bytes_of(&(g.time[2] * 1.6)), wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(4), rows_per_image: Some(1) }, wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 });
        let now = std::time::Instant::now();
        let dt_ms = (now - fsr.last).as_secs_f32() * 1e3;
        fsr.last = now;
        let info = crate::fsr::FsrDispatchInfo {
            color: t.hdr_tex.clone(),
            depth: t.depth_tex.clone(),
            motion_vectors: ft.motion.clone(),
            exposure: Some(fsr.exposure.clone()),
            reactive_mask: reactive.then(|| ft.reactive.clone()),
            transparency_and_composition: None,
            dilated_depth: ft.dilated_depth.clone(),
            dilated_motion_vectors: ft.dilated_motion.clone(),
            reconstructed_previous_depth: ft.prev_depth.clone(),
            output: t.taa_tex[0].clone(),
            // (G.taa.xy is the offset in uv, y down: FSR's is in pixels)
            jitter_offset: [g.taa[0] * w as f32, g.taa[1] * h as f32],
            // (fs_motion writes them in uv)
            motion_vector_scale: [w as f32, h as f32],
            render_size: [w, h],
            upscale_size: up,
            enable_sharpening: FSR_SHARPNESS > 0.0,
            sharpness: FSR_SHARPNESS,
            // (the game's step: what the frame shows; the clock's when paused)
            frame_time_delta: (if g.time[1] > 0.0 { g.time[1] * 1e3 } else { dt_ms }).clamp(1.0, 1000.0),
            pre_exposure: 1.0,
            reset_history: !fsr.hist_ok || g.cam_delta[3] < 0.5,
            // (reversed depth with no far plane)
            camera_near: f32::MAX,
            camera_far: cam[1],
            camera_fov_y: cam[0],
            view_space_to_meters_factor: 1.0,
        };
        match fsr.ctx.dispatch(fsr.view.as_mut().unwrap(), enc, &info, prof.as_ref()) {
            Ok(()) => fsr.hist_ok = true,
            Err(e) => {
                log::warn!("{e}");
                fsr.hist_ok = false;
            }
        }
    }

    /// The ocean's wave grids (Graphics `water`: 0 Low, 1 High; ocean.rs `grid`).
    pub fn set_water(&mut self, quality: u8) {
        self.ocean.set_quality(&self.queue, quality);
    }

    /// Multisample the G-buffer (Graphics `msaa`: every preset but Low).
    pub fn set_msaa(&mut self, on: bool) {
        if on != self.msaa {
            self.msaa = on;
            self.remake_targets();
        }
    }

    fn cloud_div(q: u8) -> u32 {
        if q >= 2 { 2 } else { 4 }
    }

    /// Wait for the display's refresh or not, from the next frame on.
    pub fn set_vsync(&mut self, on: bool) {
        let mode = if on { wgpu::PresentMode::AutoVsync } else { wgpu::PresentMode::AutoNoVsync };
        if mode != self.config.present_mode {
            self.config.present_mode = mode;
            self.surface.configure(&self.device, &self.config);
        }
    }

    /// The 3D scene's resolution as a fraction of the window's.
    pub fn render_scale(&self) -> f32 {
        self.scale
    }

    pub fn plane_shadow_res(&self) -> u32 {
        (if self.integrated { PLANE_SHADOW_RES_INTEGRATED } else { PLANE_SHADOW_RES }).min(self.shadow_res)
    }

    /// The 3D scene's resolution.
    pub fn render_size(&self) -> (u32, u32) {
        let f = |n: u32| ((n as f32 * self.scale).round() as u32).max(1);
        (f(self.config.width), f(self.config.height))
    }

    fn remake_targets(&mut self) {
        self.cloud_hist_ok = false;
        self.taa_hist_ok = false;
        self.logo_key = None;
        let (w, h) = self.render_size();
        self.targets = Self::make_targets(&self.device, (w, h), (self.config.width, self.config.height), self.msaa, self.taau(), self.fsr_active(), &self.l_dres, &self.l_ao, &self.l_composite, &self.l_composite1, &self.l_cloud, &self.l_cloud_apply, &self.l_cloud_temporal, &self.l_post, &self.l_pdraw, &self.l_glass, &self.psim_ub, &self.pbuf, &self.black, Self::cloud_div(self.clouds));
        self.fluid_t = self.fluid.draw_targets(&self.device, &self.targets.depth, w, h);
    }

    pub fn upload_tile(&self, t: &crate::sources::TileData) -> TileGpu {
        let d = &self.device;
        let vbuf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&t.verts), usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::STORAGE });
        let img = upload_mips(d, &self.queue, "imagery", t.img_size, wgpu::TextureFormat::Rgba8UnormSrgb, 4, &t.imagery);
        let mask = upload_mips(d, &self.queue, "mask", crate::sources::MASK as u32, wgpu::TextureFormat::Rgba8Unorm, 4, &t.mask);
        let bind = d.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &self.l_tile, entries: &[buf_res(0, &self.tile_ub, 48), tex_res(1, &img), tex_res(2, &mask), buf_res(3, &vbuf, 0)] });
        TileGpu { vbuf, bind }
    }

    /// Create a building tile's GPU objects. The geometry buffers start out
    /// empty: a dense downtown tile is ~100 MB, and every byte written goes
    /// through wgpu staging memory, which (without resizable BAR) sits in a
    /// 246 MiB heap. Filling them all at once overflows that heap and makes
    /// every later upload crawl, so `World::update` fills them a slice per
    /// frame with `fill_buffer`, matching `BuildGpu::targets` to
    /// `building_payload`.
    /// A photorealistic 3D tile: its mesh and its texture (with its mips).
    pub fn upload_photo(&self, m: &crate::tiles3d::MeshData) -> PhotoGpu {
        let d = &self.device;
        let vbuf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("photo"), contents: bytemuck::cast_slice(&m.verts), usage: wgpu::BufferUsages::VERTEX });
        let ibuf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("photo idx"), contents: bytemuck::cast_slice(&m.idx), usage: wgpu::BufferUsages::INDEX });
        let (w, h) = (m.mips[0].0, m.mips[0].1);
        let t = make_tex(d, "photo", w, h, 1, m.mips.len() as u32, wgpu::TextureFormat::Rgba8UnormSrgb, wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST);
        for (i, (mw, mh, px)) in m.mips.iter().enumerate() {
            self.queue.write_texture(
                wgpu::TexelCopyTextureInfo { texture: &t, mip_level: i as u32, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                px,
                wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(mw * 4), rows_per_image: Some(*mh) },
                wgpu::Extent3d { width: *mw, height: *mh, depth_or_array_layers: 1 },
            );
        }
        let tv = view(&t);
        let bind = d.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &self.l_photo, entries: &[buf_res(0, &self.photo_ub, 32), tex_res(1, &tv)] });
        let rad = m.verts.iter().map(|v| glam::Vec3::from(v.pos).length()).fold(0.0f32, f32::max);
        PhotoGpu { vbuf, ibuf, n_idx: m.idx.len() as u32, bind, view: tv, rad }
    }

    /// A destroyed building's mesh from one tile (skin.rs), with that tile's texture.
    pub fn make_skin(&self, verts: &[SkinVertex], tex: &wgpu::TextureView) -> SkinGpu {
        let d = &self.device;
        let vbuf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("skin"), contents: bytemuck::cast_slice(verts), usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST });
        let bind = d.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &self.l_skin, entries: &[tex_res(1, tex), buf_res(2, &self.skin_slots, (SKIN_SLOTS * 64) as u64)] });
        SkinGpu { vbuf, bind, n: verts.len() as u32 }
    }

    /// Rewrites some of a skin's vertices (pieces that broke in two).
    pub fn write_skin(&self, s: &SkinGpu, first: usize, verts: &[SkinVertex]) {
        self.queue.write_buffer(&s.vbuf, (first * std::mem::size_of::<SkinVertex>()) as u64, bytemuck::cast_slice(verts));
    }

    pub fn upload_buildings(&self, b: &crate::buildings::BuildingData) -> BuildGpu {
        let d = &self.device;
        let buf = |data: &[u8], usage| d.create_buffer(&wgpu::BufferDescriptor { label: None, size: (data.len().max(64) as u64).next_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT), usage: usage | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let vbuf = buf(bytemuck::cast_slice(&b.verts), wgpu::BufferUsages::VERTEX);
        let ibuf = buf(bytemuck::cast_slice(&b.indices), wgpu::BufferUsages::INDEX);
        let atlas = upload_mips(d, &self.queue, "atlas", b.atlas_size, wgpu::TextureFormat::Rgba8UnormSrgb, 4, &b.atlas);
        let bind = d.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &self.l_build, entries: &[buf_res(0, &self.build_ub, 64), tex_res(1, &atlas)] });
        let road = (!b.road_idx.is_empty()).then(|| (buf(bytemuck::cast_slice(&b.road_verts), wgpu::BufferUsages::VERTEX), buf(bytemuck::cast_slice(&b.road_idx), wgpu::BufferUsages::INDEX), b.road_idx.len() as u32));
        let trees = (!b.trees.is_empty()).then(|| buf(bytemuck::cast_slice(&b.trees), wgpu::BufferUsages::VERTEX));
        BuildGpu { vbuf, ibuf, n_idx: b.indices.len() as u32, n_main: b.n_main, n_det: b.n_det, n_bld: b.n_bld, n_far: b.n_far, n_far_tall: b.n_far_tall, cls: b.cls, bind, road, trees }
    }

    pub fn upload_field(&self, m: &crate::airfield::FieldMesh) -> FieldGpu {
        let d = &self.device;
        let init = |data: &[u8], usage| d.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("airfield"), contents: if data.is_empty() { &[0u8; 64] } else { data }, usage });
        FieldGpu {
            vbuf: init(bytemuck::cast_slice(&m.verts), wgpu::BufferUsages::VERTEX),
            ibuf: init(bytemuck::cast_slice(&m.idx), wgpu::BufferUsages::INDEX),
            n_idx: m.idx.len() as u32,
            lights: (!m.lights.is_empty()).then(|| (init(bytemuck::cast_slice(&m.lights), wgpu::BufferUsages::VERTEX), m.lights.len() as u32)),
        }
    }
    /// Bytes an airfield's upload takes, for the per-frame budget.
    pub fn field_bytes(m: &crate::airfield::FieldMesh) -> usize {
        std::mem::size_of_val(&m.verts[..]) + m.idx.len() * 4 + std::mem::size_of_val(&m.lights[..])
    }

    /// Write `data` into `buf` at byte offset `off` (both multiples of 4).
    pub fn fill_buffer(&self, buf: &wgpu::Buffer, off: u64, data: &[u8]) {
        self.queue.write_buffer(buf, off, data);
    }

    fn mesh_lib<V: Pod>(&self, meshes: &[(Vec<V>, Vec<u32>)]) -> MeshLib {
        let (mut v, mut i, mut ranges) = (Vec::new(), Vec::new(), Vec::new());
        for (mv, mi) in meshes {
            ranges.push((i.len() as u32, mi.len() as u32, v.len() as i32));
            v.extend_from_slice(mv);
            i.extend_from_slice(mi);
        }
        let d = &self.device;
        MeshLib {
            vbuf: d.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&v), usage: wgpu::BufferUsages::VERTEX }),
            ibuf: d.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&i), usage: wgpu::BufferUsages::INDEX }),
            ranges,
        }
    }
    /// Tree and undergrowth models, indexed by `variant * LODS + level of detail`
    /// (the last "level" being the model's leaf cards).
    /// Also takes the pictures for the impostors.
    pub fn set_tree_meshes(&mut self, meshes: &[(Vec<TreeVertex>, Vec<u32>)]) {
        let lib = self.mesh_lib(meshes);
        let bounds = crate::impostor::Impostors::bounds(meshes);
        self.imp.bake(&self.device, &self.queue, &self.bg0, &self.bg_tileu, &lib.vbuf, &lib.ibuf, &lib.ranges, &bounds);
        self.tree_lib = Some(lib);
    }
    /// Models for instanced solids (cars, rubble, rocks).
    pub fn set_inst_meshes(&mut self, meshes: &[(Vec<MeshVertex>, Vec<u32>)]) {
        self.inst_lib = Some(self.mesh_lib(meshes));
    }

    pub fn add_mesh(&mut self, verts: &[PartVertex], idx: &[u32]) -> usize {
        let vbuf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(verts), usage: wgpu::BufferUsages::VERTEX });
        let ibuf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(idx), usage: wgpu::BufferUsages::INDEX });
        self.meshes.push(MeshGpu { vbuf, ibuf, n_idx: idx.len() as u32 });
        self.meshes.len() - 1
    }
    /// Swap the geometry behind an existing mesh id.
    pub fn replace_mesh(&mut self, i: usize, verts: &[PartVertex], idx: &[u32]) {
        let vbuf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(verts), usage: wgpu::BufferUsages::VERTEX });
        let ibuf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(idx), usage: wgpu::BufferUsages::INDEX });
        self.meshes[i] = MeshGpu { vbuf, ibuf, n_idx: idx.len() as u32 };
    }

    /// Append particles to the ring buffer (oldest are overwritten).
    pub fn emit(&mut self, ps: &[Particle]) {
        let mut ps = ps;
        let smoky = ps.iter().filter(|p| matches!(p.kind as u32, 0 | 2 | 5 | 7 | 8)).map(|p| p.life).fold(0.0f32, f32::max);
        if smoky > 0.0 {
            self.smoke_until = self.smoke_until.max(self.now + smoky + 0.5);
        }
        if let Some(life) = ps.iter().map(|p| p.life).reduce(f32::max) {
            self.parts_until = self.parts_until.max(self.now + life + 1.0);
        }
        if ps.len() > MAX_PARTICLES as usize {
            ps = &ps[..MAX_PARTICLES as usize];
        }
        let first = (MAX_PARTICLES - self.phead).min(ps.len() as u32) as usize;
        self.queue.write_buffer(&self.pbuf, self.phead as u64 * 64, bytemuck::cast_slice(&ps[..first]));
        if first < ps.len() {
            self.queue.write_buffer(&self.pbuf, 0, bytemuck::cast_slice(&ps[first..]));
        }
        self.phead = (self.phead + ps.len() as u32) % MAX_PARTICLES;
    }
    pub fn clear_particles(&mut self) {
        self.queue.write_buffer(&self.pbuf, 0, &vec![0u8; MAX_PARTICLES as usize * 64]);
        self.phead = 0;
        self.smoke_until = 0.0;
        self.parts_until = 0.0;
    }

    /// Start loading the photo-scanned ground materials (see materials.rs).
    pub fn load_ground_photos(&mut self, fetch: crate::fetch::Fetcher) {
        self.ground_mats.photos = Some(crate::materials::spawn(fetch));
    }

    pub fn render(&mut self, mut f: Frame) {
        // The UI's texture changes are taken in whether or not a frame can be
        // drawn: each builds on the one before, so one dropped with a skipped
        // frame (a hidden window on macOS) would leave the next nothing to patch.
        if let Some((_, delta, _)) = &f.egui {
            for (id, ds) in &delta.set {
                for d in ds {
                    self.egui.update_texture(&self.device, &self.queue, *id, d);
                }
            }
        }
        let t_acq = std::time::Instant::now();
        // Headless: everything is drawn but nothing is shown, so no frame of the
        // window's is needed (and a hidden or locked screen does not stop it).
        let frame = if self.headless {
            None
        } else {
            match self.surface.get_current_texture() {
                wgpu::CurrentSurfaceTexture::Success(t) | wgpu::CurrentSurfaceTexture::Suboptimal(t) => Some(t),
                wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                    self.surface.configure(&self.device, &self.config);
                    None
                }
                _ => {
                    // Hidden or minimised (macOS hands out no frames then): idle
                    // rather than spin the loop at thousands of empty frames a second.
                    std::thread::sleep(std::time::Duration::from_millis(30));
                    None
                }
            }
        };
        if frame.is_none() && !self.headless {
            if let Some((_, delta, _)) = &f.egui {
                for id in &delta.free {
                    self.egui.free_texture(id);
                }
            }
            return;
        }
        let t_enc = std::time::Instant::now();
        self.stats[1] += (t_enc - t_acq).as_secs_f64();
        let sview = frame.as_ref().map(|f| f.texture.create_view(&Default::default()));
        let q = &self.queue;
        // which of each main-view pipeline: multisampled or not
        let mi = if self.msaa { 0 } else { 1 };
        self.cloud_flip ^= 1;
        if !self.cloud_hist_ok {
            f.g.cam_delta[3] = 0.0;
        }
        self.cloud_hist_ok = true;
        // Anti-aliasing blends into last frame's result, if there is one and
        // the camera has not jumped since.
        let taa = f.g.taa[2] > 0.5;
        let taau = taa && self.taau();
        // (FSR in place of either: it keeps its own history, fsr_dispatch)
        let fsr = taa && self.fsr_active();
        if let Some(s) = self.fsr.as_mut().filter(|_| !fsr) {
            s.hist_ok = false;
        }
        if !self.taa_hist_ok || f.g.cam_delta[3] < 0.5 {
            f.g.taa[2] = 0.0;
        }
        self.taa_hist_ok = taa;
        self.taa_flip ^= 1;
        // A clock jump (new flight) makes the bookkeeping meaningless.
        if f.g.time[0] < self.now - 1.0 {
            self.smoke_until = 0.0;
            self.parts_until += f.g.time[0] - self.now;
        } else if !f.sim_steps {
            // (paused: the particles do not age, so neither does their deadline)
            self.parts_until += (f.g.time[0] - self.now).max(0.0);
        }
        self.now = f.g.time[0];
        let parts = self.now < self.parts_until;
        let smoke = self.now < self.smoke_until || f.fluid.is_some();
        f.g.vol_u[3] = if smoke { 1.0 } else { 0.0 };
        self.ground_mats.poll(&self.device, q);
        f.g.det = self.ground_mats.params;
        let cs_full = self.cshadow.prepare(&mut f.g);
        f.g.start[1] = self.clouds as f32;
        f.g.start[2] = if self.bloom { 0.0 } else { 1.0 };
        f.g.start[3] = if crate::gpuprof::opt_on("shadow_taps") { 0.0 } else if self.shadow_fast { 1.0 } else { 2.0 };
        // Below High clouds the haze and sky marches take half their steps
        // (the temporal AA smooths their jitter): common.wgsl atm_steps.
        f.g.vol_n[3] = if crate::gpuprof::opt_on("water_edge") { 0.0 } else { 1.0 };
        f.g.vol_e[3] = if self.clouds >= 2 || !crate::gpuprof::opt_on("atm_steps") { 1.0 } else { 0.5 };
        q.write_buffer(&self.cshadow.mix_ub, 0, &self.cshadow.mix_bytes());
        q.write_buffer(&self.globals_ub, 0, bytemuck::bytes_of(&f.g));
        q.write_buffer(&self.psim_ub, 0, bytemuck::bytes_of(&f.psim));
        let n_tiles = f.tiles.len().min(MAX_TILES as usize);
        let n_shore_tiles = f.shore_tiles.len().min(MAX_TILES as usize - n_tiles);
        let n_builds = f.details.len().min(MAX_BUILDS as usize);
        let n_objs = f.objs.len().min(MAX_OBJS as usize);
        let mut blob = vec![0u8; (n_tiles + n_shore_tiles) * UB_ALIGN as usize];
        for (i, (_, u)) in f.tiles.iter().take(n_tiles).chain(f.shore_tiles.iter().take(n_shore_tiles)).enumerate() {
            blob[i * 256..i * 256 + 48].copy_from_slice(bytemuck::bytes_of(u));
        }
        q.write_buffer(&self.tile_ub, 0, &blob);
        // One extra slot after the tiles for loose trees: they are already
        // camera-relative, and offset.w = 1 keeps them to the model they were given.
        let n_fields = f.fields.len().min(MAX_FIELDS);
        let mut blob = vec![0u8; (n_builds + 1 + n_fields) * UB_ALIGN as usize];
        for (i, dd) in f.details.iter().take(n_builds).enumerate() {
            blob[i * 256..i * 256 + 64].copy_from_slice(bytemuck::bytes_of(&dd.u));
        }
        for (i, (_, o)) in f.fields.iter().take(n_fields).enumerate() {
            let k = (n_builds + 1 + i) * 256;
            blob[k..k + 12].copy_from_slice(bytemuck::cast_slice(o));
        }
        let field_slot = |i: usize| ((n_builds + 1 + i) * 256) as u32;
        let n_dyn = f.lights.len().min(MAX_DYN_LIGHTS);
        if n_dyn > 0 {
            q.write_buffer(&self.dyn_lights, 0, bytemuck::cast_slice(&f.lights[..n_dyn]));
        }
        blob[n_builds * 256 + 12..n_builds * 256 + 16].copy_from_slice(&1f32.to_ne_bytes());
        q.write_buffer(&self.build_ub, 0, &blob);
        let n_slots = f.skin_slots.len().min(SKIN_SLOT_BUF);
        if n_slots > 0 {
            q.write_buffer(&self.skin_slots, 0, bytemuck::cast_slice(&f.skin_slots[..n_slots]));
        }
        let n_photo = f.photo.len().min(MAX_PHOTO as usize);
        if n_photo > 0 {
            let mut blob = vec![0u8; n_photo * UB_ALIGN as usize];
            for (i, (_, u)) in f.photo.iter().take(n_photo).enumerate() {
                blob[i * 256..i * 256 + 32].copy_from_slice(bytemuck::cast_slice(u));
            }
            q.write_buffer(&self.photo_ub, 0, &blob);
        }
        let mut blob = vec![0u8; (CASCADES + 4) * UB_ALIGN as usize];
        for (i, m) in f.casc_vp.iter().enumerate() {
            blob[i * 256..i * 256 + 64].copy_from_slice(bytemuck::bytes_of(m));
        }
        q.write_buffer(&self.casc_ub, 0, &blob);
        // Instances grouped by model so each model is one draw.
        let group = |n_meshes: usize, items: &mut dyn Iterator<Item = u32>| -> (Vec<u32>, Vec<(u32, u32)>) {
            let mut counts = vec![0u32; n_meshes + 1];
            let ids: Vec<u32> = items.collect();
            for m in &ids {
                counts[(*m as usize).min(n_meshes.saturating_sub(1)) + 1] += 1;
            }
            for k in 0..n_meshes {
                counts[k + 1] += counts[k];
            }
            let ranges = (0..n_meshes).map(|k| (counts[k], counts[k + 1])).collect();
            let mut next = counts;
            let order = ids.iter().map(|m| { let k = (*m as usize).min(n_meshes.saturating_sub(1)); next[k] += 1; next[k] - 1 }).collect();
            (order, ranges)
        };
        let mut inst_ranges = Vec::new();
        if let Some(lib) = &self.inst_lib {
            let n = f.insts.len().min(MAX_INST);
            let (order, ranges) = group(lib.ranges.len(), &mut f.insts.iter().take(n).map(|i| i.0));
            let mut data = vec![Inst::zeroed(); n];
            for (k, (_, inst)) in f.insts.iter().take(n).enumerate() {
                data[order[k] as usize] = *inst;
            }
            q.write_buffer(&self.inst_buf, 0, bytemuck::cast_slice(&data));
            inst_ranges = ranges;
        }
        let mut fall_ranges = Vec::new();
        if let Some(lib) = &self.tree_lib {
            let n = f.fall.len().min(MAX_FALL);
            let (order, ranges) = group(lib.ranges.len(), &mut f.fall.iter().take(n).map(|i| i.0));
            let mut data = vec![TreeInst::zeroed(); n];
            for (k, (_, inst)) in f.fall.iter().take(n).enumerate() {
                data[order[k] as usize] = *inst;
            }
            q.write_buffer(&self.fall_buf, 0, bytemuck::cast_slice(&data));
            fall_ranges = ranges;
        }
        let n_near = f.near.len().min(MAX_NEAR);
        q.write_buffer(&self.near_buf, 0, bytemuck::cast_slice(&f.near[..n_near]));
        let loose_slot = n_builds as u32 * 256;
        let mut blob = vec![0u8; n_objs * UB_ALIGN as usize];
        for (i, o) in f.objs.iter().take(n_objs).enumerate() {
            let u = ObjU { model: o.model.to_cols_array_2d(), tint: o.tint, glow: o.glow, params: o.params, tear: o.tear, dmg: o.dmg, dmg2: o.dmg2 };
            blob[i * 256..i * 256 + std::mem::size_of::<ObjU>()].copy_from_slice(bytemuck::bytes_of(&u));
        }
        q.write_buffer(&self.obj_ub, 0, &blob);

        let _ = self.device.poll(wgpu::PollType::Poll);
        self.fluid.poll();
        let t = &self.targets;
        let mut enc = self.device.create_command_encoder(&Default::default());
        match &f.fluid {
            Some(ff) => self.fluid.compute(q, &mut enc, ff),
            None => self.fluid.live = false,
        }
        self.ocean.compute(q, &mut enc, &f.ocean, self.prof.as_ref());
        if f.g.weather[3] != self.lut_haze {
            self.lut_haze = f.g.weather[3];
            self.lut_bake.draw(&mut enc);
        }
        if f.sim_steps && parts {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: self.prof.as_ref().and_then(|p| p.compute("psim")) });
            cp.set_pipeline(&self.p_psim);
            cp.set_bind_group(0, &self.bg0, &[]);
            cp.set_bind_group(1, &self.bg_psim, &[]);
            cp.dispatch_workgroups(MAX_PARTICLES / 64, 1, 1);
        }
        if smoke {
            // Smoke volume: splat, smooth, then light it (see vol.wgsl).
            enc.clear_buffer(&self.vol_acc, 0, None);
            if self.fluid.live {
                self.fluid.splat_coarse(&mut enc);
            }
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: self.prof.as_ref().and_then(|p| p.compute("smoke vol")) });
            cp.set_bind_group(0, &self.bg0_pre, &[]);
            let g = [VOL_N[0] / 4, VOL_N[1] / 4, VOL_N[2] / 4];
            for (k, p) in self.p_vol.iter().enumerate() {
                cp.set_pipeline(p);
                cp.set_bind_group(1, if k == 4 { &self.bg_vol_b } else { &self.bg_vol_a }, &[]);
                if k == 0 {
                    cp.dispatch_workgroups(MAX_PARTICLES / 64, 1, 1);
                } else {
                    cp.dispatch_workgroups(g[0], g[1], g[2]);
                }
            }
        }
        self.cshadow.draw(&mut enc, &self.bg0_pre, cs_full);
        {
            let mut rp = enc.begin_render_pass(&pd(&self.prof, &[Some(color_att(&self.lut_sky, true))], None, "sky lut"));
            rp.set_pipeline(&self.p_sky);
            rp.set_bind_group(0, &self.bg0_pre, &[]);
            rp.draw(0..3, 0..1);
        }
        // Draws shared by the shadow cascades (c = Some) and the main view (c = None).
        let solids = |rp: &mut wgpu::RenderPass, c: Option<usize>| {
            let shadow = c.is_some();
            let bit = c.map(|c| 1u8 << c).unwrap_or(0);
            rp.set_pipeline(if shadow { &self.p_build_shadow } else { &self.p_build[mi] });
            let build_solid = !shadow && f.g.photo[2] <= 0.0 && crate::gpuprof::opt_on("build_solid");
            // Whether run `r` of `dd` discards nothing in buildings.wgsl fs (so
            // can take p_build_solid): no 3D tiles zone, the tile done fading in,
            // and level 0; or level 1 (2) done fading in, with the whole tile
            // short of where it starts dissolving with distance (dcam <= length(P)).
            let solid_run = |dd: &crate::render::DetailDraw, r: (u32, u32, u32)| {
                let d = dd.reach_m;
                build_solid && dd.u[14] >= 1.0 && match r.2 & 7 {
                    0 => true,
                    1 => dd.u[12] >= 1.0 && d < crate::buildings::BFAR[0],
                    2 => dd.u[13] >= 1.0 && d < 680.0 * 0.6, // (LOD2_FAR * 0.6)
                    _ => false,
                }
            };
            let mut cur_solid = Some(false);
            for (i, dd) in f.details.iter().take(n_builds).enumerate() {
                let g = dd.gpu;
                let (mid, fine) = (dd.u[12] > 0.0, dd.u[13] > 0.0);
                // Index runs (start, end, instance); the instance index tells the
                // shader the level of detail in its low 3 bits: 0 always drawn,
                // 1 middle, 2 finest, 3 full geometry dissolving out with distance,
                // 4 far stand-ins dissolving in; above them, 4x the distance scale
                // of the run's height class (buildings::HCLASS).
                let mut runs = [(0u32, 0u32, 0u32); 20];
                let mut nr = 0;
                let mut push = |r: (u32, u32, u32)| {
                    if r.1 > r.0 {
                        runs[nr] = r;
                        nr += 1;
                    }
                };
                // Level `r` (0 base, 1 middle, 2 finest, 3 far stand-ins): its start,
                // and the end of height class `c` in it.
                let start = [0, g.n_main, g.n_det, g.n_far];
                let end = |r: usize, k: usize| if k == 0 { start[r] } else { g.cls[r][k - 1] };
                // The first `k` height classes of level `r`, one draw each.
                let classes = |push: &mut dyn FnMut((u32, u32, u32)), r: usize, k: usize, lvl: u32| {
                    for c in 0..k {
                        push((end(r, c), end(r, c + 1), lvl | ((crate::buildings::HCLASS[c].1 * 4.0) as u32) << 3));
                    }
                };
                let (k_full, k_mid, k_fine) = (dd.reach[0] as usize, if mid { dd.reach[1] as usize } else { 0 }, if fine { dd.reach[2] as usize } else { 0 });
                match c {
                    Some(_) if dd.casc & bit == 0 => {}
                    Some(2) => push((g.n_far, g.n_far_tall, 0)),
                    // The second cascade swaps to the stand-ins at their usual distance.
                    Some(1) if k_full < crate::buildings::HCLASS.len() => {
                        push((g.n_bld, g.n_main, 0));
                        push((g.n_far, g.n_idx, 0));
                    }
                    Some(1) => push((0, g.n_main, 0)),
                    Some(_) => {
                        push((0, g.n_main, 0));
                        push((g.n_main, end(1, k_mid), 1));
                        push((g.n_det, if dd.u[13] > 0.5 { end(2, k_fine) } else { g.n_det }, 2));
                    }
                    None if !dd.view || sk("build") => {}
                    None => match dd.far {
                        0 => {
                            push((0, g.n_main, 0));
                            classes(&mut push, 1, k_mid, 1);
                            classes(&mut push, 2, k_fine, 2);
                        }
                        1 => {
                            classes(&mut push, 0, k_full, 3);
                            push((g.n_bld, g.n_main, 0));
                            classes(&mut push, 1, k_mid, 1);
                            classes(&mut push, 2, k_fine, 2);
                            classes(&mut push, 3, k_full, 4);
                            // Classes past their full geometry's reach: stand-ins only.
                            push((end(3, k_full), g.n_idx, 0));
                        }
                        _ => {
                            push((g.n_bld, g.n_main, 0));
                            push((g.n_far, g.n_idx, 0));
                        }
                    },
                }
                let runs = &runs[..nr];
                if runs.is_empty() {
                    continue;
                }
                rp.set_bind_group(1, &g.bind, &[i as u32 * 256]);
                rp.set_vertex_buffer(0, g.vbuf.slice(..));
                rp.set_index_buffer(g.ibuf.slice(..), wgpu::IndexFormat::Uint32);
                for r in runs.iter() {
                    if build_solid {
                        let s = solid_run(dd, *r);
                        if cur_solid != Some(s) {
                            rp.set_pipeline(if s { &self.p_build_solid[mi] } else { &self.p_build[mi] });
                            cur_solid = Some(s);
                        }
                    }
                    rp.draw_indexed(r.0..r.1, 0, r.2..r.2 + 1);
                }
            }
            if let Some(lib) = self.tree_lib.as_ref().filter(|_| shadow || !sk("trees")) {
                rp.set_vertex_buffer(0, lib.vbuf.slice(..));
                rp.set_index_buffer(lib.ibuf.slice(..), wgpu::IndexFormat::Uint32);
                // In view only the trees short of IMP_DIST are models (impostors follow below).
                // The second cascade draws whole batches with one model each.
                rp.set_pipeline(match c {
                    Some(1) => &self.p_tree_shadow_fixed,
                    Some(_) => &self.p_tree_shadow,
                    None => &self.p_tree_near[mi],
                });
                {
                    for (i, dd) in f.details.iter().take(n_builds).enumerate() {
                        let Some(tb) = &dd.gpu.trees else { continue };
                        let mut bound = false;
                        for t in &dd.trees {
                            if t.imp || if shadow { t.casc & bit == 0 } else { !t.view || !t.close } {
                                continue;
                            }
                            if !bound {
                                rp.set_bind_group(1, &self.bg_tileu, &[i as u32 * 256]);
                                rp.set_vertex_buffer(1, tb.slice(..));
                                bound = true;
                            }
                            let (first, count, base) = lib.ranges[t.mesh as usize];
                            rp.draw_indexed(first..first + count, base, t.first..t.first + t.count);
                        }
                    }
                }
                if !shadow {
                    rp.set_pipeline(&self.p_tree[(f.g.photo[2] <= 0.0 && crate::gpuprof::opt_on("photo_solid")) as usize][mi]);
                }
                let loose = c != Some(2) && fall_ranges.iter().any(|r| r.1 > r.0);
                if loose {
                    rp.set_bind_group(1, &self.bg_tileu, &[loose_slot]);
                    rp.set_vertex_buffer(1, self.fall_buf.slice(..));
                    for (k, r) in fall_ranges.iter().enumerate() {
                        if r.1 > r.0 {
                            let (first, count, base) = lib.ranges[k];
                            rp.draw_indexed(first..first + count, base, r.0..r.1);
                        }
                    }
                }
                // What stands close to the camera: undergrowth (which only casts
                // into the nearest cascade), then everyone's leaf cards.
                let near = |rp: &mut wgpu::RenderPass, cards: bool| {
                    for (i, dd) in f.details.iter().take(n_builds).enumerate() {
                        let mut bound = false;
                        for nd in dd.close.iter().filter(|nd| nd.cards == cards) {
                            let (first, count, base) = lib.ranges[nd.mesh as usize];
                            if count == 0 || nd.first + nd.count > n_near as u32 {
                                continue;
                            }
                            if !bound {
                                rp.set_bind_group(1, &self.bg_tileu, &[i as u32 * 256]);
                                rp.set_vertex_buffer(1, self.near_buf.slice(..));
                                bound = true;
                            }
                            rp.draw_indexed(first..first + count, base, nd.first..nd.first + nd.count);
                        }
                    }
                };
                if c.is_none() || c == Some(0) {
                    if !shadow {
                        rp.set_pipeline(&self.p_tree_near[mi]);
                    }
                    near(rp, false);
                }
                if c.is_none() {
                    rp.set_pipeline(&self.p_card[mi]);
                    near(rp, true);
                    // Everything further off: one picture each.
                    if self.imp.ready {
                        rp.set_pipeline(&self.imp.p_draw[mi]);
                        rp.set_bind_group(2, &self.imp.bg, &[]);
                        rp.set_index_buffer(self.imp.quad.slice(..), wgpu::IndexFormat::Uint32);
                        for (i, dd) in f.details.iter().take(n_builds).enumerate() {
                            let Some(tb) = &dd.gpu.trees else { continue };
                            let mut bound = false;
                            for t in dd.trees.iter().filter(|t| t.imp) {
                                if !bound {
                                    rp.set_bind_group(1, &self.bg_tileu, &[i as u32 * 256]);
                                    rp.set_vertex_buffer(0, tb.slice(..));
                                    bound = true;
                                }
                                rp.draw_indexed(0..6, t.mesh as i32 * 4, t.first..t.first + t.count);
                            }
                        }
                        rp.set_pipeline(&self.p_card[mi]);
                        rp.set_vertex_buffer(0, lib.vbuf.slice(..));
                        rp.set_index_buffer(lib.ibuf.slice(..), wgpu::IndexFormat::Uint32);
                    }
                    if loose {
                        rp.set_bind_group(1, &self.bg_tileu, &[loose_slot]);
                        rp.set_vertex_buffer(1, self.fall_buf.slice(..));
                        for (k, r) in fall_ranges.iter().enumerate() {
                            if r.1 > r.0 && k % crate::trees::LODS == 0 {
                                let (first, count, base) = lib.ranges[k + crate::trees::CARDS as usize];
                                rp.draw_indexed(first..first + count, base, r.0..r.1);
                            }
                        }
                    }
                }
            }
            if c == Some(2) {
                return;
            }
            if let Some(lib) = self.inst_lib.as_ref().filter(|_| shadow || !sk("inst")) {
                rp.set_pipeline(if shadow { &self.p_inst_shadow } else { &self.p_inst[(f.g.photo[2] <= 0.0 && crate::gpuprof::opt_on("photo_solid")) as usize][mi] });
                rp.set_vertex_buffer(0, lib.vbuf.slice(..));
                rp.set_vertex_buffer(1, self.inst_buf.slice(..));
                rp.set_index_buffer(lib.ibuf.slice(..), wgpu::IndexFormat::Uint32);
                for (k, r) in inst_ranges.iter().enumerate() {
                    // (rebar is too thin to cast a shadow worth its cost)
                    let thin = (crate::cars::M_REBAR..crate::cars::M_REBAR + crate::cars::N_REBAR).contains(&(k as u32));
                    if r.1 > r.0 && !(shadow && thin) {
                        let (first, count, base) = lib.ranges[k];
                        rp.draw_indexed(first..first + count, base, r.0..r.1);
                    }
                }
            }
            let mut cur: Option<bool> = None;
            for (i, o) in f.objs.iter().take(n_objs).enumerate() {
                if o.glass || if shadow { o.cast & bit == 0 } else { sk("objs") } {
                    continue;
                }
                if cur != Some(o.cut) {
                    rp.set_pipeline(match (shadow, o.cut) {
                        (true, true) => &self.p_obj_shadow,
                        (true, false) => &self.p_obj_shadow_solid,
                        (false, true) => &self.p_obj[mi],
                        (false, false) => &self.p_obj_solid[mi],
                    });
                    cur = Some(o.cut);
                }
                let m = &self.meshes[o.mesh];
                rp.set_bind_group(1, &self.bg_obj, &[i as u32 * 256]);
                rp.set_vertex_buffer(0, m.vbuf.slice(..));
                rp.set_index_buffer(m.ibuf.slice(..), wgpu::IndexFormat::Uint32);
                rp.draw_indexed(0..m.n_idx, 0, 0..1);
            }
        };
        // (first: the cascades leave out our scenery the mesh stands over)
        // Where the 3D tiles are, seen from above: our own scenery gives way there.
        if n_photo > 0 || self.photo_mask_dirty {
            let clear = wgpu::RenderPassColorAttachment { view: &self.photo_mask, depth_slice: None, resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.0, g: -1e4, b: 0.0, a: -1e4 }), store: wgpu::StoreOp::Store } };
            let mut rp = enc.begin_render_pass(&pd(&self.prof, &[Some(clear)], None, "photo mask"));
            if n_photo > 0 && f.g.photo[2] > 0.0 {
                rp.set_pipeline(&self.p_photo_mask);
                rp.set_bind_group(0, &self.bg0_pre, &[]);
                rp.set_bind_group(2, &self.bg_casc, &[MASK_SLOT as u32 * 256]);
                for (i, (pg, _)) in f.photo.iter().take(n_photo).enumerate() {
                    rp.set_bind_group(1, &pg.bind, &[i as u32 * 256]);
                    rp.set_vertex_buffer(0, pg.vbuf.slice(..));
                    rp.set_index_buffer(pg.ibuf.slice(..), wgpu::IndexFormat::Uint32);
                    rp.draw_indexed(0..pg.n_idx, 0, 0..1);
                }
                rp.set_pipeline(&self.p_water_top);
                rp.set_index_buffer(self.tile_ibuf.slice(..), wgpu::IndexFormat::Uint16);
                for (i, (tile, _)) in f.tiles.iter().take(n_tiles).enumerate() {
                    rp.set_bind_group(1, &tile.bind, &[i as u32 * 256]);
                    rp.set_vertex_buffer(0, tile.vbuf.slice(..));
                    rp.draw_indexed(0..self.tile_nidx, 0, 0..1);
                }
                // Street lamps, once the sun is low enough for them to matter.
                let sun_up = f.g.up[0] * f.g.sun_dir[0] + f.g.up[1] * f.g.sun_dir[1] + f.g.up[2] * f.g.sun_dir[2];
                if sun_up < 0.12 {
                    rp.set_pipeline(&self.p_lamp_top);
                    for (i, dd) in f.details.iter().take(n_builds).enumerate() {
                        if let (true, Some((vb, ib, n))) = (dd.view && dd.near, &dd.gpu.road) {
                            rp.set_bind_group(1, &self.bg_tileu, &[i as u32 * 256]);
                            rp.set_vertex_buffer(0, vb.slice(..));
                            rp.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                            rp.draw_indexed(0..*n, 0, 0..1);
                        }
                    }
                }
            }
            self.photo_mask_dirty = n_photo > 0;
        }
        // The shore distance field (shore.rs), a step a frame: our water from
        // above, and where there is ground at all, then the flood.
        if let Some(step) = f.shore {
            if let crate::shore::Step::Raster = step {
                // the square itself, and the wide one around it (for the exposure)
                for (view, slot, half) in [(&self.shore.mask, SHORE_SLOT, crate::shore::HALF), (&self.shore.wide, SHORE_WIDE_SLOT, crate::shore::WIDE_HALF)] {
                    let clear = wgpu::RenderPassColorAttachment { view, depth_slice: None, resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.0, g: -1e4, b: 0.0, a: 0.0 }), store: wgpu::StoreOp::Store } };
                    let mut rp = enc.begin_render_pass(&pd(&self.prof, &[Some(clear)], None, "shore"));
                    rp.set_bind_group(0, &self.bg0_pre, &[]);
                    rp.set_bind_group(2, &self.bg_casc, &[slot as u32 * 256]);
                    rp.set_index_buffer(self.tile_ibuf.slice(..), wgpu::IndexFormat::Uint16);
                    let c = glam::Vec3::from(f.shore_c);
                    let reach = half as f32 * 1.45;
                    let near = |u: &[f32; 12]| (glam::Vec3::new(u[0], u[1], u[2]) - c).length() - u[3] * crate::sources::GRID as f32 * 0.75 < reach;
                    let all = f.tiles.iter().take(n_tiles).chain(f.shore_tiles.iter().take(n_shore_tiles)).enumerate();
                    for p in [&self.p_shore_cover, &self.p_water_top] {
                        rp.set_pipeline(p);
                        for (i, (tile, u)) in all.clone() {
                            if near(u) {
                                rp.set_bind_group(1, &tile.bind, &[i as u32 * 256]);
                                rp.set_vertex_buffer(0, tile.vbuf.slice(..));
                                rp.draw_indexed(0..self.tile_nidx, 0, 0..1);
                            }
                        }
                    }
                }
            }
            self.shore.compute(q, &mut enc, step, [f.g.ocean[0], f.g.ocean[1]]);
        }
        // Sun shadows off: the cascades are not read, so not drawn (nor cleared).
        // Amortised ones keep what they hold on the frames they are not drawn.
        for c in (0..CASCADES).filter(|c| f.g.csm[3] > 0.0 && f.casc_draw & (1 << c) != 0) {
            let mut rp = enc.begin_render_pass(&pd(&self.prof, &[], Some(depth_att(&self.shadow[c], 1.0)), ["casc 0", "casc 1", "casc 2", "casc 3"][c.min(3)]));
            {
                rp.set_bind_group(0, &self.bg0_shadow, &[]);
                rp.set_bind_group(2, &self.bg_casc, &[c as u32 * 256]);
                solids(&mut rp, Some(c));
                // The 3D tiles cast their own shadows: those this cascade takes
                // in, and that the next finer one does not wholly cover (the
                // shaders read the finest there is, shadow_world).
                if n_photo > 0 && f.g.photo[2] > 0.0 && f.g.photo_sun[3] >= 0.0 {
                    let mat = |c: usize| {
                        let m = glam::Mat4::from_cols_array_2d(&f.casc_vp[c]);
                        (m, m.row(0).truncate().length(), m.row(1).truncate().length())
                    };
                    let (m, sx, sy) = mat(c);
                    let finer = (c > 0).then(|| mat(c - 1));
                    let zc = glam::Vec3::new(f.g.photo_c[0], f.g.photo_c[1], f.g.photo_c[2]);
                    // (fs_shadow stops casting at the middle of the zone's band)
                    let r_cast = f.g.photo[0] - 0.5 * f.g.photo[1];
                    let n_cut = (f.g.photo[3] as usize).min(crate::tiles3d::MAX_CUTS);
                    let mut fast: Option<bool> = None;
                    for (i, (pg, u)) in f.photo.iter().take(n_photo).enumerate() {
                        // Of a tile and the one it is dissolving into, the
                        // one more than half there casts (no dither here).
                        if u[1][0] < 0.5 {
                            continue;
                        }
                        let o = &u[0];
                        let p = glam::Vec3::new(o[0], o[1], o[2]);
                        let q = m * p.extend(1.0);
                        if q.x.abs() > 1.0 + pg.rad * sx || q.y.abs() > 1.0 + pg.rad * sy {
                            continue;
                        }
                        // (past 7% in from its edge the finer cascade is read alone)
                        if let Some((mf, fx, fy)) = finer {
                            let qf = mf * p.extend(1.0);
                            if qf.x.abs() + pg.rad * fx < 0.86 && qf.y.abs() + pg.rad * fy < 0.86 && qf.z > 0.0 && qf.z < 1.0 {
                                continue;
                            }
                        }
                        let dz = p.distance(zc);
                        if dz - pg.rad > r_cast {
                            continue;
                        }
                        // Depth only, unless something in the tile must not cast:
                        // the outer half of the band, a cut-out building. (Its
                        // still water casts: our water looks above it, composite.wgsl.)
                        let plain = dz + pg.rad < r_cast && (0..n_cut).all(|k| p.distance(glam::Vec3::from_slice(&f.g.cut[k * 3][..3])) > pg.rad + f.g.cut[k * 3][3]);
                        if fast != Some(plain) {
                            rp.set_pipeline(if plain { &self.p_photo_shadow_fast } else { &self.p_photo_shadow });
                            fast = Some(plain);
                        }
                        rp.set_bind_group(1, &pg.bind, &[i as u32 * 256]);
                        rp.set_vertex_buffer(0, pg.vbuf.slice(..));
                        rp.set_index_buffer(pg.ibuf.slice(..), wgpu::IndexFormat::Uint32);
                        rp.draw_indexed(0..pg.n_idx, 0, 0..1);
                    }
                }
            }
        }

        if f.g.pshadow[3] > 0.0 {
            // The aircraft's own fine shadow map (not read when off, so not drawn).
            let mut rp = enc.begin_render_pass(&pd(&self.prof, &[], Some(depth_att(&self.shadow[PLANE_LAYER], 1.0)), "plane shadow"));
            {
                let r = self.plane_shadow_res() as f32;
                rp.set_viewport(0.0, 0.0, r, r, 0.0, 1.0);
                rp.set_bind_group(0, &self.bg0_pre, &[]);
                rp.set_bind_group(2, &self.bg_casc, &[PLANE_LAYER as u32 * 256]);
                for (i, o) in f.objs.iter().take(n_objs).enumerate() {
                    if o.glass || o.cast & (1 << PLANE_LAYER) == 0 {
                        continue;
                    }
                    rp.set_pipeline(if o.cut { &self.p_obj_shadow } else { &self.p_obj_shadow_solid });
                    let m = &self.meshes[o.mesh];
                    rp.set_bind_group(1, &self.bg_obj, &[i as u32 * 256]);
                    rp.set_vertex_buffer(0, m.vbuf.slice(..));
                    rp.set_index_buffer(m.ibuf.slice(..), wgpu::IndexFormat::Uint32);
                    rp.draw_indexed(0..m.n_idx, 0, 0..1);
                }
            }
        }
        {
            // The composite reads original samples at water/foreground boundaries.
            let att = |v, r| wgpu::RenderPassColorAttachment { view: v, depth_slice: None, resolve_target: Some(r), ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store } };
            // (MSAA off: straight into the textures the passes after read)
            let ms = |v, r| if self.msaa { att(v, r) } else { color_att(r, true) };
            let mut rp = enc.begin_render_pass(&pd(&self.prof, &[Some(ms(&t.color_ms, &t.color)), Some(ms(&t.aux_ms, &t.aux)), Some(ms(&t.amb_ms, &t.amb))], Some(depth_att(if self.msaa { &t.depth_ms } else { &t.depth }, 0.0)), "gbuffer"));
            rp.set_bind_group(0, &self.bg0, &[]);
            // The 3D tiles first: up close they hide most of what follows.
            if n_photo > 0 && !sk("photo") {
                rp.set_pipeline(&self.p_photo[mi]);
                for (i, (pg, _)) in f.photo.iter().take(n_photo).enumerate() {
                    rp.set_bind_group(1, &pg.bind, &[i as u32 * 256]);
                    rp.set_vertex_buffer(0, pg.vbuf.slice(..));
                    rp.set_index_buffer(pg.ibuf.slice(..), wgpu::IndexFormat::Uint32);
                    rp.draw_indexed(0..pg.n_idx, 0, 0..1);
                }
            }
            // Destroyed buildings: their own mesh, on the pieces they came down in.
            if !f.skins.is_empty() {
                rp.set_pipeline(&self.p_skin[mi]);
                for (sk, first) in &f.skins {
                    if (*first as usize) + SKIN_SLOTS > SKIN_SLOT_BUF + SKIN_SLOTS || (*first as usize) >= n_slots.max(1) {
                        continue;
                    }
                    rp.set_bind_group(1, &sk.bind, &[first * 64]);
                    rp.set_vertex_buffer(0, sk.vbuf.slice(..));
                    rp.draw(0..sk.n, 0..1);
                }
            }
            // No 3D tiles zone: the terrain without its discard for it.
            let ti = (f.g.photo[2] <= 0.0 && crate::gpuprof::opt_on("photo_solid")) as usize;
            rp.set_pipeline(&self.p_terrain[ti][mi]);
            rp.set_index_buffer(self.tile_ibuf.slice(..), wgpu::IndexFormat::Uint16);
            let tess = |u: &[f32; 12]| u[10] >= 1.0 && u[11] >= 1.0;
            for (i, (tile, u)) in f.tiles.iter().take(n_tiles).enumerate().filter(|_| !sk("terrain")) {
                rp.set_bind_group(1, &tile.bind, &[i as u32 * 256]);
                rp.set_vertex_buffer(0, tile.vbuf.slice(..));
                // A tessellated tile only takes its skirts from here.
                let first = if tess(u) { (crate::sources::GRID * crate::sources::GRID * 6) as u32 } else { 0 };
                rp.draw_indexed(first..self.tile_nidx, 0, 0..1);
            }
            for (i, (tile, u)) in f.tiles.iter().take(n_tiles).enumerate() {
                if !tess(u) || sk("tess") {
                    continue;
                }
                rp.set_bind_group(1, &tile.bind, &[i as u32 * 256]);
                rp.set_pipeline(&self.p_terrain_cells[ti][mi]);
                rp.set_index_buffer(self.cell_ibuf.slice(..), wgpu::IndexFormat::Uint16);
                rp.draw_indexed(0..6, 0, 0..(crate::sources::GRID * crate::sources::GRID) as u32);
                rp.set_pipeline(&self.p_terrain_tess[ti][mi]);
                rp.set_index_buffer(self.tess_ibuf.slice(..), wgpu::IndexFormat::Uint16);
                rp.draw_indexed(0..self.tess_nidx, 0, 0..(u[10] * u[11]) as u32);
            }
            // Water High: the water of the tiles near the camera as fine grids
            // the waves stand up on (terrain.wgsl vs_water).
            if !sk("water_grid") {
                let mut set = false;
                for (i, (tile, u)) in f.tiles.iter().take(n_tiles).enumerate() {
                    let Some(g) = WATER_GRIDS.iter().position(|&n| n as f32 == u[5]) else { continue };
                    if !set {
                        rp.set_pipeline(&self.p_water[mi]);
                        set = true;
                    }
                    rp.set_bind_group(1, &tile.bind, &[i as u32 * 256]);
                    rp.set_index_buffer(self.water_ibuf[g].0.slice(..), wgpu::IndexFormat::Uint32);
                    rp.draw_indexed(0..self.water_ibuf[g].1, 0, 0..1);
                }
            }
            // Airfield pavement, then roads, straight onto the terrain.
            rp.set_pipeline(&self.p_field[mi]);
            for (i, (fg, _)) in f.fields.iter().take(n_fields).enumerate() {
                if fg.n_idx > 0 && !sk("field") {
                    rp.set_bind_group(1, &self.bg_tileu, &[field_slot(i)]);
                    rp.set_vertex_buffer(0, fg.vbuf.slice(..));
                    rp.set_index_buffer(fg.ibuf.slice(..), wgpu::IndexFormat::Uint32);
                    rp.draw_indexed(0..fg.n_idx, 0, 0..1);
                }
            }
            rp.set_pipeline(&self.p_road[mi]);
            for (i, dd) in f.details.iter().take(n_builds).enumerate().filter(|_| !sk("road")) {
                if let (true, Some((vb, ib, n))) = (dd.view && dd.near, &dd.gpu.road) {
                    rp.set_bind_group(1, &self.bg_tileu, &[i as u32 * 256]);
                    rp.set_vertex_buffer(0, vb.slice(..));
                    rp.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                    rp.draw_indexed(0..*n, 0, 0..1);
                }
            }
            solids(&mut rp, None);
        }
        if let Some(bg_dres) = &t.bg_dres {
            let mut rp = enc.begin_render_pass(&pd(&self.prof, &[], Some(depth_att(&t.depth, 0.0)), "depth resolve"));
            rp.set_pipeline(&self.p_dres);
            rp.set_bind_group(0, bg_dres, &[]);
            rp.draw(0..3, 0..1);
        }
        let full = |enc: &mut wgpu::CommandEncoder, target: &wgpu::TextureView, clear: bool, p: &wgpu::RenderPipeline, bg1: &wgpu::BindGroup, label: &'static str| {
            let mut rp = enc.begin_render_pass(&pd(&self.prof, &[Some(color_att(target, clear))], None, label));
            rp.set_pipeline(p);
            rp.set_bind_group(0, &self.bg0, &[]);
            rp.set_bind_group(1, bg1, &[]);
            rp.draw(0..3, 0..1);
        };
        if self.ao {
            full(&mut enc, &t.ao_raw, true, &self.p_gtao, &t.bg_gtao, "gtao");
            full(&mut enc, &t.ao, true, &self.p_ao_blur, &t.bg_ao_blur, "ao blur");
        } else {
            // Ambient occlusion off: the composite reads "nothing occluded".
            let att = wgpu::RenderPassColorAttachment { view: &t.ao, depth_slice: None, resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::WHITE), store: wgpu::StoreOp::Store } };
            drop(enc.begin_render_pass(&pd(&self.prof, &[Some(att)], None, "ao clear")));
        }
        if crate::gpuprof::opt_on("comp_split") {
            full(&mut enc, &t.hdr, true, &self.p_comp_land[mi], &t.bg_composite, "composite");
            full(&mut enc, &t.hdr, false, &self.p_comp_water[mi], &t.bg_composite, "composite water");
            if self.msaa {
                full(&mut enc, &t.hdr, false, &self.p_comp_edge, &t.bg_composite, "composite edge");
            }
        } else {
            full(&mut enc, &t.hdr, true, &self.p_composite[mi], &t.bg_composite, "composite");
        }
        // Lights over the lit scene, under the clouds.
        if n_fields > 0 || n_dyn > 0 {
            let ds = wgpu::RenderPassDepthStencilAttachment { view: &t.depth, depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store }), stencil_ops: None };
            let mut rp = enc.begin_render_pass(&pd(&self.prof, &[Some(color_att(&t.hdr, false))], Some(ds), "lights"));
            rp.set_bind_group(0, &self.bg0, &[]);
            rp.set_pipeline(&self.p_lights);
            for (i, (fg, _)) in f.fields.iter().take(n_fields).enumerate() {
                if let Some((buf, n)) = &fg.lights {
                    rp.set_bind_group(1, &self.bg_tileu, &[field_slot(i)]);
                    rp.set_vertex_buffer(0, buf.slice(..));
                    rp.draw(0..6, 0..*n);
                }
            }
            if n_dyn > 0 {
                rp.set_bind_group(1, &self.bg_tileu, &[loose_slot]);
                rp.set_vertex_buffer(0, self.dyn_lights.slice(..));
                rp.draw(0..6, 0..n_dyn as u32);
            }
        }
        {
            let mut rp = enc.begin_render_pass(&pd(&self.prof, &[Some(color_att(&t.cloud, true)), Some(color_att(&t.cloud_dist, true))], None, "cloud march"));
            rp.set_pipeline(&self.p_cloud);
            rp.set_bind_group(0, &self.bg0, &[]);
            rp.set_bind_group(1, &t.bg_cloud, &[]);
            rp.draw(0..3, 0..1);
        }
        let cf = self.cloud_flip;
        full(&mut enc, &t.cloud_hist[cf], true, &self.p_cloud_temporal, &t.bg_cloud_temporal[cf], "cloud temporal");
        {
            let mut rp = enc.begin_render_pass(&pd(&self.prof, &[Some(color_att(&t.hdr, false))], None, "cloud apply"));
            rp.set_bind_group(0, &self.bg0, &[]);
            rp.set_pipeline(&self.p_cloud_apply);
            rp.set_bind_group(1, &t.bg_cloud_apply[cf], &[]);
            rp.draw(0..3, 0..1);
        }
        // FSR's reactive mask: the scene before the particles and the fire, to
        // compare with after (fs_reactive). (Not the glass: in the cockpit it
        // covers most of the view, which would then go without the history.)
        let reactive = fsr && (parts || self.fluid.live);
        if let Some(ft) = t.fsr.as_ref().filter(|_| reactive) {
            let (w, h) = self.render_size();
            enc.copy_texture_to_texture(t.hdr_tex.as_image_copy(), ft.opaque.as_image_copy(), wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 });
        }
        if parts {
            let mut rp = enc.begin_render_pass(&pd(&self.prof, &[Some(color_att(&t.hdr, false))], None, "particles"));
            rp.set_bind_group(0, &self.bg0, &[]);
            rp.set_pipeline(&self.p_particles);
            rp.set_bind_group(1, &t.bg_particles, &[]);
            // Ring order = age order, so blending stays stable without sorting.
            rp.draw(0..6, self.phead..MAX_PARTICLES);
            if self.phead > 0 {
                rp.draw(0..6, 0..self.phead);
            }
        }
        // The fire is drawn over the particles: sparks and leaves behind or
        // inside it are hidden by it (those in front are dimmed with them).
        if self.fluid.live {
            let ft = &self.fluid_t;
            {
                let mut rp = enc.begin_render_pass(&pd(&self.prof, &[Some(color_att(&ft.half, true)), Some(color_att(&ft.half_z, true))], None, "fire march"));
                rp.set_bind_group(0, &self.bg0, &[]);
                rp.set_pipeline(&self.fluid.p_draw);
                rp.set_bind_group(1, &ft.bg_march, &[]);
                rp.draw(0..3, 0..1);
            }
            let mut rp = enc.begin_render_pass(&pd(&self.prof, &[Some(color_att(&t.hdr, false))], None, "fire up"));
            rp.set_bind_group(0, &self.bg0, &[]);
            rp.set_pipeline(&self.fluid.p_up);
            rp.set_bind_group(1, &ft.bg_up, &[]);
            rp.draw(0..3, 0..1);
        }
        if let Some(ft) = t.fsr.as_ref().filter(|_| reactive) {
            full(&mut enc, &view(&ft.reactive), true, &self.p_reactive, &ft.bg_reactive, "reactive");
        }
        if f.objs.iter().take(n_objs).any(|o| o.glass) {
            // The windows bend what is seen through them, so they read the
            // scene from a copy. The depth stays read-only: it is sampled too.
            let (w, h) = self.render_size();
            enc.copy_texture_to_texture(t.hdr_tex.as_image_copy(), t.scene_tex.as_image_copy(), wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 });
            let ds = wgpu::RenderPassDepthStencilAttachment { view: &t.depth, depth_ops: None, stencil_ops: None };
            let mut rp = enc.begin_render_pass(&pd(&self.prof, &[Some(color_att(&t.hdr, false))], Some(ds), "glass"));
            rp.set_bind_group(0, &self.bg0, &[]);
            rp.set_bind_group(2, &t.bg_glass, &[]);
            rp.set_pipeline(&self.p_glass);
            for (i, o) in f.objs.iter().take(n_objs).enumerate() {
                if !o.glass {
                    continue;
                }
                let m = &self.meshes[o.mesh];
                rp.set_bind_group(1, &self.bg_obj, &[i as u32 * 256]);
                rp.set_vertex_buffer(0, m.vbuf.slice(..));
                rp.set_index_buffer(m.ibuf.slice(..), wgpu::IndexFormat::Uint32);
                rp.draw_indexed(0..m.n_idx, 0, 0..1);
            }
        }
        // The finished frame: `hdr` itself, or blended into the frames before it.
        let src = if let (Some(ft), Some(s)) = (t.fsr.as_ref().filter(|_| fsr), self.fsr.as_mut()) {
            // FSR: from the scene, its depth and how each pixel moved (fs_motion), into the first `taa`.
            full(&mut enc, &view(&ft.motion), true, &self.p_motion, &t.bg_taa[0], "motion");
            let (w, h) = ((t.hdr_tex.width()), t.hdr_tex.height());
            Self::fsr_dispatch(s, &self.queue, &self.prof, t, (w, h), [self.config.width, self.config.height], &mut enc, &f.g, f.cam, reactive);
            1
        } else if taa {
            full(&mut enc, &t.taa[self.taa_flip], true, if taau { &self.p_taa_up } else { &self.p_taa }, &t.bg_taa[self.taa_flip], "taa");
            1 + self.taa_flip
        } else {
            0
        };
        // (bloom off: post.wgsl `finish` reads none of it, G.start.z)
        for i in (0..BLOOM_MIPS as usize).filter(|_| self.bloom) {
            // (upscaled: from the scene itself, the size the bloom is made for)
            full(&mut enc, &t.bloom[i], true, &self.p_down, if i == 0 { &t.bg_down0[if taau || fsr { 0 } else { src }] } else { &t.bg_down[i - 1] }, "bloom");
        }
        for i in (0..BLOOM_MIPS as usize - 1).rev().filter(|_| self.bloom) {
            full(&mut enc, &t.bloom[i], false, &self.p_up, &t.bg_up[i], "bloom");
        }
        let p_taa = if taau { &self.p_final_taau } else if self.scale < 1.0 { &self.p_final_taa_up } else { &self.p_final_taa };
        // (FSR sharpens its own result)
        full(&mut enc, &t.ldr, true, if taa && !fsr { p_taa } else { &self.p_final }, &t.bg_final[src], "final");

        let egui_jobs = f.egui.map(|(jobs, delta, ppp)| {
            let sd = egui_wgpu::ScreenDescriptor { size_in_pixels: [self.config.width, self.config.height], pixels_per_point: ppp };
            self.egui.update_buffers(&self.device, &self.queue, &mut enc, &jobs, &sd);
            (jobs, delta, sd)
        });
        if let Some((jobs, _, sd)) = &egui_jobs {
            let mut rp = enc.begin_render_pass(&pd(&self.prof, &[Some(color_att(&t.ldr, false))], None, "ui")).forget_lifetime();
            self.egui.render(&mut rp, jobs, sd);
        }
        if let Some(scale) = f.logo {
            let (w, h) = (self.config.width, self.config.height);
            if self.logo_key != Some((w, h, scale.to_bits())) {
                let m = self.logo_font.as_deref().map(|font| crate::logo::mask(font, w, h, scale)).unwrap_or_else(|| vec![0; (w * h) as usize]);
                self.queue.write_texture(t.logo_tex.as_image_copy(), &m, wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(w), rows_per_image: Some(h) }, wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 });
                self.logo_key = Some((w, h, scale.to_bits()));
            }
            full(&mut enc, &t.ldr, false, &self.p_logo, &t.bg_logo, "logo");
        }
        if let Some(sview) = &sview {
            full(&mut enc, sview, true, &self.p_blit, &t.bg_blit, "blit");
        }
        let shot = f.shot.map(|path| {
            let (w, h) = (self.config.width, self.config.height);
            let row = (w * 4).div_ceil(256) * 256;
            let buf = self.device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (row * h) as u64, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
            enc.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo { texture: &t.ldr_tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                wgpu::TexelCopyBufferInfo { buffer: &buf, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: Some(h) } },
                wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            );
            (path, buf, w, h, row)
        });
        let t_sub = std::time::Instant::now();
        self.stats[2] += (t_sub - t_enc).as_secs_f64();
        if let Some(p) = &self.prof {
            p.resolve(&mut enc);
        }
        self.queue.submit([enc.finish()]);
        self.fluid.after_submit();
        if let Some(p) = &mut self.prof {
            p.after_submit();
        }
        match frame {
            Some(frame) => self.queue.present(frame),
            // Nothing paces a frame that is not shown: wait for the GPU to finish it.
            None => {
                let _ = self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
            }
        }
        self.stats[3] += t_sub.elapsed().as_secs_f64();
        if let Some((_, delta, _)) = egui_jobs {
            for id in &delta.free {
                self.egui.free_texture(id);
            }
        }
        if let Some((path, buf, w, h, row)) = shot {
            buf.slice(..).map_async(wgpu::MapMode::Read, |_| {});
            let _ = self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
            let data = buf.slice(..).get_mapped_range().expect("mapped");
            let mut img = image::RgbaImage::new(w, h);
            for y in 0..h as usize {
                img.as_flat_samples_mut().samples[y * w as usize * 4..(y + 1) * w as usize * 4].copy_from_slice(&data[y * row as usize..y * row as usize + w as usize * 4]);
            }
            match image::DynamicImage::ImageRgba8(img).to_rgb8().save(&path) {
                Ok(_) => log::info!("saved screenshot {}", path.display()),
                Err(e) => log::error!("screenshot failed: {e}"),
            }
        }
    }
}

fn color_att(view: &wgpu::TextureView, clear: bool) -> wgpu::RenderPassColorAttachment<'_> {
    wgpu::RenderPassColorAttachment {
        view,
        depth_slice: None,
        resolve_target: None,
        ops: wgpu::Operations { load: if clear { wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT) } else { wgpu::LoadOp::Load }, store: wgpu::StoreOp::Store },
    }
}
fn depth_att(view: &wgpu::TextureView, clear: f32) -> wgpu::RenderPassDepthStencilAttachment<'_> {
    wgpu::RenderPassDepthStencilAttachment { view, depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(clear), store: wgpu::StoreOp::Store }), stencil_ops: None }
}
/// `pass_desc`, timed as `label` under --gpu-profile.
fn pd<'a>(prof: &'a Option<crate::gpuprof::GpuProf>, colors: &'a [Option<wgpu::RenderPassColorAttachment<'a>>], depth: Option<wgpu::RenderPassDepthStencilAttachment<'a>>, label: &'static str) -> wgpu::RenderPassDescriptor<'a> {
    wgpu::RenderPassDescriptor { timestamp_writes: prof.as_ref().and_then(|p| p.render(label)), ..pass_desc(colors, depth) }
}

fn pass_desc<'a>(colors: &'a [Option<wgpu::RenderPassColorAttachment<'a>>], depth: Option<wgpu::RenderPassDepthStencilAttachment<'a>>) -> wgpu::RenderPassDescriptor<'a> {
    wgpu::RenderPassDescriptor { label: None, color_attachments: colors, depth_stencil_attachment: depth, timestamp_writes: None, occlusion_query_set: None, multiview_mask: None }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        self.fluid.shutdown(&self.device);
    }
}
