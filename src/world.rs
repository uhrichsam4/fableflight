//! Streaming world: a quadtree of terrain tiles plus OSM building tiles,
//! loaded around the camera, with ground/obstacle queries for the physics.
use crate::buildings::{self, Footprint, Parked, TreeChunk};
use crate::cars::{self, Car};
use crate::geo::{self, TileId};
use crate::render::{BuildGpu, DetailDraw, Inst, NearDraw, Renderer, TileGpu, TreeDraw, TreeInst, MAX_NEAR};
use crate::roads::RoadNet;
use crate::trees::{lod_span, tree_jit, NearCell, TreeDim, Under, CARDS, CARD_RANGE, IMP_DIST, LODS, LOD_DIST, LOD_JITTER, TREE_SPECIES, VARIANTS};
use crate::sources::{Job, JobResult, Loader, DATA_ZOOM, GRID, GV};
use glam::{DMat4, DVec3, DVec4, Vec3};
use std::collections::{HashMap, VecDeque};

struct Tile {
    center: DVec3,
    radius: f64,
    gpu: TileGpu,
    heights: Vec<f32>,
    water: Vec<bool>,
    /// any of it
    has_water: bool,
    /// how much of the ocean's two longer wave cascades the water shows (sources.rs TileData::sea)
    sea: Vec<[f32; 2]>,
    last_used: u64,
    /// imagery source it was made with; a different one gets it made again
    generation: u32,
    refreshing: bool,
}
enum Slot {
    Loading,
    Ready(Box<Tile>),
    Failed,
}
struct BTile {
    center: DVec3,
    radius: f64,
    gpu: Option<BuildGpu>,
    footprints: Vec<Footprint>,
    last_used: u64,
    net: RoadNet,
    cars: Vec<Car>,
    parked: Vec<Parked>,
    trees: Vec<TreeInst>,
    tree_var: Vec<u8>,
    chunks: Vec<TreeChunk>,
    under: Vec<Under>,
    cells: Vec<NearCell>,
    near_idx: Vec<u32>,
    ax: [f32; 4],
    ay: [f32; 4],
    /// seconds the tile has been shown, and has been close enough for the
    /// middle and the finest level of detail (for fading them in)
    shown: f32,
    mid_t: f32,
    fine_t: f32,
    /// distance scale of its tallest height class (buildings::HCLASS)
    tall: f32,
}
/// An airfield near enough to be drawn and to be busy.
pub struct Field {
    pub geo: std::sync::Arc<crate::airports::Geo>,
    gpu: crate::render::FieldGpu,
    props: Vec<crate::airfield::Prop>,
    socks: Vec<Vec3>,
    pub traffic: crate::traffic::Traffic,
    last_used: u64,
}
enum FSlot {
    Loading,
    Ready(Box<Field>),
    Failed,
}
enum BSlot {
    Loading,
    Ready(Box<BTile>),
}
/// A building tile whose geometry is still being copied to the GPU.
struct Upload {
    id: TileId,
    data: Box<buildings::BuildingData>,
    gpu: BuildGpu,
    /// next buffer of `BuildGpu::targets` to fill, and how far into it
    part: usize,
    off: usize,
}
/// Bytes copied to the GPU per frame at most (plus one terrain tile or
/// atlas). Everything written sits in wgpu staging memory until the GPU has
/// consumed it, a few frames later; on cards without resizable BAR that is a
/// 246 MiB heap, and overflowing it slows every upload to a crawl.
const UPLOAD_BUDGET: usize = 24 << 20;
/// Largest single write into a building buffer.
const UPLOAD_CHUNK: usize = 8 << 20;

pub struct Ground {
    pub h: f64,
    pub water: bool,
}

/// A building or bridge deck over a point.
pub struct BHit {
    pub base: f64,
    pub top: f64,
    pub bridge: bool,
    pub tile: TileId,
    pub idx: usize,
    /// footprint area (m²)
    pub area: f32,
}

/// A standing tree, as the physics sees it.
#[derive(Clone)]
pub struct TreeRef {
    pub tile: TileId,
    pub idx: usize,
    pub pos: DVec3,
    pub height: f64,
    pub crown_r: f64,
    pub crown_base: f64,
    pub trunk_r: f64,
    pub variant: u8,
    pub inst: TreeInst,
}

/// What is left to animate when a building is knocked down.
pub struct Wreck {
    pub ring: Vec<DVec3>,
    pub base: f64,
    pub top: f64,
    pub col: [u8; 3],
    pub tile: TileId,
    /// how its walls were drawn (see `Footprint`)
    pub sty: [u8; 4],
    pub gref: f64,
    pub roof_col: [u8; 3],
    pub tiers: Vec<(f32, f32)>,
}

/// Shape of a crater as a function of distance / radius: height change in
/// units of the crater depth. Must match `crater_field` in the shaders.
/// The block of grid cells (x0, y0, w, h) of tile `id` that is drawn
/// tessellated: everything close enough to the camera for an edge to be split
/// at least once (terrain.wgsl: closer than cell / (2 TESS_K)), plus a margin.
/// Empty when the camera is too high or too far away.
fn tess_block(id: TileId, lat: f64, lon: f64, agl: f64, cell: f64) -> [f32; 4] {
    const TESS_K: f64 = 0.03;
    let r = cell / (2.0 * TESS_K) + 1.5 * cell;
    if agl >= r || agl.is_nan() {
        return [0.0; 4];
    }
    let rh = (r * r - agl.max(0.0).powi(2)).sqrt() / cell;
    let (u, v) = id.frac(lat, lon);
    let (cx, cy) = (u * GRID as f64, v * GRID as f64);
    let span = |c: f64| ((c - rh).floor().clamp(0.0, GRID as f64), (c + rh).ceil().clamp(0.0, GRID as f64));
    let ((x0, x1), (y0, y1)) = (span(cx), span(cy));
    if x1 <= x0 || y1 <= y0 {
        return [0.0; 4];
    }
    [x0 as f32, y0 as f32, (x1 - x0) as f32, (y1 - y0) as f32]
}

/// The fine water grid for a tile of width `w` (m) whose nearest point is
/// `near` metres off (render.rs `WATER_GRIDS`), 0 for none: every tile with
/// water nearer than the waves reach (terrain.wgsl `DENSE_FAR`), with
/// vertices a fraction of their distance apart.
fn water_grid(w: f64, near: f64, agl: f64) -> f32 {
    let near = near.max(agl).max(0.0);
    if near > 8200.0 || agl > 3000.0 || !crate::gpuprof::opt_on("water_grid") {
        return 0.0;
    }
    let want = w / (near * 0.004).max(0.5);
    let g = crate::render::WATER_GRIDS;
    (if want > g[0] as f64 * 1.4 { g[1] } else { g[0] }) as f32
}

pub fn crater_profile(d: f64) -> f64 {
    let k = (d - 1.1) / 0.24;
    let rim = 0.22 * (-k * k).exp();
    if d < 1.0 { rim - (1.0 - d * d).powi(2) } else { rim }
}

/// Distance (beyond a tile's bounding sphere) within which its middle and
/// finest levels of detail are drawn. The middle level goes with the full
/// geometry it dresses, which gives way to the far stand-ins by BFAR[1].
/// Intensities of a car's head and tail lights where the 3D tiles hide our
/// cars (lights.wgsl units; an airliner's nav light is 40).
const CAR_HEAD: f32 = 60.0;
const CAR_TAIL: f32 = 12.0;
/// How far off moving cars are drawn (m).
const CAR_FAR: f64 = 2200.0;

const LOD_MID: f32 = buildings::BFAR[1];
const LOD_FINE: f32 = 700.0;

pub struct World {
    /// flat ground everywhere tiles have not loaded, at this height (for tests)
    pub flat: Option<f64>,
    loader: Loader,
    tiles: HashMap<TileId, Slot>,
    builds: HashMap<TileId, BSlot>,
    frame: u64,
    in_flight: usize,
    pub max_zoom: u8,
    pub lod_k: f64,
    pub buildings_on: bool,
    draw: Vec<TileId>,
    /// selected tiles out of view but near the camera: for the shore masks (shore.rs)
    draw_shore: Vec<TileId>,
    draw_b: Vec<TileId>,
    wanted: Vec<(f32, TileId)>,
    in_view: Vec<bool>,
    frustum: Option<Frustum>,
    /// (centre, radius) of the most recent craters
    pub craters: Vec<(DVec3, f32)>,
    pub tree_dims: Vec<TreeDim>,
    tick: u32,
    pub stats: (usize, usize, usize), // drawn tiles, loading, building tiles
    last_update: Option<std::time::Instant>,
    uploads: VecDeque<Upload>,
    fields: HashMap<u32, FSlot>,
    /// photorealistic 3D tiles near the aircraft (taken out while it updates)
    photo: Option<Box<crate::tiles3d::Photo>>,
    /// the waves (their CPU copy, so what floats rides what is drawn)
    pub ocean: crate::ocean::Ocean,
    /// Water High: the water near the camera drawn as fine grids the waves
    /// stand up on (terrain.wgsl `vs_water`)
    pub dense_water: bool,
}

#[derive(Clone)]
pub struct Frustum([DVec4; 4]);
impl Frustum {
    /// From a camera-relative view-projection matrix.
    pub fn new(vp: DMat4) -> Self {
        let r = |i: usize| DVec4::new(vp.x_axis[i], vp.y_axis[i], vp.z_axis[i], vp.w_axis[i]);
        let norm = |p: DVec4| p / p.truncate().length();
        Self([norm(r(3) + r(0)), norm(r(3) - r(0)), norm(r(3) + r(1)), norm(r(3) - r(1))])
    }
    pub fn sphere(&self, c: DVec3, r: f64) -> bool {
        self.0.iter().all(|p| p.truncate().dot(c) + p.w > -r)
    }
}

impl World {
    pub fn new(loader: Loader, max_zoom: u8) -> Self {
        let photo = Some(Box::new(crate::tiles3d::Photo::new(loader.src.google.clone(), Some(loader.src.clone()))));
        Self { photo, flat: None, loader, tiles: HashMap::new(), builds: HashMap::new(), frame: 0, in_flight: 0, max_zoom, lod_k: 0.62, buildings_on: true, draw: Vec::new(), draw_shore: Vec::new(), draw_b: Vec::new(), wanted: Vec::new(), in_view: Vec::new(), frustum: None, craters: Vec::new(), tree_dims: Vec::new(), tick: 0, stats: (0, 0, 0), last_update: None, uploads: VecDeque::new(), fields: HashMap::new(), ocean: crate::ocean::Ocean::new(), dense_water: false }
    }

    /// Height of the ground (or water surface) at a lat/lon: the top of the
    /// photorealistic 3D mesh inside its zone (buildings included), our own
    /// terrain everywhere else.
    pub fn ground(&self, lat: f64, lon: f64) -> Option<Ground> {
        let g = self.terrain_ground(lat, lon);
        if let Some(mut h) = self.photo.as_deref().and_then(|p| p.ground(lat, lon, g.as_ref().map(|g| g.h))) {
            // Our water (the mesh's is a still photograph): ditching floats on it.
            let water = g.as_ref().is_some_and(|g| g.water && h < g.h + 1.5);
            if water {
                h = h.max(g.as_ref().map_or(h, |g| g.h));
            }
            return Some(Ground { h, water });
        }
        g
    }

    /// Obstruction lights on the 3D tiles' tall buildings, if they are on.
    pub fn photo_beacons(&self, cam: DVec3, out: &mut Vec<crate::airfield::Light>) {
        if let Some(p) = self.photo.as_deref() {
            p.beacons(cam, out);
        }
    }

    /// Head and tail lights of one of our moving cars (number `i` of tile
    /// `b`, at `pos`, `dir` its heading) where the 3D tiles hide the car
    /// itself, its own lamps with it: sprites on the mesh's street, so the
    /// photographed city has traffic after dark. `streets`: the mesh's tiles
    /// round `b`, gathered when a car first needs them.
    #[allow(clippy::too_many_arguments)]
    fn photo_car_light<'a>(&self, photo: &'a crate::tiles3d::Photo, streets: &mut Option<Option<crate::tiles3d::Streets<'a>>>, b: &BTile, i: usize, c: &Car, pos: DVec3, dir: Vec3, cam: DVec3, out: &mut Vec<crate::airfield::Light>) {
        use crate::airfield::Light;
        // Cars are drawn out to 2.2 km; their lights fade out before that,
        // so the traffic has no edge.
        const FADE: f64 = 600.0;
        let d = (pos - cam).length();
        // Room left for the aircraft and the obstruction lights.
        if out.len() + 4 > crate::render::MAX_DYN_LIGHTS - 1024 || !self.frustum.as_ref().map(|f| f.sphere(pos - cam, 8.0)).unwrap_or(true) {
            return;
        }
        let road = pos.length() - geo::R;
        // The street under a car changes slowly: it is looked up every 8th
        // frame (the cars in turn), or when the car comes into view.
        let frame = self.frame as u32;
        let (dh, at) = c.street.get();
        let street = if at != u32::MAX && frame.wrapping_sub(at) < 8 && (frame as usize + i) % 8 != 0 {
            road + dh as f64
        } else {
            let Some(st) = streets.get_or_insert_with(|| photo.streets(b.center, b.radius + 20.0)).as_ref().and_then(|s| s.at(pos, road)) else {
                c.street.set((0.0, u32::MAX));
                return;
            };
            c.street.set(((st - road) as f32, frame));
            st
        };
        let k = ((CAR_FAR - d) / FADE).clamp(0.0, 1.0) as f32;
        let n = pos.normalize();
        let base = (n * (geo::R + street) - cam).as_vec3();
        let u = n.as_vec3();
        let f = (dir - u * dir.dot(u)).normalize_or_zero();
        let r = f.cross(u);
        let len = cars::CAR_LEN[c.model as usize];
        let half_w = if len > 7.0 { 1.0 } else { 0.62 };
        // Seen from above or ahead (headlights) or behind (tail lights), not
        // from the other end. Far off the pair is one light: one sprite each
        // end, twice as bright.
        let mut lamp = |along: f32, lift: f32, inten: f32, col: [u8; 3], aim: Vec3| {
            let l = |side: f32, i: f32| Light {
                pos: (base + f * along + r * side + u * lift).into(),
                inten: i * k,
                dir: (aim - u * 0.15).normalize().into(),
                spread: -0.4,
                col: [col[0], col[1], col[2], 0],
                fx: [0, 0, 255, 0],
            };
            if d > 500.0 {
                out.push(l(0.0, inten * 2.0));
            } else {
                out.push(l(-half_w, inten));
                out.push(l(half_w, inten));
            }
        };
        lamp(len * 0.5 - 0.1, 0.7, CAR_HEAD, crate::airfield::L_WHITE, f);
        lamp(-len * 0.5 + 0.1, 0.85, CAR_TAIL, [255, 20, 12], -f);
    }

    /// The photorealistic 3D tiles.
    pub fn photo(&self) -> &crate::tiles3d::Photo {
        self.photo.as_deref().expect("photo tiles")
    }
    pub fn photo_mut(&mut self) -> &mut crate::tiles3d::Photo {
        self.photo.as_deref_mut().expect("photo tiles")
    }

    /// Under the 3D tiles, the building whose mesh is at a lat/lon (`top`: the
    /// mesh's height there): the one of our own footprints it stands on, or
    /// the nearest within a few metres (the mesh and the map rarely agree to
    /// the metre). None where the mesh is only ground.
    pub fn photo_building(&self, lat: f64, lon: f64, top: f64) -> Option<BHit> {
        if !self.hidden(lat, lon) {
            return None;
        }
        let g = self.terrain_ground(lat, lon)?;
        if top - g.h < 2.5 {
            return None;
        }
        let p = geo::lla_to_pos(lat, lon, g.h);
        let (e, n, _) = geo::enu(p);
        let b = std::iter::once((0.0, 0.0))
            .chain([1.5f64, 3.0, 5.0].iter().flat_map(|&r| (0..8).map(move |k| {
                let a = k as f64 * std::f64::consts::FRAC_PI_4;
                (r * a.cos(), r * a.sin())
            })))
            .find_map(|(x, y)| {
                let (la, lo, _) = geo::pos_to_lla(p + e * x + n * y);
                self.footprint_at(la, lo).filter(|b| !b.bridge)
            })?;
        Some(BHit { top, ..b })
    }

    /// The outlines of a destroyed building's parts (our frame, at the base),
    /// grown by up to `grow` metres, less where another building stands close
    /// by: what is cut out of the 3D tiles' mesh.
    pub fn cut_rings(&self, parts: &[Wreck], grow: f64) -> Vec<Vec<DVec3>> {
        let free = |p: DVec3| {
            let (la, lo, _) = geo::pos_to_lla(p);
            self.footprint_at(la, lo).is_none()
        };
        parts
            .iter()
            .map(|w| {
                let n = w.ring.len();
                if n < 3 {
                    return w.ring.clone();
                }
                let c = w.ring.iter().fold(DVec3::ZERO, |a, p| a + *p) / n as f64;
                let (e, nn, _) = geo::enu(c);
                let r2: Vec<glam::DVec2> = w.ring.iter().map(|p| glam::DVec2::new((*p - c).dot(e), (*p - c).dot(nn))).collect();
                let area: f64 = (0..n).map(|i| r2[i].perp_dot(r2[(i + 1) % n])).sum::<f64>();
                let sign = area.signum();
                (0..n)
                    .map(|i| {
                        let (p, q, v) = (r2[(i + n - 1) % n], r2[(i + 1) % n], r2[i]);
                        let (e0, e1) = ((v - p).normalize_or_zero(), (q - v).normalize_or_zero());
                        // Outward, mitred (but not into long spikes at sharp corners).
                        let (n0, n1) = (-e0.perp() * sign, -e1.perp() * sign);
                        let m = n0 + n1;
                        let k = m.length_squared();
                        let off = if k > 0.15 { m * (2.0 / k) } else { (n0 + n1).normalize_or_zero() };
                        let off = if off.length() > 2.5 { off.normalize() * 2.5 } else { off };
                        let at = |d: f64| c + e * (v.x + off.x * d) + nn * (v.y + off.y * d);
                        let d = [grow, grow * 0.6, grow * 0.3].into_iter().find(|&d| free(at(d)) && free(at(d * 0.5))).unwrap_or(0.15);
                        at(d)
                    })
                    .collect()
            })
            .collect()
    }

    /// Whether our own scenery at lat/lon is hidden under the 3D mesh (and so
    /// is not there for the physics either).
    fn hidden(&self, lat: f64, lon: f64) -> bool {
        self.photo.as_deref().is_some_and(|p| p.hides(lat, lon))
    }

    /// Streams the 3D tiles for this frame (see tiles3d.rs for the arguments).
    #[allow(clippy::too_many_arguments)]
    pub fn update_photo(&mut self, r: &Renderer, cam: DVec3, focus: DVec3, fr: &Frustum, px: f64, day: f64, dt: f64) {
        let mut p = self.photo.take().expect("photo tiles");
        let fine = 13.min(self.max_zoom);
        // (a sample is only good for the datum on fine terrain, and not on a
        // building's roof: downtown most of the mesh is roofs and walls)
        p.update(r, cam, focus, fr, px, day, dt, &|la, lo| self.terrain_ground(la, lo).map(|g| (g.h, self.ground_zoom(la, lo).is_some_and(|z| z >= fine) && self.footprint_at(la, lo).is_none())));
        self.photo = Some(p);
    }

    /// How the 3D tiles blend in this frame (see tiles3d::PhotoFrame).
    pub fn photo_frame(&mut self, cam: DVec3) -> crate::tiles3d::PhotoFrame {
        self.photo.as_deref_mut().expect("photo tiles").frame(cam)
    }

    /// The 3D tiles to draw, with their offsets from the camera.
    pub fn photo_draws(&self, cam: DVec3) -> Vec<(&crate::render::PhotoGpu, [[f32; 4]; 2])> {
        self.photo().draws(cam)
    }

    /// Height of our own terrain surface (or water surface) at a lat/lon, from
    /// the most detailed tile currently loaded there.
    pub fn terrain_ground(&self, lat: f64, lon: f64) -> Option<Ground> {
        for z in (2..=self.max_zoom).rev() {
            let id = TileId::at(lat, lon, z);
            if let Some(Slot::Ready(t)) = self.tiles.get(&id) {
                let (u, v) = id.frac(lat, lon);
                let (x, y) = ((u * GRID as f64).clamp(0.0, GRID as f64 - 1e-6), (v * GRID as f64).clamp(0.0, GRID as f64 - 1e-6));
                let (x0, y0) = (x as usize, y as usize);
                let (fx, fy) = ((x - x0 as f64) as f32, (y - y0 as f64) as f32);
                let g = |i: usize, j: usize| t.heights[j * GV + i];
                let h = (g(x0, y0) * (1.0 - fx) + g(x0 + 1, y0) * fx) * (1.0 - fy) + (g(x0, y0 + 1) * (1.0 - fx) + g(x0 + 1, y0 + 1) * fx) * fy;
                let w = t.water[(y + 0.5) as usize * GV + (x + 0.5) as usize];
                let mut h = h as f64;
                // Water rides the ocean's long waves, as much as it shows them.
                let s = |i: usize, j: usize, c: usize| t.sea[j * GV + i][c];
                let sea = |c: usize| ((s(x0, y0, c) * (1.0 - fx) + s(x0 + 1, y0, c) * fx) * (1.0 - fy) + (s(x0, y0 + 1, c) * (1.0 - fx) + s(x0 + 1, y0 + 1, c) * fx) * fy) as f64;
                let sea = [sea(0), sea(1)];
                if w && sea[0] + sea[1] > 0.0 {
                    h += self.ocean.height_w(geo::lla_to_pos(lat, lon, h), sea);
                }
                if !self.craters.is_empty() {
                    let p = geo::lla_to_pos(lat, lon, h);
                    for (c, r) in &self.craters {
                        let v = p - *c;
                        if v.length_squared() < (*r as f64 * 2.0).powi(2) {
                            let up = c.normalize();
                            let d = (v - up * v.dot(up)).length() / *r as f64;
                            if d < 1.8 {
                                h += crater_profile(d) * *r as f64 * 0.34;
                            }
                        }
                    }
                }
                return Some(Ground { h, water: w });
            }
        }
        self.flat.map(|h| Ground { h, water: false })
    }

    /// Building or bridge deck over a lat/lon.
    pub fn building(&self, lat: f64, lon: f64) -> Option<BHit> {
        if self.hidden(lat, lon) {
            return None;
        }
        self.footprint_at(lat, lon)
    }

    /// Our own building or bridge deck over a lat/lon, hidden under the 3D tiles or not.
    fn footprint_at(&self, lat: f64, lon: f64) -> Option<BHit> {
        let id = TileId::at(lat, lon, DATA_ZOOM);
        if let Some(BSlot::Ready(b)) = self.builds.get(&id) {
            let (u, v) = id.frac(lat, lon);
            let p = [u as f32, v as f32];
            for (i, f) in b.footprints.iter().enumerate() {
                if f.alive && p[0] >= f.min[0] && p[0] <= f.max[0] && p[1] >= f.min[1] && p[1] <= f.max[1] && buildings::point_in_ring(p, &f.ring) {
                    return Some(BHit { base: f.base as f64, top: f.top as f64, bridge: f.kind == 1, tile: id, idx: i, area: f.area });
                }
            }
        }
        None
    }

    /// Standing trees whose base is within `r` metres (horizontally) of a point.
    pub fn trees_near(&self, p: DVec3, r: f64, out: &mut Vec<TreeRef>) {
        let (lat, lon, _) = geo::pos_to_lla(p);
        if self.hidden(lat, lon) {
            return;
        }
        let c = TileId::at(lat, lon, DATA_ZOOM);
        let size = 1i64 << DATA_ZOOM;
        let up = p.normalize();
        for dy in -1..=1i64 {
            for dx in -1..=1i64 {
                let (x, y) = ((c.x as i64 + dx).rem_euclid(size), c.y as i64 + dy);
                if y < 0 || y >= size {
                    continue;
                }
                let id = TileId::new(DATA_ZOOM, x as u32, y as u32);
                let Some(BSlot::Ready(b)) = self.builds.get(&id) else { continue };
                let lp = (p - b.center).as_vec3();
                for ch in &b.chunks {
                    if (ch.center - lp).length() > ch.radius + r as f32 + 40.0 {
                        continue;
                    }
                    for (_, first, count) in &ch.runs {
                        for i in *first as usize..(*first + *count) as usize {
                            let t = &b.trees[i];
                            if t.a[3] <= 0.0 {
                                continue;
                            }
                            let d = glam::Vec3::new(t.a[0], t.a[1], t.a[2]) - lp;
                            let dv = d.as_dvec3();
                            let horiz = dv - up * dv.dot(up);
                            if horiz.length_squared() > r * r {
                                continue;
                            }
                            let dim = self.tree_dims.get(b.tree_var[i] as usize).copied().unwrap_or_default();
                            let (hs, ws) = (t.a[3] as f64, (t.a[3] * t.b[3]) as f64);
                            out.push(TreeRef { tile: id, idx: i, pos: b.center + glam::Vec3::new(t.a[0], t.a[1], t.a[2]).as_dvec3(), height: dim.height as f64 * hs, crown_r: dim.crown_r as f64 * ws, crown_base: dim.crown_base as f64 * hs, trunk_r: dim.trunk_r as f64 * ws, variant: b.tree_var[i], inst: *t });
                        }
                    }
                }
            }
        }
    }

    /// Take a tree out of the standing forest (it has been knocked loose).
    pub fn remove_tree(&mut self, queue: &wgpu::Queue, t: &TreeRef) {
        if let Some(BSlot::Ready(b)) = self.builds.get_mut(&t.tile) {
            if let Some(inst) = b.trees.get_mut(t.idx) {
                inst.a[3] = 0.0;
                if let Some(buf) = b.gpu.as_ref().and_then(|g| g.trees.as_ref()) {
                    queue.write_buffer(buf, t.idx as u64 * 36 + 12, &[0u8; 4]);
                }
            }
        }
    }

    /// Remove a building from the world and hand back what is needed to wreck it.
    pub fn destroy_building(&mut self, queue: &wgpu::Queue, tile: TileId, idx: usize) -> Option<Wreck> {
        let Some(BSlot::Ready(b)) = self.builds.get_mut(&tile) else { return None };
        let f = b.footprints.get_mut(idx)?;
        if !f.alive || f.kind != 0 {
            return None;
        }
        f.alive = false;
        if let Some(g) = &b.gpu {
            for (start, n) in [f.main, f.det, f.fine, f.far] {
                if n > 0 {
                    queue.write_buffer(&g.ibuf, start as u64 * 4, &vec![0u8; n as usize * 4]);
                }
            }
        }
        let ring = f.ring.iter().map(|p| {
            let (lat, lon) = tile.lat_lon(p[0] as f64, p[1] as f64);
            geo::lla_to_pos(lat, lon, f.base as f64)
        }).collect();
        Some(Wreck { ring, base: f.base as f64, top: f.top as f64, col: f.col, tile, sty: f.sty, gref: f.gref as f64, roof_col: f.roof_col, tiers: f.tiers.clone() })
    }

    /// The rest of the building a footprint belongs to, which comes down with
    /// it: parts standing on it (a tower on its podium, a spire on its tower)
    /// and, for a tall building, the parts beside it that rise about as high
    /// (towers are often mapped as several full-height pieces side by side).
    pub fn structure_of(&self, tile: TileId, idx: usize) -> Vec<(TileId, usize)> {
        let Some(BSlot::Ready(b0)) = self.builds.get(&tile) else { return Vec::new() };
        let Some(f0) = b0.footprints.get(idx) else { return Vec::new() };
        let org = b0.center + f0.center.as_dvec3();
        let (e, n, _) = geo::enu(org);
        let hit_h = f0.top - f0.base;
        let tall = hit_h >= 28.0;
        let size = 1i64 << DATA_ZOOM;
        // Every candidate nearby, with its outline in metres on a local grid.
        struct Cand {
            id: (TileId, usize),
            ring: Vec<glam::Vec2>,
            base: f32,
            top: f32,
        }
        let o0 = {
            let (la, lo, _) = geo::pos_to_lla(org);
            geo::lla_to_pos(la, lo, 0.0)
        };
        let to_local = |t: TileId, p: [f32; 2]| {
            let (la, lo) = t.lat_lon(p[0] as f64, p[1] as f64);
            let q = geo::lla_to_pos(la, lo, 0.0) - o0;
            glam::Vec2::new(q.dot(e) as f32, q.dot(n) as f32)
        };
        let mut cands: Vec<Cand> = Vec::new();
        for dy in -1..=1i64 {
            for dx in -1..=1i64 {
                let (x, y) = ((tile.x as i64 + dx).rem_euclid(size), tile.y as i64 + dy);
                if y < 0 || y >= size {
                    continue;
                }
                let id = TileId::new(DATA_ZOOM, x as u32, y as u32);
                let Some(BSlot::Ready(b)) = self.builds.get(&id) else { continue };
                for (i, f) in b.footprints.iter().enumerate() {
                    if !f.alive || f.kind != 0 || f.ring.len() < 3 {
                        continue;
                    }
                    if (b.center + f.center.as_dvec3() - org).length() > 250.0 + f.area.sqrt() as f64 {
                        continue;
                    }
                    cands.push(Cand { id: (id, i), ring: f.ring.iter().map(|p| to_local(id, *p)).collect(), base: f.base, top: f.top });
                }
            }
        }
        let inside = |r: &[glam::Vec2], p: glam::Vec2| {
            let mut c = false;
            let mut j = r.len() - 1;
            for i in 0..r.len() {
                let (p0, p1) = (r[i], r[j]);
                if (p0.y > p.y) != (p1.y > p.y) && p.x < (p1.x - p0.x) * (p.y - p0.y) / (p1.y - p0.y) + p0.x {
                    c = !c;
                }
                j = i;
            }
            c
        };
        // Outlines that overlap or share a wall (within a metre and a half).
        let touch = |a: &[glam::Vec2], b: &[glam::Vec2]| {
            if a.iter().any(|p| inside(b, *p)) || b.iter().any(|p| inside(a, *p)) {
                return true;
            }
            let near = |p: glam::Vec2, r: &[glam::Vec2]| {
                (0..r.len()).any(|i| {
                    let (s0, s1) = (r[i], r[(i + 1) % r.len()]);
                    let d = s1 - s0;
                    let t = ((p - s0).dot(d) / d.length_squared().max(1e-6)).clamp(0.0, 1.0);
                    (s0 + d * t - p).length() < 1.5
                })
            };
            a.iter().any(|p| near(*p, b)) || b.iter().any(|p| near(*p, a))
        };
        let Some(start) = cands.iter().position(|c| c.id == (tile, idx)) else { return Vec::new() };
        let mut taken = vec![false; cands.len()];
        taken[start] = true;
        let mut todo = vec![start];
        let mut out = Vec::new();
        while let Some(k) = todo.pop() {
            if out.len() >= 40 {
                break;
            }
            for j in 0..cands.len() {
                if taken[j] {
                    continue;
                }
                let (cur, c) = (&cands[k], &cands[j]);
                let stacked = c.base >= cur.base + 0.5 && c.base <= cur.top + 3.0;
                let beside = tall && c.top - c.base >= 20.0 && c.top >= f0.top - hit_h * 0.5;
                if !(stacked || beside) || !touch(&cur.ring, &c.ring) {
                    continue;
                }
                taken[j] = true;
                out.push(c.id);
                todo.push(j);
            }
        }
        out
    }

    /// Footprints of small buildings whose centre is within `r` of a point.
    /// (Under the 3D tiles too: they are cut out of the mesh when they come down.)
    /// A standing building's base, top and footprint area.
    pub fn footprint(&self, tile: TileId, idx: usize) -> Option<(f64, f64, f32)> {
        let Some(BSlot::Ready(b)) = self.builds.get(&tile) else { return None };
        b.footprints.get(idx).filter(|f| f.alive).map(|f| (f.base as f64, f.top as f64, f.area))
    }

    pub fn buildings_near(&self, p: DVec3, r: f64, max_area: f32) -> Vec<(TileId, usize)> {
        let (lat, lon, _) = geo::pos_to_lla(p);
        let id = TileId::at(lat, lon, DATA_ZOOM);
        let mut out = Vec::new();
        if let Some(BSlot::Ready(b)) = self.builds.get(&id) {
            let lp = (p - b.center).as_vec3();
            for (i, f) in b.footprints.iter().enumerate() {
                if f.alive && f.kind == 0 && f.area < max_area && (f.center - lp).length() < r as f32 {
                    out.push((id, i));
                }
            }
        }
        out
    }

    /// Standing buildings and bridge decks near a point: outline (planet
    /// frame, at the base), base and top altitude, whether it is a bridge.
    pub fn footprints_near(&self, p: DVec3, r: f64) -> Vec<(Vec<DVec3>, f64, f64, bool)> {
        let (lat, lon, _) = geo::pos_to_lla(p);
        if self.hidden(lat, lon) {
            return Vec::new();
        }
        let c = TileId::at(lat, lon, DATA_ZOOM);
        let size = 1i64 << DATA_ZOOM;
        let mut out = Vec::new();
        for dy in -1..=1i64 {
            for dx in -1..=1i64 {
                let (x, y) = ((c.x as i64 + dx).rem_euclid(size), c.y as i64 + dy);
                if y < 0 || y >= size {
                    continue;
                }
                let id = TileId::new(DATA_ZOOM, x as u32, y as u32);
                let Some(BSlot::Ready(b)) = self.builds.get(&id) else { continue };
                let lp = (p - b.center).as_vec3();
                for f in &b.footprints {
                    if !f.alive || (f.center - lp).length() > r as f32 + f.area.sqrt() {
                        continue;
                    }
                    let ring = f.ring.iter().map(|q| {
                        let (la, lo) = id.lat_lon(q[0] as f64, q[1] as f64);
                        geo::lla_to_pos(la, lo, f.base as f64)
                    }).collect();
                    out.push((ring, f.base as f64, f.top as f64, f.kind == 1));
                }
            }
        }
        out
    }

    pub fn add_crater(&mut self, pos: DVec3, radius: f32) {
        if self.craters.len() >= 8 {
            self.craters.remove(0);
        }
        self.craters.push((pos, radius));
    }

    /// Drive the traffic on the tiles around a point.
    pub fn update_traffic(&mut self, dt: f32, focus: DVec3) {
        self.tick = self.tick.wrapping_add(1);
        let tick = self.tick;
        let mut exits: Vec<(TileId, cars::Exit, glam::Vec3)> = Vec::new();
        for (id, slot) in self.builds.iter_mut() {
            let BSlot::Ready(b) = slot else { continue };
            if b.cars.is_empty() || (b.center - focus).length() > 2600.0 + b.radius {
                continue;
            }
            let up = b.center.normalize().as_vec3();
            let mut cs = std::mem::take(&mut b.cars);
            for e in cars::step(&b.net, &mut cs, dt, up, tick) {
                let was = cars::pose(&b.net, &e.car, up).0;
                exits.push((*id, e, was));
            }
            b.cars = cs;
        }
        let size = 1i64 << DATA_ZOOM;
        for (from, e, was) in exits {
            let (dx, dy) = match e.edge {
                1 => (-1, 0),
                2 => (1, 0),
                3 => (0, -1),
                _ => (0, 1),
            };
            let nid = TileId::new(DATA_ZOOM, (from.x as i64 + dx).rem_euclid(size) as u32, (from.y as i64 + dy).clamp(0, size - 1) as u32);
            let from_center = match self.builds.get(&from) {
                Some(BSlot::Ready(b)) => b.center,
                _ => continue,
            };
            let mut car = Some(e.car);
            if let Some(BSlot::Ready(n)) = self.builds.get_mut(&nid) {
                let up = n.center.normalize().as_vec3();
                let was_n = ((from_center - n.center).as_vec3()) + was;
                if let Some(c) = cars::enter(&n.net, car.clone().unwrap(), e.key, up, was_n) {
                    n.cars.push(c);
                    car = None;
                }
            }
            // Nowhere to go (next tile not loaded, or the road ends at the border): start afresh.
            if let (Some(mut c), Some(BSlot::Ready(b))) = (car, self.builds.get_mut(&from)) {
                cars::respawn(&b.net, &mut c, tick ^ (e.key as u32));
                b.cars.push(c);
            }
        }
    }

    /// Instances for every visible moving and parked car; and, at night
    /// (`sun`: sine of the sun's elevation at the camera), the lights of the
    /// moving ones that the 3D tiles hide (`photo_car_light`).
    pub fn car_instances(&self, cam: DVec3, activity: f32, sun: f64, out: &mut Vec<(u32, Inst)>, lights: &mut Vec<crate::airfield::Light>) {
        let fr = self.frustum.as_ref();
        let mut push = |pos: DVec3, fwd: glam::Vec3, up: glam::Vec3, model: u8, col: [f32; 3], out: &mut Vec<(u32, Inst)>| {
            let o = (pos - cam).as_vec3();
            if fr.map(|f| f.sphere(pos - cam, 8.0)).unwrap_or(true) {
                let right = fwd.cross(up).normalize_or_zero();
                let upv = right.cross(fwd);
                out.push((model as u32, Inst { m: [[right.x, upv.x, -fwd.x, o.x], [right.y, upv.y, -fwd.y, o.y], [right.z, upv.z, -fwd.z, o.z]], tint: [col[0], col[1], col[2], 0.0], ext: [0.0, 0.0, 0.0, 1.0] }));
            }
        };
        let photo = self.photo.as_deref().filter(|_| sun < 0.15);
        for id in &self.draw_b {
            let Some(BSlot::Ready(b)) = self.builds.get(id) else { continue };
            let d = (b.center - cam).length() - b.radius;
            if d > CAR_FAR {
                continue;
            }
            let lit = photo.filter(|p| p.covers(b.center));
            let mut streets = None;
            let up = b.center.normalize().as_vec3();
            for (i, c) in b.cars.iter().enumerate() {
                if c.rank > activity || b.net.paths[c.path as usize].hidden {
                    continue;
                }
                let (p, dir) = cars::pose(&b.net, c, up);
                let pos = b.center + (p + c.shift).as_dvec3();
                if (pos - cam).length_squared() < CAR_FAR * CAR_FAR {
                    let fwd = if c.dir == glam::Vec3::ZERO { dir } else { c.dir };
                    push(pos, fwd, up, c.model, c.col, out);
                    if let Some(photo) = lit {
                        self.photo_car_light(photo, &mut streets, b, i, c, pos, fwd, cam, lights);
                    }
                }
            }
            if d < 1100.0 {
                for p in &b.parked {
                    let pos = b.center + p.pos.as_dvec3();
                    if (pos - cam).length_squared() < 1100.0 * 1100.0 {
                        push(pos, p.fwd, p.up, p.model, p.col, out);
                    }
                }
            }
        }
    }

    pub fn update(&mut self, r: &Renderer, cam: DVec3, fr: &Frustum) {
        self.frame += 1;
        let now = std::time::Instant::now();
        let real_dt = self.last_update.map(|t| (now - t).as_secs_f32().min(0.25)).unwrap_or(0.0);
        self.last_update = Some(now);
        // Take finished tiles off the worker pool, capping the GPU uploads per
        // frame both in count and in bytes (see UPLOAD_BUDGET). Terrain goes
        // first; queued building geometry always gets at least half the bytes.
        let mut bytes = 0;
        let mut budget = 10;
        while budget > 0 && bytes < UPLOAD_BUDGET {
            match self.loader.rx.try_recv() {
                Ok(JobResult::Terrain(id, res)) => {
                    self.in_flight -= 1;
                    budget -= 1;
                    let slot = match res {
                        Ok(td) => {
                            bytes += crate::render::tile_bytes(&td);
                            Slot::Ready(Box::new(Tile { center: td.center, radius: td.radius, gpu: r.upload_tile(&td), heights: td.heights, has_water: td.water.iter().any(|w| *w), water: td.water, sea: td.sea, last_used: self.frame, generation: td.imagery_gen, refreshing: false }))
                        }
                        Err(e) => {
                            log::debug!("tile {id:?} failed: {e}");
                            // A refresh that failed keeps the tile it was replacing.
                            if let Some(Slot::Ready(t)) = self.tiles.get_mut(&id) {
                                (t.generation, t.refreshing) = (self.loader.src.google.generation(), false);
                                continue;
                            }
                            Slot::Failed
                        }
                    };
                    self.tiles.insert(id, slot);
                }
                Ok(JobResult::Buildings(id, res)) => {
                    self.in_flight -= 1;
                    budget -= 3;
                    match res {
                        Ok(bd) => {
                            // The slot stays Loading until the geometry is all on the GPU.
                            bytes += bd.atlas.iter().map(|m| m.len()).sum::<usize>();
                            let gpu = r.upload_buildings(&bd);
                            self.uploads.push_back(Upload { id, data: Box::new(bd), gpu, part: 0, off: 0 });
                        }
                        Err(_) => {
                            self.builds.remove(&id);
                        }
                    }
                }
                Ok(JobResult::Airfield(i, res)) => {
                    self.in_flight -= 1;
                    budget -= 3;
                    match res {
                        Ok((geo, mesh)) => {
                            bytes += Renderer::field_bytes(&mesh);
                            let gpu = r.upload_field(&mesh);
                            let traffic = crate::traffic::Traffic::new(&geo, i.wrapping_mul(2654435761) ^ 0x5bd1e995);
                            log::info!("airfield {} ready: {} runways, {} stands, {} lights", geo.apt().ident, geo.runways.len(), geo.stands.len(), mesh.lights.len());
                            self.fields.insert(i, FSlot::Ready(Box::new(Field { geo, gpu, props: mesh.props, socks: mesh.socks, traffic, last_used: self.frame })));
                        }
                        Err(e) => {
                            log::debug!("airfield {i} failed: {e}");
                            self.fields.insert(i, FSlot::Failed);
                        }
                    }
                }
                Err(_) => break,
            }
        }
        self.pump_uploads(r, bytes.min(UPLOAD_BUDGET / 2));

        let (clat, clon, calt) = geo::pos_to_lla(cam);
        let gh = self.ground(clat, clon).map(|g| g.h).unwrap_or(0.0);
        self.draw.clear();
        self.draw_shore.clear();
        self.wanted.clear();
        for y in 0..4 {
            for x in 0..4 {
                self.select(TileId::new(2, x, y), cam, (clat, clon, calt.max(1.0), gh), fr);
            }
        }

        // Buildings near the ground.
        self.draw_b.clear();
        self.in_view.clear();
        self.frustum = Some(fr.clone());
        let agl = calt - gh;
        if self.buildings_on && agl < 7000.0 && clat.abs() < 1.45 {
            let c = TileId::at(clat, clon, DATA_ZOOM);
            let w = c.width_m();
            let reach = (4500.0 + agl * 0.6).min(7500.0);
            let n = (reach / w).ceil() as i64;
            let size = 1i64 << DATA_ZOOM;
            for dy in -n..=n {
                for dx in -n..=n {
                    let (x, y) = ((c.x as i64 + dx).rem_euclid(size), c.y as i64 + dy);
                    if y < 0 || y >= size {
                        continue;
                    }
                    let id = TileId::new(DATA_ZOOM, x as u32, y as u32);
                    let (lat, lon) = id.lat_lon(0.5, 0.5);
                    let d = (geo::lla_to_pos(lat, lon, gh) - cam).length();
                    if d > reach + w {
                        continue;
                    }
                    match self.builds.get_mut(&id) {
                        Some(BSlot::Ready(b)) => {
                            b.last_used = self.frame;
                            // Detail arrives gradually: the tile dissolves in, then its middle and
                            // finest levels as the camera comes close enough for each.
                            let dist = ((b.center - cam).length() - b.radius) as f32;
                            b.shown += real_dt;
                            // Towers carry their detail farther (buildings::HCLASS).
                            let (mid, fine) = (LOD_MID * b.tall, LOD_FINE * b.tall);
                            b.mid_t = if dist < mid { b.mid_t + real_dt } else if dist > mid * 1.1 { 0.0 } else { b.mid_t };
                            b.fine_t = if dist < fine { b.fine_t + real_dt } else if dist > fine * 1.2 { 0.0 } else { b.fine_t };
                            if b.gpu.is_some() {
                                self.draw_b.push(id);
                                self.in_view.push(fr.sphere(b.center - cam, b.radius));
                            }
                        }
                        Some(BSlot::Loading) => {}
                        None => self.wanted.push((0.8 + (d / w) as f32 * 0.6, id.with_tag())),
                    }
                }
            }
        }

        // Airfields within reach: the bigger ones from further off.
        if self.frame % 15 == 1 && clat.abs() < 1.45 {
            let db = crate::airports::Db::get();
            let mut n = 0;
            for (i, d) in db.near(clat.to_degrees(), clon.to_degrees(), 35_000.0) {
                let a = &db.apts[i as usize];
                let reach = [9000.0, 22_000.0, 35_000.0][a.kind as usize] * if a.any_paved() { 1.0 } else { 0.5 };
                if d > reach || n >= 6 {
                    continue;
                }
                n += 1;
                match self.fields.get_mut(&i) {
                    Some(FSlot::Ready(f)) => f.last_used = self.frame,
                    Some(_) => {}
                    None if self.in_flight < 28 => {
                        self.fields.insert(i, FSlot::Loading);
                        self.in_flight += 1;
                        self.loader.submit(1.0 + (d / 9000.0) as f32, Job::Airfield(i));
                    }
                    None => {}
                }
            }
        }

        // Dispatch the most urgent requests.
        self.wanted.sort_by(|a, b| a.0.total_cmp(&b.0));
        for (prio, id) in self.wanted.drain(..) {
            if self.in_flight >= 28 {
                break;
            }
            self.in_flight += 1;
            if id.z & 0x80 != 0 {
                let id = TileId::new(id.z & 0x7f, id.x, id.y);
                self.builds.insert(id, BSlot::Loading);
                self.loader.submit(prio, Job::Buildings(id));
            } else if id.z & 0x40 != 0 {
                // keeps drawing the old tile until the new one is in
                let id = TileId::new(id.z & 0x3f, id.x, id.y);
                match self.tiles.get_mut(&id) {
                    Some(Slot::Ready(t)) if !t.refreshing => {
                        t.refreshing = true;
                        self.loader.submit(prio, Job::Terrain(id));
                    }
                    _ => self.in_flight -= 1,
                }
            } else {
                self.tiles.insert(id, Slot::Loading);
                self.loader.submit(prio, Job::Terrain(id));
            }
        }

        // Evict what has not been needed for a while.
        if self.frame % 120 == 0 {
            let f = self.frame;
            if self.tiles.len() > 700 {
                self.tiles.retain(|id, s| id.z <= 3 || !matches!(s, Slot::Ready(t) if f - t.last_used > 600));
            }
            self.tiles.retain(|_, s| !matches!(s, Slot::Failed) || f % 3600 != 0);
            self.builds.retain(|_, s| !matches!(s, BSlot::Ready(b) if f - b.last_used > 900));
            self.fields.retain(|_, s| match s {
                FSlot::Ready(fl) => f - fl.last_used < 1200,
                FSlot::Failed => f % 3600 != 0,
                FSlot::Loading => true,
            });
        }
        self.stats = (self.draw.len(), self.in_flight, self.draw_b.len());
    }

    /// Copy queued building geometry to the GPU until `bytes` (already used
    /// this frame) reaches the budget; finished tiles become drawable.
    fn pump_uploads(&mut self, r: &Renderer, mut bytes: usize) -> usize {
        while bytes < UPLOAD_BUDGET {
            let Some(u) = self.uploads.front_mut() else { break };
            let payload = crate::render::building_payload(&u.data);
            let targets = u.gpu.targets();
            while u.part < payload.len() && bytes < UPLOAD_BUDGET {
                let (Some(buf), data) = (targets[u.part], payload[u.part]) else {
                    (u.part, u.off) = (u.part + 1, 0);
                    continue;
                };
                if u.off >= data.len() {
                    (u.part, u.off) = (u.part + 1, 0);
                    continue;
                }
                // Chunks stay multiples of 4 (all the vertex types are).
                let n = (data.len() - u.off).min(UPLOAD_CHUNK).min((UPLOAD_BUDGET - bytes).max(1 << 20)) & !3;
                let n = if n == 0 { data.len() - u.off } else { n };
                r.fill_buffer(buf, u.off as u64, &data[u.off..u.off + n]);
                u.off += n;
                bytes += n;
            }
            if u.part < payload.len() {
                break;
            }
            let Upload { id, data, gpu, .. } = self.uploads.pop_front().unwrap();
            let bd = *data;
            let main = bd.cls[0];
            let tall = (0..buildings::HCLASS.len()).find(|&c| main[c] > if c == 0 { 0 } else { main[c - 1] }).map_or(1.0, |c| buildings::HCLASS[c].1);
            let cars = cars::spawn(&bd.net, id.x.wrapping_mul(2246822519) ^ id.y.wrapping_mul(3266489917));
            self.builds.insert(id, BSlot::Ready(Box::new(BTile { center: bd.center, radius: bd.radius, gpu: Some(gpu), footprints: bd.footprints, last_used: self.frame, net: bd.net, cars, parked: bd.parked, trees: bd.trees, tree_var: bd.tree_var, chunks: bd.chunks, under: bd.under, cells: bd.cells, near_idx: bd.near_idx, ax: bd.ax, ay: bd.ay, shown: 0.0, mid_t: 0.0, fine_t: 0.0, tall })));
        }
        bytes
    }

    /// Zoom of the most detailed terrain tile loaded at a spot.
    pub fn ground_zoom(&self, lat: f64, lon: f64) -> Option<u8> {
        (2..=self.max_zoom).rev().find(|z| matches!(self.tiles.get(&TileId::at(lat, lon, *z)), Some(Slot::Ready(_))))
    }

    /// The nearest airfield that is loaded, within `r` metres.
    pub fn airfield_near(&self, p: DVec3, r: f64) -> Option<&Field> {
        self.fields.values().filter_map(|s| if let FSlot::Ready(f) = s { Some(f.as_ref()) } else { None }).filter(|f| (f.geo.center - p).length() < r + f.geo.radius as f64).min_by(|a, b| (a.geo.center - p).length().total_cmp(&(b.geo.center - p).length()))
    }
    pub fn airfield(&self, idx: u32) -> Option<&Field> {
        match self.fields.get(&idx) {
            Some(FSlot::Ready(f)) => Some(f),
            _ => None,
        }
    }

    /// Move the airport traffic on. `player`: the player's aircraft position.
    pub fn update_airports(&mut self, dt: f32, wind_from: f32, wind_kt: f32, activity: f32, player: Option<DVec3>, focus: DVec3) {
        for s in self.fields.values_mut() {
            let FSlot::Ready(f) = s else { continue };
            if (f.geo.center - focus).length() > 40_000.0 {
                continue;
            }
            let pl = player.map(|p| {
                let lp = f.geo.local(p);
                (lp, geo::pos_to_lla(p).2 as f32 - f.geo.h(lp))
            });
            f.traffic.step(&f.geo, dt, wind_from, wind_kt, activity, pl);
        }
    }

    /// Pavement to draw: each airfield with its centre relative to the camera.
    pub fn field_draws(&self, cam: DVec3) -> Vec<(&crate::render::FieldGpu, [f32; 3])> {
        let mut out: Vec<(f64, &crate::render::FieldGpu, [f32; 3])> = Vec::new();
        for s in self.fields.values() {
            let FSlot::Ready(f) = s else { continue };
            let o = f.geo.center - cam;
            let d = o.length();
            if d > 60_000.0 || !self.frustum.as_ref().map(|fr| fr.sphere(o, f.geo.radius as f64 + 600.0)).unwrap_or(true) {
                continue;
            }
            out.push((d, &f.gpu, o.as_vec3().into()));
        }
        out.sort_by(|a, b| a.0.total_cmp(&b.0));
        out.into_iter().take(crate::render::MAX_FIELDS).map(|x| (x.1, x.2)).collect()
    }

    /// Fixed airport objects, parked and moving aircraft and their lights.
    pub fn airport_instances(&self, cam: DVec3, wind_from: f32, wind_kt: f32, t: f64, out: &mut Vec<(u32, Inst)>, lights: &mut Vec<crate::airfield::Light>) {
        let fr = self.frustum.as_ref();
        for s in self.fields.values() {
            let FSlot::Ready(f) = s else { continue };
            let g = &f.geo;
            let dc = (g.center - cam).length();
            if dc > 40_000.0 {
                continue;
            }
            let base = g.center - cam;
            for p in &f.props {
                let o = base + p.pos.as_dvec3();
                if o.length() > 6000.0 || !fr.map(|f| f.sphere(o, 40.0)).unwrap_or(true) {
                    continue;
                }
                let o = o.as_vec3();
                out.push((p.mesh, Inst { m: [[p.x.x, p.y.x, p.z.x, o.x], [p.x.y, p.y.y, p.z.y, o.y], [p.x.z, p.y.z, p.z.z, o.z]], tint: [p.tint[0], p.tint[1], p.tint[2], 0.0], ext: [0.0; 4] }));
            }
            // Windsocks stream downwind, and droop in a light breeze.
            let b = (wind_from + 180.0).to_radians();
            let hd = g.dir3(Vec3::new(b.sin(), b.cos(), 0.0));
            let up = g.dir3(Vec3::Z);
            for (k, sp) in f.socks.iter().enumerate() {
                let o = base + sp.as_dvec3();
                if o.length() > 5000.0 {
                    continue;
                }
                let wob = (t as f32 * (2.3 + k as f32 * 0.4)).sin() * 0.05 * (wind_kt / 10.0).min(1.5);
                let droop = (1.0 - (wind_kt / 15.0).clamp(0.0, 1.0)) * 1.25 + wob;
                let fwd = hd * droop.cos() - up * droop.sin();
                let fwd = (fwd + hd.cross(up) * wob).normalize();
                let upv = (up - fwd * fwd.dot(up)).normalize();
                let x = fwd.cross(upv);
                let z = -fwd;
                let o = o.as_vec3();
                out.push((crate::airmodels::M_SOCK, Inst { m: [[x.x, upv.x, z.x, o.x], [x.y, upv.y, z.y, o.y], [x.z, upv.z, z.z, o.z]], tint: [1.0, 1.0, 1.0, 0.0], ext: [0.0; 4] }));
            }
            f.traffic.draw(g, cam, t, out, lights);
        }
    }

    /// Nearest point of a tile to the camera, on the ground, and whether the
    /// whole tile lies below the horizon (never drawn, so never needed).
    fn tile_near(id: TileId, cam: DVec3, c: (f64, f64, f64, f64)) -> (DVec3, bool) {
        let (clat, clon, calt, gh) = c;
        let (lat_n, lon_w) = id.lat_lon(0.0, 0.0);
        let (lat_s, lon_e) = id.lat_lon(1.0, 1.0);
        let mut dl = clon;
        if dl < lon_w - std::f64::consts::PI {
            dl += std::f64::consts::TAU;
        } else if dl > lon_e + std::f64::consts::PI {
            dl -= std::f64::consts::TAU;
        }
        let near = geo::lla_to_pos(clat.clamp(lat_s, lat_n), dl.clamp(lon_w, lon_e), gh);
        let hidden = id.z > 2 && {
            let ang = (near.normalize().dot(cam.normalize())).clamp(-1.0, 1.0).acos();
            let horizon = (geo::R / (geo::R + calt)).acos() + (geo::R / (geo::R + 9000.0)).acos();
            ang > horizon
        };
        (near, hidden)
    }

    /// Returns true if this tile (or its descendants) covers its area.
    fn select(&mut self, id: TileId, cam: DVec3, c: (f64, f64, f64, f64), fr: &Frustum) -> bool {
        let (near, hidden) = Self::tile_near(id, cam, c);
        let dist = (near - cam).length().max(1.0);
        if hidden {
            return true;
        }
        let (center, radius) = match self.tiles.get_mut(&id) {
            Some(Slot::Ready(t)) => {
                t.last_used = self.frame;
                // made with the imagery source switched away from: again, after what is missing
                if !t.refreshing && t.generation != self.loader.src.google.generation() {
                    self.wanted.push((2.0 + (dist / id.width_m()) as f32, id.with_refresh()));
                }
                (t.center, t.radius)
            }
            Some(_) => return false,
            None => {
                self.wanted.push(((dist / id.width_m()) as f32, id));
                return false;
            }
        };
        let width = id.width_m();
        if id.z < self.max_zoom && width / dist > self.lod_k {
            let kids = id.children();
            let mut all = true;
            for k in kids {
                // A child below the horizon is never drawn: it need not be in
                // memory (and is evicted, not being used), so it must not
                // hold its siblings back and leave this coarser tile showing.
                if Self::tile_near(k, cam, c).1 {
                    continue;
                }
                match self.tiles.get(&k) {
                    Some(Slot::Ready(_)) => {}
                    Some(Slot::Loading) => all = false,
                    Some(Slot::Failed) => all = false,
                    None => {
                        all = false;
                        self.wanted.push(((dist / width) as f32 + 0.01, k));
                    }
                }
            }
            if all {
                for k in kids {
                    self.select(k, cam, c, fr);
                }
                return true;
            }
        }
        if fr.sphere(center - cam, radius) {
            self.draw.push(id);
        } else if (center - cam).length() - radius < crate::shore::WIDE_HALF * 1.6 {
            self.draw_shore.push(id);
        }
        true
    }

    /// Terrain tiles out of view that reach into the shore mask's square
    /// (centre `c`, both camera-relative), with their uniforms as `draw_lists`.
    pub fn shore_tiles(&self, cam: DVec3, c: DVec3) -> Vec<(&TileGpu, [f32; 12])> {
        let mut out = Vec::new();
        for id in &self.draw_shore {
            if let Some(Slot::Ready(t)) = self.tiles.get(id) {
                let o = t.center - cam;
                if (o - c).length() - t.radius > crate::shore::WIDE_HALF * 1.45 {
                    continue;
                }
                let o = o.as_vec3();
                out.push((&t.gpu, [o.x, o.y, o.z, (id.width_m() / GRID as f64) as f32, id.z as f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]));
            }
        }
        out
    }

    /// Terrain tiles and detail tiles to draw. `focus` and `casc` (radius of
    /// each shadow cascade) decide what has to be rendered into the shadow maps.
    pub fn draw_lists(&self, cam: DVec3, focus: DVec3, casc: [f32; 3]) -> (Vec<(&TileGpu, [f32; 12])>, Vec<DetailDraw<'_>>, Vec<TreeInst>) {
        let mut tiles = Vec::with_capacity(self.draw.len());
        let (clat, clon, calt) = geo::pos_to_lla(cam);
        let agl = calt - self.ground(clat, clon).map(|g| g.h).unwrap_or(0.0);
        for id in &self.draw {
            if let Some(Slot::Ready(t)) = self.tiles.get(id) {
                let o = (t.center - cam).as_vec3();
                let cell = id.width_m() / GRID as f64;
                let tess = tess_block(*id, clat, clon, agl, cell);
                let grid = if self.dense_water && t.has_water { water_grid(id.width_m(), (t.center - cam).length() - t.radius, agl) } else { 0.0 };
                tiles.push((&t.gpu, [o.x, o.y, o.z, cell as f32, id.z as f32, grid, 0.0, 0.0, tess[0], tess[1], tess[2], tess[3]]));
            }
        }
        let mut details = Vec::with_capacity(self.draw_b.len());
        let mut near_inst: Vec<TreeInst> = Vec::new();
        let mut tb = TreeBatcher::new();
        for (id, view) in self.draw_b.iter().zip(&self.in_view) {
            let Some(BSlot::Ready(b)) = self.builds.get(id) else { continue };
            let Some(g) = &b.gpu else { continue };
            let o = (b.center - cam).as_vec3();
            let dist = (o.length() as f64 - b.radius) as f32;
            let fdist = ((b.center - focus).length() - b.radius) as f32;
            let mut mask = 0u8;
            for (c, r) in casc.iter().enumerate() {
                if fdist < r * 1.45 {
                    mask |= 1 << c;
                }
            }
            if dist < 3300.0 && g.trees.is_some() {
                for ch in &b.chunks {
                    let cp = b.center + ch.center.as_dvec3();
                    let dc = (cp - cam).length() as f32;
                    if dc - ch.radius > 3200.0 {
                        continue;
                    }
                    let seen = self.frustum.as_ref().map(|f| f.sphere(cp - cam, ch.radius as f64 + 25.0)).unwrap_or(true);
                    let fd = (cp - focus).length() as f32 - ch.radius;
                    // The nearest cascade gets each tree's own model; the next one,
                    // with texels most of a metre across, one simple model for the
                    // whole chunk (no per-tree choice, so whole runs batch together).
                    let cm = (fd < casc[0] * 1.45) as u8;
                    let in_c1 = fd < casc[1] * 1.45;
                    if !seen && cm == 0 && !in_c1 {
                        continue;
                    }
                    if in_c1 {
                        let lod = LOD_DIST.iter().filter(|d| dc - ch.radius >= **d).count().clamp(C1_LOD, LODS - 2);
                        for (variant, first, count) in &ch.runs {
                            tb.push(TreeDraw { mesh: *variant as u32 * LODS as u32 + lod as u32, first: *first, count: *count, view: false, casc: 2, close: false, imp: false, fixed: true });
                        }
                    }
                    // Every model some tree in the chunk may use: the shader keeps
                    // each tree to the one for its own distance. In view, only
                    // trees short of IMP_DIST are drawn as models; the simpler
                    // ones are for the shadows.
                    let close = dc - ch.radius < IMP_DIST * (1.0 + LOD_JITTER);
                    for lod in 0..LODS - 1 {
                        let (lo, hi) = lod_span(lod);
                        if dc + ch.radius < lo || dc - ch.radius >= hi {
                            continue;
                        }
                        let view = seen && close && lod == 0;
                        if !view && cm == 0 {
                            continue;
                        }
                        for (variant, first, count) in &ch.runs {
                            tb.push(TreeDraw { mesh: *variant as u32 * LODS as u32 + lod as u32, first: *first, count: *count, view, casc: cm, close: view, imp: false, fixed: false });
                        }
                    }
                    // Further off, the trees are impostors.
                    if seen && dc + ch.radius >= IMP_DIST * (1.0 - LOD_JITTER) {
                        for (variant, first, count) in &ch.runs {
                            tb.push(TreeDraw { mesh: *variant as u32, first: *first, count: *count, view: true, casc: 0, close: false, imp: true, fixed: false });
                        }
                    }
                }
            }
            // Close to the camera: trees get their leaf cards, and the undergrowth appears.
            let mut near = Vec::new();
            if dist < CARD_RANGE && near_inst.len() < MAX_NEAR {
                let lp = -o;
                let seen = |c: Vec3, r: f32| self.frustum.as_ref().map(|f| f.sphere(c.as_dvec3(), r as f64)).unwrap_or(true);
                // Sorted into rings by distance so the nearest are drawn first and hide the rest.
                let ring = |d2: f32| if d2 < 20.0 * 20.0 { 0u16 } else if d2 < 45.0 * 45.0 { 256 } else if d2 < 80.0 * 80.0 { 512 } else { 768 };
                let mut found: Vec<(u16, TreeInst)> = Vec::new();
                for c in &b.cells {
                    let co = c.center - lp;
                    if co.length() - c.radius > CARD_RANGE || !seen(co, c.radius + 20.0) {
                        continue;
                    }
                    for i in &b.near_idx[c.trees.0 as usize..(c.trees.0 + c.trees.1) as usize] {
                        let t = &b.trees[*i as usize];
                        let to = Vec3::new(t.a[0], t.a[1], t.a[2]) - lp;
                        // Past IMP_DIST the tree is an impostor and has no cards.
                        if t.a[3] > 0.0 && to.length() < IMP_DIST * 1.01 * tree_jit(t.a) && seen(to, 26.0 * t.a[3]) {
                            found.push((ring(to.length_squared()) | b.tree_var[*i as usize] as u16, *t));
                        }
                    }
                    for u in &b.under[c.under.0 as usize..(c.under.0 + c.under.1) as usize] {
                        let uo = u.pos(c.center) - lp;
                        if uo.length_squared() > u.range() * u.range() || !seen(uo, 8.0) {
                            continue;
                        }
                        // Nothing grows in a fresh crater.
                        if self.craters.iter().any(|(cp, r)| ((*cp - cam).as_vec3() - uo).length_squared() < r * r * 1.5) {
                            continue;
                        }
                        found.push((ring(uo.length_squared()) | u.var as u16, u.inst(c.center)));
                    }
                }
                found.truncate(MAX_NEAR - near_inst.len());
                found.sort_unstable_by_key(|f| f.0);
                let mut i = 0;
                while i < found.len() {
                    let (key, first) = (found[i].0, near_inst.len() as u32);
                    let var = key & 255;
                    while i < found.len() && found[i].0 == key {
                        near_inst.push(found[i].1);
                        i += 1;
                    }
                    let (mesh, count) = (var as u32 * LODS as u32, near_inst.len() as u32 - first);
                    // Trees are already standing there as solids; undergrowth is not.
                    if var as usize / VARIANTS >= TREE_SPECIES {
                        near.push(NearDraw { mesh, first, count, cards: false });
                    }
                    near.push(NearDraw { mesh: mesh + CARDS, first, count, cards: true });
                }
            }
            let fade = |t: f32, secs: f32| (t / secs).clamp(0.0, 1.0);
            let lod = [fade(b.mid_t - 0.3, 1.2), fade(b.fine_t - 0.6, 1.6), fade(b.shown, 1.0)];
            // How many height classes, tallest first, are near enough for their
            // full geometry, their middle level and their finest level.
            let reach = [buildings::BFAR[1], LOD_MID, LOD_FINE].map(|r| buildings::HCLASS.iter().take_while(|c| dist < r * c.1).count() as u8);
            details.push(DetailDraw {
                gpu: g,
                u: [o.x, o.y, o.z, 0.0, b.ax[0], b.ax[1], b.ax[2], b.ax[3], b.ay[0], b.ay[1], b.ay[2], b.ay[3], lod[0], lod[1], lod[2], 0.0],
                view: *view,
                near: dist < 3000.0,
                far: if o.length() + (b.radius as f32) < buildings::BFAR[0] { 0 } else if reach[0] == 0 { 2 } else { 1 },
                reach_m: o.length() + b.radius as f32,
                reach,
                casc: mask,
                trees: tb.take(),
                close: near,
            });
        }
        (tiles, details, near_inst)
    }
}

/// Model the second shadow cascade draws trees with, at the least: some 25
/// triangles, enough for texels most of a metre across.
const C1_LOD: usize = 3;

/// Collects a tile's tree batches, joining each to the last one drawn the same
/// way when its instances follow straight on (a tile's trees are sorted by
/// variant, then chunk, and chunks are visited in order), so a stretch of
/// neighbouring chunks costs one draw instead of one each.
struct TreeBatcher {
    /// index + 1 into `out` of the open batch for each way of drawing (0: none)
    open: Vec<u32>,
    touched: Vec<usize>,
    out: Vec<TreeDraw>,
}
impl TreeBatcher {
    fn new() -> Self {
        Self { open: vec![0; 64 * 512], touched: Vec::new(), out: Vec::new() }
    }
    fn push(&mut self, t: TreeDraw) {
        let k = ((t.imp as usize) | (t.fixed as usize) << 1 | (t.view as usize) << 2 | (t.close as usize) << 3 | ((t.casc as usize) & 3) << 4) * 512 + (t.mesh as usize).min(511);
        let o = self.open[k] as usize;
        if o > 0 {
            let p = &mut self.out[o - 1];
            if p.mesh == t.mesh && p.first + p.count == t.first {
                p.count += t.count;
                return;
            }
        } else {
            self.touched.push(k);
        }
        self.out.push(t);
        self.open[k] = self.out.len() as u32;
    }
    /// The batches so far, ready for the next tile.
    fn take(&mut self) -> Vec<TreeDraw> {
        for k in self.touched.drain(..) {
            self.open[k] = 0;
        }
        std::mem::take(&mut self.out)
    }
}

trait Tag {
    fn with_tag(self) -> Self;
    fn with_refresh(self) -> Self;
}
impl Tag for TileId {
    /// Marks a request as a building tile inside the shared request list.
    fn with_tag(self) -> Self {
        TileId::new(self.z | 0x80, self.x, self.y)
    }
    /// Marks a request to remake a loaded terrain tile (new imagery source).
    fn with_refresh(self) -> Self {
        TileId::new(self.z | 0x40, self.x, self.y)
    }
}
