//! Map data sources: satellite imagery, elevation, and OSM vector tiles, with
//! decoded in-memory caches, plus the worker pool that builds tiles off-thread.
use crate::fetch::Fetcher;
use crate::geo::{self, TileId};
use crate::mvt;
use anyhow::{anyhow, Result};
use bytemuck::{Pod, Zeroable};
use glam::DVec3;
use parking_lot::{Condvar, Mutex};
use std::collections::HashMap;
use std::sync::Arc;

pub const DATA_ZOOM: u8 = 14; // deepest zoom of elevation + vector tiles
pub const GRID: usize = 64; // cells per tile side
pub const GV: usize = GRID + 1;
pub const MASK: usize = 256;

pub struct ElevTile {
    /// (n + 2 pad) squared samples, row by row
    pub h: Vec<f32>,
    /// samples across the tile itself
    pub n: usize,
    /// a border of this many samples taken from the neighbouring tiles, so
    /// heights interpolate smoothly across tile edges (0 or 1)
    pub pad: usize,
}
impl ElevTile {
    pub fn sample(&self, u: f64, v: f64) -> f32 {
        let w = self.n + 2 * self.pad;
        let (n, pad) = (self.n as f64, self.pad as f64);
        let x = (u * n - 0.5 + pad).clamp(0.0, w as f64 - 1.0);
        let y = (v * n - 0.5 + pad).clamp(0.0, w as f64 - 1.0);
        let (x0, y0) = (x.floor() as usize, y.floor() as usize);
        let (x1, y1) = ((x0 + 1).min(w - 1), (y0 + 1).min(w - 1));
        let (fx, fy) = ((x - x0 as f64) as f32, (y - y0 as f64) as f32);
        let g = |x: usize, y: usize| self.h[y * w + x];
        (g(x0, y0) * (1.0 - fx) + g(x1, y0) * fx) * (1.0 - fy) + (g(x0, y1) * (1.0 - fx) + g(x1, y1) * fx) * fy
    }
}

pub struct BuildingPoly {
    /// Exterior ring first, then holes; tile units.
    pub rings: Vec<Vec<[f32; 2]>>,
    pub height: f32,
    pub min_height: f32,
    /// Colour tagged in OpenStreetMap, when someone bothered.
    pub colour: Option<[u8; 3]>,
}

// Road classes, most to least important.
pub const RC_MOTORWAY: u8 = 0;
pub const RC_TRUNK: u8 = 1;
pub const RC_PRIMARY: u8 = 2;
pub const RC_SECONDARY: u8 = 3;
pub const RC_TERTIARY: u8 = 4;
pub const RC_MINOR: u8 = 5;
pub const RC_SERVICE: u8 = 6;
pub const RC_TRACK: u8 = 7;
pub const RC_PATH: u8 = 8;
pub const RC_RAIL: u8 = 9;

pub struct RoadLine {
    pub pts: Vec<[f32; 2]>,
    pub class: u8,
    /// 0 none, 1 bridge, 2 tunnel
    pub brunnel: u8,
    pub oneway: bool,
    pub ramp: bool,
    pub layer: i8,
    /// service roads: 1 = alley / driveway / parking aisle; paths: 1 = wide pedestrian way
    pub sub: u8,
}

// Land cover kinds.
pub const LC_WOOD: u8 = 1;
pub const LC_SCRUB: u8 = 2;
pub const LC_GRASS: u8 = 3;
pub const LC_FARM: u8 = 4;
pub const LC_WET: u8 = 5;
pub const LC_BARE: u8 = 6; // sand, ice
pub const LC_ROCK: u8 = 7;
// Land use kinds.
pub const LU_RESIDENTIAL: u8 = 1;
pub const LU_COMMERCIAL: u8 = 2;
pub const LU_INDUSTRIAL: u8 = 3;
pub const LU_OPEN: u8 = 4; // pitches, playgrounds: nothing grows or gets built here

/// A runway or taxiway centreline.
pub struct AeroLine {
    pub pts: Vec<[f32; 2]>,
    pub runway: bool,
    /// "10L/28R", or a taxiway's letter
    pub name: String,
    /// metres, from the airport database (runways) or the size of the airport
    pub width: f32,
    pub paved: bool,
}

// Airfield polygon kinds.
pub const AP_APRON: u8 = 1;
pub const AP_RUNWAY: u8 = 2;
pub const AP_TAXIWAY: u8 = 3;
pub const AP_HELIPAD: u8 = 4;
/// the airport's whole grounds (inside the fence)
pub const AP_GROUNDS: u8 = 5;

#[derive(Default)]
pub struct VecTile {
    pub extent: f32,
    /// (class, rings): 1 ocean, 2 lake/pond, 3 river
    pub water: Vec<(u8, Vec<Vec<[f32; 2]>>)>,
    pub buildings: Vec<BuildingPoly>,
    pub roads: Vec<RoadLine>,
    pub landcover: Vec<(u8, Vec<Vec<[f32; 2]>>)>,
    pub landuse: Vec<(u8, Vec<Vec<[f32; 2]>>)>,
    pub aero_lines: Vec<AeroLine>,
    /// (kind, rings) of aprons and other paved areas, one polygon each
    pub aero_polys: Vec<(u8, Vec<Vec<[f32; 2]>>)>,
    /// gate positions and their numbers
    pub gates: Vec<([f32; 2], String)>,
}
impl VecTile {
    pub fn has_aero(&self) -> bool {
        !self.aero_lines.is_empty() || self.aero_polys.iter().any(|p| p.0 != AP_GROUNDS)
    }
}

/// CSS-ish colour as tagged on OSM buildings: #rgb, #rrggbb or a common name.
fn parse_colour(s: &str) -> Option<[u8; 3]> {
    let s = s.trim().to_ascii_lowercase();
    if let Some(h) = s.strip_prefix('#') {
        let v = u32::from_str_radix(h, 16).ok()?;
        return match h.len() {
            6 => Some([(v >> 16) as u8, (v >> 8) as u8, v as u8]),
            3 => Some([((v >> 8) & 15) as u8 * 17, ((v >> 4) & 15) as u8 * 17, (v & 15) as u8 * 17]),
            _ => None,
        };
    }
    Some(match s.as_str() {
        "white" => [240, 240, 236],
        "black" => [40, 40, 42],
        "grey" | "gray" => [140, 140, 140],
        "lightgrey" | "lightgray" | "silver" => [196, 196, 196],
        "darkgrey" | "darkgray" => [90, 90, 92],
        "red" => [170, 60, 50],
        "maroon" | "darkred" => [120, 45, 40],
        "brown" => [125, 85, 60],
        "orange" => [215, 135, 70],
        "yellow" => [225, 200, 110],
        "beige" | "cream" | "ivory" => [226, 214, 186],
        "tan" | "sand" => [200, 175, 135],
        "pink" => [225, 180, 180],
        "green" => [100, 140, 95],
        "blue" => [95, 130, 175],
        "lightblue" => [160, 195, 220],
        _ => return None,
    })
}

struct Lru<T> {
    map: HashMap<TileId, (Arc<T>, u64)>,
    tick: u64,
    cap: usize,
}
impl<T> Lru<T> {
    fn new(cap: usize) -> Self {
        Self { map: HashMap::new(), tick: 0, cap }
    }
    fn get(&mut self, id: TileId) -> Option<Arc<T>> {
        self.tick += 1;
        let t = self.tick;
        self.map.get_mut(&id).map(|e| {
            e.1 = t;
            e.0.clone()
        })
    }
    fn put(&mut self, id: TileId, v: Arc<T>) {
        if self.map.len() >= self.cap {
            if let Some(old) = self.map.iter().min_by_key(|(_, e)| e.1).map(|(k, _)| *k) {
                self.map.remove(&old);
            }
        }
        self.tick += 1;
        self.map.insert(id, (v, self.tick));
    }
}

pub struct Sources {
    pub fetch: Fetcher,
    elev: Mutex<Lru<ElevTile>>,
    raw: Mutex<Lru<ElevTile>>,
    /// airfields that have been put together (see airports.rs)
    pub fields: Mutex<HashMap<u32, Arc<crate::airports::Geo>>>,
    vec: Mutex<Lru<VecTile>>,
    vec_url: Mutex<Option<String>>,
    pub imagery_url: String,
    /// optional Google satellite imagery, used in place of `imagery_url` when on
    pub google: Arc<crate::gmaps::Google>,
}

impl Sources {
    pub fn new(fetch: Fetcher, imagery_url: String) -> Self {
        Self::with_google(fetch, imagery_url, Arc::new(crate::gmaps::Google::disabled()))
    }

    pub fn with_google(fetch: Fetcher, imagery_url: String, google: Arc<crate::gmaps::Google>) -> Self {
        Self { fetch, elev: Mutex::new(Lru::new(96)), raw: Mutex::new(Lru::new(160)), fields: Mutex::new(HashMap::new()), vec: Mutex::new(Lru::new(96)), vec_url: Mutex::new(None), imagery_url, google }
    }

    /// Satellite image for a tile: Google's when that is on (and has the
    /// tile), the default imagery otherwise.
    pub fn imagery(&self, id: TileId) -> Result<image::RgbaImage> {
        if self.google.enabled() && !self.fetch.offline {
            match self.google.tile(&self.fetch.root, id.z, id.x, id.y).and_then(|d| Ok(image::load_from_memory(&d)?.to_rgba8())) {
                Ok(img) => return Ok(img),
                Err(e) => log::debug!("google tile {id:?}: {e}"),
            }
        }
        let url = self.imagery_url.replace("{z}", &id.z.to_string()).replace("{x}", &id.x.to_string()).replace("{y}", &id.y.to_string());
        let data = self.fetch.get_cached(&format!("img/{}/{}/{}.jpg", id.z, id.x, id.y), &url)?;
        Ok(image::load_from_memory(&data)?.to_rgba8())
    }

    /// Place names and borders to lay over the imagery (transparent PNG,
    /// Esri's reference layer that matches the default imagery).
    pub fn labels(&self, id: TileId) -> Result<image::RgbaImage> {
        let url = format!("https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{}/{}/{}", id.z, id.y, id.x);
        let data = self.fetch.get_cached(&format!("lbl/{}/{}/{}.png", id.z, id.x, id.y), &url)?;
        Ok(image::load_from_memory(&data)?.to_rgba8())
    }

    /// Elevation as the data has it.
    pub fn elev_raw(&self, id: TileId) -> Result<Arc<ElevTile>> {
        if let Some(t) = self.raw.lock().get(id) {
            return Ok(t);
        }
        let url = format!("https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{}/{}/{}.png", id.z, id.x, id.y);
        let data = self.fetch.get_cached(&format!("elev/{}/{}/{}.png", id.z, id.x, id.y), &url)?;
        let img = image::load_from_memory(&data)?.to_rgb8();
        let n = img.width() as usize;
        let h = img.pixels().map(|p| p[0] as f32 * 256.0 + p[1] as f32 + p[2] as f32 / 256.0 - 32768.0).collect();
        let t = Arc::new(ElevTile { h, n, pad: 0 });
        self.raw.lock().put(id, t.clone());
        Ok(t)
    }

    fn vec_template(&self) -> Result<String> {
        if let Some(u) = self.vec_url.lock().clone() {
            return Ok(u);
        }
        // OpenFreeMap publishes the current planet build in a TileJSON document.
        let u = self
            .fetch
            .get("https://tiles.openfreemap.org/planet")
            .ok()
            .and_then(|d| serde_json::from_slice::<serde_json::Value>(&d).ok())
            .and_then(|j| j["tiles"][0].as_str().map(String::from))
            .ok_or_else(|| anyhow!("no vector tile endpoint"))?;
        *self.vec_url.lock() = Some(u.clone());
        Ok(u)
    }

    pub fn vector(&self, id: TileId) -> Result<Arc<VecTile>> {
        if let Some(t) = self.vec.lock().get(id) {
            return Ok(t);
        }
        let key = format!("vec/{}/{}/{}.pbf", id.z, id.x, id.y);
        let data = match std::fs::read(self.fetch.root.join(&key)) {
            Ok(d) if !d.is_empty() => d,
            _ => {
                let url = self.vec_template()?.replace("{z}", &id.z.to_string()).replace("{x}", &id.x.to_string()).replace("{y}", &id.y.to_string());
                self.fetch.get_cached(&key, &url)?
            }
        };
        let mut layers = mvt::decode(&data, &["water", "building", "transportation", "landcover", "landuse", "aeroway"]);
        let mut vt = VecTile { extent: 4096.0, ..Default::default() };
        if let Some(l) = layers.remove("water") {
            vt.extent = l.extent;
            for f in l.features {
                if f.kind != 3 {
                    continue;
                }
                let class = match f.props.get("class").and_then(|v| v.str()) {
                    Some("ocean") => 1,
                    Some("river") => 3,
                    _ => 2,
                };
                vt.water.push((class, f.rings));
            }
        }
        if let Some(l) = layers.remove("building") {
            vt.extent = l.extent;
            for f in l.features {
                if f.kind != 3 || f.rings.is_empty() {
                    continue;
                }
                if f.props.get("hide_3d").is_some() {
                    continue;
                }
                let height = f.props.get("render_height").and_then(|v| v.num()).unwrap_or(6.0) as f32;
                let min_height = f.props.get("render_min_height").and_then(|v| v.num()).unwrap_or(0.0) as f32;
                let colour = f.props.get("colour").and_then(|v| v.str()).and_then(parse_colour);
                // Split multipolygons: a ring wound like the first one starts a new polygon.
                let sign = mvt::signed_area(&f.rings[0]).signum();
                let mut cur: Vec<Vec<[f32; 2]>> = Vec::new();
                for r in f.rings {
                    if r.len() < 3 {
                        continue;
                    }
                    if mvt::signed_area(&r).signum() == sign && !cur.is_empty() {
                        vt.buildings.push(BuildingPoly { rings: std::mem::take(&mut cur), height, min_height, colour });
                    }
                    cur.push(r);
                }
                if !cur.is_empty() {
                    vt.buildings.push(BuildingPoly { rings: cur, height, min_height, colour });
                }
            }
        }
        if let Some(l) = layers.remove("transportation") {
            let k = vt.extent / l.extent;
            for f in l.features {
                if f.kind != 2 {
                    continue;
                }
                let st = |k: &str| f.props.get(k).and_then(|v| v.str());
                let sub = st("subclass").unwrap_or("");
                let (class, sub) = match st("class").unwrap_or("") {
                    "motorway" => (RC_MOTORWAY, 0),
                    "trunk" => (RC_TRUNK, 0),
                    "primary" => (RC_PRIMARY, 0),
                    "secondary" => (RC_SECONDARY, 0),
                    "tertiary" => (RC_TERTIARY, 0),
                    "minor" | "raceway" => (RC_MINOR, 0),
                    "service" | "busway" => (RC_SERVICE, st("service").is_some() as u8),
                    "track" => (RC_TRACK, 0),
                    "path" => match sub {
                        "platform" | "corridor" | "steps" => continue,
                        "pedestrian" => (RC_PATH, 1),
                        _ => (RC_PATH, 0),
                    },
                    "rail" => (RC_RAIL, 0),
                    _ => continue,
                };
                let brunnel = match st("brunnel") {
                    Some("bridge") => 1,
                    Some("tunnel") => 2,
                    _ => 0,
                };
                let num = |k: &str| f.props.get(k).and_then(|v| v.num()).unwrap_or(0.0);
                let (oneway, ramp, layer) = (num("oneway") != 0.0, num("ramp") != 0.0, num("layer") as i8);
                let rev = num("oneway") < 0.0;
                for mut pts in f.rings {
                    if pts.len() < 2 {
                        continue;
                    }
                    for p in pts.iter_mut() {
                        *p = [p[0] * k, p[1] * k];
                    }
                    if rev {
                        pts.reverse();
                    }
                    vt.roads.push(RoadLine { pts, class, brunnel, oneway, ramp, layer, sub });
                }
            }
        }
        for (name, is_cover) in [("landcover", true), ("landuse", false)] {
            let Some(l) = layers.remove(name) else { continue };
            let k = vt.extent / l.extent;
            for f in l.features {
                if f.kind != 3 {
                    continue;
                }
                let class = f.props.get("class").and_then(|v| v.str()).unwrap_or("");
                let sub = f.props.get("subclass").and_then(|v| v.str()).unwrap_or("");
                let kind = if is_cover {
                    match (class, sub) {
                        ("wood", _) => LC_WOOD,
                        (_, "scrub") | (_, "heath") => LC_SCRUB,
                        ("grass", _) => LC_GRASS,
                        ("farmland", _) => LC_FARM,
                        ("wetland", _) => LC_WET,
                        ("rock", _) => LC_ROCK,
                        ("sand", _) | ("ice", _) => LC_BARE,
                        _ => continue,
                    }
                } else {
                    match class {
                        "residential" | "suburb" | "neighbourhood" | "quarter" => LU_RESIDENTIAL,
                        "commercial" | "retail" => LU_COMMERCIAL,
                        "industrial" | "railway" | "garages" => LU_INDUSTRIAL,
                        "pitch" | "playground" | "track" | "stadium" => LU_OPEN,
                        _ => continue,
                    }
                };
                let rings = f.rings.into_iter().map(|r| r.into_iter().map(|p| [p[0] * k, p[1] * k]).collect()).collect();
                if is_cover {
                    vt.landcover.push((kind, rings));
                } else {
                    vt.landuse.push((kind, rings));
                }
            }
        }
        if let Some(l) = layers.remove("aeroway") {
            let k = vt.extent / l.extent;
            let db = crate::airports::Db::get();
            let unit_m = id.width_m() / vt.extent as f64;
            for f in l.features {
                let class = f.props.get("class").and_then(|v| v.str()).unwrap_or("");
                let name = f.props.get("ref").and_then(|v| v.str()).unwrap_or("").to_string();
                let scale = |r: Vec<[f32; 2]>| -> Vec<[f32; 2]> { r.into_iter().map(|p| [p[0] * k, p[1] * k]).collect() };
                match (f.kind, class) {
                    (2, "runway") | (2, "taxiway") => {
                        for pts in f.rings {
                            if pts.len() < 2 {
                                continue;
                            }
                            let pts = scale(pts);
                            let (a, b) = (pts[0], pts[pts.len() - 1]);
                            let (lat, lon) = id.lat_lon(((a[0] + b[0]) * 0.5 / vt.extent) as f64, ((a[1] + b[1]) * 0.5 / vt.extent) as f64);
                            let (lat, lon) = (lat.to_degrees(), lon.to_degrees());
                            if class == "runway" {
                                let brg = ((b[0] - a[0]) as f64 * unit_m).atan2(-(b[1] - a[1]) as f64 * unit_m).to_degrees() as f32;
                                let m = db.runway_meta(lat, lon, brg, &name);
                                vt.aero_lines.push(AeroLine { pts, runway: true, name: name.clone(), width: m.width, paved: m.paved });
                            } else {
                                let kind = db.owner(lat, lon).map(|i| db.apts[i as usize].kind).unwrap_or(0);
                                vt.aero_lines.push(AeroLine { pts, runway: false, name: name.clone(), width: crate::airports::taxi_width(kind), paved: true });
                            }
                        }
                    }
                    (3, "apron") | (3, "runway") | (3, "taxiway") | (3, "helipad") | (3, "aerodrome") => {
                        let kind = match class { "apron" => AP_APRON, "runway" => AP_RUNWAY, "taxiway" => AP_TAXIWAY, "aerodrome" => AP_GROUNDS, _ => AP_HELIPAD };
                        if f.rings.is_empty() {
                            continue;
                        }
                        let sign = mvt::signed_area(&f.rings[0]).signum();
                        let mut cur: Vec<Vec<[f32; 2]>> = Vec::new();
                        for r in f.rings {
                            if r.len() < 3 {
                                continue;
                            }
                            if mvt::signed_area(&r).signum() == sign && !cur.is_empty() {
                                vt.aero_polys.push((kind, std::mem::take(&mut cur)));
                            }
                            cur.push(scale(r));
                        }
                        if !cur.is_empty() {
                            vt.aero_polys.push((kind, cur));
                        }
                    }
                    (1, "gate") => {
                        if let Some(p) = f.rings.first().and_then(|r| r.first()) {
                            vt.gates.push(([p[0] * k, p[1] * k], name));
                        }
                    }
                    _ => {}
                }
            }
        }
        let t = Arc::new(vt);
        self.vec.lock().put(id, t.clone());
        Ok(t)
    }

    /// Elevation as the terrain is built: the data as it comes, except that
    /// airfield pavement is levelled (see `level_airfield`).
    pub fn elev(&self, id: TileId) -> Result<Arc<ElevTile>> {
        if let Some(t) = self.elev.lock().get(id) {
            return Ok(t);
        }
        let raw = self.elev_raw(id)?;
        // Levelling reaches a little past the pavement, so a tile is levelled
        // when its neighbours have airfield pavement too (or there would be a
        // step at the border).
        let aero = id.z == DATA_ZOOM && {
            let size = 1i64 << id.z;
            (-1..=1i64).any(|dy| (-1..=1i64).any(|dx| {
                let (x, y) = ((id.x as i64 + dx).rem_euclid(size), id.y as i64 + dy);
                y >= 0 && y < size && self.vector(TileId::new(id.z, x as u32, y as u32)).is_ok_and(|vt| vt.has_aero())
            }))
        };
        let t = Arc::new(if aero { self.level_airfield(id, &raw) } else { self.padded(id, &raw) });
        self.elev.lock().put(id, t.clone());
        Ok(t)
    }

    /// Runways, taxiways and aprons are smooth; the elevation data under them
    /// is not (bumps of a metre every fifty, enough to throw an aircraft
    /// about on its take-off run). So pavement takes the average height of
    /// the pavement around it, a ~40 m Gaussian over paved samples only (the
    /// water or hillside beside a runway does not drag its edge away), and
    /// the ground eases back to the raw data over the 20 m beyond the edge.
    /// Works across tile borders: both sides compute the same function from
    /// the same neighbourhood.
    fn level_airfield(&self, id: TileId, raw: &ElevTile) -> ElevTile {
        let n = raw.n;
        let m = 3 * n;
        let size = 1i64 << id.z;
        let mean = raw.h.iter().sum::<f32>() / raw.h.len() as f32;
        let mut h = vec![mean; m * m];
        let mut mask = vec![0f32; m * m];
        let px_m = (id.width_m() / n as f64) as f32;
        // runway pieces in mosaic pixels: ends, width (m)
        let mut runs: Vec<(glam::Vec2, glam::Vec2, f32)> = Vec::new();
        for ty in 0..3usize {
            for tx in 0..3usize {
                let (x, y) = ((id.x as i64 + tx as i64 - 1).rem_euclid(size), id.y as i64 + ty as i64 - 1);
                if y < 0 || y >= size {
                    continue;
                }
                let nid = TileId::new(id.z, x as u32, y as u32);
                let e = if nid == id { None } else { self.elev_raw(nid).ok() };
                let e: &ElevTile = match &e { Some(e) if e.n == n => e, _ if nid == id => raw, _ => continue };
                for j in 0..n {
                    h[(ty * n + j) * m + tx * n..(ty * n + j) * m + tx * n + n].copy_from_slice(&e.h[j * n..(j + 1) * n]);
                }
                let Ok(vt) = self.vector(nid) else { continue };
                if !vt.has_aero() {
                    continue;
                }
                let mut tile = vec![0u8; n * n];
                let k = n as f32 / vt.extent;
                for (_, rings) in vt.aero_polys.iter().filter(|p| p.0 != AP_GROUNDS) {
                    let px: Vec<Vec<[f32; 2]>> = rings.iter().map(|r| r.iter().map(|p| [p[0] * k, p[1] * k]).collect()).collect();
                    mvt::fill_rings(&mut tile, n, n, &px, 1);
                }
                for l in &vt.aero_lines {
                    let hw = (l.width * 0.5 + 4.0) / px_m;
                    thick_line(&l.pts, k, hw, n, |i| tile[i] = 1);
                    if l.runway {
                        let o = glam::Vec2::new((tx * n) as f32, (ty * n) as f32);
                        runs.push((o + glam::Vec2::from(l.pts[0]) * k, o + glam::Vec2::from(l.pts[l.pts.len() - 1]) * k, l.width));
                    }
                }
                for j in 0..n {
                    for i in 0..n {
                        mask[(ty * n + j) * m + tx * n + i] = tile[j * n + i] as f32;
                    }
                }
            }
        }
        // Three box passes make a near-Gaussian.
        let blur = |src: &[f32], r: usize, passes: usize| -> Vec<f32> {
            let mut a = src.to_vec();
            let mut b = vec![0f32; m * m];
            let w = (2 * r + 1) as f32;
            for _ in 0..passes {
                for horiz in [true, false] {
                    for line in 0..m {
                        let at = |k: usize| if horiz { line * m + k } else { k * m + line };
                        let mut s: f32 = (0..=r.min(m - 1)).map(|k| a[at(k)]).sum::<f32>() + a[at(0)] * r as f32;
                        for k in 0..m {
                            b[at(k)] = s / w;
                            let add = a[at((k + r + 1).min(m - 1))];
                            let sub = a[at(k.saturating_sub(r))];
                            s += add - sub;
                        }
                    }
                    std::mem::swap(&mut a, &mut b);
                }
            }
            a
        };
        let r = ((42.0 / px_m / 1.1).round() as usize).clamp(2, 12);
        let hm: Vec<f32> = h.iter().zip(&mask).map(|(h, k)| h * k).collect();
        let (bh, bm) = (blur(&hm, r, 3), blur(&mask, r, 3));
        let soft = blur(&mask, ((15.0 / px_m).round() as usize).clamp(1, 4), 2);
        let mut lev = h.clone();
        for k in 0..m * m {
            if bm[k] > 1e-3 {
                let w = (soft[k] * 2.2).min(1.0);
                lev[k] += (bh[k] / bm[k] - lev[k]) * w * w * (3.0 - 2.0 * w);
            }
        }
        // Runways want more than that: the data can put a hump of several
        // metres along one, and a runway climbs or falls by a percent or so
        // at most, and changes its slope slowly. Each runway takes the
        // profile along its centreline, smoothed over ~250 m, right across
        // its width; the ground beside it eases back over the next 40 m.
        runs.sort_by(|a, b| (b.1 - b.0).length().total_cmp(&(a.1 - a.0).length()));
        let mut joined: Vec<(glam::Vec2, glam::Vec2, f32)> = Vec::new();
        for (a, b, w) in runs {
            let d = (b - a).normalize_or_zero();
            let hit = joined.iter_mut().find(|j| {
                let jd = (j.1 - j.0).normalize_or_zero();
                let (s0, s1) = ((a - j.0).dot(jd), (b - j.0).dot(jd));
                jd.dot(d).abs() > 0.997 && (a - j.0).perp_dot(jd).abs() < 2.5 && (b - j.0).perp_dot(jd).abs() < 2.5 && s0.min(s1) < (j.1 - j.0).length() + 8.0 && s0.max(s1) > -8.0
            });
            match hit {
                Some(j) => {
                    let jd = (j.1 - j.0).normalize_or_zero();
                    let (s0, s1) = ((a - j.0).dot(jd), (b - j.0).dot(jd));
                    let (lo, hi) = (s0.min(s1).min(0.0), s0.max(s1).max((j.1 - j.0).length()));
                    (j.0, j.1) = (j.0 + jd * lo, j.0 + jd * hi);
                }
                None => joined.push((a, b, w)),
            }
        }
        let sample = |f: &[f32], p: glam::Vec2| {
            let (x, y) = ((p.x - 0.5).clamp(0.0, m as f32 - 1.001), (p.y - 0.5).clamp(0.0, m as f32 - 1.001));
            let (x0, y0) = (x as usize, y as usize);
            let (fx, fy) = (x - x0 as f32, y - y0 as f32);
            (f[y0 * m + x0] * (1.0 - fx) + f[y0 * m + x0 + 1] * fx) * (1.0 - fy) + (f[(y0 + 1) * m + x0] * (1.0 - fx) + f[(y0 + 1) * m + x0 + 1] * fx) * fy
        };
        let sigma = 250.0 / px_m;
        for (a, b, width) in joined {
            let len = (b - a).length();
            if len * px_m < 150.0 {
                continue;
            }
            let d = (b - a) / len;
            let ns = len.ceil() as usize + 1;
            let mut prof: Vec<f32> = (0..ns).map(|k| sample(&lev, a + d * (k as f32).min(len))).collect();
            // The steady slope stays (an altiport's runway is a ramp); only
            // what departs from it is smoothed.
            let (mx, my) = ((ns - 1) as f32 * 0.5, prof.iter().sum::<f32>() / ns as f32);
            let slope = prof.iter().enumerate().map(|(k, v)| (k as f32 - mx) * (v - my)).sum::<f32>() / (0..ns).map(|k| (k as f32 - mx).powi(2)).sum::<f32>().max(1.0);
            let line = |k: usize| my + slope * (k as f32 - mx);
            for (k, v) in prof.iter_mut().enumerate() {
                *v -= line(k);
            }
            let reach = (sigma * 3.0) as i64;
            let smooth: Vec<f32> = (0..ns as i64).map(|k| {
                let (mut s, mut ws) = (0.0, 0.0);
                for o in -reach..=reach {
                    if k + o >= 0 && k + o < ns as i64 {
                        let w = (-0.5 * (o as f32 / sigma).powi(2)).exp();
                        s += prof[(k + o) as usize] * w;
                        ws += w;
                    }
                }
                s / ws + line(k as usize)
            }).collect();
            let (half, fade) = ((width * 0.5 + 8.0) / px_m, 40.0 / px_m);
            let pad = half + fade + 1.0;
            let (lo, hi) = (a.min(b) - pad, a.max(b) + pad);
            for y in (lo.y.floor().max(0.0) as usize)..(hi.y.ceil().min(m as f32) as usize) {
                for x in (lo.x.floor().max(0.0) as usize)..(hi.x.ceil().min(m as f32) as usize) {
                    let p = glam::Vec2::new(x as f32 + 0.5, y as f32 + 0.5);
                    let t = (p - a).dot(d).clamp(0.0, len);
                    let dist = (a + d * t - p).length();
                    if dist < half + fade {
                        let w = 1.0 - ((dist - half) / fade).clamp(0.0, 1.0);
                        let (k0, f) = (t as usize, t.fract());
                        let target = smooth[k0] * (1.0 - f) + smooth[(k0 + 1).min(ns - 1)] * f;
                        lev[y * m + x] += (target - lev[y * m + x]) * w * w * (3.0 - 2.0 * w);
                    }
                }
            }
        }
        // The tile and a one-sample border from the mosaic around it.
        let w = n + 2;
        let mut out = vec![0f32; w * w];
        for j in 0..w {
            for i in 0..w {
                out[j * w + i] = lev[(n + j - 1) * m + n + i - 1];
            }
        }
        ElevTile { h: out, n, pad: 1 }
    }
}

impl Sources {
    /// A raw tile with a one-sample border from its neighbours (its own edge
    /// repeated where a neighbour is missing).
    fn padded(&self, id: TileId, raw: &ElevTile) -> ElevTile {
        let n = raw.n;
        let w = n + 2;
        let size = 1i64 << id.z;
        let mut out = vec![0f32; w * w];
        let nb = |dx: i64, dy: i64| -> Option<Arc<ElevTile>> {
            let (x, y) = ((id.x as i64 + dx).rem_euclid(size), id.y as i64 + dy);
            if y < 0 || y >= size {
                return None;
            }
            self.elev_raw(TileId::new(id.z, x as u32, y as u32)).ok().filter(|t| t.n == n)
        };
        let tiles: Vec<Option<Arc<ElevTile>>> = (-1..=1i64).flat_map(|dy| (-1..=1i64).map(move |dx| (dx, dy))).map(|(dx, dy)| if dx == 0 && dy == 0 { None } else { nb(dx, dy) }).collect();
        for j in 0..w {
            for i in 0..w {
                // Mosaic coordinates of this sample.
                let (gx, gy) = (i as i64 - 1, j as i64 - 1);
                let (tx, ty) = (gx.div_euclid(n as i64), gy.div_euclid(n as i64));
                let (lx, ly) = (gx.rem_euclid(n as i64) as usize, gy.rem_euclid(n as i64) as usize);
                let v = if tx == 0 && ty == 0 {
                    raw.h[ly * n + lx]
                } else {
                    match &tiles[((ty + 1) * 3 + tx + 1) as usize] {
                        Some(t) => t.h[ly * n + lx],
                        None => raw.h[gy.clamp(0, n as i64 - 1) as usize * n + gx.clamp(0, n as i64 - 1) as usize],
                    }
                };
                out[j * w + i] = v;
            }
        }
        ElevTile { h: out, n, pad: 1 }
    }
}

/// Visit every pixel of an n x n raster within `hw` pixels of a polyline
/// (given in tile units; `k` scales them to pixels).
pub fn thick_line(pts: &[[f32; 2]], k: f32, hw: f32, n: usize, mut hit: impl FnMut(usize)) {
    for s in pts.windows(2) {
        let (a, b) = ([s[0][0] * k, s[0][1] * k], [s[1][0] * k, s[1][1] * k]);
        let (x0, x1) = ((a[0].min(b[0]) - hw).floor().max(0.0) as i64, (a[0].max(b[0]) + hw).ceil().min(n as f32) as i64);
        let (y0, y1) = ((a[1].min(b[1]) - hw).floor().max(0.0) as i64, (a[1].max(b[1]) + hw).ceil().min(n as f32) as i64);
        let d = [b[0] - a[0], b[1] - a[1]];
        let dd = (d[0] * d[0] + d[1] * d[1]).max(1e-6);
        for y in y0..y1 {
            for x in x0..x1 {
                let p = [x as f32 + 0.5 - a[0], y as f32 + 0.5 - a[1]];
                let t = ((p[0] * d[0] + p[1] * d[1]) / dd).clamp(0.0, 1.0);
                let (ex, ey) = (p[0] - d[0] * t, p[1] - d[1] * t);
                if ex * ex + ey * ey < hw * hw {
                    hit(y as usize * n + x as usize);
                }
            }
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct TerrainVertex {
    pub pos: [f32; 3],
    pub depth: f32,
    pub nrm: [f32; 3],
    pub skirt: f32,
    pub uv: [f32; 2],
}

pub struct TileData {
    pub id: TileId,
    pub center: DVec3,
    pub radius: f64,
    pub verts: Vec<TerrainVertex>,
    /// Surface height (water surface where there is water) on the GV x GV grid.
    pub heights: Vec<f32>,
    pub water: Vec<bool>,
    /// on the same grid: how much of the ocean's long waves the water shows
    /// (ocean.rs `sea_state`, as the shaders'), its two longer cascades; 0 on land
    pub sea: Vec<[f32; 2]>,
    pub imagery: Vec<Vec<u8>>, // mip chain, rgba
    pub img_size: u32,
    pub mask: Vec<Vec<u8>>, // mip chain, rgba (see ground_mask)
    /// imagery source generation the tile was made with (see `Google::generation`)
    pub imagery_gen: u32,
}

pub fn mip_chain(mut data: Vec<u8>, mut size: usize, ch: usize) -> Vec<Vec<u8>> {
    let mut out = Vec::new();
    while size > 1 {
        let half = size / 2;
        let mut next = vec![0u8; half * half * ch];
        for y in 0..half {
            for x in 0..half {
                for c in 0..ch {
                    let s = |dx: usize, dy: usize| data[((y * 2 + dy) * size + x * 2 + dx) * ch + c] as u32;
                    next[(y * half + x) * ch + c] = ((s(0, 0) + s(1, 0) + s(0, 1) + s(1, 1) + 2) / 4) as u8;
                }
            }
        }
        out.push(std::mem::replace(&mut data, next));
        size = half;
    }
    out.push(data);
    out
}

/// Rasterise the water polygons covering `id` into an RG mask
/// (R = coverage, G = water class * 80).
fn water_mask(vt: &VecTile, ox: f64, oy: f64, scale: f64) -> Vec<u8> {
    const SS: usize = MASK * 2;
    let mut cls = vec![0u8; SS * SS];
    let k = SS as f64 / scale / vt.extent as f64;
    let (ox, oy) = (ox * vt.extent as f64, oy * vt.extent as f64);
    // Lakes and rivers go on top of ocean so inland classes win where both exist.
    for pass in [1u8, 2, 3] {
        for (class, rings) in vt.water.iter().filter(|w| w.0 == pass) {
            let px: Vec<Vec<[f32; 2]>> =
                rings.iter().map(|r| r.iter().map(|p| [((p[0] as f64 - ox) * k) as f32, ((p[1] as f64 - oy) * k) as f32]).collect()).collect();
            let (mut lo, mut hi) = ([f32::MAX; 2], [f32::MIN; 2]);
            for p in px.iter().flatten() {
                lo = [lo[0].min(p[0]), lo[1].min(p[1])];
                hi = [hi[0].max(p[0]), hi[1].max(p[1])];
            }
            if hi[0] < 0.0 || hi[1] < 0.0 || lo[0] > SS as f32 || lo[1] > SS as f32 {
                continue;
            }
            let mut layer = vec![0u8; SS * SS];
            mvt::fill_rings(&mut layer, SS, SS, &px, *class);
            for (d, s) in cls.iter_mut().zip(&layer) {
                if *s != 0 {
                    *d = *s;
                }
            }
        }
    }
    let mut out = vec![0u8; MASK * MASK * 2];
    for y in 0..MASK {
        for x in 0..MASK {
            let s = [cls[y * 2 * SS + x * 2], cls[y * 2 * SS + x * 2 + 1], cls[(y * 2 + 1) * SS + x * 2], cls[(y * 2 + 1) * SS + x * 2 + 1]];
            let n = s.iter().filter(|c| **c != 0).count();
            out[(y * MASK + x) * 2] = (n * 255 / 4) as u8;
            out[(y * MASK + x) * 2 + 1] = s.iter().copied().max().unwrap_or(0) * 80;
        }
    }
    out
}

// Ground classes in the B channel of the tile mask (low nibble; the high
// nibble is a per-polygon seed). Land cover kinds map onto themselves.
pub const GC_PAVED: u8 = 8;
pub const GC_PITCH: u8 = 9;

/// The full RGBA tile mask: water (R coverage, G class * 80) from `water_mask`,
/// B = ground class | seed << 4, A = direction of the polygon's longest side
/// (0..255 over 0..pi, from east toward north): crop rows and mowing stripes
/// run along it.
fn ground_mask(vt: &VecTile, ox: f64, oy: f64, scale: f64, tile_m: f64) -> Vec<u8> {
    let water = water_mask(vt, ox, oy, scale);
    let mut out = vec![0u8; MASK * MASK * 4];
    for i in 0..MASK * MASK {
        out[i * 4] = water[i * 2];
        out[i * 4 + 1] = water[i * 2 + 1];
    }
    let k = MASK as f64 / scale / vt.extent as f64;
    let (ox, oy) = (ox * vt.extent as f64, oy * vt.extent as f64);
    let to_px = |p: &[f32; 2]| [((p[0] as f64 - ox) * k) as f32, ((p[1] as f64 - oy) * k) as f32];
    let mut scratch = vec![0u8; MASK * MASK];
    let mut stamp = |out: &mut [u8], rings: &[Vec<[f32; 2]>], class: u8| {
        let px: Vec<Vec<[f32; 2]>> = rings.iter().map(|r| r.iter().map(to_px).collect()).collect();
        let (mut lo, mut hi) = ([f32::MAX; 2], [f32::MIN; 2]);
        for p in px.iter().flatten() {
            lo = [lo[0].min(p[0]), lo[1].min(p[1])];
            hi = [hi[0].max(p[0]), hi[1].max(p[1])];
        }
        if px.is_empty() || hi[0] < 0.0 || hi[1] < 0.0 || lo[0] > MASK as f32 || lo[1] > MASK as f32 {
            return;
        }
        // Longest side of the outer ring.
        let r0 = &px[0];
        let (mut best, mut ang) = (0f32, 0f32);
        for i in 0..r0.len() {
            let (a, b) = (r0[i], r0[(i + 1) % r0.len()]);
            let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
            let l = dx * dx + dy * dy;
            if l > best {
                best = l;
                ang = (-dy).atan2(dx).rem_euclid(std::f32::consts::PI);
            }
        }
        let first = rings[0][0];
        let seed = ((first[0].to_bits().wrapping_mul(0x9e37_79b9) ^ first[1].to_bits().wrapping_mul(0x85eb_ca6b)) >> 28) as u8;
        let a = ((ang / std::f32::consts::PI * 256.0) as u32).min(255) as u8;
        mvt::fill_rings(&mut scratch, MASK, MASK, &px, 1);
        let (x0, y0) = (lo[0].max(0.0) as usize, lo[1].max(0.0) as usize);
        let (x1, y1) = ((hi[0].ceil().max(0.0) as usize + 1).min(MASK), (hi[1].ceil().max(0.0) as usize + 1).min(MASK));
        for y in y0..y1 {
            for x in x0..x1 {
                let i = y * MASK + x;
                if scratch[i] != 0 {
                    scratch[i] = 0;
                    out[i * 4 + 2] = class | seed << 4;
                    out[i * 4 + 3] = a;
                }
            }
        }
    };
    // Later kinds win where polygons overlap (as for tree placement).
    for want in [LC_GRASS, LC_FARM, LC_WET, LC_SCRUB, LC_WOOD, LC_BARE, LC_ROCK] {
        for (_, rings) in vt.landcover.iter().filter(|l| l.0 == want) {
            stamp(&mut out, rings, want);
        }
    }
    for (_, rings) in vt.landuse.iter().filter(|l| l.0 == LU_OPEN) {
        stamp(&mut out, rings, GC_PITCH);
    }
    // Airfield pavement, and mown strips where a runway is grass.
    let px_m = (tile_m / MASK as f64) as f32;
    for (_, rings) in vt.aero_polys.iter().filter(|p| p.0 != AP_GROUNDS) {
        stamp(&mut out, rings, GC_PAVED);
    }
    for l in &vt.aero_lines {
        let class = if l.paved { GC_PAVED } else { GC_PITCH };
        let px: Vec<[f32; 2]> = l.pts.iter().map(|p| [p[0] - ox as f32, p[1] - oy as f32]).collect();
        thick_line(&px, k as f32, (l.width * 0.5 + 1.0) / px_m, MASK, |i| out[i * 4 + 2] = class);
    }
    // Road surface: no grass, crops or rock relief under the road ribbons.
    for r in &vt.roads {
        if r.brunnel != 0 {
            continue;
        }
        let w_m = match r.class {
            RC_MOTORWAY => 13.0,
            RC_TRUNK => 11.0,
            RC_PRIMARY => 9.0,
            RC_SECONDARY => 8.0,
            RC_TERTIARY => 7.0,
            RC_MINOR => 6.0,
            RC_SERVICE => 4.0,
            RC_RAIL => 4.0,
            RC_TRACK => 3.0,
            _ => 2.0,
        };
        let hw = (w_m * 0.5 + 0.5) / px_m;
        for s in r.pts.windows(2) {
            let (a, b) = (to_px(&s[0]), to_px(&s[1]));
            let (x0, x1) = ((a[0].min(b[0]) - hw).floor().max(0.0) as i64, (a[0].max(b[0]) + hw).ceil().min(MASK as f32) as i64);
            let (y0, y1) = ((a[1].min(b[1]) - hw).floor().max(0.0) as i64, (a[1].max(b[1]) + hw).ceil().min(MASK as f32) as i64);
            let d = [b[0] - a[0], b[1] - a[1]];
            let dd = (d[0] * d[0] + d[1] * d[1]).max(1e-6);
            for y in y0..y1 {
                for x in x0..x1 {
                    let p = [x as f32 + 0.5 - a[0], y as f32 + 0.5 - a[1]];
                    let t = ((p[0] * d[0] + p[1] * d[1]) / dd).clamp(0.0, 1.0);
                    let (ex, ey) = (p[0] - d[0] * t, p[1] - d[1] * t);
                    if ex * ex + ey * ey < hw * hw {
                        let i = (y as usize * MASK + x as usize) * 4;
                        out[i + 2] = GC_PAVED;
                    }
                }
            }
        }
    }
    out
}

pub fn build_tile(src: &Sources, id: TileId) -> Result<TileData> {
    // read before the image, so a switch while this tile loads still refreshes it
    let imagery_gen = src.google.generation();
    let img = src.imagery(id)?;
    let img_size = img.width();
    if img_size != img.height() || !img_size.is_power_of_two() {
        return Err(anyhow!("unexpected imagery size"));
    }
    let (did, ox, oy, sc) = id.ancestor(DATA_ZOOM);
    let elev = src.elev(did).ok();
    // Coarser tiles near an airfield take their heights from the finer,
    // levelled data too, or the two would not meet where they join.
    let (clat0, clon0) = id.lat_lon(0.5, 0.5);
    let fine_elev = (12..DATA_ZOOM).contains(&id.z) && {
        let db = crate::airports::Db::get();
        db.near(clat0.to_degrees(), clon0.to_degrees(), id.width_m() * 0.75 + 5000.0).iter().any(|(i, _)| db.apts[*i as usize].any_paved())
    };
    let mut fine_cache: HashMap<TileId, Option<Arc<ElevTile>>> = HashMap::new();
    let mut height_at = |u: f64, v: f64| -> f32 {
        if fine_elev {
            let (lat, lon) = id.lat_lon(u, v);
            let t = TileId::at(lat, lon, DATA_ZOOM);
            if let Some(e) = fine_cache.entry(t).or_insert_with(|| src.elev(t).ok()) {
                let (fu, fv) = t.frac(lat, lon);
                return e.sample(fu, fv);
            }
        }
        elev.as_ref().map(|t| t.sample(ox + u * sc, oy + v * sc)).unwrap_or(0.0)
    };
    let mask = match src.vector(did) {
        Ok(vt) => ground_mask(&vt, ox, oy, sc, id.width_m()),
        // No vector data: fall back to "below sea level is ocean".
        Err(_) => {
            let mut m = vec![0u8; MASK * MASK * 4];
            if let Some(e) = &elev {
                for y in 0..MASK {
                    for x in 0..MASK {
                        let (u, v) = ((x as f64 + 0.5) / MASK as f64, (y as f64 + 0.5) / MASK as f64);
                        if e.sample(ox + u * sc, oy + v * sc) < -1.0 {
                            m[(y * MASK + x) * 4] = 255;
                            m[(y * MASK + x) * 4 + 1] = 80;
                        }
                    }
                }
            }
            m
        }
    };

    let (clat, clon) = id.lat_lon(0.5, 0.5);
    let center = geo::lla_to_pos(clat, clon, 0.0);
    // Grid with a one-cell border so normals are continuous across tiles.
    let e = GV + 2;
    let mut pts = vec![DVec3::ZERO; e * e];
    let mut hs = vec![(0f32, 0f32, false); e * e]; // surface height, raw elevation, water
    for j in 0..e {
        for i in 0..e {
            let (u, v) = ((i as f64 - 1.0) / GRID as f64, (j as f64 - 1.0) / GRID as f64);
            let (uc, vc) = (u.clamp(0.0, 1.0), v.clamp(0.0, 1.0));
            let raw = height_at(uc, vc);
            let (mx, my) = (((uc * MASK as f64) as usize).min(MASK - 1), ((vc * MASK as f64) as usize).min(MASK - 1));
            let water = mask[(my * MASK + mx) * 4] > 127;
            let h = if water { raw.max(0.0) } else { raw };
            let (lat, lon) = id.lat_lon(u, v);
            pts[j * e + i] = geo::lla_to_pos(lat, lon, h as f64) - center;
            hs[j * e + i] = (h, raw, water);
        }
    }
    let mask_mips = mip_chain(mask, MASK, 4);
    // Bilinear lookup in one level of the mask, as the GPU samples it.
    let mask_at = |level: usize, u: f64, v: f64, ch: usize| -> f32 {
        let n = (MASK >> level).max(1);
        let m = &mask_mips[level.min(mask_mips.len() - 1)];
        let (x, y) = (u * n as f64 - 0.5, v * n as f64 - 0.5);
        let (x0, y0) = (x.floor(), y.floor());
        let (fx, fy) = ((x - x0) as f32, (y - y0) as f32);
        let at = |i: f64, j: f64| m[((j.clamp(0.0, n as f64 - 1.0) as usize) * n + i.clamp(0.0, n as f64 - 1.0) as usize) * 4 + ch] as f32 / 255.0;
        (at(x0, y0) * (1.0 - fx) + at(x0 + 1.0, y0) * fx) * (1.0 - fy) + (at(x0, y0 + 1.0) * (1.0 - fx) + at(x0 + 1.0, y0 + 1.0) * fx) * fy
    };
    let mut verts = Vec::with_capacity(GV * GV + 4 * GV);
    let (mut heights, mut water) = (Vec::with_capacity(GV * GV), Vec::with_capacity(GV * GV));
    let mut sea = Vec::with_capacity(GV * GV);
    let mut radius = 0f64;
    for j in 1..=GV {
        for i in 1..=GV {
            let p = pts[j * e + i];
            let up = (p + center).normalize();
            let mut n = (pts[j * e + i + 1] - pts[j * e + i - 1]).cross(pts[(j - 1) * e + i] - pts[(j + 1) * e + i]);
            if n.length_squared() < 1e-9 || n.dot(up) <= 0.0 {
                n = up;
            }
            let n = n.normalize();
            let (h, raw, w) = hs[j * e + i];
            radius = radius.max(p.length());
            heights.push(h);
            water.push(w);
            let (u, v) = ((i - 1) as f64 / GRID as f64, (j - 1) as f64 / GRID as f64);
            sea.push(if w {
                let shore = mask_at(3, u, v, 0);
                let shore = ((shore - 0.5) / 0.5).clamp(0.0, 1.0);
                let shore = shore * shore * (3.0 - 2.0 * shore);
                let st = crate::ocean::sea_state(mask_at(1, u, v, 1) * 255.0 / 80.0, id.lat_lon(u, v).0.sin() as f32, shore);
                [st[0], st[1]]
            } else {
                [0.0; 2]
            });
            verts.push(TerrainVertex {
                pos: p.as_vec3().into(),
                depth: if w { (-raw).max(0.0) } else { 0.0 },
                nrm: n.as_vec3().into(),
                skirt: 0.0,
                uv: [(i - 1) as f32 / GRID as f32, (j - 1) as f32 / GRID as f32],
            });
        }
    }
    // Skirts hide the cracks between neighbouring tiles of different detail.
    let drop = (id.width_m() * 0.04).max(4.0);
    let mut skirt = |i: usize, j: usize| {
        let mut v = verts[j * GV + i];
        let up = (DVec3::from(glam::Vec3::from(v.pos).as_dvec3()) + center).normalize();
        v.pos = (glam::Vec3::from(v.pos).as_dvec3() - up * drop).as_vec3().into();
        v.skirt = 1.0;
        verts.push(v);
    };
    for i in 0..GV {
        skirt(i, 0);
    }
    for i in 0..GV {
        skirt(i, GRID);
    }
    for j in 0..GV {
        skirt(0, j);
    }
    for j in 0..GV {
        skirt(GRID, j);
    }
    Ok(TileData {
        id,
        center,
        radius: radius + drop,
        verts,
        heights,
        water,
        sea,
        imagery: mip_chain(img.into_raw(), img_size as usize, 4),
        img_size,
        mask: mask_mips,
        imagery_gen,
    })
}

/// Index buffer shared by every terrain tile (grid + four skirts).
pub fn tile_indices() -> Vec<u16> {
    let mut ix = Vec::new();
    for j in 0..GRID {
        for i in 0..GRID {
            let (a, b, c, d) = (j * GV + i, j * GV + i + 1, (j + 1) * GV + i, (j + 1) * GV + i + 1);
            ix.extend([a, c, b, b, c, d].map(|v| v as u16));
        }
    }
    let base = GV * GV;
    // (edge vertex index fn, skirt strip start, flip winding)
    let edges: [(Box<dyn Fn(usize) -> usize>, usize, bool); 4] = [
        (Box::new(|i| i), base, false),
        (Box::new(|i| GRID * GV + i), base + GV, true),
        (Box::new(|j| j * GV), base + 2 * GV, true),
        (Box::new(|j| j * GV + GRID), base + 3 * GV, false),
    ];
    for (edge, s, flip) in edges.iter() {
        for k in 0..GRID {
            let (a, b, c, d) = (edge(k), edge(k + 1), s + k, s + k + 1);
            let t = if *flip { [a, c, b, b, c, d] } else { [a, b, c, b, d, c] };
            ix.extend(t.map(|v| v as u16));
        }
    }
    ix
}

// ---------------------------------------------------------------- worker pool

pub enum Job {
    Terrain(TileId),
    Buildings(TileId),
    Airfield(u32),
}
pub enum JobResult {
    Terrain(TileId, Result<TileData>),
    Buildings(TileId, Result<crate::buildings::BuildingData>),
    Airfield(u32, Result<(Arc<crate::airports::Geo>, crate::airfield::FieldMesh)>),
}

pub struct Loader {
    queue: Arc<(Mutex<Vec<(f32, Job)>>, Condvar)>,
    pub rx: crossbeam_channel::Receiver<JobResult>,
    pub src: Arc<Sources>,
}

impl Loader {
    pub fn new(src: Arc<Sources>, threads: usize) -> Self {
        let queue = Arc::new((Mutex::new(Vec::<(f32, Job)>::new()), Condvar::new()));
        let (tx, rx) = crossbeam_channel::unbounded();
        for _ in 0..threads {
            let (queue, tx, src) = (queue.clone(), tx.clone(), src.clone());
            std::thread::spawn(move || loop {
                let job = {
                    let mut q = queue.0.lock();
                    while q.is_empty() {
                        queue.1.wait(&mut q);
                    }
                    let best = q.iter().enumerate().min_by(|a, b| a.1 .0.total_cmp(&b.1 .0)).map(|(i, _)| i).unwrap();
                    q.swap_remove(best).1
                };
                let res = match job {
                    Job::Terrain(id) => JobResult::Terrain(id, build_tile(&src, id)),
                    Job::Buildings(id) => JobResult::Buildings(id, crate::buildings::build(&src, id)),
                    Job::Airfield(i) => JobResult::Airfield(i, crate::airports::geo_for(&src, i).map(|g| {
                        let m = crate::airfield::build(&g);
                        (g, m)
                    })),
                };
                if tx.send(res).is_err() {
                    break;
                }
            });
        }
        Self { queue, rx, src }
    }
    /// Lower priority value = sooner.
    pub fn submit(&self, prio: f32, job: Job) {
        self.queue.0.lock().push((prio, job));
        self.queue.1.notify_one();
    }
}
