//! The moving map: a minimap in the HUD's lower-right corner and a large map
//! on M. Both draw the satellite imagery the ground is built from (loaded on
//! worker threads, kept as egui textures between frames), the route, nearby
//! airports, live traffic and the aircraft, with egui's painter.
use crate::geo::{self, TileId};
use crate::i18n::t;
use crate::route::{self, Route, Stop, Wp};
use crate::{airports, sources::Sources};
use egui::{pos2, vec2, Align2, Color32, FontId, Pos2, Rect, Stroke, Vec2};
use glam::DVec2;
use std::collections::{HashMap, HashSet};
use std::f64::consts::PI;
use std::sync::{mpsc, Arc, Mutex};
use std::time::Instant;

/// Tiles being loaded at once, at most.
const MAX_LOADING: usize = 6;
/// Tiles kept as textures (about 256 KB each).
const MAX_TEX: usize = 260;
/// New textures uploaded per frame, at most.
const UPLOADS: usize = 3;
const MAX_Z: u8 = 16;
const MAGENTA: Color32 = Color32::from_rgb(255, 80, 230);
const TRAFFIC: Color32 = Color32::from_rgb(150, 225, 255);
/// the other players in a multiplayer room
pub const PEER: Color32 = Color32::from_rgb(255, 150, 40);
const APT: Color32 = Color32::from_rgb(110, 200, 255);

/// A tile of one layer: 0 imagery, 1 place names.
type Key = (u8, TileId);
const IMG: u8 = 0;
const NAMES: u8 = 1;

struct Entry {
    /// None: the tile could not be had
    tex: Option<egui::TextureHandle>,
    used: u64,
}

/// The imagery tiles, as textures, and the worker threads that load them.
struct Tiles {
    jobs: mpsc::Sender<Key>,
    done: mpsc::Receiver<(Key, Option<egui::ColorImage>)>,
    have: HashMap<Key, Entry>,
    loading: HashSet<Key>,
    /// tiles wanted this frame, nearest the centre first
    want: Vec<(f64, Key)>,
    frame: u64,
}

impl Tiles {
    fn new(src: Arc<Sources>) -> Self {
        let (jobs, jrx) = mpsc::channel::<Key>();
        let (dtx, done) = mpsc::channel();
        let jrx = Arc::new(Mutex::new(jrx));
        for k in 0..3 {
            let (src, jrx, dtx) = (src.clone(), jrx.clone(), dtx.clone());
            let _ = std::thread::Builder::new().name(format!("map tiles {k}")).spawn(move || loop {
                let Ok((layer, id)) = jrx.lock().map_err(|_| ()).and_then(|r| r.recv().map_err(|_| ())) else { return };
                let img = match if layer == NAMES { src.labels(id) } else { src.imagery(id) } {
                    Ok(img) => Some(egui::ColorImage::from_rgba_unmultiplied([img.width() as usize, img.height() as usize], img.as_raw())),
                    Err(e) => {
                        log::debug!("map tile {id:?}: {e}");
                        None
                    }
                };
                if dtx.send(((layer, id), img)).is_err() {
                    return;
                }
            });
        }
        Tiles { jobs, done, have: HashMap::new(), loading: HashSet::new(), want: Vec::new(), frame: 0 }
    }

    /// Takes in what the workers have loaded, and forgets the tiles unused longest.
    fn pump(&mut self, ctx: &egui::Context) {
        self.frame += 1;
        for _ in 0..UPLOADS {
            let Ok((key, img)) = self.done.try_recv() else { break };
            self.loading.remove(&key);
            let id = key.1;
            let tex = img.map(|img| ctx.load_texture(format!("map {} {}/{}/{}", key.0, id.z, id.x, id.y), img, egui::TextureOptions::LINEAR));
            self.have.insert(key, Entry { tex, used: self.frame });
        }
        if self.have.len() > MAX_TEX {
            let mut old: Vec<(u64, Key)> = self.have.iter().filter(|(_, e)| e.used + 1 < self.frame).map(|(id, e)| (e.used, *id)).collect();
            old.sort_unstable_by_key(|o| o.0);
            for (_, id) in old.into_iter().take(self.have.len() - MAX_TEX) {
                self.have.remove(&id);
            }
        }
    }

    /// Sends the wanted tiles to the workers, nearest first, a few at a time.
    fn request(&mut self) {
        self.want.sort_by(|a, b| a.0.total_cmp(&b.0));
        for (_, id) in std::mem::take(&mut self.want) {
            if self.loading.len() >= MAX_LOADING {
                break;
            }
            if self.have.contains_key(&id) || self.loading.contains(&id) {
                continue;
            }
            if self.jobs.send(id).is_ok() {
                self.loading.insert(id);
            }
        }
    }

    /// The imagery under a view: each tile at the zoom that suits it, or
    /// part of a coarser one already loaded while it comes.
    /// Place names are drawn a pixel per point (a zoom coarser on Retina),
    /// so their text keeps its size; they fall back at most one zoom.
    fn draw(&mut self, painter: &egui::Painter, v: &View, layer: u8) {
        let ppp = if layer == NAMES { 1.0 } else { painter.ctx().pixels_per_point() as f64 };
        let z = (v.ppu * ppp / 256.0).log2().round().clamp(0.0, MAX_Z as f64) as u8;
        let n = (1u64 << z) as f64;
        let half = if v.rot == 0.0 { v.rect.size() * 0.5 } else { Vec2::splat(v.rect.size().length() * 0.5) };
        let (hx, hy) = (half.x as f64 / v.ppu, half.y as f64 / v.ppu);
        let x0 = ((v.c.x - hx) * n).floor() as i64;
        let x1 = (((v.c.x + hx) * n).floor() as i64).min(x0 + n as i64 - 1);
        let y0 = ((v.c.y - hy) * n).floor().max(0.0) as i64;
        let y1 = ((v.c.y + hy) * n).floor().min(n - 1.0) as i64;
        let frame = self.frame;
        for ty in y0..=y1 {
            for tx in x0..=x1 {
                let id = TileId::new(z, tx.rem_euclid(n as i64) as u32, ty as u32);
                let prio = ((tx as f64 + 0.5) / n - v.c.x).hypot((ty as f64 + 0.5) / n - v.c.y) * n;
                if !self.have.contains_key(&(layer, id)) {
                    self.want.push((prio + if layer == NAMES { 0.7 } else { 0.0 }, (layer, id)));
                    // a coarse one too, to show while the zoom changes
                    if z >= 3 && layer == IMG {
                        self.want.push((prio + 50.0, (layer, id.ancestor(z - 3).0)));
                    }
                }
                let mut found = None;
                for up in 0..=z.min(if layer == NAMES { 1 } else { 8 }) {
                    let (a, ox, oy, s) = id.ancestor(z - up);
                    if let Some(e) = self.have.get_mut(&(layer, a)) {
                        e.used = frame;
                        if let Some(t) = &e.tex {
                            found = Some((t.id(), ox as f32, oy as f32, s as f32));
                            break;
                        }
                    }
                }
                let Some((tid, ox, oy, s)) = found else { continue };
                let (wx0, wy0, wx1, wy1) = (tx as f64 / n, ty as f64 / n, (tx + 1) as f64 / n, (ty + 1) as f64 / n);
                let pts = [v.at_m(DVec2::new(wx0, wy0), false), v.at_m(DVec2::new(wx1, wy0), false), v.at_m(DVec2::new(wx1, wy1), false), v.at_m(DVec2::new(wx0, wy1), false)];
                let uvs = [pos2(ox, oy), pos2(ox + s, oy), pos2(ox + s, oy + s), pos2(ox, oy + s)];
                let mut mesh = egui::Mesh::with_texture(tid);
                for (p, uv) in pts.into_iter().zip(uvs) {
                    mesh.vertices.push(egui::epaint::Vertex { pos: p, uv, color: Color32::WHITE });
                }
                mesh.add_triangle(0, 1, 2);
                mesh.add_triangle(0, 2, 3);
                painter.add(egui::Shape::mesh(mesh));
            }
        }
    }
}

/// Where the map looks: its centre (web mercator, 0..1), its scale in
/// screen points per mercator unit, its rotation (track-up) and its rect.
struct View {
    c: DVec2,
    ppu: f64,
    /// radians: the heading at the top of the map (0 north-up)
    rot: f64,
    rect: Rect,
}

fn merc(lat: f64, lon: f64) -> DVec2 {
    DVec2::new(geo::lon_to_merc_x(lon.to_radians()), geo::lat_to_merc_y(lat.to_radians()))
}

fn rotate(v: Vec2, a: f32) -> Vec2 {
    let (s, c) = a.sin_cos();
    vec2(v.x * c - v.y * s, v.x * s + v.y * c)
}

impl View {
    fn at_m(&self, m: DVec2, wrap: bool) -> Pos2 {
        let mut d = m - self.c;
        if wrap {
            d.x -= d.x.round();
        }
        self.rect.center() + rotate(vec2((d.x * self.ppu) as f32, (d.y * self.ppu) as f32), -self.rot as f32)
    }
    /// A lat/lon (degrees) on screen.
    fn at(&self, lat: f64, lon: f64) -> Pos2 {
        self.at_m(merc(lat, lon), true)
    }
    /// Metres per screen point at the centre.
    fn mpp(&self) -> f64 {
        2.0 * PI * geo::R * geo::merc_y_to_lat(self.c.y).cos() / self.ppu
    }
    fn centre_ll(&self) -> (f64, f64) {
        (geo::merc_y_to_lat(self.c.y).to_degrees(), (self.c.x.rem_euclid(1.0) * 2.0 * PI - PI).to_degrees())
    }
    /// Screen direction of a compass heading (degrees).
    fn dir(&self, hdg: f64) -> Vec2 {
        let a = (hdg - self.rot.to_degrees()).to_radians() as f32;
        vec2(a.sin(), -a.cos())
    }
}

/// What the map shows of the flight.
pub struct Scene<'a> {
    /// degrees
    pub lat: f64,
    pub lon: f64,
    pub hdg: f64,
    /// ground speed and the aircraft's cruise speed (m/s), height above ground (m)
    pub gs: f64,
    pub cruise: f64,
    pub agl: f64,
    pub route: &'a Route,
    /// live traffic: lat, lon (degrees), heading (degrees)
    pub traffic: Vec<(f64, f64, f64)>,
    /// the big ships (ships.rs): lat, lon, heading (degrees), length (m), name
    pub ships: Vec<(f64, f64, f64, f64, String)>,
    /// the other players (multiplayer): lat, lon (degrees), heading (degrees), name
    pub peers: Vec<(f64, f64, f64, String)>,
}

/// Airports near a view, looked up again only when it has moved or zoomed.
#[derive(Default)]
struct Apts {
    at: (f64, f64),
    r: f64,
    list: Vec<u32>,
}

impl Apts {
    fn get(&mut self, lat: f64, lon: f64, r: f64) -> &[u32] {
        if self.r <= 0.0 || route::dist(lat, lon, self.at.0, self.at.1) > r * 0.2 || (r / self.r).ln().abs() > 0.2 {
            let db = airports::Db::get();
            self.list = db
                .near(lat, lon, (r * 1.3).min(2_500_000.0))
                .into_iter()
                .filter(|&(i, _)| {
                    let a = &db.apts[i as usize];
                    a.kind == 2 || (a.kind == 1 && r < 300_000.0) || (r < 30_000.0 && !a.rwys.is_empty())
                })
                .map(|x| x.0)
                .take(600)
                .collect();
            (self.at, self.r) = ((lat, lon), r);
        }
        &self.list
    }
}

pub struct Map {
    tiles: Tiles,
    /// the large map (M)
    pub open: bool,
    /// the minimap (a setting)
    pub mini_on: bool,
    /// the minimap turns with the aircraft (click it to switch)
    pub track_up: bool,
    /// place names over the imagery, Google-Maps style
    pub names: bool,
    /// large map: centre when panned away (None follows the aircraft), and zoom
    pan: Option<DVec2>,
    full_ppu: Option<f64>,
    /// minimap radius (m), smoothed
    mini_r: f64,
    apts_mini: Apts,
    apts_full: Apts,
    /// what drawing the maps costs (ms per frame, smoothed)
    pub cost_ms: f64,
    cost_acc: f64,
}

impl Map {
    pub fn new(src: Arc<Sources>, mini_on: bool) -> Self {
        Map { tiles: Tiles::new(src), open: false, mini_on, track_up: false, names: true, pan: None, full_ppu: None, mini_r: 0.0, apts_mini: Apts::default(), apts_full: Apts::default(), cost_ms: 0.0, cost_acc: 0.0 }
    }

    /// Call once per frame before drawing either map.
    pub fn begin(&mut self, ctx: &egui::Context) {
        let t0 = Instant::now();
        self.tiles.pump(ctx);
        self.cost_acc = t0.elapsed().as_secs_f64() * 1000.0;
    }

    /// Call once per frame after drawing.
    pub fn end(&mut self) {
        let t0 = Instant::now();
        self.tiles.request();
        let ms = self.cost_acc + t0.elapsed().as_secs_f64() * 1000.0;
        self.cost_ms += (ms - self.cost_ms) * 0.05;
        self.cost_acc = 0.0;
        if self.tiles.frame % 600 == 0 {
            log::info!("maps: {:.3} ms per frame, {} tiles kept, {} loading", self.cost_ms, self.tiles.have.len(), self.tiles.loading.len());
        }
    }

    /// How far the minimap reaches from the aircraft (m): a couple of
    /// minutes of flight, more when high.
    fn mini_radius(s: &Scene) -> f64 {
        (s.gs.max(s.cruise * 0.8) * 250.0).max(s.agl * 2.5).clamp(9_000.0, 900_000.0)
    }

    /// The minimap, in `rect`. Clicking it switches north-up / track-up.
    pub fn mini(&mut self, ctx: &egui::Context, rect: Rect, s: &Scene) {
        let t0 = Instant::now();
        let want = Self::mini_radius(s);
        self.mini_r = if self.mini_r <= 0.0 { want } else { self.mini_r * (want / self.mini_r).powf(0.03) };
        let mut clicked = false;
        egui::Area::new(egui::Id::unique("minimap")).fixed_pos(rect.min).movable(false).show(ctx, |ui| {
            let (r, resp) = ui.allocate_exact_size(rect.size(), egui::Sense::click());
            clicked = resp.clicked();
            let painter = ui.painter_at(r);
            let c = merc(s.lat, s.lon);
            let mpu = 2.0 * PI * geo::R * s.lat.to_radians().cos();
            let v = View { c, ppu: mpu * (r.width() as f64 * 0.5) / self.mini_r, rot: if self.track_up { s.hdg.to_radians() } else { 0.0 }, rect: r };
            painter.rect_filled(r, 10.0, Color32::from_rgb(18, 26, 34));
            self.tiles.draw(&painter, &v, IMG);
            painter.rect_filled(r, 0.0, Color32::from_black_alpha(25));
            if self.names {
                self.tiles.draw(&painter, &v, NAMES);
            }
            let apts = self.apts_mini.get(s.lat, s.lon, self.mini_r * 1.42).to_vec();
            layers(&painter, &v, s, &apts, 10, None);
            // north marker, scale and mode
            let nd = v.dir(0.0);
            let np = r.center() + nd * (r.width() * 0.5 - 13.0);
            painter.circle_filled(np, 9.0, Color32::from_black_alpha(150));
            painter.text(np, Align2::CENTER_CENTER, "N", FontId::proportional(12.0), Color32::from_rgb(255, 120, 100));
            scale_bar(&painter, &v, r.left_bottom() + vec2(10.0, -10.0), r.width() * 0.3, Some(if self.track_up { t!("TRACK UP", "ROUTE EN HAUT") } else { t!("NORTH UP", "NORD EN HAUT") }));
            painter.rect_stroke(r, 10.0, Stroke::new(1.5, Color32::from_white_alpha(70)), egui::StrokeKind::Inside);
            resp.on_hover_text(t!("Click: north-up / track-up.  M: large map", "Clic : nord en haut / route en haut.  M : grande carte"));
        });
        if clicked {
            self.track_up = !self.track_up;
        }
        self.cost_acc += t0.elapsed().as_secs_f64() * 1000.0;
    }

    /// The large map over most of the screen. Returns what was done to the
    /// route there (to apply with `plan`), if anything.
    pub fn full(&mut self, ctx: &egui::Context, s: &Scene) -> Option<Plan> {
        let t0 = Instant::now();
        let screen = ctx.content_rect();
        let rect = screen.shrink2(vec2((screen.width() * 0.06).max(20.0), (screen.height() * 0.07).max(20.0)));
        let me = merc(s.lat, s.lon);
        let mpu = 2.0 * PI * geo::R * s.lat.to_radians().cos();
        let mut ppu = self.full_ppu.unwrap_or_else(|| mpu * (rect.width() as f64 * 0.5) / (Self::mini_radius(s) * 5.0));
        let mut c = self.pan.unwrap_or(me);
        let mut status = None;
        let mut close = false;
        let mut follow = false;
        egui::Area::new(egui::Id::unique("full_map")).order(egui::Order::Foreground).fixed_pos(rect.min).movable(false).show(ctx, |ui| {
            let (r, resp) = ui.allocate_exact_size(rect.size(), egui::Sense::click_and_drag());
            // pan and zoom (about the pointer)
            if resp.dragged() {
                let d = resp.drag_delta();
                c -= DVec2::new(d.x as f64, d.y as f64) / ppu;
                self.pan = Some(c);
            }
            if resp.hovered() {
                let (scroll, pinch) = ui.input(|i| (i.smooth_scroll_delta.y, i.zoom_delta()));
                let k = (scroll as f64 * 0.004).exp() * pinch as f64;
                if (k - 1.0).abs() > 1e-6 {
                    let at = resp.hover_pos().unwrap_or(r.center()) - r.center();
                    let p = c + DVec2::new(at.x as f64, at.y as f64) / ppu;
                    let nppu = (ppu * k).clamp(300.0, mpu / 1.5);
                    c = p - DVec2::new(at.x as f64, at.y as f64) / nppu;
                    ppu = nppu;
                    if self.pan.is_some() || at.length() > 2.0 {
                        self.pan = Some(c);
                    }
                }
            }
            c.y = c.y.clamp(0.0, 1.0);
            let v = View { c, ppu, rot: 0.0, rect: r };
            let painter = ui.painter_at(r);
            painter.rect_filled(r, 8.0, Color32::from_rgb(18, 26, 34));
            self.tiles.draw(&painter, &v, IMG);
            if self.names {
                self.tiles.draw(&painter, &v, NAMES);
            }
            let (clat, clon) = v.centre_ll();
            let radius = v.mpp() * (r.size().length() as f64 * 0.5);
            let apts = self.apts_full.get(clat, clon, radius).to_vec();
            // the airport under the pointer: highlighted, named, and what a click does
            let mut hit = None;
            let ptr = resp.hover_pos().filter(|_| !resp.dragged());
            layers(&painter, &v, s, &apts, 60, ptr.map(|p| (p, &mut hit)));
            scale_bar(&painter, &v, r.left_bottom() + vec2(14.0, -14.0), r.width() * 0.18, None);
            painter.rect_stroke(r, 8.0, Stroke::new(1.5, Color32::from_white_alpha(80)), egui::StrokeKind::Inside);
            if let Some(i) = hit {
                let a = &airports::Db::get().apts[i as usize];
                let p = v.at(a.lat, a.lon);
                painter.circle_stroke(p, 9.0, Stroke::new(2.5, Color32::WHITE));
                let on = s.route.stops.iter().any(|st| st.wp.as_ref().is_some_and(|w| w.apt == Some(i)));
                let what = if on { t!("click / right-click: remove from the route", "clic / clic droit : retirer de l'itinéraire") } else { t!("click: add to the route", "clic : ajouter à l'itinéraire") };
                let lines = format!("{}  {}\n{}", a.code(), a.name, what);
                let galley = painter.layout(lines, FontId::proportional(13.0), Color32::WHITE, 360.0);
                let mut at = p + vec2(14.0, -galley.size().y - 10.0);
                at.x = at.x.min(r.right() - galley.size().x - 12.0);
                at.y = at.y.max(r.top() + 6.0);
                let bg = Rect::from_min_size(at, galley.size()).expand(5.0);
                painter.rect_filled(bg, 4.0, Color32::from_rgba_unmultiplied(8, 12, 18, 220));
                painter.galley(at, galley, Color32::WHITE);
                if resp.clicked() || resp.secondary_clicked() {
                    status = Some(if on { Plan::Remove(i) } else if resp.clicked() { Plan::Add(i) } else { return });
                }
            }
        });
        // title, hints and buttons over the map
        egui::Area::new(egui::Id::unique("full_map_bar")).order(egui::Order::Foreground).fixed_pos(rect.min + vec2(12.0, 10.0)).movable(false).show(ctx, |ui| {
            egui::Frame::new().fill(Color32::from_rgba_unmultiplied(8, 12, 18, 190)).corner_radius(6.0).inner_margin(8.0).show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new(t!("Map", "Carte")).size(18.0).strong().color(Color32::WHITE));
                    ui.add_space(8.0);
                    if ui.add_enabled(self.pan.is_some(), egui::Button::new(t!("Centre on aircraft", "Centrer sur l'avion"))).clicked() {
                        follow = true;
                    }
                    let any = s.route.stops.iter().any(|st| st.wp.is_some() || !st.text.is_empty());
                    if ui.add_enabled(any, egui::Button::new(t!("Clear route", "Effacer l'itinéraire"))).clicked() {
                        status = Some(Plan::Clear);
                    }
                    ui.toggle_value(&mut self.names, t!("Place names", "Noms de lieux"));
                    if ui.button(t!("Close (M)", "Fermer (M)")).clicked() {
                        close = true;
                    }
                });
                ui.label(egui::RichText::new(t!("Click airports to plan a route (From, To, then stops on the way)   Wheel: zoom   Drag: pan", "Cliquez des aéroports pour tracer un itinéraire (départ, arrivée, puis étapes)   Molette : zoom   Glisser : déplacer")).size(12.0).color(Color32::from_white_alpha(170)));
            });
        });
        self.full_ppu = Some(ppu);
        if follow {
            self.pan = None;
        }
        if close {
            self.open = false;
        }
        self.cost_acc += t0.elapsed().as_secs_f64() * 1000.0;
        status
    }
}

/// A change to the route made on the large map.
#[derive(Clone, Copy, Debug)]
pub enum Plan {
    Add(u32),
    Remove(u32),
    Clear,
}

/// Applies a change made on the map to the route; returns a status line.
pub fn plan(route: &mut Route, p: Plan) -> String {
    match p {
        Plan::Add(i) => add_stop(route, i),
        Plan::Remove(i) => {
            let Some(k) = route.stops.iter().position(|st| st.wp.as_ref().is_some_and(|w| w.apt == Some(i))) else { return String::new() };
            route.stops.remove(k);
            // a plan keeps its two lines, From and To
            if route.stops.len() < 2 {
                route.stops.push(Stop::default());
            }
            if k < route.next {
                route.next -= 1;
            }
            route.next = route.next.clamp(1, route.stops.len() - 1);
            if !route.ready() {
                route.autopilot = false;
            }
            t!("{} removed from the route", "{} retiré de l'itinéraire", airports::Db::get().apts[i as usize].ident)
        }
        Plan::Clear => {
            *route = Route::empty();
            t!("Route cleared", "Itinéraire effacé").into()
        }
    }
}

/// Adds an airport to the route: the first empty line of the plan (From,
/// then To), then before the destination (or after it once arrived).
fn add_stop(route: &mut Route, i: u32) -> String {
    let a = &airports::Db::get().apts[i as usize];
    let wp = Wp { name: format!("{} {}", a.code(), a.name), lat: a.lat, lon: a.lon, elev: a.elev as f64, apt: Some(i) };
    let stop = Stop { text: a.ident.clone(), wp: Some(wp), busy: false, missing: false };
    if let Some(s) = route.stops.iter_mut().find(|s| s.wp.is_none() && !s.busy) {
        *s = stop;
    } else if route.done {
        route.stops.push(stop);
        route.next = route.stops.len() - 1;
        route.done = false;
    } else {
        let at = route.stops.len().max(1) - 1;
        route.stops.insert(at, stop);
    }
    t!("{} added to the route", "{} ajouté à l'itinéraire", a.ident)
}

/// What both maps show over the imagery: airports, the route, traffic and
/// the aircraft. With `pick`, the airport under the click goes in it.
fn layers(painter: &egui::Painter, v: &View, s: &Scene, apts: &[u32], max_labels: usize, mut pick: Option<(Pos2, &mut Option<u32>)>) {
    let r = v.rect.expand(20.0);
    let shadow = Color32::from_black_alpha(170);
    let text = |p: Pos2, align: Align2, t: &str, size: f32, col: Color32| {
        painter.text(p + vec2(1.0, 1.0), align, t, FontId::proportional(size), shadow);
        painter.text(p, align, t, FontId::proportional(size), col);
    };
    // airports: bigger ones first in the list of labels
    let db = airports::Db::get();
    let radius = v.mpp() * v.rect.size().length() as f64 * 0.5;
    let label_kind = if radius < 25_000.0 { 0 } else if radius < 150_000.0 { 1 } else { 2 };
    let mut labels = 0;
    let mut best = f32::MAX;
    let on_route: Vec<u32> = s.route.stops.iter().filter_map(|st| st.wp.as_ref().and_then(|w| w.apt)).collect();
    for kind in [2u8, 1, 0] {
        for &i in apts {
            let a = &db.apts[i as usize];
            if a.kind != kind {
                continue;
            }
            let p = v.at(a.lat, a.lon);
            if !r.contains(p) {
                continue;
            }
            let rad = [2.3, 3.0, 4.0][kind as usize];
            painter.circle(p, rad, APT, Stroke::new(1.0, Color32::from_black_alpha(200)));
            if kind >= label_kind && labels < max_labels && !on_route.contains(&i) {
                labels += 1;
                text(p + vec2(rad + 3.0, 0.0), Align2::LEFT_CENTER, &a.ident, if kind == 2 { 12.0 } else { 11.0 }, Color32::from_rgb(200, 235, 255));
            }
            if let Some((at, hit)) = pick.as_mut() {
                let d = at.distance(p);
                if d < 12.0 && d < best {
                    best = d;
                    **hit = Some(i);
                }
            }
        }
    }
    // the route: flown legs faint, the leg being flown bright
    let ws: Vec<(usize, &Wp)> = s.route.stops.iter().enumerate().filter_map(|(k, st)| st.wp.as_ref().map(|w| (k, w))).collect();
    let flying = s.route.target().is_some();
    for p in ws.windows(2) {
        let ((_, a), (kb, b)) = (p[0], p[1]);
        let al = if !flying { 200 } else if kb < s.route.next { 70 } else if kb == s.route.next { 255 } else { 170 };
        let col = Color32::from_rgba_unmultiplied(MAGENTA.r(), MAGENTA.g(), MAGENTA.b(), al);
        let d = route::dist(a.lat, a.lon, b.lat, b.lon);
        let pts: Vec<Pos2> = route::arc((a.lat, a.lon), (b.lat, b.lon), ((d / 25_000.0) as usize).clamp(2, 64))
            .into_iter()
            .map(|u| {
                let (la, lo, _) = geo::pos_to_lla(u * geo::R);
                v.at(la.to_degrees(), lo.to_degrees())
            })
            .collect();
        for q in pts.windows(2) {
            if (q[1] - q[0]).length() < v.rect.width() {
                painter.line_segment([q[0], q[1]], Stroke::new(2.5, col));
            }
        }
    }
    for (k, w) in &ws {
        let p = v.at(w.lat, w.lon);
        if !r.contains(p) {
            continue;
        }
        let target = flying && *k == s.route.next;
        let d = if target { 7.0 } else { 5.5 };
        let pts = vec![p + vec2(0.0, -d), p + vec2(d, 0.0), p + vec2(0.0, d), p + vec2(-d, 0.0)];
        let fill = if target { MAGENTA } else { Color32::from_rgba_unmultiplied(MAGENTA.r(), MAGENTA.g(), MAGENTA.b(), 70) };
        painter.add(egui::Shape::convex_polygon(pts, fill, Stroke::new(1.5, Color32::WHITE)));
        let code = w.apt.map(|i| db.apts[i as usize].ident.clone()).unwrap_or_else(|| w.name.split(',').next().unwrap_or("").chars().take(16).collect());
        text(p + vec2(0.0, d + 2.0), Align2::CENTER_TOP, &code, 11.5, MAGENTA);
    }
    // live traffic: small arrows
    for &(la, lo, h) in &s.traffic {
        let p = v.at(la, lo);
        if !v.rect.contains(p) {
            continue;
        }
        let f = v.dir(h);
        let side = vec2(-f.y, f.x);
        let pts = vec![p + f * 6.0, p - f * 4.0 + side * 4.0, p - f * 2.0, p - f * 4.0 - side * 4.0];
        painter.add(egui::Shape::convex_polygon(pts, TRAFFIC, Stroke::new(0.8, Color32::from_black_alpha(200))));
    }
    // ships: a hull outline to scale (never smaller than an icon), named
    for (la, lo, h, len, name) in &s.ships {
        let p = v.at(*la, *lo);
        if !r.contains(p) {
            continue;
        }
        let f = v.dir(*h);
        let side = vec2(-f.y, f.x);
        let l = ((len / v.mpp()) as f32 * 0.5).max(7.0);
        let w = l * 0.22;
        let pts = vec![p + f * l, p + f * (l * 0.45) + side * w, p - f * l + side * w * 0.8, p - f * l - side * w * 0.8, p + f * (l * 0.45) - side * w];
        painter.add(egui::Shape::convex_polygon(pts, Color32::from_rgb(235, 240, 245), Stroke::new(1.2, Color32::from_rgb(20, 60, 110))));
        text(p + vec2(0.0, l.min(30.0) + 4.0), Align2::CENTER_TOP, name, 11.0, Color32::from_rgb(200, 225, 255));
    }
    // the other players: orange arrows with their names
    for (la, lo, h, name) in &s.peers {
        let p = v.at(*la, *lo);
        if !v.rect.contains(p) {
            continue;
        }
        let f = v.dir(*h);
        let side = vec2(-f.y, f.x);
        let pts = vec![p + f * 9.0, p - f * 6.0 + side * 6.0, p - f * 3.0, p - f * 6.0 - side * 6.0];
        painter.add(egui::Shape::convex_polygon(pts, PEER, Stroke::new(1.2, Color32::BLACK)));
        text(p + vec2(10.0, -2.0), Align2::LEFT_BOTTOM, name, 12.0, PEER);
    }
    // the aircraft
    let p = v.at(s.lat, s.lon);
    let f = v.dir(s.hdg);
    let side = vec2(-f.y, f.x);
    let pts = vec![p + f * 11.0, p - f * 7.0 + side * 7.0, p - f * 3.5, p - f * 7.0 - side * 7.0];
    painter.add(egui::Shape::convex_polygon(pts, Color32::from_rgb(255, 210, 60), Stroke::new(1.5, Color32::BLACK)));
}

/// A scale bar of a round length in nautical miles (and an optional tag).
fn scale_bar(painter: &egui::Painter, v: &View, at: Pos2, max_w: f32, tag: Option<&str>) {
    let nm = 1852.0;
    let want = v.mpp() * max_w as f64 / nm;
    let e = 10f64.powf(want.log10().floor());
    let len = [5.0, 2.0, 1.0].into_iter().map(|k| k * e).find(|l| *l <= want).unwrap_or(e);
    let w = (len * nm / v.mpp()) as f32;
    let y = at.y;
    let col = Color32::WHITE;
    let bg = Color32::from_black_alpha(160);
    for s in [Stroke::new(4.0, bg), Stroke::new(2.0, col)] {
        painter.line_segment([pos2(at.x, y), pos2(at.x + w, y)], s);
        painter.line_segment([pos2(at.x, y - 4.0), pos2(at.x, y)], s);
        painter.line_segment([pos2(at.x + w, y - 4.0), pos2(at.x + w, y)], s);
    }
    let label = if len >= 1.0 { format!("{len:.0} nm") } else { format!("{len} nm") };
    let label = match tag {
        Some(t) => format!("{label}   {t}"),
        None => label,
    };
    painter.text(pos2(at.x + 1.0, y - 6.0), Align2::LEFT_BOTTOM, &label, FontId::proportional(11.5), bg);
    painter.text(pos2(at.x, y - 7.0), Align2::LEFT_BOTTOM, &label, FontId::proportional(11.5), col);
}
