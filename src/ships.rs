//! Big ships at sea: an Olympic-class liner of 1912 (the Titanic and her
//! sisters), a container ship a kilometre long, a Nimitz-class aircraft
//! carrier, and a mega carrier 3.6 km long that 747s fly off and land on.
//!
//! Each sails a slow straight course and rides the waves that are drawn: its
//! heave, pitch and roll follow the sea under its bow, stern and sides
//! through a heavy, slow spring (a big ship barely answers short waves).
//! They are spawned a few at a time near the player when over open sea,
//! the same ones in the same places for each patch of ocean, and one of each
//! sails in company when the flight starts on a ship.
//!
//! The decks are solid (`Decks`, read by plane.rs through world.rs): wheels
//! roll on them and carry the ship's own motion, and the hull, island,
//! superstructure and parked aircraft are walls. The carriers launch with a
//! catapult and catch with arresting wires (`Ops`).
//!
//! Ship frame: +X starboard, +Y up, -Z forward (as the aircraft's), the
//! origin on the waterline amidships.
use crate::boat::{Mesh, MB, GLASS, LAMP, TRIM};
use crate::geo;
use crate::render::Inst;
use glam::{DMat3, DQuat, DVec2, DVec3, Vec3};
use std::collections::HashSet;
use std::sync::LazyLock;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Kind {
    Liner,
    MegaShip,
    Carrier,
    MegaCarrier,
}

impl Kind {
    /// In the order the start screen lists them.
    pub const ALL: [Kind; 4] = [Kind::Carrier, Kind::MegaCarrier, Kind::Liner, Kind::MegaShip];
    pub fn idx(self) -> usize {
        match self {
            Kind::Liner => 0,
            Kind::MegaShip => 1,
            Kind::Carrier => 2,
            Kind::MegaCarrier => 3,
        }
    }
    /// `--ship` names.
    pub fn parse(s: &str) -> Option<Kind> {
        match s.to_ascii_lowercase().replace(['-', '_', ' '], "").as_str() {
            "carrier" | "cvn" | "nimitz" | "aircraftcarrier" => Some(Kind::Carrier),
            "megacarrier" | "supercarrier" | "bigcarrier" => Some(Kind::MegaCarrier),
            "titanic" | "liner" | "olympic" => Some(Kind::Liner),
            "megaship" | "mega" | "containership" | "bigship" => Some(Kind::MegaShip),
            _ => None,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Kind::Liner => "Titanic",
            Kind::MegaShip => "Mega ship",
            Kind::Carrier => "Aircraft carrier",
            Kind::MegaCarrier => "Mega carrier",
        }
    }
    pub fn blurb(self) -> &'static str {
        match self {
            Kind::Liner => "The 1912 Olympic-class liner: 269 m, four funnels, 21 kt.",
            Kind::MegaShip => "A container ship a kilometre long, 140 m across, stacked nine high.",
            Kind::Carrier => "Nimitz class, 333 m. Start on the bow catapult: hold Space to launch. O lowers the hook to catch a wire.",
            Kind::MegaCarrier => "A carrier 3.6 km long with a 420 m deck: room for a 747 (even the ×5) to take off and land.",
        }
    }
    pub fn is_carrier(self) -> bool {
        matches!(self, Kind::Carrier | Kind::MegaCarrier)
    }
    pub fn dims(self) -> &'static Dims {
        &DIMS[self.idx()]
    }
    pub fn geom(self) -> &'static Geom {
        &GEOM[self.idx()]
    }
}

/// Size and gait.
pub struct Dims {
    /// overall length, waterline beam, height of the main (flight) deck
    /// above the waterline, draft (m)
    pub len: f64,
    pub beam: f64,
    pub deck: f64,
    pub draft: f64,
    /// cruising speed, and flight operations into the wind (m/s)
    pub cruise: f64,
    pub ops: f64,
    /// natural period of its heave and pitch (s): how slowly it answers the sea
    pub period: f64,
    /// how far back its wake shows (m)
    pub wake: f64,
}

pub const DIMS: [Dims; 4] = [
    Dims { len: 269.0, beam: 28.2, deck: 11.0, draft: 10.5, cruise: 10.8, ops: 11.3, period: 11.0, wake: 900.0 },
    Dims { len: 1000.0, beam: 140.0, deck: 30.0, draft: 16.0, cruise: 8.5, ops: 9.0, period: 22.0, wake: 2600.0 },
    Dims { len: 333.0, beam: 41.0, deck: 19.5, draft: 11.3, cruise: 10.3, ops: 14.4, period: 12.0, wake: 1600.0 },
    Dims { len: 3600.0, beam: 300.0, deck: 50.0, draft: 30.0, cruise: 9.0, ops: 14.4, period: 32.0, wake: 6000.0 },
];

// ------------------------------------------------------------------ hulls

/// A hull's lines: plan taper at the bow (`tb`) and stern (`ta`, down to
/// `transom` of the beam), flare out to the deck edge, the stem's rake, and
/// how full the midship section is (`p`: 2 a V, 8 a box with round bilges).
struct HullP {
    len: f64,
    beam: f64,
    deck: f64,
    draft: f64,
    tb: f64,
    ta: f64,
    transom: f64,
    flare: f64,
    bow_flare: f64,
    rake: f64,
    p: f64,
    sheer_b: f64,
    sheer_s: f64,
}

fn taper(t: f64, tb: f64, ta: f64, transom: f64) -> f64 {
    let f = if t < tb { 1.0 - (1.0 - t / tb).powi(2) } else { 1.0 };
    let g = if t > 1.0 - ta { 1.0 - (1.0 - transom) * ((t - (1.0 - ta)) / ta).powi(2) } else { 1.0 };
    (f * g).max(0.0)
}

impl HullP {
    /// Deck height at `t` (0 bow .. 1 stern), with the sheer.
    fn deck_at(&self, t: f64) -> f64 {
        self.deck + self.sheer_b * (1.0 - t / 0.25).max(0.0).powi(2) + self.sheer_s * ((t - 0.85) / 0.15).max(0.0).powi(2)
    }
    /// Half breadth at `t`, height `y` above the waterline.
    fn half_w(&self, t: f64, y: f64) -> f64 {
        let wl = self.beam * 0.5 * taper(t, self.tb, self.ta, self.transom);
        if y <= 0.0 {
            let full = taper(t, self.tb * 1.4, self.ta * 1.4, 0.4);
            let p = 2.0 + (self.p - 2.0) * full * full;
            let q = (-y / self.draft).clamp(0.0, 1.0);
            wl * (1.0 - q.powf(p)).max(0.0).powf(1.0 / p)
        } else {
            let wd = self.beam * 0.5 * self.flare * taper(t, self.tb * self.bow_flare, self.ta, self.transom.max(0.05));
            let k = (y / self.deck_at(t)).clamp(0.0, 1.0);
            wl + (wd - wl) * k.powf(1.3)
        }
    }
    /// The station's z at height `y` (the stem rakes forward as it rises).
    fn z_at(&self, t: f64, y: f64) -> f64 {
        -self.len * 0.5 + t * self.len - self.rake * (y + self.draft) * (1.0 - t / self.tb).max(0.0).powi(2)
    }
}

fn v3(x: f64, y: f64, z: f64) -> Vec3 {
    Vec3::new(x as f32, y as f32, z as f32)
}

/// Stations along a hull, closer together toward the ends.
fn stations(n: usize) -> Vec<f64> {
    (0..=n).map(|i| {
        let u = i as f64 / n as f64;
        0.5 - 0.5 * (std::f64::consts::PI * u).cos()
    }).collect()
}

/// The hull from `y0` to `y1` above the waterline in one colour, both sides,
/// closed at the stern; up to the deck edge (sheer) when `y1` is None.
fn hull_band(m: &mut MB, h: &HullP, y0: f64, y1: Option<f64>, col: [f32; 3]) {
    let ts = stations(56);
    for s in [-1.0, 1.0] {
        let rows: Vec<Vec<Vec3>> = ts
            .iter()
            .map(|&t| {
                let top = y1.unwrap_or_else(|| h.deck_at(t)).min(h.deck_at(t));
                let lo = y0.min(top);
                (0..=6)
                    .map(|k| {
                        let f = k as f64 / 6.0;
                        let y = lo + (top - lo) * f;
                        v3(s * h.half_w(t, y), y, h.z_at(t, y))
                    })
                    .collect()
            })
            .collect();
        m.grid(&rows, col, TRIM, s < 0.0);
    }
    // The transom (or cruiser stern's end).
    let t = 1.0;
    let top = y1.unwrap_or_else(|| h.deck_at(t)).min(h.deck_at(t));
    let lo = y0.min(top);
    let mut pts = Vec::new();
    for k in 0..=6 {
        let y = lo + (top - lo) * k as f64 / 6.0;
        pts.push(v3(h.half_w(t, y), y, h.z_at(t, y)));
    }
    for k in (0..=6).rev() {
        let y = lo + (top - lo) * k as f64 / 6.0;
        pts.push(v3(-h.half_w(t, y), y, h.z_at(t, y)));
    }
    if (top - lo) > 0.01 {
        m.poly(&pts, col, TRIM);
    }
}

/// The deck: a crowned surface edge to edge, bow to stern.
fn hull_deck(m: &mut MB, h: &HullP, col: [f32; 3]) {
    let ts = stations(56);
    let rows: Vec<Vec<Vec3>> = ts
        .iter()
        .map(|&t| {
            let y = h.deck_at(t);
            let w = h.half_w(t, y);
            (0..=4).map(|k| {
                let f = -1.0 + 0.5 * k as f64;
                v3(f * w, y + 0.002 * h.beam * (1.0 - f * f), h.z_at(t, y))
            }).collect()
        })
        .collect();
    m.grid(&rows, col, TRIM, true);
}

/// The deck's edge as a polygon in (x, z), port side forward then back up the starboard side.
fn hull_outline(h: &HullP) -> Vec<DVec2> {
    let ts = stations(40);
    let mut out: Vec<DVec2> = ts.iter().map(|&t| {
        let y = h.deck_at(t);
        DVec2::new(-h.half_w(t, y), h.z_at(t, y))
    }).collect();
    out.extend(ts.iter().rev().map(|&t| {
        let y = h.deck_at(t);
        DVec2::new(h.half_w(t, y), h.z_at(t, y))
    }));
    out.dedup_by(|a, b| (*a - *b).length() < 1e-6);
    out
}

/// Triangles covering a simple polygon (ear clipping).
fn triangulate(poly: &[DVec2]) -> Vec<[usize; 3]> {
    let n = poly.len();
    let area: f64 = (0..n).map(|i| poly[i].perp_dot(poly[(i + 1) % n])).sum();
    let mut idx: Vec<usize> = (0..n).collect();
    if area < 0.0 {
        idx.reverse();
    }
    let mut tris = Vec::new();
    let mut guard = 0;
    while idx.len() > 3 && guard < 10 * n {
        guard += 1;
        let k = idx.len();
        let mut cut = None;
        for i in 0..k {
            let (a, b, c) = (idx[(i + k - 1) % k], idx[i], idx[(i + 1) % k]);
            let (pa, pb, pc) = (poly[a], poly[b], poly[c]);
            if (pb - pa).perp_dot(pc - pb) <= 1e-9 {
                continue;
            }
            let inside = idx.iter().any(|&j| {
                if j == a || j == b || j == c {
                    return false;
                }
                let p = poly[j];
                (pb - pa).perp_dot(p - pa) > 0.0 && (pc - pb).perp_dot(p - pb) > 0.0 && (pa - pc).perp_dot(p - pc) > 0.0
            });
            if !inside {
                cut = Some(i);
                tris.push([a, b, c]);
                break;
            }
        }
        match cut {
            Some(i) => {
                idx.remove(i);
            }
            None => break,
        }
    }
    if idx.len() == 3 {
        tris.push([idx[0], idx[1], idx[2]]);
    }
    tris
}

fn inside(poly: &[DVec2], p: DVec2) -> bool {
    let mut c = false;
    let n = poly.len();
    let mut j = n - 1;
    for i in 0..n {
        let (a, b) = (poly[i], poly[j]);
        if (a.y > p.y) != (b.y > p.y) && p.x < (b.x - a.x) * (p.y - a.y) / (b.y - a.y) + a.x {
            c = !c;
        }
        j = i;
    }
    c
}

/// A flat polygon (any simple shape) at height y, its sides down to `y - th`
/// and its underside: a flight deck.
fn slab(m: &mut MB, poly: &[DVec2], y: f64, th: f64, top: [f32; 3], side: [f32; 3]) {
    let base = m.v.len() as u32;
    for p in poly {
        m.v.push(crate::render::MeshVertex { pos: v3(p.x, y, p.y).into(), nrm: [0.0, 1.0, 0.0], col: [top[0], top[1], top[2], TRIM] });
    }
    let tris = triangulate(poly);
    for t in &tris {
        m.i.extend([base + t[0] as u32, base + t[1] as u32, base + t[2] as u32]);
    }
    let base = m.v.len() as u32;
    for p in poly {
        m.v.push(crate::render::MeshVertex { pos: v3(p.x, y - th, p.y).into(), nrm: [0.0, -1.0, 0.0], col: [side[0], side[1], side[2], TRIM] });
    }
    for t in &tris {
        m.i.extend([base + t[0] as u32, base + t[2] as u32, base + t[1] as u32]);
    }
    let n = poly.len();
    for i in 0..n {
        let (a, b) = (poly[i], poly[(i + 1) % n]);
        m.poly(&[v3(a.x, y, a.y), v3(b.x, y, b.y), v3(b.x, y - th, b.y), v3(a.x, y - th, a.y)], side, TRIM);
    }
}

/// A flat quad painted on a deck at height y: corners in (x, z).
fn paint(m: &mut MB, y: f64, q: [DVec2; 4], col: [f32; 3]) {
    m.poly(&q.map(|p| v3(p.x, y, p.y)), col, TRIM);
}

/// A strip painted on a deck from a to b, `w` wide.
fn stripe(m: &mut MB, y: f64, a: DVec2, b: DVec2, w: f64, col: [f32; 3]) {
    let d = (b - a).normalize_or_zero();
    let n = DVec2::new(-d.y, d.x) * (w * 0.5);
    paint(m, y, [a - n, b - n, b + n, a + n], col);
}

/// Seven-segment digits: `o` the lower left of the first, `r` one unit
/// along the line of text, `u` one unit up the glyph (glyphs are 1 x 2).
fn digits(m: &mut MB, s: &str, o: Vec3, r: Vec3, u: Vec3, col: [f32; 3], mat: f32) {
    const T: f32 = 0.22;
    let segs: [[f32; 4]; 7] = [
        [0.0, 2.0 - T, 1.0, 2.0],
        [1.0 - T, 1.0, 1.0, 2.0],
        [1.0 - T, 0.0, 1.0, 1.0],
        [0.0, 0.0, 1.0, T],
        [0.0, 0.0, T, 1.0],
        [0.0, 1.0, T, 2.0],
        [0.0, 1.0 - T * 0.5, 1.0, 1.0 + T * 0.5],
    ];
    let on: [&[usize]; 10] = [&[0, 1, 2, 3, 4, 5], &[1, 2], &[0, 1, 6, 4, 3], &[0, 1, 6, 2, 3], &[5, 6, 1, 2], &[0, 5, 6, 2, 3], &[0, 5, 6, 4, 3, 2], &[0, 1, 2], &[0, 1, 2, 3, 4, 5, 6], &[0, 1, 2, 3, 5, 6]];
    for (k, ch) in s.chars().enumerate() {
        let Some(d) = ch.to_digit(10) else { continue };
        let ox = o + r * (k as f32 * 1.5);
        for &si in on[d as usize] {
            let [x0, y0, x1, y1] = segs[si];
            let p = |x: f32, y: f32| ox + r * x + u * (y * 0.5);
            m.poly(&[p(x0, y0), p(x1, y0), p(x1, y1), p(x0, y1)], col, mat);
        }
    }
}

/// A box with rows of windows on its long sides (lit after dark).
fn windows(m: &mut MB, x: f64, z0: f64, z1: f64, y: f64, h: f64, pitch: f64, w: f64, side: f64) {
    let n = ((z1 - z0) / pitch).floor() as i32;
    for k in 0..n {
        let z = z0 + pitch * (k as f64 + 0.5);
        let xs = x + side * 0.03;
        m.poly(&[v3(xs, y, z - w * 0.5), v3(xs, y, z + w * 0.5), v3(xs, y + h, z + w * 0.5), v3(xs, y + h, z - w * 0.5)], [0.05, 0.05, 0.045], LAMP);
    }
}

/// An upright elliptic funnel from y0 to y1, raked aft by `rake` (m over its height).
fn funnel(m: &mut MB, c: DVec2, a: f64, b: f64, y0: f64, y1: f64, rake: f64, col: [f32; 3], top: f64, top_col: [f32; 3]) {
    let ring = |y: f64| -> Vec<Vec3> {
        let k = (y - y0) / (y1 - y0);
        (0..=16).map(|j| {
            let t = j as f64 / 16.0 * std::f64::consts::TAU;
            v3(c.x + a * t.cos(), y, c.y + b * t.sin() + rake * k)
        }).collect()
    };
    let yt = y1 - top;
    m.grid(&[ring(y0), ring(yt)], col, TRIM, false);
    m.grid(&[ring(yt), ring(y1)], top_col, TRIM, false);
    let mut cap = ring(y1 - 0.8);
    cap.pop();
    m.poly(&cap, [0.02, 0.02, 0.02], TRIM);
}

const RED_BOTTOM: [f32; 3] = [0.42, 0.07, 0.05];
const WHITE: [f32; 3] = [0.86, 0.86, 0.84];
const DARKWIN: [f32; 3] = [0.05, 0.06, 0.07];
const NAVY_GREY: [f32; 3] = [0.40, 0.43, 0.46];
const DECK_DARK: [f32; 3] = [0.15, 0.155, 0.16];

// ------------------------------------------------------------------ the Titanic

fn liner_hull() -> HullP {
    HullP { len: 269.0, beam: 28.2, deck: 11.0, draft: 10.5, tb: 0.2, ta: 0.17, transom: 0.12, flare: 1.0, bow_flare: 0.9, rake: 0.04, p: 9.0, sheer_b: 3.6, sheer_s: 2.2 }
}
const LINER_FUNNELS: [f64; 4] = [-60.0, -36.5, -13.0, 10.5];
const LINER_BOATS: [f64; 8] = [-86.0, -76.0, -66.0, -56.0, 18.0, 28.0, 38.0, 48.0];

fn liner_mesh() -> Mesh {
    let h = liner_hull();
    let mut m = MB::default();
    hull_band(&mut m, &h, -h.draft, Some(0.4), RED_BOTTOM);
    hull_band(&mut m, &h, 0.4, None, [0.03, 0.03, 0.035]);
    hull_deck(&mut m, &h, [0.58, 0.48, 0.34]);
    // A gilt line along the sheer.
    for s in [-1.0, 1.0] {
        let pts: Vec<Vec3> = stations(40).iter().map(|&t| {
            let y = h.deck_at(t) - 0.6;
            v3(s * (h.half_w(t, y) + 0.04), y, h.z_at(t, y))
        }).collect();
        m.tube(&pts, 0.12, 4, [0.62, 0.48, 0.18], TRIM);
    }
    // Portholes: three rows along the black hull.
    for s in [-1.0, 1.0] {
        for y in [4.6, 7.0, 9.3] {
            let mut z = -112.0;
            while z < 112.0 {
                let t = (z + 134.5) / 269.0;
                let x = h.half_w(t, y);
                let xs = s * (x + 0.04);
                m.poly(&[v3(xs, y, z - 0.25), v3(xs, y, z + 0.25), v3(xs, y + 0.5, z + 0.25), v3(xs, y + 0.5, z - 0.25)], [0.06, 0.06, 0.05], LAMP);
                z += 2.1;
            }
        }
    }
    // White superstructure: B and A decks flush with the hull amidships,
    // the boat deck on top, and its deckhouses.
    let (z0, z1) = (-88.0, 68.0);
    let hw = 14.0;
    m.cube(Vec3::new(-hw as f32, 11.0, z0 as f32), Vec3::new(hw as f32, 16.4, z1 as f32), WHITE, TRIM);
    m.cube(Vec3::new(-hw as f32 - 0.05, 16.3, z0 as f32), Vec3::new(hw as f32 + 0.05, 16.5, z1 as f32), [0.56, 0.47, 0.34], TRIM);
    for s in [-1.0, 1.0] {
        windows(&mut m, s * hw, z0 + 4.0, z1 - 4.0, 12.3, 1.2, 2.4, 1.1, s);
        // A deck's promenade: open forward, a dark band.
        m.poly(&[v3(s * (hw + 0.03), 14.2, z0 + 6.0), v3(s * (hw + 0.03), 14.2, -20.0), v3(s * (hw + 0.03), 15.8, -20.0), v3(s * (hw + 0.03), 15.8, z0 + 6.0)], [0.08, 0.08, 0.075], TRIM);
        windows(&mut m, s * hw, -20.0, z1 - 4.0, 14.5, 1.1, 3.0, 1.4, s);
    }
    m.cube(Vec3::new(-8.0, 16.4, -80.0), Vec3::new(8.0, 19.2, 40.0), WHITE, TRIM);
    for s in [-1.0, 1.0] {
        windows(&mut m, s * 8.0, -78.0, 38.0, 17.4, 1.0, 3.5, 1.0, s);
    }
    // Bridge and wheelhouse forward, its wings out to the sides.
    m.cube(Vec3::new(-7.0, 16.4, -88.0), Vec3::new(7.0, 19.6, -81.0), WHITE, TRIM);
    m.cube(Vec3::new(-14.5, 18.6, -87.0), Vec3::new(14.5, 19.0, -84.0), WHITE, TRIM);
    m.poly(&[v3(-6.0, 17.6, -88.03), v3(6.0, 17.6, -88.03), v3(6.0, 18.8, -88.03), v3(-6.0, 18.8, -88.03)], DARKWIN, GLASS);
    // The aft deckhouse and the docking bridge at the stern.
    m.cube(Vec3::new(-6.0, 16.4, 50.0), Vec3::new(6.0, 18.8, 66.0), WHITE, TRIM);
    m.cube(Vec3::new(-12.0, 15.6, 112.0), Vec3::new(12.0, 16.2, 114.0), WHITE, TRIM);
    // Four buff funnels with black tops, raked aft.
    for &z in &LINER_FUNNELS {
        funnel(&mut m, DVec2::new(0.0, z), 3.0, 3.8, 16.4, 46.0, 1.7, [0.74, 0.53, 0.26], 4.6, [0.03, 0.03, 0.03]);
        // (the casing it stands on, and the steam pipes up its front)
        m.cube(Vec3::new(-3.6, 19.2, z as f32 - 4.3), Vec3::new(3.6, 20.2, z as f32 + 4.3), WHITE, TRIM);
        m.tube(&[v3(-1.2, 20.0, z - 4.0), v3(-1.2, 45.0, z - 2.6)], 0.3, 5, [0.7, 0.5, 0.25], TRIM);
    }
    // Masts, with the crow's nest on the foremast and the yards.
    let mast_col = [0.6, 0.48, 0.3];
    m.tube(&[v3(0.0, 14.0, -102.0), v3(0.0, 40.0, -101.4), v3(0.0, 62.0, -100.8)], 0.45, 8, mast_col, TRIM);
    m.tube(&[v3(0.0, 13.0, 92.0), v3(0.0, 38.0, 92.6), v3(0.0, 57.0, 93.2)], 0.4, 8, mast_col, TRIM);
    m.tube(&[v3(0.0, 29.0, -101.6), v3(0.0, 31.4, -101.55)], 1.2, 10, WHITE, TRIM);
    m.tube(&[v3(-5.0, 36.0, -101.5), v3(5.0, 36.0, -101.5)], 0.15, 4, mast_col, TRIM);
    // The wireless aerials between the mastheads, and the stays to the bow and stern.
    m.tube(&[v3(-1.2, 60.0, -100.8), v3(-1.2, 55.0, 93.2)], 0.08, 3, [0.15, 0.15, 0.15], TRIM);
    m.tube(&[v3(1.2, 60.0, -100.8), v3(1.2, 55.0, 93.2)], 0.08, 3, [0.15, 0.15, 0.15], TRIM);
    m.tube(&[v3(0.0, 61.0, -100.8), v3(0.0, 15.0, -133.0)], 0.07, 3, [0.15, 0.15, 0.15], TRIM);
    m.tube(&[v3(0.0, 56.0, 93.2), v3(0.0, 14.0, 128.0)], 0.07, 3, [0.15, 0.15, 0.15], TRIM);
    // Lifeboats under their davits along the boat deck.
    for s in [-1.0f32, 1.0] {
        for &z in &LINER_BOATS {
            let x = s * 12.4;
            m.rbox(Vec3::new(x - 1.3, 17.0, z as f32 - 4.2), Vec3::new(x + 1.3, 18.3, z as f32 + 4.2), 0.6, [0.9, 0.9, 0.87], TRIM);
            for dz in [-4.5f32, 4.5] {
                m.tube(&[Vec3::new(s * 13.6, 16.4, z as f32 + dz), Vec3::new(s * 13.6, 19.4, z as f32 + dz), Vec3::new(s * 12.2, 19.8, z as f32 + dz)], 0.12, 4, WHITE, TRIM);
            }
        }
    }
    // Cargo cranes in the well decks, and the anchors.
    for z in [-108.0f32, -96.0, 80.0, 100.0] {
        m.tube(&[Vec3::new(0.0, 11.0, z), Vec3::new(0.0, 16.0, z), Vec3::new(0.0, 15.0, z + 7.0)], 0.25, 5, mast_col, TRIM);
    }
    for s in [-1.0, 1.0] {
        let t = 0.06;
        let y = 9.0;
        m.cube(v3(s * (h.half_w(t, y) + 0.1) - 0.15, y - 1.5, h.z_at(t, y) - 1.0), v3(s * (h.half_w(t, y) + 0.1) + 0.15, y, h.z_at(t, y) + 1.0), [0.1, 0.1, 0.1], TRIM);
    }
    m.done()
}

// ------------------------------------------------------------------ the mega ship

fn mega_hull() -> HullP {
    HullP { len: 1000.0, beam: 140.0, deck: 30.0, draft: 16.0, tb: 0.14, ta: 0.08, transom: 0.7, flare: 1.02, bow_flare: 0.8, rake: 0.4, p: 10.0, sheer_b: 7.0, sheer_s: 0.0 }
}
const MEGA_ACCOM: (f64, f64) = (372.0, 404.0);

/// The mega ship's container bays: (z0, z1, half width, height of the stacks) and their colours.
fn mega_bays() -> Vec<(f64, f64, f64, Vec<(f64, f64, f64, [f32; 3])>)> {
    let h = mega_hull();
    let pal: [[f32; 3]; 10] = [[0.08, 0.25, 0.5], [0.55, 0.1, 0.08], [0.75, 0.38, 0.08], [0.12, 0.38, 0.22], [0.55, 0.55, 0.55], [0.82, 0.82, 0.8], [0.35, 0.2, 0.12], [0.8, 0.65, 0.12], [0.06, 0.4, 0.42], [0.2, 0.2, 0.22]];
    let mut out = Vec::new();
    let mut z = -405.0;
    let mut k = 0u32;
    while z < 360.0 {
        let z1 = z + 12.4;
        if z1 > MEGA_ACCOM.0 - 6.0 {
            break;
        }
        let t = (z + 500.0) / 1000.0;
        let hw = h.half_w(t.max(0.0), h.deck).min(h.half_w(((z1 + 500.0) / 1000.0).min(1.0), h.deck)) - 3.0;
        if hw > 8.0 {
            let mut stacks = Vec::new();
            let n = (2.0 * hw / 9.8).floor() as i32;
            let x0 = -(n as f64) * 9.8 * 0.5;
            for j in 0..n {
                let hsh = |a: u32| ((k.wrapping_mul(2654435761) ^ (j as u32).wrapping_mul(40503) ^ a.wrapping_mul(97)).wrapping_mul(2246822519) >> 13) % 1000;
                let tiers = 5 + hsh(1) % 5;
                let mut y = 30.0;
                for tier in 0..tiers {
                    let c = pal[(hsh(tier + 7) % 10) as usize];
                    stacks.push((x0 + j as f64 * 9.8, y, y + 2.6, c));
                    y += 2.6;
                }
            }
            out.push((z, z1, hw, stacks));
        }
        k += 1;
        z = z1 + if k % 2 == 0 { 3.0 } else { 0.6 };
    }
    out
}

fn mega_mesh() -> Mesh {
    let h = mega_hull();
    let mut m = MB::default();
    hull_band(&mut m, &h, -h.draft, Some(1.0), RED_BOTTOM);
    hull_band(&mut m, &h, 1.0, Some(h.deck - 2.5), [0.05, 0.11, 0.24]);
    hull_band(&mut m, &h, h.deck - 2.5, None, [0.75, 0.75, 0.73]);
    hull_deck(&mut m, &h, [0.32, 0.3, 0.28]);
    // Containers: each stack a column of boxes in many colours.
    for (z0, z1, _, stacks) in mega_bays() {
        for (x, y0, y1, c) in stacks {
            let lo = Vec3::new(x as f32 + 0.15, y0 as f32, z0 as f32);
            let hi = Vec3::new((x + 9.8) as f32 - 0.15, y1 as f32 - 0.08, z1 as f32);
            m.cube(lo, hi, c, TRIM);
        }
    }
    // The accommodation block near the stern: ten decks of windows, the
    // bridge across the full width, the funnel behind.
    let (a0, a1) = MEGA_ACCOM;
    m.cube(Vec3::new(-36.0, 30.0, a0 as f32), Vec3::new(36.0, 84.0, a1 as f32), WHITE, TRIM);
    m.cube(Vec3::new(-70.0, 84.0, a0 as f32), Vec3::new(70.0, 91.0, a0 as f32 + 14.0), WHITE, TRIM);
    m.poly(&[v3(-69.0, 86.0, a0 - 0.05), v3(69.0, 86.0, a0 - 0.05), v3(69.0, 89.5, a0 - 0.05), v3(-69.0, 89.5, a0 - 0.05)], DARKWIN, GLASS);
    for k in 0..10 {
        let y = 33.0 + k as f64 * 5.0;
        for zz in [a0 - 0.04, a1 + 0.04] {
            let mut x = -33.0;
            while x < 32.0 {
                m.poly(&[v3(x, y, zz), v3(x + 1.6, y, zz), v3(x + 1.6, y + 1.5, zz), v3(x, y + 1.5, zz)], [0.06, 0.06, 0.05], LAMP);
                x += 3.2;
            }
        }
    }
    m.cube(Vec3::new(-12.0, 30.0, 418.0), Vec3::new(12.0, 98.0, 446.0), [0.15, 0.15, 0.16], TRIM);
    m.cube(Vec3::new(-12.1, 88.0, 417.9), Vec3::new(12.1, 92.0, 446.1), [0.6, 0.1, 0.08], TRIM);
    m.tube(&[v3(0.0, 91.0, a0 + 7.0), v3(0.0, 108.0, a0 + 7.0)], 0.7, 6, WHITE, TRIM);
    // Big ship-to-shore cranes are not carried; a pair of gantry masts forward.
    for z in [-430.0f32, -440.0] {
        m.tube(&[Vec3::new(-40.0, 30.0 + 7.0, z), Vec3::new(-40.0, 62.0, z)], 1.0, 6, [0.8, 0.6, 0.1], TRIM);
        m.tube(&[Vec3::new(40.0, 37.0, z), Vec3::new(40.0, 62.0, z)], 1.0, 6, [0.8, 0.6, 0.1], TRIM);
        m.tube(&[Vec3::new(-40.0, 62.0, z), Vec3::new(40.0, 62.0, z)], 0.8, 6, [0.8, 0.6, 0.1], TRIM);
    }
    m.done()
}

// ------------------------------------------------------------------ the carriers

/// A carrier's flight deck and what is on it, scaled from the Nimitz.
struct CarrierP {
    hull: HullP,
    outline: Vec<DVec2>,
    /// the angled deck (if any): its ramp at the stern, direction toward the bow, width, length
    landing: Option<Landing>,
    /// catapults: (x, z start, z end)
    cats: Vec<(f64, f64, f64)>,
    /// the island: (x0, x1, z0, z1) and its two heights above the deck
    island: (f64, f64, f64, f64, f64, f64),
    /// the runway down the deck of the mega carrier: its centre line x and half width
    runway: Option<(f64, f64)>,
    number: &'static str,
}

/// A carrier's landing area: where the ramp is, the way toward the bow
/// (unit, in the deck's x, z), its width and length, where the arresting
/// wires cross it (m from the ramp) and how far they run out.
#[derive(Clone, Debug)]
pub struct Landing {
    pub ramp: DVec2,
    pub dir: DVec2,
    pub width: f64,
    pub len: f64,
    pub wires: Vec<f64>,
    pub runout: f64,
}

fn carrier_p() -> CarrierP {
    let hull = HullP { len: 317.0, beam: 41.0, deck: 17.8, draft: 11.3, tb: 0.26, ta: 0.1, transom: 0.72, flare: 1.32, bow_flare: 0.7, rake: 0.5, p: 6.0, sheer_b: 0.0, sheer_s: 0.0 };
    let o = [(-10.0, -166.5), (10.0, -166.5), (22.0, -150.0), (31.0, -120.0), (35.0, -80.0), (36.0, -40.0), (36.0, 110.0), (34.0, 140.0), (28.0, 166.5), (-14.0, 166.5), (-20.0, 150.0), (-30.0, 100.0), (-40.0, 40.0), (-43.0, -10.0), (-43.0, -55.0), (-36.0, -75.0), (-25.0, -100.0), (-20.0, -130.0), (-15.0, -155.0)];
    let a = 9f64.to_radians();
    CarrierP {
        hull,
        outline: o.iter().map(|&(x, z)| DVec2::new(x, z)).collect(),
        landing: Some(Landing { ramp: DVec2::new(2.0, 166.5), dir: DVec2::new(-a.sin(), -a.cos()), width: 24.0, len: 210.0, wires: vec![48.0, 60.0, 72.0, 84.0], runout: 100.0 }),
        cats: vec![(7.0, -55.0, -150.0), (-7.0, -55.0, -150.0)],
        island: (27.0, 35.0, 10.0, 48.0, 14.0, 23.0),
        runway: None,
        number: "68",
    }
}

fn megacarrier_p() -> CarrierP {
    let hull = HullP { len: 3450.0, beam: 300.0, deck: 47.0, draft: 30.0, tb: 0.2, ta: 0.06, transom: 0.8, flare: 1.25, bow_flare: 0.7, rake: 0.45, p: 7.0, sheer_b: 0.0, sheer_s: 0.0 };
    let o = [(-130.0, -1800.0), (130.0, -1800.0), (205.0, -1550.0), (210.0, -1300.0), (210.0, 1760.0), (195.0, 1800.0), (-195.0, 1800.0), (-210.0, 1760.0), (-210.0, -1300.0), (-205.0, -1550.0)];
    CarrierP {
        hull,
        outline: o.iter().map(|&(x, z)| DVec2::new(x, z)).collect(),
        landing: None,
        cats: vec![],
        island: (150.0, 190.0, 200.0, 500.0, 30.0, 58.0),
        runway: Some((-50.0, 30.0)),
        number: "99",
    }
}

/// Where parked aircraft stand on a carrier's deck: (x, z, yaw (rad, 0 = bow), which).
fn parked(kind: Kind) -> Vec<(f64, f64, f64, u8)> {
    match kind {
        Kind::Carrier => {
            let mut v = Vec::new();
            for z in [-122.0, -106.0, -90.0, -74.0] {
                v.push((25.0, z, 1.75, 0));
            }
            for z in [92.0, 108.0, 124.0, 140.0] {
                v.push((25.0, z, 1.4, 0));
            }
            v
        }
        Kind::MegaCarrier => vec![(110.0, -1250.0, 1.57, 1), (110.0, -1150.0, 1.57, 1), (110.0, -1050.0, 1.57, 1), (110.0, 900.0, 1.57, 1), (110.0, 1000.0, 1.57, 1)],
        _ => vec![],
    }
}

fn carrier_mesh(p: &CarrierP) -> Mesh {
    let h = &p.hull;
    let mut m = MB::default();
    let big = h.len / 317.0;
    let fy = DIMS[if big > 2.0 { 3 } else { 2 }].deck;
    hull_band(&mut m, h, -h.draft, Some(0.6 * big.sqrt()), RED_BOTTOM);
    hull_band(&mut m, h, 0.6 * big.sqrt(), None, NAVY_GREY);
    hull_deck(&mut m, h, [0.3, 0.32, 0.34]);
    // (the hangar deck's open bays along the sides under the flight deck)
    for s in [-1.0, 1.0] {
        let y = h.deck - 1.0 * big;
        for k in 0..6 {
            let z0 = -h.len * 0.3 + k as f64 * h.len * 0.11;
            let t = (z0 + h.len * 0.5) / h.len;
            let x = s * (h.half_w(t, y) + 0.05);
            m.poly(&[v3(x, y - 5.0 * big, z0), v3(x, y - 5.0 * big, z0 + h.len * 0.06), v3(x, y, z0 + h.len * 0.06), v3(x, y, z0)], [0.08, 0.085, 0.09], TRIM);
        }
    }
    // The flight deck: a slab over the hull, overhanging it to port for the angled deck.
    slab(&mut m, &p.outline, fy, fy - h.deck, if big > 2.0 { [0.075, 0.078, 0.082] } else { DECK_DARK }, NAVY_GREY);
    let py = fy + 0.06 * big.max(1.0);
    let wht = [0.82, 0.82, 0.8];
    let yel = [0.78, 0.62, 0.1];
    if let Some(l) = &p.landing {
        // Edges and dashed centre line of the landing area, the ramp's
        // stripes, the wires across it.
        let n = DVec2::new(-l.dir.y, l.dir.x);
        let a = l.ramp;
        let b = l.ramp + l.dir * l.len;
        for s in [-1.0, 1.0] {
            stripe(&mut m, py, a + n * (s * l.width * 0.5), b + n * (s * l.width * 0.5), 0.5, wht);
        }
        let mut d = 6.0;
        while d < l.len - 8.0 {
            stripe(&mut m, py, a + l.dir * d, a + l.dir * (d + 6.0), 0.6, wht);
            d += 12.0;
        }
        for k in 0..6 {
            let x = -l.width * 0.5 + (k as f64 + 0.5) * l.width / 6.0;
            stripe(&mut m, py, a + n * x + l.dir * 2.0, a + n * x + l.dir * 14.0, 1.2, if k % 2 == 0 { wht } else { [0.75, 0.15, 0.1] });
        }
        for &w in &l.wires {
            let c = a + l.dir * w;
            let (e0, e1) = (c - n * (l.width * 0.5 + 1.0), c + n * (l.width * 0.5 + 1.0));
            m.tube(&[v3(e0.x, fy + 0.18, e0.y), v3(e1.x, fy + 0.18, e1.y)], 0.09, 5, [0.55, 0.55, 0.55], TRIM);
            for e in [e0, e1] {
                m.cube(v3(e.x - 0.4, fy, e.y - 0.4), v3(e.x + 0.4, fy + 0.35, e.y + 0.4), [0.3, 0.3, 0.3], TRIM);
            }
        }
        // The foul line: yellow, from the ramp's starboard corner forward.
        stripe(&mut m, py, a + n * (l.width * 0.5 + 3.0), b + n * (l.width * 0.5 + 3.0), 0.35, yel);
        // The landing mirror (the "meatball") on the port side.
        let mp = a + l.dir * 95.0 - n * (l.width * 0.5 + 8.0);
        m.cube(v3(mp.x - 1.5, fy, mp.y - 0.6), v3(mp.x + 1.5, fy + 2.4, mp.y + 0.6), [0.2, 0.2, 0.2], TRIM);
    }
    for &(x, z0, z1) in &p.cats {
        // The catapult's shuttle slot, its yellow edges and the launch line.
        stripe(&mut m, py, DVec2::new(x, z0), DVec2::new(x, z1), 0.35, [0.05, 0.05, 0.05]);
        for s in [-1.0, 1.0] {
            stripe(&mut m, py, DVec2::new(x + s * 1.0, z0), DVec2::new(x + s * 1.0, z1), 0.15, yel);
        }
        stripe(&mut m, py, DVec2::new(x - 4.0, z0 + 1.0), DVec2::new(x + 4.0, z0 + 1.0), 0.4, wht);
        // the jet blast deflector, down, behind the start
        m.cube(v3(x - 5.0, fy, z0 + 15.0), v3(x + 5.0, fy + 0.35, z0 + 19.0), [0.32, 0.33, 0.34], TRIM);
    }
    if let Some((x, w)) = p.runway {
        // A runway down the deck: edges, centre line, thresholds, aiming points.
        let (z0, z1) = (1770.0, -1770.0);
        for s in [-1.0, 1.0] {
            stripe(&mut m, py, DVec2::new(x + s * w, z0), DVec2::new(x + s * w, z1), 1.0, wht);
        }
        let mut z = z0 - 80.0;
        while z > z1 + 80.0 {
            stripe(&mut m, py, DVec2::new(x, z), DVec2::new(x, z - 30.0), 1.0, wht);
            z -= 50.0;
        }
        for zt in [z0, z1] {
            let dir = if zt > 0.0 { -1.0 } else { 1.0 };
            for k in 0..8 {
                for s in [-1.0, 1.0] {
                    let xx = x + s * (3.0 + k as f64 * 3.3);
                    stripe(&mut m, py, DVec2::new(xx, zt), DVec2::new(xx, zt + dir * 30.0), 1.8, wht);
                }
            }
            for s in [-1.0, 1.0] {
                stripe(&mut m, py, DVec2::new(x + s * 13.0, zt + dir * 300.0), DVec2::new(x + s * 13.0, zt + dir * 360.0), 9.0, wht);
                for k in 0..3 {
                    let zz = zt + dir * (450.0 + k as f64 * 150.0);
                    stripe(&mut m, py, DVec2::new(x + s * 13.0, zz), DVec2::new(x + s * 13.0, zz + dir * 22.0), 5.0, wht);
                }
            }
        }
        // a taxi line to the parking on the starboard side
        stripe(&mut m, py, DVec2::new(x + 40.0, 1500.0), DVec2::new(80.0, 1100.0), 0.8, yel);
    }
    // The hull number, big on the deck toward the bow, read from astern.
    {
        let k = big as f32;
        let (gw, z) = (6.0 * k, (-h.len * 0.5 + 40.0 * big) as f32);
        let w = gw * (1.5 * p.number.len() as f32 - 0.5);
        let x = if p.runway.is_some() { p.runway.unwrap().0 as f32 } else { 0.0 };
        digits(&mut m, p.number, Vec3::new(x - w * 0.5, py as f32, z), Vec3::new(gw, 0.0, 0.0), Vec3::new(0.0, 0.0, -gw * 2.0), wht, TRIM);
    }
    // The island: two tiers, the bridge windows, the mast and its radars,
    // the stack, the number on its side.
    let (x0, x1, z0, z1, h1, h2) = p.island;
    let k = big as f32;
    let (x0f, x1f, z0f, z1f) = (x0 as f32, x1 as f32, z0 as f32, z1 as f32);
    let fyf = fy as f32;
    m.cube(Vec3::new(x0f, fyf, z0f), Vec3::new(x1f, fyf + h1 as f32, z1f), NAVY_GREY, TRIM);
    let (ix0, ix1, iz0, iz1) = (x0f + 1.0 * k, x1f - 1.0 * k, z0f + 6.0 * k, z1f - 6.0 * k);
    m.cube(Vec3::new(ix0, fyf + h1 as f32, iz0), Vec3::new(ix1, fyf + h2 as f32, iz1), NAVY_GREY, TRIM);
    for (yy, hh) in [(h1 - 2.4 * big, 1.4 * big), (h2 - 2.6 * big, 1.6 * big)] {
        let (xa, xb, za, zb) = if yy < h1 { (x0, x1, z0, z1) } else { (ix0 as f64, ix1 as f64, iz0 as f64, iz1 as f64) };
        let y = fy + yy;
        for (p0, p1) in [((xa - 0.03, za - 0.03), (xb + 0.03, za - 0.03)), ((xa - 0.03, za - 0.03), (xa - 0.03, zb + 0.03)), ((xb + 0.03, za - 0.03), (xb + 0.03, zb + 0.03))] {
            m.poly(&[v3(p0.0, y, p0.1), v3(p1.0, y, p1.1), v3(p1.0, y + hh, p1.1), v3(p0.0, y + hh, p0.1)], DARKWIN, GLASS);
        }
    }
    // the side's number, white
    let num_h = (h1 as f32 * 0.55).min(9.0 * k);
    for (xs, dir) in [(x0f - 0.05, -1.0f32), (x1f + 0.05, 1.0)] {
        let gw = num_h * 0.5;
        let w = gw * (1.5 * p.number.len() as f32 - 0.5);
        let zc = (z0f + z1f) * 0.5;
        digits(&mut m, p.number, Vec3::new(xs, fyf + h1 as f32 * 0.25, zc + dir * w * 0.5), Vec3::new(0.0, 0.0, -dir * gw), Vec3::new(0.0, num_h, 0.0), wht, TRIM);
    }
    let mx = (x0 + x1) * 0.5;
    let mz = z0 + (z1 - z0) * 0.55;
    let top = fy + h2;
    m.tube(&[v3(mx, top, mz), v3(mx, top + 17.0 * big, mz)], 0.6 * k, 6, NAVY_GREY, TRIM);
    m.tube(&[v3(mx - 5.0 * big, top + 12.0 * big, mz), v3(mx + 5.0 * big, top + 12.0 * big, mz)], 0.2 * k, 4, NAVY_GREY, TRIM);
    m.tube(&[v3(mx - 3.5 * big, top + 15.5 * big, mz), v3(mx + 3.5 * big, top + 15.5 * big, mz)], 0.18 * k, 4, NAVY_GREY, TRIM);
    // the big search radar: a tilted dish on a pedestal
    let rz = z0 + (z1 - z0) * 0.3;
    m.cube(v3(mx - 0.8 * big, top, rz - 0.8 * big), v3(mx + 0.8 * big, top + 3.0 * big, rz + 0.8 * big), NAVY_GREY, TRIM);
    m.poly(&[v3(mx - 3.5 * big, top + 3.0 * big, rz - 0.5 * big), v3(mx + 3.5 * big, top + 3.0 * big, rz - 0.5 * big), v3(mx + 3.5 * big, top + 8.0 * big, rz + 1.2 * big), v3(mx - 3.5 * big, top + 8.0 * big, rz + 1.2 * big)], [0.3, 0.32, 0.34], TRIM);
    // the phased array faces and the stack's exhaust
    m.cube(v3(ix0 as f64 - 0.3, top - 6.0 * big, iz0 as f64 + 2.0 * big), v3(ix0 as f64, top - 2.0 * big, iz0 as f64 + 6.0 * big), [0.25, 0.27, 0.29], TRIM);
    m.cube(v3(mx - 2.5 * big, top, z1 - 9.0 * big), v3(mx + 2.5 * big, top + 4.0 * big, z1 - 3.0 * big), [0.1, 0.1, 0.11], TRIM);
    // Deck edge: the catwalks' rails as a dark lip, and the elevators' outlines.
    let n = p.outline.len();
    for i in 0..n {
        let (a, b) = (p.outline[i], p.outline[(i + 1) % n]);
        m.tube(&[v3(a.x, fy + 0.5, a.y), v3(b.x, fy + 0.5, b.y)], 0.05 * k, 3, [0.25, 0.25, 0.25], TRIM);
    }
    m.done()
}

/// A Hornet as it stands parked on deck (for the instances): origin on the
/// deck under the main wheels, nose to -Z.
fn fighter_mesh() -> Mesh {
    let mut m = MB::default();
    let grey = [0.36, 0.38, 0.41];
    let dark = [0.1, 0.1, 0.11];
    // Fuselage: stations (z, half width, half height, centre height).
    let st: [(f32, f32, f32, f32); 9] = [(-10.2, 0.05, 0.05, 2.25), (-9.0, 0.35, 0.35, 2.25), (-7.0, 0.6, 0.6, 2.3), (-5.0, 0.8, 0.75, 2.4), (-2.5, 1.3, 0.85, 2.35), (0.5, 1.45, 0.85, 2.3), (3.5, 1.3, 0.75, 2.3), (5.8, 1.0, 0.6, 2.3), (7.0, 0.85, 0.5, 2.3)];
    let rows: Vec<Vec<Vec3>> = st.iter().map(|&(z, w, h, y)| (0..=12).map(|j| {
        let a = j as f32 / 12.0 * std::f32::consts::TAU;
        let (c, s) = (a.cos(), a.sin());
        // (a flattened section, flat-bottomed)
        Vec3::new(w * c.signum() * c.abs().powf(0.7), y + h * s.signum() * s.abs().powf(if s < 0.0 { 0.4 } else { 0.9 }), z)
    }).collect()).collect();
    m.grid(&rows, grey, TRIM, false);
    // the canopy
    let can: Vec<Vec<Vec3>> = [(-6.6f32, 0.0f32), (-5.6, 0.45), (-4.2, 0.55), (-2.8, 0.4), (-1.8, 0.0)].iter().map(|&(z, h)| (0..=8).map(|j| {
        let a = j as f32 / 8.0 * std::f32::consts::PI;
        Vec3::new(0.48 * a.cos(), 3.0 + h * a.sin(), z)
    }).collect()).collect();
    m.grid(&can, [0.12, 0.13, 0.15], GLASS, false);
    for s in [-1.0f32, 1.0] {
        // the leading edge extension along the nose, the wing, the stabilator
        m.poly(&[Vec3::new(s * 0.7, 2.35, -6.5), Vec3::new(s * 1.5, 2.3, -1.0), Vec3::new(s * 1.4, 2.3, 0.5), Vec3::new(s * 0.75, 2.35, 0.5)], grey, TRIM);
        m.poly(&[Vec3::new(s * 1.4, 2.25, -1.0), Vec3::new(s * 6.15, 2.15, 1.6), Vec3::new(s * 6.15, 2.15, 3.2), Vec3::new(s * 1.4, 2.25, 3.6)], grey, TRIM);
        m.poly(&[Vec3::new(s * 1.0, 2.2, 5.0), Vec3::new(s * 3.3, 2.1, 7.1), Vec3::new(s * 3.3, 2.1, 8.0), Vec3::new(s * 1.0, 2.2, 7.8)], grey, TRIM);
        // the twin fins, canted out
        m.poly(&[Vec3::new(s * 0.9, 2.9, 3.6), Vec3::new(s * 1.9, 5.6, 5.6), Vec3::new(s * 1.95, 5.6, 6.9), Vec3::new(s * 1.0, 2.9, 7.0)], grey, TRIM);
        // intakes, the main gear and its wheel
        m.cube(Vec3::new(s * 1.1 - 0.35, 1.7, -2.6), Vec3::new(s * 1.1 + 0.35, 2.3, -1.6), dark, TRIM);
        m.tube(&[Vec3::new(s * 1.6, 0.4, 0.0), Vec3::new(s * 1.3, 1.9, 0.2)], 0.1, 4, [0.3; 3], TRIM);
        m.tube(&[Vec3::new(s * 1.6, 0.36, -0.3), Vec3::new(s * 1.6, 0.36, 0.3)], 0.36, 8, dark, TRIM);
        // the two exhausts
        m.tube(&[Vec3::new(s * 0.5, 2.25, 6.8), Vec3::new(s * 0.5, 2.25, 7.5)], 0.42, 8, [0.2, 0.19, 0.18], TRIM);
    }
    m.tube(&[Vec3::new(0.0, 0.3, -5.9), Vec3::new(0.0, 1.9, -5.8)], 0.08, 4, [0.3; 3], TRIM);
    m.tube(&[Vec3::new(-0.15, 0.3, -5.9), Vec3::new(0.15, 0.3, -5.9)], 0.3, 6, dark, TRIM);
    m.done()
}

/// The ships' meshes, in `MESH` order: liner, mega ship, carrier, mega carrier, parked Hornet.
pub fn meshes() -> Vec<Mesh> {
    vec![liner_mesh(), mega_mesh(), carrier_mesh(&carrier_p()), carrier_mesh(&megacarrier_p()), fighter_mesh()]
}
/// Index of the first ship mesh among the instanced solids (after the speedboat's two).
pub const MESH: u32 = crate::boat::MESH + 2;
pub const M_FIGHTER: u32 = MESH + 4;
pub const N_MESHES: usize = 5;

// ------------------------------------------------------------------ what is solid

/// What can be touched on a ship, in its own frame.
pub struct Geom {
    /// the deck's edge in (x, z), and its height along the ship (z, y)
    pub outline: Vec<DVec2>,
    pub prof: Vec<(f64, f64)>,
    pub draft: f64,
    /// solid boxes on the deck: superstructure, island, funnels, parked aircraft
    pub blocks: Vec<(DVec3, DVec3)>,
    pub landing: Option<Landing>,
    pub cats: Vec<(f64, f64, f64)>,
    /// where an aircraft starts on it: the deck spot (x, z) its nose wheel
    /// stands on (the catapult's start), for small and for big aircraft
    pub start_small: Option<DVec2>,
    pub start_big: Option<DVec2>,
    /// farthest any of it is from the origin
    pub radius: f64,
}

impl Geom {
    pub fn deck_y(&self, z: f64) -> f64 {
        let p = &self.prof;
        if z <= p[0].0 {
            return p[0].1;
        }
        for w in p.windows(2) {
            if z <= w[1].0 {
                let k = (z - w[0].0) / (w[1].0 - w[0].0).max(1e-9);
                return w[0].1 + (w[1].1 - w[0].1) * k;
            }
        }
        p[p.len() - 1].1
    }
}

fn hull_prof(h: &HullP) -> Vec<(f64, f64)> {
    stations(40).iter().map(|&t| (h.z_at(t, h.deck_at(t)), h.deck_at(t))).collect()
}

fn bx(x0: f64, x1: f64, y0: f64, y1: f64, z0: f64, z1: f64) -> (DVec3, DVec3) {
    (DVec3::new(x0, y0, z0), DVec3::new(x1, y1, z1))
}

/// A parked aircraft's box: `half` its half size along and across.
fn parked_block(x: f64, z: f64, yaw: f64, along: f64, across: f64, y0: f64, h: f64) -> (DVec3, DVec3) {
    let (c, s) = (yaw.cos().abs(), yaw.sin().abs());
    let hx = across * c + along * s;
    let hz = along * c + across * s;
    bx(x - hx, x + hx, y0, y0 + h, z - hz, z + hz)
}

static GEOM: LazyLock<[Geom; 4]> = LazyLock::new(|| {
    let liner = {
        let h = liner_hull();
        let mut blocks = vec![bx(-14.0, 14.0, 11.0, 16.4, -88.0, 68.0), bx(-8.0, 8.0, 16.4, 19.2, -80.0, 40.0)];
        for &z in &LINER_FUNNELS {
            blocks.push(bx(-3.3, 3.3, 16.4, 46.0, z - 4.1, z + 5.8));
        }
        Geom { outline: hull_outline(&h), prof: hull_prof(&h), draft: h.draft, blocks, landing: None, cats: vec![], start_small: None, start_big: None, radius: 140.0 }
    };
    let mega = {
        let h = mega_hull();
        let mut blocks = vec![bx(-36.0, 36.0, 30.0, 84.0, MEGA_ACCOM.0, MEGA_ACCOM.1), bx(-70.0, 70.0, 84.0, 91.0, MEGA_ACCOM.0, MEGA_ACCOM.0 + 14.0), bx(-12.0, 12.0, 30.0, 98.0, 418.0, 446.0)];
        for (z0, z1, _, stacks) in mega_bays() {
            for (x, _, y1, _) in stacks.iter().filter(|s| s.2 > 0.0) {
                blocks.push(bx(*x, x + 9.8, 30.0, *y1, z0, z1));
            }
        }
        // (one block per stack: keep only each stack's top)
        blocks.sort_by(|a, b| (a.0.x, a.0.z, a.1.y).partial_cmp(&(b.0.x, b.0.z, b.1.y)).unwrap());
        blocks.dedup_by(|b, a| a.0.x == b.0.x && a.0.z == b.0.z && a.0.y == b.0.y && { a.1.y = a.1.y.max(b.1.y); true });
        Geom { outline: hull_outline(&h), prof: hull_prof(&h), draft: h.draft, blocks, landing: None, cats: vec![], start_small: None, start_big: None, radius: 520.0 }
    };
    let carrier = |p: CarrierP, kind: Kind| {
        let d = kind.dims();
        let (x0, x1, z0, z1, h1, h2) = p.island;
        let k = p.hull.len / 317.0;
        let mut blocks = vec![bx(x0, x1, d.deck, d.deck + h1, z0, z1), bx(x0 + k, x1 - k, d.deck + h1, d.deck + h2, z0 + 6.0 * k, z1 - 6.0 * k)];
        for (x, z, yaw, which) in parked(kind) {
            blocks.push(if which == 0 { parked_block(x, z, yaw, 8.5, 2.5, d.deck, 4.5) } else { parked_block(x, z, yaw, 32.0, 3.4, d.deck, 8.0) });
        }
        let radius = p.outline.iter().map(|q| q.length()).fold(0.0, f64::max) + 10.0;
        let start_small = p.cats.first().map(|c| DVec2::new(c.0, c.1));
        let start_big = Some(match p.runway {
            Some((x, _)) => DVec2::new(x, 1700.0),
            None => DVec2::new(0.0, 0.0),
        });
        Geom { outline: p.outline, prof: vec![(-1e9, d.deck), (1e9, d.deck)], draft: p.hull.draft, blocks, landing: p.landing, cats: p.cats, start_small, start_big, radius }
    };
    [liner, mega, carrier(carrier_p(), Kind::Carrier), carrier(megacarrier_p(), Kind::MegaCarrier)]
});

/// A ship as the physics sees it this step.
#[derive(Clone, Copy, Debug)]
pub struct DeckPose {
    pub kind: Kind,
    pub id: u64,
    pub pos: DVec3,
    pub rot: DQuat,
    pub vel: DVec3,
    /// angular velocity (planet frame)
    pub omega: DVec3,
}

impl DeckPose {
    pub fn local(&self, p: DVec3) -> DVec3 {
        self.rot.inverse() * (p - self.pos)
    }
    pub fn world(&self, l: DVec3) -> DVec3 {
        self.pos + self.rot * l
    }
    /// How fast a point of it moves.
    pub fn vel_at(&self, p: DVec3) -> DVec3 {
        self.vel + self.omega.cross(p - self.pos)
    }
}

/// The ships near the aircraft (world.rs `World::decks`).
#[derive(Default, Clone)]
pub struct Decks(pub Vec<DeckPose>);

/// A point inside a ship: how deep (below what it should rest on), how
/// that surface moves, and whether it is a wall rather than a deck.
#[derive(Clone, Copy, Debug)]
pub struct DeckHit {
    pub pen: f64,
    pub vel: DVec3,
    pub wall: bool,
    pub id: u64,
}

impl Decks {
    pub fn near(&self, p: DVec3, margin: f64) -> bool {
        self.0.iter().any(|d| (p - d.pos).length() < d.kind.geom().radius + margin)
    }
    pub fn probe(&self, p: DVec3) -> Option<DeckHit> {
        for d in &self.0 {
            let g = d.kind.geom();
            if (p - d.pos).length() > g.radius + 5.0 {
                continue;
            }
            let l = d.local(p);
            for (lo, hi) in &g.blocks {
                if l.x > lo.x && l.x < hi.x && l.z > lo.z && l.z < hi.z && l.y > lo.y - 0.5 && l.y < hi.y {
                    let pen = hi.y - l.y;
                    return Some(DeckHit { pen, vel: d.vel_at(p), wall: pen > 1.2, id: d.id });
                }
            }
            let dy = g.deck_y(l.z);
            if l.y < dy && l.y > -g.draft - 2.0 && inside(&g.outline, DVec2::new(l.x, l.z)) {
                let pen = dy - l.y;
                return Some(DeckHit { pen, vel: d.vel_at(p), wall: pen > 2.5, id: d.id });
            }
        }
        None
    }
    /// Something afloat (the speedboat) inside a hull: pushed out over the
    /// nearest side, its speed into the side taken away (returns the new
    /// position and velocity).
    pub fn push_out(&self, p: DVec3, v: DVec3) -> Option<(DVec3, DVec3)> {
        for d in &self.0 {
            let g = d.kind.geom();
            if (p - d.pos).length() > g.radius + 5.0 {
                continue;
            }
            let l = d.local(p);
            let q = DVec2::new(l.x, l.z);
            if l.y > g.deck_y(l.z) || !inside(&g.outline, q) {
                continue;
            }
            let n = g.outline.len();
            let mut best = (f64::MAX, DVec2::ZERO);
            for i in 0..n {
                let (a, b) = (g.outline[i], g.outline[(i + 1) % n]);
                let ab = b - a;
                let t = ((q - a).dot(ab) / ab.length_squared().max(1e-9)).clamp(0.0, 1.0);
                let c = a + ab * t;
                let dd = (q - c).length();
                if dd < best.0 {
                    best = (dd, c);
                }
            }
            let out = (best.1 - q).normalize_or_zero();
            let to = best.1 + out * 1.0;
            let lw = DVec3::new(to.x, l.y, to.y);
            let nw = d.rot * DVec3::new(out.x, 0.0, out.y);
            let rel = v - d.vel_at(p);
            let vn = rel.dot(nw);
            let v2 = if vn < 0.0 { v - nw * vn * 1.4 } else { v };
            return Some((d.world(lw), d.vel_at(p) + (v2 - d.vel_at(p)) * 0.7));
        }
        None
    }
}

// ------------------------------------------------------------------ ships at sea

pub struct Ship {
    pub kind: Kind,
    pub id: u64,
    pub name: String,
    /// where its origin would be on a calm sea (on the sphere)
    base: DVec3,
    /// course (rad from north) and the one it turns to, speed (m/s)
    pub heading: f64,
    want: f64,
    pub speed: f64,
    /// heave (m), pitch (rad, bow up), roll (rad, starboard up), their
    /// rates, and where the sea would put them
    heave: [f64; 3],
    pitch: [f64; 3],
    roll: [f64; 3],
    pub pos: DVec3,
    pub rot: DQuat,
    pub vel: DVec3,
    pub omega: DVec3,
    /// the patch of ocean it was spawned for (None: placed for a start)
    region: Option<(i32, i32)>,
    pub flagship: bool,
    look_t: f64,
    /// (tests) pitch and roll forced as sines: amplitudes (rad) and period (s)
    pub forced: Option<(f64, f64, f64)>,
    t: f64,
}

/// What the sea is under a point: its height above the sphere, and whether it is water.
pub type Sea<'a> = dyn FnMut(DVec3) -> Option<(f64, bool)> + 'a;

impl Ship {
    pub fn new(kind: Kind, id: u64, name: String, lat: f64, lon: f64, heading: f64, speed: f64) -> Self {
        let base = geo::lla_to_pos(lat, lon, 0.0);
        let mut s = Self { kind, id, name, base, heading, want: heading, speed, heave: [0.0; 3], pitch: [0.0; 3], roll: [0.0; 3], pos: base, rot: DQuat::IDENTITY, vel: DVec3::ZERO, omega: DVec3::ZERO, region: None, flagship: false, look_t: 0.0, forced: None, t: 0.0 };
        s.advance(0.0);
        s
    }
    pub fn fwd(&self) -> DVec3 {
        self.rot * DVec3::NEG_Z
    }
    /// The frame of a ship sailing level on its course.
    fn level(&self) -> (DVec3, DVec3, DVec3) {
        let (e, n, u) = geo::enu(self.base);
        let f = n * self.heading.cos() + e * self.heading.sin();
        (f.cross(u).normalize(), u, f)
    }
    pub fn pose(&self) -> DeckPose {
        DeckPose { kind: self.kind, id: self.id, pos: self.pos, rot: self.rot, vel: self.vel, omega: self.omega }
    }
    pub fn lla(&self) -> (f64, f64) {
        let (la, lo, _) = geo::pos_to_lla(self.base);
        (la, lo)
    }

    /// Once a frame: where the waves would put it, and a look ahead for land.
    pub fn frame(&mut self, dt: f64, sea: &mut Sea<'_>) {
        let d = self.kind.dims();
        let (r, u, f) = self.level();
        let base = self.base;
        let at = |x: f64, z: f64| base + r * x - f * z;
        let mut h = |p: DVec3| sea(p).map(|s| s.0).unwrap_or(0.0);
        let hb = h(at(0.0, -0.4 * d.len));
        let hs = h(at(0.0, 0.4 * d.len));
        let hp = h(at(-0.45 * d.beam, 0.0));
        let hst = h(at(0.45 * d.beam, 0.0));
        let hc = h(at(0.0, 0.0));
        let _ = u;
        self.heave[2] = ((hb + hs + hp + hst + 2.0 * hc) / 6.0).clamp(-6.0, 6.0);
        self.pitch[2] = ((hb - hs) / (0.8 * d.len)).atan().clamp(-0.08, 0.08);
        self.roll[2] = ((hst - hp) / (0.9 * d.beam)).atan().clamp(-0.1, 0.1);
        if let Some((ap, ar, per)) = self.forced {
            let w = std::f64::consts::TAU / per;
            self.pitch[2] = ap * (w * self.t).sin();
            self.roll[2] = ar * (w * self.t * 0.83 + 1.0).sin();
        }
        // Land ahead: come about.
        self.look_t -= dt;
        if self.look_t <= 0.0 && (self.want - self.heading).abs() < 1e-3 {
            self.look_t = 5.0;
            let ahead = self.base + f * (0.6 * d.len + 2500.0);
            if let Some((_, false)) = sea(ahead) {
                self.want = self.heading + std::f64::consts::PI;
            }
        }
    }

    /// Move on by `h` seconds.
    pub fn advance(&mut self, h: f64) {
        let d = self.kind.dims();
        self.t += h;
        // Turn toward the course wanted, slowly (a big ship more slowly).
        let rate = (0.012 * (300.0 / d.len).sqrt()).min(0.02);
        let dh = (self.want - self.heading).clamp(-rate * h, rate * h);
        self.heading += dh;
        if (self.want - self.heading).abs() < 1e-4 {
            self.heading = self.heading.rem_euclid(std::f64::consts::TAU);
            self.want = self.heading;
        }
        let (_, _, f) = self.level();
        self.base = (self.base + f * (self.speed * h)).normalize() * geo::R;
        // A heavy, well damped spring toward where the sea would put it.
        let w0 = std::f64::consts::TAU / d.period;
        for s in [&mut self.heave, &mut self.pitch, &mut self.roll] {
            let a = w0 * w0 * (s[2] - s[0]) - 2.0 * 0.45 * w0 * s[1];
            s[1] += a * h;
            s[0] += s[1] * h;
        }
        let (r, u, f) = self.level();
        let rot0 = DQuat::from_mat3(&DMat3::from_cols(r, u, -f));
        self.rot = (rot0 * DQuat::from_rotation_x(self.pitch[0]) * DQuat::from_rotation_z(self.roll[0])).normalize();
        self.pos = self.base + u * self.heave[0];
        self.vel = f * self.speed + u * self.heave[1];
        self.omega = rot0 * DVec3::new(self.pitch[1], dh / h.max(1e-9), self.roll[1]);
    }
}

/// The ships around the player.
#[derive(Default)]
pub struct Fleet {
    pub ships: Vec<Ship>,
    done: HashSet<(i32, i32)>,
    spawn_t: f64,
    next_id: u64,
}

/// The patches of ocean ships are spawned for (degrees).
const REGION: f64 = 0.2;

fn hash(a: i64, b: i64, c: u64) -> u64 {
    let mut x = (a as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (b as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F) ^ c.wrapping_mul(0x1656_67B1_9E37_79F9);
    x ^= x >> 31;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 29;
    x
}
fn unit(x: u64) -> f64 {
    (x >> 11) as f64 / (1u64 << 53) as f64
}

fn ship_name(kind: Kind, k: u64) -> String {
    let pick = |v: &[&str]| v[(k % v.len() as u64) as usize].to_string();
    match kind {
        Kind::Liner => pick(&["RMS Titanic", "RMS Olympic", "HMHS Britannic"]),
        Kind::MegaShip => pick(&["MV Colossus", "MV Leviathan", "MV Goliath", "MV Behemoth"]),
        Kind::Carrier => pick(&["Carrier Valiant", "Carrier Resolute", "Carrier Vigilant", "Carrier Intrepid"]),
        Kind::MegaCarrier => pick(&["Mega carrier Atlas", "Mega carrier Titan", "Mega carrier Olympus"]),
    }
}

impl Fleet {
    fn id(&mut self) -> u64 {
        self.next_id += 1;
        self.next_id
    }
    pub fn by_id(&self, id: u64) -> Option<&Ship> {
        self.ships.iter().find(|s| s.id == id)
    }
    pub fn flagship(&self) -> Option<&Ship> {
        self.ships.iter().find(|s| s.flagship)
    }
    /// A start somewhere else: no ship is kept around for it any more.
    pub fn release_flagship(&mut self) {
        for s in self.ships.iter_mut() {
            s.flagship = false;
        }
    }
    pub fn clear(&mut self) {
        self.ships.clear();
        self.done.clear();
    }

    /// Whether a ship of `kind` fits here heading `hdg`, with open water to
    /// sail on: None while the ground there is not known yet.
    pub fn fits_at(kind: Kind, lat: f64, lon: f64, hdg: f64, wet: &mut impl FnMut(f64, f64) -> Option<bool>) -> Option<bool> {
        Self::fits(kind, lat, lon, hdg, wet)
    }
    fn fits(kind: Kind, lat: f64, lon: f64, hdg: f64, wet: &mut impl FnMut(f64, f64) -> Option<bool>) -> Option<bool> {
        let d = kind.dims();
        let c = geo::lla_to_pos(lat, lon, 0.0);
        let (e, n, _) = geo::enu(c);
        let f = n * hdg.cos() + e * hdg.sin();
        let r = f.cross(c.normalize());
        let mut pts = vec![c];
        for k in [-1.0, 1.0] {
            pts.push(c + f * (k * (0.5 * d.len + 300.0)));
            pts.push(c + r * (k * (0.5 * d.beam + 250.0)));
        }
        pts.push(c + f * (0.5 * d.len + 3000.0));
        let mut all = true;
        for p in pts {
            let (la, lo, _) = geo::pos_to_lla(p);
            match wet(la, lo) {
                None => return None,
                Some(false) => all = false,
                Some(true) => {}
            }
        }
        Some(all)
    }

    /// Once a frame: waves, course, spawning near `at` and letting go of
    /// ships left far behind. `sea` gives the surface (height above the
    /// sphere, water) under a point.
    pub fn update(&mut self, dt: f64, at: DVec3, sea: &mut Sea<'_>) {
        for s in self.ships.iter_mut() {
            s.frame(dt, sea);
        }
        self.spawn_t -= dt;
        if self.spawn_t > 0.0 {
            return;
        }
        self.spawn_t = 1.5;
        let (la, lo, _) = geo::pos_to_lla(at);
        let (la, lo) = (la.to_degrees(), lo.to_degrees());
        let (ci, cj) = ((la / REGION).floor() as i32, (lo / REGION).floor() as i32);
        let mut wet = |a: f64, b: f64| sea(geo::lla_to_pos(a, b, 0.0)).map(|s| s.1);
        for di in -2..=2 {
            for dj in -2..=2 {
                let key = (ci + di, cj + dj);
                if self.done.contains(&key) {
                    continue;
                }
                let hsh = hash(key.0 as i64, key.1 as i64, 77);
                // (some patches are empty)
                if unit(hsh) < 0.2 {
                    self.done.insert(key);
                    continue;
                }
                let kind = Kind::ALL[(hash(key.0 as i64, key.1 as i64, 3) % 4) as usize];
                let lat = (key.0 as f64 + 0.2 + 0.6 * unit(hash(key.0 as i64, key.1 as i64, 5))) * REGION;
                let lon = (key.1 as f64 + 0.2 + 0.6 * unit(hash(key.0 as i64, key.1 as i64, 7))) * REGION;
                let hdg = unit(hash(key.0 as i64, key.1 as i64, 11)) * std::f64::consts::TAU;
                let here = geo::lla_to_pos(lat.to_radians(), lon.to_radians(), 0.0);
                if (here - at).length() > 45_000.0 {
                    continue;
                }
                // (not on top of one already there)
                if self.ships.iter().any(|s| (s.pos - here).length() < 0.5 * (s.kind.dims().len + kind.dims().len) + 1500.0) {
                    self.done.insert(key);
                    continue;
                }
                match Self::fits(kind, lat.to_radians(), lon.to_radians(), hdg, &mut wet) {
                    None => {}
                    Some(false) => {
                        self.done.insert(key);
                    }
                    Some(true) => {
                        self.done.insert(key);
                        let id = self.id();
                        let mut s = Ship::new(kind, id, ship_name(kind, hsh >> 7), lat.to_radians(), lon.to_radians(), hdg, kind.dims().cruise);
                        s.region = Some(key);
                        self.ships.push(s);
                    }
                }
            }
        }
        // Far behind: gone (its patch spawns it again on coming back).
        let mut gone = Vec::new();
        self.ships.retain(|s| {
            let keep = s.flagship || (s.pos - at).length() < 90_000.0;
            if !keep {
                if let Some(r) = s.region {
                    gone.push(r);
                }
            }
            keep
        });
        for r in gone {
            self.done.remove(&r);
        }
        let far: Vec<(i32, i32)> = self.done.iter().copied().filter(|&(i, j)| (i - ci).abs() > 4 || (j - cj).abs() > 4).collect();
        for k in far {
            if !self.ships.iter().any(|s| s.region == Some(k)) {
                self.done.remove(&k);
            }
        }
    }

    pub fn advance(&mut self, h: f64) {
        for s in self.ships.iter_mut() {
            s.advance(h);
        }
    }

    /// Start on a ship: `kind` here, heading `hdg` at flight-operations
    /// speed, the others sailing in company. Returns its id.
    pub fn start_at(&mut self, kind: Kind, lat: f64, lon: f64, hdg: f64, wet: &mut impl FnMut(f64, f64) -> Option<bool>) -> u64 {
        for s in self.ships.iter_mut() {
            s.flagship = false;
        }
        // (ships already there that would be in the way: gone)
        let c = geo::lla_to_pos(lat, lon, 0.0);
        self.ships.retain(|s| (s.pos - c).length() > 9000.0);
        let id = self.id();
        let mut s = Ship::new(kind, id, ship_name(kind, 0), lat, lon, hdg, kind.dims().ops);
        s.flagship = true;
        self.ships.push(s);
        let (e, n, _) = geo::enu(c);
        let f = n * hdg.cos() + e * hdg.sin();
        let r = f.cross(c.normalize());
        let flag = kind.dims();
        let mut side = 1.0;
        let mut ahead = 0.0;
        for (k, other) in Kind::ALL.iter().filter(|k| **k != kind).enumerate() {
            let od = other.dims();
            let off = 0.5 * flag.beam + 0.5 * od.beam + 900.0 + 400.0 * k as f64;
            let p = c + r * (side * off) + f * (ahead - 0.3 * od.len);
            side = -side;
            ahead += 0.5 * flag.len + 600.0;
            let (la, lo, _) = geo::pos_to_lla(p);
            if Self::fits(*other, la, lo, hdg, wet) != Some(false) {
                let id = self.id();
                self.ships.push(Ship::new(*other, id, ship_name(*other, k as u64 + 1), la, lo, hdg, flag.ops.min(od.cruise + 2.0)));
            }
        }
        id
    }

    /// For the maps: lat, lon, heading (degrees), length (m) and name of each.
    pub fn map_list(&self) -> Vec<(f64, f64, f64, f64, String)> {
        self.ships.iter().map(|s| {
            let (la, lo) = s.lla();
            (la.to_degrees(), lo.to_degrees(), s.heading.to_degrees(), s.kind.dims().len, s.name.clone())
        }).collect()
    }

    /// Their poses, for the ones within `r` of `p`.
    pub fn decks_near(&self, p: DVec3, r: f64) -> Decks {
        Decks(self.ships.iter().filter(|s| (s.pos - p).length() < r + s.kind.geom().radius).map(|s| s.pose()).collect())
    }

    /// The instanced solids: hulls, and the aircraft parked on the carriers.
    pub fn instances(&self, cam: DVec3, out: &mut Vec<(u32, Inst)>) {
        let inst = |m: glam::DMat4, tint: [f32; 4]| {
            let (x, y, z, o) = (m.x_axis, m.y_axis, m.z_axis, m.w_axis - cam.extend(0.0));
            let r = |i: usize| [x[i] as f32, y[i] as f32, z[i] as f32, o[i] as f32];
            Inst { m: [r(0), r(1), r(2)], tint, ext: [0.0; 4] }
        };
        for s in &self.ships {
            let d = (s.pos - cam).length();
            if d > 70_000.0 {
                continue;
            }
            let base = glam::DMat4::from_rotation_translation(s.rot, s.pos);
            out.push((MESH + s.kind.idx() as u32, inst(base, [0.6, 0.6, 0.6, 0.0])));
            if d > 12_000.0 {
                continue;
            }
            let dy = s.kind.dims().deck;
            for (x, z, yaw, which) in parked(s.kind) {
                let m = base * glam::DMat4::from_rotation_translation(DQuat::from_rotation_y(yaw), DVec3::new(x, dy, z));
                if which == 0 {
                    out.push((M_FIGHTER, inst(m, [0.36, 0.38, 0.41, 0.0])));
                } else {
                    let w = crate::airmodels::M_BODY + 3 * 4;
                    out.push((w, inst(m, [0.92, 0.92, 0.93, 0.0])));
                    out.push((w + 1, inst(m, [0.1, 0.25, 0.55, 0.0])));
                    out.push((w + 2, inst(m, [0.5, 0.5, 0.5, 0.0])));
                }
            }
        }
    }

    /// Navigation lights after dark, and the carriers' deck lighting.
    pub fn lights(&self, cam: DVec3, out: &mut Vec<crate::airfield::Light>) {
        use crate::airfield::{Light, L_GREEN, L_RED, L_WHITE};
        for s in &self.ships {
            if (s.pos - cam).length() > 25_000.0 {
                continue;
            }
            let d = s.kind.dims();
            let g = s.kind.geom();
            let mut l = |p: DVec3, inten: f32, col: [u8; 3], day: u8| {
                if out.len() < crate::render::MAX_DYN_LIGHTS {
                    let w = (s.pos + s.rot * p - cam).as_vec3();
                    out.push(Light { pos: w.into(), inten, dir: [0.0; 3], spread: 0.0, col: [col[0], col[1], col[2], 0], fx: [0, 0, 255, day] });
                }
            };
            let k = (d.len / 300.0).sqrt() as f32;
            let top = match s.kind {
                Kind::Liner => 62.0,
                Kind::MegaShip => 108.0,
                Kind::Carrier => d.deck + 40.0,
                Kind::MegaCarrier => d.deck + 120.0,
            };
            l(DVec3::new(0.0, top, -0.35 * d.len), 60.0 * k, L_WHITE, 0);
            l(DVec3::new(0.0, d.deck + 2.0, 0.5 * d.len - 2.0), 40.0 * k, L_WHITE, 0);
            l(DVec3::new(-0.5 * d.beam, d.deck + 3.0, -0.15 * d.len), 40.0 * k, L_RED, 0);
            l(DVec3::new(0.5 * d.beam, d.deck + 3.0, -0.15 * d.len), 40.0 * k, L_GREEN, 0);
            if let Some(ln) = &g.landing {
                // The landing area's centre line and edges, and the mirror's
                // amber ball and green datum bars, always on.
                let n = DVec2::new(-ln.dir.y, ln.dir.x);
                let mut sd = 0.0;
                while sd < ln.len {
                    let c = ln.ramp + ln.dir * sd;
                    l(DVec3::new(c.x, d.deck + 0.1, c.y), 6.0, L_WHITE, 0);
                    for e in [-1.0, 1.0] {
                        let q = c + n * (e * ln.width * 0.5);
                        l(DVec3::new(q.x, d.deck + 0.1, q.y), 5.0, [255, 190, 80], 0);
                    }
                    sd += 15.0;
                }
                let mp = ln.ramp + ln.dir * 95.0 - n * (ln.width * 0.5 + 8.0);
                l(DVec3::new(mp.x, d.deck + 1.6, mp.y), 30.0, [255, 150, 30], 1);
                for k in [-1.0, 1.0] {
                    for j in 0..3 {
                        let q = mp + DVec2::new(0.0, k * (1.5 + 1.2 * j as f64));
                        l(DVec3::new(q.x, d.deck + 1.6, q.y), 15.0, L_GREEN, 1);
                    }
                }
            }
            if s.kind == Kind::MegaCarrier {
                let mut z = 1760.0;
                while z > -1760.0 {
                    l(DVec3::new(-80.0, d.deck + 0.2, z), 20.0, L_WHITE, 0);
                    l(DVec3::new(-20.0, d.deck + 0.2, z), 20.0, L_WHITE, 0);
                    z -= 120.0;
                }
            }
        }
    }

    /// The wakes for the shaders (common.wgsl `Globals.ships`): the nearest
    /// ones under way, three rows each; returns how many.
    pub fn wake_rows(&self, cam: DVec3, rows: &mut [[f32; 4]]) -> usize {
        let mut near: Vec<&Ship> = self.ships.iter().filter(|s| s.speed > 0.5 && (s.pos - cam).length() < 20_000.0 + s.kind.dims().wake).collect();
        near.sort_by(|a, b| (a.pos - cam).length().partial_cmp(&(b.pos - cam).length()).unwrap());
        let n = near.len().min(rows.len() / 3);
        for (i, s) in near.iter().take(n).enumerate() {
            let d = s.kind.dims();
            let (_, _, f) = s.level();
            let stern = s.base - f * (0.48 * d.len);
            let p = (stern - cam).as_vec3();
            rows[3 * i] = [p.x, p.y, p.z, d.beam as f32];
            rows[3 * i + 1] = [f.x as f32, f.y as f32, f.z as f32, s.speed as f32];
            rows[3 * i + 2] = [d.len as f32, d.wake as f32, 0.0, 0.0];
        }
        n
    }
}

// ------------------------------------------------------------------ flight operations

/// The carrier's catapult and arresting gear, and the tail hook.
#[derive(Default)]
pub struct Ops {
    /// tail hook down
    pub hook: bool,
    cat: Option<Cat>,
    trap: Option<Trap>,
    /// how long full power has been held on the catapult
    held: f64,
    /// where the hook was along the landing area last step (m from the ramp)
    last_s: Option<f64>,
    /// something to tell the pilot
    pub note: Option<&'static str>,
    /// standing on a catapult, ready (for the hint)
    pub on_cat: bool,
}

struct Cat {
    ship: u64,
    x: f64,
    z_end: f64,
    accel: f64,
}

struct Trap {
    ship: u64,
    s0: f64,
    stop: Option<f64>,
}

/// Where the hook is on an aircraft (its body frame): under the tail.
pub fn hook_point(spec: &crate::aircraft::Spec) -> DVec3 {
    let low = spec.gear.iter().map(|l| l.at.y).fold(0.0f64, f64::min);
    DVec3::new(0.0, low + 0.15 * spec.size_k as f64, spec.hard[3].z * 0.8)
}

impl Ops {
    pub fn trapped(&self) -> bool {
        self.trap.is_some()
    }
    pub fn launching(&self) -> bool {
        self.cat.is_some()
    }
    pub fn reset(&mut self) {
        *self = Ops::default();
    }

    /// The force (planet frame, N) the deck gear puts on the aircraft this
    /// step. `full` while the pilot holds full power for the catapult.
    pub fn step(&mut self, h: f64, plane: &crate::plane::Plane, decks: &Decks, full: bool) -> DVec3 {
        let spec = plane.spec;
        let m = spec.mass;
        self.on_cat = false;
        let carrier = |id: u64| decks.0.iter().find(|d| d.id == id);
        // On the catapult's stroke.
        if let Some(c) = &self.cat {
            let Some(d) = carrier(c.ship) else {
                self.cat = None;
                return DVec3::ZERO;
            };
            let l = d.local(plane.pos);
            let nose = spec.gear.first().map(|g| g.at.z).unwrap_or(-3.0);
            if l.z + nose < c.z_end || !plane.on_ground && l.z < c.z_end + 30.0 {
                self.cat = None;
                self.note = Some("launch");
                return DVec3::ZERO;
            }
            let fwd = d.rot * DVec3::NEG_Z;
            let right = d.rot * DVec3::X;
            let rel = plane.vel - d.vel_at(plane.pos);
            let side = -right * (m * (6.0 * (l.x - c.x) + 4.0 * rel.dot(right)));
            return fwd * (m * c.accel) + side;
        }
        // Caught by a wire: run out to a stop, and held there.
        if let Some(t) = &mut self.trap {
            let Some(d) = carrier(t.ship) else {
                self.trap = None;
                return DVec3::ZERO;
            };
            let Some(ln) = &d.kind.geom().landing else {
                self.trap = None;
                return DVec3::ZERO;
            };
            if !self.hook {
                self.trap = None;
                return DVec3::ZERO;
            }
            let l = d.local(plane.pos);
            let s = (DVec2::new(l.x, l.z) - ln.ramp).dot(ln.dir);
            let axis = d.rot * DVec3::new(ln.dir.x, 0.0, ln.dir.y);
            let v = (plane.vel - d.vel_at(plane.pos)).dot(axis);
            let run = s - t.s0;
            if t.stop.is_none() && v > 0.4 {
                // What stops it in the run out left, and enough over that to
                // hold back the engines at full power (pilots go to full
                // power as they touch down, in case the hook skips).
                let left = (ln.runout - run).max(2.0);
                let a = (v * v / (2.0 * left)).min(5.0 * 9.81) + 0.8 * 9.81;
                return -axis * (m * a);
            }
            let stop = *t.stop.get_or_insert(s);
            // (the wire pulls back only)
            let f = (m * (3.0 * (s - stop) + 4.0 * v)).max(0.0).min(m * 3.0 * 9.81);
            return -axis * f;
        }
        // On a carrier's deck: on a catapult, or crossing the wires.
        // (the carrier it is over, or nearest)
        let mut on_deck = None;
        let mut best = f64::MAX;
        for d in &decks.0 {
            let g = d.kind.geom();
            let dist = (plane.pos - d.pos).length();
            if !d.kind.is_carrier() || dist > g.radius + 200.0 {
                continue;
            }
            let l = d.local(plane.pos);
            let over = if inside(&g.outline, DVec2::new(l.x, l.z)) { 0.0 } else { dist };
            if over < best {
                best = over;
                on_deck = Some(*d);
            }
        }
        let Some(d) = on_deck else {
            self.held = 0.0;
            self.last_s = None;
            return DVec3::ZERO;
        };
        let g = d.kind.geom();
        let l = d.local(plane.pos);
        let dy = d.kind.dims().deck;
        let nose = spec.gear.first().map(|g| g.at.z).unwrap_or(-3.0);
        if plane.on_ground {
            for &(x, z0, z1) in &g.cats {
                let nz = l.z + nose;
                if (l.x - x).abs() < 3.0 && (nz - z0).abs() < 8.0 && l.y < dy + 6.0 {
                    let rel = plane.vel - d.vel_at(plane.pos);
                    if rel.length() < 3.0 {
                        self.on_cat = true;
                        // (full power first: then the shot)
                        self.held = if full && plane.throttle > 0.9 { self.held + h } else { 0.0 };
                        if self.held > 0.6 {
                            // Fire: up to flying speed over the stroke.
                            let fwd = d.rot * DVec3::NEG_Z;
                            let over_deck = (d.vel - plane.wind).dot(fwd).max(0.0);
                            let need = (spec.approach.0 * 1.12 - over_deck).clamp(12.0, 95.0);
                            let accel = need * need / (2.0 * (z0 - z1).abs());
                            self.cat = Some(Cat { ship: d.id, x, z_end: z1, accel });
                            self.held = 0.0;
                            self.note = Some("cat");
                        }
                    }
                }
            }
        }
        if let (true, Some(ln)) = (self.hook, &g.landing) {
            let hp = plane.pos + plane.rot * hook_point(spec);
            let hl = d.local(hp);
            let q = DVec2::new(hl.x, hl.z) - ln.ramp;
            let s = q.dot(ln.dir);
            let lat = q.perp_dot(ln.dir).abs();
            let axis = d.rot * DVec3::new(ln.dir.x, 0.0, ln.dir.y);
            let v = (plane.vel - d.vel_at(plane.pos)).dot(axis);
            if let Some(s_prev) = self.last_s {
                if lat < ln.width * 0.5 + 1.0 && hl.y < dy + 1.5 && v > 8.0 && ln.wires.iter().any(|&w| s_prev < w && s >= w) {
                    self.trap = Some(Trap { ship: d.id, s0: s, stop: None });
                    self.note = Some("trap");
                }
            }
            self.last_s = Some(s);
        } else {
            self.last_s = None;
        }
        DVec3::ZERO
    }
}

/// Where an aircraft starts on a ship of `kind`: its origin and attitude in
/// the ship's frame (wheels on the deck, facing the bow), or None if it
/// cannot start on that deck.
pub fn start_spot(kind: Kind, spec: &crate::aircraft::Spec) -> Option<(DVec3, DQuat)> {
    let g = kind.geom();
    let d = kind.dims();
    let sag = -spec.gear.iter().map(|l| l.at.y).fold(0.0f64, f64::min);
    let nose = spec.gear.first().map(|l| l.at.z).unwrap_or(-2.0);
    match kind {
        Kind::Carrier => {
            let s = g.start_small?;
            Some((DVec3::new(s.x, d.deck + sag, s.y - nose), DQuat::IDENTITY))
        }
        Kind::MegaCarrier => {
            let s = g.start_big?;
            // (the tail clear of the stern)
            let tail = spec.hard[3].z.max(10.0);
            let z = (1800.0 - 25.0 - tail).min(s.y);
            Some((DVec3::new(s.x, d.deck + sag, z), DQuat::IDENTITY))
        }
        _ => None,
    }
}

/// Whether `spec` belongs on the mega carrier rather than the carrier (a 747).
pub fn too_big_for_carrier(spec: &crate::aircraft::Spec) -> bool {
    spec.b > 30.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plane::{Controls, Plane};
    use crate::world::World;
    use std::sync::Arc;

    fn empty_world() -> World {
        let dir = std::env::temp_dir().join("fablegame-test-cache");
        let src = Arc::new(crate::sources::Sources::new(crate::fetch::Fetcher::new(dir, true), String::new()));
        World::new(crate::sources::Loader::new(src, 0), 14)
    }

    #[test]
    fn meshes_hold_together() {
        for (v, i) in meshes() {
            assert!(!v.is_empty() && i.len() % 3 == 0);
            assert!(i.iter().all(|&k| (k as usize) < v.len()));
            assert!(v.iter().all(|x| x.pos.iter().all(|c| c.is_finite()) && x.nrm.iter().all(|c| c.is_finite())));
        }
        for k in Kind::ALL {
            let g = k.geom();
            assert!(g.outline.len() >= 4);
            assert!(inside(&g.outline, DVec2::ZERO), "{k:?}");
            assert_eq!(Kind::parse(match k { Kind::Liner => "titanic", Kind::MegaShip => "megaship", Kind::Carrier => "carrier", Kind::MegaCarrier => "megacarrier" }), Some(k));
        }
        // The flight deck's triangles cover its area.
        let p = carrier_p();
        let tris = triangulate(&p.outline);
        let area: f64 = tris.iter().map(|t| 0.5 * (p.outline[t[1]] - p.outline[t[0]]).perp_dot(p.outline[t[2]] - p.outline[t[0]]).abs()).sum();
        let n = p.outline.len();
        let poly: f64 = 0.5 * (0..n).map(|i| p.outline[i].perp_dot(p.outline[(i + 1) % n])).sum::<f64>().abs();
        assert!((area - poly).abs() < 1.0, "{area} vs {poly}");
    }

    /// How far a Hornet parked with its brakes on drifts on a carrier under
    /// way in 40 s, the deck pitching and rolling as `forced` says (None:
    /// steady); and the share of the time it was on its wheels.
    fn parked_drift(forced: Option<(f64, f64, f64)>) -> (f64, f64) {
        let spec: &'static crate::aircraft::Spec = &crate::aircraft::fa18::SPEC;
        let mut w = empty_world();
        let mut fleet = Fleet::default();
        let id = fleet.start_at(Kind::Carrier, 0.6, -2.1, 0.4, &mut |_, _| Some(true));
        fleet.ships.retain(|s| s.id == id);
        fleet.ships[0].forced = forced;
        let (spot, q) = start_spot(Kind::Carrier, spec).unwrap();
        // Parked off the catapult, on the deck's flat amidships.
        let spot = spot + DVec3::new(-12.0, 0.0, 60.0);
        let s = &fleet.ships[0];
        let mut p = Plane::spawn_ground(spec, 0.6, -2.1, 0.0, 0.4, 0.0);
        p.pos = s.pos + s.rot * spot;
        p.rot = s.rot * q;
        p.vel = s.vel;
        p.hold = false;
        p.t = 2.0;
        let c = Controls { brake: true, ..Default::default() };
        let dt = 1.0 / 180.0;
        let mut worst = 0.0f64;
        let mut ground = 0;
        let steps = (40.0 / dt) as usize;
        for k in 0..steps {
            fleet.update(dt, p.pos, &mut |_| Some((0.0, true)));
            fleet.advance(dt);
            w.decks = fleet.decks_near(p.pos, 500.0);
            assert!(p.step(dt, &c, &w).is_none(), "crashed at {:.1} s", k as f64 * dt);
            let s = fleet.by_id(id).unwrap();
            let l = s.rot.inverse() * (p.pos - s.pos);
            if k % 1800 == 0 {
                eprintln!("  t {:.0}: off the spot by {:.2?}, pitch {:.2}°, roll {:.2}°", k as f64 * dt, l - spot, s.pitch[0].to_degrees(), s.roll[0].to_degrees());
            }
            if k as f64 * dt > 3.0 {
                worst = worst.max((l - spot).length());
                ground += p.on_ground as usize;
            }
        }
        (worst, ground as f64 / (steps as f64 - 3.0 / dt))
    }

    /// Parked with its brakes on, on a carrier under way pitching ±2.5° and
    /// rolling ±4° every 9 s (a heavy sea for a carrier): it stays where it
    /// was put, on its wheels, no further than on a steady deck (where the
    /// engines' idle thrust creeps it along against the brakes).
    #[test]
    fn parked_aircraft_stays_on_a_pitching_deck() {
        let (calm, _) = parked_drift(None);
        let (rough, wheels) = parked_drift(Some((2.5f64.to_radians(), 4f64.to_radians(), 9.0)));
        eprintln!("parked 40 s: drift {calm:.2} m on a steady deck, {rough:.2} m pitching and rolling, on its wheels {:.0}% of the time", wheels * 100.0);
        assert!(rough < 3.0 && rough - calm < 0.6, "slid {rough} m ({calm} steady)");
        assert!(wheels > 0.98);
    }

    /// The 747s off the mega carrier steaming into the wind: full power
    /// from the stern, rotate at 150 kt through the air, and off the deck
    /// with room to spare; the ×5 too.
    #[test]
    fn a_747_takes_off_from_the_mega_carrier() {
        for spec in [&crate::aircraft::b747::SPEC, &*crate::aircraft::b747big::X5] {
            let spec: &'static crate::aircraft::Spec = spec;
            let mut w = empty_world();
            let mut fleet = Fleet::default();
            let id = fleet.start_at(Kind::MegaCarrier, 0.6, -2.1, 0.4, &mut |_, _| Some(true));
            fleet.ships.retain(|s| s.id == id);
            let (spot, q) = start_spot(Kind::MegaCarrier, spec).unwrap();
            let s = &fleet.ships[0];
            let mut p = Plane::spawn_ground(spec, 0.6, -2.1, 0.0, 0.4, 0.0);
            p.pos = s.pos + s.rot * spot;
            p.rot = s.rot * q;
            p.vel = s.vel;
            p.hold = false;
            p.t = 2.0;
            let dt = 1.0 / 180.0;
            let mut off = None;
            for k in 0..(120.0 / dt) as usize {
                fleet.advance(dt);
                w.decks = fleet.decks_near(p.pos, 4000.0);
                let air = p.airspeed().length();
                let s = &fleet.ships[0];
                let up = p.pos.normalize();
                let pitch = p.forward().dot(up).asin();
                let c = Controls { throttle: 1.0, pitch: if air > 77.0 { ((0.17 - pitch) * 4.0).clamp(-1.0, 1.0) } else { 0.0 }, ..Default::default() };
                assert!(p.step(dt, &c, &w).is_none(), "crashed at {:.1} s", k as f64 * dt);
                let l = s.rot.inverse() * (p.pos - s.pos);
                if off.is_none() && !p.on_ground && l.y > Kind::MegaCarrier.dims().deck + 15.0 * spec.size_k as f64 {
                    off = Some((l.z, k as f64 * dt, air));
                }
                if off.is_some() {
                    break;
                }
            }
            let (z, t, v) = off.expect("never left the deck");
            let used = spot.z - z;
            eprintln!("{}: off the deck after {used:.0} m ({:.0} m of deck left) in {t:.0} s at {:.0} kt", spec.name, z + 1800.0, v * 1.94384);
            assert!(z > -1800.0, "ran off the bow");
        }
    }

    /// A Hornet rolling onto the carrier's deck at landing speed, hook down:
    /// the wire takes it from 130 kt to a stop in under 120 m, at a few g.
    #[test]
    fn arrested_landing_decelerates() {
        let spec: &'static crate::aircraft::Spec = &crate::aircraft::fa18::SPEC;
        let mut w = empty_world();
        let mut fleet = Fleet::default();
        let id = fleet.start_at(Kind::Carrier, 0.6, -2.1, 0.4, &mut |_, _| Some(true));
        fleet.ships.retain(|s| s.id == id);
        fleet.advance(0.0);
        let ln = Kind::Carrier.geom().landing.clone().unwrap();
        let sag = -spec.gear.iter().map(|l| l.at.y).fold(0.0f64, f64::min);
        // On the deck just short of the first wire, lined up with the landing area.
        let c0 = ln.ramp + ln.dir * 30.0;
        let yaw = (-ln.dir.x).atan2(-ln.dir.y);
        let s = &fleet.ships[0];
        let q = DQuat::from_rotation_y(yaw);
        let mut p = Plane::spawn_ground(spec, 0.6, -2.1, 0.0, 0.4, 0.0);
        p.rot = s.rot * q;
        p.pos = s.pos + s.rot * DVec3::new(c0.x, Kind::Carrier.dims().deck + sag, c0.y);
        let axis = s.rot * DVec3::new(ln.dir.x, 0.0, ln.dir.y);
        p.vel = s.vel + axis * (130.0 / 1.94384);
        p.hold = false;
        p.t = 2.0;
        p.throttle = 1.0;
        let mut ops = Ops { hook: true, ..Default::default() };
        let c = Controls { throttle: 1.0, ..Default::default() };
        let dt = 1.0 / 180.0;
        let (mut peak, mut caught_at, mut stop_at) = (0.0f64, None, None);
        let mut prev_v = 130.0 / 1.94384;
        for k in 0..(8.0 / dt) as usize {
            fleet.advance(dt);
            w.decks = fleet.decks_near(p.pos, 500.0);
            p.push = ops.step(dt, &p, &w.decks, false);
            assert!(p.step(dt, &c, &w).is_none(), "crashed");
            let s = &fleet.ships[0];
            let l = s.rot.inverse() * (p.pos - s.pos);
            let along = (DVec2::new(l.x, l.z) - ln.ramp).dot(ln.dir);
            let v = (p.vel - s.vel).dot(s.rot * DVec3::new(ln.dir.x, 0.0, ln.dir.y));
            if ops.trapped() && caught_at.is_none() {
                caught_at = Some(along);
            }
            if caught_at.is_some() {
                peak = peak.max((prev_v - v) / dt / 9.81);
                if v < 0.5 && stop_at.is_none() {
                    stop_at = Some((along, k as f64 * dt));
                }
            }
            prev_v = v;
            if k % 90 == 0 {
                eprintln!("  t {:.1}: {along:.1} m from the ramp, {v:.1} m/s, height {:.2}, trapped {}, ground {}", k as f64 * dt, l.y - Kind::Carrier.dims().deck, ops.trapped(), p.on_ground);
            }
        }
        let (caught, (stop, t)) = (caught_at.expect("never caught a wire"), stop_at.expect("never stopped"));
        eprintln!("caught at {caught:.1} m from the ramp, stopped at {stop:.1} m ({:.1} m run out) after {t:.2} s, peak {peak:.1} g", stop - caught);
        assert!(stop - caught < 120.0 && stop - caught > 40.0);
        assert!(peak > 2.0 && peak < 7.0, "peak {peak} g");
        assert!(stop < ln.len, "off the end of the deck");
    }
}
