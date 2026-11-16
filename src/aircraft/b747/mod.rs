//! A Boeing 747-400 with Pratt & Whitney PW4056 engines, its exterior built
//! to the dimensions in Boeing's airport planning document (D6-58326-1 rev F):
//! 70.67 m long, 64.44 m span, 19.4 m to the top of the fin, a 6.50 m wide
//! fuselage with the upper deck in its hump. The fuselage, wing, nacelle and
//! tail outlines were traced from its general-arrangement drawings; the gear
//! sits where its wheelbase (25.60 m), wing gear track (11.00 m) and body gear
//! track (3.84 m) put it, the engines at 11.68 m and 21.0 m out.
//!
//! The flight deck is in `deck.rs`.
//!
//! Body frame: the origin is the centre of gravity, 31.6 m behind the nose
//! (about 20% of the mean aerodynamic chord) and 5.0 m above the ground with
//! the aircraft at rest on its gear. Drawing positions are given as `d`, the
//! distance back from the nose, and `h`, the height above the ground.
use crate::aircraft::{Engine, Fan, Handling, Hole, Hull, Leg, Part, Sound, Spec, Tail};
use crate::airframe::*;
use glam::{DVec3, Quat, Vec2, Vec3};
use std::f32::consts::{FRAC_PI_2, PI, TAU};

pub mod deck;

/// Nose to the centre of gravity, and the ground to it (m).
pub const ZN: f32 = 31.6;
pub const YG: f32 = 5.0;
/// Body z and y from drawing distance-from-nose and height-above-ground.
const fn bz(d: f32) -> f32 {
    d - ZN
}
const fn by(h: f32) -> f32 {
    h - YG
}

// ---------------------------------------------------------------- fuselage

/// Fuselage sections traced from the side, plan and front views: distance
/// from the nose, half width, top, height of the widest point, bottom
/// (heights above the ground), squareness of the upper and the lower half.
/// The upper half carries the upper deck's hump forward of 30 m; the lower
/// half squares off around the wing root under the belly fairing.
#[rustfmt::skip]
const FUSE: [[f32; 7]; 62] = [
    [0.00, 0.06, 5.36, 5.30, 5.24, 2.0, 2.0],
    [0.10, 0.33, 5.62, 5.30, 4.98, 2.0, 2.0],
    [0.25, 0.58, 5.96, 5.30, 4.71, 2.0, 2.0],
    [0.50, 0.86, 6.14, 5.30, 4.48, 2.0, 2.0],
    [0.75, 1.07, 6.33, 5.30, 4.33, 2.0, 2.0],
    [1.00, 1.22, 6.49, 5.30, 4.14, 2.0, 2.0],
    [1.50, 1.40, 6.77, 5.30, 3.92, 1.95, 2.0],
    [2.00, 1.54, 7.02, 5.30, 3.70, 1.9, 2.0],
    [2.50, 1.67, 7.30, 5.30, 3.51, 1.85, 2.0],
    [3.00, 1.81, 7.55, 5.30, 3.32, 1.8, 2.0],
    [3.50, 1.93, 7.80, 5.30, 3.17, 1.75, 2.0],
    [4.00, 2.03, 8.08, 5.30, 3.04, 1.7, 2.0],
    [5.00, 2.24, 8.86, 5.30, 2.79, 1.62, 2.0],
    [5.50, 2.37, 9.17, 5.30, 2.68, 1.62, 1.98],
    [6.00, 2.47, 9.43, 5.35, 2.60, 1.65, 1.96],
    [7.00, 2.61, 9.65, 5.45, 2.45, 1.72, 1.94],
    [8.00, 2.79, 9.80, 5.55, 2.32, 1.78, 1.92],
    [9.00, 2.91, 9.93, 5.62, 2.23, 1.82, 1.90],
    [10.0, 3.00, 9.99, 5.66, 2.14, 1.85, 1.88],
    [11.0, 3.09, 10.05, 5.70, 2.07, 1.88, 1.86],
    [12.0, 3.18, 10.05, 5.70, 2.01, 1.9, 1.85],
    [13.0, 3.22, 10.05, 5.70, 1.98, 1.9, 1.85],
    [14.0, 3.25, 10.05, 5.70, 1.95, 1.9, 1.85],
    [17.0, 3.25, 10.05, 5.70, 1.95, 1.9, 1.85],
    [20.0, 3.25, 10.05, 5.70, 1.92, 1.9, 2.6],
    [22.0, 3.25, 10.02, 5.70, 1.90, 1.9, 3.1],
    [23.0, 3.25, 9.96, 5.70, 1.89, 1.92, 3.2],
    [24.0, 3.25, 9.90, 5.70, 1.88, 1.94, 3.2],
    [25.0, 3.25, 9.83, 5.70, 1.88, 1.96, 3.2],
    [26.0, 3.25, 9.74, 5.70, 1.88, 1.98, 3.2],
    [27.0, 3.25, 9.62, 5.70, 1.88, 2.0, 3.2],
    [28.0, 3.25, 9.49, 5.70, 1.88, 2.03, 3.2],
    [29.0, 3.25, 9.34, 5.70, 1.88, 2.06, 3.2],
    [30.0, 3.25, 9.24, 5.70, 1.88, 2.08, 3.2],
    [31.0, 3.25, 9.21, 5.70, 1.88, 2.1, 3.2],
    [36.0, 3.25, 9.21, 5.70, 1.92, 2.1, 3.1],
    [38.0, 3.25, 9.21, 5.70, 2.00, 2.1, 2.6],
    [40.0, 3.25, 9.21, 5.70, 2.12, 2.1, 2.0],
    [42.0, 3.25, 9.21, 5.72, 2.16, 2.1, 1.85],
    [44.0, 3.25, 9.21, 5.75, 2.18, 2.1, 1.85],
    [46.0, 3.24, 9.21, 5.82, 2.25, 2.1, 1.85],
    [48.0, 3.19, 9.21, 5.98, 2.53, 2.1, 1.88],
    [50.0, 3.11, 9.21, 6.19, 2.97, 2.1, 1.9],
    [52.0, 3.01, 9.20, 6.40, 3.41, 2.1, 1.95],
    [54.0, 2.82, 9.20, 6.63, 3.85, 2.1, 2.0],
    [56.0, 2.57, 9.19, 6.86, 4.29, 2.1, 2.0],
    [58.0, 2.23, 9.17, 7.09, 4.69, 2.1, 2.0],
    [60.0, 1.84, 9.15, 7.34, 5.13, 2.1, 2.0],
    [62.0, 1.50, 9.12, 7.58, 5.57, 2.1, 2.0],
    [64.0, 1.14, 9.08, 7.83, 6.01, 2.1, 2.0],
    [66.0, 0.79, 9.03, 8.08, 6.44, 2.1, 2.0],
    [67.0, 0.62, 8.98, 8.16, 6.66, 2.1, 2.0],
    [68.0, 0.45, 8.92, 8.25, 6.88, 2.0, 2.0],
    [68.6, 0.32, 8.86, 8.35, 7.40, 2.0, 2.0],
    [69.0, 0.24, 8.85, 8.48, 7.95, 2.0, 2.0],
    [69.2, 0.20, 8.85, 8.55, 8.15, 2.0, 2.0],
    [69.4, 0.15, 8.84, 8.63, 8.35, 2.0, 2.0],
    [69.55, 0.11, 8.84, 8.70, 8.50, 2.0, 2.0],
    [69.7, 0.08, 8.83, 8.75, 8.62, 2.0, 2.0],
    [69.8, 0.05, 8.82, 8.78, 8.71, 2.0, 2.0],
    [69.86, 0.03, 8.82, 8.79, 8.76, 2.0, 2.0],
    [69.9, 0.01, 8.82, 8.80, 8.79, 2.0, 2.0],
];
/// Fuselage from the nose tip to the APU exhaust (drawing distances).
const D_NOSE: f32 = 0.0;
const D_TAIL: f32 = 69.9;
/// Where the radome, the flight deck section and the tail cone join the rest.
const D_RADOME: f32 = 2.2;
const D_DECK: f32 = 8.4;
const D_CONE: f32 = 53.0;

#[derive(Clone, Copy, Debug)]
pub struct Sec {
    pub w: f32,
    pub top: f32,
    pub mid: f32,
    pub bot: f32,
    pub nt: f32,
    pub nb: f32,
}

/// Monotone cubic through the table (no overshoot between stations).
fn pchip(col: usize, d: f32) -> f32 {
    let k = &FUSE;
    let n = k.len();
    if d <= k[0][0] {
        return k[0][col];
    }
    if d >= k[n - 1][0] {
        return k[n - 1][col];
    }
    let i = (0..n - 1).find(|&i| d < k[i + 1][0]).unwrap_or(n - 2);
    let h = |i: usize| k[i + 1][0] - k[i][0];
    let dd = |i: usize| (k[i + 1][col] - k[i][col]) / h(i);
    let slope = |i: usize| -> f32 {
        if i == 0 || i == n - 1 {
            return if i == 0 { dd(0) } else { dd(n - 2) };
        }
        let (a, b) = (dd(i - 1), dd(i));
        if a * b <= 0.0 {
            0.0
        } else {
            let (w1, w2) = (2.0 * h(i) + h(i - 1), h(i) + 2.0 * h(i - 1));
            (w1 + w2) / (w1 / a + w2 / b)
        }
    };
    let t = (d - k[i][0]) / h(i);
    let (y0, y1, m0, m1) = (k[i][col], k[i + 1][col], slope(i) * h(i), slope(i + 1) * h(i));
    let (t2, t3) = (t * t, t * t * t);
    (2.0 * t3 - 3.0 * t2 + 1.0) * y0 + (t3 - 2.0 * t2 + t) * m0 + (-2.0 * t3 + 3.0 * t2) * y1 + (t3 - t2) * m1
}
/// The section at body station z (body frame heights).
pub fn sec(z: f32) -> Sec {
    let d = z + ZN;
    Sec { w: pchip(1, d), top: by(pchip(2, d)), mid: by(pchip(3, d)), bot: by(pchip(4, d)), nt: pchip(5, d), nb: pchip(6, d) }
}
/// Point on a section at angle th (0 = top, pi/2 = right side, pi = bottom).
pub fn sec_pt(s: &Sec, th: f32) -> Vec2 {
    let (sn, cs) = th.sin_cos();
    let n = if cs > 0.0 { s.nt } else { s.nb };
    let e = |v: f32| v.signum() * v.abs().powf(2.0 / n);
    let h = if cs > 0.0 { s.top - s.mid } else { s.mid - s.bot };
    Vec2::new(s.w * e(sn), s.mid + h * e(cs))
}
pub fn fuse_pt(z: f32, th: f32) -> Vec3 {
    let p = sec_pt(&sec(z), th);
    Vec3::new(p.x, p.y, z)
}
/// Outward normal of the fuselage skin at (z, th).
pub fn fuse_n(z: f32, th: f32) -> Vec3 {
    let e = 1e-3;
    let du = fuse_pt(z + e, th) - fuse_pt(z - e, th);
    let dv = fuse_pt(z, th + e) - fuse_pt(z, th - e);
    // (Down the right side as th grows, aft as z grows: their cross product points out.)
    du.cross(dv).normalize()
}
/// Angle on the section at z whose point is at height y on the right side.
fn fuse_th(z: f32, y: f32) -> f32 {
    let s = sec(z);
    let (mut a, mut b) = (0.0f32, PI);
    for _ in 0..30 {
        let m = (a + b) * 0.5;
        if sec_pt(&s, m).y > y { a = m } else { b = m }
    }
    (a + b) * 0.5
}
/// Point and normal on the skin at station z, height y, on side sx (+1 right), lifted by `off`.
fn on_side(z: f32, y: f32, sx: f32, off: f32) -> (Vec3, Vec3) {
    let th = fuse_th(z, y);
    let (mut p, mut n) = (fuse_pt(z, th), fuse_n(z, th));
    p.x *= sx;
    n.x *= sx;
    (p + n * off, n)
}
/// Height of the lower fuselage skin at station z, lateral position x (None outside it).
fn fuse_bottom_y(z: f32, x: f32) -> Option<f32> {
    let s = sec(z);
    if x.abs() >= s.w * 0.999 {
        return None;
    }
    let x = x.abs();
    let (mut a, mut b) = (FRAC_PI_2, PI);
    for _ in 0..30 {
        let m = (a + b) * 0.5;
        if sec_pt(&s, m).x > x { a = m } else { b = m }
    }
    Some(sec_pt(&s, (a + b) * 0.5).y)
}

// ---------------------------------------------------------------- wing

/// Span station of the wing tip, where the winglet stands.
pub const SPAN: f32 = 31.4;
/// Leading and trailing edge positions (drawing distance from the nose) at a
/// span station, traced from the plan view: 42 then 40 degrees of leading-edge
/// sweep, and the trailing edge's inboard extension out to the kink at 13.4 m.
fn le_d(x: f32) -> f32 {
    if x < 21.0 { 18.06 + 0.906 * x } else { 37.086 + 0.841 * (x - 21.0) }
}
fn te_d(x: f32) -> f32 {
    if x < 13.4 { 34.93 + 0.315 * x } else { 39.151 + 0.580 * (x - 13.4) }
}
/// Height above the ground of the wing's mean plane (from the front view: about 5° of dihedral at rest).
fn wing_mid_h(x: f32) -> f32 {
    if x < 15.0 { 3.72 + 0.10 * (x - 3.25) } else { 4.895 + 0.083 * (x - 15.0) }
}
/// Thickness to chord: 13.4% at the root, 9.3% at the kink, 8% at the tip.
fn wing_tc(x: f32) -> f32 {
    if x <= 3.3 {
        0.134
    } else if x < 13.4 {
        0.134 + (0.093 - 0.134) * (x - 3.3) / 10.1
    } else {
        0.093 + (0.08 - 0.093) * ((x - 13.4) / 18.2).min(1.0)
    }
}
/// Incidence: 2° at the root washing out to -1.5° at the tip.
fn wing_inc(x: f32) -> f32 {
    let d = if x <= 3.3 { 2.0 } else if x < 13.4 { 2.0 - 1.5 * (x - 3.3) / 10.1 } else { 0.5 - 2.0 * ((x - 13.4) / 18.2).min(1.0) };
    d.to_radians()
}
/// Right wing section at span station x.
pub fn wing_sect(x: f32) -> Sect {
    let x = x.max(0.0);
    let le = bz(le_d(x));
    let c = te_d(x) - le_d(x);
    let inc = wing_inc(x);
    let dih = 0.093f32;
    let ym = by(wing_mid_h(x));
    Sect { le: Vec3::new(x, ym + inc.sin() * c * 0.5, le), c, cdir: Vec3::new(0.0, -inc.sin(), inc.cos()), up: Vec3::new(-dih, inc.cos(), inc.sin()).normalize(), tf: wing_tc(x) / 0.134 }
}
pub fn wing_foil() -> Foil {
    Foil { at: Box::new(wing_sect), t: 0.134, camber: 0.018 }
}
/// Control surfaces along the span: inboard flap, the inboard (all-speed)
/// aileron behind the inboard engine, outboard flap, outboard (low-speed) aileron.
pub const IB_FLAP: [f32; 2] = [3.3, 10.95];
pub const IB_AIL: [f32; 2] = [11.0, 13.35];
pub const OB_FLAP: [f32; 2] = [13.4, 24.0];
pub const OB_AIL: [f32; 2] = [24.05, 30.5];
/// The chord fraction each starts at.
const XF: [f32; 4] = [0.70, 0.76, 0.73, 0.77];
const TANK: [f32; 2] = [3.3, 29.0];
/// Engines: span station, height of the centreline above the ground, inlet lip from the nose.
pub const ENGINES: [[f32; 3]; 2] = [[11.68, 2.50, 23.24], [21.0, 3.42, 32.41]];

/// Point and outward normal on a foil.
fn foil_frame(f: &Foil, s: f32, xc: f32, side: f32) -> (Vec3, Vec3) {
    let p = f.pt(s, xc, side, -1.0);
    let ds = f.pt(s + 0.01, xc, side, -1.0) - f.pt(s - 0.01, xc, side, -1.0);
    let dx = f.pt(s, xc + 0.002, side, -1.0) - f.pt(s, (xc - 0.002).max(0.0), side, -1.0);
    let n = (ds.cross(dx) * side).normalize();
    (p, n)
}
/// Height of the wing's lower skin at (x, z), if the wing is there.
fn wing_bottom_y(x: f32, z: f32) -> Option<f32> {
    let x = x.abs();
    if x > SPAN {
        return None;
    }
    let k = wing_sect(x);
    let xc = (z - k.le.z) / k.c;
    if !(0.0..=1.0).contains(&xc) {
        return None;
    }
    Some(wing_foil().pt(x, xc, -1.0, -1.0).y)
}
/// The lowest skin at (z, x): the fuselage's or the wing's.
fn skin_bottom(z: f32, x: f32) -> f32 {
    match (fuse_bottom_y(z, x), wing_bottom_y(x, z)) {
        (Some(a), Some(b)) => a.min(b),
        (Some(a), None) => a,
        (None, Some(b)) => b,
        _ => sec(z).bot,
    }
}

// ---------------------------------------------------------------- openings

/// The flight deck windows. The windshields are drawn in the front view
/// (|x|, y): their outline less its 6 cm corner radius. The side windows are
/// in (distance from the nose, angle round the section from the crown, about
/// a point 0.3 m up): two each side (the first entry is unused). Keep in step
/// with `window747` in plane.wgsl.
pub const WSHIELD: [[f32; 2]; 5] = [[0.13, 3.18], [1.08, 2.94], [1.20, 3.05], [0.84, 3.49], [0.13, 3.81]];
pub const WIN: [[f32; 4]; 3] = [[0.0, 0.0, 0.0, 0.0], [5.30, 5.86, 0.30, 0.55], [5.98, 6.40, 0.36, 0.55]];
/// Signed distance to the nearest window opening (negative inside), body frame.
pub fn window(p: Vec3) -> f32 {
    let d = p.z + ZN;
    if !(3.9..=6.7).contains(&d) || p.y < 2.0 {
        return 1.0;
    }
    let mut best = 1.0f32;
    if d < 5.45 {
        let poly = WSHIELD.map(|v| Vec2::new(v[0], v[1]));
        best = (sd_poly(Vec2::new(p.x.abs(), p.y), &poly) - 0.06).max(d - 5.26);
    }
    let th = p.x.abs().atan2(p.y - 0.3);
    for w in &WIN[1..] {
        // A rounded box in metres: the angle scaled by the local radius.
        let q = Vec2::new((d - (w[0] + w[1]) * 0.5).abs() - (w[1] - w[0]) * 0.5, ((th - (w[2] + w[3]) * 0.5).abs() - (w[3] - w[2]) * 0.5) * 3.4);
        let r = 0.06;
        let q = q + Vec2::splat(r);
        best = best.min(q.max(Vec2::ZERO).length() + q.x.max(q.y).min(0.0) - r);
    }
    best
}

/// Gear wells in the belly and the lower wing root, in plan (z, |x|): the
/// nose gear's, the wing gear's with the slot its leg swings through, the
/// body gear's likewise. Keep in step with `well747` in plane.wgsl.
const NOSE_WELL: [f32; 4] = [-27.35, -22.95, 0.0, 0.78];
const WING_WELL: [f32; 4] = [-0.95, 0.45, 1.15, 4.0];
const BODY_WELL: [f32; 4] = [0.15, 3.05, 1.05, 2.8];
const BODY_SLOT: [f32; 4] = [3.0, 5.75, 1.6, 2.24];
/// The wing gear leg's slot: from its trunnion inboard to where the truck stows.
const WING_SLOT: [[f32; 2]; 2] = [[0.85, 5.75], [-0.30, 2.75]];
const WING_SLOT_W: f32 = 0.36;

fn sd_box(p: Vec2, b: [f32; 4]) -> f32 {
    let c = Vec2::new((b[0] + b[1]) * 0.5, (b[2] + b[3]) * 0.5);
    let h = Vec2::new((b[1] - b[0]) * 0.5, (b[3] - b[2]) * 0.5);
    let q = (p - c).abs() - h;
    q.max(Vec2::ZERO).length() + q.x.max(q.y).min(0.0)
}
fn sd_seg(p: Vec2, a: Vec2, b: Vec2) -> f32 {
    let (pa, ba) = (p - a, b - a);
    (pa - ba * (pa.dot(ba) / ba.dot(ba)).clamp(0.0, 1.0)).length()
}
/// Signed distance to the gear wells (5 mm inside their walls).
pub fn well(p: Vec3) -> f32 {
    if p.y > -1.2 {
        return 1.0;
    }
    let q = Vec2::new(p.z, p.x.abs());
    let mut d = sd_box(q, [NOSE_WELL[0], NOSE_WELL[1], -NOSE_WELL[3], NOSE_WELL[3]]);
    d = d.min(sd_box(q, WING_WELL)).min(sd_box(q, BODY_WELL)).min(sd_box(q, BODY_SLOT));
    let [a, b] = WING_SLOT.map(|v| Vec2::new(v[0], v[1]));
    d = d.min(sd_seg(q, a, b) - WING_SLOT_W);
    d + 0.005
}

// ---------------------------------------------------------------- the gear

/// Tyres: 49 x 19 on the mains, 49 x 17 on the nose (1.245 m across).
const TYRE_R: f32 = 0.622;
/// Nose gear: trunnion and axle (fully extended), the wheels either side.
const NOSE_PIVOT: Vec3 = Vec3::new(0.0, -1.50, -23.40);
const NOSE_AXLE: Vec3 = Vec3::new(0.0, -4.53, -23.85);
const NOSE_STOWED: Vec3 = Vec3::new(0.0, -1.35, -26.46);
const NOSE_TRACK: f32 = 0.44;
/// Wing gear (right): trunnion, truck centre extended and stowed.
const WING_PIVOT: Vec3 = Vec3::new(5.62, -1.50, 0.72);
const WING_AXLE: Vec3 = Vec3::new(5.5, -4.65, 0.22);
const WING_STOWED: Vec3 = Vec3::new(2.58, -1.34, -0.24);
/// Body gear (right).
const BODY_PIVOT: Vec3 = Vec3::new(1.92, -1.40, 5.45);
const BODY_AXLE: Vec3 = Vec3::new(1.92, -4.65, 3.29);
const BODY_STOWED: Vec3 = Vec3::new(1.92, -1.40, 1.55);
/// Four-wheel trucks: 1.47 m between axles, 1.12 m between the tyres' centres;
/// tilted 13° aft-down until weight comes on them.
const TRUCK_L: f32 = 1.47;
const TRUCK_W: f32 = 1.12;
const TILT: f32 = 0.23;

/// How far a gear leg turns to stow (rad) and about which axis.
fn stow(pivot: Vec3, axle: Vec3, stowed: Vec3) -> (Vec3, f32) {
    let (a, b) = ((axle - pivot).normalize(), (stowed - pivot).normalize());
    (a.cross(b).normalize(), a.dot(b).clamp(-1.0, 1.0).acos())
}

/// The cockpit eye: the captain's seat.
pub const EYE: Vec3 = Vec3::new(-0.53, 3.40, -25.95);

// ---------------------------------------------------------------- materials

const SKIN: Mat = WHITE;
const GREY_SKIN: Mat = mt(mat::LIVERY, 0.45, 0.47, 0.50);
const WELL: Mat = mt(mat::SATIN, 0.17, 0.19, 0.17);
const DARK: Mat = mt(mat::BLACK, 0.02, 0.02, 0.022);
const LINER: Mat = mt(mat::SATIN, 0.62, 0.63, 0.64);
const TITANIUM: Mat = mt(mat::STEEL, 0.50, 0.50, 0.52);
const NOZZLE: Mat = mt(mat::EXHAUST, 0.3, 0.28, 0.26);
const STRUT: Mat = mt(mat::PAINT, 0.62, 0.63, 0.64);
const HOSE: Mat = mt(mat::BLACK, 0.03, 0.03, 0.032);
const PIPE: Mat = mt(mat::STEEL, 0.55, 0.56, 0.58);

/// A static discharger: a thin composite rod trailing aft from a trailing edge,
/// its brush tip at the end.
fn wick(m: &mut Mesh, at: Vec3, dir: Vec3) {
    let d = dir.normalize();
    m.rbox(&mt(mat::SATIN, 0.5, 0.5, 0.5), at + d * 0.01, Vec3::new(0.012, 0.006, 0.03), 0.004, Quat::from_rotation_arc(Vec3::Z, d));
    m.tube(&DARK, &[at - d * 0.02, at + d * 0.17], &|t| 0.0065 * (1.0 - 0.45 * t), 6, true);
    m.tube(&mt(mat::SATIN, 0.35, 0.35, 0.36), &[at + d * 0.17, at + d * 0.205], &|_| 0.0022, 4, true);
}

// ---------------------------------------------------------------- the model

pub fn build() -> Model {
    let mut b = Builder::default();
    fuselage(&mut b);
    flight_deck(&mut b);
    wings(&mut b);
    engines(&mut b);
    tail(&mut b);
    gear(&mut b);
    bake_ao(&mut b.pieces, &AoBake { vox: 0.14, reach: 14.0, rays: 20, open: &|q, pat| window(q) < 0.0 || (pat != pat::LINING && well(q) < 0.0) });
    // The flight deck gets its own, much finer bake: the nose's skin shuts
    // it in, the light comes through the windows.
    let mut d = Builder::default();
    deck::build(&mut d);
    let n_deck = d.pieces.len();
    if let Some(k) = b.pieces.iter().position(|p| p.name == "flight deck section") {
        let mut skin = b.pieces[k].mesh.clone();
        skin.i = skin.i.chunks(3).filter(|t| t.iter().all(|&i| skin.v[i as usize].pos[2] < bz(9.5))).flatten().copied().collect();
        d.fixed("skin (occluder)", G_ENGINE, skin);
    }
    bake_ao(&mut d.pieces, &AoBake { vox: 0.02, reach: 4.5, rays: 48, open: &|q, pat| if pat == pat::LINING { deck::lining_window(q) < 0.0 } else { window(q) < 0.0 } });
    let off = b.pieces.len();
    for mut p in d.pieces.into_iter().take(n_deck) {
        p.parent = p.parent.map(|i| i + off);
        b.pieces.push(p);
    }
    // The doors on the gear legs are baked tucked against the strut, but
    // spend most of their time flush with the belly in the open air.
    for p in b.pieces.iter_mut().filter(|p| p.name.ends_with("gear leg")) {
        for v in p.mesh.v.iter_mut().filter(|v| v.mat[0] == mat::LIVERY) {
            v.col[3] = 0.9;
        }
    }
    Model::finish(b.pieces)
}

/// The fuselage skin between two stations, both halves, with `pattern`.
fn skin_rows(d0: f32, d1: f32, pattern: f32, inset: f32, flip: bool, m: &Mat) -> Mesh {
    let mut ds = Vec::new();
    let mut d = d0;
    while d < d1 - 1e-4 {
        ds.push(d);
        d += if d < 1.0 { 0.05 } else if d < 8.0 { 0.15 } else if d > 66.0 { 0.08 } else { 0.4 };
    }
    ds.push(d1);
    const NT: usize = 96;
    let rows: Vec<Vec<SP>> = ds
        .iter()
        .map(|&d| {
            let z = bz(d);
            let s = sec(z);
            let mut arc = 0.0;
            let mut prev = sec_pt(&s, 0.0);
            (0..=NT)
                .map(|k| {
                    let th = k as f32 / NT as f32 * PI;
                    let p = sec_pt(&s, th);
                    arc += (p - prev).length();
                    prev = p;
                    let mut q = Vec3::new(p.x, p.y, z);
                    if inset != 0.0 {
                        q -= fuse_n(z, th) * inset;
                    }
                    (q, [z, arc], 0.0)
                })
                .collect()
        })
        .collect();
    let mut r = Mesh::default();
    r.grid(&m.pat(pattern), &rows, false, flip);
    let mut l = r.mirror_x();
    // The left half is the mirror image, its surface coordinate negative.
    for v in &mut l.v {
        v.uv[1] = -v.uv[1];
    }
    // Mirroring turns the triangles inside out: wind them back.
    for t in l.i.chunks_mut(3) {
        t.swap(1, 2);
    }
    r.append(&l);
    r
}

fn fuselage(b: &mut Builder) {
    let radome = skin_rows(D_NOSE, D_RADOME, pat::FUSE, 0.0, false, &SKIN);
    let mut deck = skin_rows(D_RADOME, D_DECK, pat::FUSE, 0.0, false, &SKIN);
    let mut body = skin_rows(D_DECK, D_CONE, pat::FUSE, 0.0, false, &SKIN);
    let mut cone = skin_rows(D_CONE, D_TAIL, pat::FUSE, 0.0, false, &SKIN);
    // The APU exhaust at the very tip of the tail cone.
    let s = sec(bz(69.55));
    cone.disc(&DARK, Vec3::new(0.0, s.mid, bz(69.6)), Vec3::Z, Vec3::Y, 0.10, 16);

    // Gear wells: walls rising from the openings in the skin to their roofs.
    wells(&mut body, &mut deck);

    // Lights: red beacons on top behind the wing and under the belly, a white
    // light and a strobe on the tail cone.
    let beacon = mt(mat::LAMP, 0.9, 0.06, 0.03).x(lamp::BEACON);
    let top = Vec3::new(0.0, sec(bz(34.0)).top, bz(34.0));
    body.lathe(&beacon, top, Vec3::Y, &[[0.09, 0.0], [0.09, 0.04], [0.06, 0.11], [0.0, 0.13]], 16, true);
    let bot = Vec3::new(0.0, sec(bz(25.0)).bot, bz(25.0));
    body.lathe(&beacon, bot, Vec3::NEG_Y, &[[0.09, 0.0], [0.09, 0.04], [0.06, 0.11], [0.0, 0.13]], 16, true);
    let tail = Vec3::new(0.0, sec(bz(69.0)).mid, bz(69.0));
    for sx in [-1.0f32, 1.0] {
        cone.lathe(&mt(mat::LAMP, 0.95, 0.95, 0.95).x(lamp::TAIL), tail + Vec3::new(sx * 0.23, 0.15, 0.0), Vec3::new(sx, 0.0, 0.3), &[[0.04, 0.0], [0.04, 0.02], [0.0, 0.04]], 10, true);
    }
    cone.lathe(&mt(mat::LAMP, 0.9, 0.9, 0.9).x(lamp::STROBE), tail + Vec3::new(0.0, 0.37, 0.0), Vec3::Y, &[[0.035, 0.0], [0.035, 0.03], [0.0, 0.05]], 10, true);

    // Antennas: blades on the hump and the crown, under the belly; the satcom
    // fairing on top behind the hump.
    let blade = |m: &mut Mesh, d: f32, up: bool, h: f32| {
        let z = bz(d);
        let s = sec(z);
        let (y, dir) = if up { (s.top, 1.0) } else { (s.bot, -1.0) };
        let o = Vec2::new(-0.18, 0.0);
        let pts = [o, Vec2::new(0.18, 0.0), Vec2::new(0.06, h * dir), Vec2::new(-0.04, h * dir)];
        m.slab(&DARK, &pts, &[], Vec3::new(0.012, y, z), Vec3::Z, Vec3::Y, 0.024);
    };
    for (d, up, h) in [(14.0, true, 0.32), (19.0, true, 0.28), (39.0, true, 0.30), (44.0, true, 0.25), (20.0, false, 0.22), (45.0, false, 0.22), (12.0, false, 0.16)] {
        blade(if d > D_CONE { &mut cone } else { &mut body }, d, up, h);
    }
    let z = bz(33.0);
    body.rbox(&mt(mat::PAINT, 0.80, 0.80, 0.78), Vec3::new(0.0, sec(z).top + 0.05, z), Vec3::new(0.35, 0.10, 0.9), 0.09, Quat::IDENTITY);
    // Pitot probes and the angle-of-attack vanes on the nose.
    for sx in [-1.0f32, 1.0] {
        for (d, y) in [(3.6, -0.55), (4.0, -0.85)] {
            let (p, n) = on_side(bz(d), y, sx, 0.0);
            deck.tube(&ALU, &[p - n * 0.02, p + n * 0.09, p + n * 0.11 - Vec3::Z * 0.16], &|_| 0.012, 8, true);
        }
        let (p, n) = on_side(bz(4.5), -0.25, sx, 0.0);
        deck.slab(&DARK, &[Vec2::new(0.0, 0.0), Vec2::new(0.09, 0.0), Vec2::new(0.06, 0.06), Vec2::new(0.0, 0.06)], &[], p, Vec3::Z, n, 0.01);
    }
    // Drain masts under the belly, ahead of and behind the wing: small swept blades.
    for d in [16.0f32, 47.0] {
        let z = bz(d);
        let y = sec(z).bot;
        let pts = [Vec2::new(-0.07, 0.0), Vec2::new(0.07, 0.0), Vec2::new(0.11, -0.15), Vec2::new(0.07, -0.16)];
        body.slab(&mt(mat::PAINT, 0.70, 0.70, 0.69), &pts, &[], Vec3::new(0.012, y + 0.01, z), Vec3::Z, Vec3::Y, 0.024);
        body.tube(&DARK, &[Vec3::new(0.0, y - 0.15, z + 0.08), Vec3::new(0.0, y - 0.16, z + 0.12)], &|_| 0.006, 6, true);
    }
    // The total air temperature probe on the left of the nose, and an ice detector on the right.
    {
        let (p, n) = on_side(bz(6.3), -0.75, -1.0, 0.0);
        deck.slab(&ALU, &[Vec2::new(-0.05, 0.0), Vec2::new(0.07, 0.0), Vec2::new(0.06, 0.09), Vec2::new(0.0, 0.10)], &[], p - n * 0.01, Vec3::Z, n, 0.014);
        deck.tube(&ALU, &[p + n * 0.09, p + n * 0.10 + Vec3::Z * 0.03], &|_| 0.012, 8, true);
        let (p, n) = on_side(bz(6.6), -0.55, 1.0, 0.0);
        deck.tube(&ALU, &[p - n * 0.01, p + n * 0.07, p + n * 0.08 + Vec3::Z * 0.02], &|t| 0.014 - 0.006 * t, 8, true);
    }
    // Rows of fasteners round the radome joint.
    let zr = bz(D_RADOME) + 0.03;
    let s = sec(zr);
    for k in 0..96 {
        let th = k as f32 / 96.0 * TAU;
        let (t, sx) = if th > PI { (TAU - th, -1.0) } else { (th, 1.0) };
        let q = sec_pt(&s, t);
        let p = Vec3::new(q.x * sx, q.y, zr);
        let n = fuse_n(zr, t) * Vec3::new(sx, 1.0, 1.0);
        deck.screw(&SCREW_PAINT, p + n * 0.001, n, 0.008, k as f32);
    }

    b.fixed("radome", G_PROP, radome);
    deck::wipers(&mut deck);
    b.fixed("flight deck section", G_ENGINE, deck);
    b.fixed("fuselage", G_FUSE, body);
    let t = b.fixed("tail cone", G_TAIL, cone);
    b.role(t, Role::TailCone);
}

/// Wall from the skin up to `roof` along an outline in plan (z, x).
fn well_wall(m: &mut Mesh, pts: &[(f32, f32)], roof: f32) {
    for k in 0..pts.len() {
        let (a, c) = (pts[k], pts[(k + 1) % pts.len()]);
        let n = 10;
        let rows: Vec<Vec<SP>> = (0..=n)
            .map(|i| {
                let t = i as f32 / n as f32;
                let (z, x) = (a.0 + (c.0 - a.0) * t, a.1 + (c.1 - a.1) * t);
                let ys = skin_bottom(z, x) - 0.01;
                let yt = roof.max(ys + 0.05);
                vec![(Vec3::new(x, ys, z), [z, ys], 0.0), (Vec3::new(x, yt, z), [z, yt], 0.0)]
            })
            .collect();
        m.grid(&WELL, &rows, false, false);
    }
}
/// A flat roof over a convex outline in plan (z, x), facing down.
fn well_roof(m: &mut Mesh, pts: &[(f32, f32)], roof: f32) {
    let ring: Vec<Vec3> = pts.iter().map(|&(z, x)| Vec3::new(x, roof.max(skin_bottom(z, x) + 0.05), z)).collect();
    let c = ring.iter().copied().sum::<Vec3>() / ring.len() as f32;
    m.fan(&WELL, c, &ring, Vec3::NEG_Y);
}
fn box_pts(b: [f32; 4]) -> [(f32, f32); 4] {
    [(b[0], b[2]), (b[1], b[2]), (b[1], b[3]), (b[0], b[3])]
}
fn wells(body: &mut Mesh, deck: &mut Mesh) {
    // The nose gear's, under the flight deck section.
    let [z0, z1, _, hw] = NOSE_WELL;
    let nose = [(z0, -hw), (z1, -hw), (z1, hw), (z0, hw)];
    well_wall(deck, &nose.iter().rev().copied().collect::<Vec<_>>(), -0.62);
    well_roof(deck, &nose, -0.62);
    let mut side = Mesh::default();
    // Wing gear well and the slot its leg swings through; the body gear's.
    let ww = box_pts(WING_WELL);
    well_wall(&mut side, &ww.iter().rev().copied().collect::<Vec<_>>(), -0.6);
    well_roof(&mut side, &ww, -0.6);
    let [a, c] = WING_SLOT;
    let d = Vec2::new(c[0] - a[0], c[1] - a[1]).normalize();
    let n = Vec2::new(-d.y, d.x) * WING_SLOT_W;
    let slot = [(a[0] + n.x, a[1] + n.y), (c[0] + n.x, c[1] + n.y), (c[0] - n.x, c[1] - n.y), (a[0] - n.x, a[1] - n.y)];
    well_wall(&mut side, &slot, -1.3);
    well_roof(&mut side, &slot, -1.3);
    for bx in [BODY_WELL, BODY_SLOT] {
        let p = box_pts(bx);
        well_wall(&mut side, &p.iter().rev().copied().collect::<Vec<_>>(), -1.2);
        well_roof(&mut side, &p, -1.2);
    }
    let mut l = side.mirror_x();
    for t in l.i.chunks_mut(3) {
        t.swap(1, 2);
    }
    body.append(&side);
    body.append(&l);
}

/// The flight deck windows' glass (the deck itself is in deck.rs).
fn flight_deck(b: &mut Builder) {
    // The windows: a sheet just inside the skin over the whole window band.
    let mut g = Mesh::default();
    let rows: Vec<Vec<SP>> = (0..=110)
        .map(|i| {
            let d = 3.9 + 2.65 * i as f32 / 110.0;
            let z = bz(d);
            let s = sec(z);
            (0..=72)
                .map(|j| {
                    let th = 0.9 * j as f32 / 72.0;
                    let p = sec_pt(&s, th);
                    let q = Vec3::new(p.x, p.y, z) - fuse_n(z, th) * 0.012;
                    (q, [d, th], 0.0)
                })
                .collect()
        })
        .collect();
    g.grid(&mt(mat::GLASS, 0.1, 0.1, 0.1), &rows, false, false);
    let mut l = g.mirror_x();
    for t in l.i.chunks_mut(3) {
        t.swap(1, 2);
    }
    g.append(&l);
    let k = b.fixed("windows", G_ENGINE, g);
    b.pieces[k].glass = true;
    b.pieces[k].occluder = false;
}

// ---------------------------------------------------------------- wings

fn wings(b: &mut Builder) {
    let f = wing_foil();
    let skin = GREY_SKIN.pat(pat::WING);
    let ctrl = GREY_SKIN.pat(pat::CTRL);
    let one = |_: f32| 1.0;
    let zero = |_: f32| 0.0;
    let mut w = Mesh::default();
    // Skin, leaving the trailing edge where the flaps and ailerons are.
    let spans = [(0.0, IB_FLAP[0], 1.0), (IB_FLAP[0], IB_FLAP[1], XF[0]), (IB_FLAP[1], IB_AIL[0], 1.0), (IB_AIL[0], IB_AIL[1], XF[1]), (IB_AIL[1], OB_FLAP[0], 1.0), (OB_FLAP[0], OB_FLAP[1], XF[2]), (OB_FLAP[1], OB_AIL[0], 1.0), (OB_AIL[0], OB_AIL[1], XF[3]), (OB_AIL[1], SPAN, 1.0)];
    for &(s0, s1, x1) in &spans {
        let ns = ((s1 - s0) * 2.5).ceil().max(1.0) as usize;
        f.skin(&mut w, &skin, s0, s1, ns, &zero, &|_| x1, 34, false);
        if x1 < 1.0 {
            f.web(&mut w, &WELL, s0, s1, ns, &|_| x1);
            f.cap(&mut w, &WELL, s0, x1, 1.0, false);
            f.cap(&mut w, &WELL, s1, x1, 1.0, false);
        }
    }
    f.cap(&mut w, &skin, SPAN, 0.0, 1.0, false);
    // Landing lights in the leading edge at the root (two each side).
    for s0 in [3.9f32, 4.6] {
        let lens = mt(mat::LAMP, 0.95, 0.92, 0.85).x(lamp::LANDING);
        w.patch(&lens, 8, 10, false, |a, bb| {
            let s = s0 + (a - 0.5) * 0.5;
            let side = if bb < 0.5 { 1.0 } else { -1.0 };
            let xc = (bb - 0.5).abs() * 2.0 * 0.012;
            let (p, n) = foil_frame(&f, s, xc.max(0.0004), side);
            (p + n * 0.006, [(a - 0.5) * 2.0, (bb - 0.5) * 2.0], 0.0)
        });
    }
    // Fuel vents and the overwing fuelling-panel dots of paint; fasteners along the spars.
    for k in 0..90 {
        let s = 3.5 + k as f32 * 0.31;
        if s > SPAN - 0.4 {
            break;
        }
        for (xc, side) in [(0.13, 1.0), (0.13, -1.0)] {
            let (p, n) = foil_frame(&f, s, xc, side);
            w.screw(&SCREW_PAINT, p + n * 0.001, n, 0.011, s * 7.0 + side);
        }
    }
    for &(s, xc) in &[(6.0f32, 0.40f32), (9.0, 0.42), (15.0, 0.40), (18.5, 0.41), (25.0, 0.40), (28.5, 0.42)] {
        let (p, n) = foil_frame(&f, s, xc, -1.0);
        inspection_plate(&mut w, p + n * 0.003, n, Vec3::Z, 0.13, 8);
    }

    // Flap track fairings ("canoes") under the wing: the front of each fixed,
    // the back riding down with its flap.
    let canoe = |wf: &Foil, m_fixed: &mut Mesh, m_flap: &mut Mesh, s: f32, xf: f32| {
        let k = (wf.at)(s);
        let path = |xc: f32| -> Vec3 {
            let base = if xc <= 1.0 { wf.pt(s, xc, -1.0, -1.0) } else { wf.pt(s, 1.0, -1.0, -1.0) + k.cdir * ((xc - 1.0) * k.c) };
            base
        };
        // Deep and blunt, tapering to a point a fifth of a chord past the trailing edge.
        let r = |xc: f32| -> f32 {
            let t = ((xc - 0.45) / 0.77).clamp(0.0, 1.0);
            0.52 * (t * PI).sin().max(0.0).powf(0.4) * (1.0 - 0.3 * t)
        };
        let sample = |a: f32, bb: f32, n: usize| -> (Vec<Vec3>, Vec<f32>) {
            let xs: Vec<f32> = (0..=n).map(|i| a + (bb - a) * i as f32 / n as f32).collect();
            (xs.iter().map(|&x| path(x) - k.up * r(x) * 0.55).collect(), xs.iter().map(|&x| r(x).max(0.02)).collect())
        };
        let (p1, r1) = sample(0.45, xf, 12);
        m_fixed.tube(&GREY_SKIN, &p1, &|t| r1[((t * 12.0).round() as usize).min(12)], 14, false);
        let (p2, r2) = sample(xf, 1.22, 12);
        m_flap.tube(&GREY_SKIN, &p2, &|t| r2[((t * 12.0).round() as usize).min(12)], 14, true);
    };

    // Flaps: triple-slotted, running back and down on their tracks (a hinge
    // well below the wing gives the motion).
    for (span, xf, name, tracks) in [(IB_FLAP, XF[0], ["left inboard flap", "right inboard flap"], [5.8f32, 9.6]), (OB_FLAP, XF[2], ["left outboard flap", "right outboard flap"], [16.0, 22.5])] {
        let mut fl = Mesh::default();
        let (s0, s1) = (span[0] + 0.03, span[1] - 0.03);
        let ns = ((s1 - s0) * 2.0).ceil() as usize;
        f.skin(&mut fl, &ctrl, s0, s1, ns, &|_| xf + 0.004, &one, 16, true);
        f.cap(&mut fl, &ctrl, s0, xf + 0.004, 1.0, true);
        f.cap(&mut fl, &ctrl, s1, xf + 0.004, 1.0, true);
        // The vane slots: dark lines across the flap's upper surface.
        for xs in [xf + 0.07, xf + 0.17] {
            fl.patch(&DARK, ns, 1, false, |a, bb| {
                let s = s0 + (s1 - s0) * a;
                let (p, n) = foil_frame(&f, s, xs + bb * 0.006, 1.0);
                (p + n * 0.004, [s, bb], 0.0)
            });
        }
        for t in tracks {
            canoe(&f, &mut w, &mut fl, t, xf);
        }
        let sm = (s0 + s1) * 0.5;
        let k = wing_sect(sm);
        let axis = (f.pt(s1, xf, 0.0, -1.0) - f.pt(s0, xf, 0.0, -1.0)).normalize();
        let pivot = f.pt(sm, xf + 0.05, -1.0, -1.0) - k.up * (0.11 * k.c);
        let (l, r) = b.pair(name, [G_LWING, G_RWING], fl, |_| Anim::hinge(Drive::new(Input::Flaps, 30f32.to_radians())), pivot, axis, [None, None]);
        b.role(l, Role::Flap(0)).role(r, Role::Flap(1));
    }
    // A fairing under each engine pylon's back end, and the outboard canoe.
    for e in ENGINES {
        let mut dummy = Mesh::default();
        canoe(&f, &mut w, &mut dummy, e[0], 0.95);
    }
    canoe(&f, &mut w, &mut Mesh::default(), 27.5, 0.98);

    // Ailerons.
    for (span, xf, name) in [(IB_AIL, XF[1], ["left inboard aileron", "right inboard aileron"]), (OB_AIL, XF[3], ["left outboard aileron", "right outboard aileron"])] {
        let mut al = Mesh::default();
        let (s0, s1) = (span[0] + 0.02, span[1] - 0.02);
        let ns = ((s1 - s0) * 2.0).ceil() as usize;
        f.skin(&mut al, &ctrl, s0, s1, ns, &|_| xf + 0.005, &one, 14, true);
        f.cap(&mut al, &ctrl, s0, xf + 0.005, 1.0, true);
        f.cap(&mut al, &ctrl, s1, xf + 0.005, 1.0, true);
        if span == OB_AIL {
            for k in 0..4 {
                let s = s0 + 0.5 + k as f32 * (s1 - s0 - 1.0) / 3.0;
                wick(&mut al, f.pt(s, 1.0, 0.0, -1.0), wing_sect(s).cdir);
            }
        } else {
            let s = (s0 + s1) * 0.5;
            wick(&mut al, f.pt(s, 1.0, 0.0, -1.0), wing_sect(s).cdir);
        }
        let axis = (f.pt(s1, xf + 0.02, 0.0, -1.0) - f.pt(s0, xf + 0.02, 0.0, -1.0)).normalize();
        let pivot = f.pt((s0 + s1) * 0.5, xf + 0.02, 0.0, -1.0);
        let (l, r) = b.pair(name, [G_LWING, G_RWING], al, |s| Anim::hinge(Drive::new(Input::Aileron(s), 0.35)), pivot, axis, [None, None]);
        b.role(l, Role::Aileron(0)).role(r, Role::Aileron(1));
    }

    // Spoilers: six panels on each upper surface, ahead of the flaps; they
    // rise on the wing going down, as speedbrakes, and all of them on landing.
    let spoilers: [(f32, f32, f32); 6] = [(4.2, 7.4, XF[0]), (7.4, 10.6, XF[0]), (13.6, 16.1, XF[2]), (16.1, 18.6, XF[2]), (18.6, 21.1, XF[2]), (21.1, 23.6, XF[2])];
    for (k, &(s0, s1, xf)) in spoilers.iter().enumerate() {
        let mut sp = Mesh::default();
        let (x0, x1) = (0.585, xf - 0.003);
        sp.patch(&skin, 10, 6, false, |a, c| {
            let s = s0 + (s1 - s0) * a;
            let (p, n) = foil_frame(&f, s, x0 + (x1 - x0) * c, 1.0);
            (p + n * 0.009, [s, (x0 + (x1 - x0) * c) * wing_sect(s).c], 0.0)
        });
        sp.patch(&WELL, 10, 6, true, |a, c| {
            let s = s0 + (s1 - s0) * a;
            let (p, n) = foil_frame(&f, s, x0 + (x1 - x0) * c, 1.0);
            (p + n * 0.003, [s, c], 0.0)
        });
        let axis = (f.pt(s1, x0, 1.0, -1.0) - f.pt(s0, x0, 1.0, -1.0)).normalize();
        let pivot = f.pt((s0 + s1) * 0.5, x0, 1.0, -1.0);
        const NAMES: [[&str; 2]; 6] = [["left spoiler 1", "right spoiler 1"], ["left spoiler 2", "right spoiler 2"], ["left spoiler 3", "right spoiler 3"], ["left spoiler 4", "right spoiler 4"], ["left spoiler 5", "right spoiler 5"], ["left spoiler 6", "right spoiler 6"]];
        b.pair(NAMES[k], [G_LWING, G_RWING], sp, |s| Anim::hinge(Drive::new(Input::Spoiler(s), -0.95)), pivot, axis, [None, None]);
    }

    // Leading-edge Krueger flaps under the leading edge: three inboard, the
    // variable-camber ones outboard; they swing down and forward.
    let kruegers: [(f32, f32); 12] = [(4.0, 6.3), (6.3, 8.6), (8.6, 10.9), (12.3, 14.4), (14.4, 16.4), (16.4, 18.4), (18.4, 20.4), (21.6, 23.4), (23.4, 25.2), (25.2, 27.0), (27.0, 28.8), (28.8, 30.6)];
    const KN: [[&str; 2]; 12] = [
        ["left krueger 1", "right krueger 1"], ["left krueger 2", "right krueger 2"], ["left krueger 3", "right krueger 3"], ["left krueger 4", "right krueger 4"],
        ["left krueger 5", "right krueger 5"], ["left krueger 6", "right krueger 6"], ["left krueger 7", "right krueger 7"], ["left krueger 8", "right krueger 8"],
        ["left krueger 9", "right krueger 9"], ["left krueger 10", "right krueger 10"], ["left krueger 11", "right krueger 11"], ["left krueger 12", "right krueger 12"],
    ];
    for (k, &(s0, s1)) in kruegers.iter().enumerate() {
        let mut kr = Mesh::default();
        let (x0, x1) = (0.006, 0.115);
        kr.patch(&ALU, 8, 5, true, |a, c| {
            let s = s0 + (s1 - s0) * a;
            let (p, n) = foil_frame(&f, s, x0 + (x1 - x0) * c, -1.0);
            (p + n * 0.008, [s, c], 0.2)
        });
        kr.patch(&WELL, 8, 5, false, |a, c| {
            let s = s0 + (s1 - s0) * a;
            let (p, n) = foil_frame(&f, s, x0 + (x1 - x0) * c, -1.0);
            (p + n * 0.003, [s, c], 0.0)
        });
        let axis = (f.pt(s1, x0, -1.0, -1.0) - f.pt(s0, x0, -1.0, -1.0)).normalize();
        let pivot = f.pt((s0 + s1) * 0.5, x0, -1.0, -1.0);
        b.pair(KN[k], [G_LWING, G_RWING], kr, |_| Anim::hinge(Drive::new(Input::Slats, 2.1)), pivot, axis, [None, None]);
    }

    // Winglet: 1.8 m tall, canted out 25°, its leading edge swept 62°.
    let cant = 25f32.to_radians();
    let root = wing_sect(SPAN);
    let span_dir = Vec3::new(cant.sin(), cant.cos(), 0.0);
    let wl = Foil {
        at: Box::new(move |s: f32| {
            let u = (s / 1.83).clamp(0.0, 1.0);
            let c = root.c + (1.25 - root.c) * u;
            let mut le = root.le + span_dir * s + Vec3::Z * (s.max(0.0) * 62f32.to_radians().tan());
            let mut tf = 1.0;
            let mut cc = c;
            if s > 1.72 {
                let v = ((s - 1.72) / 0.11).min(1.0);
                let k = 1.0 - (1.0 - v * v).max(0.0).sqrt();
                le += Vec3::Z * (c * 0.2 * k);
                cc -= c * 0.3 * k;
                tf = (1.0 - v * v).max(0.05).sqrt();
            }
            Sect { le, c: cc, cdir: Vec3::Z, up: Vec3::new(cant.cos(), -cant.sin(), 0.0), tf }
        }),
        t: 0.09,
        camber: 0.0,
    };
    let mut tip = Mesh::default();
    wl.skin(&mut tip, &GREY_SKIN.pat(pat::TAIL), -0.25, 1.83, 16, &zero, &one, 20, false);
    // Lights at the tip: the green navigation light ahead of the winglet,
    // a strobe beside it, and the white tail light at its trailing edge.
    let (p, _) = foil_frame(&f, SPAN - 0.12, 0.02, 0.0);
    tip.lathe(&mt(mat::LAMP, 0.1, 0.9, 0.2).x(lamp::NAV_GREEN), p + Vec3::new(0.0, 0.0, -0.06), Vec3::new(0.5, 0.0, -0.86), &[[0.07, 0.0], [0.07, 0.04], [0.05, 0.10], [0.0, 0.12]], 16, true);
    tip.lathe(&mt(mat::LAMP, 0.9, 0.9, 0.9).x(lamp::STROBE), p + Vec3::new(0.05, 0.08, 0.12), Vec3::X, &[[0.05, 0.0], [0.05, 0.05], [0.0, 0.08]], 12, true);
    let te = wl.pt(0.15, 1.0, 0.0, -1.0);
    tip.lathe(&mt(mat::LAMP, 0.95, 0.95, 0.95).x(lamp::TAIL), te, Vec3::Z, &[[0.04, 0.0], [0.04, 0.03], [0.0, 0.05]], 10, true);
    // Static dischargers on the winglet's trailing edge.
    for s in [0.55f32, 1.1, 1.6] {
        wick(&mut tip, wl.pt(s, 1.0, 0.0, -1.0), Vec3::Z);
    }
    w.append(&tip);
    // Static dischargers on the wing tip's trailing edge, the fuel jettison
    // nozzle beside them.
    for s in [30.75f32, 31.15] {
        wick(&mut w, f.pt(s, 1.0, 0.0, -1.0), wing_sect(s).cdir);
    }
    let jp = f.pt(30.6, 0.95, -1.0, -1.0);
    w.tube(&ALU, &[jp + Vec3::Y * 0.03, jp + wing_sect(30.6).cdir * 0.32 - Vec3::Y * 0.02], &|t| 0.028 - 0.008 * t, 10, true);
    w.disc(&DARK, jp + wing_sect(30.6).cdir * 0.322 - Vec3::Y * 0.02, wing_sect(30.6).cdir, Vec3::Y, 0.018, 10);
    // A row of vortex generators along the outboard wing, ahead of the ailerons.
    let mut k = 0;
    let mut s = 21.8f32;
    while s < 30.2 {
        let (p, n) = foil_frame(&f, s, 0.10, 1.0);
        let fwd = -wing_sect(s).cdir;
        let tw = if k % 2 == 0 { 0.26f32 } else { -0.26 };
        let ax = Quat::from_axis_angle(n, tw) * fwd;
        let side = n.cross(ax).normalize();
        let pts = [Vec2::new(-0.05, 0.0), Vec2::new(0.05, 0.0), Vec2::new(-0.03, 0.045), Vec2::new(-0.05, 0.045)];
        w.slab(&GREY_SKIN, &pts, &[], p - n * 0.004 + side * 0.0015, ax, n, 0.003);
        s += 0.42;
        k += 1;
    }

    let mut lw = w.mirror_x();
    for t in lw.i.chunks_mut(3) {
        t.swap(1, 2);
    }
    // The left wing's tip light is red.
    for v in &mut lw.v {
        if v.mat[0] == mat::LAMP && v.mat[3] == lamp::NAV_GREEN {
            v.mat[3] = lamp::NAV_RED;
            v.col = [0.95, 0.1, 0.05, v.col[3]];
        }
    }
    b.fixed("left wing", G_LWING, lw);
    b.fixed("right wing", G_RWING, w);
    // Spars and ribs for the stump when a wing breaks.
    let mut ws = Mesh::default();
    let mut ribs = Vec::new();
    let mut s = 3.4;
    while s < SPAN - 0.3 {
        let to = spans.iter().find(|sp| s >= sp.0 && s < sp.1).map(|sp| sp.2).unwrap_or(1.0) - 0.03;
        ribs.push((s, 0.03, to.min(0.97)));
        s += 0.9;
    }
    structure(&f, &mut ws, 2.0, SPAN - 0.3, &[0.15, 0.62], &ribs);
    let mut lws = ws.mirror_x();
    for t in lws.i.chunks_mut(3) {
        t.swap(1, 2);
    }
    let l = b.fixed("left wing structure", G_LWING, lws);
    let r = b.fixed("right wing structure", G_RWING, ws);
    b.role(l, Role::WingStructure(0)).role(r, Role::WingStructure(1));
}

// ---------------------------------------------------------------- engines

/// One PW4056 in its nacelle with its pylon (right side), lip at `lip`: the
/// nacelle, the reverser sleeve that slides back, and the fan that turns.
fn engine(e: [f32; 3], f: &Foil) -> (Mesh, Mesh, Mesh, Vec3) {
    let c = Vec3::new(e[0], by(e[1]), bz(e[2]));
    let ax = Vec3::Z;
    let cowl = SKIN.pat(pat::NACELLE);
    let mut m = Mesh::default();
    // Inlet: the acoustic liner inside, the polished lip, the fan cowl outside.
    m.lathe(&LINER, c, ax, &[[1.195, 1.30], [1.19, 1.0], [1.16, 0.6], [1.10, 0.22], [1.088, 0.12]], 48, true);
    m.lathe(&mt(mat::ALU, 0.80, 0.81, 0.83), c, ax, &[[1.088, 0.12], [1.10, 0.035], [1.135, 0.004], [1.18, 0.0], [1.225, 0.025], [1.255, 0.09], [1.27, 0.20]], 48, true);
    m.lathe(&cowl, c, ax, &[[1.27, 0.20], [1.285, 0.45], [1.292, 0.9], [1.292, 1.9]], 48, true);
    // Behind the fan: the dark duct and the core's intake.
    m.disc(&DARK, c + ax * 1.48, Vec3::NEG_Z, Vec3::Y, 1.19, 40);
    // The core's splitter lip, and the ring of outlet guide vanes behind the fan.
    m.lathe(&mt(mat::ALU, 0.6, 0.61, 0.63), c, ax, &[[0.70, 1.47], [0.68, 1.40], [0.655, 1.385], [0.63, 1.40], [0.62, 1.47]], 40, true);
    let ogv = mt(mat::SATIN, 0.22, 0.22, 0.23);
    for k in 0..44 {
        let a = (k as f32 + 0.5) / 44.0 * TAU;
        let rot = Quat::from_rotation_z(a);
        let (rad, tang) = (rot * Vec3::Y, rot * Vec3::NEG_X);
        // Cambered to turn the swirl back to axial: leading edge angled with the flow.
        for (flip, off) in [(false, 0.004f32), (true, -0.004)] {
            m.patch(&ogv, 2, 4, flip, |u, v| {
                let r = 0.70 + (1.185 - 0.70) * u;
                let z = 1.28 + 0.14 * v;
                let sw = 0.05 * (1.0 - v) * (1.0 - v);
                (c + rad * r + tang * (sw + off) + ax * z, [r, z], 0.0)
            });
        }
    }
    // The reverser cascades the sleeve uncovers, and the fan duct's inner wall.
    m.lathe(&DARK, c, ax, &[[1.27, 1.88], [1.27, 2.5]], 48, false);
    m.lathe(&mt(mat::SATIN, 0.15, 0.15, 0.16), c, ax, &[[1.15, 2.95], [1.17, 2.2], [1.19, 1.48]], 48, true);
    // Core cowl, primary nozzle and plug.
    m.lathe(&mt(mat::SATIN, 0.42, 0.43, 0.45), c, ax, &[[0.62, 1.5], [0.86, 2.2], [0.88, 2.97], [0.82, 3.8], [0.70, 4.8], [0.56, 5.94]], 40, true);
    m.lathe(&NOZZLE, c, ax, &[[0.56, 5.94], [0.53, 5.96], [0.50, 5.80]], 40, false);
    m.lathe(&NOZZLE, c, ax, &[[0.47, 5.70], [0.47, 5.94], [0.38, 6.4], [0.20, 6.95], [0.02, 7.27], [0.0, 7.28]], 40, true);
    // Cowl latches along the bottom, and the pylon.
    for k in 0..5 {
        let t = 0.5 + k as f32 * 0.3;
        m.rbox(&mt(mat::SATIN, 0.55, 0.55, 0.56), c + Vec3::new(0.0, -1.292, t), Vec3::new(0.06, 0.006, 0.03), 0.004, Quat::IDENTITY);
    }
    // The pylon: from the top of the nacelle up into the wing, swept forward
    // over the fan cowl and back along the core to under the wing.
    let wk = wing_sect(e[0]);
    let wing_lo = f.pt(e[0], 0.25, -1.0, -1.0).y;
    let (y0, y1) = (c.y + 1.10, wing_lo + 0.35);
    let (f0, f1) = (c.z + 0.85, wk.le.z - 0.6);
    let (b0, b1) = (c.z + 6.0, wk.le.z + 0.62 * wk.c);
    let pylon = Foil {
        at: Box::new(move |s: f32| {
            let u = s.clamp(0.0, 1.0);
            let le = Vec3::new(e[0], y0 + (y1 - y0) * u, f0 + (f1 - f0) * u);
            let c = (b0 + (b1 - b0) * u) - le.z;
            Sect { le, c, cdir: Vec3::Z, up: Vec3::X, tf: 1.0 }
        }),
        t: 0.11,
        camber: 0.0,
    };
    pylon.skin(&mut m, &SKIN.pat(pat::NACELLE), 0.0, 1.0, 10, &|_| 0.0, &|_| 1.0, 18, false);
    // The reverser sleeve.
    let mut rev = Mesh::default();
    rev.lathe(&cowl, c, ax, &[[1.292, 1.9], [1.29, 2.3], [1.265, 2.7], [1.225, 2.97]], 48, true);
    rev.lathe(&mt(mat::SATIN, 0.5, 0.5, 0.52), c, ax, &[[1.225, 2.97], [1.20, 2.98], [1.15, 2.95]], 48, false);
    // The fan: spinner and 38 blades.
    let mut fan = Mesh::default();
    // (tip first, so the lathe's normals face out)
    let sp: Vec<[f32; 2]> = (0..=14).map(|k| { let u = k as f32 / 14.0; [0.44 * (1.0 - (1.0 - u).powf(2.2)).max(0.0).sqrt(), 0.72 + 0.42 * u] }).collect();
    fan.lathe(&mt(mat::SATIN, 0.20, 0.20, 0.21).pat(pat::SPINNER), c, ax, &sp, 32, true);
    fan.disc(&mt(mat::SATIN, 0.20, 0.20, 0.21), c + ax * 1.14, Vec3::Z, Vec3::Y, 0.44, 32);
    for k in 0..38 {
        let a0 = k as f32 / 38.0 * TAU;
        let rot = Quat::from_rotation_z(a0);
        let blade = Foil {
            at: Box::new(move |r: f32| {
                // Stagger from about 30° at the hub to 62° at the tip.
                let beta = (30.0 + 32.0 * ((r - 0.44) / 0.74).clamp(0.0, 1.0)).to_radians();
                let radial = rot * Vec3::Y;
                let tang = rot * Vec3::NEG_X;
                let cdir = (Vec3::Z * beta.cos() + tang * beta.sin()).normalize();
                let up = (tang * beta.cos() - Vec3::Z * beta.sin()).normalize();
                let ch = 0.22 + 0.06 * ((r - 0.44) / 0.74).clamp(0.0, 1.0);
                Sect { le: c + Vec3::Z * 1.06 + radial * r - cdir * (ch * 0.5), c: ch, cdir, up, tf: 1.0 - 0.5 * ((r - 0.44) / 0.74).clamp(0.0, 1.0) }
            }),
            t: 0.06,
            camber: 0.03,
        };
        blade.skin(&mut fan, &TITANIUM, 0.45, 1.175, 6, &|_| 0.0, &|_| 1.0, 6, false);
    }
    (m, rev, fan, c)
}

fn engines(b: &mut Builder) {
    let f = wing_foil();
    const N: [[&str; 2]; 6] = [
        ["left inboard nacelle", "right inboard nacelle"], ["left inboard reverser", "right inboard reverser"], ["left inboard fan", "right inboard fan"],
        ["left outboard nacelle", "right outboard nacelle"], ["left outboard reverser", "right outboard reverser"], ["left outboard fan", "right outboard fan"],
    ];
    for (k, e) in ENGINES.iter().enumerate() {
        let (m, rev, fan, c) = engine(*e, &f);
        let g = [G_LWING, G_RWING];
        b.pair(N[k * 3], g, m, |_| Anim::Fixed, Vec3::ZERO, Vec3::X, [None, None]);
        b.pair(N[k * 3 + 1], g, rev, |_| Anim::slide(Drive::new(Input::Reverser, 0.55), Vec3::Z), Vec3::ZERO, Vec3::X, [None, None]);
        b.pair(N[k * 3 + 2], g, fan, |_| Anim::hinge(Drive::new(Input::Prop, -1.0)), c, Vec3::Z, [None, None]);
    }
}

// ---------------------------------------------------------------- tail

/// Horizontal stabiliser leading and trailing edges and its mean height at span station x.
fn stab_le(x: f32) -> f32 {
    60.3 + 0.92 * (x - 3.4)
}
fn stab_te(x: f32) -> f32 {
    67.95 + 0.27 * (x - 3.4)
}
const STAB_SPAN: f32 = 11.085;
fn stab_hinge(x: f32) -> f32 {
    0.707 + (0.80 - 0.707) * ((x - 3.4) / 5.6).clamp(0.0, 1.3)
}
/// The fin: root at 9.2 m (on the crown), 10.16 m tall.
const FIN_ROOT_H: f32 = 9.2;
const FIN_H: f32 = 10.16;

fn tail(b: &mut Builder) {
    let stab = Foil {
        at: Box::new(|s: f32| {
            let x = s.abs();
            let mut le = bz(stab_le(x));
            let mut c = stab_te(x) - stab_le(x);
            let mut tf = 1.0 - 0.2 * (x / STAB_SPAN);
            if x > STAB_SPAN - 0.2 {
                let u = ((x - (STAB_SPAN - 0.2)) / 0.2).min(1.0);
                let k = 1.0 - (1.0 - u * u).max(0.0).sqrt();
                le += c * 0.25 * k;
                c -= c * 0.3 * k;
                tf *= (1.0 - u * u).max(0.05).sqrt();
            }
            let y = by(7.65 + 0.1228 * (x - 3.4));
            Sect { le: Vec3::new(s, y, le), c, cdir: Vec3::Z, up: Vec3::new(-0.122 * s.signum(), 0.992, 0.0), tf }
        }),
        t: 0.10,
        camber: 0.0,
    };
    let mut m = Mesh::default();
    let grey = GREY_SKIN.pat(pat::WING);
    // Right half then left, each from inside the tail cone out to the tip.
    for sgn in [1.0f32, -1.0] {
        let (s0, s1) = (0.0, STAB_SPAN * sgn);
        let (a, bb) = if sgn > 0.0 { (s0, s1) } else { (s1, s0) };
        stab.skin(&mut m, &grey, a, bb, 40, &|_| 0.0, &|s| stab_hinge(s.abs()) - 0.004, 24, false);
        stab.web(&mut m, &WELL, a, bb, 20, &|s| stab_hinge(s.abs()) - 0.004);
    }
    let ctrl = GREY_SKIN.pat(pat::CTRL);
    let mut elevators = Vec::new();
    for (x0, x1) in [(1.1f32, 6.5f32), (6.55, 10.9)] {
        let mut el = Mesh::default();
        stab.skin(&mut el, &ctrl, x0, x1, 16, &|s| stab_hinge(s) + 0.004, &|_| 1.0, 12, true);
        stab.cap(&mut el, &ctrl, x0, stab_hinge(x0) + 0.004, 1.0, true);
        stab.cap(&mut el, &ctrl, x1, stab_hinge(x1) + 0.004, 1.0, true);
        let p0 = stab.pt(x0, stab_hinge(x0), 0.0, -1.0);
        let p1 = stab.pt(x1, stab_hinge(x1), 0.0, -1.0);
        if x1 > 10.0 {
            for x in [8.0f32, 9.3, 10.5] {
                wick(&mut el, stab.pt(x, 1.0, 0.0, -1.0), Vec3::Z);
            }
        }
        elevators.push((el, p0, p1 - p0));
    }

    let fin = Foil {
        at: Box::new(|s: f32| {
            let u = (s / FIN_H).clamp(0.0, 1.0);
            let mut le = 23.04 + 12.36 * u;
            let te = 35.07 + 4.03 * u;
            let mut c = te - le;
            let mut tf = 1.0 - 0.1 * u;
            if s > FIN_H - 0.15 {
                let v = ((s - (FIN_H - 0.15)) / 0.15).min(1.0);
                let k = 1.0 - (1.0 - v * v).max(0.0).sqrt();
                le += c * 0.08 * k;
                c -= c * 0.1 * k;
                tf *= (1.0 - v * v).max(0.05).sqrt();
            }
            Sect { le: Vec3::new(0.0, by(FIN_ROOT_H) + s, le), c, cdir: Vec3::Z, up: Vec3::X, tf }
        }),
        t: 0.10,
        camber: 0.0,
    };
    let fh = |_: f32| 0.707;
    let mut fm = Mesh::default();
    let tl = SKIN.pat(pat::TAIL);
    fin.skin(&mut fm, &tl, -0.6, 0.9, 4, &|_| 0.0, &|_| 1.0, 26, false);
    fin.skin(&mut fm, &tl, 0.9, 9.7, 36, &|_| 0.0, &|s| fh(s) - 0.004, 26, false);
    fin.skin(&mut fm, &tl, 9.7, FIN_H, 4, &|_| 0.0, &|_| 1.0, 26, false);
    fin.web(&mut fm, &WELL, 0.9, 9.7, 20, &|s| fh(s) - 0.004);
    fin.cap(&mut fm, &WELL, 0.9, fh(0.9) - 0.004, 1.0, false);
    fin.cap(&mut fm, &WELL, 9.7, fh(9.7) - 0.004, 1.0, false);
    fin.cap(&mut fm, &SKIN.pat(pat::TAIL), FIN_H, 0.0, 1.0, false);
    wick(&mut fm, fin.pt(FIN_H - 0.25, 1.0, 0.0, -1.0), Vec3::Z);
    // A dorsal fillet blending the fin's root into the crown.
    let fillet = Foil {
        at: Box::new(|s: f32| {
            let u = (s / 0.9).clamp(0.0, 1.0);
            let le = bz(52.2) + (23.04 + 0.9 * 12.36 / FIN_H - bz(52.2)) * u.sqrt();
            Sect { le: Vec3::new(0.0, by(FIN_ROOT_H) - 0.3 + s, le), c: bz(60.0) - le, cdir: Vec3::Z, up: Vec3::X, tf: 1.0 }
        }),
        t: 0.13,
        camber: 0.0,
    };
    fillet.skin(&mut fm, &SKIN.pat(pat::TAIL), 0.0, 0.9, 8, &|_| 0.0, &|_| 1.0, 16, false);
    // Spars and ribs for when they break.
    let mut ts = Mesh::default();
    structure(&stab, &mut ts, -10.5, 10.5, &[0.15, 0.62], &[(-9.0, 0.03, 0.66), (-6.0, 0.03, 0.66), (-3.0, 0.03, 0.66), (3.0, 0.03, 0.66), (6.0, 0.03, 0.66), (9.0, 0.03, 0.66)]);
    structure(&fin, &mut ts, 0.2, 9.8, &[0.15, 0.6], &[(2.0, 0.03, 0.68), (4.5, 0.03, 0.68), (7.0, 0.03, 0.68)]);
    let k = b.fixed("tail structure", G_TAIL, ts);
    b.role(k, Role::TailStructure);
    let k = b.fixed("stabiliser", G_TAIL, m);
    b.role(k, Role::Stab);
    let k = b.fixed("fin", G_TAIL, fm);
    b.role(k, Role::Stab);
    for (i, (el, p0, axis)) in elevators.into_iter().enumerate() {
        let names = [["left inboard elevator", "right inboard elevator"], ["left outboard elevator", "right outboard elevator"]];
        let (l, r) = b.pair(names[i], [G_TAIL, G_TAIL], el, |_| Anim::hinge(Drive::new(Input::Elevator, -0.40)), p0, axis, [None, None]);
        b.role(l, Role::Elevator).role(r, Role::Elevator);
    }
    // Upper and lower rudder.
    for (s0, s1, name) in [(0.9f32, 5.0f32, "lower rudder"), (5.05, 9.65, "upper rudder")] {
        let mut ru = Mesh::default();
        fin.skin(&mut ru, &SKIN.pat(pat::CTRL), s0, s1, 16, &|s| fh(s) + 0.004, &|_| 1.0, 12, true);
        fin.cap(&mut ru, &SKIN.pat(pat::CTRL), s0, fh(s0) + 0.004, 1.0, true);
        fin.cap(&mut ru, &SKIN.pat(pat::CTRL), s1, fh(s1) + 0.004, 1.0, true);
        if s1 > 9.0 {
            for s in [6.6f32, 7.9, 9.2] {
                wick(&mut ru, fin.pt(s, 1.0, 0.0, -1.0), Vec3::Z);
            }
        }
        let p0 = fin.pt(s0, 0.707, 0.0, -1.0);
        let p1 = fin.pt(s1, 0.707, 0.0, -1.0);
        let k = b.add(name, G_TAIL, ru, Anim::hinge(Drive::new(Input::Rudder, 0.45)), p0, p1 - p0, None);
        b.role(k, Role::Rudder);
    }
}

// ---------------------------------------------------------------- landing gear

/// A truck's four wheels on two axles: the front pair, the rear pair (each one mesh).
fn wheel_pair(axle: Vec3, half_w: f32, rim: f32) -> Mesh {
    let mut m = Mesh::default();
    for sx in [-1.0f32, 1.0] {
        let c = axle + Vec3::X * (sx * TRUCK_W * 0.5);
        wheel(&mut m, c, Vec3::X * sx, TYRE_R, half_w, rim, true);
    }
    m.tube(&STRUT, &[axle - Vec3::X * (TRUCK_W * 0.5 - 0.2), axle + Vec3::X * (TRUCK_W * 0.5 - 0.2)], &|_| 0.08, 12, true);
    // Carbon brake stacks inboard of each wheel: the housing with its ring of
    // pistons, the heat-tinted stack showing through the hub.
    let heat = mt(mat::STEEL, 0.32, 0.29, 0.27).x(2.0);
    for sx in [-1.0f32, 1.0] {
        let inner = axle + Vec3::X * (sx * (TRUCK_W * 0.5 - half_w));
        m.tube(&heat, &[inner, inner - Vec3::X * (sx * 0.07)], &|_| rim * 0.82, 24, true);
        m.tube(&STRUT, &[inner - Vec3::X * (sx * 0.07), inner - Vec3::X * (sx * 0.11)], &|_| rim * 0.6, 20, true);
        for k in 0..8 {
            let a = k as f32 / 8.0 * TAU;
            let o = Vec3::new(0.0, a.cos(), a.sin()) * (rim * 0.68);
            m.tube(&STRUT, &[inner + o - Vec3::X * (sx * 0.07), inner + o - Vec3::X * (sx * 0.12)], &|_| 0.022, 8, true);
        }
    }
    m
}

/// A main gear leg (right side) and its truck. `twist` is the extra tilt the
/// truck takes to stow (rad), `leg` its index among the legs.
#[allow(clippy::too_many_arguments)]
fn main_gear(b: &mut Builder, names: [[&'static str; 2]; 5], pivot: Vec3, axle: Vec3, stowed: Vec3, legs: [u8; 2], extra: f32, slot: &[Vec3]) {
    let (axis, ang) = stow(pivot, axle, stowed);
    let down = (axle - pivot).normalize();
    let len = (axle - pivot).length();
    // The leg: the outer cylinder from the trunnion, its side brace, and the
    // door that covers its slot once it is stowed.
    let mut leg = Mesh::default();
    let top = pivot - down * 0.1;
    let low = pivot + down * (len - 1.25);
    leg.tube(&STRUT, &[top, low], &|t| 0.24 - 0.04 * t, 18, true);
    leg.tube(&STEEL, &[pivot - axis * 0.45, pivot + axis * 0.45], &|_| 0.13, 12, true);
    let brace_a = pivot + down * (len * 0.45);
    let side = axis.cross(down).normalize();
    leg.tube(&STRUT, &[brace_a, pivot + side * 1.1 - down * 0.15], &|_| 0.08, 10, true);
    // The brace's lock link, and the hydraulic lines clamped down the leg.
    leg.tube(&STEEL, &[brace_a + side * 0.55 - down * 0.08, pivot + down * (len * 0.25) + side * 0.05], &|_| 0.035, 8, true);
    let q = (axis * 0.8 + side * 0.6).normalize();
    for (k, o) in [0.0f32, 0.05].iter().enumerate() {
        let off = q * (0.255 + o * 0.6) + axis.cross(q) * (o - 0.025);
        leg.tube(if k == 0 { &HOSE } else { &PIPE }, &[top + off * 1.08, low + off * 0.92], &|_| 0.013, 8, true);
    }
    for t in [0.2f32, 0.45, 0.7, 0.92] {
        let p = top.lerp(low, t);
        let rr = 0.24 - 0.04 * t;
        leg.lathe(&STRUT, p, down, &[[rr + 0.005, -0.025], [rr + 0.035, -0.02], [rr + 0.035, 0.02], [rr + 0.005, 0.025]], 18, false);
    }
    // The slot door: flush with the skin when stowed, so modelled there and
    // carried back to where it sits with the gear down.
    if slot.len() >= 4 {
        let back = Quat::from_axis_angle(axis, -ang);
        let mut door = Mesh::default();
        let n = 12;
        door.patch(&GREY_SKIN, n, 2, true, |a, c| {
            let p = slot[0].lerp(slot[1], a).lerp(slot[3].lerp(slot[2], a), c);
            let q = Vec3::new(p.x, skin_bottom(p.z, p.x) - 0.006, p.z);
            (pivot + back * (q - pivot), [q.z, q.x], 0.0)
        });
        door.patch(&WELL, n, 2, false, |a, c| {
            let p = slot[0].lerp(slot[1], a).lerp(slot[3].lerp(slot[2], a), c);
            let q = Vec3::new(p.x, skin_bottom(p.z, p.x) + 0.03, p.z);
            (pivot + back * (q - pivot), [q.z, q.x], 0.0)
        });
        // Its arms to the leg.
        for t in [0.25f32, 0.75] {
            let p = slot[0].lerp(slot[1], t).lerp(slot[3].lerp(slot[2], t), 0.5);
            let q = Vec3::new(p.x, skin_bottom(p.z, p.x) + 0.02, p.z);
            let at = pivot + back * (q - pivot);
            let on_leg = pivot + down * (at - pivot).dot(down).clamp(0.2, len - 1.0);
            door.tube(&STRUT, &[at, on_leg], &|_| 0.04, 8, true);
        }
        leg.append(&door);
    }
    let stow_d = Drive::new(Input::Gear, -ang).smooth(0.15, 0.85).offset(ang);
    let (ll, lr) = b.pair(names[0], [G_GEAR, G_GEAR], leg, |_| Anim::hinge(stow_d), pivot, axis, [None, None]);
    b.role(ll, Role::Gear(legs[0])).role(lr, Role::Gear(legs[1]));
    // The oleo: the chromed piston sliding up into the cylinder as it takes the weight.
    let mut oleo = Mesh::default();
    oleo.tube(&CHROME, &[low - down * 0.3, axle - down * 0.12], &|_| 0.16, 16, true);
    oleo.rbox(&STRUT, axle - down * 0.05, Vec3::new(0.22, 0.14, 0.26), 0.05, Quat::from_rotation_arc(Vec3::Y, -down));
    // Torque links in front of the strut.
    let fwd = Vec3::NEG_Z;
    let mid = (low + axle) * 0.5 + fwd * 0.28;
    oleo.tube(&STRUT, &[low + fwd * 0.12, mid], &|_| 0.05, 8, true);
    oleo.tube(&STRUT, &[mid, axle - down * 0.25 + fwd * 0.1], &|_| 0.045, 8, true);
    oleo.tube(&STEEL, &[mid - Vec3::X * 0.07, mid + Vec3::X * 0.07], &|_| 0.03, 10, true);
    // The brake hoses looping down to the truck.
    let q = (axis * 0.8 + side * 0.6).normalize();
    oleo.tube(&HOSE, &[low + q * 0.22 - down * 0.25, low + q * 0.30 + down * 0.25, axle - down * 0.3 + q * 0.2], &|_| 0.016, 8, true);
    let up = -down;
    let (ol, or) = b.pair(names[1], [G_GEAR, G_GEAR], oleo, |s| Anim::slide(Drive::new(Input::Compress(if s < 0.0 { legs[0] } else { legs[1] }), 1.0), up * Vec3::new(s, 1.0, 1.0)), axle, Vec3::X, [Some(ll), Some(lr)]);
    b.role(ol, Role::Gear(legs[0])).role(or, Role::Gear(legs[1]));
    // The truck beam, tilted aft-down in the air (and further to stow).
    let mut truck = Mesh::default();
    truck.tube(&STRUT, &[axle + Vec3::Z * (TRUCK_L * 0.5 + 0.12), axle - Vec3::Z * (TRUCK_L * 0.5 + 0.12)], &|_| 0.13, 14, true);
    // Brake rods and the pitch trimmer.
    truck.tube(&STEEL, &[axle + Vec3::new(0.0, 0.35, -0.15), axle + Vec3::new(0.0, 0.12, -TRUCK_L * 0.5)], &|_| 0.035, 8, true);
    for sx in [-1.0f32, 1.0] {
        truck.tube(&PIPE, &[axle + Vec3::new(sx * 0.11, 0.10, -TRUCK_L * 0.5 + 0.1), axle + Vec3::new(sx * 0.11, 0.14, 0.0), axle + Vec3::new(sx * 0.11, 0.10, TRUCK_L * 0.5 - 0.1)], &|_| 0.012, 6, true);
        for dz in [-TRUCK_L * 0.5, TRUCK_L * 0.5] {
            truck.tube(&HOSE, &[axle + Vec3::new(sx * 0.11, 0.10, dz * 0.85), axle + Vec3::new(sx * (TRUCK_W * 0.5 - 0.30), 0.12, dz)], &|_| 0.012, 6, true);
        }
    }
    // Axle-end fittings with the towing lugs.
    for dz in [-TRUCK_L * 0.5 - 0.12, TRUCK_L * 0.5 + 0.12] {
        truck.lathe(&STEEL, axle + Vec3::Z * dz, Vec3::Z * dz.signum(), &[[0.13, 0.0], [0.10, 0.04], [0.0, 0.05]], 14, true);
    }
    let tilt = |s: f32| {
        let k = if s < 0.0 { legs[0] } else { legs[1] };
        Anim::hinge2(Drive::new(Input::Compress(k), -TILT).smooth(0.0, 0.05).offset(TILT), Drive::new(Input::Gear, -extra).smooth(0.86, 0.98).offset(extra))
    };
    let (tl, tr) = b.pair(names[2], [G_GEAR, G_GEAR], truck, tilt, axle, Vec3::X, [Some(ol), Some(or)]);
    b.role(tl, Role::Gear(legs[0])).role(tr, Role::Gear(legs[1]));
    // The wheels, a pair on each axle.
    for (k, dz) in [(3usize, -TRUCK_L * 0.5), (4, TRUCK_L * 0.5)] {
        let a = axle + Vec3::Z * dz;
        let wm = wheel_pair(a, 0.24, 0.28);
        let spin = |s: f32| Anim::hinge(Drive::new(Input::Wheel(if s < 0.0 { legs[0] } else { legs[1] }), -1.0));
        let (wl, wr) = b.pair(names[k], [G_GEAR, G_GEAR], wm, spin, a, Vec3::X, [Some(tl), Some(tr)]);
        b.role(wl, Role::Gear(legs[0])).role(wr, Role::Gear(legs[1]));
    }
}

/// A door over a well (plan box), following the skin, hinged along its inner edge.
fn well_door(bx: [f32; 4], gap: f32) -> (Mesh, Vec3, Vec3) {
    let mut m = Mesh::default();
    let at = |a: f32, c: f32, lift: f32| -> SP {
        let z = bx[0] + gap + (bx[1] - bx[0] - 2.0 * gap) * a;
        let x = bx[2] + gap + (bx[3] - bx[2] - 2.0 * gap) * c;
        (Vec3::new(x, skin_bottom(z, x) + lift, z), [z, x], 0.0)
    };
    m.patch(&GREY_SKIN, 16, 8, true, |a, c| at(a, c, -0.004));
    m.patch(&WELL, 16, 8, false, |a, c| at(a, c, 0.025));
    let edge = |m: &mut Mesh, f: &dyn Fn(f32) -> (f32, f32)| {
        let rows: Vec<Vec<SP>> = (0..=16).map(|i| { let (a, c) = f(i as f32 / 16.0); vec![at(a, c, -0.004), at(a, c, 0.025)] }).collect();
        m.grid(&WELL, &rows, false, false);
    };
    edge(&mut m, &|t| (t, 0.0));
    edge(&mut m, &|t| (t, 1.0));
    edge(&mut m, &|t| (0.0, t));
    edge(&mut m, &|t| (1.0, t));
    let h0 = at(0.0, 0.0, 0.0).0;
    let h1 = at(1.0, 0.0, 0.0).0;
    (m, h0, h1 - h0)
}

fn gear(b: &mut Builder) {
    // Nose gear: forward-retracting, steering, two wheels.
    let (axis, ang) = stow(NOSE_PIVOT, NOSE_AXLE, NOSE_STOWED);
    let down = (NOSE_AXLE - NOSE_PIVOT).normalize();
    let len = (NOSE_AXLE - NOSE_PIVOT).length();
    let mut nl = Mesh::default();
    let low = NOSE_PIVOT + down * (len - 1.05);
    nl.tube(&STRUT, &[NOSE_PIVOT - down * 0.08, low], &|t| 0.17 - 0.02 * t, 16, true);
    nl.tube(&STEEL, &[NOSE_PIVOT - Vec3::X * 0.4, NOSE_PIVOT + Vec3::X * 0.4], &|_| 0.1, 12, true);
    // The drag brace forward to the well's front, and the taxi lights on the strut.
    nl.tube(&STRUT, &[NOSE_PIVOT + down * 1.1, NOSE_PIVOT + Vec3::new(0.0, 0.05, -1.45)], &|_| 0.07, 10, true);
    for sx in [-0.12f32, 0.12] {
        let c = NOSE_PIVOT + down * 1.35 + Vec3::new(sx, 0.0, -0.2);
        nl.rbox(&STRUT, c, Vec3::new(0.06, 0.07, 0.06), 0.02, Quat::IDENTITY);
        nl.disc(&mt(mat::LAMP, 0.95, 0.92, 0.85).x(lamp::LANDING), c + Vec3::new(0.0, 0.0, -0.062), Vec3::NEG_Z, Vec3::Y, 0.05, 14);
    }
    for o in [0.0f32, 0.04] {
        let q = Vec3::new(0.19, 0.0, 0.0) + Vec3::Z * (0.08 + o);
        nl.tube(&HOSE, &[NOSE_PIVOT + down * 0.1 + q, low + q * 0.95], &|_| 0.011, 8, true);
    }
    nl.lathe(&STRUT, low - down * 0.05, down, &[[0.17, -0.03], [0.21, -0.025], [0.21, 0.025], [0.17, 0.03]], 18, false);
    let leg = b.add("nose leg", G_GEAR, nl, Anim::hinge(Drive::new(Input::Gear, -ang).smooth(0.15, 0.85).offset(ang)), NOSE_PIVOT, axis, None);
    b.role(leg, Role::Gear(0));
    // Steering collar, then the oleo, then the wheels.
    let mut st = Mesh::default();
    st.lathe(&STEEL, low + down * 0.05, down, &[[0.22, 0.0], [0.22, 0.12], [0.18, 0.15]], 18, false);
    st.tube(&STRUT, &[low + down * 0.2 + Vec3::Z * 0.2, low + down * 0.55 + Vec3::Z * 0.32, NOSE_AXLE - down * 0.3 + Vec3::Z * 0.18], &|_| 0.045, 8, true);
    let steer = b.add("nose steering", G_GEAR, st, Anim::hinge(Drive::new(Input::Steer, 1.0)), NOSE_AXLE, down, Some(leg));
    b.role(steer, Role::Gear(0));
    let mut ol = Mesh::default();
    ol.tube(&CHROME, &[low - down * 0.25, NOSE_AXLE - down * 0.1], &|_| 0.12, 14, true);
    ol.tube(&STRUT, &[NOSE_AXLE - Vec3::X * 0.25, NOSE_AXLE + Vec3::X * 0.25], &|_| 0.09, 12, true);
    let oleo = b.add("nose oleo", G_GEAR, ol, Anim::slide(Drive::new(Input::Compress(0), 1.0), -down), NOSE_AXLE, Vec3::X, Some(steer));
    b.role(oleo, Role::Gear(0));
    let mut nw = Mesh::default();
    for sx in [-1.0f32, 1.0] {
        wheel(&mut nw, NOSE_AXLE + Vec3::X * (sx * NOSE_TRACK), Vec3::X * sx, TYRE_R, 0.215, 0.254, false);
    }
    let k = b.add("nose wheels", G_GEAR, nw, Anim::hinge(Drive::new(Input::Wheel(0), -1.0)), NOSE_AXLE, Vec3::X, Some(oleo));
    b.role(k, Role::Gear(0));
    // Nose gear doors: the long forward ones close again behind the gear, the
    // aft ones on the strut's sides stay open while it is down.
    let [z0, z1, _, hw] = NOSE_WELL;
    let zm = z1 - 1.25;
    for (a, c, anim) in [(z0, zm, Drive::new(Input::Gear, 1.45).pulse(0.0, 0.1, 0.9, 1.0)), (zm, z1, Drive::new(Input::Gear, 1.45).smooth(0.0, 0.1))] {
        let mut dr = Mesh::default();
        let at = |u: f32, v: f32, lift: f32| -> SP {
            let z = a + 0.01 + (c - a - 0.02) * u;
            let x = 0.004 + (hw - 0.01) * v;
            (Vec3::new(x, skin_bottom(z, x) + lift, z), [z, x], 0.0)
        };
        dr.patch(&SKIN, 12, 6, true, |u, v| at(u, v, -0.004));
        dr.patch(&WELL, 12, 6, false, |u, v| at(u, v, 0.02));
        let h0 = Vec3::new(hw, skin_bottom(a, hw), a);
        let h1 = Vec3::new(hw, skin_bottom(c, hw), c);
        let names = if a == z0 { ["left nose door", "right nose door"] } else { ["left aft nose door", "right aft nose door"] };
        b.pair(names, [G_ENGINE, G_ENGINE], dr, move |_| Anim::hinge(anim), h0, h1 - h0, [None, None]);
    }

    // Wing gear: swinging inboard, the truck tilted near upright first so it
    // lies across the belly once stowed.
    let [a, c] = WING_SLOT;
    let d = Vec2::new(c[0] - a[0], c[1] - a[1]).normalize();
    let n = Vec2::new(-d.y, d.x) * WING_SLOT_W;
    let slot = [Vec3::new(a[1] + n.y, 0.0, a[0] + n.x), Vec3::new(c[1] + n.y, 0.0, c[0] + n.x), Vec3::new(c[1] - n.y, 0.0, c[0] - n.x), Vec3::new(a[1] - n.y, 0.0, a[0] - n.x)];
    main_gear(
        b,
        [["left wing gear leg", "right wing gear leg"], ["left wing gear oleo", "right wing gear oleo"], ["left wing gear truck", "right wing gear truck"], ["left wing gear front wheels", "right wing gear front wheels"], ["left wing gear rear wheels", "right wing gear rear wheels"]],
        WING_PIVOT,
        WING_AXLE,
        WING_STOWED,
        [1, 2],
        1.35,
        &slot,
    );
    // Body gear: swinging forward, the truck kept level as it goes.
    let (_, body_ang) = stow(BODY_PIVOT, BODY_AXLE, BODY_STOWED);
    let bs = BODY_SLOT;
    let slot = [Vec3::new(bs[2], 0.0, bs[1]), Vec3::new(bs[2], 0.0, bs[0]), Vec3::new(bs[3], 0.0, bs[0]), Vec3::new(bs[3], 0.0, bs[1])];
    main_gear(
        b,
        [["left body gear leg", "right body gear leg"], ["left body gear oleo", "right body gear oleo"], ["left body gear truck", "right body gear truck"], ["left body gear front wheels", "right body gear front wheels"], ["left body gear rear wheels", "right body gear rear wheels"]],
        BODY_PIVOT,
        BODY_AXLE,
        BODY_STOWED,
        [3, 4],
        -(body_ang + TILT),
        &slot,
    );
    // The well doors: closed with the gear up and down, open while it moves.
    for (bx, names, open) in [(WING_WELL, ["left wing gear door", "right wing gear door"], -1.5f32), (BODY_WELL, ["left body gear door", "right body gear door"], -1.45)] {
        let (m, h0, ax) = well_door(bx, 0.01);
        b.pair(names, [G_GEAR, G_GEAR], m, |_| Anim::hinge(Drive::new(Input::Gear, open).pulse(0.0, 0.1, 0.9, 1.0)), h0, ax, [None, None]);
    }
}

// ---------------------------------------------------------------- the spec

const KT: f64 = 1.0 / 1.94384;

const fn main_leg(x: f64, z: f64, name: &'static str, kick: f64) -> Leg {
    Leg {
        at: DVec3::new(x, -5.27, z),
        wheel_r: TYRE_R as f64,
        travel: 0.5,
        k: 2.6e6,
        c: 4.5e5,
        nf_max: 8.0e6,
        steer: 0.0,
        brakes: true,
        side_k: 8.0e5,
        side_mu: 0.8,
        roll_k: 1.5e4,
        roll_mu: 0.015,
        brake_k: 4.0e5,
        brake_mu: 0.5,
        fold_vn: 5.0,
        fold_kick: DVec3::new(0.3, 0.0, kick),
        name,
    }
}

pub static SPEC: Spec = Spec {
    id: 1,
    name: "Boeing 747-400",
    keys: &["747", "b747", "747-400", "jumbo"],
    blurb: "Four-engine wide-body airliner, 290 t. Heavy and slow to answer the controls; Mach 0.85 cruise, 150 kt approach. / speedbrake, Z reversers.",
    still: include_bytes!("../../../assets/aircraft/747.png"),
    build,
    groups: [("fuselage", 120_000.0), ("tail", 12_000.0), ("left wing", 40_000.0), ("right wing", 40_000.0), ("flight deck", 8_000.0), ("radome", 300.0), ("gear", 14_000.0)],
    span: SPAN,
    wing_sect,
    fuse: |z| {
        let s = sec(z);
        (s.top, s.mid, s.bot)
    },
    aileron: [IB_AIL[0], OB_AIL[1]],
    tank: TANK,
    tail: Tail {
        fin_y0: by(FIN_ROOT_H),
        fin_h: FIN_H,
        rudder_y0: by(FIN_ROOT_H) + 0.9,
        rudder_h: 8.75,
        stab_root: 1.5,
        elev_root: 1.1,
        break_z: [bz(55.8), bz(60.0)],
        fin_base: DVec3::new(0.0, by(FIN_ROOT_H) as f64, bz(61.0) as f64),
        fin_tip: DVec3::new(0.0, by(FIN_ROOT_H + FIN_H) as f64, bz(69.0) as f64),
        stab_tip: DVec3::new(STAB_SPAN as f64 - 0.1, by(7.65 + 0.1228 * (STAB_SPAN - 3.4)) as f64, bz(68.7) as f64),
    },
    eye: EYE,
    sees_inside: deck::sees_inside,
    contacts: &[
        ([0.0, 0.3, -31.62], Part::Nose),
        ([0.0, -2.0, -29.0], Part::Nose),
        ([0.0, 3.6, -27.0], Part::Nose),
        ([-1.9, -0.3, -28.5], Part::Nose),
        ([1.9, -0.3, -28.5], Part::Nose),
        ([0.0, -3.0, -20.0], Part::Belly),
        ([0.0, -3.1, -8.0], Part::Belly),
        ([0.0, -3.12, 0.0], Part::Belly),
        ([0.0, -2.95, 10.0], Part::Belly),
        ([-3.25, 0.7, -20.0], Part::Side),
        ([3.25, 0.7, -20.0], Part::Side),
        ([-3.25, 0.7, 0.0], Part::Side),
        ([3.25, 0.7, 0.0], Part::Side),
        ([-3.2, 0.9, 13.0], Part::Side),
        ([3.2, 0.9, 13.0], Part::Side),
        ([0.0, 5.05, -20.0], Part::Roof),
        ([0.0, 4.2, 0.0], Part::Roof),
        ([0.0, 4.2, 15.0], Part::Roof),
        ([-11.68, -3.78, -6.9], Part::Pod(0)),
        ([11.68, -3.78, -6.9], Part::Pod(1)),
        ([-21.0, -2.86, 2.3], Part::Pod(0)),
        ([21.0, -2.86, 2.3], Part::Pod(1)),
    ],
    tail_pts: &[[0.0, -2.77, 14.0], [0.0, -0.8, 24.0], [0.0, 1.14, 33.0], [0.0, 3.8, 38.3]],
    wing_st: [3.6 / SPAN, 9.0 / SPAN, 15.0 / SPAN, 21.0 / SPAN, 27.0 / SPAN],
    hard: [
        DVec3::new(0.0, 0.3, -31.65),
        DVec3::new(-32.0, 1.4, 16.3),
        DVec3::new(32.0, 1.4, 16.3),
        DVec3::new(0.0, 3.8, 38.3),
        DVec3::new(0.0, 14.36, 37.5),
        DVec3::new(0.0, -3.12, 0.0),
        DVec3::new(11.68, -3.78, -6.5),
    ],
    gear: &[
        Leg {
            at: DVec3::new(0.0, -5.15, -23.85),
            wheel_r: TYRE_R as f64,
            travel: 0.40,
            k: 1.5e6,
            c: 2.6e5,
            nf_max: 4.0e6,
            steer: 1.0,
            brakes: false,
            side_k: 3.0e5,
            side_mu: 0.8,
            roll_k: 1.0e4,
            roll_mu: 0.015,
            brake_k: 0.0,
            brake_mu: 0.0,
            fold_vn: 3.8,
            fold_kick: DVec3::new(-0.4, 0.0, 0.0),
            name: "NOSE GEAR",
        },
        main_leg(-5.5, 0.22, "LEFT WING GEAR", 0.3),
        main_leg(5.5, 0.22, "RIGHT WING GEAR", -0.3),
        main_leg(-1.92, 3.29, "LEFT BODY GEAR", 0.2),
        main_leg(1.92, 3.29, "RIGHT BODY GEAR", -0.2),
    ],
    hull: Hull { k: 1.6e7, c: 2.4e6, nf_max: 3.2e7, drag: 8.0e5 },
    hole: Hole { fus: [3.25, 4.1, 1.0, 2.5], wing: [-1.3, 0.093, 0.9, 0.25], stab: [3.0, 0.35, 11.1], fin: [4.2, 10.16, 0.6, 0.2], pods: [[11.68, -2.5, 1.3], [21.0, -1.58, 1.3]] },
    // A mid-weight: about 290 t (maximum takeoff weight is 396.9 t, landing 285.8 t).
    mass: 290_000.0,
    // Pitch, yaw, roll (Heffley & Jewell, the 747 at 288 t).
    inertia: DVec3::new(44.9e6, 67.4e6, 24.7e6),
    wing_area: 541.2,
    b: 64.4,
    cl0: 0.22,
    cl_a: 5.5,
    cd0: 0.0175,
    k_ind: 0.052,
    cd_gear: 0.012,
    cd_flap: [0.03, 0.045],
    flap_cl: 0.85,
    stall_alpha: 0.20,
    stall_flap: 0.055,
    stall_slat: 0.035,
    alpha_trim: 0.20,
    mcrit: 0.78,
    cd_wave: 0.0,
    spoiler_cd: 0.06,
    spoiler_cl: 0.45,
    flaps: &[0.0, 1.0, 5.0, 10.0, 20.0, 25.0, 30.0],
    vfe: &[1e9, 280.0 * KT, 260.0 * KT, 240.0 * KT, 230.0 * KT, 205.0 * KT, 180.0 * KT],
    flap_rate: 1.0 / 40.0,
    slats_at: 1,
    gear_rate: 1.0 / 16.0,
    // VMO 365 kt, MMO 0.92.
    vne: 365.0 * KT,
    mmo: 0.92,
    flutter_v: 420.0 * KT,
    // Limit load 2.5 g, ultimate 3.75 g.
    g_limit: 2.6,
    g_break: 3.8,
    fbw_g: 0.0,
    steer_max: 1.0,
    steer_fade: 12.0,
    // Four PW4056s, 252 kN (56,750 lbf) each.
    engine: Engine::Fan(Fan { n: 4, thrust: 252_000.0, idle: 0.25, idle_thrust: 0.035, n1_rpm: 3600.0, spool_up: 0.6, spool_down: 0.5, reverse: 0.35, reverser_time: 2.0, boost: 1.08, mach_lapse: [0.78, 0.38] }),
    handling: Handling {
        q_ref: 11_000.0,
        a_max: 1.6,
        damp_v: 68.0,
        ad_min: 0.3,
        ad_max: 1.6,
        pitch_k: 3.0,
        pitch_k_lost: 1.0,
        pitch_ctl: 0.25,
        alpha_ctl: 0.2,
        pitch_d: 5.5,
        roll_ctl: 0.6,
        roll_d: 2.75,
        roll_beta: 0.25,
        roll_clean: 0.5,
        yaw_ctl: 0.18,
        yaw_beta: 1.2,
        yaw_d: 2.5,
        dihedral: 0.08,
        adverse: 0.1,
        stall_pitch: 0.3,
        stall_drop: 0.5,
        stall_yaw: 0.8,
    },
    cruise: 128.6,
    approach: (80.0, 6),
    cam_dist: 85.0,
    shadow_r: 38.0,
    sound: Sound::Jet,
    size_k: 1.0,
    speed_k: 1.0,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_builds_whole() {
        let t0 = std::time::Instant::now();
        let m = build();
        let nv: usize = m.pieces.iter().map(|p| p.mesh.v.len()).sum();
        let (c, h) = m.bounds();
        println!("747: {} pieces, {nv} vertices, {:.1} s, bounds centre {c:?} half {h:?}", m.pieces.len(), t0.elapsed().as_secs_f64());
        for p in &m.pieces {
            assert!(p.mesh.v.iter().all(|v| v.pos.iter().chain(v.nrm.iter()).all(|x| x.is_finite())), "{} has NaNs", p.name);
            assert!(!p.mesh.i.is_empty(), "{} is empty", p.name);
        }
        // Overall: 70.67 m long, 64.44 m across the winglets, 19.4 m to the top of the fin.
        let len = h.z * 2.0;
        let span = h.x * 2.0;
        let top = c.y + h.y + YG;
        println!("length {len:.2} m, span {span:.2} m, height {top:.2} m");
        assert!((len - 70.67).abs() < 0.8, "{len}");
        assert!((span - 64.44).abs() < 0.8, "{span}");
        assert!((top - 19.4).abs() < 0.5, "{top}");
    }

    #[test]
    fn gear_stows_inside_the_skin() {
        // Every stowed wheel centre is above the belly skin under it.
        for (axle, stowed) in [(WING_AXLE, WING_STOWED), (BODY_AXLE, BODY_STOWED), (NOSE_AXLE, NOSE_STOWED)] {
            let skin = skin_bottom(stowed.z, stowed.x);
            println!("stowed at {stowed:?}: skin {skin:.2} (from {axle:?})");
            assert!(stowed.y - 0.3 > skin);
            assert!(well(Vec3::new(stowed.x, skin + 0.01, stowed.z)) < 0.0);
        }
    }
}

#[cfg(test)]
mod pose_tests {
    use super::*;

    #[test]
    fn slot_doors_close_flush() {
        let m = build();
        let mut p = crate::plane::Plane::spawn(&SPEC, 0.6, -2.1, 1000.0, 0.0, 120.0);
        p.gear = 0.0;
        let pose = p.pose(&m.pieces);
        for (i, pc) in m.pieces.iter().enumerate() {
            if pc.name != "right body gear leg" && pc.name != "right wing gear leg" {
                continue;
            }
            // The door's outer face: the grey-paint vertices.
            let mut worst: f32 = 0.0;
            let mut n = 0;
            for v in &pc.mesh.v {
                if v.mat[0] != mat::LIVERY {
                    continue;
                }
                let q = pose[i].transform_point3(Vec3::from(v.pos));
                let s = skin_bottom(q.z, q.x);
                worst = worst.max((q.y - s).abs());
                n += 1;
            }
            println!("{}: {n} door vertices, worst {worst:.3} m off the skin; axis {:?}", pc.name, pc.axis);
            assert!(n > 0 && worst < 0.05);
        }
    }
}

