//! A Cessna 172-class four-seat, high-wing single with retractable gear
//! (180 hp, about 1100 kg, 16.2 m² of strut-braced wing): a lofted
//! superellipse fuselage, NACA-section wings and tail, a twisted propeller.
//! The cockpit lives in `cabin.rs`.
pub mod cabin;

use crate::aircraft::{Engine, Handling, Hole, Hull, Leg, Part, Piston, Sound, Spec, Tail};
use crate::airframe::*;
use glam::{DVec3, Mat4, Quat, Vec2, Vec3};
use std::f32::consts::{FRAC_PI_2, PI, TAU};

pub fn build() -> Model {
    let mut b = Builder::default();
    fuselage(&mut b);
    wings(&mut b);
    tail(&mut b);
    engine(&mut b);
    gear(&mut b);
    cabin::build(&mut b);
    bake_ao(&mut b.pieces, &AoBake { vox: 0.025, reach: 2.6, rays: 28, open: &|q, pat| window(q) < 0.0 || (well_cut_by(pat) && well(q) < 0.0) });
    Model::finish(b.pieces)
}

const KT: f64 = 1.0 / 1.94384;
const V_NONE: f64 = 1e9;

const fn leg(at: DVec3, r: f32, travel: f64, steer: f64, fold_vn: f64, fold_kick: DVec3, name: &'static str) -> Leg {
    Leg { at, wheel_r: r as f64, travel, k: 70_000.0, c: 7_000.0, nf_max: 120_000.0, steer, brakes: true, side_k: 5000.0, side_mu: 0.9, roll_k: 80.0, roll_mu: 0.025, brake_k: 2500.0, brake_mu: 0.55, fold_vn, fold_kick, name }
}

pub static SPEC: Spec = Spec {
    id: 0,
    name: "Cessna 172RG",
    keys: &["c172", "172", "cessna"],
    blurb: "Four-seat single, 180 hp, retractable gear. Forgiving and slow: 125 kt flat out.",
    still: include_bytes!("../../../assets/aircraft/c172.png"),
    build,
    groups: [("fuselage", 420.0), ("tail", 150.0), ("left wing", 170.0), ("right wing", 170.0), ("engine", 260.0), ("propeller", 30.0), ("gear", 60.0)],
    span: SPAN,
    wing_sect,
    fuse: |z| {
        let s = sec(z);
        (s.top, s.mid, s.bot)
    },
    aileron: AILERON,
    tank: TANK,
    tail: Tail {
        fin_y0: 0.27,
        fin_h: 1.53,
        rudder_y0: 0.22,
        rudder_h: 1.57,
        stab_root: 0.34,
        elev_root: 0.12,
        break_z: [1.4, 2.7],
        fin_base: DVec3::new(0.0, 0.27, 4.35),
        fin_tip: DVec3::new(0.0, 1.80, 4.1),
        stab_tip: DVec3::new(1.65, 0.11, 4.15),
    },
    eye: EYE,
    sees_inside: cabin::sees_inside,
    contacts: &[
        ([0.0, -0.035, -3.17], Part::Nose),
        ([0.0, -0.60, -2.6], Part::Nose),
        ([0.0, 0.31, -2.45], Part::Nose),
        ([-0.5, -0.1, -2.6], Part::Nose),
        ([0.5, -0.1, -2.6], Part::Nose),
        ([0.0, -0.72, -1.5], Part::Belly),
        ([0.0, -0.73, 0.0], Part::Belly),
        ([0.0, -0.70, 0.9], Part::Belly),
        ([-0.58, 0.05, -1.2], Part::Side),
        ([0.58, 0.05, -1.2], Part::Side),
        ([-0.58, 0.05, 0.4], Part::Side),
        ([0.58, 0.05, 0.4], Part::Side),
        ([0.0, 0.52, -1.35], Part::Roof),
        ([0.0, 0.50, 1.6], Part::Roof),
    ],
    tail_pts: &[[0.0, -0.45, 2.4], [0.0, -0.12, 3.6], [0.0, 0.19, 4.62], [0.0, 0.34, 3.2]],
    wing_st: [0.45 / SPAN, 1.5 / SPAN, 2.6 / SPAN, 3.7 / SPAN, 4.7 / SPAN],
    hard: [
        DVec3::new(0.0, -0.03, -3.62),
        DVec3::new(-5.5, 0.82, -0.2),
        DVec3::new(5.5, 0.82, -0.2),
        DVec3::new(0.0, 0.19, 4.62),
        DVec3::new(0.0, 1.83, 4.1),
        DVec3::new(0.0, -0.73, 0.0),
        DVec3::new(0.0, -1.0, -3.30),
    ],
    gear: &[
        leg(DVec3::new(0.0, -1.50, -2.87), NOSE_R, 0.12, 1.0, 4.5, DVec3::new(-0.6, 0.0, 0.0), "NOSE GEAR"),
        leg(DVec3::new(-1.25, -1.50, 0.30), MAIN_R, 0.10, 0.0, 6.5, DVec3::new(0.4, 0.0, 0.5), "LEFT MAIN GEAR"),
        leg(DVec3::new(1.25, -1.50, 0.30), MAIN_R, 0.10, 0.0, 6.5, DVec3::new(0.4, 0.0, -0.5), "RIGHT MAIN GEAR"),
    ],
    hull: Hull { k: 60_000.0, c: 9_000.0, nf_max: 120_000.0, drag: 3000.0 },
    hole: Hole { fus: [0.82, 0.92, -0.04, 0.5], wing: [0.72, 0.03, 0.36, 0.24], stab: [0.2, 0.2, 1.75], fin: [0.3, 1.55, 0.2, 0.14], pods: [[0.0; 3]; 2] },
    // Two aboard and most of the fuel (maximum takeoff weight is 1157 kg).
    mass: 1100.0,
    // Published figures for a 172.
    inertia: DVec3::new(1825.0, 2667.0, 1285.0),
    wing_area: 16.2,
    b: 11.0,
    // Aspect ratio 7.5: a strut-braced high wing, Cd0 ~0.034 cleaned up; the
    // big slotted flaps are mostly drag past 20 degrees.
    cl0: 0.25,
    cl_a: 5.0,
    cd0: 0.034,
    k_ind: 0.055,
    cd_gear: 0.009,
    cd_flap: [0.02, 0.06],
    flap_cl: 0.55,
    stall_alpha: 0.27,
    stall_flap: 0.04,
    stall_slat: 0.0,
    alpha_trim: 0.24,
    mcrit: 0.0,
    cd_wave: 0.0,
    spoiler_cd: 0.0,
    spoiler_cl: 0.0,
    flaps: &[0.0, 10.0, 20.0, 30.0],
    // 110 kt for the first notch, 85 kt beyond.
    vfe: &[V_NONE, 110.0 * KT, 85.0 * KT, 85.0 * KT],
    flap_rate: 1.0 / 7.0,
    slats_at: 0,
    gear_rate: 1.0 / 6.0,
    // 163 kt indicated; flutter sets in past 185.
    vne: 83.9,
    mmo: 0.0,
    flutter_v: 95.0,
    // Limit load is 4.4 g; hold more than that and the spar fatigues, 6.6 g snaps it outright.
    g_limit: 4.6,
    g_break: 6.6,
    fbw_g: 0.0,
    steer_max: 0.45,
    steer_fade: 0.0,
    // 180 hp at 2700 rpm turning a 76 in fixed-pitch propeller.
    engine: Engine::Piston(Piston {
        power: 134_000.0,
        w_rated: 2700.0 * std::f64::consts::TAU / 60.0,
        prop_d: 1.93,
        ct0: 0.096,
        cp0: 0.059,
        j_t: 1.05,
        j_p: 1.5,
        prop_i: 3.0,
        static_thrust: 2500.0,
        fric0: 15.0,
        fric1: 35.0,
        blade_break_y: PROP_HUB.y + 0.42,
    }),
    handling: Handling {
        q_ref: 2200.0,
        a_max: 2.6,
        damp_v: 68.0,
        ad_min: 0.3,
        ad_max: 2.6,
        pitch_k: 7.0,
        pitch_k_lost: 2.0,
        pitch_ctl: 0.6,
        alpha_ctl: 0.26,
        pitch_d: 3.2,
        roll_ctl: 4.0,
        roll_d: 4.0,
        roll_beta: 0.8,
        roll_clean: 1.0,
        yaw_ctl: 1.3,
        yaw_beta: 4.5,
        yaw_d: 2.4,
        dihedral: 0.3,
        adverse: 0.35,
        stall_pitch: 0.9,
        stall_drop: 1.8,
        stall_yaw: 2.5,
    },
    cruise: 55.0,
    approach: (36.0, 2),
    cam_dist: 16.0,
    shadow_r: 6.2,
    sound: Sound::Piston,
    size_k: 1.0,
    speed_k: 1.0,
};

// ---------------------------------------------------------------- fuselage

/// Fuselage cross-section keys: station z, half width, top, waist height,
/// bottom, squareness of the upper and the lower half.
const FUSE: [[f32; 7]; 20] = [
    [-3.17, 0.215, 0.175, -0.035, -0.245, 2.0, 2.0],
    [-3.14, 0.300, 0.215, -0.040, -0.330, 2.2, 2.2],
    [-3.08, 0.385, 0.245, -0.050, -0.440, 2.4, 2.4],
    [-2.98, 0.445, 0.268, -0.060, -0.540, 2.6, 2.7],
    [-2.75, 0.505, 0.292, -0.070, -0.610, 2.8, 3.0],
    [-2.40, 0.538, 0.310, -0.075, -0.650, 3.0, 3.2],
    [-2.05, 0.556, 0.322, -0.075, -0.680, 3.2, 3.4],
    [-1.72, 0.570, 0.335, -0.065, -0.705, 3.4, 3.6],
    [-1.35, 0.580, 0.500, -0.050, -0.720, 3.7, 3.6],
    [-0.97, 0.585, 0.655, -0.040, -0.730, 4.2, 3.6],
    [-0.60, 0.585, 0.640, -0.040, -0.730, 4.4, 3.6],
    [0.50, 0.585, 0.640, -0.040, -0.730, 4.4, 3.6],
    [0.85, 0.565, 0.625, -0.045, -0.725, 4.0, 3.5],
    [1.30, 0.520, 0.560, -0.050, -0.715, 3.4, 3.2],
    [1.80, 0.450, 0.480, -0.045, -0.620, 3.0, 2.9],
    [2.40, 0.360, 0.405, -0.020, -0.450, 2.6, 2.6],
    [3.20, 0.245, 0.335, 0.040, -0.220, 2.4, 2.4],
    [4.00, 0.135, 0.300, 0.100, -0.040, 2.2, 2.2],
    [4.45, 0.065, 0.280, 0.150, 0.060, 2.0, 2.0],
    [4.62, 0.020, 0.262, 0.170, 0.120, 2.0, 2.0],
];
pub const NOSE_Z: f32 = -3.17;
pub const TAIL_Z: f32 = 4.62;
pub const FIREWALL_Z: f32 = -2.05;

#[derive(Clone, Copy, Debug)]
pub struct Sec {
    pub w: f32,
    pub top: f32,
    pub mid: f32,
    pub bot: f32,
    pub nt: f32,
    pub nb: f32,
}

/// Monotone cubic through the keys (no overshoot between stations).
fn pchip(col: usize, z: f32) -> f32 {
    let k = &FUSE;
    let n = k.len();
    if z <= k[0][0] {
        return k[0][col];
    }
    if z >= k[n - 1][0] {
        return k[n - 1][col];
    }
    let i = (0..n - 1).find(|&i| z < k[i + 1][0]).unwrap_or(n - 2);
    let h = |i: usize| k[i + 1][0] - k[i][0];
    let d = |i: usize| (k[i + 1][col] - k[i][col]) / h(i);
    let slope = |i: usize| -> f32 {
        if i == 0 || i == n - 1 {
            return if i == 0 { d(0) } else { d(n - 2) };
        }
        let (a, b) = (d(i - 1), d(i));
        if a * b <= 0.0 {
            0.0
        } else {
            let (w1, w2) = (2.0 * h(i) + h(i - 1), h(i) + 2.0 * h(i - 1));
            (w1 + w2) / (w1 / a + w2 / b)
        }
    };
    let t = (z - k[i][0]) / h(i);
    let (y0, y1, m0, m1) = (k[i][col], k[i + 1][col], slope(i) * h(i), slope(i + 1) * h(i));
    let (t2, t3) = (t * t, t * t * t);
    (2.0 * t3 - 3.0 * t2 + 1.0) * y0 + (t3 - 2.0 * t2 + t) * m0 + (-2.0 * t3 + 3.0 * t2) * y1 + (t3 - t2) * m1
}
pub fn sec(z: f32) -> Sec {
    Sec { w: pchip(1, z), top: pchip(2, z), mid: pchip(3, z), bot: pchip(4, z), nt: pchip(5, z), nb: pchip(6, z) }
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
/// Angle on the section at z whose point is at height y on the right side.
pub fn fuse_th(z: f32, y: f32) -> f32 {
    let s = sec(z);
    let (mut a, mut b) = (0.0f32, PI);
    for _ in 0..30 {
        let m = (a + b) * 0.5;
        if sec_pt(&s, m).y > y { a = m } else { b = m }
    }
    (a + b) * 0.5
}
/// Height of the upper skin at station z, lateral position x.
pub fn fuse_top_y(z: f32, x: f32) -> f32 {
    let s = sec(z);
    let x = x.abs().min(s.w * 0.999);
    let (mut a, mut b) = (0.0, FRAC_PI_2);
    for _ in 0..30 {
        let m = (a + b) * 0.5;
        if sec_pt(&s, m).x < x { a = m } else { b = m }
    }
    sec_pt(&s, (a + b) * 0.5).y
}
/// Height of the lower skin at station z, lateral position x.
pub fn fuse_bottom_y(z: f32, x: f32) -> f32 {
    let s = sec(z);
    let x = x.abs().min(s.w * 0.999);
    let (mut a, mut b) = (FRAC_PI_2, PI);
    for _ in 0..30 {
        let m = (a + b) * 0.5;
        if sec_pt(&s, m).x > x { a = m } else { b = m }
    }
    sec_pt(&s, (a + b) * 0.5).y
}
/// Height of a well roof at (z, x): `top`, but never closer than 5 cm to the skin.
fn roof_y(z: f32, x: f32, top: f32) -> f32 {
    top.max(fuse_bottom_y(z, x) + 0.05)
}
/// Walls rising from an opening in the lower skin, outline in (z, x, roof
/// height of the edge that starts there).
fn well_walls(m: &mut Mesh, outline: &[(f32, f32, f32)]) {
    for k in 0..outline.len() {
        let (a, b) = (outline[k], outline[(k + 1) % outline.len()]);
        let n = 12;
        let rows: Vec<Vec<SP>> = (0..=n)
            .map(|i| {
                let t = i as f32 / n as f32;
                let (z, x) = (a.0 + (b.0 - a.0) * t, a.1 + (b.1 - a.1) * t);
                let ys = fuse_bottom_y(z, x) - 0.004;
                let yt = roof_y(z, x, a.2);
                vec![(Vec3::new(x, ys, z), [z, ys], 0.0), (Vec3::new(x, yt, z), [z, yt], 0.0)]
            })
            .collect();
        m.grid(&PRIMER, &rows, false, false);
    }
}
/// A roof over a convex opening, outline in (z, x).
fn well_roof(m: &mut Mesh, outline: &[(f32, f32)], top: f32) {
    let mut ring = Vec::new();
    for k in 0..outline.len() {
        let (a, b) = (outline[k], outline[(k + 1) % outline.len()]);
        for i in 0..8 {
            let t = i as f32 / 8.0;
            let (z, x) = (a.0 + (b.0 - a.0) * t, a.1 + (b.1 - a.1) * t);
            ring.push(Vec3::new(x, roof_y(z, x, top), z));
        }
    }
    let c = ring.iter().copied().sum::<Vec3>() / ring.len() as f32;
    m.fan(&PRIMER, c, &ring, Vec3::NEG_Y);
}
/// x span of a convex outline in (z, x) at station z.
pub fn span_at(outline: &[[f32; 2]], z: f32) -> (f32, f32) {
    let (mut lo, mut hi) = (f32::MAX, f32::MIN);
    for k in 0..outline.len() {
        let (a, b) = (outline[k], outline[(k + 1) % outline.len()]);
        if (a[0] - z) * (b[0] - z) <= 0.0 && a[0] != b[0] {
            let x = a[1] + (b[1] - a[1]) * (z - a[0]) / (b[0] - a[0]);
            lo = lo.min(x);
            hi = hi.max(x);
        }
    }
    (lo, hi)
}
/// A gear door filling a convex opening between stations z0 and z1, flush
/// with the skin, `gap` short of the edges: painted outside, primer inside.
fn gear_door(outline: &[[f32; 2]], z0: f32, z1: f32, gap: f32) -> Mesh {
    let mut m = Mesh::default();
    let at = |a: f32, c: f32, lift: f32| -> SP {
        let z = z0 + (z1 - z0) * a;
        let (lo, hi) = span_at(outline, z);
        let x = lo + gap + (hi - lo - 2.0 * gap) * c;
        (Vec3::new(x, fuse_bottom_y(z, x) + lift, z), [z, x], 0.0)
    };
    m.patch(&WHITE, 16, 8, true, |a, c| at(a, c, -0.002));
    m.patch(&PRIMER, 16, 8, false, |a, c| at(a, c, 0.004));
    // Thin edges all round, so the door has a thickness when it opens.
    let edge = |m: &mut Mesh, f: &dyn Fn(f32) -> (f32, f32)| {
        let rows: Vec<Vec<SP>> = (0..=16).map(|i| { let (a, c) = f(i as f32 / 16.0); vec![at(a, c, -0.002), at(a, c, 0.004)] }).collect();
        m.grid(&PRIMER, &rows, false, false);
    };
    edge(&mut m, &|t| (t, 0.0));
    edge(&mut m, &|t| (t, 1.0));
    edge(&mut m, &|t| (0.0, t));
    edge(&mut m, &|t| (1.0, t));
    m
}
/// Outward normal of the fuselage skin at (z, th).
pub fn fuse_n(z: f32, th: f32) -> Vec3 {
    let e = 1e-3;
    let du = fuse_pt(z + e, th) - fuse_pt(z - e, th);
    let dv = fuse_pt(z, th + e) - fuse_pt(z, th - e);
    // (Down the right side as th grows, aft as z grows: their cross product
    // points out. It used to be the other way round, which sank the door
    // handles, fasteners, steps and plates placed with `on_side` into the skin.)
    du.cross(dv).normalize()
}
/// Point on the skin at station z, height y, on side sx (+1 right), lifted by `off`.
pub fn on_side(z: f32, y: f32, sx: f32, off: f32) -> (Vec3, Vec3) {
    let th = fuse_th(z, y);
    let (mut p, mut n) = (fuse_pt(z, th), fuse_n(z, th));
    p.x *= sx;
    n.x *= sx;
    (p + n * off, n)
}

/// Signed distance to the nearest window opening (negative inside), body frame.
/// Keep in step with `window()` in plane.wgsl.
pub fn window(p: Vec3) -> f32 {
    let zy = Vec2::new(p.z, p.y);
    let ws = [Vec2::new(-1.72, 0.31), Vec2::new(-1.52, 0.07), Vec2::new(-1.03, 0.64), Vec2::new(-1.0, 0.80), Vec2::new(-1.75, 0.80)];
    let mut d = sd_poly(zy, &ws) + 0.01;
    if p.x.abs() > 0.3 && p.y < 0.6 {
        d = d.min(sd_poly(zy, &v2([[-1.36, 0.12], [-0.60, 0.12], [-0.60, 0.48], [-0.98, 0.48]])) - 0.04);
        d = d.min(sd_poly(zy, &v2([[-0.42, 0.12], [0.62, 0.12], [0.42, 0.48], [-0.42, 0.48]])) - 0.04);
    }
    if p.y > 0.35 {
        d = d.min(sd_poly(Vec2::new(p.z, p.x), &v2([[0.84, -0.30], [1.48, -0.17], [1.48, 0.17], [0.84, 0.30]])) - 0.04);
    }
    d
}
/// Nose gear bay in the belly, (z, x): it runs back under the footwell hump.
pub const NOSE_BAY: [f32; 3] = [-3.0, -1.915, 0.135];
/// Main gear bays, right side, (z, x): a trough the leg folds into, from the
/// slot at its root aft to the bay the wheel lies in.
pub const MAIN_TROUGH: [[f32; 2]; 4] = [[0.215, 0.37], [1.15, 0.183], [1.15, 0.427], [0.215, 0.55]];
pub const MAIN_BAY: [[f32; 2]; 5] = [[1.14, 0.03], [1.545, 0.03], [1.545, 0.43], [1.27, 0.505], [1.14, 0.44]];
/// Roof heights over the trough and the wheel bay, and where they meet.
pub const TROUGH_TOP: f32 = -0.47;
pub const BAY_TOP: f32 = -0.25;
pub const BAY_Z: f32 = 1.14;
/// The trough door starts behind the slot the leg passes through when down.
pub const TROUGH_DOOR_Z: f32 = 0.372;

/// Signed distance to the landing gear wells in the skin (5 mm inside their
/// walls). Keep in step with plane.wgsl.
pub fn well(p: Vec3) -> f32 {
    let mut d = 1.0f32;
    if p.y < -0.40 {
        let [z0, z1, hw] = NOSE_BAY;
        d = d.min(sd_poly(Vec2::new(p.z, p.x), &v2([[z0, -hw], [z1, -hw], [z1, hw], [z0, hw]])) + 0.005);
        let q = Vec2::new(p.z, p.x.abs());
        let bay = MAIN_BAY.map(|v| Vec2::new(v[0], v[1]));
        d = d.min(sd_poly(q, &v2(MAIN_TROUGH)).min(sd_poly(q, &bay)) + 0.005);
    }
    d
}


// Landing gear geometry (body frame, right side for the mains).
pub const NOSE_PIVOT: Vec3 = Vec3::new(0.0, -0.50, -2.95);
pub const NOSE_AXLE: Vec3 = Vec3::new(0.0, -1.335, -2.87);
pub const NOSE_R: f32 = 0.165;
/// how far the nose leg swings back to retract (rad)
pub const NOSE_FOLD: f32 = 1.487;
pub const MAIN_ROOT: Vec3 = Vec3::new(0.45, -0.60, 0.30);
pub const MAIN_AXLE: Vec3 = Vec3::new(1.25, -1.31, 0.30);
pub const MAIN_R: f32 = 0.19;
/// where a main wheel's centre ends up with the gear retracted
pub const MAIN_STOWED: Vec3 = Vec3::new(0.235, -0.46, 1.338);
pub const PROP_HUB: Vec3 = Vec3::new(0.0, -0.03, -3.30);
/// the pilot's eye
pub const EYE: Vec3 = Vec3::new(-0.27, 0.46, -0.52);

// ---------------------------------------------------------------- fuselage

fn fuselage(b: &mut Builder) {
    // Stations: finer around the nose bowl and the tail cap.
    let mut zs = Vec::new();
    let mut z = NOSE_Z;
    while z < TAIL_Z - 0.005 {
        zs.push(z);
        z += if z < -3.0 { 0.012 } else if z > 4.4 { 0.02 } else { 0.035 };
    }
    zs.push(TAIL_Z);
    for split in [FIREWALL_Z, 1.0] {
        let k = zs.iter().position(|&z| z > split).unwrap();
        zs[k - 1] = split;
    }
    const NT: usize = 72; // around one side, top to bottom
    let rows: Vec<Vec<SP>> = zs
        .iter()
        .map(|&z| {
            let s = sec(z);
            let mut arc = 0.0;
            let mut prev = sec_pt(&s, 0.0);
            (0..=NT)
                .map(|k| {
                    let th = k as f32 / NT as f32 * PI;
                    let p = sec_pt(&s, th);
                    arc += (p - prev).length();
                    prev = p;
                    (Vec3::new(p.x, p.y, z), [z, arc], 0.0)
                })
                .collect()
        })
        .collect();
    let mut full = Mesh::default();
    full.grid(&WHITE, &rows, false, false);
    let nc = NT + 1;
    let row_of = |z: f32| zs.iter().position(|&q| (q - z).abs() < 1e-6).unwrap();
    let sub = |r0: usize, r1: usize, pattern: f32| -> Mesh {
        let mut m = Mesh::default();
        m.v = full.v[r0 * nc..(r1 + 1) * nc].to_vec();
        for v in &mut m.v {
            v.mat[1] = pattern;
        }
        for i in r0..r1 {
            for j in 0..nc - 1 {
                let a = ((i - r0) * nc + j) as u32;
                let c = a + nc as u32;
                m.i.extend([a, c, a + 1, a + 1, c, c + 1]);
            }
        }
        // The left half is the mirror image, its surface coordinate negative.
        let mut l = m.mirror_x();
        for v in &mut l.v {
            v.uv[1] = -v.uv[1];
        }
        m.append(&l);
        m
    };
    let (rf, rt) = (row_of(FIREWALL_Z), row_of(1.0));
    let mut cowl = sub(0, rf, pat::COWL);
    // The engine face behind the spinner.
    cowl.disc(&mt(mat::BLACK, 0.03, 0.03, 0.03), Vec3::new(0.0, -0.035, NOSE_Z + 0.03), Vec3::NEG_Z, Vec3::Y, 0.26, 24);
    let mut cabin = sub(rf, rt, pat::FUSE);
    let mut tailcone = sub(rt, zs.len() - 1, pat::FUSE);
    let s = sec(TAIL_Z);
    let ring: Vec<Vec3> = (0..48).map(|k| { let p = sec_pt(&s, k as f32 / 48.0 * TAU); Vec3::new(p.x, p.y, TAIL_Z) }).collect();
    tailcone.fan(&WHITE.pat(pat::FUSE), Vec3::new(0.0, s.mid, TAIL_Z + 0.004), &ring, Vec3::Z);

    // Gear wells: walls and a roof behind each opening in the skin.
    let [z0, z1, hw] = NOSE_BAY;
    const NOSE_TOP: f32 = -0.31;
    let nose = [(z0, -hw), (z1, -hw), (z1, hw), (z0, hw)];
    well_walls(&mut cowl, &nose.map(|(z, x)| (z, x, NOSE_TOP)));
    well_roof(&mut cowl, &nose, NOSE_TOP);
    let mut bays = Mesh::default();
    let (t, w) = (MAIN_TROUGH, MAIN_BAY);
    let (ti, to) = (span_at(&t, BAY_Z).0, span_at(&t, BAY_Z).1);
    // One outline round the trough and the bay, each edge with its roof height.
    well_walls(&mut bays, &[
        (t[0][0], t[0][1], TROUGH_TOP),
        (BAY_Z, ti, BAY_TOP),
        (w[0][0], w[0][1], BAY_TOP),
        (w[1][0], w[1][1], BAY_TOP),
        (w[2][0], w[2][1], BAY_TOP),
        (w[3][0], w[3][1], BAY_TOP),
        (w[4][0], w[4][1], BAY_TOP),
        (BAY_Z, to, TROUGH_TOP),
        (t[3][0], t[3][1], TROUGH_TOP),
    ]);
    well_roof(&mut bays, &[(t[0][0], t[0][1]), (BAY_Z, ti), (BAY_Z, to), (t[3][0], t[3][1])], TROUGH_TOP);
    well_roof(&mut bays, &w.map(|v| (v[0], v[1])), BAY_TOP);
    // The step up from the trough roof to the bay roof.
    let rows: Vec<Vec<SP>> = (0..=8)
        .map(|i| {
            let x = ti + (to - ti) * i as f32 / 8.0;
            let (a, b) = (roof_y(BAY_Z, x, TROUGH_TOP), roof_y(BAY_Z, x, BAY_TOP));
            vec![(Vec3::new(x, a, BAY_Z), [x, a], 0.0), (Vec3::new(x, b, BAY_Z), [x, b], 0.0)]
        })
        .collect();
    bays.grid(&PRIMER, &rows, false, false);
    cabin.append(&bays.mirror_x());
    cabin.append(&bays);

    // Cowling fasteners along the split line and around the nose bowl and firewall.
    for sx in [-1.0f32, 1.0] {
        let mut z = -2.92;
        while z < -2.1 {
            let (p, n) = on_side(z, -0.085, sx, 0.0005);
            cowl.screw(&SCREW_PAINT, p, n, 0.0065, z * 13.0 + sx);
            z += 0.105;
        }
        for k in 0..9 {
            let y = 0.24 - k as f32 * 0.105;
            for zz in [-2.975, FIREWALL_Z - 0.025] {
                let (p, n) = on_side(zz, y, sx, 0.0005);
                cowl.screw(&SCREW_PAINT, p, n, 0.006, y * 7.0 + zz + sx);
            }
        }
    }
    // Oil filler door on top of the cowl, two quarter-turn latches.
    for x in [-0.07f32, 0.07] {
        let z = -2.36;
        let th = (x / 0.55).asin().abs() * 0.7;
        let p = fuse_pt(z, th);
        cowl.screw(&SCREW_STEEL, Vec3::new(x, p.y + 0.0005, z), Vec3::Y, 0.008, x * 3.0);
    }
    // Rows of screws around the cabin door frames and along the wing root fairing.
    let door: [(f32, f32); 5] = [(-1.47, -0.42), (-0.47, -0.42), (-0.47, 0.59), (-0.99, 0.59), (-1.47, 0.03)];
    for sx in [-1.0f32, 1.0] {
        for k in 0..door.len() {
            let (a, c) = (door[k], door[(k + 1) % door.len()]);
            let len = ((c.0 - a.0).powi(2) + (c.1 - a.1).powi(2)).sqrt();
            let (nz_, ny) = ((c.1 - a.1) / len, -(c.0 - a.0) / len);
            let n = (len / 0.085).round().max(1.0) as usize;
            for i in 0..n {
                let t = (i as f32 + 0.5) / n as f32;
                let (z, y) = (a.0 + (c.0 - a.0) * t - nz_ * 0.022, a.1 + (c.1 - a.1) * t - ny * 0.022);
                let (p, nn) = on_side(z, y, sx, 0.0004);
                if window(p) > 0.015 && y < 0.57 {
                    cabin.screw(&SCREW_PAINT, p, nn, 0.0045, y * 31.0 + z + sx);
                }
            }
        }
        let mut z = -0.95;
        while z < 0.7 {
            let (p, n) = on_side(z, 0.60, sx, 0.0004);
            cabin.screw(&SCREW_PAINT, p, n, 0.0045, z * 17.0 + sx);
            z += 0.08;
        }
    }
    // Inspection plates on the belly and tail cone.
    for &(z, x, r) in &[(2.10f32, 0.0f32, 0.07f32), (3.05, 0.0, 0.055), (-0.30, 0.0, 0.08)] {
        let s = sec(z);
        let p = Vec3::new(x, s.bot - 0.0012, z);
        inspection_plate(if z > 1.0 { &mut tailcone } else { &mut cabin }, p, Vec3::NEG_Y, Vec3::Z, r, 6);
    }
    for sx in [-1.0f32, 1.0] {
        let (p, n) = on_side(2.6, 0.05, sx, 0.0012);
        inspection_plate(&mut tailcone, p, n, Vec3::Y, 0.05, 6);
    }
    // Static port, left side.
    let (p, n) = on_side(0.95, 0.02, -1.0, 0.0);
    cabin.lathe(&ALU, p, n, &[[0.016, 0.0], [0.016, 0.0015], [0.004, 0.0015], [0.004, -0.001], [0.0, -0.001]], 16, false);
    // Door handles and hinges, the fuel-up step and grab handle.
    for sx in [-1.0f32, 1.0] {
        // Below the window (it used to sit on the glass).
        let (p, n) = on_side(-0.62, -0.01, sx, 0.004);
        let rot = Quat::from_rotation_arc(Vec3::X, n);
        cabin.rbox(&CHROME, p, Vec3::new(0.004, 0.012, 0.055), 0.004, rot);
        cabin.rbox(&BLACK, p - n * 0.003, Vec3::new(0.002, 0.02, 0.07), 0.002, rot);
        for y in [-0.30f32, -0.04] {
            let (p, n) = on_side(-1.49, y, sx, 0.006);
            cabin.rbox(&STEEL, p, Vec3::new(0.006, 0.035, 0.012), 0.004, Quat::from_rotation_arc(Vec3::X, n));
            cabin.bolt(&STEEL, p + n * 0.006, n, 0.0035);
        }
        let (p, n) = on_side(-1.88, -0.36, sx, 0.0);
        cowl.tube(&STEEL, &[p - n * 0.02, p + n * 0.06, p + n * 0.08 + Vec3::Y * 0.012], &|_| 0.011, 10, true);
        let (p, n) = on_side(-1.80, 0.12, sx, 0.0);
        let path: Vec<Vec3> = (0..=10).map(|k| { let a = k as f32 / 10.0 * PI; p + n * (0.035 * a.sin()) + Vec3::Y * (0.06 * a.cos()) }).collect();
        cowl.tube(&STEEL, &path, &|_| 0.008, 8, true);
    }

    // Antennas: two com whips on the cabin top, the transponder blade under
    // the belly, the ELT whip and the GPS puck on the tail cone.
    for &(z, x) in &[(1.68f32, 0.0f32), (2.25, 0.0)] {
        let s = sec(z);
        let p = Vec3::new(x, s.top, z);
        tailcone.rbox(&BLACK, p + Vec3::Y * 0.006, Vec3::new(0.02, 0.006, 0.035), 0.004, Quat::IDENTITY);
        let tip = p + Vec3::new(0.0, 0.55, 0.32);
        tailcone.tube(&BLACK, &[p + Vec3::Y * 0.01, tip], &|t| 0.006 - t * 0.0035, 6, true);
        tailcone.dome(&BLACK, tip, tip - p, 0.004, 0.004, 6, 0.0);
    }
    let s = sec(0.15);
    let p = Vec3::new(0.0, s.bot, 0.15);
    let blade = [Vec2::new(-0.03, 0.0), Vec2::new(0.035, 0.0), Vec2::new(0.02, -0.085), Vec2::new(0.0, -0.09)];
    cabin.slab(&BLACK, &blade, &[], p + Vec3::X * 0.003, Vec3::Z, Vec3::Y, 0.006);
    let s = sec(2.9);
    let p = Vec3::new(0.0, s.top, 2.9);
    tailcone.tube(&BLACK, &[p, p + Vec3::new(0.0, 0.38, 0.12)], &|t| 0.004 - t * 0.0025, 6, true);
    let s = sec(3.35);
    tailcone.lathe(&mt(mat::PLASTIC, 0.62, 0.62, 0.60), Vec3::new(0.0, s.top, 3.35), Vec3::Y, &[[0.045, 0.0], [0.045, 0.006], [0.038, 0.014], [0.02, 0.019], [0.0, 0.02]], 20, true);
    // Tail tie-down ring.
    let s = sec(4.25);
    let p = Vec3::new(0.0, s.bot - 0.005, 4.25);
    let ring: Vec<Vec3> = (0..=16).map(|k| { let a = k as f32 / 16.0 * TAU; p + Vec3::new(0.0, -0.03 - 0.03 * a.cos(), 0.03 * a.sin()) }).collect();
    tailcone.tube(&STEEL, &ring, &|_| 0.005, 6, false);

    b.fixed("cowling", G_ENGINE, cowl);
    b.fixed("cabin skin", G_FUSE, cabin);
    let t = b.fixed("tail cone", G_TAIL, tailcone);
    b.role(t, Role::TailCone);
}

// ---------------------------------------------------------------- wings

pub const SPAN: f32 = 5.5;
pub const FLAP: [f32; 2] = [0.62, 3.05];
pub const AILERON: [f32; 2] = [3.07, 5.13];
/// The span the fuel tank in each wing fills.
pub const TANK: [f32; 2] = [0.62, 2.55];

/// Right wing section at span station xs.
pub fn wing_sect(xs: f32) -> Sect {
    let (mut le, mut c) = if xs <= 2.55 { (-0.97, 1.65) } else { let t = (xs - 2.55) / (SPAN - 2.55); let c = 1.65 - 0.50 * t; (0.68 - c, c) };
    let mut tf = 1.0;
    let mut y = 0.70 + 0.0302 * xs;
    if xs > 5.3 {
        // Rounded, slightly drooped tip.
        let u = ((xs - 5.3) / (SPAN - 5.3)).min(1.0);
        let k = 1.0 - (1.0 - u * u).max(0.0).sqrt();
        le += c * 0.3 * k;
        c -= c * 0.3 * k;
        tf = (1.0 - u * u).max(0.0).sqrt();
    }
    if xs > 5.0 {
        y -= 0.05 * ((xs - 5.0) / 0.5).powi(2);
    }
    let inc = (1.5 - 3.0 * xs / SPAN).to_radians();
    let dih = 0.0302f32;
    Sect { le: Vec3::new(xs, y, le), c, cdir: Vec3::new(0.0, -inc.sin(), inc.cos()), up: Vec3::new(-dih, inc.cos(), inc.sin()).normalize(), tf }
}
pub fn wing_foil() -> Foil {
    Foil { at: Box::new(wing_sect), t: 0.12, camber: 0.02 }
}
/// Point and outward normal on a foil.
pub fn foil_frame(f: &Foil, s: f32, xc: f32, side: f32) -> (Vec3, Vec3) {
    let p = f.pt(s, xc, side, -1.0);
    let ds = f.pt(s + 0.002, xc, side, -1.0) - f.pt(s - 0.002, xc, side, -1.0);
    let dx = f.pt(s, xc + 0.002, side, -1.0) - f.pt(s, (xc - 0.002).max(0.0), side, -1.0);
    let n = (ds.cross(dx) * side).normalize();
    (p, n)
}

fn wings(b: &mut Builder) {
    let f = wing_foil();
    let skin = WHITE.pat(pat::WING);
    let ctrl = WHITE.pat(pat::CTRL);
    let mut w = Mesh::default();
    let one = |_: f32| 1.0;
    let zero = |_: f32| 0.0;
    f.skin(&mut w, &skin, 0.0, FLAP[0], 6, &zero, &one, 28, false);
    f.skin(&mut w, &skin, FLAP[0], FLAP[1], 24, &zero, &|_| 0.70, 28, false);
    f.skin(&mut w, &skin, FLAP[1], AILERON[1], 22, &zero, &|_| 0.75, 28, false);
    f.skin(&mut w, &skin, AILERON[1], SPAN, 18, &zero, &one, 28, false);
    f.web(&mut w, &PRIMER, FLAP[0], FLAP[1], 12, &|_| 0.70);
    f.web(&mut w, &PRIMER, FLAP[1], AILERON[1], 12, &|_| 0.75);
    f.cap(&mut w, &PRIMER, FLAP[0], 0.70, 1.0, false);
    f.cap(&mut w, &PRIMER, FLAP[1], 0.70, 0.75, false);
    f.cap(&mut w, &PRIMER, AILERON[1], 0.75, 1.0, false);

    // Fuel filler, sump drain, inspection plates, tie-down ring.
    let (p, n) = foil_frame(&f, 1.55, 0.30, 1.0);
    w.lathe(&ALU, p, n, &[[0.05, -0.002], [0.05, 0.002], [0.042, 0.004], [0.0, 0.004]], 28, false);
    w.lathe(&mt(mat::ALU, 0.5, 0.5, 0.52), p + n * 0.004, n, &[[0.036, 0.0], [0.036, 0.006], [0.03, 0.009], [0.0, 0.009]], 24, false);
    let tab = n.any_orthonormal_vector();
    w.rbox(&mt(mat::PAINT, 0.5, 0.04, 0.03), p + n * 0.016, Vec3::new(0.026, 0.007, 0.003), 0.0025, Quat::from_mat3(&glam::Mat3::from_cols(tab, n, tab.cross(n))));
    for &(s, xc) in &[(0.9f32, 0.40f32), (2.2, 0.42), (3.4, 0.40), (4.5, 0.38)] {
        let (p, n) = foil_frame(&f, s, xc, -1.0);
        inspection_plate(&mut w, p + n * 0.0012, n, Vec3::Z, 0.055, 6);
    }
    let (p, n) = foil_frame(&f, 0.85, 0.42, -1.0);
    w.tube(&STEEL, &[p, p + n * 0.025], &|_| 0.004, 8, true);
    let (p, _) = foil_frame(&f, 2.75, 0.30, -1.0);
    let ring: Vec<Vec3> = (0..=16).map(|k| { let a = k as f32 / 16.0 * TAU; p + Vec3::new(0.0, -0.03 + 0.03 * a.cos(), 0.03 * a.sin()) }).collect();
    w.tube(&STEEL, &ring, &|_| 0.005, 6, false);
    // Rows of screws along the leading-edge skin joint and the tip.
    for k in 0..36 {
        let s = 0.7 + k as f32 * 0.13;
        for side in [-1.0f32, 1.0] {
            let (p, n) = foil_frame(&f, s, 0.13, side);
            w.screw(&SCREW_PAINT, p + n * 0.0003, n, 0.0035, s * 7.0 + side);
        }
    }

    // Nav light and strobe on the tip, the strut and its fittings.
    let mut tip = Mesh::default();
    let (p, _) = foil_frame(&f, 5.47, 0.05, 0.0);
    tip.lathe(&mt(mat::LAMP, 0.9, 0.9, 0.9).x(lamp::STROBE), p + Vec3::new(0.0, 0.0, 0.05), Vec3::X, &[[0.016, 0.0], [0.016, 0.015], [0.012, 0.03], [0.0, 0.034]], 14, true);
    tip.lathe(&mt(mat::LAMP, 0.1, 0.9, 0.2).x(lamp::NAV_GREEN), p + Vec3::new(0.0, 0.0, -0.01), Vec3::new(0.6, 0.0, -0.8), &[[0.02, 0.0], [0.02, 0.012], [0.013, 0.028], [0.0, 0.032]], 16, true);
    w.append(&tip);

    let strut_w = foil_frame(&f, 2.70, 0.24, -1.0).0;
    let strut_f = Vec3::new(0.575, -0.40, -0.18);
    let d = (strut_w - strut_f).normalize();
    let cd = (Vec3::Z - d * d.z).normalize();
    let up = d.cross(cd).normalize();
    let len = (strut_w - strut_f).length();
    let strut = Foil {
        at: Box::new(move |s: f32| Sect { le: strut_f + d * s - cd * 0.04, c: 0.11, cdir: cd, up, tf: 1.0 }),
        t: 0.34,
        camber: 0.0,
    };
    let mut st = Mesh::default();
    strut.skin(&mut st, &WHITE, 0.03, len - 0.03, 24, &zero, &one, 14, false);
    for (p, n) in [(strut_f, -d), (strut_w, d)] {
        let c = p + n * 0.015;
        st.rbox(&WHITE, c - n * 0.02, Vec3::new(0.035, 0.035, 0.035) * Vec3::new(1.0, 1.0, 1.0), 0.015, Quat::from_rotation_arc(Vec3::Y, d));
        st.bolt(&STEEL, c + up * 0.03, up, 0.006);
        st.bolt(&STEEL, c - up * 0.03, -up, 0.006);
    }
    w.append(&st);

    // Landing lights in the left leading edge (built on the right, moved later).
    let mut ll = Mesh::default();
    for s0 in [1.25f32, 1.42] {
        let lens = mt(mat::LAMP, 0.95, 0.92, 0.85).x(lamp::LANDING);
        ll.patch(&lens, 8, 10, false, |a, bb| {
            let s = s0 + (a - 0.5) * 0.13;
            let side = if bb < 0.5 { 1.0 } else { -1.0 };
            let xc = (bb - 0.5).abs() * 2.0 * 0.025;
            let (p, n) = foil_frame(&f, s, xc.max(0.0005), side);
            (p + n * 0.0015, [(a - 0.5) * 2.0, (bb - 0.5) * 2.0], 0.0)
        });
    }
    // The left wing gets the landing lights, a pitot tube, the stall warning
    // vane and a fuel vent; the right one the green light.
    let mut left_extra = ll;
    let (p, n) = foil_frame(&f, 3.5, 0.2, -1.0);
    let bend = p + n * 0.09;
    left_extra.slab(&WHITE, &[Vec2::new(-0.03, 0.0), Vec2::new(0.03, 0.0), Vec2::new(0.012, -0.085), Vec2::new(-0.012, -0.085)], &[], p + Vec3::X * 0.003, Vec3::Z, Vec3::Y, 0.006);
    left_extra.tube(&STEEL, &[bend + Vec3::Z * 0.04, bend - Vec3::Z * 0.17], &|t| if t > 0.9 { 0.006 } else { 0.0075 }, 10, true);
    let (p, n) = foil_frame(&f, 1.72, 0.004, -1.0);
    left_extra.rbox(&ALU, p + n * 0.003, Vec3::new(0.012, 0.002, 0.008), 0.0015, Quat::IDENTITY);
    let (p, n) = foil_frame(&f, 2.4, 0.35, -1.0);
    left_extra.tube(&ALU, &[p, p + n * 0.05, p + n * 0.06 - Vec3::Z * 0.06], &|_| 0.005, 8, true);
    // Mirror the extras: built on the right, they belong to the left wing.
    let left_extra = left_extra.mirror_x();
    let mut lw = w.mirror_x();
    lw.append(&left_extra);
    // Left wing tip light is red.
    for v in &mut lw.v {
        if v.mat[0] == mat::LAMP && v.mat[3] == lamp::NAV_GREEN {
            v.mat[3] = lamp::NAV_RED;
            v.col = [0.95, 0.1, 0.05, v.col[3]];
        }
    }
    b.fixed("left wing", G_LWING, lw);
    b.fixed("right wing", G_RWING, w);
    let ws = wing_structure(&f);
    let l = b.fixed("left wing structure", G_LWING, ws.mirror_x());
    let r = b.fixed("right wing structure", G_RWING, ws);
    b.role(l, Role::WingStructure(0)).role(r, Role::WingStructure(1));

    // Flaps (Fowler: they ride down and back on a hinge below the wing).
    let mut fl = Mesh::default();
    f.skin(&mut fl, &ctrl, FLAP[0] + 0.01, FLAP[1] - 0.01, 22, &|_| 0.70, &one, 14, true);
    f.cap(&mut fl, &ctrl, FLAP[0] + 0.01, 0.70, 1.0, true);
    f.cap(&mut fl, &ctrl, FLAP[1] - 0.01, 0.70, 1.0, true);
    let sm = (FLAP[0] + FLAP[1]) * 0.5;
    let k = wing_sect(sm);
    let axis = (f.pt(FLAP[1], 0.7, 0.0, -1.0) - f.pt(FLAP[0], 0.7, 0.0, -1.0)).normalize();
    let pivot = f.pt(sm, 0.66, -1.0, -1.0) - k.up * 0.09;
    // Flap tracks hanging under the wing.
    for s in [FLAP[0] + 0.35, FLAP[1] - 0.35] {
        let (p, n) = foil_frame(&f, s, 0.62, -1.0);
        let pts = [p, p + n * 0.05 + k.cdir * 0.04, p + n * 0.07 + k.cdir * 0.12];
        w_track(&mut fl, &pts);
        let _ = n;
    }
    let (l, r) = b.pair(["left flap", "right flap"], [G_LWING, G_RWING], fl, |_| Anim::hinge(Drive::new(Input::Flaps, 30f32.to_radians())), pivot, axis, [None, None]);
    b.role(l, Role::Flap(0)).role(r, Role::Flap(1));

    // Ailerons.
    let mut al = Mesh::default();
    f.skin(&mut al, &ctrl, AILERON[0], AILERON[1], 20, &|_| 0.755, &one, 14, true);
    f.cap(&mut al, &ctrl, AILERON[0], 0.755, 1.0, true);
    f.cap(&mut al, &ctrl, AILERON[1], 0.755, 1.0, true);
    let axis = (f.pt(AILERON[1], 0.78, 0.0, -1.0) - f.pt(AILERON[0], 0.78, 0.0, -1.0)).normalize();
    let pivot = f.pt((AILERON[0] + AILERON[1]) * 0.5, 0.78, 0.0, -1.0);
    // Hinge brackets.
    for s in [AILERON[0] + 0.3, AILERON[1] - 0.4] {
        let (p, n) = foil_frame(&f, s, 0.76, -1.0);
        al.rbox(&ALU, p + n * 0.012, Vec3::new(0.004, 0.012, 0.03), 0.003, Quat::IDENTITY);
    }
    let (l, r) = b.pair(["left aileron", "right aileron"], [G_LWING, G_RWING], al, |s| Anim::hinge(Drive::new(Input::Aileron(s), 0.35)), pivot, axis, [None, None]);
    b.role(l, Role::Aileron(0)).role(r, Role::Aileron(1));
}

fn w_track(m: &mut Mesh, pts: &[Vec3]) {
    m.tube(&GREY, pts, &|_| 0.007, 6, true);
}

fn wing_structure(f: &Foil) -> Mesh {
    let mut m = Mesh::default();
    let mut ribs = Vec::new();
    let mut s = 0.62;
    while s < 5.3 {
        // Ribs stop at the control surface webs; the tip bay is closed.
        let to = if s < FLAP[1] { 0.68 } else if s < AILERON[1] { 0.73 } else { 0.97 };
        ribs.push((s, 0.03, to));
        s += 0.305;
    }
    structure(f, &mut m, 0.45, SPAN - 0.1, &[0.25, 0.64], &ribs);
    m
}

// ---------------------------------------------------------------- tail

fn tail(b: &mut Builder) {
    let stab = Foil {
        at: Box::new(|s: f32| {
            let a = s.abs();
            let mut le = 3.50 + a / 1.70 * 0.38;
            let te = 4.58 - a / 1.70 * 0.10;
            let mut c = te - le;
            let mut tf = 1.0;
            if a > 1.58 {
                let u = ((a - 1.58) / 0.12).min(1.0);
                let k = 1.0 - (1.0 - u * u).max(0.0).sqrt();
                le += c * 0.25 * k;
                c -= c * 0.25 * k;
                tf = (1.0 - u * u).max(0.0).sqrt();
            }
            Sect { le: Vec3::new(s, 0.11, le), c, cdir: Vec3::Z, up: Vec3::Y, tf }
        }),
        t: 0.10,
        camber: 0.0,
    };
    let hinge = |s: f32| {
        let k = (stab.at)(s);
        (4.16 - k.le.z) / k.c
    };
    let mut m = Mesh::default();
    stab.skin(&mut m, &WHITE.pat(pat::TAIL), -1.70, 1.70, 60, &|_| 0.0, &|s| hinge(s) - 0.004, 24, false);
    stab.web(&mut m, &PRIMER, -1.70, 1.70, 30, &|s| hinge(s) - 0.004);
    for s in [-1.0f32, 1.0] {
        let (p, n) = (stab.pt(s * 0.9, 0.3, -1.0, -1.0), Vec3::NEG_Y);
        inspection_plate(&mut m, p - Vec3::Y * 0.0012, n, Vec3::Z, 0.04, 4);
    }
    let mut el = Mesh::default();
    let ctrl = WHITE.pat(pat::CTRL);
    stab.skin(&mut el, &ctrl, -1.68, 1.68, 60, &|s| hinge(s) + 0.004, &|_| 1.0, 12, true);
    for s in [-1.68f32, 1.68] {
        stab.cap(&mut el, &ctrl, s, hinge(s) + 0.004, 1.0, true);
    }

    let fin = Foil {
        at: Box::new(|s: f32| {
            let u = s / 1.53;
            let mut le = 3.98 - 0.98 * (1.0 - u).max(0.0).powf(1.6);
            let te = 4.66 - 0.04 * u;
            let mut c = te - le;
            let mut tf = 1.0;
            if s > 1.43 {
                let v = ((s - 1.43) / 0.10).min(1.0);
                let k = 1.0 - (1.0 - v * v).max(0.0).sqrt();
                le += c * 0.15 * k;
                c -= c * 0.15 * k;
                tf = (1.0 - v * v).max(0.05).sqrt();
            }
            Sect { le: Vec3::new(0.0, 0.27 + s, le), c, cdir: Vec3::Z, up: Vec3::X, tf }
        }),
        t: 0.09,
        camber: 0.0,
    };
    let fh = |s: f32| {
        let k = (fin.at)(s);
        (4.30 + 0.12 * s / 1.53 - k.le.z) / k.c
    };
    fin.skin(&mut m, &WHITE.pat(pat::TAIL), 0.0, 1.53, 40, &|_| 0.0, &|s| fh(s) - 0.004, 24, false);
    fin.web(&mut m, &PRIMER, 0.0, 1.53, 20, &|s| fh(s) - 0.004);
    // Beacon on the fin tip, VOR whiskers, the white tail light.
    let top = fin.pt(1.50, 0.35, 0.0, -1.0);
    m.lathe(&BLACK, top + Vec3::Y * 0.01, Vec3::Y, &[[0.022, 0.0], [0.022, 0.012], [0.0, 0.012]], 16, false);
    m.lathe(&mt(mat::LAMP, 0.9, 0.06, 0.03).x(lamp::BEACON), top + Vec3::Y * 0.022, Vec3::Y, &[[0.018, 0.0], [0.018, 0.012], [0.013, 0.028], [0.0, 0.032]], 16, true);
    for sx in [-1.0f32, 1.0] {
        let p = fin.pt(1.2, 0.08, 0.0, -1.0);
        m.tube(&BLACK, &[p, p + Vec3::new(sx * 0.38, 0.08, -0.2)], &|t| 0.005 - 0.002 * t, 6, true);
    }
    m.lathe(&mt(mat::LAMP, 0.95, 0.95, 0.95).x(lamp::TAIL), Vec3::new(0.0, 0.19, TAIL_Z + 0.004), Vec3::Z, &[[0.012, 0.0], [0.012, 0.01], [0.0, 0.016]], 12, true);

    let mut ru = Mesh::default();
    fin.skin(&mut ru, &ctrl, -0.05, 1.52, 40, &|s| fh(s) + 0.004, &|_| 1.0, 12, true);
    fin.cap(&mut ru, &ctrl, -0.05, fh(-0.05) + 0.004, 1.0, true);
    // Rudder and elevator control horns with their cable bolts.
    let hp = Vec3::new(0.0, 0.18, 4.36);
    ru.rbox(&GREY, hp, Vec3::new(0.08, 0.012, 0.025), 0.004, Quat::IDENTITY);
    for sx in [-1.0f32, 1.0] {
        ru.bolt(&STEEL, hp + Vec3::new(sx * 0.075, 0.012, 0.0), Vec3::Y, 0.004);
    }
    let rax = Vec3::new(0.0, 1.53, 0.12).normalize();
    // Spars and ribs for the stump when either surface is broken off.
    let mut ts = Mesh::default();
    structure(&stab, &mut ts, -1.6, 1.6, &[0.22, 0.52], &[(-1.3, 0.03, 0.56), (-0.9, 0.03, 0.56), (-0.5, 0.03, 0.56), (0.5, 0.03, 0.56), (0.9, 0.03, 0.56), (1.3, 0.03, 0.56)]);
    structure(&fin, &mut ts, 0.05, 1.45, &[0.3, 0.6], &[(0.4, 0.03, 0.72), (0.8, 0.03, 0.72), (1.2, 0.03, 0.72)]);
    let k = b.fixed("tail structure", G_TAIL, ts);
    b.role(k, Role::TailStructure);
    let k = b.fixed("stabiliser", G_TAIL, m);
    b.role(k, Role::Stab);
    // The trim tab holds the elevator a little up or down from the stick.
    let k = b.add("elevator", G_TAIL, el, Anim::hinge2(Drive::new(Input::Elevator, -0.40), Drive::new(Input::Trim, -0.5).offset(0.02)), Vec3::new(0.0, 0.11, 4.16), Vec3::X, None);
    b.role(k, Role::Elevator);
    let k = b.add("rudder", G_TAIL, ru, Anim::hinge(Drive::new(Input::Rudder, 0.42)), Vec3::new(0.0, 0.27, 4.30), rax, None);
    b.role(k, Role::Rudder);
}

// ---------------------------------------------------------------- engine, propeller

fn engine(b: &mut Builder) {
    let mut m = Mesh::default();
    // Exhaust stack out of the lower right cowl, and the cowl flaps.
    let pts = [Vec3::new(0.18, -0.45, -2.35), Vec3::new(0.28, -0.60, -2.26), Vec3::new(0.32, -0.70, -2.17), Vec3::new(0.335, -0.75, -2.10)];
    m.tube(&mt(mat::EXHAUST, 0.3, 0.28, 0.26), &pts, &|t| 0.03 + 0.006 * smoothstep(0.85, 1.0, t), 16, false);
    m.tube(&mt(mat::BLACK, 0.01, 0.01, 0.01), &[pts[3] - (pts[3] - pts[2]).normalize() * 0.05, pts[3] + (pts[3] - pts[2]).normalize() * 0.001], &|_| 0.029, 16, true);
    for sx in [-1.0f32, 1.0] {
        let p = Vec3::new(0.17 * sx, sec(-2.12).bot + 0.015, -2.12);
        m.rbox(&WHITE, p, Vec3::new(0.13, 0.004, 0.055), 0.003, Quat::from_rotation_x(0.18));
    }
    b.fixed("exhaust", G_ENGINE, m);

    // Spinner and blades.
    let mut sp = Mesh::default();
    let prof: Vec<[f32; 2]> = (0..=16).map(|k| { let h = k as f32 / 16.0 * 0.42; [0.205 * (1.0 - (h / 0.42).powf(2.2)).max(0.0).sqrt(), h] }).collect();
    sp.lathe(&WHITE, Vec3::new(0.0, -0.035, NOSE_Z - 0.01), Vec3::NEG_Z, &prof, 40, true);
    sp.lathe(&GREY, Vec3::new(0.0, -0.035, NOSE_Z - 0.01), Vec3::NEG_Z, &[[0.21, -0.012], [0.21, 0.0], [0.0, 0.0]], 40, false);
    for k in 0..6 {
        let a = k as f32 / 6.0 * TAU + 0.5;
        let p = Vec3::new(0.0, -0.035, NOSE_Z - 0.03) + Vec3::new(a.cos(), a.sin(), 0.0) * 0.196;
        sp.screw(&SCREW_STEEL, p, Vec3::new(a.cos(), a.sin(), -0.15), 0.005, k as f32);
    }
    let blade_m = mt(mat::PROP, 0.5, 0.5, 0.5);
    let mut blades = Mesh::default();
    for k in 0..2 {
        let mut bl = Mesh::default();
        let blade = Foil {
            at: Box::new(|r: f32| {
                let beta = (1.55 / (TAU * r.max(0.1))).atan() + 0.035;
                let shank = smoothstep(0.24, 0.12, r);
                let mut c = 0.155 - 0.11 * ((r - 0.55) / 0.42).powi(2);
                c = c * (1.0 - shank) + 0.055 * shank;
                let mut tf = (0.18 - 0.11 * (r - 0.22) / 0.75) / 0.10;
                tf = tf * (1.0 - shank) + 9.0 * shank;
                if r > 0.93 {
                    let u = ((r - 0.93) / 0.035).min(1.0);
                    let q = (1.0 - u * u).max(0.0).sqrt();
                    c *= 0.4 + 0.6 * q;
                    tf *= q.max(0.05);
                }
                let t = Vec3::X;
                let cdir = -t * beta.cos() + Vec3::Z * beta.sin();
                let up = Vec3::NEG_Z * beta.cos() - t * beta.sin();
                Sect { le: PROP_HUB + Vec3::Y * r - cdir * (0.35 * c) - up * (0.03 * c), c, cdir, up, tf }
            }),
            t: 0.10,
            camber: 0.035,
        };
        blade.skin(&mut bl, &blade_m, 0.10, 0.965, 36, &|_| 0.0, &|_| 1.0, 14, false);
        bl.transform(0, Mat4::from_translation(PROP_HUB) * Mat4::from_rotation_z(k as f32 * PI) * Mat4::from_translation(-PROP_HUB));
        blades.append(&bl);
    }
    let spin = Anim::hinge(Drive::new(Input::Prop, -1.0));
    let k = b.add("spinner", G_PROP, sp, spin, PROP_HUB, Vec3::Z, None);
    b.role(k, Role::Prop);
    let prop = b.add("blades", G_PROP, blades, spin, PROP_HUB, Vec3::Z, None);
    b.role(prop, Role::Prop);
    b.pieces[prop].vis = Vis::SlowProp;
    let mut disc = Mesh::default();
    disc.disc(&mt(mat::PROPDISC, 0.1, 0.1, 0.1), PROP_HUB + Vec3::Z * 0.005, Vec3::Z, Vec3::Y, 0.975, 64);
    let d = b.add("prop disc", G_PROP, disc, spin, PROP_HUB, Vec3::Z, None);
    b.role(d, Role::Prop);
    b.pieces[d].vis = Vis::FastProp;
    b.pieces[d].glass = true;
    b.pieces[d].occluder = false;
    // Spinning blades are a blur, not something that shades the cowl.
    b.pieces[prop].occluder = false;
}

// ---------------------------------------------------------------- landing gear

fn gear(b: &mut Builder) {
    // Main legs: tapered spring-steel tubes swinging back into the belly.
    let mut leg = Mesh::default();
    let end = Vec3::new(1.152, -1.29, 0.30);
    leg.tube(&PAINT_WHITE, &[MAIN_ROOT, MAIN_ROOT.lerp(end, 0.5) + Vec3::new(0.0, 0.01, 0.0), end], &|t| 0.036 - 0.01 * t, 14, true);
    leg.rbox(&PAINT_WHITE, end + Vec3::new(-0.005, 0.0, 0.0), Vec3::new(0.012, 0.045, 0.035), 0.01, Quat::IDENTITY);
    leg.lathe(&GREY, MAIN_ROOT, (end - MAIN_ROOT).normalize(), &[[0.05, -0.02], [0.05, 0.05], [0.04, 0.07]], 16, false);
    // Brake caliper and its line.
    let cal = MAIN_AXLE + Vec3::new(-0.087, 0.075, 0.045);
    leg.rbox(&mt(mat::SATIN, 0.08, 0.08, 0.09), cal, Vec3::new(0.014, 0.03, 0.022), 0.006, Quat::from_rotation_x(-0.5));
    leg.bolt(&STEEL, cal + Vec3::new(-0.014, 0.012, 0.0), Vec3::NEG_X, 0.004);
    leg.bolt(&STEEL, cal + Vec3::new(-0.014, -0.012, 0.0), Vec3::NEG_X, 0.004);
    let line: Vec<Vec3> = (0..=8).map(|k| { let t = k as f32 / 8.0; MAIN_ROOT.lerp(end, 1.0 - t) + Vec3::new(0.0, 0.0, 0.04) + if k == 0 { Vec3::new(-0.02, 0.04, 0.0) } else { Vec3::ZERO } }).collect();
    leg.tube(&RUBBER, &line, &|_| 0.0045, 6, false);
    let mut wh = Mesh::default();
    wheel(&mut wh, MAIN_AXLE, Vec3::X, MAIN_R, 0.075, 0.08, true);
    let (ll, lr) = b.pair(["left main leg", "right main leg"], [G_GEAR, G_GEAR], leg, Anim::MainLeg, MAIN_ROOT, Vec3::Z, [None, None]);
    b.role(ll, Role::Gear(1)).role(lr, Role::Gear(2));
    let (l, r) = b.pair(["left main wheel", "right main wheel"], [G_GEAR, G_GEAR], wh, |s| Anim::hinge(Drive::new(Input::Wheel(if s < 0.0 { 1 } else { 2 }), -1.0)), MAIN_AXLE, Vec3::X, [Some(ll), Some(lr)]);
    b.role(l, Role::Gear(1)).role(r, Role::Gear(2));
    // Main gear doors, closed with the gear up and down: one over the trough,
    // hinged on its inboard edge, and one over the wheel bay, hinged near the keel.
    let gap = 0.004;
    let (z0, z1) = (TROUGH_DOOR_Z, BAY_Z - 0.003);
    let hinge = |z: f32| { let x = span_at(&MAIN_TROUGH, z).0 + gap; Vec3::new(x, fuse_bottom_y(z, x), z) };
    let (h0, h1) = (hinge(z0), hinge(z1));
    b.pair(["left trough door", "right trough door"], [G_GEAR, G_GEAR], gear_door(&MAIN_TROUGH, z0, z1, gap), |_| Anim::hinge(Drive::new(Input::Gear, -1.75).pulse(0.0, 0.12, 0.88, 1.0)), h0, h1 - h0, [None, None]);
    let (z0, z1) = (BAY_Z + 0.003, MAIN_BAY[1][0] - gap);
    let x = MAIN_BAY[0][1] + gap;
    let (h0, h1) = (Vec3::new(x, fuse_bottom_y(z0, x), z0), Vec3::new(x, fuse_bottom_y(z1, x), z1));
    b.pair(["left bay door", "right bay door"], [G_GEAR, G_GEAR], gear_door(&MAIN_BAY, z0, z1, gap), |_| Anim::hinge(Drive::new(Input::Gear, -1.50).pulse(0.0, 0.12, 0.88, 1.0)), h0, h1 - h0, [None, None]);

    // Nose leg: oleo strut that steers, with torque links and a shimmy damper.
    let (px, top) = (NOSE_PIVOT, NOSE_PIVOT + Vec3::Y * 0.04);
    let mut nl = Mesh::default();
    nl.tube(&PAINT_WHITE, &[top, Vec3::new(0.0, -0.97, px.z)], &|t| 0.044 - 0.004 * t, 18, true);
    nl.tube(&STEEL, &[px - Vec3::X * 0.1, px + Vec3::X * 0.1], &|_| 0.02, 12, true);
    nl.rbox(&PAINT_WHITE, px + Vec3::new(0.0, -0.02, 0.0), Vec3::new(0.06, 0.04, 0.032), 0.012, Quat::IDENTITY);
    nl.lathe(&GREY, Vec3::new(0.0, -0.66, px.z), Vec3::Y, &[[0.058, 0.0], [0.058, 0.035], [0.045, 0.045]], 20, false);
    // Shimmy damper alongside the strut, so it folds into the bay with it.
    let dmp = [Vec3::new(0.064, -0.62, px.z + 0.02), Vec3::new(0.064, -0.79, px.z + 0.02)];
    nl.tube(&mt(mat::SATIN, 0.1, 0.1, 0.11), &dmp, &|_| 0.016, 10, true);
    nl.tube(&CHROME, &[dmp[1], dmp[1] - Vec3::Y * 0.07], &|_| 0.006, 8, true);
    for y in [-0.64f32, -0.86] {
        nl.tube(&GREY, &[Vec3::new(0.04, y, px.z + 0.02), Vec3::new(0.064, y, px.z + 0.02)], &|_| 0.008, 8, true);
    }
    nl.rbox(&GREY, Vec3::new(0.0, -0.94, px.z + 0.044), Vec3::new(0.03, 0.012, 0.012), 0.004, Quat::IDENTITY);
    let leg = b.add("nose leg", G_GEAR, nl, Anim::NoseLeg, NOSE_PIVOT, Vec3::X, None);
    b.role(leg, Role::Gear(0));

    let mut fk = Mesh::default();
    fk.tube(&CHROME, &[Vec3::new(0.0, -0.72, px.z), Vec3::new(0.0, -1.09, px.z)], &|_| 0.031, 16, true);
    let crown = Vec3::new(0.0, -1.105, px.z + 0.01);
    fk.rbox(&PAINT_WHITE, crown, Vec3::new(0.105, 0.022, 0.05), 0.012, Quat::IDENTITY);
    for sx in [-1.0f32, 1.0] {
        let a = crown + Vec3::new(sx * 0.088, 0.0, 0.0);
        let c = NOSE_AXLE + Vec3::new(sx * 0.088, 0.0, 0.0);
        let mid = a.lerp(c, 0.5);
        let dir = (c - a).normalize();
        fk.rbox(&PAINT_WHITE, mid, Vec3::new(0.008, (c - a).length() * 0.5 + 0.02, 0.025), 0.006, Quat::from_rotation_arc(Vec3::Y, -dir));
        fk.bolt(&STEEL, c + Vec3::X * sx * 0.008, Vec3::X * sx, 0.007);
    }
    fk.rbox(&GREY, Vec3::new(0.0, -1.10, px.z + 0.044), Vec3::new(0.03, 0.012, 0.012), 0.004, Quat::IDENTITY);
    let fork = b.add("nose fork", G_GEAR, fk, Anim::slide(Drive::new(Input::Compress(0), 1.0), Vec3::Y), NOSE_PIVOT, Vec3::Y, Some(leg));
    b.role(fork, Role::Gear(0));
    let mut nw = Mesh::default();
    wheel(&mut nw, NOSE_AXLE, Vec3::X, NOSE_R, 0.062, 0.065, false);
    let k = b.add("nose wheel", G_GEAR, nw, Anim::hinge(Drive::new(Input::Wheel(0), -1.0)), NOSE_AXLE, Vec3::X, Some(fork));
    b.role(k, Role::Gear(0));
    // Torque links: two arms meeting at an apex behind the strut (facing up
    // into the bay once the leg has folded back).
    let (u, l) = (Vec3::new(0.0, -0.94, px.z + 0.05), Vec3::new(0.0, -1.10, px.z + 0.05));
    let apex = (u + l) * 0.5 + Vec3::Z * (0.095f32 * 0.095 - 0.08 * 0.08).sqrt();
    for (a, c, anim, parent) in [(u, apex, Anim::LinkUpper, leg), (l, apex, Anim::LinkLower, fork)] {
        let mut lk = Mesh::default();
        for sx in [-1.0f32, 1.0] {
            let off = Vec3::X * sx * 0.018;
            lk.rbox(&GREY, (a + c) * 0.5 + off, Vec3::new(0.004, 0.012, (c - a).length() * 0.5 + 0.008), 0.004, Quat::from_rotation_arc(Vec3::Z, (c - a).normalize()));
        }
        lk.tube(&STEEL, &[a - Vec3::X * 0.026, a + Vec3::X * 0.026], &|_| 0.006, 8, true);
        lk.tube(&STEEL, &[c - Vec3::X * 0.026, c + Vec3::X * 0.026], &|_| 0.006, 8, true);
        let k = b.add(if anim == Anim::LinkUpper { "upper torque link" } else { "lower torque link" }, G_GEAR, lk, anim, a, Vec3::X, Some(parent));
        b.role(k, Role::Gear(0));
    }

    // Nose gear doors, following the cowl bottom.
    let bottom_y = fuse_bottom_y;
    let mut dr = Mesh::default();
    let (z0, z1) = (NOSE_BAY[0] + 0.01, NOSE_BAY[1] - 0.01);
    dr.patch(&WHITE, 16, 6, true, |a, c| {
        let z = z0 + (z1 - z0) * a;
        let x = 0.002 + 0.128 * c;
        (Vec3::new(x, bottom_y(z, x) - 0.002, z), [z, x], 0.0)
    });
    dr.patch(&PRIMER, 16, 6, false, |a, c| {
        let z = z0 + (z1 - z0) * a;
        let x = 0.002 + 0.128 * c;
        (Vec3::new(x, bottom_y(z, x) + 0.004, z), [z, x], 0.0)
    });
    let h0 = Vec3::new(0.13, bottom_y(z0, 0.13), z0);
    let h1 = Vec3::new(0.13, bottom_y(z1, 0.13), z1);
    b.pair(["left nose door", "right nose door"], [G_ENGINE, G_ENGINE], dr, |_| Anim::hinge(Drive::new(Input::Gear, 1.45).smooth(0.0, 0.12)), h0, h1 - h0, [None, None]);
}

// ---------------------------------------------------------------- gear kinematics

/// Where the Cessna's gear legs and torque links are: the nose leg folds back
/// and steers, the links scissor with the oleo, the main legs swing back and
/// up into the belly and flex with the load. `legs` is how far the legs have
/// come down (0..1), `steer` the nose wheel angle, `comp` each oleo's travel.
pub fn kinematics(anim: Anim, p: &Piece, legs: f32, steer: f32, comp: &[f32]) -> Mat4 {
    let hinge = |a: f32| Mat4::from_translation(p.pivot) * Mat4::from_axis_angle(p.axis, a) * Mat4::from_translation(-p.pivot);
    // Torque link angles for a given oleo compression.
    let link = |c: f32, upper: bool| -> f32 {
        let ang = |d: f32| {
            let half = d * 0.5;
            let h = (0.095f32 * 0.095 - half * half).max(0.0).sqrt();
            if upper { (-h).atan2(-half) } else { (-h).atan2(half) }
        };
        ang(0.16 - c) - ang(0.16)
    };
    match anim {
        Anim::NoseLeg => {
            let q = Quat::from_rotation_x(-(1.0 - legs) * NOSE_FOLD) * Quat::from_rotation_y(-steer);
            Mat4::from_translation(p.pivot) * Mat4::from_quat(q) * Mat4::from_translation(-p.pivot)
        }
        // The links sit behind the strut: mirrored fore and aft, the angles change sign.
        Anim::LinkUpper => hinge(-link(comp[0], true)),
        Anim::LinkLower => hinge(-link(comp[0], false)),
        Anim::MainLeg(side) => {
            let m = Vec3::new(side, 1.0, 1.0);
            let (root, axle, stowed) = (MAIN_ROOT * m, MAIN_AXLE * m, MAIN_STOWED * m);
            let up = Quat::from_rotation_arc((axle - root).normalize(), (stowed - root).normalize());
            let q = Quat::IDENTITY.slerp(up, 1.0 - legs) * Quat::from_rotation_z(side * comp[if side < 0.0 { 1 } else { 2 }] / 1.07);
            Mat4::from_translation(root) * Mat4::from_quat(q) * Mat4::from_translation(-root)
        }
        _ => Mat4::IDENTITY,
    }
}
