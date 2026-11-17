//! A speedboat you drive yourself: a 7.4 m open runabout with an outboard.
//!
//! The hull is a set of columns over its bottom (a deep V aft, finer and
//! rising to the bow). Each column floats by how deep it sits under the
//! sea surface the GPU draws (world.rs `ground`, ocean.rs `height_w`), so the
//! boat heaves, pitches and rolls on the actual waves. Water striking the
//! bottom pushes back along its normal (the pressure of a planing surface:
//! at speed the trimmed bottom lifts the hull out until it skims; a hull
//! falling back off a crest slams); water sliding along it drags. The keel
//! and the motor's lower unit resist sliding sideways, and the motor turns
//! with the steering, so its thrust and its fin swing the stern round. Out
//! of the water the propeller races and pushes nothing. Land is solid and
//! rough: the boat runs aground and stops.
//!
//! Body frame as the aircraft's (airframe.rs): +X right, +Y up, -Z forward,
//! the origin at the centre of gravity.
use crate::render::MeshVertex;
use glam::{DQuat, DVec3, Vec3};
use std::collections::VecDeque;

/// What the start screen calls it, and says about it.
pub const NAME: &str = "Speedboat";
pub const BLURB: &str = "7.4 m centre-console runabout, 300 hp outboard. W/S throttle and reverse, A/D steer, Shift for the last few knots: about 40 kt flat out.";
/// Names it answers to on the command line (`--aircraft boat`).
pub fn is_key(k: &str) -> bool {
    matches!(k.to_ascii_lowercase().as_str(), "boat" | "speedboat" | "runabout")
}
/// Its picture in the start screen: rendered in the game, at rest off San Francisco.
pub const STILL: &[u8] = include_bytes!("../assets/aircraft/boat.png");
/// Chase camera distance (m).
pub const CAM_DIST: f64 = 13.0;
/// Index of its meshes among the instanced solids (main.rs): the hull, then the motor.
pub const MESH: u32 = crate::airmodels::BASE + crate::airmodels::COUNT as u32;

const RHO: f64 = 1025.0;
const G: f64 = 9.81;
/// Mass with fuel and two aboard (kg), and moments of inertia about the body
/// axes: pitch (x), yaw (y), roll (z) (kg m²).
pub const MASS: f64 = 1500.0;
const INERTIA: DVec3 = DVec3::new(7400.0, 7900.0, 1100.0);
/// Rotational damping in the water about pitch, yaw and roll (N m s).
const DAMP: DVec3 = DVec3::new(30_000.0, 1_500.0, 22_000.0);
/// Bow and transom (body z, m), and the hull's widest half beam.
pub const BOW: f64 = -5.45;
pub const STERN: f64 = 1.95;
const BEAM: f64 = 1.22;
/// Propeller thrust at full throttle (N), and with the boost held.
const THRUST: f64 = 10_500.0;
const BOOST: f64 = 1.4;
/// Largest steering angle of the motor (rad).
const STEER_MAX: f64 = 0.45;
/// Lift slope of the planing bottom (per radian of trim, on its wetted area).
const PLANE_K: f64 = 0.5;
/// Side area of the motor's lower unit (m²): its fin steers.
const FIN: f64 = 0.35;
/// How much of the sea surface's rise under a column (seen from the moving
/// hull, so mostly the slope it runs over, already felt as depth) counts
/// as water moving at the bottom.
const WS_K: f64 = 0.0;
/// Where the motor swivels (body frame) and where its propeller is.
pub const MOTOR: DVec3 = DVec3::new(0.0, 0.42, STERN + 0.08);
const PROP: DVec3 = DVec3::new(0.0, -0.62, STERN + 0.42);

// ------------------------------------------------------------------ hull shape
// Along the hull, t = 0 at the bow and 1 at the transom.
fn t_of(z: f64) -> f64 {
    ((z - BOW) / (STERN - BOW)).clamp(0.0, 1.0)
}
/// Half beam at the chine.
fn chine_w(t: f64) -> f64 {
    1.06 * (1.0 - (1.0 - t).powf(2.6)).max(0.0).powf(0.55)
}
/// Half beam at the sheer (the topsides flare out above the chine).
fn sheer_w(t: f64) -> f64 {
    BEAM * (1.0 - (1.0 - t).powf(2.3)).max(0.0).powf(0.5)
}
/// Height of the keel: flat aft, sweeping up to the stem.
fn keel_y(t: f64) -> f64 {
    -0.56 + 0.9 * ((0.4 - t).max(0.0) / 0.4).powf(1.7)
}
/// Deadrise (rad): 21 degrees at the transom, sharper forward.
fn deadrise(t: f64) -> f64 {
    (21.0 + 30.0 * (1.0 - t).powi(2)).to_radians()
}
fn chine_y(t: f64) -> f64 {
    keel_y(t) + chine_w(t) * deadrise(t).tan()
}
fn sheer_y(t: f64) -> f64 {
    0.5 + 0.42 * (1.0 - t).powi(2)
}
/// Bottom of the hull at body (x, z), or None outside its plan.
fn bottom_y(x: f64, z: f64) -> Option<f64> {
    let t = t_of(z);
    let w = chine_w(t);
    (x.abs() <= w + 1e-6 && z >= BOW && z <= STERN).then(|| keel_y(t) + x.abs() * deadrise(t).tan())
}

/// A buoyancy column: where it is on the bottom, its share of the plan
/// area (m²), how tall it is (to the sheer), and the bottom's outward normal.
#[derive(Clone, Copy, Debug)]
struct Col {
    at: DVec3,
    area: f64,
    top: f64,
    n: DVec3,
}

fn columns() -> Vec<Col> {
    const NZ: usize = 9;
    const NX: usize = 4;
    let dz = (STERN - BOW) / NZ as f64;
    let mut out = Vec::new();
    for i in 0..NZ {
        let z = BOW + dz * (i as f64 + 0.5);
        let t = t_of(z);
        let w = chine_w(t);
        if w < 0.05 {
            continue;
        }
        let dx = 2.0 * w / NX as f64;
        for j in 0..NX {
            let x = -w + dx * (j as f64 + 0.5);
            let y = bottom_y(x, z).unwrap();
            // The bottom's slope across (deadrise) and along (the bow's sweep up).
            let dydz = (keel_y(t_of(z + 0.05)) - keel_y(t_of(z - 0.05))) / 0.1;
            let n = DVec3::new(x.signum() * deadrise(t).tan(), -1.0, dydz).normalize();
            out.push(Col { at: DVec3::new(x, y, z), area: dx * dz, top: sheer_y(t), n });
        }
    }
    out
}

/// The sea (or land) under a point: its surface height above the sphere
/// (geo::R) and whether it is water.
#[derive(Clone, Copy, Debug)]
pub struct Surface {
    pub h: f64,
    pub water: bool,
}

/// What the driver asks for.
#[derive(Clone, Copy, Debug, Default)]
pub struct Input {
    /// +1 full ahead, -1 full astern
    pub throttle: f64,
    /// +1 hard to starboard (right)
    pub steer: f64,
    pub boost: bool,
}

/// A point of the wake's track: where the transom passed (on the water), when
/// (s, `Boat::t`), how fast, and how hard it was pushing (0..1).
#[derive(Clone, Copy, Debug)]
pub struct WakePt {
    pub pos: DVec3,
    pub t: f64,
    pub speed: f64,
    pub power: f64,
}

pub struct Boat {
    pub pos: DVec3,
    pub vel: DVec3,
    pub rot: DQuat,
    /// angular velocity, body frame (rad/s)
    pub omega: DVec3,
    /// throttle the motor is at (-1..1, follows the lever), its steering
    /// angle (rad), and its revs (rpm)
    pub throttle: f64,
    pub steer: f64,
    pub rpm: f64,
    /// share of the hull's plan in the water (0..1), and of the propeller
    pub wet: f64,
    pub prop_wet: f64,
    /// on land
    pub aground: bool,
    /// seconds since it was last in the water (0 while it is)
    pub air_t: f64,
    pub t: f64,
    cols: Vec<Col>,
    /// last step's sea height at each column, for how fast the surface rises
    prev_h: Vec<f64>,
    /// where it hit the water, how fast it fell (m/s): for the splash
    pub splashes: Vec<(DVec3, f64)>,
    /// spray thrown off the bow this step: (where, velocity, size 0..1)
    pub spray: Vec<(DVec3, DVec3, f32)>,
    spray_acc: f64,
    pub wake: VecDeque<WakePt>,
}

/// Wake track points kept, and how long (s).
pub const WAKE_N: usize = 40;
pub const WAKE_LIFE: f64 = 45.0;

impl Boat {
    /// Afloat at `pos` (on the surface), heading `heading` (rad from north).
    pub fn new(pos: DVec3, heading: f64) -> Self {
        let (e, n, u) = crate::geo::enu(pos);
        let fwd = n * heading.cos() + e * heading.sin();
        let rot = DQuat::from_mat3(&glam::DMat3::from_cols(fwd.cross(u).normalize(), u, -fwd));
        let cols = columns();
        Self {
            pos,
            vel: DVec3::ZERO,
            rot,
            omega: DVec3::ZERO,
            throttle: 0.0,
            steer: 0.0,
            rpm: 750.0,
            wet: 1.0,
            prop_wet: 1.0,
            aground: false,
            air_t: 0.0,
            t: 0.0,
            prev_h: vec![f64::NAN; cols.len()],
            cols,
            splashes: Vec::new(),
            spray: Vec::new(),
            spray_acc: 0.0,
            wake: VecDeque::new(),
        }
    }

    pub fn forward(&self) -> DVec3 {
        self.rot * DVec3::NEG_Z
    }
    /// Speed through the water, along the hull (m/s).
    pub fn speed(&self) -> f64 {
        self.vel.dot(self.forward())
    }

    /// One step of `h` seconds. `sea` gives the surface under a point.
    pub fn step(&mut self, h: f64, input: &Input, sea: &mut impl FnMut(DVec3) -> Surface) {
        self.t += h;
        let up = self.pos.normalize();
        let alt = |p: DVec3| p.length() - crate::geo::R;
        // The motor answers the lever and the wheel with a little lag.
        let want = input.throttle.clamp(-1.0, 1.0) * if input.boost && input.throttle > 0.0 { BOOST } else { 1.0 };
        self.throttle += (want - self.throttle) * (h * 2.5).min(1.0);
        let st = input.steer.clamp(-1.0, 1.0) * STEER_MAX;
        self.steer += (st - self.steer).clamp(-1.6 * h, 1.6 * h);

        let mut force = -up * (MASS * G);
        let mut torque = DVec3::ZERO; // world frame, about the centre of gravity
        let wb = self.rot * self.omega; // angular velocity, world frame
        let add = |f: DVec3, at: DVec3, force: &mut DVec3, torque: &mut DVec3| {
            *force += f;
            *torque += (at - self.pos).cross(f);
        };
        let mut wet_area = 0.0;
        let mut total_area = 0.0;
        let mut aground = false;
        let mut slam: (f64, DVec3, f64) = (0.0, DVec3::ZERO, 0.0);
        for (i, c) in self.cols.iter().enumerate() {
            let p = self.pos + self.rot * c.at;
            let s = sea(p);
            total_area += c.area;
            let y = alt(p);
            let v = self.vel + wb.cross(p - self.pos);
            if !s.water {
                // Land: stiff, damped, and rough.
                let pen = s.h - y;
                if pen > 0.0 {
                    aground = true;
                    let vn = v.dot(up);
                    let nf = (3.0e5 * pen * c.area - 2.0e4 * c.area * vn).max(0.0);
                    let vt = v - up * vn;
                    let fr = -vt.normalize_or_zero() * (nf * 0.7).min(vt.length() * 4.0e4 * c.area);
                    add(up * nf + fr, p, &mut force, &mut torque);
                }
                self.prev_h[i] = f64::NAN;
                continue;
            }
            let depth = (s.h - y).min(c.top - c.at.y);
            // How fast the surface itself rises here.
            let ws = if self.prev_h[i].is_finite() { ((s.h - self.prev_h[i]) / h).clamp(-6.0, 6.0) } else { 0.0 };
            self.prev_h[i] = s.h;
            if depth <= 0.0 {
                continue;
            }
            wet_area += c.area * (depth / 0.12).min(1.0);
            // Buoyancy.
            add(up * (RHO * G * c.area * depth), p, &mut force, &mut torque);
            let vr = v - up * (ws * WS_K);
            let n = self.rot * c.n;
            let vn = vr.dot(n);
            let sub = (depth / 0.15).min(1.0);
            // Water struck by the bottom pushes back along its normal: the
            // planing lift, and the slam of a hull falling onto the sea.
            if vn > 0.0 {
                // Cross flow, and the lift of a planing surface: the water
                // running aft along it at a small angle of attack (trim:
                // sin a ~ vn / |v|, from the motion in the hull's own fore
                // and aft plane only - sideways the bottom slides, it does
                // not plane).
                let vb = self.rot.inverse() * vr;
                let vl = DVec3::new(0.0, vb.y, vb.z);
                let nb = c.n;
                let vnl = vl.dot(nb).max(0.0);
                let p_dyn = (0.5 * RHO * (0.4 * vn * vn + PLANE_K * vb.z.abs() * vnl) + 900.0 * vn).min(12000.0);
                let f = -n * (p_dyn * c.area * sub);
                add(f, p, &mut force, &mut torque);
                if vn > slam.0 {
                    slam = (vn, p, 0.0);
                }
            } else {
                // (water falling away from it pulls only a little)
                add(-n * (400.0 * vn * c.area * sub), p, &mut force, &mut torque);
            }
            // Friction along the bottom, with the spray it throws.
            let vt = vr - n * vn;
            let f = -vt * (0.5 * RHO * 0.0055 * vt.length() * c.area * sub * 1.6);
            add(f, p, &mut force, &mut torque);
        }
        let _ = slam.2;
        self.wet = (wet_area / total_area.max(1e-6)).clamp(0.0, 1.0);
        self.aground = aground;

        // Keel and motor: sideways water drag (the keel ahead of the centre of
        // lateral resistance, the motor's fin at the stern turning with it).
        let right = self.rot * DVec3::X;
        let keel_at = self.pos + self.rot * DVec3::new(0.0, -0.3, 0.2);
        let sk = sea(keel_at);
        if sk.water && sk.h > alt(keel_at) {
            let v = self.vel + wb.cross(keel_at - self.pos);
            let vx = v.dot(right);
            add(-right * (0.5 * RHO * 1.1 * 0.5 * vx * vx.abs() + 600.0 * vx), keel_at, &mut force, &mut torque);
        }
        let mrot = self.rot * DQuat::from_rotation_y(self.steer);
        let prop_at = self.pos + self.rot * MOTOR + mrot * (PROP - MOTOR);
        let sp = sea(prop_at);
        let pdepth = if sp.water { sp.h - alt(prop_at) } else { -1.0 };
        self.prop_wet = ((pdepth + 0.12) / 0.3).clamp(0.0, 1.0);
        let thrust_dir = mrot * DVec3::NEG_Z;
        let side = mrot * DVec3::X;
        let v = self.vel + wb.cross(prop_at - self.pos);
        if self.prop_wet > 0.0 {
            // Reverse pushes less (the propeller's blades are made for ahead).
            let t = self.throttle * THRUST * if self.throttle < 0.0 { 0.45 } else { 1.0 };
            // (it pushes less the faster the water already comes at it)
            let ahead = v.dot(thrust_dir).max(0.0);
            let lapse = (1.0 - ahead / 40.0).clamp(0.2, 1.0);
            add(thrust_dir * (t * lapse * self.prop_wet), prop_at, &mut force, &mut torque);
            let vx = v.dot(side);
            add(-side * ((0.5 * RHO * 1.2 * FIN * vx * vx.abs() + 200.0 * vx) * self.prop_wet), prop_at, &mut force, &mut torque);
        }
        // The revs: loaded in the water, racing out of it.
        let rev = 750.0 + 5000.0 * self.throttle.abs().min(1.0) * (0.75 + 0.25 * (1.0 - self.prop_wet)) + 600.0 * (self.throttle - 1.0).max(0.0);
        self.rpm += (rev - self.rpm) * (h * if self.prop_wet < 0.5 { 6.0 } else { 2.0 }).min(1.0);

        // Air drag on the hull and windshield.
        force -= self.vel * (0.5 * 1.2 * 0.55 * 3.2 * self.vel.length());
        // Damping of the turning, pitching and above all the rolling (the
        // water's added mass and the eddies off the chines), body axes.
        torque -= self.rot * (self.omega * DAMP * self.wet.max(0.35));
        // Off a crest, the hull's own drag (and its wide flat sides) turns
        // it toward the way it flies, as an arrow, a few degrees bow up.
        if self.wet < 0.02 && self.vel.length() > 3.0 {
            let fwd = self.forward();
            let want = (self.vel.normalize() + up * 0.08).normalize();
            torque += fwd.cross(want) * (MASS * 9.0 * self.vel.length().min(25.0));
        }

        // Integrate: the centre of gravity, then the rotation (body frame).
        self.vel += force / MASS * h;
        self.pos += self.vel * h;
        let tb = self.rot.inverse() * torque;
        let w = self.omega;
        let iw = INERTIA * w;
        let dw = (tb - w.cross(iw)) / INERTIA;
        self.omega += dw * h;
        let ang = self.omega * h;
        if ang.length_squared() > 1e-18 {
            self.rot = (self.rot * DQuat::from_scaled_axis(ang)).normalize();
        }

        // Back in the water after a jump: a splash as big as the fall.
        if self.wet > 0.005 {
            if self.air_t > 0.25 {
                let vs = -self.vel.dot(up);
                if vs > 1.0 {
                    self.splashes.push((self.pos - up * 0.3, vs));
                }
            }
            self.air_t = 0.0;
        } else {
            self.air_t += h;
        }

        // Spray off the bow where it cuts the water: up the hull's sides at
        // speed, thrown out sideways and back. Its rate and size grow with speed.
        let spd = self.speed().max(0.0);
        if self.wet > 0.01 && spd > 4.0 {
            self.spray_acc += h * (spd - 4.0) * 9.0;
            let fwd = self.forward();
            while self.spray_acc >= 1.0 {
                self.spray_acc -= 1.0;
                let k = (self.t * 7919.0).sin() * 0.5 + 0.5;
                let sgn = if (self.t * 104_729.0).sin() > 0.0 { 1.0 } else { -1.0 };
                // Where the water meets the bow, further aft as it planes.
                let z = BOW + 1.1 + 1.6 * (spd / 22.0).min(1.0) + 0.8 * k;
                let tz = t_of(z);
                let at = self.pos + self.rot * DVec3::new(sgn * chine_w(tz) * 1.02, chine_y(tz) + 0.05, z);
                let out = right * sgn;
                // (sheets thrown out low and sideways, then falling back: a few
                // tenths of a second in the air, a metre or two out)
                let vel = self.vel * 0.8 + out * (1.6 + 0.16 * spd) * (0.7 + 0.6 * k) + up * (1.0 + 0.07 * spd) * (0.6 + 0.8 * (1.0 - k)) - fwd * (0.05 * spd);
                self.spray.push((at, vel, (spd / 20.0).min(1.0) as f32));
            }
        }

        // The wake's track: where the transom passes, every so often.
        let stern = self.pos + self.rot * DVec3::new(0.0, -0.3, STERN);
        let gap = self.wake.back().map(|w| (w.pos - stern).length()).unwrap_or(f64::MAX);
        let since = self.wake.back().map(|w| self.t - w.t).unwrap_or(f64::MAX);
        if self.air_t < 0.15 && (gap > 2.0 + 0.35 * spd || (since > 1.2 && gap > 0.3)) {
            let ss = sea(stern);
            let on = stern - up * (alt(stern) - ss.h);
            self.wake.push_back(WakePt { pos: on, t: self.t, speed: spd, power: self.throttle.abs().min(1.0) * self.prop_wet });
        }
        while self.wake.len() > WAKE_N || self.wake.front().is_some_and(|w| self.t - w.t > WAKE_LIFE) {
            self.wake.pop_front();
        }
    }

    /// The wake for the shaders (common.wgsl `Globals.wake`): the newest track
    /// points first, beginning at the transom now; each two rows: position
    /// (camera-relative) + age (s); speed (m/s), push (0..1), 0, 0. Row 0:
    /// how many points, 0, 0, 0.
    pub fn wake_rows(&self, cam: DVec3, rows: &mut [[f32; 4]]) {
        let n_max = (rows.len() - 1) / 2;
        let up = self.pos.normalize();
        let stern = self.pos + self.rot * DVec3::new(0.0, -0.3, STERN);
        let mut pts = Vec::with_capacity(n_max);
        if self.air_t < 0.4 {
            // (the transom itself, on the water)
            let p = stern - up * (stern - self.pos).dot(up) - up * 0.3;
            pts.push(WakePt { pos: p, t: self.t, speed: self.speed().max(0.0), power: self.throttle.abs().min(1.0) * self.prop_wet });
        }
        pts.extend(self.wake.iter().rev().copied());
        pts.truncate(n_max);
        rows.fill([0.0; 4]);
        rows[0][0] = pts.len() as f32;
        for (i, w) in pts.iter().enumerate() {
            let p = (w.pos - cam).as_vec3();
            rows[1 + 2 * i] = [p.x, p.y, p.z, (self.t - w.t) as f32];
            rows[2 + 2 * i] = [w.speed as f32, w.power as f32, 0.0, 0.0];
        }
    }
}

impl Boat {
    /// The hull and the motor, swung by the steering (instanced solids, inst.wgsl).
    pub fn instances(&self, cam: DVec3, out: &mut Vec<(u32, crate::render::Inst)>) {
        let inst = |m: glam::DMat4| {
            let (x, y, z, o) = (m.x_axis, m.y_axis, m.z_axis, m.w_axis - cam.extend(0.0));
            let r = |i: usize| [x[i] as f32, y[i] as f32, z[i] as f32, o[i] as f32];
            crate::render::Inst { m: [r(0), r(1), r(2)], tint: HULL_TINT, ext: [0.0; 4] }
        };
        let base = glam::DMat4::from_rotation_translation(self.rot, self.pos);
        out.push((MESH, inst(base)));
        out.push((MESH + 1, inst(base * glam::DMat4::from_rotation_translation(DQuat::from_rotation_y(self.steer), MOTOR))));
    }
}

/// What the anti-aliasing takes to move with the boat (common.wgsl
/// `plane_box`): centre and half size in its body frame.
pub const TAA_BOX: [[f32; 4]; 2] = [[0.0, 0.25, ((BOW + STERN + 0.6) * 0.5) as f32, 0.0], [1.5, 1.35, ((STERN + 0.6 - BOW) * 0.5 + 0.4) as f32, 0.0]];

// ------------------------------------------------------------------ model

pub(crate) type Mesh = (Vec<MeshVertex>, Vec<u32>);

/// inst.wgsl's materials: paint takes the instance tint, glass, lights, trim as coloured.
pub(crate) const PAINT: f32 = 0.0;
pub(crate) const LAMP: f32 = 0.25;
pub(crate) const GLASS: f32 = 0.5;
pub(crate) const TAIL: f32 = 0.75;
pub(crate) const TRIM: f32 = 1.0;

#[derive(Default)]
pub(crate) struct MB {
    pub(crate) v: Vec<MeshVertex>,
    pub(crate) i: Vec<u32>,
}

impl MB {
    /// A grid of points (rows of equal length) as one smooth surface.
    pub(crate) fn grid(&mut self, rows: &[Vec<Vec3>], col: [f32; 3], mat: f32, flip: bool) {
        let base = self.v.len() as u32;
        let nr = rows.len();
        let nc = rows[0].len();
        let mut nrm = vec![Vec3::ZERO; nr * nc];
        for r in 0..nr - 1 {
            for c in 0..nc - 1 {
                let (a, b, d, e) = (rows[r][c], rows[r][c + 1], rows[r + 1][c], rows[r + 1][c + 1]);
                let n = (b - a).cross(d - a) + (e - b).cross(d - b);
                let n = if flip { -n } else { n };
                for k in [r * nc + c, r * nc + c + 1, (r + 1) * nc + c, (r + 1) * nc + c + 1] {
                    nrm[k] += n;
                }
            }
        }
        for r in 0..nr {
            for c in 0..nc {
                self.v.push(MeshVertex { pos: rows[r][c].into(), nrm: nrm[r * nc + c].normalize_or_zero().into(), col: [col[0], col[1], col[2], mat] });
            }
        }
        for r in 0..nr as u32 - 1 {
            for c in 0..nc as u32 - 1 {
                let (a, b, d, e) = (base + r * nc as u32 + c, base + r * nc as u32 + c + 1, base + (r + 1) * nc as u32 + c, base + (r + 1) * nc as u32 + c + 1);
                self.i.extend([a, b, d, b, e, d]);
            }
        }
    }
    /// A flat polygon (convex, in order), one normal.
    pub(crate) fn poly(&mut self, pts: &[Vec3], col: [f32; 3], mat: f32) {
        let base = self.v.len() as u32;
        let n = (pts[1] - pts[0]).cross(pts[2] - pts[0]).normalize_or_zero();
        for p in pts {
            self.v.push(MeshVertex { pos: (*p).into(), nrm: n.into(), col: [col[0], col[1], col[2], mat] });
        }
        for k in 1..pts.len() as u32 - 1 {
            self.i.extend([base, base + k, base + k + 1]);
        }
    }
    /// A box between two corners, faceted.
    pub(crate) fn cube(&mut self, lo: Vec3, hi: Vec3, col: [f32; 3], mat: f32) {
        let c = |x: bool, y: bool, z: bool| Vec3::new(if x { hi.x } else { lo.x }, if y { hi.y } else { lo.y }, if z { hi.z } else { lo.z });
        let faces = [
            [c(true, false, false), c(true, true, false), c(true, true, true), c(true, false, true)],
            [c(false, false, true), c(false, true, true), c(false, true, false), c(false, false, false)],
            [c(false, true, false), c(false, true, true), c(true, true, true), c(true, true, false)],
            [c(false, false, true), c(false, false, false), c(true, false, false), c(true, false, true)],
            [c(false, false, true), c(true, false, true), c(true, true, true), c(false, true, true)],
            [c(true, false, false), c(false, false, false), c(false, true, false), c(true, true, false)],
        ];
        for f in faces {
            self.poly(&f, col, mat);
        }
    }
    /// A tube along a polyline (closed ring of `k` sides), radius `r`.
    pub(crate) fn tube(&mut self, path: &[Vec3], r: f32, k: usize, col: [f32; 3], mat: f32) {
        let mut rows = Vec::new();
        for (i, p) in path.iter().enumerate() {
            let d = if i + 1 < path.len() { path[i + 1] - *p } else { *p - path[i - 1] }.normalize();
            let a = if d.y.abs() < 0.9 { Vec3::Y } else { Vec3::X };
            let u = d.cross(a).normalize();
            let w = d.cross(u);
            rows.push((0..=k).map(|j| *p + (u * (j as f32 / k as f32 * std::f32::consts::TAU).cos() + w * (j as f32 / k as f32 * std::f32::consts::TAU).sin()) * r).collect());
        }
        self.grid(&rows, col, mat, false);
    }
    /// A box between two corners with its upright edges and top edges
    /// rounded (radius `r`), smooth-shaded; open underneath.
    pub(crate) fn rbox(&mut self, lo: Vec3, hi: Vec3, r: f32, col: [f32; 3], mat: f32) {
        let (hx, hz) = ((hi.x - lo.x) * 0.5, (hi.z - lo.z) * 0.5);
        let r = r.min(hx * 0.95).min(hz * 0.95).min((hi.y - lo.y) * 0.95).max(0.001);
        let c = Vec3::new((lo.x + hi.x) * 0.5, 0.0, (lo.z + hi.z) * 0.5);
        const K: usize = 4;
        // A rounded rectangle `inset` in from the sides, at height y.
        let ring = |inset: f32, y: f32| -> Vec<Vec3> {
            let rc = (r - inset).max(0.0);
            let (ax, az) = (hx - inset - rc, hz - inset - rc);
            let mut pts = Vec::new();
            for (q, (sx, sz)) in [(1.0f32, 1.0f32), (-1.0, 1.0), (-1.0, -1.0), (1.0, -1.0)].iter().enumerate() {
                for k in 0..=K {
                    let a = (q as f32 + k as f32 / K as f32) * std::f32::consts::FRAC_PI_2;
                    pts.push(c + Vec3::new(sx * ax + rc * a.cos(), y, sz * az + rc * a.sin()));
                }
            }
            pts.push(pts[0]);
            pts
        };
        let mut rows = vec![ring(0.0, lo.y), ring(0.0, hi.y - r)];
        for k in 1..=K {
            let a = k as f32 / K as f32 * std::f32::consts::FRAC_PI_2;
            rows.push(ring(r * (1.0 - a.cos()), hi.y - r + r * a.sin()));
        }
        self.grid(&rows, col, mat, false);
        let top = rows.last().unwrap().clone();
        let mut cap: Vec<Vec3> = top[..top.len() - 1].to_vec();
        cap.reverse();
        self.poly(&cap, col, mat);
    }
    pub(crate) fn done(self) -> Mesh {
        (self.v, self.i)
    }
}

const NAVY: [f32; 3] = [0.03, 0.07, 0.2];
const RED: [f32; 3] = [0.55, 0.05, 0.04];
const DECK: [f32; 3] = [0.8, 0.8, 0.77];
const FLOOR: [f32; 3] = [0.52, 0.53, 0.52];
const DARK: [f32; 3] = [0.035, 0.037, 0.04];
const STEEL: [f32; 3] = [0.62, 0.64, 0.66];
const SEAT: [f32; 3] = [0.82, 0.8, 0.74];
/// The hull's paint (the instance tint): white.
pub const HULL_TINT: [f32; 4] = [0.86, 0.87, 0.88, 0.0];

/// The boat: hull, deck, console, windshield and seats (one mesh), and the
/// outboard motor (its own, to swing with the steering about `MOTOR`).
pub fn meshes() -> [Mesh; 2] {
    let mut m = MB::default();
    const NS: usize = 34;
    let zs: Vec<f64> = (0..=NS).map(|i| BOW + (STERN - BOW) * (1.0 - (1.0 - i as f64 / NS as f64).powf(1.6))).collect();
    let v3 = |x: f64, y: f64, z: f64| Vec3::new(x as f32, y as f32, z as f32);
    // Topsides in bands, chine to sheer: a navy boot stripe just above the
    // chine, a red pinstripe above it, white above that.
    let side = |s: f64, a: f64, b: f64| -> Vec<Vec<Vec3>> {
        zs.iter()
            .map(|&z| {
                let t = t_of(z);
                let (cw, cy, sw, sy) = (chine_w(t) + 0.05, chine_y(t), sheer_w(t), sheer_y(t));
                (0..=3)
                    .map(|k| {
                        let f = a + (b - a) * k as f64 / 3.0;
                        // (a slight convex curve to the flare)
                        let x = cw + (sw - cw) * f + 0.03 * (f * (1.0 - f)) * 4.0 * (t * 4.0).min(1.0);
                        v3(s * x, cy + (sy - cy) * f, z)
                    })
                    .collect()
            })
            .collect()
    };
    for s in [-1.0, 1.0] {
        let flip = s < 0.0;
        m.grid(&side(s, 0.0, 0.16), NAVY, TRIM, flip);
        m.grid(&side(s, 0.16, 0.2), [0.95, 0.95, 0.95], TRIM, flip);
        m.grid(&side(s, 0.2, 0.245), RED, TRIM, flip);
        m.grid(&side(s, 0.245, 1.0), [1.0; 3], PAINT, flip);
        // Bottom: keel to chine, then the spray rail's lip.
        let bottom: Vec<Vec<Vec3>> = zs
            .iter()
            .map(|&z| {
                let t = t_of(z);
                (0..=5).map(|k| {
                    let x = chine_w(t) * k as f64 / 5.0;
                    v3(s * x, keel_y(t) + x * deadrise(t).tan(), z)
                }).chain(std::iter::once(v3(s * (chine_w(t) + 0.05), chine_y(t), z))).collect()
            })
            .collect();
        m.grid(&bottom, [0.9, 0.9, 0.9], PAINT, !flip);
        // Gunwale cap: a rounded rubbing strake along the sheer.
        let rail: Vec<Vec3> = zs.iter().map(|&z| v3(s * (sheer_w(t_of(z)) + 0.015), sheer_y(t_of(z)) - 0.015, z)).collect();
        m.tube(&rail, 0.035, 6, [0.25, 0.26, 0.28], TRIM);
    }
    // Transom.
    {
        let t = 1.0;
        let mut pts = vec![v3(0.0, keel_y(t), STERN)];
        for k in 0..=5 {
            let x = chine_w(t) * k as f64 / 5.0;
            pts.push(v3(x, keel_y(t) + x * deadrise(t).tan(), STERN));
        }
        pts.push(v3(sheer_w(t), sheer_y(t), STERN));
        pts.push(v3(0.0, sheer_y(t), STERN));
        let mirror: Vec<Vec3> = pts.iter().rev().map(|p| Vec3::new(-p.x, p.y, p.z)).collect();
        // (two halves, each convex enough)
        m.poly(&pts.iter().rev().copied().collect::<Vec<_>>(), [0.92; 3], TRIM);
        m.poly(&mirror.iter().rev().copied().collect::<Vec<_>>(), [0.92; 3], TRIM);
    }
    // Deck: the foredeck crowned over the bow, the cockpit sunk behind it.
    let cockpit_z = -0.9;
    let fore: Vec<Vec<Vec3>> = zs
        .iter()
        .filter(|&&z| z <= cockpit_z + 1e-9)
        .map(|&z| {
            let t = t_of(z);
            (0..=6).map(|k| {
                let f = -1.0 + 2.0 * k as f64 / 6.0;
                let w = sheer_w(t);
                v3(f * w, sheer_y(t) + 0.07 * (1.0 - f * f) * (w / BEAM), z)
            }).collect()
        })
        .collect();
    if fore.len() >= 2 {
        m.grid(&fore, DECK, TRIM, true);
    }
    // Cockpit: floor, and the inner faces of the sides and the bulkheads.
    let floor_y = -0.05;
    let inner = 0.12;
    let cz: Vec<f64> = zs.iter().copied().filter(|&z| z >= cockpit_z).collect();
    let floor: Vec<Vec<Vec3>> = cz.iter().map(|&z| (0..=2).map(|k| v3((-1.0 + k as f64) * (sheer_w(t_of(z)) - inner), floor_y, z)).collect()).collect();
    m.grid(&floor, FLOOR, TRIM, true);
    for s in [-1.0, 1.0] {
        let wall: Vec<Vec<Vec3>> = cz.iter().map(|&z| vec![v3(s * (sheer_w(t_of(z)) - inner), floor_y, z), v3(s * (sheer_w(t_of(z)) - inner), sheer_y(t_of(z)), z)]).collect();
        m.grid(&wall, DECK, TRIM, s > 0.0);
        // the top of the gunwale, inner wall to the rail
        let top: Vec<Vec<Vec3>> = cz.iter().map(|&z| vec![v3(s * (sheer_w(t_of(z)) - inner), sheer_y(t_of(z)), z), v3(s * sheer_w(t_of(z)), sheer_y(t_of(z)), z)]).collect();
        m.grid(&top, DECK, TRIM, s < 0.0);
    }
    {
        // the bulkhead under the foredeck's aft edge, and the one at the transom
        let t = t_of(cockpit_z);
        let w = sheer_w(t) - inner;
        m.poly(&[v3(-w, floor_y, cockpit_z), v3(w, floor_y, cockpit_z), v3(w, sheer_y(t), cockpit_z), v3(-w, sheer_y(t), cockpit_z)], DECK, TRIM);
        let w = sheer_w(1.0) - inner;
        let zt = STERN - 0.55;
        m.poly(&[v3(w, floor_y, zt), v3(-w, floor_y, zt), v3(-w, sheer_y(1.0), zt), v3(w, sheer_y(1.0), zt)], DECK, TRIM);
        // the splash well's top behind it, a step down to the motor
        m.poly(&[v3(-w, sheer_y(1.0), zt), v3(w, sheer_y(1.0), zt), v3(w, sheer_y(1.0), STERN), v3(-w, sheer_y(1.0), STERN)], DECK, TRIM);
        // a sun pad over the stern
        m.rbox(Vec3::new(-0.95, sheer_y(1.0) as f32 - 0.02, zt as f32 - 0.9), Vec3::new(0.95, sheer_y(1.0) as f32 + 0.12, zt as f32), 0.06, SEAT, TRIM);
        m.cube(Vec3::new(-0.95, floor_y as f32, zt as f32 - 0.9), Vec3::new(0.95, sheer_y(1.0) as f32, zt as f32 - 0.05), DECK, TRIM);
    }
    // Centre console with its wheel, and the windshield wrapped round its front.
    let (c0, c1) = (-0.35f32, 0.35f32);
    let cy0 = floor_y as f32;
    m.rbox(Vec3::new(-0.42, cy0, c0), Vec3::new(0.42, cy0 + 0.92, c1), 0.12, DECK, TRIM);
    m.poly(&[Vec3::new(-0.42, cy0 + 0.92, c1), Vec3::new(0.42, cy0 + 0.92, c1), Vec3::new(0.42, cy0 + 1.02, c0 + 0.1), Vec3::new(-0.42, cy0 + 1.02, c0 + 0.1)], DARK, TRIM);
    let wheel: Vec<Vec3> = (0..=16).map(|k| {
        let a = k as f32 / 16.0 * std::f32::consts::TAU;
        Vec3::new(0.18 * a.cos(), cy0 + 1.08 + 0.16 * a.sin() * 0.8, c1 + 0.02 + 0.16 * a.sin() * 0.55)
    }).collect();
    m.tube(&wheel, 0.018, 5, DARK, TRIM);
    // Windshield: a curved sheet leaning back from the console's front.
    let ws: Vec<Vec<Vec3>> = (0..=8)
        .map(|k| {
            let a = -1.0 + 2.0 * k as f32 / 8.0;
            let x = 0.52 * a;
            let z = c0 - 0.06 + 0.18 * a * a;
            vec![Vec3::new(x, cy0 + 0.92, z), Vec3::new(x * 0.93, cy0 + 1.38, z + 0.3)]
        })
        .collect();
    m.grid(&ws, [0.4, 0.45, 0.5], GLASS, false);
    let frame: Vec<Vec3> = (0..=8).map(|k| {
        let a = -1.0 + 2.0 * k as f32 / 8.0;
        Vec3::new(0.52 * a * 0.93, cy0 + 1.38, c0 - 0.06 + 0.18 * a * a + 0.3)
    }).collect();
    m.tube(&frame, 0.016, 5, STEEL, TRIM);
    // Two helm seats behind the console, a bench forward of it.
    for x in [-0.45f32, 0.45] {
        m.rbox(Vec3::new(x - 0.2, cy0, 0.95), Vec3::new(x + 0.2, cy0 + 0.5, 1.25), 0.06, DECK, TRIM);
        m.rbox(Vec3::new(x - 0.27, cy0 + 0.46, 0.8), Vec3::new(x + 0.27, cy0 + 0.64, 1.38), 0.07, SEAT, TRIM);
        m.rbox(Vec3::new(x - 0.27, cy0 + 0.6, 1.28), Vec3::new(x + 0.27, cy0 + 1.15, 1.44), 0.07, SEAT, TRIM);
    }
    m.rbox(Vec3::new(-0.6, cy0, -0.85), Vec3::new(0.6, cy0 + 0.45, -0.45), 0.08, SEAT, TRIM);
    // Bow rail and the navigation lights.
    for s in [-1.0f32, 1.0] {
        let pts: Vec<Vec3> = (0..=8)
            .map(|k| {
                let z = BOW + 0.35 + (cockpit_z - 0.2 - BOW - 0.35) * k as f64 / 8.0;
                let t = t_of(z);
                v3(s as f64 * (sheer_w(t) - 0.06), sheer_y(t) + 0.32, z)
            })
            .collect();
        m.tube(&pts, 0.016, 5, STEEL, TRIM);
        for k in [1usize, 4, 8] {
            let p = pts[k];
            m.tube(&[Vec3::new(p.x, p.y - 0.33, p.z), p], 0.014, 4, STEEL, TRIM);
        }
        let t = t_of(BOW + 0.6);
        let nl = v3(s as f64 * (sheer_w(t) * 0.9), sheer_y(t) + 0.03, BOW + 0.6);
        m.cube(nl - Vec3::new(0.03, 0.0, 0.05), nl + Vec3::new(0.03, 0.05, 0.05), if s < 0.0 { [0.9, 0.05, 0.05] } else { [0.05, 0.85, 0.2] }, TAIL);
    }
    {
        let t = t_of(BOW + 0.35);
        let a = v3(0.0, sheer_y(t) + 0.32, BOW + 0.35);
        let b = v3(0.0, sheer_y(t) + 0.32, BOW + 0.35);
        let _ = b;
        m.tube(&[Vec3::new(-(sheer_w(t) as f32 - 0.06), a.y, a.z), Vec3::new(sheer_w(t) as f32 - 0.06, a.y, a.z)], 0.016, 5, STEEL, TRIM);
        // the white light on its pole at the stern
        let p = Vec3::new(0.7, sheer_y(1.0) as f32, STERN as f32 - 0.6);
        m.tube(&[p, p + Vec3::Y * 0.9], 0.015, 4, STEEL, TRIM);
        m.cube(p + Vec3::new(-0.03, 0.9, -0.03), p + Vec3::new(0.03, 0.97, 0.03), [1.0, 1.0, 0.95], LAMP);
    }
    let hull = m.done();

    // The outboard, about its swivel (`MOTOR`): cowling, the leg, the lower
    // unit with its skeg and cavitation plate, and the propeller.
    let mut o = MB::default();
    let at = |x: f32, y: f32, z: f32| Vec3::new(x, y, z) - MOTOR.as_vec3();
    let cow: Vec<Vec<Vec3>> = (0..=6)
        .map(|k| {
            let f = k as f32 / 6.0;
            let y = 0.35 + 0.72 * f;
            let r = 0.24 * (1.0 - 0.75 * (f - 0.55).powi(2) * 3.0).max(0.4);
            (0..=10).map(|j| {
                let a = j as f32 / 10.0 * std::f32::consts::TAU;
                at(r * a.cos() * 0.85, y, STERN as f32 + 0.38 + r * 1.4 * a.sin())
            }).collect()
        })
        .collect();
    o.grid(&cow, [0.92, 0.92, 0.92], PAINT, true);
    o.poly(&cow[6].iter().rev().copied().collect::<Vec<_>>(), [0.92; 3], PAINT);
    o.cube(at(-0.1, -0.5, STERN as f32 + 0.25), at(0.1, 0.4, STERN as f32 + 0.5), DARK, TRIM);
    o.cube(at(-0.2, -0.45, STERN as f32 + 0.12), at(0.2, -0.42, STERN as f32 + 0.62), DARK, TRIM);
    let torp: Vec<Vec3> = (0..=6).map(|k| at(0.0, -0.62, STERN as f32 + 0.15 + 0.5 * k as f32 / 6.0)).collect();
    o.tube(&torp, 0.08, 8, DARK, TRIM);
    o.cube(at(-0.015, -0.85, STERN as f32 + 0.4), at(0.015, -0.6, STERN as f32 + 0.62), DARK, TRIM);
    for b in 0..3 {
        let a = b as f32 / 3.0 * std::f32::consts::TAU;
        let (c, s) = (a.cos(), a.sin());
        let z = STERN as f32 + 0.68;
        o.poly(&[at(0.03 * c, -0.62 + 0.03 * s, z), at(0.17 * c - 0.08 * s, -0.62 + 0.17 * s + 0.08 * c, z + 0.03), at(0.18 * c + 0.06 * s, -0.62 + 0.18 * s - 0.06 * c, z - 0.02)], STEEL, TRIM);
    }
    o.cube(at(-0.03, -0.66, STERN as f32 + 0.62), at(0.03, -0.58, STERN as f32 + 0.72), DARK, TRIM);
    [hull, o.done()]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn calm(_: DVec3) -> Surface {
        Surface { h: 0.0, water: true }
    }

    fn afloat() -> Boat {
        let pos = crate::geo::lla_to_pos(0.6, -2.0, 0.2);
        Boat::new(pos, 0.3)
    }

    /// Dropped into calm water, it settles level at a sensible draft, its
    /// weight carried by the water it displaces.
    #[test]
    fn floats_level() {
        let mut b = afloat();
        let h = 1.0 / 180.0;
        for _ in 0..(25.0 / h) as usize {
            b.step(h, &Input::default(), &mut calm);
        }
        let up = b.pos.normalize();
        let keel = b.pos + b.rot * DVec3::new(0.0, keel_y(1.0), STERN);
        let draft = -(keel.length() - crate::geo::R);
        let pitch = b.forward().dot(up).asin().to_degrees();
        let roll = (b.rot * DVec3::X).dot(up).asin().to_degrees();
        eprintln!("draft at the transom {draft:.3} m, pitch {pitch:.2}°, roll {roll:.2}°, v {:.4}", b.vel.length());
        assert!(draft > 0.2 && draft < 0.6, "draft {draft}");
        assert!(pitch.abs() < 4.0 && roll.abs() < 0.5);
        assert!(b.vel.length() < 0.02);
        // Archimedes: the columns under water carry the weight.
        let disp: f64 = b.cols.iter().map(|c| {
            let p = b.pos + b.rot * c.at;
            (-(p.length() - crate::geo::R)).clamp(0.0, c.top - c.at.y) * c.area
        }).sum();
        assert!((disp * RHO / MASS - 1.0).abs() < 0.03, "displaced {disp} m³");
    }

    /// Full throttle on calm water: it gets up on the plane (rises, trims
    /// bow up a few degrees) and runs at a speedboat's speed.
    #[test]
    fn planes() {
        let mut b = afloat();
        let h = 1.0 / 180.0;
        let up0 = b.pos.normalize();
        let z0 = b.pos.length();
        for _ in 0..(10.0 / h) as usize {
            b.step(h, &Input::default(), &mut calm);
        }
        let rest = b.pos.length() - z0;
        let input = Input { throttle: 1.0, steer: 0.0, boost: false };
        for _ in 0..(60.0 / h) as usize {
            b.step(h, &input, &mut calm);
        }
        let kt = b.speed() * 1.94384;
        let trim = b.forward().dot(b.pos.normalize()).asin().to_degrees();
        let lift = b.pos.length() - z0 - rest;
        eprintln!("full throttle: {kt:.1} kt, trim {trim:.2}°, risen {lift:.3} m, wet {:.2}", b.wet);
        assert!(kt > 30.0 && kt < 55.0, "{kt} kt");
        assert!(trim > 1.0 && trim < 8.0, "trim {trim}");
        assert!(lift > 0.05, "did not rise onto the plane");
        let _ = up0;
        // Hard over at speed: it turns, and stays upright.
        let input = Input { throttle: 1.0, steer: 1.0, boost: false };
        let hdg = |b: &Boat| {
            let (e, n, _) = crate::geo::enu(b.pos);
            b.forward().dot(e).atan2(b.forward().dot(n))
        };
        let (h0, mut turned, mut prev) = (hdg(&b), 0.0, hdg(&b));
        let mut worst_roll = 0.0f64;
        for k in 0..(4.0 / h) as usize {
            b.step(h, &input, &mut calm);
            let a = hdg(&b);
            turned += (a - prev + std::f64::consts::PI).rem_euclid(std::f64::consts::TAU) - std::f64::consts::PI;
            prev = a;
            let roll = (b.rot * DVec3::X).dot(b.pos.normalize()).asin().to_degrees();
            worst_roll = if roll.abs() > worst_roll.abs() { roll } else { worst_roll };
            if k % 90 == 0 {
                eprintln!("  t {:.1}: turned {:.0}°, roll {roll:.1}° (+ right side up), {:.1} kt", k as f64 * h, turned.to_degrees(), b.speed() * 1.94384);
            }
        }
        let _ = h0;
        eprintln!("hard to starboard 4 s: turned {:.0}°, worst roll {worst_roll:.1}°", turned.to_degrees());
        assert!(turned.to_degrees() > 40.0, "turned the wrong way or not enough: {}", turned.to_degrees());
        // (it banks into the turn: the right side goes down)
        assert!(worst_roll < 2.0 && worst_roll > -30.0, "roll {worst_roll}");
    }

    /// Full throttle into a moderate sea: it rides it, in the water most of
    /// the time (no endless flying), and stays upright.
    #[test]
    fn rides_waves() {
        for wind in [7.0, 11.0] {
            let mut o = crate::ocean::Ocean::new();
            let mut b = afloat();
            let (e, n, _) = crate::geo::enu(b.pos);
            let cam = b.pos;
            let h = 1.0 / 180.0;
            let (mut air, mut worst_roll, mut worst_trim, mut tot, mut v) = (0.0, 0.0f64, 0.0f64, 0.0, 0.0);
            for k in 0..(70.0 / h) as usize {
                let t = k as f64 * h;
                if k % 3 == 0 {
                    o.update(h * 3.0, t, wind, 300.0, 0.6, cam, e, n, glam::DVec2::ZERO);
                }
                let mut sea = |p: DVec3| Surface { h: o.height_w(p, [1.0, 1.0]), water: true };
                let input = Input { throttle: if t > 5.0 { 1.0 } else { 0.0 }, steer: 0.0, boost: false };
                b.step(h, &input, &mut sea);
                if t > 25.0 {
                    tot += h;
                    if b.wet < 0.01 {
                        air += h;
                    }
                    let up = b.pos.normalize();
                    worst_roll = worst_roll.max((b.rot * DVec3::X).dot(up).asin().to_degrees().abs());
                    worst_trim = worst_trim.max(b.forward().dot(up).asin().to_degrees().abs());
                    v += b.speed() * h;
                }
            }
            eprintln!("wind {wind}: Hs {:.1} m, {:.1} kt, airborne {:.0}%, worst trim {worst_trim:.1}°, roll {worst_roll:.1}°", o.hs, v / tot * 1.94384, air / tot * 100.0);
            assert!(air / tot < 0.25 && worst_roll < 35.0 && worst_trim < 25.0);
        }
    }

    /// The mesh holds together: indices in range, normals set.
    #[test]
    fn model() {
        for (v, i) in meshes() {
            assert!(!v.is_empty() && i.len() % 3 == 0);
            assert!(i.iter().all(|&k| (k as usize) < v.len()));
        }
    }
}
