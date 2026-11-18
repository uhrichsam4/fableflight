//! Destruction mode: twin lasers, a meteor shower, a wrecking ball on a
//! chain under the aircraft and a count of the player's deaths; and mouse
//! steering (War Thunder's "mouse aim"), which works in every mode.
//!
//! What breaks is broken by destruct.rs: this module decides what was hit
//! and how hard, and hands the game requests (`Req`) for the parts only it
//! can do (taking a building out of the world, craters, killing the pilot).

use crate::airfield::Light;
use crate::crash::{Rng, DUST, FLASH, GAS, GRIT, GLASS, SPARK, DROPS};
use crate::geo::{self, TileId};
use crate::render::{Inst, MeshVertex, Particle};
use crate::world::World;
use glam::{DQuat, DVec2, DVec3, Mat3, Quat};
use std::collections::{HashMap, VecDeque};

/// Free flight (the sim as it always was) or Destruction.
#[derive(Copy, Clone, PartialEq, Eq, Debug, Default)]
pub enum GameMode {
    #[default]
    Free,
    Destruction,
}

impl GameMode {
    pub const ALL: [GameMode; 2] = [GameMode::Free, GameMode::Destruction];
    pub fn code(self) -> &'static str {
        match self {
            GameMode::Free => "free",
            GameMode::Destruction => "destruction",
        }
    }
    pub fn parse(s: &str) -> Option<GameMode> {
        let s = s.trim().to_ascii_lowercase();
        match s.as_str() {
            "free" | "freeflight" | "free-flight" | "flight" => Some(GameMode::Free),
            "destruction" | "destroy" | "destruct" => Some(GameMode::Destruction),
            _ => None,
        }
    }
}

/// How far the lasers reach (m), how fast their bolts are drawn going, and
/// how often the pair fires (s).
pub const RANGE: f64 = 1600.0;
const BOLT_SPEED: f64 = 1500.0;
const FIRE_EVERY: f64 = 1.0 / 7.0;
/// Within this angle of the nose (rad) the lasers swing to the mouse's aim point.
const GIMBAL: f64 = 0.21;
/// The meteor shower's settings: seconds between meteors, by rate (0 off).
pub const METEOR_EVERY: [f64; 4] = [0.0, 6.0, 2.6, 0.9];
/// Seconds after a death before the aircraft comes back (Destruction mode).
pub const RESPAWN_AFTER: f64 = 3.0;

// ---------------------------------------------------------------- mouse aim

/// Turns the aim direction by a mouse movement (pixels): sideways round the
/// local vertical, up and down toward it, never quite straight up or down.
pub fn turn_aim(dir: DVec3, up: DVec3, dx: f64, dy: f64, sens: f64) -> DVec3 {
    let k = 0.0022 * sens;
    let d = DQuat::from_axis_angle(up, -dx * k) * dir;
    let right = d.cross(up).normalize_or_zero();
    if right == DVec3::ZERO {
        return d;
    }
    let el = d.dot(up).clamp(-1.0, 1.0).asin();
    let to = (el - dy * k).clamp(-1.45, 1.45);
    let flat = (d - up * d.dot(up)).normalize_or_zero();
    (flat * to.cos() + up * to.sin()).normalize()
}

fn smooth(a: f64, b: f64, x: f64) -> f64 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn wrap(a: f64) -> f64 {
    (a + std::f64::consts::PI).rem_euclid(std::f64::consts::TAU) - std::f64::consts::PI
}

/// The "instructor" of mouse aim: stick pitch, roll and yaw (each -1..1, as
/// `plane::Controls`) that bring the nose round onto `aim`. `rot` is the
/// aircraft's attitude (body: x right, y up, -z forward), `omega` its body
/// rates (rad/s, as `Plane::omega`), `up` the local vertical, `speed` its
/// airspeed (m/s). Small errors
/// are flown out with a bank toward the target, the elevator and a touch of
/// rudder; big ones (beyond about 20°) by rolling the target overhead and
/// pulling, as a pilot would. `g_room` (0..1) eases the pull off near the
/// airframe's g limit.
pub fn instructor(rot: DQuat, omega: DVec3, up: DVec3, aim: DVec3, speed: f64, g_room: f64) -> (f64, f64, f64) {
    let inv = rot.inverse();
    let l = inv * aim.normalize_or(DVec3::NEG_Z);
    let ub = inv * up;
    let fwd = -l.z;
    let off = fwd.clamp(-1.0, 1.0).acos();
    let pitch_err = l.y.atan2(fwd.max(0.0) + 1e-6 * (fwd <= 0.0) as u8 as f64).clamp(-1.4, 1.4);
    let yaw_err = l.x.atan2(fwd.abs().max(1e-6));
    // (+ = right wing down)
    let bank = (-ub.x).atan2(ub.y);
    // Roll that puts the target straight overhead (for a big turn: roll, then pull).
    let over = l.x.atan2(l.y);
    // Close in: a bank into the turn, as much as it needs, and the nose
    // pulled or pushed onto it.
    // (the bank that turns at 1.2 times the error a second, at this speed)
    let level = (yaw_err * 1.2 * speed.max(20.0) / 9.81).atan().clamp(-1.15, 1.15) - bank;
    let w = smooth(0.3, 0.6, off);
    let roll_err = wrap(level + wrap(over - level) * w);
    // (the rate of roll to the right is -omega.z)
    let roll = (roll_err * 2.4 + omega.z * 0.45).clamp(-1.0, 1.0);
    // Pull (or push) once the target is roughly in the plane of the wings' lift.
    let lined = (1.0 - roll_err.abs() / 1.6).clamp(0.0, 1.0);
    let mut pitch = pitch_err * 4.5 * (0.25 + 0.75 * lined) - omega.x * 0.9;
    pitch = pitch.clamp(-0.6, 1.0) * if pitch > 0.0 { g_room.clamp(0.0, 1.0) } else { 1.0 };
    // The rudder trims out the last of it (yaw right rate is -omega.y).
    let yaw = ((yaw_err * 3.0).clamp(-1.0, 1.0) * (1.0 - 0.8 * w) + omega.y * 0.6).clamp(-1.0, 1.0);
    (pitch.clamp(-1.0, 1.0), roll, yaw)
}

// ---------------------------------------------------------------- ray tests

/// Distance along a ray (origin `o`, unit direction `d`, both in a local frame
/// with z up) to where it first enters an upright prism: the outline `ring`
/// (x, y) from `z0` up to `z1`. Zero if it starts inside; None if it misses
/// within `tmax`. The outline need not be convex.
pub fn ray_prism(o: DVec3, d: DVec3, ring: &[DVec2], z0: f64, z1: f64, tmax: f64) -> Option<f64> {
    let n = ring.len();
    if n < 3 {
        return None;
    }
    let inside = |p: DVec2| {
        let mut c = false;
        let mut j = n - 1;
        for i in 0..n {
            let (a, b) = (ring[i], ring[j]);
            if (a.y > p.y) != (b.y > p.y) && p.x < (b.x - a.x) * (p.y - a.y) / (b.y - a.y) + a.x {
                c = !c;
            }
            j = i;
        }
        c
    };
    let (o2, d2) = (o.truncate(), d.truncate());
    if o.z >= z0 && o.z <= z1 && inside(o2) {
        return Some(0.0);
    }
    let mut best = tmax;
    let mut hit = false;
    if d.z.abs() > 1e-9 {
        for z in [z0, z1] {
            let t = (z - o.z) / d.z;
            if t >= 0.0 && t < best && inside(o2 + d2 * t) {
                best = t;
                hit = true;
            }
        }
    }
    for i in 0..n {
        let (a, b) = (ring[i], ring[(i + 1) % n]);
        let e = b - a;
        let den = d2.perp_dot(e);
        if den.abs() < 1e-12 {
            continue;
        }
        let w = a - o2;
        let t = w.perp_dot(e) / den;
        let s = w.perp_dot(d2) / den;
        if t >= 0.0 && t < best && (0.0..=1.0).contains(&s) {
            let z = o.z + d.z * t;
            if z >= z0 && z <= z1 {
                best = t;
                hit = true;
            }
        }
    }
    hit.then_some(best)
}

/// Ray (unit `d`) against a sphere: distance to where it enters, if within `tmax`.
pub fn ray_sphere(o: DVec3, d: DVec3, c: DVec3, r: f64, tmax: f64) -> Option<f64> {
    let oc = o - c;
    let b = oc.dot(d);
    let q = oc.length_squared() - r * r;
    if q <= 0.0 {
        return Some(0.0);
    }
    let disc = b * b - q;
    if disc < 0.0 || b > 0.0 {
        return None;
    }
    let t = -b - disc.sqrt();
    (t >= 0.0 && t <= tmax).then_some(t)
}

/// What a laser bolt (or a meteor) ends at.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Target {
    Ground { water: bool },
    Building(TileId, usize),
    Meteor(u32),
}

/// A building that was hit, where and how: the game knocks a few pieces
/// off it (`Destruction::chip`) or, past its strength, brings it down.
pub enum Req {
    /// a laser or the ball took a bite out of a wall: where, outward, how hard (0..1+)
    Chip { tile: TileId, idx: usize, at: DVec3, out: DVec3, hard: f32 },
    /// knock the whole building down
    BringDown { tile: TileId, idx: usize, at: DVec3, vel: DVec3 },
    /// a small dent in open ground (merged into one nearby) or a crater
    Crater { at: DVec3, r: f32, small: bool },
    /// a meteor shot to pieces in the air: its rock in bits
    Rocks { at: DVec3, vel: DVec3, r: f32 },
    /// a meteor coming down: crater, blast, everything near it
    Blast { at: DVec3, vel: DVec3, r: f32, building: Option<(TileId, usize)>, water: bool },
    /// the pilot is dead: why (as the damage messages say it)
    Kill(&'static str),
    Sound(crate::audio::Event),
    /// an explosion's sound: how big, where
    Boom(f32, DVec3),
}

struct Bolt {
    pos: DVec3,
    dir: DVec3,
    /// distance left to the target (or the end of the range)
    left: f64,
    target: Option<Target>,
    col: [f32; 3],
    travelled: f64,
}

pub struct Meteor {
    pub id: u32,
    pub pos: DVec3,
    pub vel: DVec3,
    pub r: f64,
    rot: DQuat,
    spin: DVec3,
    hp: f32,
    mesh: u32,
    fx: f32,
    age: f32,
}

/// The wrecking ball: steel, on a chain from the aircraft's belly.
pub struct Ball {
    pub pos: DVec3,
    pub vel: DVec3,
    pub r: f64,
    pub m: f64,
    pub len: f64,
    /// where the chain is fastened, as of the last step
    top: DVec3,
    /// the chain's links (verlet points from top to ball), for drawing
    links: Vec<DVec3>,
    links_prev: Vec<DVec3>,
    /// seconds before the same building can be hit again
    cool: HashMap<(TileId, usize), f32>,
    ground_cool: f32,
    /// time spent wedged against a wall
    stuck: f64,
}

/// The death count, the respawn and the session (see `Destroy`).
#[derive(Default)]
pub struct Tally {
    pub deaths: u32,
    pub destroyed: u32,
    /// game time this life ended (None: alive)
    pub dead_at: Option<f64>,
}

impl Tally {
    /// A death (once per life); true if it was a new one.
    pub fn died(&mut self, now: f64) -> bool {
        if self.dead_at.is_some() {
            return false;
        }
        self.deaths += 1;
        self.dead_at = Some(now);
        true
    }
    /// A new life.
    pub fn revived(&mut self) {
        self.dead_at = None;
    }
    /// A new session (a start from the globe): the counts start again.
    pub fn new_session(&mut self) {
        *self = Tally::default();
    }
}

pub struct Destroy {
    pub mode: GameMode,
    pub tally: Tally,
    /// meteor shower 0 off, 1 light, 2 heavy, 3 storm
    pub meteor_rate: u8,
    pub ball_on: bool,
    /// the trigger held (and the pair of lasers firing), this frame
    pub firing: bool,
    bolts: Vec<Bolt>,
    pub meteors: Vec<Meteor>,
    pub ball: Option<Ball>,
    /// force of the chain on the aircraft (N, planet frame), for its next step
    pub ball_force: DVec3,
    /// damage taken by standing buildings, against `strength`
    wear: HashMap<(TileId, usize), f32>,
    cool: f64,
    next_meteor: f64,
    next_id: u32,
    /// where the aircraft was in good health lately: (time, lat, lon, alt, heading, speed)
    trail: VecDeque<(f64, f64, f64, f64, f64, f64)>,
    pub reqs: Vec<Req>,
    rng: Rng,
    /// short glows: impact flashes, muzzle flashes (pos, age, life, colour, intensity)
    glows: Vec<(DVec3, f32, f32, [u8; 3], f32)>,
    /// the first of this module's meshes in the renderer's solids (`meshes`)
    pub mesh0: u32,
    time: f64,
}

/// Laser red: a white core with a red glow.
const LASER: [f32; 3] = [1.0, 0.16, 0.08];
const LASER_LIGHT: [u8; 3] = [255, 60, 40];

/// How much a building takes before it comes down, in laser hits: a house a
/// few, a tower about eighty.
pub fn strength(area: f32, height: f64) -> f32 {
    (area * height as f32 / 1500.0).clamp(4.0, 80.0)
}

/// The renderer mesh this module adds to the solids: a smooth sphere (the
/// wrecking ball, and a meteor's white-hot front).
pub fn meshes() -> Vec<(Vec<MeshVertex>, Vec<u32>)> {
    let (vs, fs) = crate::trees::ico(3);
    let v = vs.iter().map(|p| MeshVertex { pos: p.normalize().into(), nrm: p.normalize().into(), col: [1.0, 1.0, 1.0, 0.0] }).collect();
    vec![(v, fs.iter().flat_map(|f| [f[0], f[1], f[2]]).collect())]
}

/// What the module needs to know of the aircraft each frame.
pub struct Craft {
    pub pos: DVec3,
    pub vel: DVec3,
    pub fwd: DVec3,
    /// how far round it a meteor touches it (m)
    pub radius: f64,
    pub alive: bool,
}

impl Destroy {
    pub fn new(seed: u64) -> Self {
        Self {
            mode: GameMode::Free,
            tally: Tally::default(),
            meteor_rate: 0,
            ball_on: false,
            firing: false,
            bolts: Vec::new(),
            meteors: Vec::new(),
            ball: None,
            ball_force: DVec3::ZERO,
            wear: HashMap::new(),
            cool: 0.0,
            next_meteor: 2.0,
            next_id: 1,
            trail: VecDeque::new(),
            reqs: Vec::new(),
            rng: Rng(seed | 1),
            glows: Vec::new(),
            mesh0: 0,
            time: 0.0,
        }
    }

    pub fn on(&self) -> bool {
        self.mode == GameMode::Destruction
    }

    /// Everything flying is let go (a new start, or Free flight).
    pub fn clear(&mut self) {
        self.bolts.clear();
        self.meteors.clear();
        self.glows.clear();
        self.trail.clear();
        self.ball = None;
        self.ball_force = DVec3::ZERO;
        self.reqs.clear();
        self.firing = false;
    }

    /// Remember where the aircraft is, flying well (for coming back after a death).
    pub fn note_good(&mut self, time: f64, pos: DVec3, fwd: DVec3, speed: f64, agl: f64) {
        if self.trail.back().is_some_and(|b| time - b.0 < 0.5) || agl < 25.0 {
            return;
        }
        let (lat, lon, alt) = geo::pos_to_lla(pos);
        let (e, n, _) = geo::enu(pos);
        let hdg = fwd.dot(e).atan2(fwd.dot(n));
        self.trail.push_back((time, lat, lon, alt, hdg, speed));
        while self.trail.len() > 120 {
            self.trail.pop_front();
        }
    }

    /// Where to come back after a death at `died`: four seconds before it, a little higher,
    /// as it was then (lat, lon, alt, heading, speed).
    pub fn respawn_spot(&self, died: f64) -> Option<(f64, f64, f64, f64, f64)> {
        let mut it = self.trail.iter().rev().filter(|s| s.0 < died - 4.0);
        it.next().or_else(|| self.trail.front()).map(|s| (s.1, s.2, s.3 + 40.0, s.4, s.5))
    }

    /// The meteor shower's setting, as words.
    pub fn rate_name(rate: u8) -> &'static str {
        match rate {
            0 => crate::i18n::t!("off", "désactivée"),
            1 => crate::i18n::t!("light", "légère"),
            2 => crate::i18n::t!("heavy", "forte"),
            _ => crate::i18n::t!("storm", "tempête"),
        }
    }

    /// The first thing a ray hits within `len`: distance and what.
    pub fn trace(&self, world: &World, from: DVec3, dir: DVec3, len: f64) -> (f64, Option<Target>) {
        let mut tmax = len;
        let mut tgt = None;
        if let Some((t, tile, idx)) = world.ray_building(from, dir, len) {
            tmax = t;
            tgt = Some(Target::Building(tile, idx));
        }
        for m in &self.meteors {
            if let Some(t) = ray_sphere(from, dir, m.pos, m.r * 1.3, tmax) {
                tmax = t;
                tgt = Some(Target::Meteor(m.id));
            }
        }
        // The ground (and under the 3D tiles, its mesh): marched, in steps
        // as long as the height above it allows, then pinned down.
        let mut t = 1.0;
        let mut last = 0.0;
        while t < tmax {
            let p = from + dir * t;
            let (la, lo, h) = geo::pos_to_lla(p);
            let Some(g) = world.ground(la, lo) else { break };
            if h <= g.h {
                let (mut a, mut b) = (last, t);
                for _ in 0..6 {
                    let m = (a + b) * 0.5;
                    let (la, lo, h) = geo::pos_to_lla(from + dir * m);
                    if world.ground(la, lo).is_some_and(|g| h <= g.h) {
                        b = m;
                    } else {
                        a = m;
                    }
                }
                let (la, lo, _) = geo::pos_to_lla(from + dir * b);
                let g = world.ground(la, lo).unwrap_or(g);
                let what = match world.photo_building(la, lo, g.h) {
                    Some(bh) => Target::Building(bh.tile, bh.idx),
                    None => Target::Ground { water: g.water },
                };
                return (b, Some(what));
            }
            // (well above anything and climbing: nothing more to hit)
            if h - g.h > 3000.0 && dir.dot(p.normalize()) > 0.0 {
                break;
            }
            last = t;
            t += ((h - g.h) * 0.5).clamp(3.0, 60.0);
        }
        (tmax, tgt)
    }

    /// Fire the pair, if it is time: from the two muzzles at the point the
    /// aim ray meets (when it is near the nose), else straight ahead.
    pub fn fire(&mut self, world: &World, muzzles: [DVec3; 2], eye: DVec3, nose: DVec3, aim: Option<DVec3>, emit: &mut Vec<(DVec3, Particle)>) {
        if self.cool > 0.0 {
            return;
        }
        self.cool += FIRE_EVERY;
        let dir = match aim {
            Some(a) if a.dot(nose).clamp(-1.0, 1.0).acos() < GIMBAL => a,
            _ => nose,
        };
        let (t, _) = self.trace(world, eye, dir, RANGE);
        // (nothing in the way: they cross a little short of the range)
        let meet = eye + dir * t.min(RANGE * 0.6);
        for m in muzzles {
            let d = (meet - m).normalize_or(dir);
            let (t, tgt) = self.trace(world, m, d, RANGE);
            self.bolts.push(Bolt { pos: m, dir: d, left: t, target: tgt, col: LASER, travelled: 0.0 });
            self.glows.push((m + d * 1.5, 0.0, 0.08, LASER_LIGHT, 3_000.0));
            let r = &mut self.rng;
            emit.push((m + d * 1.0, Particle { life: 0.06, size0: 0.9, size1: 1.6, kind: FLASH, alpha: 1.2, seed: r.f(), ..Default::default() }));
        }
        self.reqs.push(Req::Sound(crate::audio::Event::Laser(1.0)));
    }

    /// One frame: bolts fly and land, the meteors fall, the glows fade.
    pub fn update(&mut self, dt: f64, world: &World, craft: &Craft, emit: &mut Vec<(DVec3, Particle)>) {
        self.time += dt;
        self.cool = (self.cool - dt).max(0.0);
        if !self.firing {
            self.cool = 0.0;
        }
        for g in &mut self.glows {
            g.1 += dt as f32;
        }
        self.glows.retain(|g| g.1 < g.2);
        // ---- the bolts
        let mut landed = Vec::new();
        for (i, b) in self.bolts.iter_mut().enumerate() {
            let step = BOLT_SPEED * dt;
            b.travelled += step;
            if step >= b.left {
                b.pos += b.dir * b.left;
                b.left = 0.0;
                landed.push(i);
            } else {
                b.pos += b.dir * step;
                b.left -= step;
            }
        }
        for &i in landed.iter().rev() {
            let b = self.bolts.swap_remove(i);
            if let Some(t) = b.target {
                self.bolt_hit(world, b.pos, b.dir, t, emit);
            }
        }
        // ---- the meteor shower
        if self.on() && self.meteor_rate > 0 && craft.alive {
            self.next_meteor -= dt;
            if self.next_meteor <= 0.0 {
                let every = METEOR_EVERY[self.meteor_rate.min(3) as usize];
                self.next_meteor = every * self.rng.range(0.5, 1.5) as f64;
                self.spawn_meteor(world, craft, None);
            }
        }
        self.meteors_step(dt, world, craft, emit);
    }

    fn bolt_hit(&mut self, world: &World, at: DVec3, dir: DVec3, t: Target, emit: &mut Vec<(DVec3, Particle)>) {
        let up = at.normalize();
        let back = -dir;
        self.glows.push((at + back * 0.8, 0.0, 0.25, [255, 140, 80], 30_000.0));
        let r = &mut self.rng;
        emit.push((at + back * 0.5, Particle { life: 0.16, size0: 3.0, size1: 7.0, kind: FLASH, alpha: 2.0, seed: r.f(), ..Default::default() }));
        // A spit of flame, a shower of sparks, and smoke that hangs where it hit.
        emit.push((at + back * 0.8, Particle { vel: (back.as_vec3() * 4.0).into(), life: r.range(0.4, 0.7), size0: 0.8, size1: r.range(2.2, 3.2), temp: r.range(1.8, 2.4), kind: GAS, drag: 2.0, buoy: 3.0, seed: r.f(), alpha: 0.85, ..Default::default() }));
        if r.f() < 0.2 {
            emit.push((at + back * 1.2, Particle { vel: (back.as_vec3() * 2.0).into(), life: r.range(3.0, 5.0), size0: 0.8, size1: r.range(3.0, 5.0), temp: 0.05, kind: GAS, drag: 1.2, buoy: 0.8, seed: r.f(), alpha: r.range(0.18, 0.3), ..Default::default() }));
        }
        for _ in 0..16 {
            let d = (back.as_vec3() * 0.8 + r.dir()).normalize();
            emit.push((at + back * 0.3, Particle { vel: (d * r.range(10.0, 40.0)).into(), life: r.range(0.4, 1.3), size0: r.range(0.08, 0.16), size1: 0.04, kind: SPARK, drag: r.range(0.3, 0.9), seed: r.f(), alpha: r.range(0.7, 1.0), ..Default::default() }));
        }
        match t {
            Target::Building(tile, idx) => {
                // Masonry dust and glass blown out of the wall.
                if r.f() < 0.5 {
                    let d = (back.as_vec3() + r.dir() * 0.6).normalize();
                    emit.push((at + back, Particle { vel: (d * r.range(2.0, 6.0)).into(), life: r.range(1.5, 3.0), size0: r.range(0.3, 0.6), size1: r.range(1.5, 3.0), kind: GRIT, drag: 1.4, buoy: 0.2, seed: r.f(), alpha: r.range(0.2, 0.35), ..Default::default() }));
                }
                for _ in 0..5 {
                    let d = (back.as_vec3() + r.dir() * 0.7).normalize();
                    emit.push((at + back * 0.4, Particle { vel: (d * r.range(4.0, 12.0)).into(), life: r.range(2.0, 4.0), size0: r.range(0.08, 0.18), size1: 0.1, kind: GLASS, drag: 0.3, seed: r.f(), alpha: 0.9, ..Default::default() }));
                }
                let Some((base, top, area)) = world.footprint(tile, idx) else { return };
                let w = self.wear.entry((tile, idx)).or_insert(0.0);
                *w += 1.0;
                let gone = *w >= strength(area, top - base);
                self.reqs.push(Req::Chip { tile, idx, at, out: back, hard: 0.6 });
                if gone {
                    self.wear.remove(&(tile, idx));
                    self.reqs.push(Req::BringDown { tile, idx, at, vel: dir * 40.0 });
                }
            }
            Target::Ground { water } => {
                if water {
                    for _ in 0..8 {
                        let d = (up.as_vec3() * 1.5 + r.dir()).normalize();
                        emit.push((at, Particle { vel: (d * r.range(4.0, 14.0)).into(), life: r.range(0.8, 1.8), size0: r.range(0.2, 0.4), size1: r.range(0.6, 1.4), kind: DROPS, drag: 0.4, seed: r.f(), alpha: 0.5, ..Default::default() }));
                    }
                } else {
                    for _ in 0..5 {
                        let d = (up.as_vec3() * 1.4 + r.dir()).normalize();
                        emit.push((at + up * 0.3, Particle { vel: (d * r.range(2.0, 9.0)).into(), life: r.range(1.5, 4.0), size0: r.range(0.4, 0.8), size1: r.range(2.0, 4.5), kind: DUST, drag: 1.4, buoy: 0.2, seed: r.f(), alpha: r.range(0.4, 0.65), ..Default::default() }));
                    }
                    self.reqs.push(Req::Crater { at, r: 1.2, small: true });
                }
            }
            Target::Meteor(id) => {
                if let Some(k) = self.meteors.iter().position(|m| m.id == id) {
                    self.meteors[k].hp -= 1.0;
                    if self.meteors[k].hp <= 0.0 {
                        let m = self.meteors.swap_remove(k);
                        self.break_up(&m, emit);
                    }
                }
            }
        }
    }

    /// A new meteor, coming down at an angle on a spot within a few km of
    /// the aircraft (mostly ahead of it); one in five at the aircraft itself.
    /// `at`: a given spot to come down on instead (scripted shots).
    pub fn spawn_meteor(&mut self, world: &World, craft: &Craft, at: Option<DVec3>) {
        let up = craft.pos.normalize();
        let (e, n, _) = geo::enu(craft.pos);
        let r = &mut self.rng;
        let speed = r.range(170.0, 300.0) as f64;
        let hunt = at.is_none() && r.f() < 0.2;
        let target = match at {
            Some(p) => p,
            None if hunt => craft.pos + craft.vel * 9.0,
            None => {
                let fwd = (craft.fwd - up * craft.fwd.dot(up)).normalize_or(n);
                let a = r.range(0.0, std::f32::consts::TAU) as f64;
                let d = r.range(150.0, 2600.0) as f64;
                let p = craft.pos + fwd * 900.0 + (e * a.cos() + n * a.sin()) * d;
                let (la, lo, _) = geo::pos_to_lla(p);
                let g = world.ground(la, lo).map(|g| g.h).unwrap_or(0.0);
                geo::lla_to_pos(la, lo, g)
            }
        };
        // From high up, 25..55° off the vertical, from any side.
        let tilt = r.range(0.45, 0.95) as f64;
        let az = r.range(0.0, std::f32::consts::TAU) as f64;
        let side = e * az.cos() + n * az.sin();
        let dir = -(up * tilt.cos() + side * tilt.sin());
        // (a given spot: eight seconds out, so scripted shots know when it lands)
        let lead = if hunt { 9.0 } else if at.is_some() { 8.0 } else { r.range(10.0, 16.0) as f64 };
        // (raised by what it falls on the way: `meteors_step` gives it half of gravity)
        let pos = target - dir * speed * lead + up * (0.25 * 9.81 * lead * lead);
        let rad = r.range(2.0, 5.5) as f64 * if hunt { 0.8 } else { 1.0 };
        let id = self.next_id;
        self.next_id += 1;
        let spin = r.dir().as_dvec3() * r.range(0.5, 2.0) as f64;
        let mesh = crate::cars::M_ROCK + (id % 3);
        self.meteors.push(Meteor { id, pos, vel: dir * speed, r: rad, rot: DQuat::from_axis_angle(r.dir().as_dvec3(), r.range(0.0, 6.0) as f64), spin, hp: (rad * 1.5) as f32, mesh, fx: 0.0, age: 0.0 });
    }

    fn meteors_step(&mut self, dt: f64, world: &World, craft: &Craft, emit: &mut Vec<(DVec3, Particle)>) {
        let mut down = Vec::new();
        for (k, m) in self.meteors.iter_mut().enumerate() {
            let up = m.pos.normalize();
            let p0 = m.pos;
            m.vel -= up * 9.81 * dt * 0.5;
            m.pos += m.vel * dt;
            m.rot = (DQuat::from_scaled_axis(m.spin * dt) * m.rot).normalize();
            m.age += dt as f32;
            // A burning trail: fire close behind, then smoke that hangs.
            let sp = m.vel.length();
            let back = -m.vel / sp.max(1.0);
            // (spaced by the distance flown, so the trail is unbroken at any speed)
            m.fx += ((m.pos - p0).length() / (m.r * 0.9)) as f32;
            let r = &mut self.rng;
            let rr = m.r as f32;
            // The head's glare.
            emit.push((m.pos, Particle { life: 0.09, size0: rr * 6.0, size1: rr * 6.0, kind: FLASH, alpha: 2.0, seed: r.f(), ..Default::default() }));
            while m.fx >= 1.0 {
                m.fx -= 1.0;
                let f = r.f() as f64;
                let at = p0 + (m.pos - p0) * f + back * m.r * 0.6;
                // Flame streaming off it, then a long column of smoke.
                emit.push((at, Particle { vel: (back.as_vec3() * r.range(20.0, 50.0) + r.dir() * 5.0).into(), life: r.range(0.4, 0.9), size0: rr * r.range(0.9, 1.3), size1: rr * r.range(2.0, 3.0), temp: r.range(2.2, 3.0), kind: GAS, drag: 2.0, buoy: 2.0, seed: r.f(), alpha: r.range(0.8, 0.95), ..Default::default() }));
                if r.f() < 0.7 {
                    emit.push((at + back * m.r * 1.5, Particle { vel: (back.as_vec3() * r.range(4.0, 12.0) + r.dir() * 2.0).into(), life: r.range(7.0, 12.0), size0: rr * r.range(1.0, 1.4), size1: rr * r.range(4.5, 7.5), temp: r.range(0.05, 0.3), kind: GAS, drag: 1.0, buoy: 0.5, seed: r.f(), alpha: r.range(0.35, 0.55), ..Default::default() }));
                }
            }
            if r.f() < (dt * 20.0) as f32 {
                emit.push((m.pos, Particle { vel: (back.as_vec3() * 30.0 + r.dir() * 15.0).into(), life: r.range(0.5, 1.4), size0: 0.15, size1: 0.05, kind: SPARK, drag: 0.4, seed: r.f(), alpha: 1.0, ..Default::default() }));
            }
            // What it hits on its way this frame.
            if craft.alive && seg_dist(craft.pos, p0, m.pos) < m.r + craft.radius {
                down.push((k, None, false, true));
                continue;
            }
            let n = ((m.pos - p0).length() / 4.0).ceil().clamp(1.0, 40.0) as usize;
            for s in 1..=n {
                let p = p0 + (m.pos - p0) * (s as f64 / n as f64);
                let (la, lo, h) = geo::pos_to_lla(p);
                if let Some(b) = world.building(la, lo).filter(|b| !b.bridge && h - m.r * 0.5 <= b.top) {
                    m.pos = p;
                    down.push((k, Some((b.tile, b.idx)), false, false));
                    break;
                }
                if let Some(g) = world.ground(la, lo) {
                    if h - m.r * 0.3 <= g.h {
                        m.pos = p;
                        let b = world.photo_building(la, lo, g.h).map(|b| (b.tile, b.idx));
                        down.push((k, b, g.water && b.is_none(), false));
                        break;
                    }
                }
            }
            // (far behind, or out of the world: gone)
            if m.age > 40.0 || (m.pos - craft.pos).length() > 12_000.0 {
                down.push((k, None, true, false));
                m.r = -1.0;
            }
        }
        down.sort_by(|a, b| b.0.cmp(&a.0));
        down.dedup_by_key(|d| d.0);
        for (k, building, water, plane) in down {
            let m = self.meteors.swap_remove(k);
            if m.r < 0.0 {
                continue;
            }
            if plane {
                self.reqs.push(Req::Kill("METEOR"));
                self.explode(m.pos, m.vel, m.r as f32 * 0.8, emit);
                continue;
            }
            // Where it comes down: a fireball, and a crater or a building gone.
            let up = m.pos.normalize();
            let (la, lo, _) = geo::pos_to_lla(m.pos);
            let at = if building.is_some() { m.pos } else { world.ground(la, lo).map(|g| geo::lla_to_pos(la, lo, g.h)).unwrap_or(m.pos) };
            let rr = (m.r * 2.2 + 3.0) as f32;
            self.explode(at + up * 1.0, m.vel, (m.r as f32 * 1.6).max(4.0), emit);
            self.reqs.push(Req::Blast { at, vel: m.vel, r: rr, building, water });
            // Close by, the blast is deadly too.
            if craft.alive && (craft.pos - at).length() < rr as f64 * 2.5 + craft.radius {
                self.reqs.push(Req::Kill("METEOR BLAST"));
            }
        }
    }

    /// A fireball: flash, rolling flame, smoke and embers (`size` ~ metres).
    pub fn explode(&mut self, at: DVec3, vel: DVec3, size: f32, emit: &mut Vec<(DVec3, Particle)>) {
        let r = &mut self.rng;
        let up = at.normalize().as_vec3();
        let p = (size / 3.0).clamp(0.6, 2.5);
        self.glows.push((at + at.normalize() * 3.0, 0.0, 0.9, [255, 170, 90], 120_000.0 * p));
        emit.push((at + at.normalize() * 2.0, Particle { life: 0.3, size0: 10.0 * p, size1: 34.0 * p, kind: FLASH, alpha: 1.8, ..Default::default() }));
        let carry = (vel.as_vec3() * 0.04).clamp_length_max(8.0);
        for _ in 0..(18.0 * p) as usize {
            let d = (r.dir() + up * 0.5).normalize();
            emit.push((at + (d * r.range(0.0, 3.0) * p).as_dvec3(), Particle { vel: (d * r.range(10.0, 26.0) * p + up * r.range(3.0, 8.0) + carry).into(), life: r.range(2.6, 4.6), size0: r.range(2.4, 3.8) * p, size1: r.range(9.0, 13.0) * p, temp: r.range(2.2, 3.0), kind: GAS, drag: r.range(1.9, 2.4), buoy: r.range(6.0, 9.0), seed: r.f(), alpha: r.range(0.75, 0.95), ..Default::default() }));
        }
        for _ in 0..(10.0 * p) as usize {
            let d = (r.dir() + up * 0.9).normalize();
            emit.push((at, Particle { vel: (d * r.range(4.0, 12.0) * p).into(), life: r.range(8.0, 14.0), size0: r.range(3.0, 5.0) * p, size1: r.range(14.0, 22.0) * p, temp: 0.1, kind: GAS, drag: 1.0, buoy: r.range(1.0, 2.5), seed: r.f(), alpha: r.range(0.4, 0.6), ..Default::default() }));
        }
        for _ in 0..(40.0 * p) as usize {
            let d = (r.dir() + up * 0.6).normalize();
            emit.push((at, Particle { vel: (d * r.range(20.0, 70.0) * p.sqrt()).into(), life: r.range(0.8, 2.4), size0: r.range(0.08, 0.16), size1: 0.04, kind: SPARK, drag: r.range(0.2, 0.7), seed: r.f(), alpha: r.range(0.6, 1.0), ..Default::default() }));
        }
        self.reqs.push(Req::Boom(p * 1.2, at));
    }

    /// Shot to pieces in the air: a burst of fire and the rock in bits.
    fn break_up(&mut self, m: &Meteor, emit: &mut Vec<(DVec3, Particle)>) {
        self.explode(m.pos, m.vel, (m.r as f32 * 0.7).max(1.5), emit);
        self.reqs.push(Req::Rocks { at: m.pos, vel: m.vel * 0.3, r: m.r as f32 });
    }

    // ---------------------------------------------------------- wrecking ball

    /// Hang the ball (or take it away). `top`: where the chain is fastened;
    /// `span`: the aircraft's half span (`Spec::span`), `mass` its mass.
    pub fn set_ball(&mut self, on: bool, top: DVec3, vel: DVec3, span: f64, mass: f64) {
        self.ball_on = on;
        if !on {
            self.ball = None;
            self.ball_force = DVec3::ZERO;
            return;
        }
        let up = top.normalize();
        // Heavy for the aircraft, but not so heavy it cannot fly it: about
        // a fifth of its mass, a ball of steel that size (bigger to see on
        // the light ones), on a chain of a couple of spans.
        let m = (mass * 0.25).clamp(150.0, 40_000.0);
        let r = ((m / 7850.0 * 3.0 / (4.0 * std::f64::consts::PI)).cbrt()).max(1.1 * (span / 5.5).sqrt());
        let len = (span * 5.0).clamp(24.0, 90.0);
        let pos = top - up * len;
        let n = 14;
        let links: Vec<DVec3> = (0..=n).map(|i| top + (pos - top) * (i as f64 / n as f64)).collect();
        self.ball = Some(Ball { pos, vel, r, m, len, top, links_prev: links.clone(), links, cool: HashMap::new(), ground_cool: 0.0, stuck: 0.0 });
    }

    /// One physics step of the ball (the aircraft's own step, h s): the chain
    /// pulls it along like a stiff spring; it swings, hits the ground and
    /// smashes into buildings. Sets `ball_force`, the chain's pull on the aircraft.
    pub fn ball_step(&mut self, h: f64, top: DVec3, top_vel: DVec3, world: &World) {
        let Some(b) = self.ball.as_mut() else {
            self.ball_force = DVec3::ZERO;
            return;
        };
        b.top = top;
        let up = b.pos.normalize();
        let k = b.m * (std::f64::consts::TAU * 3.0).powi(2);
        let c = 2.0 * 0.5 * (k * b.m).sqrt();
        let d = b.pos - top;
        let dist = d.length().max(1e-6);
        let nrm = d / dist;
        let mut tension = 0.0;
        if dist > b.len {
            let vrel = (b.vel - top_vel).dot(nrm);
            tension = (k * (dist - b.len) + c * vrel).clamp(0.0, b.m * 9.81 * 6.0);
        }
        let rho = 1.2;
        // (as dense as solid steel: the drag of a ball of its mass, not of the size drawn)
        let rs = (b.m / 7850.0 * 3.0 / (4.0 * std::f64::consts::PI)).cbrt();
        let drag = -b.vel * b.vel.length() * (0.5 * rho * 0.5 * std::f64::consts::PI * rs * rs);
        let f = -nrm * tension + drag - up * (9.81 * b.m);
        b.vel += f / b.m * h;
        let p0 = b.pos;
        b.pos += b.vel * h;
        self.ball_force = nrm * tension;
        // What it hits.
        for v in b.cool.values_mut() {
            *v -= h as f32;
        }
        b.cool.retain(|_, v| *v > 0.0);
        b.ground_cool -= h as f32;
        let (la, lo, alt) = geo::pos_to_lla(b.pos);
        let hit_b = world.building(la, lo).filter(|bh| !bh.bridge && alt - b.r <= bh.top).map(|bh| (bh.tile, bh.idx, bh.top));
        let g = world.ground(la, lo);
        let photo_b = match (hit_b, &g) {
            (None, Some(g)) if alt - b.r <= g.h => world.photo_building(la, lo, g.h).map(|bh| (bh.tile, bh.idx, g.h)),
            _ => None,
        };
        if let Some((tile, idx, top_alt)) = hit_b.or(photo_b) {
            let speed = b.vel.length();
            let ke = 0.5 * b.m * speed * speed;
            // On its roof (coming down on it from above) or into a wall.
            let from_above = geo::pos_to_lla(p0).2 - b.r > top_alt - 0.5;
            let out = if from_above { up } else { -(b.vel - up * b.vel.dot(up)).normalize_or(-up) };
            let mut breaks = false;
            if !b.cool.contains_key(&(tile, idx)) && speed > 6.0 {
                b.cool.insert((tile, idx), 0.3);
                // Damage by the energy it brings: 40 kJ a laser hit (a
                // light aircraft's ball at cruise is a dozen).
                let dmg = (ke / 40_000.0) as f32;
                if let Some((base, top, area)) = world.footprint(tile, idx) {
                    let w = self.wear.entry((tile, idx)).or_insert(0.0);
                    *w += dmg;
                    let hp = strength(area, top - base);
                    self.reqs.push(Req::Chip { tile, idx, at: b.pos - out * b.r, out, hard: (dmg / 4.0).clamp(0.8, 3.0) });
                    self.reqs.push(Req::Sound(crate::audio::Event::Crunch((dmg / 6.0).clamp(0.5, 2.0))));
                    if *w >= hp {
                        self.wear.remove(&(tile, idx));
                        self.reqs.push(Req::BringDown { tile, idx, at: b.pos, vel: b.vel * 0.8 });
                        breaks = true;
                    }
                }
            }
            if breaks {
                // Through it: it hardly slows.
                b.vel *= 0.8;
                b.stuck = 0.0;
            } else if from_above {
                // Dragged over its roof.
                b.pos += up * (top_alt + b.r - alt).max(0.0);
                let vn = b.vel.dot(up);
                if vn < 0.0 {
                    b.vel -= up * vn * 1.25;
                }
                b.vel -= (b.vel - up * b.vel.dot(up)) * (h * 1.5).min(1.0);
                b.stuck = 0.0;
            } else {
                // Off a wall that holds: back out, bounced. Wedged in for long,
                // it is hauled up over the top.
                b.stuck += h;
                b.pos = p0;
                let vn = b.vel.dot(out);
                if vn < 0.0 {
                    b.vel -= out * vn * 1.3;
                }
                b.vel *= 0.7;
                if b.stuck > 0.25 {
                    b.pos += up * (top_alt + b.r - geo::pos_to_lla(b.pos).2).max(0.0);
                    b.stuck = 0.0;
                }
            }
        } else if let Some(g) = g {
            if alt - b.r <= g.h {
                let vn = b.vel.dot(up);
                b.pos += up * (g.h + b.r - alt);
                if vn < -12.0 && b.ground_cool <= 0.0 {
                    b.ground_cool = 0.5;
                    self.reqs.push(Req::Crater { at: geo::lla_to_pos(la, lo, g.h), r: (b.r as f32 * 1.5).clamp(1.0, 4.0), small: true });
                    self.reqs.push(Req::Sound(crate::audio::Event::Crunch(((-vn) / 25.0).clamp(0.4, 1.6) as f32)));
                }
                if vn < 0.0 {
                    b.vel -= up * vn * if g.water { 1.0 } else { 1.25 };
                }
                // Dragged along: friction (water slows it hard too).
                let tang = b.vel - up * b.vel.dot(up);
                b.vel -= tang * (h * if g.water { 2.5 } else { 1.5 }).min(1.0);
            }
        }
    }

    /// The chain's links follow, as a rope of verlet points (drawing only).
    pub fn chain_frame(&mut self, dt: f64) {
        let Some(b) = self.ball.as_mut() else { return };
        let n = b.links.len();
        if n < 2 {
            return;
        }
        let up = b.top.normalize();
        let seg = b.len / (n - 1) as f64;
        let g = -up * 9.81 * dt * dt;
        for i in 1..n - 1 {
            let p = b.links[i];
            let v = (p - b.links_prev[i]) * 0.96;
            b.links_prev[i] = p;
            b.links[i] = p + v + g;
        }
        b.links[0] = b.top;
        b.links[n - 1] = b.pos;
        b.links_prev[0] = b.top;
        b.links_prev[n - 1] = b.pos;
        for _ in 0..8 {
            for i in 0..n - 1 {
                let (a, c) = (b.links[i], b.links[i + 1]);
                let d = c - a;
                let l = d.length().max(1e-6);
                if l <= seg {
                    continue;
                }
                let corr = d * ((l - seg) / l);
                match (i == 0, i + 1 == n - 1) {
                    (true, true) => {}
                    (true, false) => b.links[i + 1] -= corr,
                    (false, true) => b.links[i] += corr,
                    _ => {
                        b.links[i] += corr * 0.5;
                        b.links[i + 1] -= corr * 0.5;
                    }
                }
            }
        }
    }

    // ---------------------------------------------------------------- drawing

    /// The bolts, meteors, ball and chain as instanced solids, and their glows.
    /// `peers`: others in the room firing their lasers (where, which way, how big).
    pub fn instances(&self, cam: DVec3, insts: &mut Vec<(u32, Inst)>, lights: &mut Vec<Light>, peers: &[(DVec3, DQuat, f32)], t: f64) {
        let rel = |p: DVec3| (p - cam).as_vec3();
        let mut light = |p: DVec3, inten: f32, col: [u8; 3]| {
            if lights.len() < crate::render::MAX_DYN_LIGHTS {
                lights.push(Light { pos: rel(p).into(), inten, dir: [0.0; 3], spread: 0.0, col: [col[0], col[1], col[2], 0], fx: [0, 0, 255, 1] });
            }
        };
        let beam = |insts: &mut Vec<(u32, Inst)>, head: DVec3, dir: DVec3, len: f64, w: f32, col: [f32; 3], k: f32| {
            let o = rel(head - dir * (len * 0.5));
            let y = dir.as_vec3();
            let x = y.any_orthonormal_vector();
            let z = x.cross(y);
            let m = Mat3::from_cols(x * w, y * (len as f32 * 0.5), z * w);
            insts.push((crate::cars::M_CUBE, Inst { m: [[m.x_axis.x, m.y_axis.x, m.z_axis.x, o.x], [m.x_axis.y, m.y_axis.y, m.z_axis.y, o.y], [m.x_axis.z, m.y_axis.z, m.z_axis.z, o.z]], tint: [col[0] * k, col[1] * k, col[2] * k, 4.0], ext: [0.0; 4] }));
        };
        for b in &self.bolts {
            let d = (b.pos - cam).length();
            // (wider far off, so it stays a few pixels wide; a long streak)
            let w = (0.06 + d as f32 * 0.0011).min(1.5);
            // (no longer than the way it has come: it starts at the muzzle)
            let len = (b.travelled).clamp(0.5, 22.0);
            beam(insts, b.pos, b.dir, len, w, b.col, 110.0);
            light(b.pos, 20_000.0, LASER_LIGHT);
        }
        // Other players' lasers: drawn, not hitting anything here.
        for (k, (pos, rot, size)) in peers.iter().enumerate() {
            let fwd = *rot * DVec3::NEG_Z;
            for s in [-1.0, 1.0] {
                let m = *pos + *rot * DVec3::new(s * 0.25 * *size as f64, -0.1, -0.12 * *size as f64);
                let ph = ((t * 7.0 + k as f64 * 0.37 + if s > 0.0 { 0.5 } else { 0.0 }).fract()) * RANGE * 0.6;
                let head = m + fwd * (40.0 + ph);
                let d = (head - cam).length();
                let w = (0.06 + d as f32 * 0.0011).min(1.5);
                beam(insts, head, fwd, 22.0, w, LASER, 110.0);
                light(head, 20_000.0, LASER_LIGHT);
            }
        }
        for g in &self.glows {
            let k = 1.0 - g.1 / g.2;
            light(g.0, g.4 * k * k, g.3);
        }
        for m in &self.meteors {
            let o = rel(m.pos);
            if o.length() > 15_000.0 {
                continue;
            }
            let r3 = Mat3::from_quat(m.rot.as_quat()) * (m.r as f32);
            insts.push((m.mesh, Inst { m: [[r3.x_axis.x, r3.y_axis.x, r3.z_axis.x, o.x], [r3.x_axis.y, r3.y_axis.y, r3.z_axis.y, o.y], [r3.x_axis.z, r3.y_axis.z, r3.z_axis.z, o.z]], tint: [0.32, 0.27, 0.24, 3.0], ext: [1.0, 0.0, 0.0, 0.0] }));
            // The white-hot front, and its light.
            let f = (m.vel.normalize_or_zero() * m.r * 0.35).as_vec3();
            let s = m.r as f32 * 0.8;
            let oc = o + f;
            insts.push((self.mesh0, Inst { m: [[s, 0.0, 0.0, oc.x], [0.0, s, 0.0, oc.y], [0.0, 0.0, s, oc.z]], tint: [40.0, 14.0, 4.0, 4.0], ext: [0.0; 4] }));
            light(m.pos, 60_000.0 * m.r as f32, [255, 150, 70]);
        }
        if let Some(b) = &self.ball {
            let o = rel(b.pos);
            let s = b.r as f32;
            insts.push((self.mesh0, Inst { m: [[s, 0.0, 0.0, o.x], [0.0, s, 0.0, o.y], [0.0, 0.0, s, o.z]], tint: [0.30, 0.31, 0.33, 0.0], ext: [0.0; 4] }));
            // The chain: a link every so often along the rope, each turned a quarter from the last.
            let n = b.links.len();
            let lw = (b.r as f32 * 0.22).clamp(0.12, 0.6);
            for i in 0..n - 1 {
                let (a, c) = (b.links[i], b.links[i + 1]);
                let along = (c - a).length() as f32;
                let steps = (along / (lw * 2.2)).ceil().max(1.0) as usize;
                let y = (c - a).normalize_or(DVec3::Z).as_vec3();
                let x0 = y.any_orthonormal_vector();
                for j in 0..steps {
                    let p = a + (c - a) * ((j as f64 + 0.5) / steps as f64);
                    let q = Quat::from_axis_angle(y, if (i * 31 + j) % 2 == 0 { 0.0 } else { std::f32::consts::FRAC_PI_2 });
                    let x = q * x0;
                    let z = x.cross(y);
                    let o = rel(p);
                    let m = Mat3::from_cols(x * lw, y * lw * 1.1, z * lw * 0.3);
                    insts.push((crate::cars::M_CUBE, Inst { m: [[m.x_axis.x, m.y_axis.x, m.z_axis.x, o.x], [m.x_axis.y, m.y_axis.y, m.z_axis.y, o.y], [m.x_axis.z, m.y_axis.z, m.z_axis.z, o.z]], tint: [0.22, 0.22, 0.23, 0.0], ext: [0.0; 4] }));
                }
            }
        }
    }
}

/// Distance from a point to a segment.
fn seg_dist(p: DVec3, a: DVec3, b: DVec3) -> f64 {
    let ab = b - a;
    let t = ((p - a).dot(ab) / ab.length_squared().max(1e-9)).clamp(0.0, 1.0);
    (a + ab * t - p).length()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aircraft::{c172, fa18};
    use crate::plane::{Controls, Plane};
    use std::sync::Arc;

    fn empty_world() -> World {
        let dir = std::env::temp_dir().join("fablegame-test-cache");
        let src = Arc::new(crate::sources::Sources::new(crate::fetch::Fetcher::new(dir, true), String::new()));
        World::new(crate::sources::Loader::new(src, 0), 14)
    }

    fn angle(a: DVec3, b: DVec3) -> f64 {
        a.normalize().dot(b.normalize()).clamp(-1.0, 1.0).acos().to_degrees()
    }

    /// Mouse aim: the instructor brings the nose onto the aim, in the real
    /// flight model, without breaking the aircraft: a turn off to one side,
    /// a target behind, one a little below.
    #[test]
    fn mouse_aim_converges() {
        let w = empty_world();
        for spec in [&c172::SPEC, &fa18::SPEC] {
            for (yaw, pitch, secs) in [(40.0f64, 8.0f64, 14.0), (150.0, 3.0, 30.0), (-10.0, -7.0, 10.0)] {
                let mut p = Plane::spawn(spec, 0.6, -2.1, 2500.0, 0.3, spec.cruise);
                p.trim_to_climb();
                let up = p.pos.normalize();
                let (e, n, _) = geo::enu(p.pos);
                let hd = 0.3 + yaw.to_radians();
                let flat = e * hd.sin() + n * hd.cos();
                let aim = (flat * pitch.to_radians().cos() + up * pitch.to_radians().sin()).normalize();
                let dt = 1.0 / 180.0;
                let mut best = 180.0f64;
                for _ in 0..(secs / dt) as usize {
                    let room = 1.0 - smooth(0.7, 0.95, p.g_load / spec.g_limit as f64);
                    let (pi, ro, ya) = instructor(p.rot, p.omega, p.pos.normalize(), aim, p.airspeed().length(), room);
                    let c = Controls { pitch: pi, roll: ro, yaw: ya, throttle: 1.0, ..Default::default() };
                    assert!(p.step(dt, &c, &w).is_none(), "{} crashed", spec.name);
                    best = best.min(angle(p.forward(), aim));
                }
                let a = angle(p.forward(), aim);
                assert!(a < 4.0, "{}: aim {yaw},{pitch}: still {a:.1}° off after {secs} s (best {best:.1}°)", spec.name);
                assert!(!p.wrecked(), "{}: broke itself following the aim", spec.name);
            }
        }
    }

    #[test]
    fn deaths_count_and_reset() {
        let mut t = Tally::default();
        assert!(t.died(3.0));
        // (the same death, seen again as the wreck hits the ground: once)
        assert!(!t.died(4.0));
        assert_eq!(t.deaths, 1);
        t.revived();
        assert!(t.died(10.0));
        assert_eq!(t.deaths, 2);
        t.destroyed = 5;
        t.new_session();
        assert_eq!((t.deaths, t.destroyed, t.dead_at), (0, 0, None));
    }

    #[test]
    fn crash_counts_as_death() {
        let w = empty_world();
        let mut d = Destroy::new(7);
        d.mode = GameMode::Destruction;
        // A meteor dropped straight onto the aircraft kills it.
        let craft = Craft { pos: geo::lla_to_pos(0.6, -2.1, 1500.0), vel: DVec3::ZERO, fwd: DVec3::X, radius: 6.0, alive: true };
        d.spawn_meteor(&w, &craft, Some(craft.pos));
        let mut emit = Vec::new();
        for _ in 0..(30.0 * 60.0) as usize {
            d.update(1.0 / 60.0, &w, &craft, &mut emit);
            if d.reqs.iter().any(|r| matches!(r, Req::Kill(_))) {
                break;
            }
        }
        assert!(d.reqs.iter().any(|r| matches!(r, Req::Kill(_))), "the meteor never hit");
        assert!(d.tally.died(1.0));
        assert_eq!(d.tally.deaths, 1);
    }

    /// A laser ray against a building's box: it hits the wall facing it at
    /// the right distance, goes over the roof, and counts from inside as 0.
    #[test]
    fn laser_hits_building_box() {
        let ring = [DVec2::new(-10.0, -10.0), DVec2::new(10.0, -10.0), DVec2::new(10.0, 10.0), DVec2::new(-10.0, 10.0)];
        // From 100 m west, 5 m up, heading east: the west wall at x = -10.
        let t = ray_prism(DVec3::new(-100.0, 0.0, 5.0), DVec3::X, &ring, 0.0, 30.0, 1000.0).unwrap();
        assert!((t - 90.0).abs() < 1e-9, "{t}");
        // Over the roof: a miss.
        assert!(ray_prism(DVec3::new(-100.0, 0.0, 35.0), DVec3::X, &ring, 0.0, 30.0, 1000.0).is_none());
        // Diving onto the roof from above.
        let d = DVec3::new(1.0, 0.0, -1.0).normalize();
        let t = ray_prism(DVec3::new(-25.0, 0.0, 50.0), d, &ring, 0.0, 30.0, 1000.0).unwrap();
        let p = DVec3::new(-25.0, 0.0, 50.0) + d * t;
        assert!((p.z - 30.0).abs() < 1e-6 && p.x > -10.0, "{p}");
        // Out of range, and from inside.
        assert!(ray_prism(DVec3::new(-100.0, 0.0, 5.0), DVec3::X, &ring, 0.0, 30.0, 50.0).is_none());
        assert_eq!(ray_prism(DVec3::new(0.0, 0.0, 5.0), DVec3::X, &ring, 0.0, 30.0, 50.0), Some(0.0));
        // An L-shaped (concave) outline: through the notch, the far arm.
        let l = [DVec2::new(0.0, 0.0), DVec2::new(30.0, 0.0), DVec2::new(30.0, 10.0), DVec2::new(10.0, 10.0), DVec2::new(10.0, 30.0), DVec2::new(0.0, 30.0)];
        let t = ray_prism(DVec3::new(20.0, 50.0, 2.0), DVec3::NEG_Y, &l, 0.0, 10.0, 1000.0).unwrap();
        assert!((t - 40.0).abs() < 1e-9, "{t}");
        // And a meteor in the way.
        assert!((ray_sphere(DVec3::ZERO, DVec3::X, DVec3::new(50.0, 0.0, 0.0), 5.0, 100.0).unwrap() - 45.0).abs() < 1e-9);
        assert!(ray_sphere(DVec3::ZERO, DVec3::X, DVec3::new(-50.0, 0.0, 0.0), 5.0, 100.0).is_none());
    }

    #[test]
    fn aim_turns_with_the_mouse() {
        let up = DVec3::Z;
        let d = turn_aim(DVec3::Y, up, 100.0, 0.0, 1.0);
        assert!(d.x > 0.1 && d.z.abs() < 1e-9, "{d}");
        let d = turn_aim(DVec3::Y, up, 0.0, -100.0, 1.0);
        assert!(d.z > 0.1, "{d}");
        // Never past straight up.
        let d = turn_aim(DVec3::Y, up, 0.0, -1e5, 1.0);
        assert!(d.z < 0.999 && d.y > 0.0, "{d}");
    }
}
