//! The aircraft's systems and a six-degree-of-freedom flight model, driven
//! by the type's `Spec` (aircraft/): lift, drag and stall from its wing,
//! flaps, slats and spoilers; a piston engine turning a fixed-pitch propeller
//! (revs and thrust follow the airspeed, power falls with altitude, torque
//! and slipstream yaw it) or turbofans (the fans spool up and down, thrust
//! lapses with height and Mach, reversers on the ground); the airspeed
//! indicator reading indicated (not true) airspeed; adverse yaw, stall
//! buffet and wing drop, ground effect, wave drag near the speed of sound,
//! and the turbulence the weather stirs up.
//! The models are built in `aircraft/`; `pose` animates their pieces;
//! `damage.rs` describes what has been knocked off. Trees, rooftops, the
//! ground, over-speed and over-G all take parts off the airframe here, and
//! the aircraft flies on (as best it can) with what is left.
use crate::aircraft::{Engine, Part, Spec, MAX_GEAR};
use crate::airframe::{Anim, Drive, Input, Piece, Role, G_ENGINE, G_FUSE, G_LWING, G_RWING, G_TAIL};
use crate::crash::Rng;
use crate::damage::{self, Damage, Sel, ShedReq, Tear};
use crate::weather::Air;
use crate::geo;
use crate::world::{TreeRef, World};
use glam::{DMat3, DQuat, DVec3, Mat4, Vec3};

#[derive(Default, Clone, Copy)]
pub struct Controls {
    pub pitch: f64, // +1 = nose up
    pub roll: f64,  // +1 = roll right
    pub yaw: f64,   // +1 = nose right
    pub throttle: f64,
    pub brake: bool,
    pub boost: bool,
    /// thrust reversers (on the ground, jets)
    pub reverse: bool,
}

pub struct Impact {
    pub pos: DVec3,
    pub vel: DVec3,
    pub water: bool,
    pub ground_h: f64,
    /// the building that was hit, if any (tile, footprint index)
    pub building: Option<(geo::TileId, usize)>,
}

pub struct Plane {
    /// what it is
    pub spec: &'static Spec,
    pub pos: DVec3,
    pub vel: DVec3,
    /// the air's own motion where the aircraft is (planet frame); the wing only feels `vel - wind`
    pub wind: DVec3,
    /// how rough the air is here (set by the caller from the weather each frame)
    pub air: Air,
    /// the turbulent part of the air's motion (planet frame), on top of `wind`
    pub gust: DVec3,
    /// rotational gusts (body frame, rad/s): the air rolling, pitching and yawing across the airframe
    pub gust_rot: DVec3,
    /// turbulence filter states: translational in the flight-path frame (along, across, up), and rotational
    gust_a: DVec3,
    gust_b: DVec3,
    rot_a: DVec3,
    rot_b: DVec3,
    /// the pilot's head swaying on its neck as the airframe is shaken (body frame, m)
    pub head: DVec3,
    /// specific force (what an accelerometer reads: everything but gravity), body frame, m/s²
    pub sf: DVec3,
    head_v: DVec3,
    /// shaft power being made, as a fraction of rated (a jet: thrust as a fraction of rated)
    pub power: f64,
    /// how far open the throttle effectively is (the engine takes a moment to answer it)
    charge: f64,
    /// a jet's fan speed, fraction of 100% N1
    pub n1: f64,
    /// how far the afterburners are lit, 0..1 (what is seen and heard of them; the thrust is immediate)
    pub burner: f64,
    /// which wing a stall would rather drop (slowly wandering, -1..1)
    drop: f64,
    pub rot: DQuat,
    pub omega: DVec3,
    pub throttle: f64,
    pub prop: f64,
    pub on_ground: bool,
    pub stalled: bool,
    pub alpha: f64,
    pub g_load: f64,
    /// smoothed control inputs: pitch, yaw, roll
    pub stick: DVec3,
    /// flap extension, 0 up .. 1 full travel, and the selected lever notch
    pub flaps: f64,
    pub flap_set: u8,
    /// leading-edge devices, 0 stowed .. 1 out
    pub slats: f64,
    /// the speedbrake lever (0 down .. 1) and where the spoiler panels are (0..1)
    pub speedbrake: f64,
    pub spoilers: f64,
    /// thrust reversers, 0 stowed .. 1 deployed
    pub reverser: f64,
    /// gear position, 0 stowed .. 1 down and locked, and where the handle is
    pub gear: f64,
    pub gear_down: bool,
    /// per landing gear leg (see `Spec::gear`): how far its oleo or spring is
    /// compressed (m), how far its wheels have turned (rad) and how fast (rad/s)
    pub compress: [f64; MAX_GEAR],
    pub wheel_rot: [f64; MAX_GEAR],
    pub wheel_spin: [f64; MAX_GEAR],
    /// propeller (or fan) speed, rad/s
    pub prop_rate: f64,
    /// the angle of attack the auto-trim is holding (drives the trim wheel)
    pub trim: f64,
    /// trees struck during the last step, for the caller to knock down
    pub tree_hits: Vec<TreeRef>,
    /// recent brushes with foliage
    pub brushes: f64,
    /// what is broken, and what was broken at the start of the step (so the
    /// caller can still find the pieces that have just gone)
    pub dmg: Damage,
    pub dmg_prev: Damage,
    /// parts that came off during the last step, for the caller to let go of
    pub shed: Vec<ShedReq>,
    /// one line per thing that just broke, for the HUD and the sound
    pub notes: Vec<String>,
    /// where metal is grinding along the ground: position, velocity of the sparks
    pub sparks: Vec<(DVec3, DVec3)>,
    /// seconds flown
    pub t: f64,
    near_trees: Vec<TreeRef>,
    /// trees something went into that are still standing, with how deep the
    /// worst contact went (1 = foliage, 2 = trunk), so they are not struck again
    struck: Vec<(geo::TileId, usize, u8)>,
    scan: u32,
    rng: Rng,
    /// held where it is (waiting on the ground for the terrain under it to load)
    pub hold: bool,
    /// where it is splashing through water: position, velocity of the spray
    pub splashes: Vec<(DVec3, DVec3)>,
    /// where something is ploughing or rolling through the ground: position,
    /// velocity the dirt is thrown at, how much (wheel ~1, airframe more)
    pub dirt: Vec<(DVec3, DVec3, f32)>,
    /// resting or sliding on the airframe itself rather than the wheels
    pub on_hull: bool,
    /// a force from outside the airframe this step (planet frame, N): a
    /// carrier's catapult or arresting wire (ships.rs `Ops`)
    pub push: DVec3,
    /// contact points (reused each step)
    pts: Vec<(DVec3, Part)>,
    /// moments since the left wing, right wing, tail and nose last broke
    /// something (so one blow is one break, not one per step)
    cool: [f64; 4],
    /// torque the ground puts on each hanging wing this step
    fold_tq: [f64; 2],
    /// how far a ditched aircraft has settled into the water
    sink: f64,
    /// what made it go up in flames, once it has
    pub cause: &'static str,
}

/// The breakable group a contact point belongs to.
fn part_group(p: Part) -> usize {
    match p {
        Part::Nose | Part::Prop => G_ENGINE,
        Part::Wing(0, _) | Part::Fold(0) | Part::Pod(0) => G_LWING,
        Part::Wing(..) | Part::Fold(_) | Part::Pod(_) => G_RWING,
        Part::Tail | Part::Fin | Part::Stab(_) => G_TAIL,
        _ => G_FUSE,
    }
}

/// Sea-level air density (kg/m³).
pub const RHO0: f64 = 1.225;

impl Plane {
    pub fn spawn(spec: &'static Spec, lat: f64, lon: f64, alt: f64, heading: f64, speed: f64) -> Self {
        let pos = geo::lla_to_pos(lat, lon, alt);
        let (e, n, u) = geo::enu(pos);
        let f = n * heading.cos() + e * heading.sin();
        let mut p = Self {
            spec,
            pos,
            vel: f * speed,
            wind: DVec3::ZERO,
            air: Air::default(),
            gust: DVec3::ZERO,
            gust_rot: DVec3::ZERO,
            gust_a: DVec3::ZERO,
            gust_b: DVec3::ZERO,
            rot_a: DVec3::ZERO,
            rot_b: DVec3::ZERO,
            head: DVec3::ZERO,
            sf: DVec3::new(0.0, 9.81, 0.0),
            head_v: DVec3::ZERO,
            power: 0.7,
            charge: 0.0,
            n1: 0.0,
            burner: 0.0,
            drop: 0.0,
            rot: DQuat::IDENTITY,
            omega: DVec3::ZERO,
            throttle: 0.0,
            prop: 0.0,
            on_ground: false,
            stalled: false,
            alpha: 0.0,
            g_load: 1.0,
            stick: DVec3::ZERO,
            flaps: 0.0,
            flap_set: 0,
            slats: 0.0,
            speedbrake: 0.0,
            spoilers: 0.0,
            reverser: 0.0,
            // Airborne starts are cleaned up: gear stowed.
            gear: 0.0,
            gear_down: false,
            compress: [0.0; MAX_GEAR],
            wheel_rot: [0.0; MAX_GEAR],
            wheel_spin: [0.0; MAX_GEAR],
            prop_rate: 0.0,
            trim: 0.0,
            tree_hits: Vec::new(),
            brushes: 0.0,
            dmg: Damage::new(spec),
            dmg_prev: Damage::new(spec),
            shed: Vec::new(),
            notes: Vec::new(),
            sparks: Vec::new(),
            dirt: Vec::new(),
            t: 0.0,
            near_trees: Vec::new(),
            struck: Vec::new(),
            scan: 0,
            rng: Rng(pos.x.to_bits() | 1),
            hold: false,
            splashes: Vec::new(),
            on_hull: false,
            push: DVec3::ZERO,
            pts: Vec::new(),
            cool: [0.0; 4],
            fold_tq: [0.0; 2],
            sink: 0.0,
            cause: "",
        };
        // Nose up by the angle of attack that carries the weight at this speed
        // and height, and trimmed there, so it starts out in level flight.
        let rho = density(alt);
        p.trim = p.alpha_level(speed, rho, 0.0);
        // Power for level flight too, and the propeller (or fans) already
        // turning as that power has them, or it starts out climbing or sinking.
        p.throttle = if speed > 20.0 { p.throttle_for(speed, alt, 0.0, 0.0, 0.0) } else { 0.0 };
        p.charge = p.throttle;
        p.settle_engine(speed, alt);
        p.rot = DQuat::from_mat3(&DMat3::from_cols(f.cross(u), u, -f)) * DQuat::from_rotation_x(p.trim);
        p
    }

    /// Standing on its wheels, engine idling, brakes off: ready to taxi.
    /// `h` is the ground height under it. It is held still until released.
    pub fn spawn_ground(spec: &'static Spec, lat: f64, lon: f64, h: f64, heading: f64, pitch: f64) -> Self {
        // The wheels just touching (it settles onto its oleos).
        let sag = -spec.gear.iter().map(|l| l.at.y).fold(0.0f64, f64::min);
        let mut p = Self::spawn(spec, lat, lon, h + sag, heading, 0.0);
        let (e, n, u) = geo::enu(p.pos);
        let f = n * heading.cos() + e * heading.sin();
        p.rot = DQuat::from_mat3(&DMat3::from_cols(f.cross(u), u, -f)) * DQuat::from_rotation_x(pitch);
        p.vel = DVec3::ZERO;
        p.gear = 1.0;
        p.gear_down = true;
        p.throttle = 0.0;
        p.charge = 0.0;
        p.power = 0.0;
        match &spec.engine {
            Engine::Piston(_) => p.prop_rate = 65.0,
            Engine::Fan(fe) => {
                p.n1 = fe.idle;
                p.prop_rate = fe.idle * fe.n1_rpm * std::f64::consts::TAU / 60.0;
            }
        }
        p.trim = 0.08;
        p.on_ground = true;
        p.hold = true;
        p
    }
    /// How high above the ground the origin is with the wheels just touching it.
    pub fn wheel_height(&self) -> f64 {
        -self.spec.gear.iter().map(|l| l.at.y).fold(0.0f64, f64::min)
    }
    pub fn forward(&self) -> DVec3 {
        self.rot * DVec3::NEG_Z
    }
    pub fn up(&self) -> DVec3 {
        self.rot * DVec3::Y
    }
    pub fn right(&self) -> DVec3 {
        self.rot * DVec3::X
    }

    /// Nose wheel steering angle (rad, + = right), only with the gear down;
    /// on an airliner it fades to what the rudder pedals give as it speeds up.
    pub fn steer_angle(&self) -> f64 {
        let s = self.spec;
        let fade = if s.steer_fade > 0.0 { 0.2 + 0.8 * (1.0 - smooth(3.0, s.steer_fade, self.vel.length())) } else { 1.0 };
        self.stick.y * s.steer_max * fade * self.gear
    }
    /// Angle of attack the wing stalls at (flaps make it stall a little earlier, slats later).
    pub fn stall_alpha(&self) -> f64 {
        let s = self.spec;
        s.stall_alpha - s.stall_flap * self.flaps + s.stall_slat * self.slats
    }
    pub fn flap_deg(&self) -> f64 {
        self.flaps * self.spec.flaps.last().copied().unwrap_or(0.0)
    }
    /// The flap lever's notch, as the pilot calls it (degrees).
    pub fn flap_notch_deg(&self) -> f64 {
        self.spec.flaps.get(self.flap_set as usize).copied().unwrap_or(0.0)
    }
    pub fn toggle_gear(&mut self) {
        // The squat switch blocks retraction on the ground.
        if !(self.gear_down && self.on_ground) {
            self.gear_down = !self.gear_down;
        }
    }
    pub fn set_flaps(&mut self, delta: i32) {
        self.flap_set = (self.flap_set as i32 + delta).clamp(0, self.spec.max_notch() as i32) as u8;
    }
    /// Speedbrake lever: down, or out to the flight detent (airliners only).
    pub fn toggle_speedbrake(&mut self) {
        if self.spec.spoiler_cd > 0.0 {
            self.speedbrake = if self.speedbrake > 0.0 { 0.0 } else { 0.5 };
        }
    }
    /// The highest speed (indicated) the flaps may be out at as they are now.
    pub fn vfe(&self) -> f64 {
        let s = self.spec;
        let k = (0..s.flaps.len()).find(|&k| s.flap_frac(k as u8) >= self.flaps - 0.01).unwrap_or(s.flaps.len() - 1);
        s.vfe.get(k).copied().unwrap_or(1e9)
    }
    /// Indicated airspeed (m/s): what the pitot tube feels, true airspeed scaled
    /// by the square root of the density. It is what the wing flies by: the
    /// aircraft stalls, and its limits are set, at the same indicated speed at any height.
    pub fn ias(&self) -> f64 {
        let (_, _, h) = geo::pos_to_lla(self.pos);
        self.airspeed().length() * (density(h) / RHO0).sqrt()
    }
    /// Mach number.
    pub fn mach(&self) -> f64 {
        let (_, _, h) = geo::pos_to_lla(self.pos);
        self.airspeed().length() / sound_speed(h)
    }
    /// How far past its speed limits it is (m/s indicated; the Mach limit counted
    /// as the same margin in speed), negative within them.
    pub fn overspeed(&self) -> f64 {
        self.excess(self.ias(), self.mach() / self.spec.speed_k)
    }
    /// (In the base type's m/s for a scaled-up one, and with its Mach number
    /// divided by `Spec::speed_k`, so its margins are the base type's.)
    fn excess(&self, ias: f64, mach: f64) -> f64 {
        let s = self.spec;
        let e = (ias - s.vne) / s.speed_k;
        if s.mmo > 0.0 { e.max((mach - s.mmo) * 300.0) } else { e }
    }
    /// Start out with the flaps at `notch` and the gear as given, re-trimmed
    /// for the airspeed it has (to fly level, or on a glide of `descent` m/s),
    /// the power set to match.
    pub fn configure(&mut self, notch: u8, gear_down: bool, descent: f64) {
        self.flap_set = notch.min(self.spec.max_notch());
        self.flaps = self.spec.flap_frac(self.flap_set);
        self.slats = if self.spec.slats_at > 0 && self.flap_set >= self.spec.slats_at { 1.0 } else { 0.0 };
        self.gear_down = gear_down;
        self.gear = if gear_down { 1.0 } else { 0.0 };
        let (_, _, h) = geo::pos_to_lla(self.pos);
        let rho = density(h);
        let v = self.airspeed().length();
        let flap_cl = self.spec.flap_cl * self.flaps;
        let a_old = self.trim;
        self.trim = self.alpha_level(v, rho, flap_cl);
        // Pitch the nose to the new angle of attack and the flight path down the glide.
        let up = self.pos.normalize();
        let fp = (self.airspeed() - up * descent).normalize();
        self.vel = self.wind + fp * v;
        let right = self.right();
        self.rot = DQuat::from_axis_angle(right, self.trim - a_old - (descent / v).asin()) * self.rot;
        self.throttle = self.throttle_for(v, h, self.flaps, self.gear, descent);
        self.charge = self.throttle;
        self.settle_engine(v, h);
    }

    /// The value an animation input has this instant (see airframe::Input).
    fn input(&self, i: Input) -> f32 {
        let s = self.stick.as_vec3();
        match i {
            Input::Aileron(side) => -side * s.z,
            Input::Elevator | Input::Pitch => s.x,
            Input::Rudder | Input::Yaw => s.y,
            Input::Roll => s.z,
            Input::Flaps => self.flaps as f32,
            Input::FlapLever => self.flap_set as f32 / self.spec.max_notch().max(1) as f32,
            Input::Slats => self.slats as f32,
            Input::Spoiler(side) => {
                // Roll spoilers rise on the wing going down, on top of the speedbrake.
                let roll = (side * s.z).max(0.0) * if self.spec.spoiler_cd > 0.0 { 0.55 } else { 0.0 };
                (self.spoilers as f32 + roll).min(1.0)
            }
            Input::Gear => self.gear as f32,
            Input::GearHandle => if self.gear_down { 1.0 } else { -1.0 },
            Input::Compress(k) => self.compress[k as usize] as f32,
            Input::Wheel(k) => self.wheel_rot[k as usize] as f32,
            Input::Steer => self.steer_angle() as f32,
            Input::Prop => (self.prop % std::f64::consts::TAU) as f32,
            Input::Throttle => self.throttle as f32,
            Input::Trim => self.trim as f32,
            Input::Reverser => self.reverser as f32,
        }
    }

    /// Where every piece of the model is, relative to the body, this instant.
    pub fn pose(&self, pieces: &[Piece]) -> Vec<Mat4> {
        let gear = self.gear as f32;
        // The doors open first, the legs swing, then the doors close again behind them.
        let legs = smooth01(0.15, 0.85, gear);
        let comp: Vec<f32> = self.compress.iter().map(|c| *c as f32).collect();
        let t = self.t as f32;
        // Past the never-exceed speed the control surfaces buzz on their hinges.
        let fl = smooth01(5.0, 30.0, self.overspeed() as f32) * 0.12;
        let d = &self.dmg;
        let drive = |dr: &Drive| -> f32 {
            let mut a = dr.eval(self.input(dr.input));
            if dr.gain != 0.0 {
                match dr.input {
                    Input::Aileron(side) => {
                        let k = if side < 0.0 { 0 } else { 1 };
                        if d.aileron[k] > 0.0 && d.aileron[k] < 1.0 {
                            // Hanging from one hinge: drooped, flapping in the slipstream.
                            return 0.55 + 0.14 * (t * 27.0 + side).sin();
                        }
                        a += fl * (t * 41.0 + side).sin();
                    }
                    Input::Elevator => a += fl * 0.7 * (t * 37.0).sin(),
                    Input::Rudder => a += fl * (t * 43.0).sin(),
                    _ => {}
                }
            }
            a
        };
        let mut out: Vec<Mat4> = Vec::with_capacity(pieces.len());
        for p in pieces {
            let local = match p.anim {
                Anim::Fixed => Mat4::IDENTITY,
                Anim::Move(m) => {
                    let a = drive(&m.turn[0]) + drive(&m.turn[1]);
                    let mut mat = if a != 0.0 { Mat4::from_translation(p.pivot) * Mat4::from_axis_angle(p.axis, a) * Mat4::from_translation(-p.pivot) } else { Mat4::IDENTITY };
                    if m.slide.gain != 0.0 || m.slide.offset != 0.0 {
                        mat = Mat4::from_translation(m.dir * drive(&m.slide)) * mat;
                    }
                    mat
                }
                a => crate::aircraft::c172::kinematics(a, p, legs, self.steer_angle() as f32 * smooth01(0.9, 1.0, gear), &comp),
            };
            out.push(match p.parent {
                Some(i) => out[i] * local,
                None => local,
            });
        }
        out
    }

    // ---------------------------------------------------------------- breaking things

    fn note(&mut self, s: impl Into<String>) {
        self.notes.push(s.into());
    }
    fn scuff(&mut self, group: usize, amount: f32) {
        self.dmg.scuff[group] = (self.dmg.scuff[group] + amount).min(1.0);
    }
    /// A blow that sets it turning (rad/s for a light single; a heavy
    /// aircraft takes the same blow far more calmly).
    fn kick(&mut self, w: DVec3) {
        self.omega += w * (1100.0 / self.spec.mass).cbrt();
    }
    fn wing_group(side: usize) -> usize {
        if side == 0 { G_LWING } else { G_RWING }
    }
    fn lr(side: usize) -> &'static str {
        if side == 0 { "LEFT" } else { "RIGHT" }
    }
    /// A span station given as a fraction of the half span.
    fn sta(&self, f: f32) -> f32 {
        self.spec.span * f
    }
    /// How much bigger this aircraft is than the light single the break
    /// distances were first worked out for.
    fn scale(&self) -> f32 {
        self.spec.span / 5.5
    }

    /// The wing on `side` (0 left) breaks off outboard of span `station`.
    fn break_wing(&mut self, side: usize, station: f32, why: &str) {
        self.snap_wing(side, station, 0.5, 0.0, true, why);
    }
    /// The wing on `side` breaks near span `station`, and how is left to the
    /// break: `rough` 0 is a clean shear (an edge struck hard), 1 skin torn
    /// apart, slanting across the chord and through the thickness; `fold` is
    /// the chance it buckles and hangs on by its skin rather than letting go
    /// (bending `up` or down). Breaks further in leave the bits between as
    /// pieces of their own.
    fn snap_wing(&mut self, side: usize, station: f32, rough: f32, fold: f32, up: bool, why: &str) {
        // A wing already hanging lets go before it breaks again.
        if self.dmg.fold[side].on {
            self.release_fold(side, "");
        }
        let k = self.scale();
        let station = station.max(0.3 * k);
        if station >= self.dmg.wing[side] - 0.05 * k {
            return;
        }
        let spec = self.spec;
        let sgn = if side == 0 { -1.0 } else { 1.0 };
        let r = &mut self.rng;
        let tz = r.range(-1.0, 1.0) * (0.12 + 0.6 * rough) * k;
        let ty = r.range(-1.0, 1.0) * (0.04 + 0.22 * rough);
        let jag = (0.04 + 0.26 * rough) * r.range(0.6, 1.25) * k.sqrt();
        let folds = r.f() < fold;
        let hold = r.range(500.0, 2600.0) * self.fold_k();
        let prev = self.dmg.stump(side);
        let tear = damage::wing_break(spec, side, station, tz, ty, jag, &prev);
        let tip = self.dmg.wing[side];
        let lost = tip - station;
        let tank = station < spec.tank[1] && tip >= spec.tank[1];
        self.dmg.wing[side] = station;
        self.dmg.wing_tear[side] = tear;
        self.scuff(Self::wing_group(side), 0.5);
        // On the ground the blow itself does the shoving.
        let (sg, lo) = (sgn as f64 * if self.on_hull || self.on_ground { 0.3 } else { 1.0 }, (lost / k) as f64);
        if folds && lost > 0.7 * k {
            self.dmg.fold[side] = damage::fold_at(spec, side, station, tear, prev, tip, up, hold);
            // Still attached: less of a shove than losing it.
            self.kick(DVec3::new(0.15, -sg * 0.3 * lo, -sg * (0.4 + 0.4 * lo)));
            self.note(format!("{} WING BUCKLED ({why})", Self::lr(side)));
        } else {
            // The part that leaves: the far side of the break, with whatever flap
            // or aileron (or engine) was riding on it, as one piece.
            let mut req = ShedReq::new(Sel::GroupAll(Self::wing_group(side)), tear.flip(), Vec3::new(sgn * 2.5, 1.5, 2.0), 5.0 / k);
            req.tear2 = prev;
            req.own2 = true;
            req.rigid = true;
            self.shed.push(req);
            if station < spec.aileron[0] + 0.05 * spec.span {
                self.dmg.aileron[side] = 0.0;
            }
            // Shoved: it rolls toward the stump and yaws into the drag of the break.
            self.kick(DVec3::new(0.3, -sg * 0.6 * lo, -sg * (0.8 + 0.9 * lo)));
            self.note(format!("{} WING TORN OFF ({why})", Self::lr(side)));
        }
        if tank {
            self.note(format!("{} FUEL TANK RUPTURED", Self::lr(side)));
        }
    }
    /// How much heavier a hanging wing is to swing than the light single's
    /// (mass per metre times the square of the length).
    fn fold_k(&self) -> f32 {
        let k = self.scale();
        self.wing_kg_m() / 24.0 * k * k
    }
    /// Mass of the wing per metre of span (kg).
    fn wing_kg_m(&self) -> f32 {
        24.0 * (self.spec.groups[G_LWING].1 / 170.0) * (5.5 / self.spec.span)
    }
    /// A buckled wing lets go of the stump.
    fn release_fold(&mut self, side: usize, why: &str) {
        let f = self.dmg.fold[side];
        if !f.on {
            return;
        }
        self.dmg.fold[side].on = false;
        let cm = f.mat().transform_point3(damage::wing_pt(self.spec, side, (f.root + f.tip) * 0.5, 0.4, 0.0));
        let swing = f.axis.cross(cm - f.pivot) * f.rate;
        let mut req = ShedReq::new(Sel::GroupAll(Self::wing_group(side)), f.tear.flip(), swing + Vec3::new(0.0, 0.5, 1.5), (2.0 + f.rate.abs()) / self.scale());
        req.tear2 = f.outer;
        req.own2 = true;
        req.rigid = true;
        req.pre = f.mat();
        self.shed.push(req);
        if f.root < self.spec.aileron[0] + 0.05 * self.spec.span {
            self.dmg.aileron[side] = 0.0;
        }
        if !why.is_empty() {
            self.note(format!("{} WING TORN AWAY ({why})", Self::lr(side)));
        }
    }
    /// Let go of anything still hanging (the aircraft is about to go up in flames).
    fn release_folds(&mut self) {
        for s in 0..2 {
            self.release_fold(s, "");
        }
    }
    /// The tail breaks off through the tail cone, somewhere about `z`.
    fn snap_tail(&mut self, z: f32, why: &str) {
        if self.dmg.tail.is_some() {
            return;
        }
        let k = self.scale();
        let r = &mut self.rng;
        let tear = damage::tail_break(self.spec, z, r.range(-0.35, 0.35), r.range(-0.45, 0.45), r.range(0.06, 0.24) * k.sqrt());
        let mut req = ShedReq::new(Sel::Group(G_TAIL), tear.flip(), Vec3::new(r.range(-1.0, 1.0), 1.5, 3.0), 4.0 / k);
        req.rigid = true;
        self.shed.push(req);
        self.dmg.tail = tear;
        self.dmg.stab = [0.0; 2];
        self.dmg.elevator = [0.0; 2];
        self.dmg.fin = 0.0;
        self.dmg.rudder = 0.0;
        self.dmg.hull = (self.dmg.hull + 0.2).min(1.0);
        self.kick(DVec3::new(0.8, 0.0, 0.0));
        self.note(format!("TAIL BROKEN OFF ({why})"));
    }
    fn tail_z(&mut self) -> f32 {
        let [a, b] = self.spec.tail.break_z;
        self.rng.range(a, b)
    }
    /// Something hits hard enough to explode: anything still hanging off goes too.
    fn impact(&mut self, imp: Impact, cause: &'static str) -> Option<Impact> {
        self.cause = cause;
        self.release_folds();
        Some(imp)
    }
    fn lose_aileron(&mut self, side: usize, hanging: bool, why: &str) {
        if self.dmg.aileron[side] <= 0.0 || (hanging && self.dmg.aileron[side] < 1.0) {
            return;
        }
        if hanging {
            self.dmg.aileron[side] = 0.5;
            self.note(format!("{} AILERON HINGE BROKEN ({why})", Self::lr(side)));
            return;
        }
        self.dmg.aileron[side] = 0.0;
        self.shed.push(ShedReq::new(Sel::Role(Role::Aileron(side as u8)), Tear::NONE, Vec3::new(0.0, 1.0, 3.0), 9.0 / self.scale()));
        self.note(format!("{} AILERON TORN OFF ({why})", Self::lr(side)));
    }
    fn lose_flap(&mut self, side: usize, why: &str) {
        if self.dmg.flap[side] <= 0.0 {
            return;
        }
        self.dmg.flap[side] = 0.0;
        self.shed.push(ShedReq::new(Sel::Role(Role::Flap(side as u8)), Tear::NONE, Vec3::new(0.0, -1.0, 4.0), 8.0 / self.scale()));
        self.kick(DVec3::new(0.0, 0.0, if side == 0 { -0.4 } else { 0.4 }));
        self.note(format!("{} FLAP TORN OFF ({why})", Self::lr(side)));
    }
    fn lose_elevator(&mut self, side: usize, with_stab: bool, why: &str) {
        let tl = &self.spec.tail;
        let sgn = if side == 0 { -1.0 } else { 1.0 };
        if with_stab && self.dmg.stab[side] > 0.5 {
            self.dmg.stab[side] = 0.0;
            self.dmg.elevator[side] = 0.0;
            self.shed.push(ShedReq::new(Sel::Roles(&[Role::Stab, Role::Elevator]), Tear::new(Vec3::new(sgn, 0.0, 0.0), tl.stab_root, -1.0), Vec3::new(sgn * 2.0, 0.5, 2.0), 6.0 / self.scale()));
            self.kick(DVec3::new(-0.5, 0.0, sgn as f64 * 0.3));
            self.note(format!("{} STABILISER TORN OFF ({why})", Self::lr(side)));
        } else if self.dmg.elevator[side] > 0.5 {
            self.dmg.elevator[side] = 0.0;
            self.shed.push(ShedReq::new(Sel::Role(Role::Elevator), Tear::new(Vec3::new(sgn, 0.0, 0.0), tl.elev_root, -1.0), Vec3::new(sgn, 1.0, 3.0), 9.0 / self.scale()));
            self.note(format!("{} ELEVATOR TORN OFF ({why})", Self::lr(side)));
        }
    }
    /// The top of the rudder (and perhaps the fin with it) goes.
    fn lose_tail_top(&mut self, rudder: f32, fin: f32, why: &str) {
        let tl = &self.spec.tail;
        let rudder = rudder.min(self.dmg.rudder);
        let fin = fin.min(self.dmg.fin);
        let spin = 1.0 / self.scale();
        if rudder < self.dmg.rudder - 0.01 {
            if rudder <= 0.0 {
                self.shed.push(ShedReq::new(Sel::Role(Role::Rudder), Tear::NONE, Vec3::new(1.0, 1.0, 3.0), 7.0 * spin));
                self.note(format!("RUDDER TORN OFF ({why})"));
            } else {
                self.shed.push(ShedReq::new(Sel::Role(Role::Rudder), Tear::new(Vec3::Y, tl.rudder_y0 + tl.rudder_h * rudder, -1.0), Vec3::new(1.0, 1.5, 3.0), 8.0 * spin));
                self.note(format!("RUDDER DAMAGED ({why})"));
            }
            self.dmg.rudder = rudder;
        }
        if fin < self.dmg.fin - 0.01 {
            self.shed.push(ShedReq::new(Sel::Role(Role::Stab), Tear::new(Vec3::Y, tl.fin_y0 + tl.fin_h * fin, -1.0), Vec3::new(1.0, 2.0, 2.0), 6.0 * spin));
            self.dmg.fin = fin;
            self.note(format!("FIN BROKEN ({why})"));
        }
    }
    /// The propeller hits something solid (a jet: an engine pod scrapes the ground).
    fn prop_strike(&mut self, why: &str) {
        if let Engine::Piston(pe) = &self.spec.engine {
            if self.dmg.prop <= 0.0 {
                return;
            }
            let spinning = self.prop_rate > 40.0;
            if self.dmg.prop > 0.75 {
                self.dmg.prop = 0.5;
                self.shed.push(ShedReq::new(Sel::Role(Role::Prop), Tear::new(Vec3::Y, pe.blade_break_y, -1.0), Vec3::new(2.0, 4.0, -3.0), 30.0));
                self.note(format!("PROPELLER BLADE BROKEN ({why})"));
            }
            if spinning && self.dmg.engine > 0.0 {
                // A sudden stop wrecks the crankshaft.
                self.dmg.engine = 0.0;
                self.dmg.smoke_black = true;
                self.note("ENGINE SEIZED");
            }
            self.scuff(G_ENGINE, 0.6);
        }
    }
    /// An engine pod hanging under a wing scrapes or strikes the ground.
    fn pod_strike(&mut self, side: usize, hard: f32, why: &str) {
        if self.cool[side] > 0.0 {
            return;
        }
        self.cool[side] = 0.4;
        self.scuff(Self::wing_group(side), 0.5);
        self.engine_hit(0.12 + 0.2 * hard, why);
        self.dmg.smoke_black = true;
        self.note(format!("{} ENGINE POD STRUCK ({why})", Self::lr(side)));
    }
    fn engine_hit(&mut self, amount: f32, why: &str) {
        if self.dmg.engine <= 0.0 {
            return;
        }
        let was = self.dmg.engine;
        self.dmg.engine = (self.dmg.engine - amount).max(0.0);
        if was >= 1.0 {
            self.note(if self.spec.is_jet() { format!("ENGINE DAMAGED ({why})") } else { format!("ENGINE DAMAGED, LOSING OIL ({why})") });
        }
    }
    fn crack_windshield(&mut self, why: &str) {
        if self.dmg.crack[2] > 0.0 {
            return;
        }
        self.dmg.crack = [self.rng.range(-1.55, -1.25), self.rng.range(0.3, 0.6), self.rng.range(0.14, 0.3)];
        self.note(format!("WINDSHIELD CRACKED ({why})"));
    }
    fn lose_gear(&mut self, gi: usize, why: &str) {
        if !self.dmg.gear[gi] {
            return;
        }
        self.dmg.gear[gi] = false;
        self.shed.push(ShedReq::new(Sel::Role(Role::Gear(gi as u8)), Tear::NONE, Vec3::new(0.0, -1.0, 2.0), 6.0 / self.scale()));
        self.note(format!("{} TORN OFF ({why})", self.spec.gear[gi].name));
    }
    /// Above the never-exceed speed something lets go: whatever is next.
    fn flutter_loss(&mut self) {
        let r = self.rng.f();
        let s = (r * 2.0) as usize;
        if self.dmg.aileron[s] > 0.0 {
            self.lose_aileron(s, r > 0.6, "FLUTTER");
        } else if self.dmg.aileron[1 - s] > 0.0 {
            self.lose_aileron(1 - s, false, "FLUTTER");
        } else if self.dmg.rudder > 0.5 {
            self.lose_tail_top(0.45, 1.0, "FLUTTER");
        } else if self.dmg.elevator[s] > 0.5 {
            self.lose_elevator(s, false, "FLUTTER");
        } else if self.dmg.elevator[1 - s] > 0.5 {
            self.lose_elevator(1 - s, false, "FLUTTER");
        } else {
            self.break_wing(s, self.sta(0.78), "FLUTTER");
        }
    }

    /// Velocity through the air (planet frame).
    pub fn airspeed(&self) -> DVec3 {
        self.vel - self.wind - self.gust
    }

    // ---------------------------------------------------------------- engines

    /// The torque the engine's cylinders make (N m, before friction) with the
    /// throttle `open` (0 at the idle stop .. 1). A normally aspirated engine
    /// loses power with the air (Gagg-Ferrar); `boost` is the game's
    /// over-boost, nothing a real 172 has.
    fn engine_torque(pe: &crate::aircraft::Piston, open: f64, rho: f64, boost: bool, health: f64) -> f64 {
        let sigma = rho / RHO0;
        let alt = ((sigma - 0.12) / 0.88).max(0.0);
        (pe.power / pe.w_rated + Self::friction(pe, pe.w_rated)) * alt * (0.11 + 0.89 * open.clamp(0.0, 1.0)) * health * if boost { 1.35 } else { 1.0 }
    }
    /// Friction and pumping losses (N m) at shaft speed `w` (rad/s).
    fn friction(pe: &crate::aircraft::Piston, w: f64) -> f64 {
        pe.fric0 + pe.fric1 * (w / pe.w_rated).powi(2)
    }
    /// The fixed-pitch propeller turning at `w` rad/s with the air coming at
    /// it at `v`: its thrust (N) and the torque it takes (N m; negative when
    /// the air drives it round). `k` is what is left of the blades.
    /// Ct = Ct0 (1 - (J/Jt)²), Cp = Cp0 (1 - (J/Jp)²), written in n² so a
    /// stopped propeller needs no special case.
    pub fn prop_load(pe: &crate::aircraft::Piston, w: f64, v: f64, rho: f64, k: f64) -> (f64, f64) {
        let n = w / std::f64::consts::TAU;
        let vd = v.max(0.0) / pe.prop_d;
        let t = pe.ct0 * rho * pe.prop_d.powi(4) * (n * n - (vd / pe.j_t).powi(2));
        let q = pe.cp0 * rho * pe.prop_d.powi(5) / std::f64::consts::TAU * (n * n - (vd / pe.j_p).powi(2));
        // Windmilling, or stopped, it drags no worse than a flat plate the size of its blades.
        let t = t.max(-0.28 * 0.5 * rho * v * v);
        (t * k, q * k)
    }
    /// The revs (rad/s) an undamaged engine and propeller settle at.
    pub fn prop_equilibrium(pe: &crate::aircraft::Piston, throttle: f64, v: f64, rho: f64) -> f64 {
        let (mut lo, mut hi) = (0.0, 2.0 * pe.w_rated);
        for _ in 0..50 {
            let w = 0.5 * (lo + hi);
            if Self::engine_torque(pe, throttle, rho, false, 1.0) - Self::friction(pe, w) > Self::prop_load(pe, w, v, rho, 1.0).1 { lo = w } else { hi = w }
        }
        0.5 * (lo + hi)
    }
    /// The fan speed (fraction of N1) the throttle asks for.
    fn n1_target(fe: &crate::aircraft::Fan, throttle: f64) -> f64 {
        fe.idle + (1.0 - fe.idle) * throttle.clamp(0.0, 1.0)
    }
    /// Thrust of all the fans together (N) at fan speed `n1`, true airspeed
    /// `v`, height `h`: a fraction of the static thrust that grows steeply
    /// with the fan speed, lapsing with the density and with the Mach number
    /// (the faster it flies, the less the fan adds to the air it swallows).
    pub fn fan_thrust(fe: &crate::aircraft::Fan, n1: f64, v: f64, h: f64) -> f64 {
        let x = ((n1 - fe.idle) / (1.0 - fe.idle)).max(0.0);
        let frac = if n1 >= fe.idle { fe.idle_thrust + (1.0 - fe.idle_thrust) * x.powf(1.7) } else { fe.idle_thrust * (n1 / fe.idle).powi(2) };
        let m = v / sound_speed(h);
        let sigma = density(h) / RHO0;
        fe.n as f64 * fe.thrust * frac * sigma.powf(0.8) * (1.0 - fe.mach_lapse[0] * m + fe.mach_lapse[1] * m * m).max(0.05)
    }
    /// Thrust (N) once the engine has settled at this throttle.
    pub fn thrust(&self, throttle: f64, v: f64, h: f64) -> f64 {
        match &self.spec.engine {
            Engine::Piston(pe) => {
                let rho = density(h);
                Self::prop_load(pe, Self::prop_equilibrium(pe, throttle, v, rho), v, rho, 1.0).0
            }
            Engine::Fan(fe) => Self::fan_thrust(fe, Self::n1_target(fe, throttle), v / self.spec.speed_k, h),
        }
    }
    /// The engine already running as the throttle has it.
    fn settle_engine(&mut self, v: f64, h: f64) {
        match &self.spec.engine {
            Engine::Piston(pe) => self.prop_rate = Self::prop_equilibrium(pe, self.throttle, v, density(h)),
            Engine::Fan(fe) => {
                self.n1 = Self::n1_target(fe, self.throttle);
                self.prop_rate = self.n1 * fe.n1_rpm * std::f64::consts::TAU / 60.0;
            }
        }
    }

    // ---------------------------------------------------------------- trim

    /// Drag coefficient of the shock waves near the speed of sound (Lock's fourth-power rise).
    fn wave_cd(&self, mach: f64) -> f64 {
        if self.spec.cd_wave > 0.0 {
            self.spec.cd_wave * smooth(self.spec.mcrit, self.spec.mcrit + 0.25, mach)
        } else if self.spec.mcrit > 0.0 && mach > self.spec.mcrit {
            20.0 * (mach - self.spec.mcrit).powi(4)
        } else {
            0.0
        }
    }
    /// Angle of attack (rad) that carries the weight at true airspeed `v`.
    /// Trimmed to climb away at its present speed (off a catapult, hands off: the
    /// flight controls rotate it to its flying attitude).
    pub fn trim_to_climb(&mut self) {
        let v = self.airspeed().length().max(20.0);
        let a = self.alpha_level(v, RHO0, self.spec.flap_cl * self.flaps);
        self.trim = (a * 1.35 + 0.02).min(self.spec.alpha_trim);
    }
    fn alpha_level(&self, v: f64, rho: f64, flap_cl: f64) -> f64 {
        let s = self.spec;
        let q = 0.5 * rho * v * v;
        ((s.mass * 9.81 / (q * s.wing_area).max(1.0) - s.cl0 - flap_cl) / s.cl_a).clamp(-0.05, s.alpha_trim)
    }
    /// Drag (N) in steady flight at true airspeed `v`, height `h`.
    fn drag_level(&self, v: f64, h: f64, flaps: f64, gear: f64) -> f64 {
        let s = self.spec;
        let qs = 0.5 * density(h) * v * v * s.wing_area;
        let cl = s.mass * 9.81 / qs.max(1.0);
        qs * (s.cd0 + s.k_ind * cl * cl + s.cd_flap[0] * flaps + s.cd_flap[1] * flaps * flaps + s.cd_gear * gear + self.wave_cd(v / (sound_speed(h) * s.speed_k)))
    }
    /// Throttle to hold `v` while coming down at `descent` m/s.
    fn throttle_for(&self, v: f64, h: f64, flaps: f64, gear: f64, descent: f64) -> f64 {
        let need = self.drag_level(v, h, flaps, gear) - self.spec.mass * 9.81 * (descent / v.max(1.0));
        let (mut lo, mut hi) = (0.0, 1.0);
        for _ in 0..30 {
            let t = 0.5 * (lo + hi);
            if self.thrust(t, v, h) < need { lo = t } else { hi = t }
        }
        0.5 * (lo + hi)
    }

    /// Gaussian noise, unit variance (sum of three uniforms: close enough, and cheap).
    fn gauss(&mut self) -> f64 {
        (self.rng.range(-1.0, 1.0) + self.rng.range(-1.0, 1.0) + self.rng.range(-1.0, 1.0)) as f64
    }

    /// Turbulence: white noise through two first-order lags in series (a
    /// smoothed Dryden spectrum), the length scales turned into time by the
    /// airspeed. `a` is the coarse state, `b` the smoothed one; the second lag
    /// takes a fifth of the variance, which the output scale gives back.
    fn turbulence(&mut self, dt: f64, v: f64, upw: DVec3) {
        let air = self.air;
        let vt = v.max(12.0);
        let len = DVec3::new(air.len_h, air.len_h, air.len_w);
        let sig = DVec3::new(air.sigma_h, air.sigma_h, air.sigma_w);
        let n = DVec3::new(self.gauss(), self.gauss(), self.gauss());
        let e1 = (-vt * dt / len).exp();
        let e2 = (-vt * dt * 4.0 / len).exp();
        let one = DVec3::ONE;
        self.gust_a = self.gust_a * e1 + sig * (one - e1 * e1).max(DVec3::ZERO).sqrt() * n;
        self.gust_b = self.gust_b * e2 + self.gust_a * (one - e2);
        let g = self.gust_b * 1.118;
        // Along and across the flight path, and up.
        let air_v = self.vel - self.wind;
        let mut along = air_v - upw * air_v.dot(upw);
        if along.length_squared() < 1.0 {
            along = self.forward() - upw * self.forward().dot(upw);
        }
        let along = along.normalize_or_zero();
        let across = along.cross(upw);
        self.gust = along * g.x + across * g.y + upw * g.z;
        // Rotational gusts: the vertical gust differs from one wing tip to the
        // other (roll), from the nose to the tail (pitch), and the side gust
        // from the nose to the fin (yaw). Eddies about the size of the airframe;
        // a big aircraft averages more of them out.
        let lr = self.spec.b * 2.5;
        let big = (11.0 / self.spec.b).sqrt();
        let sr = DVec3::new(0.012 * air.sigma_w, 0.020 * air.sigma_h, 0.03 * air.sigma_w) * big;
        let n = DVec3::new(self.gauss(), self.gauss(), self.gauss());
        let e1 = (-vt * dt / lr).exp();
        let e2 = (-vt * dt * 4.0 / lr).exp();
        self.rot_a = self.rot_a * e1 + sr * (1.0 - e1 * e1).max(0.0).sqrt() * n;
        self.rot_b = self.rot_b * e2 + self.rot_a * (1.0 - e2);
        self.gust_rot = self.rot_b * 1.118 * smooth(5.0, 25.0, v);
    }

    /// Points on the airframe that meet the ground (body frame), as it is now:
    /// what has broken off has no points left, a hanging wing has its own.
    fn contacts(&self, out: &mut Vec<(DVec3, Part)>) {
        let s = self.spec;
        let d = &self.dmg;
        let v = DVec3::new;
        if d.prop > 0.0 && !s.is_jet() {
            out.push((s.hard[0], Part::Prop));
            out.push((s.hard[6], Part::Prop));
        }
        for &(p, part) in s.contacts {
            // An engine pod goes with the wing it hangs from.
            if let Part::Pod(side) = part {
                if p[0].abs() as f32 > d.wing[side] - 0.3 || d.fold[side].on {
                    continue;
                }
            }
            out.push((v(p[0], p[1], p[2]), part));
        }
        let tl = &s.tail;
        if d.tail.is_some() {
            // The stump of the tail cone, a little ahead of the break.
            let z = (d.tail.d / d.tail.n.z.max(0.5)) as f64 - 0.15 * self.scale() as f64;
            let (top, _, bot) = (s.fuse)(z as f32);
            out.push((v(0.0, bot as f64, z), Part::Belly));
            out.push((v(0.0, top as f64, z), Part::Roof));
        } else {
            out.extend(s.tail_pts.iter().map(|p| (v(p[0], p[1], p[2]), Part::Tail)));
            if d.fin > 0.05 {
                out.push((tl.fin_base.lerp(tl.fin_tip, d.fin as f64), Part::Fin));
                out.push((tl.fin_base.lerp(tl.fin_tip, 0.52 * d.fin as f64), Part::Fin));
            }
            for side in 0..2 {
                if d.stab[side] > 0.5 {
                    out.push((tl.stab_tip * DVec3::new(if side == 0 { -1.0 } else { 1.0 }, 1.0, 1.0), Part::Stab(side)));
                }
            }
        }
        for side in 0..2 {
            let b = d.wing[side];
            let k = self.scale();
            let mut at = |st: f32, tip: bool| {
                for (xc, skin) in [(0.35, 1.0), (0.35, -1.0), (0.0, 0.0)] {
                    out.push((damage::wing_pt(s, side, st, xc, skin).as_dvec3(), Part::Wing(side, st)));
                }
                if tip {
                    out.push((damage::wing_pt(s, side, st, 0.92, 0.0).as_dvec3(), Part::Wing(side, st)));
                }
            };
            for f in s.wing_st {
                let st = f * s.span;
                if st < b - 0.3 * k {
                    at(st, false);
                }
            }
            if b > 0.4 * k {
                at(b - 0.08 * k, true);
            }
            let f = &d.fold[side];
            if f.on {
                let m = f.mat();
                for (st, skin) in [(f.tip - 0.08 * k, 1.0), (f.tip - 0.08 * k, -1.0), ((f.root + f.tip) * 0.5, -1.0), ((f.root + f.tip) * 0.5, 1.0)] {
                    out.push((m.transform_point3(damage::wing_pt(s, side, st, 0.4, skin)).as_dvec3(), Part::Fold(side)));
                }
            }
        }
    }

    /// Wrecked: it will not fly again (a wing or the tail gone, the cabin
    /// caved in, or lying on its back).
    pub fn wrecked(&self) -> bool {
        let d = &self.dmg;
        let short = self.sta(0.58);
        d.wing[0] < short || d.wing[1] < short || d.fold.iter().any(|f| f.on) || d.tail.is_some() || d.hull > 0.6 || self.sink > 0.3 || (self.on_hull && self.up().dot(self.pos.normalize()) < 0.2)
    }

    /// Advance one physics step. Returns an impact if the aircraft was destroyed.
    pub fn step(&mut self, dt: f64, c: &Controls, world: &World) -> Option<Impact> {
        let spec = self.spec;
        let hd = &spec.handling;
        self.t += dt;
        if self.hold {
            self.vel = self.wind * 0.0;
            self.omega = DVec3::ZERO;
            if !spec.is_jet() {
                self.prop_rate += (65.0 - self.prop_rate) * (dt * 2.0).min(1.0);
            }
            self.prop += dt * self.prop_rate;
            return None;
        }
        self.dmg_prev = self.dmg;
        self.dmg.vibe = 0.0;
        for c in self.cool.iter_mut() {
            *c -= dt;
        }
        self.fold_tq = [0.0; 2];
        self.throttle = (self.throttle + c.throttle * dt * 0.6).clamp(0.0, 1.0);
        // Control surfaces do not snap.
        let target = DVec3::new(c.pitch, c.yaw, c.roll);
        self.stick += (target - self.stick) * (dt * 7.0).min(1.0);
        // Flaps run on an electric motor (or hydraulics), the gear on a pump;
        // the squat switch keeps the gear down while there is weight on the wheels.
        let flap_target = spec.flap_frac(self.flap_set);
        self.flaps += (flap_target - self.flaps).clamp(-spec.flap_rate * dt, spec.flap_rate * dt);
        let slat_target = if spec.slats_at > 0 && self.flap_set >= spec.slats_at { 1.0 } else { 0.0 };
        self.slats += (slat_target - self.slats).clamp(-dt * 0.12, dt * 0.12);
        if !self.gear_down && self.on_ground {
            self.gear_down = true;
        }
        self.gear = (self.gear + if self.gear_down { spec.gear_rate * dt } else { -spec.gear_rate * dt }).clamp(0.0, 1.0);

        let (lat, lon, h) = geo::pos_to_lla(self.pos);
        let upw = self.pos.normalize();
        let rho = density(h);
        // Limits are set in indicated airspeed (the dynamic pressure the airframe feels).
        let eas_k = (rho / RHO0).sqrt();
        let ground = world.ground(lat, lon);
        // The turbulent air the aircraft is flying through this instant.
        let v_air = (self.vel - self.wind).length();
        self.turbulence(dt, v_air, upw);
        let vb = self.rot.inverse() * self.airspeed();
        let v = vb.length();
        // The Mach number the airframe feels: one flying `speed_k` times as fast
        // as the type it scales up meets its shock waves that much later.
        let mach = v / (sound_speed(h) * spec.speed_k);
        let mut force_b = DVec3::ZERO;
        let mut wdot = DVec3::ZERO;
        let q = 0.5 * rho * v * v;
        let k = self.scale();

        // Spoilers: the speedbrake lever in flight; on the ground, with the
        // power off and still rolling fast, they all come up to dump the lift.
        let ground_spoil = self.on_ground && self.throttle < 0.08 && self.vel.length() > 15.0 && spec.spoiler_cd > 0.0;
        let sp_target = if ground_spoil { 1.0 } else { self.speedbrake };
        self.spoilers += (sp_target - self.spoilers).clamp(-dt * 0.8, dt * 0.8);

        // What is left to fly with.
        let dm = self.dmg;
        let lost = [dm.wing_lost(dm.wing[0]) as f64, dm.wing_lost(dm.wing[1]) as f64];
        let area = (spec.wing_area - lost[0] - lost[1]).max(spec.wing_area * 0.25);
        let ail = [dm.aileron[0] * dm.aileron_frac(dm.wing[0]), dm.aileron[1] * dm.aileron_frac(dm.wing[1])];
        // A hanging aileron does nothing for you.
        let roll_auth = ail.iter().map(|&a| if a < 0.99 && a > 0.0 { 0.0 } else { a as f64 }).sum::<f64>() * 0.5;
        let flap_eff = (dm.flap[0] + dm.flap[1]) as f64 * 0.5;
        let el = (dm.elevator[0] * dm.stab[0] + dm.elevator[1] * dm.stab[1]) as f64 * 0.5;
        let st = (dm.stab[0] + dm.stab[1]) as f64 * 0.5;
        let fin = (dm.fin * 0.6 + dm.rudder * 0.4) as f64;
        let rud = dm.rudder as f64;

        if v > 0.5 {
            let vh = vb / v;
            let alpha = (-vb.y).atan2(-vb.z);
            let beta = (vb.x / v).clamp(-1.0, 1.0).asin();
            self.alpha = alpha;
            // Flaps add lift (and stall a little earlier); flaps and gear add drag.
            let flap_cl = spec.flap_cl * self.flaps * flap_eff;
            let stall_a = self.stall_alpha();
            let stall = smooth(stall_a, stall_a + 0.13, alpha.abs());
            self.stalled = stall > 0.5 && v > 8.0;
            // Ground effect: within a span of the ground the wing tip vortices are
            // squashed, so there is less induced drag and a little more lift (the
            // float in the flare).
            let ge = match &ground {
                Some(g) if !self.on_ground || v > 5.0 => {
                    let r = 16.0 * ((h - g.h - 0.5).max(0.2) / spec.b);
                    r * r / (1.0 + r * r)
                }
                _ => 1.0,
            };
            let cl = ((spec.cl0 + flap_cl + spec.cl_a * alpha) * (1.0 - stall) + (2.0 * alpha).sin() * 0.95 * stall) * (1.0 + 0.12 * (1.0 - ge)) - spec.spoiler_cl * self.spoilers;
            // Ragged stumps and missing fairings all drag.
            let rag = ((spec.span - dm.wing[0]) + (spec.span - dm.wing[1])) as f64 / k as f64;
            let cd = spec.cd0
                + spec.k_ind * cl * cl * ge
                + stall * 1.2 * alpha.sin().powi(2)
                + 0.6 * beta * beta
                + spec.cd_flap[0] * self.flaps
                + spec.cd_flap[1] * self.flaps * self.flaps
                + spec.cd_gear * self.gear
                + spec.spoiler_cd * self.spoilers
                + self.wave_cd(mach)
                + 0.02 * rag
                + 0.03 * (1.0 - st)
                + 0.01 * (1.0 - rud);
            // Buffet as the airflow starts to break away over the wing, before it
            // stalls (and, near its Mach limit, as the shock waves shake it).
            self.dmg.vibe += (smooth(stall_a - 0.07, stall_a + 0.03, alpha.abs()) * (1.0 - 0.5 * stall) * 0.9 * (q / (hd.q_ref * 0.7)).min(1.0)) as f32;
            if spec.mcrit > 0.0 {
                self.dmg.vibe += smooth(spec.mmo - 0.02, spec.mmo + 0.05, mach) as f32 * 0.8;
            }
            let lift_dir = DVec3::X.cross(vh).normalize_or_zero();
            force_b += lift_dir * (q * area * cl) - vh * (q * area * cd) + DVec3::X * (-0.9 * beta * q * spec.wing_area);

            // Direct angular accelerations, scaled by dynamic pressure; the
            // damping of a rotation grows only with airspeed (it comes from the
            // angle the rotation adds at the wing tip or tail, ω·arm / V), so a
            // slow aircraft is sluggish and a fast one crisp.
            let a = (q / hd.q_ref).clamp(0.0, hd.a_max);
            let ad = (a * hd.damp_v / v.max(20.0 * spec.speed_k)).clamp(hd.ad_min, hd.ad_max);
            // Relative to the air, which is itself turning in the eddies.
            let om = self.omega - self.gust_rot;
            // Auto-trim: while the pilot is flying the stick it winds the trim to
            // the angle that carries the weight at this speed; let go and it stays
            // put, so the aircraft holds the airspeed it was trimmed for, as a real
            // one does: power then makes it climb or descend, and a bump sets it
            // porpoising gently back to that speed. Without an elevator the trim
            // stays where it was.
            let level = self.up().dot(upw).max(0.0);
            let alpha_1g = ((spec.mass * 9.81 * level / (q * area).max(1.0) - spec.cl0 - flap_cl + spec.spoiler_cl * self.spoilers) / spec.cl_a).clamp(-0.05, spec.alpha_trim);
            if el > 0.05 && self.stick.x.abs() > 0.06 {
                self.trim += (alpha_1g - self.trim) * (dt * 0.5).min(1.0);
            }
            let mut alpha_cmd = self.trim + self.stick.x * hd.alpha_ctl * el;
            if spec.fbw_g > 0.0 {
                // Fly-by-wire: the stick asks for load, never more than the limit.
                let alpha_for = |g: f64| (g * spec.mass * 9.81 / (q * area).max(1.0) - spec.cl0 - flap_cl) / spec.cl_a;
                alpha_cmd = alpha_cmd.clamp(alpha_for(-0.4 * spec.fbw_g), alpha_for(spec.fbw_g));
            }
            // Pitch stiffness comes from the tailplane; without one the nose wanders off.
            let stiff = hd.pitch_k * st - hd.pitch_k_lost * (1.0 - st);
            wdot.x = a * (stiff * (alpha_cmd - alpha) + self.stick.x * hd.pitch_ctl * el) - hd.pitch_d * ad * (0.3 + 0.7 * st) * om.x;
            // Roll damping fades in a stall, and past it turns round: the wing going
            // down meets the air at a still higher angle and loses still more lift.
            // An airliner's outboard ailerons lock out once the flaps are up.
            let roll_damp = (1.0 - 1.5 * stall).max(-0.35);
            let clean = hd.roll_clean + (1.0 - hd.roll_clean) * smooth(0.0, 0.1, self.flaps);
            wdot.z = a * (-hd.roll_ctl * clean * self.stick.z * roll_auth - hd.roll_beta * beta) - hd.roll_d * ad * roll_damp * om.z;
            wdot.y = a * (-hd.yaw_ctl * self.stick.y * rud - hd.yaw_beta * beta * fin) - hd.yaw_d * ad * (0.25 + 0.75 * fin) * om.y;
            // Dihedral: a banked aircraft slips toward the low wing, which then
            // meets the air at a higher angle and lifts it back, slowly (but not upside down).
            wdot.z -= a * hd.dihedral * self.right().dot(upw) * self.up().dot(upw).max(0.0);
            // Adverse yaw: the aileron that goes down drags its wing back.
            wdot.y += a * hd.adverse * self.stick.z * roll_auth * cl.clamp(0.0, 1.6);
            // A stalled wing drops the nose, and one wing goes first: which one
            // wanders, and a yawing aircraft drops the wing that is going back (a spin).
            // (Only with real airflow over the wing: parked in a tailwind the
            // angle of attack is backwards, and that is no stall.)
            let flow = smooth(0.02, 0.1, a);
            wdot.x -= stall * flow * hd.stall_pitch * alpha.signum();
            self.drop = (self.drop + (self.gauss() * 1.2 - self.drop * 0.4) * dt).clamp(-1.0, 1.0);
            wdot.z += stall * flow * a.max(0.25) * (hd.stall_drop * self.drop + hd.stall_yaw * om.y);
            // A wing that ends short of the tip lifts less than its opposite: it
            // drops, and the drag of its ragged end pulls the nose round.
            let arm = |b: f32| ((b + spec.span) * 0.5) as f64;
            let roll = (lost[0] * arm(dm.wing[0]) - lost[1] * arm(dm.wing[1])) * q * (cl.max(0.05) + 0.1) * 0.6 / spec.inertia.z;
            wdot.z += roll;
            wdot.y -= (lost[1] * arm(dm.wing[1]) - lost[0] * arm(dm.wing[0])) * q * 0.06 / spec.inertia.y;
            // One flap gone with the other out: the lift is lopsided (retract them!).
            wdot.z += (dm.flap[1] - dm.flap[0]) as f64 * self.flaps * q * spec.wing_area * 0.275 * 1.8 * k as f64 * 0.35 / spec.inertia.z;
            // A hanging aileron droops (lifting that wing) and flaps about.
            for s in 0..2 {
                if dm.aileron[s] > 0.0 && dm.aileron[s] < 1.0 {
                    let sgn = if s == 0 { -1.0 } else { 1.0 };
                    wdot.z += sgn * a * (0.45 + 0.35 * (self.t * 23.0 + s as f64).sin()) * hd.roll_ctl / 4.0;
                    self.dmg.vibe += 0.12 * a as f32;
                }
            }
        }
        // The engines. A damaged engine (oil leaking away, a pod struck) loses
        // power and fails; a snapped propeller blade shakes it to pieces; with
        // no blades it just windmills.
        if self.dmg.engine < 1.0 && self.dmg.engine > 0.0 {
            self.dmg.engine = (self.dmg.engine - dt as f32 * (0.002 + 0.01 * (1.0 - self.dmg.engine))).max(0.0);
            if self.dmg.engine <= 0.0 {
                self.note(if spec.is_jet() { "ENGINES FAILED" } else { "ENGINE FAILED" });
            }
        }
        let eng = (self.dmg.engine as f64).powf(0.6);
        // The engine takes a moment to answer the throttle.
        self.charge += (self.throttle - self.charge) * (dt * 4.0).min(1.0);
        match &spec.engine {
            Engine::Piston(pe) => {
                let prop_k = if dm.prop > 0.75 { 1.0 } else if dm.prop > 0.25 { 0.45 } else { 0.0 };
                // The fixed-pitch propeller: the engine's torque against what the
                // blades take at these revs and this airspeed. Faster through the air,
                // the blades meet it at a smaller angle and take less, so the revs
                // rise (and over-rev in a dive); slow, the propeller holds the engine
                // well below its rated revs and power. Without power it windmills, and
                // only slowing nearly to the stall stops it.
                let w = self.prop_rate.max(0.0);
                let v_ax = (-vb.z).max(0.0);
                let (mut thrust, q_prop) = Self::prop_load(pe, w, v_ax, rho, prop_k);
                let q_eng = Self::engine_torque(pe, self.charge, rho, c.boost, eng);
                let drive = q_eng - q_prop;
                if prop_k <= 0.0 {
                    // No blades left: nothing to turn.
                    self.prop_rate += (0.0 - self.prop_rate) * (dt * 2.0).min(1.0);
                } else if w > 1.0 || drive > 40.0 {
                    self.prop_rate = (w + (drive - Self::friction(pe, w)) / pe.prop_i * dt).max(0.0);
                } else {
                    // Stopped: the compression holds it until the air pushes harder.
                    self.prop_rate = 0.0;
                }
                self.power = ((q_eng - Self::friction(pe, w)) * w / pe.power).max(0.0);
                if dm.prop <= 0.25 {
                    thrust = 0.0;
                }
                force_b += DVec3::NEG_Z * thrust;
                // The propeller turns clockwise seen from the cockpit: its torque rolls the
                // aircraft left and its spiralling slipstream (and P-factor, climbing
                // slowly) yaws it left. The fin and ailerons are rigged for the cruise, so
                // it is full power at low speed that needs right rudder.
                if v > 2.0 {
                    let slow = (1.0 - v / 60.0).max(0.0);
                    let tk = thrust.max(0.0) / pe.static_thrust;
                    wdot.y += tk * slow * (0.22 + 0.25 * self.alpha.max(0.0));
                    wdot.z += tk * slow * 0.12 * if self.on_ground { 0.3 } else { 1.0 };
                }
                if dm.prop > 0.25 && dm.prop < 0.75 {
                    let spin = (self.prop_rate / 250.0) as f32;
                    self.dmg.vibe += spin * 1.4;
                    if self.dmg.engine > 0.0 {
                        self.dmg.engine = (self.dmg.engine - dt as f32 * 0.06 * spin).max(0.0);
                    }
                }
                if self.dmg.engine < 1.0 && self.dmg.engine > 0.0 {
                    self.dmg.oil = (self.dmg.oil + dt as f32 * 0.04 * (1.0 - self.dmg.engine) * (self.prop_rate / 250.0) as f32).min(1.0);
                }
                self.dmg.smoke = if self.dmg.engine < 1.0 { (1.0 - self.dmg.engine) * (0.3 + 0.7 * (self.prop_rate / 250.0) as f32) * if self.dmg.engine <= 0.0 { 0.5 } else { 1.0 } } else { 0.0 };
            }
            Engine::Fan(fe) => {
                // Reversers: only with weight on the wheels and the thrust levers
                // closed; deployed, the fans spool up to reverse thrust.
                let rev_want = c.reverse && self.on_ground && self.throttle < 0.05;
                let rate = dt / fe.reverser_time;
                self.reverser = (self.reverser + if rev_want { rate } else { -rate }).clamp(0.0, 1.0);
                let reversing = rev_want && self.reverser > 0.95;
                // The fans spool slowly from idle and quicker near the top; a
                // dead engine windmills in the airflow.
                let mut target = if reversing { 0.84 } else { Self::n1_target(fe, self.throttle) };
                target = target.min(fe.idle + (1.0 - fe.idle) * eng);
                if self.dmg.engine <= 0.0 {
                    target = (0.02 + v / 900.0).min(fe.idle * 0.8);
                }
                let up = target > self.n1;
                let kr = if up { fe.spool_up * (0.3 + 0.7 * self.n1) } else { fe.spool_down };
                self.n1 += (target - self.n1) * (dt * kr).min(1.0);
                self.prop_rate = self.n1 * fe.n1_rpm * std::f64::consts::TAU / 60.0;
                let mut thrust = Self::fan_thrust(fe, self.n1, v.max(0.0) / spec.speed_k, h) * if c.boost { fe.boost } else { 1.0 };
                // An afterburner (not an airliner's take-off bump) lights with a
                // thump a moment after it is asked for and dies away as quickly.
                let lit = c.boost && fe.boost > 1.2 && self.dmg.engine > 0.0 && self.n1 > fe.idle * 0.9;
                self.burner += ((if lit { 1.0 } else { 0.0 }) - self.burner) * (dt * if lit { 5.0 } else { 8.0 }).min(1.0);
                self.power = thrust / (fe.n as f64 * fe.thrust);
                thrust *= 1.0 - self.reverser * (1.0 + fe.reverse);
                force_b += DVec3::NEG_Z * thrust;
                self.dmg.smoke = if self.dmg.engine < 1.0 { (1.0 - self.dmg.engine) * (0.3 + 0.7 * self.n1 as f32) } else { 0.0 };
            }
        }
        self.prop += dt * self.prop_rate;

        // Ground contact.
        let mut force_w = self.rot * force_b - upw * (spec.mass * 9.81);
        let mut torque_b = DVec3::ZERO;
        self.on_ground = false;
        // Nothing is solid for the first moments after a spawn: the terrain
        // under the aircraft is still a coarse guess until its tile arrives.
        let reach = 60.0 + 2.0 * spec.b;
        // (a ship's deck is there at once: it does not wait on any tile)
        let on_ship = world.decks.near(self.pos, reach);
        let close = (self.t > 1.5 && (ground.as_ref().map(|g| h - g.h < reach).unwrap_or(false) || world.building(lat, lon).is_some())) || on_ship;
        self.brushes = (self.brushes - dt * 0.5).max(0.0);
        let mut skidding = false;
        if close {
            if let Some(imp) = self.trees(world) {
                return self.impact(imp, "tree");
            }
            // How deep a point is in what it touches: depth, water, the
            // building hit, whether that is a wall (a building's or a ship's
            // side), and how fast the surface itself moves (a ship's deck).
            let hit = |p: DVec3, world: &World| -> Option<(f64, bool, Option<(geo::TileId, usize)>, bool, DVec3)> {
                if on_ship {
                    if let Some(d) = world.decks.probe(p) {
                        return Some((d.pen, false, None, d.wall, d.vel));
                    }
                }
                let (la, lo, ph) = geo::pos_to_lla(p);
                let g = world.ground(la, lo)?;
                if let Some(b) = world.building(la, lo) {
                    if ph < b.top && ph > b.base - if b.bridge { 0.0 } else { 3.0 } {
                        return Some((b.top - ph, false, Some((b.tile, b.idx)), true, DVec3::ZERO));
                    }
                } else if ph < g.h {
                    // Under the 3D tiles the ground is the mesh, buildings and all:
                    // into one of them, it is that building that was hit.
                    if let Some(b) = world.photo_building(la, lo, g.h) {
                        if ph > b.base - 3.0 {
                            return Some((g.h - ph, false, Some((b.tile, b.idx)), true, DVec3::ZERO));
                        }
                    }
                }
                Some((g.h - ph, g.water, None, false, DVec3::ZERO))
            };
            // Every point of the airframe that can touch something, and what it
            // takes to break it: gentle enough, things only scrape; harder, the
            // part that hit comes off (a wing, the tail, the gear); only a hit
            // hard enough to burst the cabin or the tanks ends in fire.
            let mut pts = std::mem::take(&mut self.pts);
            pts.clear();
            self.contacts(&mut pts);
            let mut walled = false;
            for &(r, part) in &pts {
                let p = self.pos + self.rot * r;
                let Some((mut pen, water, building, wall, sv)) = hit(p, world) else { continue };
                if water {
                    // A ditched aircraft settles lower in the water as it floods.
                    pen -= self.sink;
                }
                if pen <= 0.0 {
                    continue;
                }
                // (relative to what it touches: a ship's deck moves)
                let mut pv = self.vel + self.rot * self.omega.cross(r) - sv;
                if let Part::Fold(s) = part {
                    let f = &self.dmg.fold[s];
                    pv += self.rot * (f.axis.cross(r.as_vec3() - f.pivot) * f.rate).as_dvec3();
                }
                let vn = pv.dot(upw);
                let closing = (-vn).max(0.0);
                let sp = pv.length();
                let crash = Impact { pos: p, vel: self.vel, water, ground_h: geo::pos_to_lla(p).2 + if wall { 0.0 } else { pen }, building };
                let up_right = self.up().dot(upw) > 0.0;
                if wall && pen > 1.0 {
                    // Into the side of a building: all of the speed goes into it.
                    match part {
                        Part::Wing(s, st) => {
                            // A wing's root goes in with the cabin right behind it.
                            if st < self.sta(0.33) && sp > 28.0 {
                                return self.impact(crash, "wing root into building");
                            }
                            if self.cool[s] <= 0.0 {
                                let into = (self.rng.range(0.1, 0.6) + (sp as f32 / 30.0).min(2.0) * self.rng.f()) * k;
                                let (rough, up) = (self.rng.range(0.0, 0.5), self.rng.f() < 0.5);
                                self.snap_wing(s, st - into, rough, if sp < 25.0 { 0.35 } else { 0.1 }, up, "BUILDING");
                                self.cool[s] = 0.25;
                                self.vel *= 0.85;
                            }
                        }
                        Part::Pod(s) => self.pod_strike(s, 1.0, "BUILDING"),
                        Part::Fold(s) => self.release_fold(s, "BUILDING"),
                        Part::Fin => self.lose_tail_top(0.0, 0.3, "BUILDING"),
                        Part::Stab(s) => self.lose_elevator(s, true, "BUILDING"),
                        Part::Tail if sp > 12.0 => {
                            let z = self.tail_z();
                            self.snap_tail(z, "BUILDING");
                        }
                        _ if sp > 18.0 => return self.impact(crash, "building"),
                        _ if !walled => {
                            // Taxiing into a wall: crumpled, stopped and pushed back off it.
                            walled = true;
                            if matches!(part, Part::Prop | Part::Nose) {
                                self.prop_strike("BUILDING");
                                if sp > 8.0 {
                                    self.engine_hit(0.35, "BUILDING");
                                    self.crack_windshield("BUILDING");
                                }
                            }
                            self.dmg.hull = (self.dmg.hull + sp as f32 * 0.03).min(1.0);
                            self.scuff(part_group(part), 0.6);
                            let back = (self.vel - upw * self.vel.dot(upw)).normalize_or_zero();
                            self.pos -= back * 0.25;
                            self.vel = -self.vel * 0.15;
                            self.omega *= 0.4;
                            self.dmg.vibe += 2.0;
                        }
                        _ => {}
                    }
                    continue;
                }
                // Ground, a roof or the water. First what this hit breaks.
                let nose_on = matches!(part, Part::Nose | Part::Prop);
                let body = matches!(part, Part::Belly | Part::Side | Part::Roof);
                if water && (nose_on || body) && (closing > 9.0 || (sp > 42.0 && closing > 4.0)) {
                    return self.impact(crash, "water");
                }
                match part {
                    Part::Prop | Part::Nose => {
                        if self.dmg.prop > 0.25 && (part == Part::Prop || self.prop_rate > 30.0) && (self.prop_rate > 5.0 || sp > 3.0) {
                            self.prop_strike(if water { "WATER" } else { "GROUND" });
                        }
                        // Nose first into the ground: the engine is driven back into the cabin.
                        if closing > 15.0 || (sp > 50.0 && closing > 7.5) {
                            return self.impact(crash, "water");
                        }
                        if closing > 6.0 && self.cool[3] <= 0.0 {
                            self.cool[3] = 0.3;
                            self.engine_hit(0.15 + (closing as f32 - 6.0) * 0.05, "NOSE IMPACT");
                            self.crack_windshield("NOSE IMPACT");
                            self.dmg.hull = (self.dmg.hull + (closing as f32 - 6.0) * 0.04).min(1.0);
                            if closing > 8.5 {
                                self.lose_gear(0, "NOSE IMPACT");
                            }
                        }
                    }
                    Part::Pod(s) => {
                        // An engine pod dug in hard enough tears open its tanks' wing.
                        if closing > 14.0 && !water {
                            return self.impact(crash, "engine");
                        }
                        if closing > 0.5 || sp > 5.0 {
                            self.pod_strike(s, (closing as f32 / 6.0).min(1.0), if water { "WATER" } else { "GROUND" });
                        }
                    }
                    Part::Belly | Part::Side | Part::Roof => {
                        // The cabin can take a hard landing on its belly; on its side or
                        // its roof it caves in sooner. Fast and steep, nothing survives.
                        let lim = if part == Part::Belly { 15.0 } else { 12.0 };
                        if closing > lim || (sp > 55.0 && closing > lim * 0.55) {
                            return self.impact(crash, "cabin");
                        }
                        if closing > 4.5 && self.cool[3] <= 0.0 {
                            self.cool[3] = 0.3;
                            self.dmg.hull = (self.dmg.hull + (closing as f32 - 4.5) * 0.05).min(1.0);
                            self.dmg.vibe += 3.0;
                            if closing > 6.5 || part != Part::Belly {
                                self.crack_windshield(if part == Part::Roof { "ROLLED OVER" } else { "HARD IMPACT" });
                            }
                            if part == Part::Belly && closing > 7.0 {
                                for gi in 0..spec.gear.len() {
                                    if self.gear > 0.5 {
                                        self.lose_gear(gi, "HARD IMPACT");
                                    }
                                }
                            }
                        }
                    }
                    Part::Tail => {
                        // A tail slammed down (or dug in sliding fast) snaps off the cone.
                        if closing > 7.5 || (closing > 4.0 && sp > 25.0) {
                            let z = self.tail_z();
                            self.snap_tail(z, "GROUND");
                            continue;
                        }
                    }
                    Part::Fin => {
                        // Scraped along the ground upside down, the fin wears away.
                        if self.cool[2] <= 0.0 && (closing > 2.5 || (sp > 7.0 && pen > 0.04)) {
                            self.cool[2] = 0.35;
                            let fin = (self.dmg.fin - self.rng.range(0.2, 0.5)).max(0.0);
                            self.lose_tail_top((self.dmg.rudder - 0.4).max(0.0).min(fin + 0.1), fin, "GROUND");
                            continue;
                        }
                    }
                    Part::Stab(s) => {
                        if closing > 3.0 || (sp > 12.0 && pen > 0.06) {
                            self.lose_elevator(s, true, "GROUND");
                            continue;
                        }
                    }
                    Part::Wing(s, st) => {
                        // A wing struck into the ground breaks in from where it hit,
                        // the further the harder; scraped along fast it tears too.
                        let hard = closing as f32 / 4.0 + sp as f32 / 45.0;
                        if self.cool[s] <= 0.0 && (closing > 3.5 || (sp > 14.0 && pen > 0.1) || (closing > 2.0 && sp > 25.0)) {
                            // (One blow takes a couple of metres at most: breaking the
                            // outer part soaks up what would have hit further in.)
                            let into = (self.rng.range(0.05, 0.5) + (hard - 1.0).clamp(0.0, 3.0) * self.rng.range(0.2, 0.6)) * k;
                            let station = (st - into).max(0.3 * k);
                            // Through the tank fast: the fuel goes up.
                            if station < spec.tank[1] && sp > 32.0 && closing > 8.0 && !water {
                                return self.impact(crash, "wing tank");
                            }
                            let rough = self.rng.range(0.35, 1.0);
                            let fold = if water { 0.2 } else if sp < 30.0 { 0.5 } else { 0.25 };
                            self.snap_wing(s, station, rough, fold, up_right, if water { "WATER" } else { "GROUND" });
                            self.cool[s] = 0.3;
                            self.vel *= 0.94;
                            continue;
                        }
                    }
                    Part::Fold(_) => {}
                }
                // Then it skids: a spring out of the surface, friction along it.
                // Water gives less and drags far more.
                let hl = &spec.hull;
                self.pos += upw * if water { pen * 0.3 } else { pen };
                let (kk, cc, mu) = if water { (hl.k * 0.42, hl.c * 0.67, 0.9) } else { (hl.k, hl.c, 0.45) };
                let nf = (pen * kk - vn * cc).clamp(0.0, hl.nf_max);
                let tang = pv - upw * vn;
                let tl = tang.length();
                let f = upw * nf - tang.normalize_or_zero() * (mu * nf).min(tl * hl.drag);
                force_w += f;
                let fb_ = self.rot.inverse() * f;
                torque_b += r.cross(fb_);
                if let Part::Fold(s) = part {
                    let fd = &self.dmg.fold[s];
                    self.fold_tq[s] += (r.as_vec3() - fd.pivot).as_dvec3().cross(fb_).dot(fd.axis.as_dvec3());
                }
                skidding = true;
                if water {
                    self.sink = (self.sink + dt * 0.03).min(1.6 * k as f64);
                    if tl > 4.0 {
                        self.splashes.push((p, tang * 0.4 + upw * (tl * 0.15)));
                    }
                    continue;
                }
                self.scuff(part_group(part), (dt * tl * 0.04) as f32);
                self.dmg.vibe += (tl * 0.03) as f32;
                if tl > 3.0 {
                    self.sparks.push((p, tang * 0.5));
                }
                if tl > 1.5 && building.is_none() && sv == DVec3::ZERO {
                    self.dirt.push((p, tang * 0.45 + upw * (2.0 + tl * 0.12), 2.5));
                }
            }
            self.pts = pts;
            let mut touching = [false; MAX_GEAR];
            for (gi, leg) in spec.gear.iter().enumerate() {
                // Only gear that is down and locked carries the aircraft.
                if self.gear < 0.999 {
                    break;
                }
                if !self.dmg.gear[gi] {
                    continue;
                }
                let r = leg.at;
                let p = self.pos + self.rot * r;
                if let Some((pen, water, building, wall, sv)) = hit(p, world) {
                    if pen <= 0.0 {
                        continue;
                    }
                    let pv = self.vel + self.rot * self.omega.cross(r) - sv;
                    let vn = pv.dot(upw);
                    // Wheels do nothing in water or against a wall, and a leg driven
                    // in far too hard just goes: the airframe takes the rest.
                    if water || (wall && pen > 1.0) || pen > leg.travel * 4.0 + 0.8 {
                        if !water {
                            self.lose_gear(gi, if building.is_some() { "BUILDING" } else if wall { "SHIP" } else { "HARD LANDING" });
                        }
                        continue;
                    }
                    if vn < -(leg.fold_vn + 4.5) {
                        self.lose_gear(gi, "HARD LANDING");
                        continue;
                    }
                    // Too hard a landing and the leg folds; the airframe comes down on its belly.
                    if vn < -leg.fold_vn {
                        self.lose_gear(gi, "HARD LANDING");
                        self.kick(leg.fold_kick);
                        self.crack_windshield("HARD LANDING");
                        continue;
                    }
                    self.on_ground = true;
                    touching[gi] = true;
                    self.compress[gi] = pen.min(leg.travel);
                    let nf = (pen * leg.k - vn * leg.c).clamp(0.0, leg.nf_max);
                    let mut fwd = (self.forward() - upw * self.forward().dot(upw)).normalize_or_zero();
                    if leg.steer != 0.0 {
                        // The nose wheel steers with the rudder pedals.
                        fwd = DQuat::from_axis_angle(upw, -self.steer_angle() * leg.steer) * fwd;
                    }
                    let side = fwd.cross(upw);
                    let (vl, vs) = (pv.dot(fwd), pv.dot(side));
                    let f_side = (-vs * leg.side_k).clamp(-leg.side_mu * nf, leg.side_mu * nf);
                    let braking = c.brake && leg.brakes;
                    let mu = if braking { leg.brake_mu } else { leg.roll_mu };
                    let f_long = (-vl * if braking { leg.brake_k } else { leg.roll_k }).clamp(-mu * nf, mu * nf);
                    let f = upw * nf + side * f_side + fwd * f_long;
                    force_w += f;
                    torque_b += r.cross(self.rot.inverse() * f);
                    self.wheel_spin[gi] = vl / leg.wheel_r;
                    if pv.length() > 3.0 && building.is_none() && sv == DVec3::ZERO {
                        self.dirt.push((p, (pv - upw * vn) * 0.3 + upw * 1.5, 1.0));
                    }
                }
            }
            for gi in 0..spec.gear.len() {
                if !touching[gi] {
                    self.compress[gi] *= (1.0 - dt * 12.0).max(0.0);
                }
            }
        } else {
            self.compress = [0.0; MAX_GEAR];
        }
        // Wheels spin down in the air; brakes stop the mains at once.
        for (gi, leg) in spec.gear.iter().enumerate() {
            if !self.on_ground {
                self.wheel_spin[gi] *= (1.0 - dt * if c.brake && leg.brakes && leg.steer == 0.0 { 8.0 } else { 0.25 }).max(0.0);
            }
            self.wheel_rot[gi] = (self.wheel_rot[gi] + self.wheel_spin[gi] * dt) % std::f64::consts::TAU;
        }

        self.on_hull = skidding;
        force_w += self.push;
        let acc = force_w / spec.mass;
        self.swing_folds(dt, acc, upw, vb, rho);
        self.g_load = (acc + upw * 9.81).dot(self.up()) / 9.81;
        // The pilot's head on its neck: a stiff, well-damped spring shoved about
        // by every change in the load, about 3 cm per g.
        self.sf = self.rot.inverse() * (acc + upw * 9.81);
        let fb = self.sf - DVec3::Y * 9.81;
        let (w0, zeta) = (11.0, 0.55);
        self.head_v += (-self.head * w0 * w0 - self.head_v * 2.0 * zeta * w0 - fb * 0.35) * dt;
        if self.on_ground {
            // Rolling over the pavement's joints and patches: a few millimetres
            // of shake by taxi speed, more on the takeoff roll.
            let s = smooth(0.5, 35.0, self.vel.length()) * 0.375 * dt.sqrt();
            let mut k = || self.rng.range(-1.0, 1.0) as f64;
            self.head_v += DVec3::new(k() * 0.35, k(), k() * 0.2) * s;
        }
        self.head = (self.head + self.head_v * dt).clamp_length_max(0.09);
        // Rough air rattles the cabin.
        self.dmg.vibe += ((self.air.sigma_w - 0.6).max(0.0) * 0.07 * smooth(10.0, 40.0, v)) as f32;
        self.vel += acc * dt;
        self.omega += (wdot + torque_b / spec.inertia) * dt;
        self.omega = self.omega.clamp_length_max(6.0);
        self.pos += self.vel * dt;
        self.rot = (self.rot * DQuat::from_scaled_axis(self.omega * dt)).normalize();

        // Airframe limits: pull too hard and a wing goes; too fast and the
        // control surfaces flutter themselves off; flaps out too fast tear away.
        if !self.on_ground && !skidding {
            let g = self.g_load.abs() as f32;
            // Past the limit load the spar fatigues; at the ultimate load it snaps outright.
            self.dmg.fatigue = (self.dmg.fatigue + ((g - spec.g_limit).max(0.0) * 1.5 - 0.3) * dt as f32).max(-1.5);
            if self.dmg.fatigue < 0.0 {
                self.dmg.fatigue = (self.dmg.fatigue + dt as f32).min(0.0);
            }
            if (g > spec.g_break && self.dmg.fatigue >= 0.0) || self.dmg.fatigue > 2.2 {
                let s = (self.rng.f() * 2.0) as usize;
                self.break_wing(s, self.sta(0.64), "OVER G");
                // The load comes off the rest of the structure for a moment.
                self.dmg.fatigue = -1.5;
            }
        }
        let ve = v * eas_k;
        let excess = self.excess(ve, mach);
        let over = (excess - (spec.flutter_v - spec.vne) / spec.speed_k).max(0.0) as f32;
        self.dmg.flutter = (self.dmg.flutter + (over * 0.035 - 0.25) * dt as f32).max(0.0);
        self.dmg.vibe += smooth(5.0, 30.0, excess) as f32 * 0.6;
        if self.dmg.flutter > 1.0 {
            self.flutter_loss();
            self.dmg.flutter = 0.35;
        }
        let vfe = self.vfe();
        if self.flaps > 0.25 * spec.flap_frac(1) && ve > vfe {
            self.dmg.flap_stress += (dt * (ve - vfe) / spec.speed_k * 0.06 * self.flaps.max(0.3)) as f32;
            if self.dmg.flap_stress > 1.0 {
                let s = (self.rng.f() * 2.0) as usize;
                let s = if self.dmg.flap[s] > 0.0 { s } else { 1 - s };
                self.lose_flap(s, "OVERSPEED");
                self.dmg.flap_stress = 0.3;
            }
        } else {
            self.dmg.flap_stress = (self.dmg.flap_stress - dt as f32 * 0.3).max(0.0);
        }
        None
    }

    /// A buckled wing swings on its torn skin: its weight and the aircraft's
    /// own acceleration, the air on it as on a flat plate, the ground under
    /// it, and what stiffness the crumpled skin has left. Load it past what
    /// the skin holds (as the slipstream does at once in flight) and it goes.
    fn swing_folds(&mut self, dt: f64, acc: DVec3, upw: DVec3, vb: DVec3, rho: f64) {
        let spec = self.spec;
        let wk = self.fold_k() as f64;
        let kg_m = self.wing_kg_m() as f64;
        for s in 0..2 {
            let mut f = self.dmg.fold[s];
            if !f.on {
                continue;
            }
            let len = (f.tip - f.root).max(0.3) as f64;
            let m = kg_m * len;
            let inertia = m * len * len / 3.0 + 2.0 * wk;
            let mat = f.mat();
            let ax = f.axis.as_dvec3();
            let cm = mat.transform_point3(damage::wing_pt(spec, s, (f.root + f.tip) * 0.5, 0.4, 0.0)).as_dvec3();
            let arm = cm - f.pivot.as_dvec3();
            // Weight, and the shove of the aircraft's own acceleration, as the part feels them.
            let felt = self.rot.inverse() * (acc + upw * 9.81);
            let t_g = arm.cross(-felt * m).dot(ax);
            // The air on it: a plate across its swung upper skin, the air meeting
            // it at the aircraft's speed plus its own swinging.
            let np = mat.transform_vector3(Vec3::Y).as_dvec3();
            let area = (self.dmg.wing_lost(f.root) - self.dmg.wing_lost(f.tip)) as f64;
            let rel = -(vb + self.omega.cross(cm)) - ax.cross(arm) * f.rate as f64;
            let vr = rel.length();
            let mut t_a = 0.0;
            if vr > 0.5 {
                let a_n = (rel.dot(np) / vr).clamp(-1.0, 1.0).asin();
                let cn = (spec.cl_a * a_n + spec.cl0 * (1.0 - a_n.abs() * 2.0).max(0.0)).clamp(-1.2, 1.2);
                let fa = np * (0.5 * rho * vr * vr * area * cn);
                t_a = arm.cross(fa).dot(ax);
                // The aircraft feels it through the hinge.
                self.vel += self.rot * fa * (dt / spec.mass);
                self.omega += cm.cross(fa) / spec.inertia * dt;
            }
            // The crumpled skin still springs back a little, and damps the swing.
            let w = f.rate as f64;
            let t_s = -350.0 * wk * f.ang as f64 - 220.0 * wk * w - 0.5 * rho * 1.2 * 1.4 * w * w.abs() * len.powi(4) / 4.0 * (kg_m / 24.0).sqrt();
            let t_c = self.fold_tq[s];
            f.rate += ((t_g + t_a + t_s + t_c) / inertia * dt) as f32;
            f.ang += f.rate * dt as f32;
            // Folded right over (or down against the strut), it stops hard.
            let (lo, hi) = (-1.75, 2.7);
            if f.ang > hi || f.ang < lo {
                f.ang = f.ang.clamp(lo, hi);
                f.stress += (f.rate.abs() * 0.04).min(0.3);
                f.rate *= -0.2;
            }
            // How hard the skin is being worked: steady load past what it holds,
            // and every big swing.
            let load = (t_g + t_a + t_c).abs() as f32;
            // Crumpled skin in a fast airflow flutters itself apart in moments.
            let flog = 0.5 * ((0.5 * rho * vb.length_squared()) / 600.0).powf(1.5);
            // (Tearing takes a moment, however hard it is pulled: at least a fifth of a second.)
            let over = (((load - f.hold).max(0.0) / f.hold) * 4.0).min(5.0);
            f.stress += (over + (f.rate.abs() - 4.0).max(0.0) * 0.1 + 0.01 + flog as f32) * dt as f32;
            self.dmg.vibe += (f.rate.abs() * 0.05).min(1.0);
            self.dmg.fold[s] = f;
            if f.stress >= 1.0 {
                self.release_fold(s, if self.on_hull || self.on_ground { "GROUND" } else { "AIRFLOW" });
            }
        }
    }

    /// Trees: refresh the list of nearby ones a few times a second, test every
    /// step. Foliage scrapes and tears things off; a trunk through the nose
    /// or belly is the end. A tree only goes over when the blow carries more
    /// energy than its trunk can take: a light aircraft's wing shears off
    /// against a mature trunk and leaves it standing, an airliner mows them down.
    fn trees(&mut self, world: &World) -> Option<Impact> {
        let spec = self.spec;
        if self.scan == 0 {
            self.near_trees.clear();
            world.trees_near(self.pos, 30.0 + spec.b * 0.5 + self.vel.length() * 0.2, &mut self.near_trees);
            let (pos, reach) = (self.pos, 60.0 + spec.b + self.vel.length());
            let near = &self.near_trees;
            self.struck.retain(|s| near.iter().any(|t| t.tile == s.0 && t.idx == s.1 && (t.pos - pos).length() < reach));
        }
        self.scan = (self.scan + 1) % 12;
        let v = self.vel.length();
        let mut k = 0;
        while k < self.near_trees.len() {
            let t = &self.near_trees[k];
            let tu = t.pos.normalize();
            let (cc, ch) = ((t.crown_base + t.height) * 0.5, ((t.height - t.crown_base) * 0.5).max(0.5));
            // 0 = miss, 1 = through the foliage, 2 = into the trunk or the heart of the crown.
            let mut worst = 0;
            let mut at = 5;
            let mut core: Option<DVec3> = None;
            for (hi, r) in spec.hard.into_iter().enumerate() {
                let d = self.pos + self.rot * r - t.pos;
                let hgt = d.dot(tu);
                if hgt < -1.0 || hgt > t.height + 1.0 {
                    continue;
                }
                let flat = d - tu * hgt;
                let rad = flat.length();
                let trunk = rad < t.trunk_r + 0.35 && hgt < cc;
                if trunk && matches!(hi, 0 | 5 | 6) {
                    core = Some(flat / rad.max(1e-3));
                }
                let e = ((rad / t.crown_r.max(0.5)).powi(2) + ((hgt - cc) / ch).powi(2)).sqrt();
                let level = if trunk || e < 0.45 { 2 } else if e < 1.0 { 1 } else { 0 };
                if level > worst {
                    worst = level;
                    at = hi;
                }
            }
            let prev = self.struck.iter().position(|s| s.0 == t.tile && s.1 == t.idx);
            let before = prev.map(|i| self.struck[i].2).unwrap_or(0);
            if worst <= before {
                // Already went through this one: a trunk still standing stays solid
                // against the fuselage instead of letting it creep through.
                if before == 2 {
                    if let Some(n) = core {
                        let vn = self.vel.dot(n);
                        if vn < 0.0 {
                            self.vel -= n * vn;
                        }
                    }
                }
                k += 1;
                continue;
            }
            // Energy the trunk soaks up before it snaps, against the share of
            // the aircraft's that goes into it before the part that hit gives.
            let ke = 0.5 * spec.mass * v * v;
            let snap_e = 3.0e5 * (t.trunk_r / 0.25).powi(3) * (t.height / 15.0).clamp(0.6, 1.4);
            let share = match (worst, at) {
                (2, 1 | 2) => 0.6,
                (2, 3 | 4) => 0.4,
                (2, _) => 1.0,
                _ => 0.03,
            };
            let snaps = ke * share > snap_e;
            let base_h = geo::pos_to_lla(t.pos).2;
            if snaps {
                let t = self.near_trees.swap_remove(k);
                if let Some(i) = prev {
                    self.struck.swap_remove(i);
                }
                // Breaking it costs what it took to break.
                let v2 = (v * v - 2.0 * snap_e / spec.mass).max(0.36 * v * v);
                self.vel *= (v2 / (v * v).max(1e-6)).sqrt();
                self.tree_hits.push(t);
            } else {
                let id = (t.tile, t.idx);
                match prev {
                    Some(i) => self.struck[i].2 = worst,
                    None => self.struck.push((id.0, id.1, worst)),
                }
                k += 1;
            }
            self.brushes += 1.0;
            let side = if at == 1 { 0 } else { 1 };
            let sgn = if side == 0 { -1.0 } else { 1.0 };
            let r = self.rng.f();
            // A tree that holds: what hit it is stopped, and the aircraft pivots on it.
            let hold = if snaps { 1.0 } else { 1.7 };
            if worst == 2 {
                match at {
                    1 | 2 => {
                        // A trunk through the outer wing shears it off.
                        self.break_wing(side, self.sta(if v > 45.0 { 0.64 } else { 0.78 }), "TREE");
                        self.vel *= if snaps { 0.9 } else { 0.72 };
                        self.kick(DVec3::new(0.2, -sgn * 0.7, -sgn * 1.5) * hold);
                    }
                    3 | 4 => {
                        self.lose_tail_top(0.4, 0.45, "TREE");
                        self.lose_elevator((r * 2.0) as usize, true, "TREE");
                        self.vel *= if snaps { 0.92 } else { 0.8 };
                        self.kick(DVec3::new(-0.8, 0.6, 0.4) * hold);
                    }
                    // Rolling slowly into a trunk crumples the nose and stops it dead.
                    _ if v < 18.0 => {
                        self.prop_strike("TREE");
                        self.crack_windshield("TREE");
                        self.dmg.hull = (self.dmg.hull + 0.2).min(1.0);
                        self.vel *= if snaps { 0.4 } else { 0.1 };
                        self.omega *= 0.3;
                    }
                    _ => return Some(Impact { pos: self.pos, vel: self.vel, water: false, ground_h: base_h + 0.2, building: None }),
                }
                continue;
            }
            // Clipped the canopy: it costs speed, throws the aircraft about, and
            // whatever went through the branches comes out worse for it.
            if !snaps {
                self.vel *= 1.0 - 0.08 * (1100.0 / spec.mass).cbrt();
            }
            let kick = (self.scan as f64 * 2.4).sin();
            self.kick(DVec3::new(0.25 * kick, 0.5 * kick, 0.9 * (self.scan as f64 * 1.7).cos()) * hold.sqrt());
            let hard = (v / 45.0) as f32;
            match at {
                1 | 2 => {
                    self.scuff(Self::wing_group(side), 0.35);
                    if r < 0.12 * hard {
                        self.break_wing(side, self.sta(0.91), "TREE");
                    } else if r < 0.4 * hard {
                        self.lose_aileron(side, r > 0.2 * hard, "TREE");
                    }
                }
                0 | 6 => {
                    self.scuff(G_ENGINE, 0.4);
                    if r < 0.3 * hard {
                        self.prop_strike("TREE");
                    } else if r < 0.6 * hard {
                        self.engine_hit(0.15, "TREE");
                    }
                    if self.rng.f() < 0.45 * hard {
                        self.crack_windshield("TREE");
                    }
                }
                3 | 4 => {
                    self.scuff(G_TAIL, 0.35);
                    if r < 0.3 * hard {
                        self.lose_tail_top(0.5, 1.0, "TREE");
                    } else if r < 0.5 * hard {
                        self.lose_elevator((self.scan % 2) as usize, false, "TREE");
                    }
                }
                _ => {
                    self.scuff(damage::hard_group(at), 0.3);
                    if self.gear > 0.5 && r < 0.3 * hard {
                        self.lose_gear(1 + (self.scan % 2) as usize, "TREE");
                    }
                }
            }
        }
        None
    }
}

/// Air density (kg/m³) at height `h` (m) in the standard atmosphere: the
/// troposphere's lapse rate up to 11 km, isothermal above.
pub fn density(h: f64) -> f64 {
    let h = h.max(-500.0);
    let t = 1.0 - 2.25577e-5 * h.min(11_000.0);
    RHO0 * t.powf(4.2559) * (-(h - 11_000.0).max(0.0) / 6341.6).exp()
}

/// Static pressure as a fraction of sea level's, in the standard atmosphere.
pub fn pressure_ratio(h: f64) -> f64 {
    let h = h.max(-500.0);
    let t = 1.0 - 2.25577e-5 * h.min(11_000.0);
    t.powf(5.2559) * (-(h - 11_000.0).max(0.0) / 6341.6).exp()
}

/// Speed of sound (m/s) at height `h` in the standard atmosphere.
pub fn sound_speed(h: f64) -> f64 {
    let t = 288.15 - 0.0065 * h.clamp(-500.0, 11_000.0);
    (1.4 * 287.05 * t).sqrt()
}

fn smooth(a: f64, b: f64, x: f64) -> f64 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn smooth01(a: f32, b: f32, x: f32) -> f32 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aircraft::{b747, c172};
    use std::sync::Arc;

    const C172: &Spec = &c172::SPEC;
    const B747: &Spec = &b747::SPEC;

    fn empty_world() -> World {
        let dir = std::env::temp_dir().join("fablegame-test-cache");
        let src = Arc::new(crate::sources::Sources::new(crate::fetch::Fetcher::new(dir, true), String::new()));
        World::new(crate::sources::Loader::new(src, 0), 14)
    }

    /// Fly for `secs` with a pitch law; returns (airspeed, vertical speed) averaged over the last 20 s.
    fn fly(p: &mut Plane, w: &World, secs: f64, throttle: f64, mut pitch: impl FnMut(&Plane) -> f64) -> (f64, f64) {
        let dt = 1.0 / 180.0;
        p.throttle = throttle;
        let (mut sv, mut svs, mut n) = (0.0, 0.0, 0.0);
        let steps = (secs / dt) as usize;
        for i in 0..steps {
            let roll_err = -p.right().dot(p.pos.normalize());
            let c = Controls { pitch: pitch(p).clamp(-1.0, 1.0), roll: (-roll_err * 3.0 + p.omega.z * 0.5).clamp(-1.0, 1.0), ..Default::default() };
            assert!(p.step(dt, &c, w).is_none());
            if (steps - i) as f64 * dt < 20.0 {
                sv += p.airspeed().length();
                svs += p.vel.dot(p.pos.normalize());
                n += 1.0;
            }
        }
        (sv / n, svs / n)
    }

    const KT: f64 = 1.94384;
    fn rpm(p: &Plane) -> f64 {
        p.prop_rate * 60.0 / std::f64::consts::TAU
    }
    fn piston() -> &'static crate::aircraft::Piston {
        match &C172.engine {
            Engine::Piston(pe) => pe,
            _ => unreachable!(),
        }
    }
    /// An altitude hold for the test pilot (gentle, or it porpoises).
    fn hold_alt(alt0: f64) -> impl FnMut(&Plane) -> f64 {
        move |p| {
            let (_, _, h) = geo::pos_to_lla(p.pos);
            let vs = p.vel.dot(p.pos.normalize());
            (alt0 - h) * 0.002 - vs * 0.02
        }
    }

    #[test]
    fn performance_is_a_hornet() {
        let hornet: &Spec = &crate::aircraft::fa18::SPEC;
        let w = empty_world();
        let dt = 1.0 / 180.0;
        let level = |boost: bool| {
            let mut p = Plane::spawn(hornet, 0.6, -2.1, 3000.0, 0.0, 200.0);
            p.throttle = 1.0;
            let mut hold = hold_alt(3000.0);
            let mut gmax: f64 = 0.0;
            for i in 0..(180.0 / dt) as usize {
                // the stick does a lot at Mach 1: the test pilot goes gently
                let gentle = (120.0 / p.airspeed().length()).powi(2).min(1.0);
                let roll_err = -p.right().dot(p.pos.normalize());
                let c = Controls { pitch: (hold(&p) * gentle).clamp(-1.0, 1.0), roll: (-roll_err * 3.0 + p.omega.z * 0.5).clamp(-1.0, 1.0), throttle: 1.0, boost, ..Default::default() };
                assert!(p.step(dt, &c, &w).is_none());
                if i as f64 * dt > 60.0 {
                    gmax = gmax.max((p.g_load - 1.0).abs());
                }
            }
            println!("  boost {boost}: Mach {:.3}, worst load swing {gmax:.2} g", p.mach());
            assert!(gmax < 0.5, "porpoising: {gmax}");
            p.mach()
        };
        let (dry, wet) = (level(false), level(true));
        println!("level at 3000 m: Mach {dry:.2} dry, {wet:.2} in afterburner");
        assert!(dry > 0.85 && dry < 1.02 && wet > 1.1 && wet < 1.45, "{dry} {wet}");

        // A full pull at 450 kt: the fly-by-wire holds it near 7.5 g.
        let mut p = Plane::spawn(hornet, 0.6, -2.1, 3000.0, 0.0, 232.0);
        let mut gmax: f64 = 0.0;
        for _ in 0..(3.0 / dt) as usize {
            let c = Controls { pitch: 1.0, throttle: 1.0, ..Default::default() };
            assert!(p.step(dt, &c, &w).is_none());
            gmax = gmax.max(p.g_load);
        }
        println!("full pull at 450 kt: {gmax:.1} g");
        assert!(gmax > 6.0 && gmax < 8.5, "{gmax}");

        // On approach: gear and full flaps at 135 kt, the angle of attack it flies at.
        let mut p = Plane::spawn(hornet, 0.6, -2.1, 600.0, 0.0, 70.0);
        p.gear_down = true;
        p.gear = 1.0;
        p.flap_set = 2;
        p.flaps = 1.0;
        let mut hold = hold_alt(600.0);
        for _ in 0..(60.0 / dt) as usize {
            // auto-throttle on 135 kt
            let v = p.airspeed().length();
            let thr = (0.7 + (70.0 - v) * 0.08).clamp(0.0, 1.0);
            let roll_err = -p.right().dot(p.pos.normalize());
            let c = Controls { pitch: hold(&p).clamp(-1.0, 1.0), roll: (-roll_err * 3.0 + p.omega.z * 0.5).clamp(-1.0, 1.0), throttle: thr, ..Default::default() };
            p.throttle = thr;
            assert!(p.step(dt, &c, &w).is_none());
        }
        println!("approach: {:.0} kt, alpha {:.1} deg, throttle {:.0}%", p.airspeed().length() * KT, p.alpha.to_degrees(), p.throttle * 100.0);
        assert!(p.alpha.to_degrees() > 5.0 && p.alpha.to_degrees() < 14.0 && !p.stalled);
    }

    #[test]
    fn performance_is_a_cessna_172() {
        let w = empty_world();
        // Flat out and level near sea level: about 125 kt with the gear up, a
        // few knots less with it down (a fixed-gear 172 does 126 kt).
        let mut p = Plane::spawn(C172, 0.6, -2.1, 300.0, 0.0, 55.0);
        let (v, _) = fly(&mut p, &w, 240.0, 1.0, hold_alt(300.0));
        println!("max level speed {:.1} m/s = {:.0} kt at {:.0} rpm", v, v * KT, rpm(&p));
        assert!(v * KT > 118.0 && v * KT < 132.0, "{}", v * KT);
        assert!(rpm(&p) > 2550.0 && rpm(&p) < 2800.0, "{}", rpm(&p));
        let mut p = Plane::spawn(C172, 0.6, -2.1, 300.0, 0.0, 55.0);
        p.configure(0, true, 0.0);
        let (vg, _) = fly(&mut p, &w, 240.0, 1.0, hold_alt(300.0));
        println!("gear down: {:.0} kt", vg * KT);
        assert!(vg < v - 1.0 && vg * KT > 112.0, "{}", vg * KT);

        // 75% power at 8000 ft: about 120 kt true.
        let mut p = Plane::spawn(C172, 0.6, -2.1, 2438.0, 0.0, 58.0);
        let mut thr = 0.8;
        for _ in 0..40 {
            fly(&mut p, &w, 5.0, thr, hold_alt(2438.0));
            thr = (thr + (0.75 - p.power) * 0.5).clamp(0.0, 1.0);
        }
        let (v, _) = fly(&mut p, &w, 120.0, thr, hold_alt(2438.0));
        println!("75% at 8000 ft: {:.0} kt true, {:.0} kt indicated, {:.0} rpm, power {:.2}", v * KT, p.ias() * KT, rpm(&p), p.power);
        assert!(v * KT > 110.0 && v * KT < 128.0, "{}", v * KT);

        // Full power at the best-rate speed (74 kt): 700-900 fpm.
        let mut p = Plane::spawn(C172, 0.6, -2.1, 300.0, 0.0, 40.0);
        let (v, vs) = fly(&mut p, &w, 90.0, 1.0, |p| (p.airspeed().length() - 38.0) * 0.04);
        println!("climb at {:.0} kt: {:.0} fpm, {:.0} rpm", v * KT, vs * 196.85, rpm(&p));
        assert!(vs * 196.85 > 650.0 && vs * 196.85 < 950.0, "{}", vs * 196.85);

        // Engine at idle at the best glide speed (68 kt): about 9 to 1.
        let mut p = Plane::spawn(C172, 0.6, -2.1, 1500.0, 0.0, 35.0);
        let (v, vs) = fly(&mut p, &w, 90.0, 0.0, |p| (p.airspeed().length() - 35.0) * 0.04);
        let glide = (v * v - vs * vs).sqrt() / -vs;
        println!("idle glide at {:.0} kt: {:.0} fpm, {:.1}:1, {:.0} rpm", v * KT, vs * 196.85, glide, rpm(&p));
        assert!(glide > 7.5 && glide < 11.0, "{glide}");
        assert!(rpm(&p) > 700.0 && rpm(&p) < 1600.0, "{}", rpm(&p));

        // Stall: about 50 kt clean, 42 kt with full flaps.
        let s = C172;
        let vs_at = |flap_cl: f64, a: f64| (2.0 * s.mass * 9.81 / (RHO0 * s.wing_area * (s.cl0 + flap_cl + s.cl_a * a))).sqrt() * KT;
        let (clean, full) = (vs_at(0.0, 0.27), vs_at(0.55, 0.23));
        println!("stall {clean:.0} kt clean, {full:.0} kt full flaps");
        assert!(clean > 46.0 && clean < 54.0 && full > 39.0 && full < 46.0);

        // On the ground: about 2300 rpm flat out, 600-800 at idle, static thrust a
        // quarter of the weight.
        let pe = piston();
        let ws = Plane::prop_equilibrium(pe, 1.0, 0.0, RHO0) * 60.0 / std::f64::consts::TAU;
        let wi = Plane::prop_equilibrium(pe, 0.0, 0.0, RHO0) * 60.0 / std::f64::consts::TAU;
        let t0 = Plane::spawn(C172, 0.6, -2.1, 0.0, 0.0, 0.0).thrust(1.0, 0.0, 0.0);
        println!("static {ws:.0} rpm, idle {wi:.0} rpm, thrust {t0:.0} N = {:.2} W", t0 / (s.mass * 9.81));
        assert!(ws > 2200.0 && ws < 2450.0 && wi > 550.0 && wi < 850.0, "{ws} {wi}");
        assert!(t0 / (s.mass * 9.81) > 0.18 && t0 / (s.mass * 9.81) < 0.28, "{t0}");
        // Diving at Vne at full throttle the fixed-pitch propeller over-revs.
        let wd = Plane::prop_equilibrium(pe, 1.0, s.vne, RHO0) * 60.0 / std::f64::consts::TAU;
        println!("full throttle at Vne: {wd:.0} rpm");
        assert!(wd > 2850.0, "{wd}");
    }

    #[test]
    fn approach_is_trimmed_down_the_glide() {
        let w = empty_world();
        // As the game sets up a final approach: 70 kt, flaps 20, a 3 degree glide.
        let mut p = Plane::spawn(C172, 0.6, -2.1, 400.0, 0.0, 36.0);
        let sink = 36.0 * 3f64.to_radians().sin();
        p.configure(2, true, sink);
        let thr = p.throttle;
        let (v, vs) = fly(&mut p, &w, 30.0, thr, |_| 0.0);
        println!("approach: throttle {thr:.2}, {:.0} kt, {:.0} fpm, {:.0} rpm", v * KT, vs * 196.85, rpm(&p));
        assert!((v - 36.0).abs() < 3.0, "{v}");
        assert!((vs + sink).abs() < 1.0, "{vs}");
    }

    #[test]
    fn turbulence_bumps_the_aircraft() {
        let w = empty_world();
        let mut spread = Vec::new();
        for sigma in [0.0, 1.5, 4.0] {
            let mut p = Plane::spawn(C172, 0.6, -2.1, 1000.0, 0.0, 55.0);
            p.air = Air { sigma_w: sigma, sigma_h: sigma, len_w: 300.0, len_h: 300.0, draft: 0.0 };
            let dt = 1.0 / 180.0;
            let (mut s, mut s2, mut n, mut bank) = (0.0, 0.0, 0.0, 0.0f64);
            for i in 0..(60.0 / dt) as usize {
                // A pilot holding the wings roughly level.
                let roll_err = -p.right().dot(p.pos.normalize());
                let c = Controls { roll: (-roll_err * 3.0 + p.omega.z * 0.5).clamp(-1.0, 1.0), ..Default::default() };
                p.step(dt, &c, &w);
                if i > 1800 {
                    s += p.g_load;
                    s2 += p.g_load * p.g_load;
                    n += 1.0;
                    bank = bank.max(roll_err.abs().asin().to_degrees());
                }
            }
            let sd = (s2 / n - (s / n).powi(2)).max(0.0).sqrt();
            println!("sigma {sigma}: g rms {sd:.3}, worst bank {bank:.1} deg");
            spread.push(sd);
        }
        assert!(spread[0] < 0.02 && spread[1] > 0.05 && spread[2] > spread[1] * 1.8, "{spread:?}");
    }

    fn flat_world() -> World {
        let mut w = empty_world();
        w.flat = Some(0.0);
        w
    }
    /// Run until it explodes or `secs` pass; every note it made along the way.
    fn run(p: &mut Plane, w: &World, secs: f64, c: Controls) -> (bool, Vec<String>) {
        run_pushed(p, w, secs, c, 0.0)
    }
    /// The same, with something rolling it over (rad/s²) for the first 0.8 s: a gust under one wing.
    fn run_pushed(p: &mut Plane, w: &World, secs: f64, c: Controls, push: f64) -> (bool, Vec<String>) {
        let dt = 1.0 / 180.0;
        let mut notes = Vec::new();
        for k in 0..(secs / dt) as usize {
            if (k as f64) * dt < 0.8 {
                p.omega.z += push * dt;
            }
            let imp = p.step(dt, &c, w);
            notes.extend(p.notes.drain(..));
            p.shed.clear();
            if imp.is_some() {
                return (true, notes);
            }
        }
        (false, notes)
    }
    /// Rolling down the runway at `kt`, banked over on one main wheel and
    /// still rolling `roll` rad/s (a gust under a wing, a botched crosswind landing).
    fn rolling(kt: f64, roll: f64, seed: u64) -> Plane {
        let mut p = Plane::spawn_ground(C172, 37.8, -122.46, 0.0, 0.5, 0.0);
        p.hold = false;
        p.t = 2.0;
        p.rng = Rng(seed);
        p.throttle = 0.0;
        p.charge = 0.0;
        let bank = 0.12 * roll.signum();
        // Pivot about the downhill main wheel so it stays on the ground.
        let wheel = C172.gear[if roll > 0.0 { 1 } else { 2 }].at;
        let before = p.pos + p.rot * wheel;
        p.rot = p.rot * DQuat::from_rotation_z(-bank);
        p.pos += before - (p.pos + p.rot * wheel);
        p.vel = p.forward() * (kt / KT);
        p.omega = DVec3::new(0.0, 0.0, -roll);
        p
    }

    #[test]
    fn tipping_over_at_40_kt_breaks_a_wing_without_fire() {
        let w = flat_world();
        let mut fires = 0;
        for seed in 1..=12u64 {
            let mut p = rolling(40.0, if seed % 2 == 0 { 2.2 } else { -2.2 }, seed * 7919);
            let (boom, notes) = run_pushed(&mut p, &w, 12.0, Controls::default(), if seed % 2 == 0 { -9.0 } else { 9.0 });
            println!("seed {seed}: boom {boom} {}, wings {:?}, folds {:?}, speed {:.1}, wrecked {}: {notes:?}", p.cause, p.dmg.wing, p.dmg.fold.map(|f| f.on), p.vel.length(), p.wrecked());
            fires += boom as usize;
            assert!(p.dmg.wing.iter().any(|&b| b < C172.span) || p.dmg.fold.iter().any(|f| f.on) || !notes.is_empty(), "something breaks");
        }
        assert_eq!(fires, 0, "a slow tip-over never burns");
    }

    #[test]
    fn a_cartwheel_at_70_kt_tears_things_off() {
        let w = flat_world();
        let mut fires = 0;
        for seed in 1..=8u64 {
            let mut p = rolling(70.0, 3.5, seed * 104_729);
            p.vel += p.pos.normalize() * -3.0;
            let (boom, notes) = run(&mut p, &w, 14.0, Controls { roll: 1.0, ..Default::default() });
            println!("seed {seed}: boom {boom} {}, wings {:?}, tail {}, hull {:.2}: {notes:?}", p.cause, p.dmg.wing, p.dmg.tail.is_some(), p.dmg.hull);
            fires += boom as usize;
            assert!(boom || !notes.is_empty());
        }
        assert!(fires < 8, "not every cartwheel burns");
    }

    #[test]
    fn belly_landing_slides_to_a_stop() {
        let w = flat_world();
        let mut p = Plane::spawn(C172, 37.8, -122.46, 2.0, 0.5, 30.0);
        p.t = 2.0;
        p.throttle = 0.0;
        p.charge = 0.0;
        p.vel += p.pos.normalize() * -2.0;
        let (boom, notes) = run(&mut p, &w, 20.0, Controls::default());
        println!("belly: boom {boom}, speed {:.1}, prop {}, {notes:?}", p.vel.length(), p.dmg.prop);
        assert!(!boom);
        assert!(p.vel.length() < 2.0);
        assert!(p.dmg.wing == [C172.span; 2]);
    }

    #[test]
    fn diving_into_the_ground_explodes() {
        let w = flat_world();
        let mut p = Plane::spawn(C172, 37.8, -122.46, 40.0, 0.5, 60.0);
        p.t = 2.0;
        p.rot = p.rot * DQuat::from_rotation_x(-0.7);
        p.vel = p.forward() * 60.0;
        let (boom, _) = run(&mut p, &w, 5.0, Controls { pitch: -1.0, ..Default::default() });
        assert!(boom);
    }

    #[test]
    fn a_buckled_wing_rips_off_in_the_slipstream() {
        let w = empty_world();
        let mut p = Plane::spawn(C172, 37.8, -122.46, 1500.0, 0.5, 50.0);
        p.dmg.apply_spec("rfold=3.0");
        let dt = 1.0 / 180.0;
        let mut gone = None;
        for k in 0..360 {
            p.step(dt, &Controls::default(), &w);
            if !p.dmg.fold[1].on {
                gone = Some(k as f64 * dt);
                break;
            }
        }
        println!("fold let go after {gone:?} s");
        assert!(gone.is_some_and(|t| t < 1.5));
        assert!(!p.shed.is_empty());
    }

    #[test]
    fn breaks_stay_inside_each_other() {
        let mut rng = Rng(99);
        for spec in [C172, B747] {
            let k = spec.span / 5.5;
            for s in 0..2 {
                let mut prev = Tear::NONE;
                let mut st = spec.span;
                while st > 1.0 * k {
                    st -= rng.range(0.4, 1.4) * k;
                    let t = damage::wing_break(spec, s, st.max(0.5 * k), rng.range(-0.8, 0.8) * k, rng.range(-0.3, 0.3), 0.2, &prev);
                    // The new break's mid-chord point is kept by the old stump, the old tip is not kept by the new.
                    let mid = damage::wing_pt(spec, s, st.max(0.5 * k), 0.45, 0.0);
                    assert!(prev.keeps(mid));
                    assert!(!t.keeps(damage::wing_pt(spec, s, st.max(0.5 * k) + 0.3 * k, 0.45, 0.0)));
                    prev = t;
                }
            }
        }
    }

    // ---------------------------------------------------------------- the 747

    #[test]
    fn performance_is_a_747() {
        let w = empty_world();
        let s = B747;
        let fe = match &s.engine {
            Engine::Fan(f) => f,
            _ => unreachable!(),
        };
        // Cruise at FL350, Mach 0.85, about 90% N1 or a little less.
        let h = 10_668.0;
        let v = 0.85 * sound_speed(h);
        let mut p = Plane::spawn(s, 0.6, -2.1, h, 0.0, v);
        println!("cruise trim: alpha {:.1} deg, throttle {:.2}, N1 {:.0}%", p.trim.to_degrees(), p.throttle, p.n1 * 100.0);
        assert!(p.trim.to_degrees() > 0.5 && p.trim.to_degrees() < 4.5, "{}", p.trim.to_degrees());
        assert!(p.n1 > 0.78 && p.n1 < 0.97, "{}", p.n1);
        let thr = p.throttle;
        let (va, vs) = fly(&mut p, &w, 120.0, thr, hold_alt(h));
        println!("cruise: M {:.3}, {:.0} kt true, {:.0} kt indicated, vs {vs:.1} m/s", va / sound_speed(h), va * KT, p.ias() * KT);
        assert!((va / sound_speed(h) - 0.85).abs() < 0.03);
        // Flat out at FL350 it is held back by the shock waves before Mach 0.95.
        let mut p = Plane::spawn(s, 0.6, -2.1, h, 0.0, v);
        let (vm, _) = fly(&mut p, &w, 300.0, 1.0, hold_alt(h));
        println!("flat out at FL350: M {:.3}", vm / sound_speed(h));
        assert!(vm / sound_speed(h) > 0.87 && vm / sound_speed(h) < 0.96);

        // Stall at 290 t: about 155 kt clean, about 125 kt with flaps 30.
        let vs_at = |flap_cl: f64, a: f64| (2.0 * s.mass * 9.81 / (RHO0 * s.wing_area * (s.cl0 + flap_cl + s.cl_a * a))).sqrt() * KT;
        let clean = vs_at(0.0, s.stall_alpha);
        let full = vs_at(s.flap_cl, s.stall_alpha - s.stall_flap + s.stall_slat);
        println!("stall {clean:.0} kt clean, {full:.0} kt flaps 30 (Vref ~{:.0})", full * 1.23);
        assert!(clean > 145.0 && clean < 170.0 && full > 115.0 && full < 135.0);

        // Static thrust a quarter of the weight; the fans take several seconds to spool up.
        let t0 = Plane::fan_thrust(fe, 1.0, 0.0, 0.0);
        println!("static thrust {:.0} kN = {:.2} W", t0 / 1000.0, t0 / (s.mass * 9.81));
        assert!(t0 / (s.mass * 9.81) > 0.3 && t0 / (s.mass * 9.81) < 0.4);
    }

    #[test]
    fn a_747_takes_off_and_climbs() {
        let w = flat_world();
        let s = B747;
        let mut p = Plane::spawn_ground(s, 37.6, -122.37, 0.0, 0.5, 0.0);
        p.hold = false;
        p.t = 2.0;
        p.flap_set = 4; // flaps 20
        p.flaps = s.flap_frac(4);
        p.slats = 1.0;
        let dt = 1.0 / 120.0;
        let (mut lifted, mut t) = (None, 0.0);
        let start = p.pos;
        let mut v_rot = 0.0;
        while t < 90.0 {
            let v = p.vel.length();
            // Rotate at 155 kt to about 12 degrees nose up, then hold the climb.
            let pitch_up = p.up().dot(p.pos.normalize()).acos();
            let pitch_deg = (p.forward().dot(p.pos.normalize())).asin().to_degrees();
            let _ = pitch_up;
            let pitch = if v * KT > 155.0 { ((12.0 - pitch_deg) * 0.12).clamp(-1.0, 1.0) } else { 0.0 };
            let c = Controls { throttle: 1.0, pitch, ..Default::default() };
            assert!(p.step(dt, &c, &w).is_none(), "{:?}", p.notes);
            if lifted.is_none() && !p.on_ground && t > 5.0 {
                lifted = Some(((p.pos - start).length(), v));
            }
            if v * KT > 155.0 && v_rot == 0.0 {
                v_rot = t;
            }
            t += dt;
        }
        let (dist, vlo) = lifted.expect("it never left the ground");
        let (_, _, h) = geo::pos_to_lla(p.pos);
        println!("rotate after {v_rot:.0} s, lift-off after {dist:.0} m at {:.0} kt; at 90 s: {h:.0} m, {:.0} kt, N1 {:.0}%", vlo * KT, p.vel.length() * KT, p.n1 * 100.0);
        assert!(dist > 1200.0 && dist < 3200.0, "{dist}");
        assert!(h > 300.0, "{h}");
        assert!(p.notes.is_empty() && p.dmg.gear.iter().all(|g| *g));
    }

    #[test]
    fn a_747_settles_on_its_gear() {
        let w = flat_world();
        for (spec, wind) in [(C172, 0.0), (B747, 0.0), (C172, 6.0), (B747, 6.0)] {
            let mut p = Plane::spawn_ground(spec, 37.6, -122.37, 0.0, 0.5, 0.0);
            // A tailwind, parked: no airflow worth the name over the wing.
            p.wind = p.forward() * wind;
            p.hold = false;
            p.t = 2.0;
            let dt = 1.0 / 180.0;
            let mut worst_vn: f64 = 0.0;
            for _ in 0..(10.0 / dt) as usize {
                assert!(p.step(dt, &Controls::default(), &w).is_none());
                worst_vn = worst_vn.min(p.vel.dot(p.pos.normalize()));
            }
            let pitch = p.forward().dot(p.pos.normalize()).asin().to_degrees();
            println!("{}: notes {:?}, compress {:?}, pitch {pitch:.2} deg, worst sink {worst_vn:.2} m/s, speed {:.2}", spec.name, p.notes, &p.compress[..spec.gear.len()], p.vel.length());
            assert!(p.dmg.gear.iter().all(|g| *g), "{:?}", p.notes);
            assert!(pitch.abs() < 2.0);
        }
    }

    #[test]
    fn a_747_lands_on_its_gear() {
        let w = flat_world();
        let s = B747;
        // Over the threshold at 150 kt, flaps 30, sinking 3 m/s.
        let mut p = Plane::spawn(s, 37.6, -122.37, 12.0, 0.5, 150.0 / KT);
        p.t = 2.0;
        p.configure(6, true, 3.0);
        let dt = 1.0 / 120.0;
        let mut t = 0.0;
        let mut touched = false;
        while t < 60.0 {
            let c = Controls { throttle: -1.0, brake: touched, reverse: touched, pitch: if touched { -0.1 } else { 0.05 }, ..Default::default() };
            assert!(p.step(dt, &c, &w).is_none(), "{:?}", p.notes);
            touched |= p.on_ground;
            t += dt;
        }
        println!("after 60 s: {:.1} m/s, notes {:?}, spoilers {:.2}, reversers {:.2}", p.vel.length(), p.notes, p.spoilers, p.reverser);
        assert!(touched);
        assert!(p.vel.length() < 3.0, "{}", p.vel.length());
        assert!(p.dmg.gear.iter().all(|g| *g));
    }
}
