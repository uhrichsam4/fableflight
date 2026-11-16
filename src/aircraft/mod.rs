//! The aircraft you can fly. Each type is one module with its model (built
//! with the toolkit in `airframe.rs`) and a `Spec`: everything the flight
//! model, the damage model, the landing gear, the engines, the camera and the
//! sound need to know about it. Adding a type is a new module and an entry in
//! `ALL`; nothing else in the game is specific to one aircraft except its
//! paint and surface detail, which `plane.wgsl` picks by `Spec::id`.
//!
//! Body frame (as in airframe.rs): +X right, +Y up, -Z forward, the origin at
//! the centre of gravity.
pub mod b747;
pub mod b747big;
pub mod c172;
pub mod fa18;

use crate::airframe::{Anim, Model, Sect};
use glam::{DVec3, Vec3};
use std::sync::LazyLock;

/// Every type, in the order the menu lists them.
pub static ALL: LazyLock<[&'static Spec; 5]> = LazyLock::new(|| [&c172::SPEC, &b747::SPEC, &*b747big::X5, &*b747big::X20, &fa18::SPEC]);

/// A type by its short name (`c172`, `747`, ...), as typed on the command line.
pub fn by_key(key: &str) -> Option<&'static Spec> {
    let k = key.to_ascii_lowercase();
    ALL.iter().copied().find(|s| s.keys.iter().any(|x| *x == k))
}

/// The most landing gear legs any type has.
pub const MAX_GEAR: usize = 6;

/// What a contact point on the airframe belongs to (see `Plane::contacts`).
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Part {
    /// spinner and cowling, or a jet's nose
    Nose,
    /// the propeller's tips
    Prop,
    /// the bottom of an engine pod hanging under that wing (0 left)
    Pod(usize),
    Belly,
    /// the cabin's sides
    Side,
    /// over the cabin and behind the wing
    Roof,
    /// the tail cone
    Tail,
    Fin,
    Stab(usize),
    /// a wing (side) at a span station
    Wing(usize, f32),
    /// a wing that is hanging from its stump
    Fold(usize),
}

/// One landing gear leg: where its tyres meet the ground fully extended
/// (body frame), and how it carries, steers and brakes the aircraft.
#[derive(Clone)]
pub struct Leg {
    pub at: DVec3,
    pub wheel_r: f64,
    /// oleo travel (m), and the spring and damper (N/m, N s/m) with the most force it takes
    pub travel: f64,
    pub k: f64,
    pub c: f64,
    pub nf_max: f64,
    /// how much of the steering angle it follows (1 a nose wheel)
    pub steer: f64,
    pub brakes: bool,
    /// tyre side force per m/s of sideways slip and its friction limit; rolling
    /// resistance likewise; braking force per m/s and its friction limit
    pub side_k: f64,
    pub side_mu: f64,
    pub roll_k: f64,
    pub roll_mu: f64,
    pub brake_k: f64,
    pub brake_mu: f64,
    /// sink rate (m/s) that folds it, and how it throws the aircraft about when it does
    pub fold_vn: f64,
    pub fold_kick: DVec3,
    /// what the HUD calls it
    pub name: &'static str,
}

/// How the airframe itself slides along the ground once it is down on it.
#[derive(Clone)]
pub struct Hull {
    pub k: f64,
    pub c: f64,
    pub nf_max: f64,
    /// friction force per m/s of sliding (before the friction limit)
    pub drag: f64,
}

/// Where the tail surfaces are, for breaking them and for what still touches the ground.
#[derive(Clone)]
pub struct Tail {
    /// height of the fin's root and how tall it is above that; the same for the rudder
    pub fin_y0: f32,
    pub fin_h: f32,
    pub rudder_y0: f32,
    pub rudder_h: f32,
    /// |x| beyond which a stabiliser half breaks off, and an elevator half
    pub stab_root: f32,
    pub elev_root: f32,
    /// where along the tail cone it snaps (z range)
    pub break_z: [f32; 2],
    /// the fin's root and tip, and a stabiliser tip (right side), as contact points
    pub fin_base: DVec3,
    pub fin_tip: DVec3,
    pub stab_tip: DVec3,
}

/// A piston engine turning a fixed-pitch propeller.
#[derive(Clone)]
pub struct Piston {
    /// rated power (W) at rated revs (rad/s)
    pub power: f64,
    pub w_rated: f64,
    /// propeller diameter (m); thrust and power coefficients at rest; the
    /// advance ratios where thrust and then the power it takes run out
    pub prop_d: f64,
    pub ct0: f64,
    pub cp0: f64,
    pub j_t: f64,
    pub j_p: f64,
    /// propeller, crankshaft and flywheel (kg m²)
    pub prop_i: f64,
    /// about the static thrust at full power (scale for torque and slipstream effects)
    pub static_thrust: f64,
    /// friction and pumping losses: constant and at rated revs (N m)
    pub fric0: f64,
    pub fric1: f64,
    /// height of the propeller hub and where a broken blade snaps (body y)
    pub blade_break_y: f32,
}

/// High-bypass turbofans.
#[derive(Clone)]
pub struct Fan {
    pub n: usize,
    /// sea-level static thrust of each (N)
    pub thrust: f64,
    /// fan speed at idle (fraction of N1) and the thrust it gives (fraction)
    pub idle: f64,
    pub idle_thrust: f64,
    /// fan revs at 100% N1
    pub n1_rpm: f64,
    /// how quickly the fan spools up and down (1/s near the top)
    pub spool_up: f64,
    pub spool_down: f64,
    /// reverse thrust as a fraction of forward, and how long the reversers take to deploy (s)
    pub reverse: f64,
    pub reverser_time: f64,
    /// thrust with the boost key held, as a multiple of full dry thrust (an afterburner)
    pub boost: f64,
    /// thrust lapse with Mach number m: 1 - a m + b m² (a high-bypass fan loses thrust
    /// as it speeds up; a low-bypass afterburning engine gains it from the ram air)
    pub mach_lapse: [f64; 2],
}

#[derive(Clone)]
pub enum Engine {
    Piston(Piston),
    Fan(Fan),
}

/// The handling: angular accelerations straight from dynamic pressure (see
/// `Plane::step`). `q_ref` is the dynamic pressure (Pa) at which control
/// authority is "1"; `damp_v` turns rotation rates into damping.
#[derive(Clone)]
pub struct Handling {
    pub q_ref: f64,
    pub a_max: f64,
    pub damp_v: f64,
    pub ad_min: f64,
    pub ad_max: f64,
    /// pitch: stiffness about the trimmed angle of attack (and without a
    /// tailplane), stick authority, how far the stick moves the angle of attack, damping
    pub pitch_k: f64,
    pub pitch_k_lost: f64,
    pub pitch_ctl: f64,
    pub alpha_ctl: f64,
    pub pitch_d: f64,
    /// roll: authority, damping, roll from sideslip
    pub roll_ctl: f64,
    pub roll_d: f64,
    pub roll_beta: f64,
    /// how much roll authority is left at high speed with the flaps up (outboard ailerons locked out)
    pub roll_clean: f64,
    /// yaw: rudder authority, weathercock stiffness, damping
    pub yaw_ctl: f64,
    pub yaw_beta: f64,
    pub yaw_d: f64,
    /// a banked aircraft rolling back level; adverse yaw from the ailerons
    pub dihedral: f64,
    pub adverse: f64,
    /// a stall: nose drop, wing drop, and how much a yaw rate feeds the wing drop
    pub stall_pitch: f64,
    pub stall_drop: f64,
    pub stall_yaw: f64,
}

/// The head-on outline of the hole it punches through a building (m, body
/// frame seen from ahead), see `destruct::Hitter`.
#[derive(Clone)]
pub struct Hole {
    /// fuselage half width and half height, its centre's height, corner radius
    pub fus: [f32; 4],
    /// wing: root height, dihedral (rise per metre out), half thickness at root and tip
    pub wing: [f32; 4],
    /// stabiliser height, half thickness, half span
    pub stab: [f32; 3],
    /// fin root height, height, half width at root and tip
    pub fin: [f32; 4],
    /// engine pods: (|x|, y, radius); radius 0 for none
    pub pods: [[f32; 3]; 2],
}

#[derive(Clone)]
pub enum Sound {
    Piston,
    Jet,
    /// a pair of afterburning engines, from recordings (audio.rs `Fighter`)
    Fighter,
}

pub struct Spec {
    /// what plane.wgsl knows it by (its paint scheme, windows, panel lines)
    pub id: u32,
    pub name: &'static str,
    /// names it answers to on the command line
    pub keys: &'static [&'static str],
    pub blurb: &'static str,
    /// its picture in the start screen's aircraft window: a PNG with alpha,
    /// 680 x 392 (the window's 340 x 196 at twice the size, for hi-dpi)
    pub still: &'static [u8],
    pub build: fn() -> Model,
    /// the breakable groups (see airframe::G_*): name, mass (kg)
    pub groups: [(&'static str, f32); 7],

    // ---- geometry the flight and damage models need
    /// span station (m out from the centreline) of the wing tip
    pub span: f32,
    /// the right wing's section at a span station
    pub wing_sect: fn(f32) -> Sect,
    /// fuselage section at station z: top, the height of its widest point, bottom (body y)
    pub fuse: fn(f32) -> (f32, f32, f32),
    /// span stations the ailerons run between, and the fuel tanks
    pub aileron: [f32; 2],
    pub tank: [f32; 2],
    pub tail: Tail,
    /// the pilot's eye
    pub eye: Vec3,
    /// does a view from the eye along this direction see the cabin (rather than out a window)?
    pub sees_inside: fn(Vec3, Vec3) -> bool,
    /// points of the fuselage that can touch the ground, and of the tail cone
    pub contacts: &'static [([f64; 3], Part)],
    pub tail_pts: &'static [[f64; 3]],
    /// wing contact stations as fractions of the span
    pub wing_st: [f32; 5],
    /// points that must not touch anything: nose, left tip, right tip, tail,
    /// fin top, belly, lowest point of the propeller or engine pods
    pub hard: [DVec3; 7],
    pub gear: &'static [Leg],
    pub hull: Hull,
    pub hole: Hole,

    // ---- mass and aerodynamics
    pub mass: f64,
    /// pitch, yaw, roll (kg m²)
    pub inertia: DVec3,
    pub wing_area: f64,
    /// wing span (m)
    pub b: f64,
    /// lift at zero angle of attack and its slope per radian
    pub cl0: f64,
    pub cl_a: f64,
    /// drag: clean, the induced-drag factor, the lowered gear, flaps (linear and squared in the extension)
    pub cd0: f64,
    pub k_ind: f64,
    pub cd_gear: f64,
    pub cd_flap: [f64; 2],
    /// lift the flaps add at full travel
    pub flap_cl: f64,
    /// stall angle of attack clean, how much full flaps take off it and leading-edge devices add
    pub stall_alpha: f64,
    pub stall_flap: f64,
    pub stall_slat: f64,
    /// the highest angle of attack the trim will hold
    pub alpha_trim: f64,
    /// Mach number where wave drag starts (0: never flies fast enough to care)
    pub mcrit: f64,
    /// supersonic wave drag it levels off at by `mcrit` + 0.25 (0: Lock's fourth-power rise, for subsonic types)
    pub cd_wave: f64,
    /// spoilers fully up: the drag they add and the lift they dump
    pub spoiler_cd: f64,
    pub spoiler_cl: f64,
    /// flap lever notches (degrees) and the fastest each may be out at (m/s indicated)
    pub flaps: &'static [f64],
    pub vfe: &'static [f64],
    /// how quickly the flaps run (fraction of full travel per second)
    pub flap_rate: f64,
    /// from which notch the leading-edge devices are out (0: it has none)
    pub slats_at: u8,
    pub gear_rate: f64,
    /// never-exceed speed (m/s indicated) and Mach limit (0: none); flutter sets in at `flutter_v`
    pub vne: f64,
    pub mmo: f64,
    pub flutter_v: f64,
    /// load where the spar starts to fatigue, and where it snaps
    pub g_limit: f32,
    pub g_break: f32,
    /// fly-by-wire: the most load (g) the stick can ask for (0: none, the stick moves the elevator)
    pub fbw_g: f64,
    /// nose wheel steering: largest angle, and the speed it fades down to a fifth by (0: it does not)
    pub steer_max: f64,
    pub steer_fade: f64,
    pub engine: Engine,
    pub handling: Handling,

    // ---- the game around it
    /// starting speed in the air, and on a final approach (m/s) with that flap notch
    pub cruise: f64,
    pub approach: (f64, u8),
    /// chase camera distance and the size of its own fine shadow map (m)
    pub cam_dist: f64,
    pub shadow_r: f32,
    pub sound: Sound,
    /// how many times the size of the model it is drawn as (plane.wgsl's
    /// `kscale`, sent in glow.z), and how many times the speeds of the type it
    /// scales up (its Mach number, speed limits and wind noise are read
    /// divided by it); 1 for a type of its own (see `scaled`)
    pub size_k: f32,
    pub speed_k: f64,
}

fn leak<T>(v: Vec<T>) -> &'static [T] {
    Box::leak(v.into_boxed_slice())
}

/// A copy of `base` `kl` times its size in every dimension, flying `kv`
/// times as fast. Its geometry scales by `kl`. Lift and drag at `kv` times the
/// speed on `kl`² times the wing grow by f = kv²·kl², so mass, thrust and every
/// force scale by f (wing loading over dynamic pressure, the trim angles and
/// the thrust-to-weight stay the 747's); moments of inertia by f·kl². Speeds
/// (limits, the start, approach) scale by `kv`, the handling's reference
/// dynamic pressure by kv² and its damping speed by kv, so it pitches, rolls
/// and yaws at the base type's rates in rad/s (gravity does not scale: with
/// `kv` > 1 it turns and climbs in wider arcs than the base type scaled).
/// The landing gear and the hull sink `kl` times as far under f times the
/// weight at the same damping ratio (springs f/kl, dampers f/√kl). With
/// `kv` = 1 this is a plain scale model (mass k², inertia k⁴). The function
/// fields still point at `base`'s: the caller supplies scaled ones (they
/// cannot carry `kl`), and its own name, keys and id.
pub fn scaled(base: &Spec, kl: f64, kv: f64) -> Spec {
    let klf = kl as f32;
    let f = kv * kv * kl * kl;
    let ff = f as f32;
    let (fk, fc, fv) = (f / kl, f / kl.sqrt(), f / kv);
    let t = &base.tail;
    let h = &base.hole;
    let hd = &base.handling;
    Spec {
        id: base.id,
        name: base.name,
        keys: base.keys,
        blurb: base.blurb,
        still: base.still,
        build: base.build,
        groups: base.groups.map(|(n, m)| (n, m * ff)),
        span: base.span * klf,
        wing_sect: base.wing_sect,
        fuse: base.fuse,
        aileron: base.aileron.map(|x| x * klf),
        tank: base.tank.map(|x| x * klf),
        tail: Tail {
            fin_y0: t.fin_y0 * klf,
            fin_h: t.fin_h * klf,
            rudder_y0: t.rudder_y0 * klf,
            rudder_h: t.rudder_h * klf,
            stab_root: t.stab_root * klf,
            elev_root: t.elev_root * klf,
            break_z: t.break_z.map(|z| z * klf),
            fin_base: t.fin_base * kl,
            fin_tip: t.fin_tip * kl,
            stab_tip: t.stab_tip * kl,
        },
        eye: base.eye * klf,
        sees_inside: base.sees_inside,
        contacts: leak(base.contacts.iter().map(|(p, part)| (p.map(|x| x * kl), *part)).collect()),
        tail_pts: leak(base.tail_pts.iter().map(|p| p.map(|x| x * kl)).collect()),
        wing_st: base.wing_st,
        hard: base.hard.map(|p| p * kl),
        gear: leak(
            base.gear
                .iter()
                .map(|l| Leg {
                    at: l.at * kl,
                    wheel_r: l.wheel_r * kl,
                    travel: l.travel * kl,
                    k: l.k * fk,
                    c: l.c * fc,
                    nf_max: l.nf_max * f,
                    side_k: l.side_k * fv,
                    roll_k: l.roll_k * fv,
                    brake_k: l.brake_k * fv,
                    fold_vn: l.fold_vn * kv * kl.sqrt(),
                    ..l.clone()
                })
                .collect(),
        ),
        hull: Hull { k: base.hull.k * fk, c: base.hull.c * fc, nf_max: base.hull.nf_max * f, drag: base.hull.drag * fv },
        hole: Hole {
            fus: h.fus.map(|x| x * klf),
            // (the dihedral is a slope)
            wing: [h.wing[0] * klf, h.wing[1], h.wing[2] * klf, h.wing[3] * klf],
            stab: h.stab.map(|x| x * klf),
            fin: h.fin.map(|x| x * klf),
            pods: h.pods.map(|p| p.map(|x| x * klf)),
        },
        mass: base.mass * f,
        inertia: base.inertia * (f * kl * kl),
        wing_area: base.wing_area * kl * kl,
        b: base.b * kl,
        vfe: leak(base.vfe.iter().map(|v| v * kv).collect()),
        vne: base.vne * kv,
        flutter_v: base.flutter_v * kv,
        steer_fade: base.steer_fade * kv,
        engine: match &base.engine {
            Engine::Fan(fe) => Engine::Fan(Fan { thrust: fe.thrust * f, ..fe.clone() }),
            Engine::Piston(p) => Engine::Piston(Piston {
                power: p.power * f * kv,
                prop_d: p.prop_d * kl,
                prop_i: p.prop_i * f * kl * kl,
                static_thrust: p.static_thrust * f,
                fric0: p.fric0 * f,
                fric1: p.fric1 * f,
                blade_break_y: p.blade_break_y * klf,
                ..p.clone()
            }),
        },
        handling: Handling { q_ref: hd.q_ref * kv * kv, damp_v: hd.damp_v * kv, ..hd.clone() },
        cruise: base.cruise * kv,
        approach: (base.approach.0 * kv, base.approach.1),
        cam_dist: base.cam_dist * kl,
        shadow_r: base.shadow_r * klf,
        sound: base.sound.clone(),
        size_k: base.size_k * klf,
        speed_k: base.speed_k * kv,
        ..*base
    }
}

/// `m` made `k` times its size: every vertex, hinge and slide.
pub fn scale_model(mut m: Model, k: f32) -> Model {
    let grow = |mesh: &mut crate::airframe::Mesh| {
        for v in &mut mesh.v {
            v.pos = v.pos.map(|x| x * k);
        }
    };
    for p in &mut m.pieces {
        grow(&mut p.mesh);
        p.pivot *= k;
        if let Anim::Move(mo) = &mut p.anim {
            mo.slide.gain *= k;
            mo.slide.offset *= k;
        }
    }
    m.groups.iter_mut().for_each(grow);
    m
}

impl std::fmt::Debug for Spec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name)
    }
}

impl Spec {
    pub fn is_jet(&self) -> bool {
        matches!(self.engine, Engine::Fan(_))
    }
    /// The flap notch fraction (of full travel) for a lever notch.
    pub fn flap_frac(&self, notch: u8) -> f64 {
        let full = *self.flaps.last().unwrap_or(&1.0);
        self.flaps.get(notch as usize).map(|d| d / full.max(1e-6)).unwrap_or(1.0)
    }
    pub fn max_notch(&self) -> u8 {
        (self.flaps.len().max(1) - 1) as u8
    }
}
