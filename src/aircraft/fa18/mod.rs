//! McDonnell Douglas F/A-18C Hornet: a twin-engine carrier fighter.
//!
//! Unlike the other types, the airframe is modelled in Blender
//! (`assets/blender/fa18.py`, which also saves `fa18.blend`) and exported to
//! `assets/fa18.bin.gz`: one mesh per piece, with the hinge each one turns
//! about. How the pieces move, what they belong to and what they are to the
//! damage model is set here, by name. Rebuild the asset with
//!
//!   blender -b --factory-startup --python assets/blender/fa18.py -- assets
//!
//! The cockpit inside is built here in Rust (`pit.rs`), like the 747's flight
//! deck, so its panels can carry legends and live displays.
//!
//! Its paint is a picture too (`assets/fa18_paint.jpg`, see
//! `assets/blender/fa18_paint.py`), projected onto the skin by plane.wgsl;
//! so are its exhaust plume (the "afterburner" piece, shells the shader lights
//! up) and its engines' sound (audio.rs `Fighter`).
//!
//! Body frame as everywhere: +X right, +Y up, -Z forward, origin at the centre
//! of gravity.
use super::*;
use crate::airframe::*;
use crate::render::PartVertex;
use std::collections::HashMap;
use std::io::Read;

mod pit;

const KT: f64 = 1.0 / 1.94384;

// ---------------------------------------------------------------- geometry (in step with fa18.py)

/// Fuselage stations: y forward (Blender frame), half width, top, height of
/// the widest point, bottom, superellipse exponent.
const FUSE: [[f32; 6]; 19] = [
    [8.90, 0.000, -0.03, -0.03, -0.03, 2.0],
    [8.65, 0.120, 0.07, -0.04, -0.15, 2.0],
    [8.20, 0.240, 0.17, -0.05, -0.27, 2.0],
    [7.70, 0.345, 0.28, -0.07, -0.39, 2.1],
    [7.20, 0.430, 0.39, -0.08, -0.48, 2.2],
    [6.50, 0.515, 0.51, -0.09, -0.56, 2.3],
    [5.80, 0.575, 0.60, -0.09, -0.61, 2.4],
    [5.00, 0.615, 0.65, -0.08, -0.63, 2.6],
    [4.00, 0.650, 0.68, -0.06, -0.65, 2.7],
    [3.00, 0.700, 0.70, -0.06, -0.67, 2.8],
    [2.00, 0.800, 0.70, -0.08, -0.70, 3.0],
    [1.00, 0.960, 0.68, -0.10, -0.72, 3.1],
    [-0.50, 1.060, 0.64, -0.11, -0.74, 3.2],
    [-2.00, 1.100, 0.60, -0.12, -0.74, 3.2],
    [-3.50, 1.100, 0.55, -0.12, -0.72, 3.2],
    [-5.00, 1.070, 0.48, -0.10, -0.66, 3.0],
    [-6.20, 1.020, 0.40, -0.08, -0.58, 2.8],
    [-7.00, 0.950, 0.33, -0.06, -0.50, 2.6],
    [-7.40, 0.900, 0.28, -0.05, -0.45, 2.5],
];

fn fuse_at(y: f32) -> [f32; 6] {
    if y >= FUSE[0][0] {
        return FUSE[0];
    }
    for w in FUSE.windows(2) {
        let (a, b) = (w[0], w[1]);
        if a[0] >= y && y >= b[0] {
            let t = (a[0] - y) / (a[0] - b[0]);
            let mut r = [0.0; 6];
            for k in 0..6 {
                r[k] = a[k] + (b[k] - a[k]) * t;
            }
            return r;
        }
    }
    FUSE[FUSE.len() - 1]
}

/// Fuselage at body station z: top, the height of its widest point, bottom.
fn fuse(z: f32) -> (f32, f32, f32) {
    let r = fuse_at(-z);
    (r[2], r[3], r[4])
}

// The wing: root (inside the fuselage) and tip span stations, leading edge
// at the root, its sweep (m back per m out), chords, height, anhedral, thickness.
const W_X0: f32 = 0.9;
const W_X1: f32 = 5.75;
const W_LE0: f32 = 1.0;
const W_SWEEP: f32 = 0.503;
const W_C0: f32 = 4.15;
const W_C1: f32 = 1.65;
const W_Z0: f32 = 0.12;
const W_ANH: f32 = 0.052;

pub fn wing_sect(x: f32) -> Sect {
    let x = x.max(0.0);
    let d = (x - W_X0).max(0.0);
    let c = W_C0 + (W_C1 - W_C0) * (d / (W_X1 - W_X0)).min(1.0);
    Sect { le: Vec3::new(x, W_Z0 - W_ANH * d, -(W_LE0 - W_SWEEP * d)), c, cdir: Vec3::Z, up: Vec3::new(W_ANH, 1.0, 0.0).normalize(), tf: 1.0 }
}

/// The pilot's eye, under the canopy's highest point.
const EYE: Vec3 = Vec3::new(0.0, 1.0, -4.55);


// ---------------------------------------------------------------- the model

struct Obj {
    pivot: Vec3,
    axis: Vec3,
    /// the gear legs: how far they turn to stow (rad)
    angle: f32,
    mesh: Mesh,
}

struct Reader<'a> {
    b: &'a [u8],
    at: usize,
}
impl Reader<'_> {
    fn take(&mut self, n: usize) -> &[u8] {
        let s = &self.b[self.at..self.at + n];
        self.at += n;
        s
    }
    fn u32(&mut self) -> u32 {
        u32::from_le_bytes(self.take(4).try_into().unwrap())
    }
    fn f32(&mut self) -> f32 {
        f32::from_le_bytes(self.take(4).try_into().unwrap())
    }
    fn v3(&mut self) -> Vec3 {
        Vec3::new(self.f32(), self.f32(), self.f32())
    }
}

/// The pieces exported from Blender, by name.
fn load() -> HashMap<String, Obj> {
    let mut raw = Vec::new();
    flate2::read::GzDecoder::new(&include_bytes!("../../../assets/fa18.bin.gz")[..]).read_to_end(&mut raw).expect("fa18 model");
    let mut r = Reader { b: &raw, at: 0 };
    assert_eq!(r.take(4), b"FFAC", "fa18 model: not a model file");
    assert_eq!(r.u32(), 1, "fa18 model: unknown version");
    let n = r.u32();
    let mut out = HashMap::new();
    for _ in 0..n {
        let len = r.u32() as usize;
        let name = String::from_utf8_lossy(r.take(len)).into_owned();
        let (pivot, axis, angle) = (r.v3(), r.v3(), r.f32());
        let nv = r.u32() as usize;
        let mut mesh = Mesh::default();
        mesh.v.reserve(nv);
        for _ in 0..nv {
            let mut f = [0.0f32; 13];
            for x in f.iter_mut() {
                *x = r.f32();
            }
            mesh.v.push(PartVertex { pos: [f[0], f[1], f[2]], nrm: [f[3], f[4], f[5]], col: [f[6], f[7], f[8], 1.0], uv: [0.0, 0.0], mat: [f[9], f[10], f[11], f[12]] });
        }
        let ni = r.u32() as usize;
        mesh.i = (0..ni).map(|_| r.u32()).collect();
        out.insert(name, Obj { pivot, axis, angle, mesh });
    }
    out
}

/// Every piece: name (as in the .blend), breakable group, what it is to the
/// damage model, and the piece it hangs from (listed before it).
const PIECES: &[(&str, usize, Role, Option<&str>)] = &[
    ("radome", G_PROP, Role::Other, None),
    ("nose section", G_ENGINE, Role::Other, None),
    ("fuselage", G_FUSE, Role::Other, None),
    ("tail cone", G_TAIL, Role::TailCone, None),
    ("spine", G_FUSE, Role::Other, None),
    ("left intake", G_FUSE, Role::Other, None),
    ("right intake", G_FUSE, Role::Other, None),
    ("left lex", G_FUSE, Role::Other, None),
    ("right lex", G_FUSE, Role::Other, None),
    ("left nozzle", G_TAIL, Role::Other, None),
    ("right nozzle", G_TAIL, Role::Other, None),
    ("canopy", G_ENGINE, Role::Other, None),
    ("canopy frame", G_ENGINE, Role::Other, None),
    ("left wing", G_LWING, Role::Other, None),
    ("right wing", G_RWING, Role::Other, None),
    ("left le flap", G_LWING, Role::Other, None),
    ("right le flap", G_RWING, Role::Other, None),
    ("left flap", G_LWING, Role::Flap(0), None),
    ("right flap", G_RWING, Role::Flap(1), None),
    ("left aileron", G_LWING, Role::Aileron(0), None),
    ("right aileron", G_RWING, Role::Aileron(1), None),
    ("left tip rail", G_LWING, Role::Other, None),
    ("right tip rail", G_RWING, Role::Other, None),
    ("left nav light", G_LWING, Role::Other, None),
    ("right nav light", G_RWING, Role::Other, None),
    ("left stabilator", G_TAIL, Role::Elevator, None),
    ("right stabilator", G_TAIL, Role::Elevator, None),
    ("left fin", G_TAIL, Role::Stab, None),
    ("right fin", G_TAIL, Role::Stab, None),
    ("left rudder", G_TAIL, Role::Rudder, None),
    ("right rudder", G_TAIL, Role::Rudder, None),
    ("speedbrake", G_FUSE, Role::Other, None),
    ("tail light", G_TAIL, Role::Other, None),
    ("nose leg", G_GEAR, Role::Gear(0), None),
    ("nose fork", G_GEAR, Role::Gear(0), Some("nose leg")),
    ("nose wheel", G_GEAR, Role::Gear(0), Some("nose fork")),
    ("left main leg", G_GEAR, Role::Gear(1), None),
    ("left main strut", G_GEAR, Role::Gear(1), Some("left main leg")),
    ("left main wheel", G_GEAR, Role::Gear(1), Some("left main strut")),
    ("right main leg", G_GEAR, Role::Gear(2), None),
    ("right main strut", G_GEAR, Role::Gear(2), Some("right main leg")),
    ("right main wheel", G_GEAR, Role::Gear(2), Some("right main strut")),
    ("left nose door", G_ENGINE, Role::Other, None),
    ("right nose door", G_ENGINE, Role::Other, None),
    ("left strut door", G_FUSE, Role::Other, None),
    ("right strut door", G_FUSE, Role::Other, None),
    ("left main door", G_FUSE, Role::Other, None),
    ("right main door", G_FUSE, Role::Other, None),
    // The exhaust plume: drawn after everything else, in the transparent pass.
    ("afterburner", G_TAIL, Role::Other, None),
];

/// How a piece moves. Hinges point so that a positive angle puts a trailing
/// edge down (the rudders: to the right); the gear is modelled down and locked,
/// its doors open. A piece's `angle` is how far it turns to stow: the legs
/// swing up, the main struts' lower parts twist to lay the wheels flat, the
/// doors shut.
fn anim(name: &str, stow: f32) -> Anim {
    let left = name.starts_with("left");
    let s = if left { -1.0 } else { 1.0 };
    let leg = if name.starts_with("nose") { 0 } else if left { 1 } else { 2 };
    let gear = || Anim::hinge(Drive::new(Input::Gear, -stow).smooth(0.15, 0.85).offset(stow));
    if name.ends_with("le flap") {
        // leading-edge flaps droop with the trailing-edge flaps
        Anim::hinge(Drive::new(Input::Slats, -0.55))
    } else if name.ends_with(" flap") {
        Anim::hinge(Drive::new(Input::Flaps, 45f32.to_radians()))
    } else if name.ends_with("aileron") {
        Anim::hinge(Drive::new(Input::Aileron(s), 0.40))
    } else if name.ends_with("stabilator") {
        // all-moving tailplanes: together for pitch, apart to help the ailerons roll
        Anim::hinge2(Drive::new(Input::Elevator, 0.42), Drive::new(Input::Aileron(s), 0.10))
    } else if name.ends_with("rudder") {
        Anim::hinge(Drive::new(Input::Rudder, 0.52))
    } else if name == "speedbrake" {
        Anim::hinge(Drive::new(Input::Spoiler(0.0), -1.1))
    } else if name.ends_with(" leg") || name.ends_with("main strut") {
        gear()
    } else if name.ends_with("main door") {
        // open for the leg to pass, shut again behind it either way
        Anim::hinge(Drive::new(Input::Gear, -stow).pulse(0.0, 0.18, 0.88, 1.0).offset(stow))
    } else if name.ends_with(" door") {
        // open before the legs come down, shut once they are up
        Anim::hinge(Drive::new(Input::Gear, -stow).smooth(0.0, 0.2).offset(stow))
    } else if name == "nose fork" {
        Anim::turn_slide(Drive::new(Input::Steer, 1.0), Drive::new(Input::Compress(0), 1.0), Vec3::Y)
    } else if name == "nose wheel" {
        Anim::hinge(Drive::new(Input::Wheel(0), -1.0))
    } else if name.ends_with("main wheel") {
        Anim::turn_slide(Drive::new(Input::Wheel(leg), -1.0), Drive::new(Input::Compress(leg), 1.0), Vec3::Y)
    } else {
        Anim::Fixed
    }
}

pub fn build() -> Model {
    let objs = load();
    let mut b = Builder::default();
    let mut at: HashMap<&str, usize> = HashMap::new();
    for &(name, group, role, parent) in PIECES {
        let o = objs.get(name).unwrap_or_else(|| panic!("fa18 model has no piece called {name:?}"));
        let k = b.add(name, group, o.mesh.clone(), anim(name, o.angle), o.pivot, o.axis, parent.map(|p| at[p]));
        b.role(k, role);
        if name == "canopy" || name == "afterburner" {
            b.pieces[k].glass = true;
        }
        at.insert(name, k);
    }
    bake_ao(&mut b.pieces, &AoBake { vox: 0.06, reach: 3.0, rays: 24, open: &|_, _| false });
    // The cockpit gets its own, much finer bake, shut in by the nose's skin
    // and the canopy's frame; the light comes in through the glass.
    let mut c = Builder::default();
    pit::build(&mut c);
    let n_pit = c.pieces.len();
    for name in ["nose section", "canopy frame"] {
        if let Some(k) = b.pieces.iter().position(|p| p.name == name) {
            let mut skin = b.pieces[k].mesh.clone();
            skin.i = skin.i.chunks(3).filter(|t| t.iter().all(|&i| { let z = skin.v[i as usize].pos[2]; z > -6.6 && z < -2.6 })).flatten().copied().collect();
            c.fixed("skin (occluder)", G_ENGINE, skin);
        }
    }
    bake_ao(&mut c.pieces, &AoBake { vox: 0.012, reach: 1.6, rays: 48, open: &|_, _| false });
    let off = b.pieces.len();
    for mut p in c.pieces.into_iter().take(n_pit) {
        p.parent = p.parent.map(|i| i + off);
        b.pieces.push(p);
    }
    Model::finish(b.pieces)
}

// ---------------------------------------------------------------- the type

const fn main_leg(x: f64, name: &'static str, kick: f64) -> Leg {
    Leg {
        at: DVec3::new(x, -1.95, 0.60),
        wheel_r: 0.38,
        travel: 0.35,
        k: 5.5e5,
        c: 7.0e4,
        nf_max: 1.6e6,
        steer: 0.0,
        brakes: true,
        side_k: 1.6e5,
        side_mu: 0.8,
        roll_k: 2.0e3,
        roll_mu: 0.02,
        brake_k: 5.0e4,
        brake_mu: 0.6,
        // carrier gear: built to take a no-flare arrival at 7 m/s and more
        fold_vn: 7.5,
        fold_kick: DVec3::new(0.3, 0.0, kick),
        name,
    }
}

pub static SPEC: Spec = Spec {
    id: 2,
    name: "F/A-18C Hornet",
    keys: &["fa18", "f18", "fa-18", "f/a-18", "hornet"],
    blurb: "Twin-engine carrier fighter, 15.5 t. Fly-by-wire holds it to 7.5 g; Space for afterburner, / speedbrake. 135 kt approach.",
    still: include_bytes!("../../../assets/aircraft/fa18.png"),
    build,
    groups: [("fuselage", 6_500.0), ("tail", 900.0), ("left wing", 1_200.0), ("right wing", 1_200.0), ("nose section", 900.0), ("radome", 60.0), ("gear", 500.0)],
    span: W_X1,
    wing_sect,
    fuse,
    aileron: [3.5, 5.6],
    tank: [1.0, 4.2],
    tail: Tail {
        fin_y0: 0.25,
        fin_h: 2.6,
        rudder_y0: 0.25,
        rudder_h: 1.55,
        stab_root: 0.9,
        elev_root: 0.9,
        break_z: [5.0, 6.2],
        fin_base: DVec3::new(0.95, 0.25, 4.6),
        fin_tip: DVec3::new(1.89, 2.83, 6.1),
        stab_tip: DVec3::new(3.25, -0.2, 7.6),
    },
    eye: EYE,
    sees_inside: pit::sees_inside,
    contacts: &[
        ([0.0, -0.03, -8.9], Part::Nose),
        ([0.0, -0.40, -7.7], Part::Nose),
        ([0.0, 0.28, -7.7], Part::Nose),
        ([0.0, -0.62, -5.4], Part::Belly),
        ([0.0, -0.66, -3.5], Part::Belly),
        ([0.0, -0.74, 0.5], Part::Belly),
        ([0.0, -0.73, 2.5], Part::Belly),
        ([-1.06, -0.70, 0.0], Part::Belly),
        ([1.06, -0.70, 0.0], Part::Belly),
        ([-1.36, -0.30, -1.0], Part::Side),
        ([1.36, -0.30, -1.0], Part::Side),
        ([-0.62, 0.0, -4.5], Part::Side),
        ([0.62, 0.0, -4.5], Part::Side),
        ([-1.08, -0.10, 3.5], Part::Side),
        ([1.08, -0.10, 3.5], Part::Side),
        ([0.0, 1.22, -4.5], Part::Roof),
        ([0.0, 0.85, -2.5], Part::Roof),
        ([0.0, 0.66, 1.0], Part::Roof),
    ],
    tail_pts: &[[0.0, -0.64, 5.2], [0.0, -0.52, 6.7], [-0.52, -0.48, 8.0], [0.52, -0.48, 8.0]],
    wing_st: [1.6 / W_X1, 2.6 / W_X1, 3.6 / W_X1, 4.6 / W_X1, 5.4 / W_X1],
    hard: [
        DVec3::new(0.0, -0.03, -8.9),
        DVec3::new(-5.9, -0.2, 1.3),
        DVec3::new(5.9, -0.2, 1.3),
        DVec3::new(0.0, -0.05, 8.0),
        DVec3::new(1.89, 2.83, 6.1),
        DVec3::new(0.0, -0.74, 0.5),
        DVec3::new(0.0, -0.74, 2.0),
    ],
    gear: &[
        Leg {
            at: DVec3::new(0.0, -1.95, -4.80),
            wheel_r: 0.28,
            travel: 0.30,
            k: 2.5e5,
            c: 3.0e4,
            nf_max: 6.0e5,
            steer: 1.0,
            brakes: false,
            side_k: 6.0e4,
            side_mu: 0.8,
            roll_k: 1.0e3,
            roll_mu: 0.02,
            brake_k: 0.0,
            brake_mu: 0.0,
            fold_vn: 7.0,
            fold_kick: DVec3::new(-0.2, 0.0, 0.0),
            name: "NOSE GEAR",
        },
        main_leg(-1.57, "LEFT MAIN GEAR", 0.2),
        main_leg(1.57, "RIGHT MAIN GEAR", -0.2),
    ],
    hull: Hull { k: 1.2e6, c: 1.6e5, nf_max: 2.5e6, drag: 5.0e4 },
    hole: Hole { fus: [1.1, 0.72, -0.05, 0.4], wing: [0.12, -W_ANH, 0.10, 0.04], stab: [-0.15, 0.06, 3.3], fin: [0.25, 2.6, 0.06, 0.025], pods: [[0.0; 3]; 2] },
    // About 15.5 t: full internal fuel and a pair of Sidewinders (empty 10.4 t, maximum 23.4 t).
    mass: 15_500.0,
    // Pitch, yaw, roll (NASA's F/A-18 HARV simulation data).
    inertia: DVec3::new(165.7e3, 205.1e3, 31.2e3),
    wing_area: 37.16,
    b: 11.43,
    // A low aspect ratio wing helped by the leading-edge extensions' vortices.
    cl0: 0.05,
    cl_a: 4.6,
    cd0: 0.028,
    k_ind: 0.12,
    cd_gear: 0.02,
    cd_flap: [0.03, 0.04],
    flap_cl: 0.5,
    stall_alpha: 0.42,
    stall_flap: 0.04,
    stall_slat: 0.06,
    alpha_trim: 0.38,
    mcrit: 0.85,
    cd_wave: 0.032,
    spoiler_cd: 0.045,
    spoiler_cl: 0.0,
    // Flaps: up, half, full.
    flaps: &[0.0, 30.0, 45.0],
    vfe: &[1e9, 250.0 * KT, 230.0 * KT],
    flap_rate: 0.3,
    slats_at: 1,
    gear_rate: 1.0 / 5.0,
    // 750 kt, Mach 1.8.
    vne: 750.0 * KT,
    mmo: 1.8,
    flutter_v: 820.0 * KT,
    g_limit: 9.0,
    g_break: 12.5,
    fbw_g: 7.5,
    steer_max: 0.9,
    steer_fade: 15.0,
    // Two F404-GE-402s: 79.2 kN each in afterburner (Space); the dry figure is a
    // little under the book 48.9 kN so it does not supercruise down low.
    engine: Engine::Fan(Fan { n: 2, thrust: 44_000.0, idle: 0.62, idle_thrust: 0.06, n1_rpm: 13_200.0, spool_up: 1.0, spool_down: 0.8, reverse: 0.0, reverser_time: 1.0, boost: 1.8, mach_lapse: [0.25, 0.45] }),
    handling: Handling {
        q_ref: 8_000.0,
        a_max: 3.5,
        damp_v: 68.0,
        ad_min: 0.3,
        ad_max: 3.5,
        pitch_k: 6.0,
        pitch_k_lost: 2.0,
        pitch_ctl: 0.15,
        alpha_ctl: 0.32,
        pitch_d: 5.0,
        roll_ctl: 5.5,
        roll_d: 3.0,
        roll_beta: 0.3,
        roll_clean: 1.0,
        yaw_ctl: 0.8,
        yaw_beta: 3.0,
        yaw_d: 2.5,
        dihedral: 0.05,
        adverse: 0.1,
        stall_pitch: 0.4,
        stall_drop: 0.6,
        stall_yaw: 1.0,
    },
    cruise: 180.0,
    approach: (70.0, 2),
    cam_dist: 26.0,
    shadow_r: 11.0,
    sound: Sound::Fighter,
    size_k: 1.0,
    speed_k: 1.0,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_builds_whole() {
        let m = build();
        let (c, h) = m.bounds();
        for p in &m.pieces {
            assert!(p.mesh.v.iter().all(|v| v.pos.iter().chain(v.nrm.iter()).all(|x| x.is_finite())), "{} has NaNs", p.name);
            assert!(!p.mesh.i.is_empty(), "{} is empty", p.name);
        }
        // 17.07 m long, 12.31 m across the wingtip missiles, 4.66 m tall on its wheels.
        let (len, span, top) = (h.z * 2.0, h.x * 2.0, c.y + h.y + 1.95);
        println!("{} pieces, length {len:.2} m, span {span:.2} m, height {top:.2} m", m.pieces.len());
        assert!((len - 17.07).abs() < 0.3, "{len}");
        assert!((span - 12.31).abs() < 0.3, "{span}");
        assert!((top - 4.66).abs() < 0.3, "{top}");
    }

    #[test]
    fn gear_stows_inside_the_fuselage() {
        let m = build();
        let mut p = crate::plane::Plane::spawn(&SPEC, 0.6, -2.1, 1000.0, 0.0, 120.0);
        p.gear = 0.0;
        let pose = p.pose(&m.pieces);
        for (i, pc) in m.pieces.iter().enumerate().filter(|(_, pc)| pc.name.ends_with("wheel")) {
            let c = pose[i].transform_point3(pc.pivot);
            let (top, _, bot) = fuse(c.z);
            let hw = fuse_at(-c.z)[1].max(if c.z.abs() < 3.0 { 1.36 } else { 0.0 });
            println!("{} stowed at {c:?}", pc.name);
            assert!(c.y > bot && c.y < top && c.x.abs() < hw, "{} sticks out: {c:?}", pc.name);
        }
    }
}
