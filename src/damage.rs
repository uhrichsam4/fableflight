//! Progressive damage: what has been torn off the aircraft, what each loss
//! does to the way it flies, and the pieces that tumble away afterwards.
//!
//! A break is a ragged plane through the body frame (`Tear`). The renderer
//! cuts any piece along it (see `tear_d` in plane.wgsl), so the stump left on
//! the aircraft and the part fluttering down are the same mesh seen from the
//! two sides of the same tear, with the same ragged edge.
use crate::aircraft::{Spec, MAX_GEAR};
use crate::airframe::{Piece, Role, G_ENGINE, G_FUSE, G_LWING, G_RWING, G_TAIL, N_GROUPS};
use crate::crash::{Rng, GRIT, SPARK};
use crate::geo;
use crate::render::{DrawObj, Particle};
use crate::world::World;
use glam::{DQuat, DVec3, Mat4, Vec3};

/// How ragged a torn edge is (metres of wander either side of the break).
pub const JAG: f32 = 0.14;

/// A break through a piece. `mode` 1 keeps the side where n·L < d, -1 the
/// other side; ±2 the same with |L.x| in place of L.x (both halves at once).
/// `jag` is how ragged its edge is (m): a clean shear is a few centimetres, torn skin far more.
#[derive(Clone, Copy, Default, PartialEq, Debug)]
pub struct Tear {
    pub n: Vec3,
    pub d: f32,
    pub mode: f32,
    pub jag: f32,
}

impl Tear {
    pub const NONE: Tear = Tear { n: Vec3::ZERO, d: 0.0, mode: 0.0, jag: 0.0 };
    pub fn new(n: Vec3, d: f32, mode: f32) -> Self {
        Self { n, d, mode, jag: JAG }
    }
    pub fn is_some(&self) -> bool {
        self.mode != 0.0
    }
    /// The same break seen from the other side.
    pub fn flip(&self) -> Self {
        Self { mode: -self.mode, ..*self }
    }
    /// Moved so the kept side grows by `by` metres (spars poking out past torn skin).
    pub fn grow(&self, by: f32) -> Self {
        if self.mode == 0.0 {
            return *self;
        }
        Self { d: self.d + by * self.mode.signum(), ..*self }
    }
    /// Does the break (without its ragged edge) keep this body-frame point?
    pub fn keeps(&self, p: Vec3) -> bool {
        if self.mode == 0.0 {
            return true;
        }
        let q = if self.mode.abs() > 1.5 { Vec3::new(p.x.abs(), p.y, p.z) } else { p };
        let v = q.dot(self.n) - self.d;
        (v < 0.0) == (self.mode > 0.0)
    }
    pub fn gpu(&self) -> [f32; 4] {
        [self.n.x, self.n.y, self.n.z, self.d]
    }
}

/// A wing that has buckled but not let go: the part outboard of the break
/// hangs from the torn skin on one side and swings about it (body frame).
#[derive(Clone, Copy, Debug, Default)]
pub struct Fold {
    pub on: bool,
    /// a point on the hinge line, and its direction (positive angles lift the tip)
    pub pivot: Vec3,
    pub axis: Vec3,
    pub ang: f32,
    pub rate: f32,
    /// the break, as the stump sees it (mode 1); and the hanging part's outer
    /// end if the tip had already gone (NONE: it runs out to the tip)
    pub tear: Tear,
    pub outer: Tear,
    /// span stations of the break and of the hanging part's outer end
    pub root: f32,
    pub tip: f32,
    /// how close the torn skin is to letting go (1 = gone), and how much load it takes (N m)
    pub stress: f32,
    pub hold: f32,
}

impl Fold {
    /// Where the hanging part is drawn: its rest pose swung about the hinge.
    pub fn mat(&self) -> Mat4 {
        Mat4::from_translation(self.pivot) * Mat4::from_axis_angle(self.axis, self.ang) * Mat4::from_translation(-self.pivot)
    }
}

/// Everything that can be wrong with the airframe.
#[derive(Clone, Copy, Debug)]
pub struct Damage {
    /// the aircraft it is
    pub spec: &'static Spec,
    /// span station each wing now ends at (`Spec::span` = whole), left then right
    pub wing: [f32; 2],
    /// the break each wing stump ends in (NONE with a shorter `wing`: a straight cut there)
    pub wing_tear: [Tear; 2],
    /// a wing that has snapped but still hangs on
    pub fold: [Fold; 2],
    /// the tail broken off through the tail cone: the break the stump keeps (NONE: attached)
    pub tail: Tear,
    /// how battered the fuselage is, 0..1
    pub hull: f32,
    /// control surfaces: 1 sound, 0.5 hanging from one hinge, 0 gone
    pub aileron: [f32; 2],
    pub flap: [f32; 2],
    /// left and right halves of the elevator and of the stabiliser (1 or 0)
    pub elevator: [f32; 2],
    pub stab: [f32; 2],
    /// how much of the rudder's and the fin's height is left
    pub rudder: f32,
    pub fin: f32,
    /// 1 both blades, 0.5 one blade snapped, 0 gone
    pub prop: f32,
    /// engine health: below 1 it leaks oil and loses power, at 0 it has seized
    pub engine: f32,
    /// each landing gear leg still attached (see `Spec::gear`)
    pub gear: [bool; MAX_GEAR],
    /// oil film over the windshield
    pub oil: f32,
    /// a crack in the windshield: body-frame z and y of its centre, and its size
    pub crack: [f32; 3],
    /// scrapes and dents, by breakable group
    pub scuff: [f32; N_GROUPS],
    /// how hard the airframe is shaking (camera, HUD)
    pub vibe: f32,
    /// smoke pouring from the engine, 0..1
    pub smoke: f32,
    /// black smoke rather than oil-white
    pub smoke_black: bool,
    pub fatigue: f32,
    pub flutter: f32,
    pub flap_stress: f32,
}

impl Damage {
    pub fn new(spec: &'static Spec) -> Self {
        Self {
            spec,
            wing: [spec.span; 2],
            wing_tear: [Tear::NONE; 2],
            fold: [Fold::default(); 2],
            tail: Tear::NONE,
            hull: 0.0,
            aileron: [1.0; 2],
            flap: [1.0; 2],
            elevator: [1.0; 2],
            stab: [1.0; 2],
            rudder: 1.0,
            fin: 1.0,
            prop: 1.0,
            engine: 1.0,
            gear: [true; MAX_GEAR],
            oil: 0.0,
            crack: [0.0; 3],
            scuff: [0.0; N_GROUPS],
            vibe: 0.0,
            smoke: 0.0,
            smoke_black: false,
            fatigue: 0.0,
            flutter: 0.0,
            flap_stress: 0.0,
        }
    }
}

fn sgn(side: usize) -> f32 {
    if side == 0 { -1.0 } else { 1.0 }
}
/// The break through one or both halves of a tail surface that lives at |x| > d.
fn halves(h: [f32; 2], d: f32) -> Tear {
    match (h[0] < 0.5, h[1] < 0.5) {
        (true, true) => Tear::new(Vec3::X, d, 2.0),
        (true, false) => Tear::new(Vec3::NEG_X, d, 1.0),
        (false, true) => Tear::new(Vec3::X, d, 1.0),
        _ => Tear::NONE,
    }
}

impl Damage {
    /// Pieces that are no longer there at all.
    pub fn hidden(&self, p: &Piece) -> bool {
        if self.tail.is_some() && p.group == G_TAIL && p.role != Role::TailCone {
            return true;
        }
        match p.role {
            Role::Aileron(s) => self.aileron[s as usize] <= 0.0,
            Role::Flap(s) => self.flap[s as usize] <= 0.0,
            Role::Rudder => self.rudder <= 0.0,
            Role::Prop => self.prop <= 0.0,
            Role::Gear(i) => !self.gear[i as usize],
            // Internal structure only shows once the skin over it is torn.
            Role::WingStructure(s) => self.wing[s as usize] >= self.spec.span,
            Role::TailStructure => self.stab[0] > 0.5 && self.stab[1] > 0.5 && self.fin >= 1.0,
            _ => false,
        }
    }

    /// The breaks that cut a piece as it is drawn on the aircraft.
    pub fn tears_for(&self, p: &Piece) -> [Tear; 2] {
        let mut t = [Tear::NONE; 2];
        let tl = &self.spec.tail;
        let structure = matches!(p.role, Role::WingStructure(_) | Role::TailStructure);
        // Spars poke out a little past the torn skin.
        let stick = if structure { 0.12 * self.spec.span / 5.5 } else { 0.0 };
        match p.group {
            G_LWING | G_RWING => {
                let s = if p.group == G_RWING { 1 } else { 0 };
                t[0] = self.stump(s).grow(stick);
            }
            G_TAIL if self.tail.is_some() => {
                if p.role == Role::TailCone {
                    t[0] = self.tail;
                }
            }
            G_TAIL => match p.role {
                Role::Elevator => t[0] = halves(self.elevator, tl.elev_root),
                Role::Stab | Role::TailStructure => {
                    t[0] = halves(self.stab, tl.stab_root + stick);
                    if self.fin < 1.0 {
                        t[1] = Tear::new(Vec3::Y, tl.fin_y0 + tl.fin_h * self.fin + stick, 1.0);
                    }
                }
                Role::Rudder => {
                    if self.rudder < 1.0 {
                        t[0] = Tear::new(Vec3::Y, tl.rudder_y0 + tl.rudder_h * self.rudder, 1.0);
                    }
                }
                _ => {}
            },
            _ => {
                if let (Role::Prop, crate::aircraft::Engine::Piston(pe)) = (p.role, &self.spec.engine) {
                    if self.prop > 0.0 && self.prop < 1.0 {
                        t[0] = Tear::new(Vec3::Y, pe.blade_break_y, 1.0);
                    }
                }
            }
        }
        t
    }

    /// The break a wing stump ends in (NONE: the whole wing).
    pub fn stump(&self, s: usize) -> Tear {
        if self.wing_tear[s].is_some() {
            self.wing_tear[s]
        } else if self.wing[s] < self.spec.span {
            Tear::new(Vec3::new(sgn(s), 0.0, 0.0), self.wing[s], 1.0)
        } else {
            Tear::NONE
        }
    }
    /// How ragged the edges drawn on a piece are.
    pub fn jag_for(&self, p: &Piece) -> f32 {
        match p.group {
            G_LWING | G_RWING => {
                let t = self.stump(if p.group == G_RWING { 1 } else { 0 });
                if t.jag > 0.0 { t.jag } else { JAG }
            }
            G_TAIL if self.tail.is_some() => self.tail.jag,
            _ => JAG,
        }
    }

    /// Wing area outboard of span station `b` on one side (m²).
    pub fn wing_lost(&self, b: f32) -> f32 {
        let span = self.spec.span;
        let b = b.clamp(0.0, span);
        let n = 16;
        (0..n).map(|k| (self.spec.wing_sect)(b + (span - b) * (k as f32 + 0.5) / n as f32).c).sum::<f32>() * (span - b) / n as f32
    }
    /// How much of the ailerons is still on a wing that ends at `b`.
    pub fn aileron_frac(&self, b: f32) -> f32 {
        let [a0, a1] = self.spec.aileron;
        ((b - a0) / (a1 - a0)).clamp(0.0, 1.0)
    }

    /// What is wrong, one line each, for the HUD.
    pub fn report(&self) -> Vec<String> {
        let mut r = Vec::new();
        for s in 0..2 {
            let lr = if s == 0 { "L" } else { "R" };
            if self.fold[s].on {
                r.push(format!("{lr} WING BUCKLED, HANGING"));
            } else if self.wing[s] < self.spec.span {
                r.push(format!("{lr} WING  -{:.1} m", self.spec.span - self.wing[s]));
            }
            if self.wing[s] < self.spec.tank[1] {
                r.push(format!("{lr} TANK RUPTURED"));
            }
            if self.aileron[s] <= 0.0 {
                r.push(format!("{lr} AILERON GONE"));
            } else if self.aileron[s] < 1.0 {
                r.push(format!("{lr} AILERON HANGING"));
            }
            if self.flap[s] <= 0.0 {
                r.push(format!("{lr} FLAP GONE"));
            }
            if self.stab[s] < 0.5 {
                r.push(format!("{lr} STABILISER GONE"));
            } else if self.elevator[s] < 0.5 {
                r.push(format!("{lr} ELEVATOR GONE"));
            }
        }
        if self.tail.is_some() {
            r.push("TAIL BROKEN OFF".into());
        } else if self.fin < 1.0 {
            r.push(format!("FIN {:.0}%", self.fin * 100.0));
        }
        if self.tail.is_some() {
        } else if self.rudder <= 0.0 {
            r.push("RUDDER GONE".into());
        } else if self.rudder < 1.0 {
            r.push(format!("RUDDER {:.0}%", self.rudder * 100.0));
        }
        let jet = self.spec.is_jet();
        if self.prop <= 0.0 && !jet {
            r.push("PROPELLER GONE".into());
        } else if self.prop < 1.0 && !jet {
            r.push("PROP BLADE BROKEN".into());
        }
        if self.engine <= 0.0 {
            r.push(if jet { "ENGINES DEAD" } else { "ENGINE DEAD" }.into());
        } else if self.engine < 1.0 {
            r.push(if jet { format!("ENGINES {:.0}%", self.engine * 100.0) } else { format!("ENGINE {:.0}%  OIL LEAK", self.engine * 100.0) });
        }
        for (g, leg) in self.gear.iter().zip(self.spec.gear) {
            if !g {
                r.push(format!("{} GONE", leg.name));
            }
        }
        if self.oil > 0.25 {
            r.push("OIL ON WINDSHIELD".into());
        }
        if self.crack[2] > 0.0 {
            r.push("WINDSHIELD CRACKED".into());
        }
        if self.hull > 0.5 {
            r.push("FUSELAGE CRUMPLED".into());
        } else if self.hull > 0.15 {
            r.push("FUSELAGE DENTED".into());
        }
        r
    }

    /// Fill in from a spec like `rwing=4.3,laileron=0,rudder=0.5,prop=0.5,engine=0.4,oil=0.6,crack=1,nosegear=0`
    /// (for testing from the command line).
    pub fn apply_spec(&mut self, spec: &str) {
        for item in spec.split(',') {
            let (k, v) = item.split_once('=').unwrap_or((item, "0"));
            let v: f32 = v.trim().parse().unwrap_or(0.0);
            match k.trim() {
                "lwing" => self.wing[0] = v,
                "rwing" => self.wing[1] = v,
                "laileron" => self.aileron[0] = v,
                "raileron" => self.aileron[1] = v,
                "lflap" => self.flap[0] = v,
                "rflap" => self.flap[1] = v,
                "lelevator" => self.elevator[0] = v,
                "relevator" => self.elevator[1] = v,
                "lstab" => self.stab[0] = v,
                "rstab" => self.stab[1] = v,
                "rudder" => self.rudder = v,
                "fin" => self.fin = v,
                "prop" => self.prop = v,
                "engine" => self.engine = v,
                "nosegear" => self.gear[0] = v > 0.5,
                "lgear" => self.gear[1] = v > 0.5,
                "rgear" => self.gear[2] = v > 0.5,
                "lbodygear" => self.gear[3] = v > 0.5,
                "rbodygear" => self.gear[4] = v > 0.5,
                "oil" => self.oil = v,
                "crack" => self.crack = [-1.45, 0.45, 0.22 * v],
                "scuff" => self.scuff = [v; N_GROUPS],
                // An angled break at that station, or a buckled wing hanging there
                // (random shape, seeded by the station), or the tail broken off at z.
                "lsnap" | "rsnap" | "lfold" | "rfold" => {
                    let s = if k.trim().starts_with('l') { 0 } else { 1 };
                    let mut rng = Rng(v.to_bits() as u64 * 7919 + 1 + s as u64);
                    let ks = self.spec.span / 5.5;
                    let tear = wing_break(self.spec, s, v, rng.range(-0.5, 0.5) * ks, rng.range(-0.2, 0.2), rng.range(0.06, 0.3) * ks.sqrt(), &Tear::NONE);
                    self.wing[s] = v;
                    self.wing_tear[s] = tear;
                    if k.trim().ends_with("fold") {
                        self.fold[s] = fold_at(self.spec, s, v, tear, Tear::NONE, self.spec.span, rng.f() < 0.5, 2500.0 * ks * ks * ks);
                    }
                }
                "tail" => self.tail = tail_break(self.spec, v, 0.1, -0.1, 0.2),
                other => log::warn!("unknown damage item {other}"),
            }
        }
    }
}

/// A point on wing `s` (body frame): span station, fraction of the chord back
/// from the leading edge, and which skin (+1 upper, -1 lower, 0 the mean line).
pub fn wing_pt(spec: &Spec, s: usize, station: f32, xc: f32, skin: f32) -> Vec3 {
    let k = (spec.wing_sect)(station.clamp(0.0, spec.span));
    let half = 0.06 * k.c * k.tf * (1.0 - (2.0 * xc - 0.6).powi(2)).max(0.15);
    let p = k.le + k.cdir * (k.c * xc) + k.up * (half * skin + 0.02 * k.c);
    Vec3::new(p.x * sgn(s), p.y, p.z)
}

/// An angled break across wing `s` near span `station`: `tz` slants it
/// across the chord (a diagonal snap), `ty` through the thickness, `jag` is
/// how ragged it is. It is kept wholly inboard of the break `prev` the wing
/// already ends in (straightened if it has to be).
pub fn wing_break(spec: &Spec, s: usize, station: f32, tz: f32, ty: f32, jag: f32, prev: &Tear) -> Tear {
    let sg = sgn(s);
    let k = (spec.wing_sect)(station.clamp(0.0, spec.span));
    let mid = wing_pt(spec, 1, station, 0.45, 0.0);
    let sc = spec.span / 5.5;
    let make = |tz: f32| {
        let n = Vec3::new(sg, ty, tz).normalize();
        let p = Vec3::new(sg * station, mid.y, mid.z);
        Tear { n, d: n.dot(p), mode: 1.0, jag }
    };
    let mut tz = tz;
    for _ in 0..5 {
        let t = make(tz);
        // Where it crosses the leading and trailing edges must still be on the wing.
        let fits = [k.le.z - 0.05 * sc, k.le.z + k.c + 0.05 * sc].iter().all(|&z| {
            let x = (t.d - t.n.y * mid.y - t.n.z * z) / t.n.x;
            prev.keeps(Vec3::new(x + sg * 0.2 * sc, mid.y, z))
        });
        if fits {
            return t;
        }
        tz *= 0.5;
    }
    make(0.0)
}

/// A buckled wing hanging from the stump at `tear` (span `root`), out to `tip`
/// where it ends in `outer`: hinged on its upper skin if it folds `up`, on
/// its lower skin if it folds down.
pub fn fold_at(spec: &Spec, s: usize, root: f32, tear: Tear, outer: Tear, tip: f32, up: bool, hold: f32) -> Fold {
    let q = wing_pt(spec, s, root, 0.4, if up { 1.0 } else { -1.0 });
    let x = (tear.d - tear.n.y * q.y - tear.n.z * q.z) / tear.n.x;
    let pivot = Vec3::new(x, q.y, q.z);
    let axis = tear.n.cross(Vec3::Y).normalize();
    Fold { on: true, pivot, axis, ang: 0.0, rate: if up { 0.6 } else { -0.6 }, tear, outer, root, tip, stress: 0.0, hold }
}

/// The tail broken off through the tail cone at `z`, slanted by `tx`, `ty`.
pub fn tail_break(spec: &Spec, z: f32, tx: f32, ty: f32, jag: f32) -> Tear {
    let n = Vec3::new(tx, ty, 1.0).normalize();
    let p = Vec3::new(0.0, (spec.fuse)(z).1, z);
    Tear { n, d: n.dot(p), mode: 1.0, jag }
}

/// Which pieces a shed request refers to.
#[derive(Clone, Copy, Debug)]
pub enum Sel {
    Group(usize),
    /// the group with its internal structure (a wing torn off shows its spars)
    GroupAll(usize),
    Role(Role),
    Roles(&'static [Role]),
}

/// Something has just come off: the flight model asks the game to let it go.
#[derive(Clone, Copy, Debug)]
pub struct ShedReq {
    pub sel: Sel,
    /// the break, seen from the side that leaves (NONE: the whole piece goes),
    /// and the part's other end if it already had one
    pub tear: Tear,
    pub tear2: Tear,
    /// extra velocity of the part, body frame
    pub kick: Vec3,
    pub spin: f32,
    /// all its pieces go as one body (a wing with its flap and aileron)
    pub rigid: bool,
    /// where the pieces were relative to their usual pose (a wing that had folded)
    pub pre: Mat4,
    /// `tear2` is given; otherwise each piece keeps the first break it already had
    pub own2: bool,
}

impl ShedReq {
    pub fn new(sel: Sel, tear: Tear, kick: Vec3, spin: f32) -> Self {
        Self { sel, tear, tear2: Tear::NONE, kick, spin, rigid: false, pre: Mat4::IDENTITY, own2: false }
    }
}

/// A part that has left the aircraft and is on its way down.
pub struct Shed {
    pub mesh: usize,
    /// the piece's pose at the moment it left
    local: Mat4,
    /// the point (after `local`) that `pos` tracks
    origin: Vec3,
    pos: DVec3,
    vel: DVec3,
    rot: DQuat,
    omega: DVec3,
    tear: [Tear; 2],
    jag: f32,
    cut: bool,
    radius: f64,
    span: f32,
    scuff: f32,
    seed: f32,
    /// which aircraft it came off (its paint, see plane.wgsl), and its `Spec::size_k`
    kind: f32,
    size: f32,
    age: f32,
    bounces: u8,
    resting: bool,
    sunk: bool,
}

#[derive(Default)]
pub struct Sheds {
    pub items: Vec<Shed>,
}

impl Sheds {
    /// Spawn whatever `req` tears off: every piece it names, cut along the tear.
    #[allow(clippy::too_many_arguments)]
    pub fn spawn(&mut self, req: &ShedReq, dmg: &Damage, pieces: &[Piece], piece_mesh: &[usize], piece_cut: &[bool], pose: &[Mat4], pos: DVec3, rot: DQuat, vel: DVec3, rng: &mut Rng, emit: &mut Vec<(DVec3, Particle)>) {
        // The pieces that go, each with its pose and the breaks it is cut along.
        let mut picked: Vec<(usize, Mat4, [Tear; 2], Vec3, Vec3, Vec3, usize)> = Vec::new();
        for (i, p) in pieces.iter().enumerate() {
            let structure = matches!(p.role, Role::WingStructure(_) | Role::TailStructure);
            if p.glass {
                continue;
            }
            let take = match req.sel {
                Sel::Group(g) => p.group == g && !structure,
                Sel::GroupAll(g) => p.group == g,
                Sel::Role(r) => p.role == r,
                Sel::Roles(rs) => rs.contains(&p.role),
            };
            // Structure counts as there whenever its skin is being torn.
            if !take || (dmg.hidden(p) && !(structure && matches!(req.sel, Sel::GroupAll(_)))) {
                continue;
            }
            let stick = if structure { 0.12 * dmg.spec.span / 5.5 } else { 0.0 };
            let second = if req.own2 { req.tear2.grow(stick) } else { dmg.tears_for(p)[0] };
            let tears = [req.tear.grow(stick), second];
            // Where the leaving part is: the mean of its vertices, body frame.
            let (mut c, mut n, mut lo, mut hi) = (Vec3::ZERO, 0usize, Vec3::splat(1e9), Vec3::splat(-1e9));
            let local = req.pre * pose[i];
            for v in &p.mesh.v {
                let q = Vec3::from(v.pos);
                if tears.iter().all(|t| t.keeps(q)) {
                    let w = local.transform_point3(q);
                    c += w;
                    n += 1;
                    lo = lo.min(w);
                    hi = hi.max(w);
                }
            }
            if n < 3 {
                continue;
            }
            picked.push((i, local, tears, c / n as f32, lo, hi, n));
        }
        if picked.is_empty() {
            return;
        }
        // One body: shared centre, motion and size, so its pieces stay together.
        let whole = if req.rigid {
            let n: usize = picked.iter().map(|k| k.6).sum();
            let c = picked.iter().map(|k| k.3 * k.6 as f32).sum::<Vec3>() / n as f32;
            let lo = picked.iter().fold(Vec3::splat(1e9), |a, k| a.min(k.4));
            let hi = picked.iter().fold(Vec3::splat(-1e9), |a, k| a.max(k.5));
            Some((c, hi - lo, rng.dir() * 1.5, rng.dir().as_dvec3() * (req.spin * rng.range(0.6, 1.3)) as f64, rng.f()))
        } else {
            None
        };
        let mut first = true;
        for (i, local, tears, c, lo, hi, _) in picked {
            let p = &pieces[i];
            let (origin, ext, jitter, omega, seed) = whole.unwrap_or_else(|| (c, hi - lo, rng.dir() * 1.5, rng.dir().as_dvec3() * (req.spin * rng.range(0.6, 1.3)) as f64, rng.f()));
            let wpos = pos + rot * origin.as_dvec3();
            let kick = rot * (req.kick + jitter).as_dvec3();
            let jag = [tears[0].jag, tears[1].jag].into_iter().fold(0.0f32, f32::max);
            self.items.push(Shed {
                mesh: piece_mesh[i],
                local,
                origin,
                pos: wpos,
                vel: vel * 0.97 + kick,
                rot,
                omega,
                tear: tears,
                jag: if jag > 0.0 { jag } else { JAG },
                cut: piece_cut[i] || tears.iter().any(|t| t.is_some()),
                radius: (ext.max_element() * 0.5).clamp(0.1, 3.0 * dmg.spec.span / 5.5) as f64,
                span: ext.max_element(),
                scuff: dmg.scuff[p.group].max(0.4),
                seed,
                kind: dmg.spec.id as f32,
                size: dmg.spec.size_k,
                age: 0.0,
                bounces: 0,
                resting: false,
                sunk: false,
            });
            if first {
                first = false;
                // A spray of rivets and shreds of skin where it let go.
                let up = wpos.normalize().as_vec3();
                for _ in 0..18 {
                    let d = rng.dir();
                    emit.push((wpos, Particle { vel: (vel.as_vec3() * 0.6 + d * rng.range(2.0, 9.0) + up * 2.0).into(), life: rng.range(0.4, 1.3), size0: 0.04, size1: 0.02, kind: SPARK, drag: 0.8, seed: rng.f(), alpha: 0.9, ..Default::default() }));
                }
                for _ in 0..6 {
                    let d = rng.dir();
                    emit.push((wpos, Particle { vel: (vel.as_vec3() * 0.5 + d * rng.range(1.0, 4.0)).into(), life: rng.range(1.5, 3.0), size0: 0.4, size1: 1.6, temp: 0.1, kind: GRIT, drag: 1.6, buoy: 0.3, seed: rng.f(), alpha: 0.3, ..Default::default() }));
                }
            }
        }
        if self.items.len() > 200 {
            self.items.drain(0..50);
        }
    }

    pub fn update(&mut self, dt: f32, world: &World) {
        let dtf = dt as f64;
        for s in self.items.iter_mut() {
            s.age += dt;
            if s.resting || s.sunk {
                continue;
            }
            let upw = s.pos.normalize();
            // Torn panels tumble: a lot of drag, more when they turn flat to the
            // airflow, and a sideways skitter as they do.
            let flat = 0.5 + 0.5 * (s.age as f64 * s.omega.length() * 0.7 + s.seed as f64 * 6.0).sin();
            let drag = (0.12 + 0.9 * flat) * (s.span as f64).sqrt() * 0.7;
            s.vel -= upw * 9.81 * dtf;
            s.vel *= (-dtf * drag).exp();
            let skitter = s.rot * DVec3::new((s.age as f64 * 3.1 + s.seed as f64 * 9.0).sin(), 0.0, (s.age as f64 * 2.3).cos()) * (flat * 2.5 * dtf);
            s.vel += skitter;
            s.pos += s.vel * dtf;
            let w = s.omega * dtf;
            if w.length_squared() > 1e-12 {
                s.rot = (DQuat::from_scaled_axis(w) * s.rot).normalize();
            }
            s.omega *= (-dtf * 0.12).exp();
            let (lat, lon, h) = geo::pos_to_lla(s.pos);
            let Some(g) = world.ground(lat, lon) else { continue };
            let mut floor = (g.h, g.water);
            if let Some(b) = world.building(lat, lon) {
                if h < b.top + 1.0 && (!b.bridge || h > b.base) {
                    floor = (b.top, false);
                }
            }
            let r = s.radius * 0.3;
            if h < floor.0 + r {
                if floor.1 {
                    s.sunk = true;
                    continue;
                }
                s.pos += upw * (floor.0 + r - h);
                let vn = s.vel.dot(upw);
                let tang = s.vel - upw * vn;
                if vn < -2.0 && s.bounces < 2 {
                    s.bounces += 1;
                    s.vel = tang * 0.5 - upw * vn * 0.2;
                    s.omega *= 0.5;
                } else if tang.length() > 1.0 && s.bounces < 6 {
                    s.bounces += 1;
                    s.vel = tang * 0.7;
                    s.omega *= 0.7;
                } else {
                    s.resting = true;
                    s.vel = DVec3::ZERO;
                    s.omega = DVec3::ZERO;
                }
            }
        }
        self.items.retain(|s| !s.sunk && s.age < 300.0);
    }

    pub fn draw(&self, cam: DVec3, params: [[f32; 4]; 6], objs: &mut Vec<DrawObj>) {
        for s in &self.items {
            let o = s.pos - cam;
            if o.length_squared() > 3000.0 * 3000.0 {
                continue;
            }
            let model = Mat4::from_translation(o.as_vec3()) * Mat4::from_quat(s.rot.as_quat()) * Mat4::from_translation(-s.origin) * s.local;
            objs.push(DrawObj {
                mesh: s.mesh,
                model,
                glow: [0.25, s.kind, s.size, s.seed],
                params,
                cast: 0b11,
                cut: s.cut,
                tear: [s.tear[0].gpu(), s.tear[1].gpu()],
                dmg: [s.tear[0].mode, s.tear[1].mode, s.scuff, s.jag],
                ..DrawObj::DEFAULT
            });
        }
    }
}

/// Which breakable group a hard point of the airframe belongs to (see `plane::HARD`).
pub fn hard_group(hi: usize) -> usize {
    match hi {
        0 | 6 => G_ENGINE,
        1 => G_LWING,
        2 => G_RWING,
        3 | 4 => G_TAIL,
        _ => G_FUSE,
    }
}
