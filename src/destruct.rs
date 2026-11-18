//! Things the aircraft (and its explosion) can break: trees topple, buildings
//! come down as rubble, the ground is cratered and throws out rock.
//!
//! A building is cut into blocks along its own grain, one or a few storeys
//! high. Low buildings crumble in a wave spreading from the point of impact.
//! Tall ones fail the way towers do: the floors at the impact are blown out,
//! the part above leans towards the hole and then drops as one block, and a
//! crush front runs down the building, throwing facade and dust out at every
//! floor, until the block reaches the ground and breaks up on the pile.
//! Rubble heaps up on a coarse height field, so piles have slopes, and heavy
//! pieces falling on a smaller building next door can bring that down too.
use crate::cars::{M_CUBE, M_REBAR, M_ROCK, M_SLAB, N_REBAR};
use crate::crash::{Rng, DUST, GLASS, GRIT, LEAF};
use crate::geo::{self, TileId};
use crate::render::{Inst, MeshVertex, Particle, TreeInst};
use crate::skin::{Group, PieceRef, NO_SKIN};
use crate::tiles3d::SkinSrc;
use crate::world::{TreeRef, World, Wreck};
use glam::{DQuat, DVec3, Mat3, Vec2, Vec3};
use std::collections::HashMap;

/// How many shapes of broken concrete there are (meshes from `M_SLAB` on).
pub const N_SLAB: u32 = 12;

fn is_rock(mesh: u32) -> bool {
    (M_ROCK..M_SLAB).contains(&mesh)
}

fn is_slab(mesh: u32) -> bool {
    (M_SLAB..M_SLAB + N_SLAB).contains(&mesh)
}

/// A broken edge of a slab shape (unit square in local x, z): the middle of
/// the side it broke along, its outward normal and half length, where rebar
/// sticks out.
#[derive(Clone, Copy)]
struct Edge {
    mid: Vec2,
    n: Vec2,
    half: f32,
}

/// A flat piece of broken concrete, thin along local y: a floor slab or a
/// wall panel. Its outline is the unit square in x, z with one to three sides
/// (and sometimes a corner) broken off along a ragged line, and the broken
/// edges crumbled thinner. Faces of the break get col.a = 1; the rest keep 0,
/// so the outer face of a piece of facade (+y) can still show the wall.
fn slab_mesh(seed: u32) -> (Vec<MeshVertex>, Vec<u32>, Vec<Edge>) {
    let mut r = Rng(0x9E37_79B9_7F4A_7C15 ^ (seed as u64 + 1).wrapping_mul(0x2545_F491_4F6C_DD1D) | 1);
    for _ in 0..4 {
        r.f();
    }
    const MASKS: [u8; 12] = [0b0001, 0b0011, 0b0111, 0b0101, 0b1011, 0b0010, 0b0110, 0b1101, 0b1001, 0b1110, 0b0100, 0b1111];
    let mask = MASKS[seed as usize % MASKS.len()];
    let corners = [Vec2::new(-1.0, -1.0), Vec2::new(1.0, -1.0), Vec2::new(1.0, 1.0), Vec2::new(-1.0, 1.0)];
    // Outline (counter-clockwise in x, z): each point, whether the edge from
    // it on is broken, and how much it is crumbled.
    let mut pts: Vec<(Vec2, bool, f32)> = Vec::new();
    let mut edges = Vec::new();
    // On some a corner is knocked off: a ragged line across it instead.
    let cut = if seed % 3 == 1 { Some(((seed / 3) as usize % 4, r.range(0.5, 1.3), r.range(0.5, 1.3))) } else { None };
    // A ragged line from p towards q (q itself not included), dented inwards.
    let ragged = |r: &mut Rng, pts: &mut Vec<(Vec2, bool, f32)>, p: Vec2, q: Vec2, n: Vec2, deep: f32| {
        let m = 3 + (r.f() * 3.0) as usize + ((q - p).length() > 1.2) as usize;
        for k in 1..m {
            let t = (k as f32 + r.range(-0.3, 0.3)) / m as f32;
            pts.push((p + (q - p) * t - n * r.range(0.0, deep), true, r.range(0.2, 0.65)));
        }
    };
    for s in 0..4 {
        let (a, b) = (corners[s], corners[(s + 1) % 4]);
        let d = (b - a) * 0.5;
        let n = -d.perp();
        let broken = mask & (1 << s) != 0;
        let prev_broken = mask & (1 << ((s + 3) % 4)) != 0;
        let start = match cut {
            Some((c, _, kb)) if c == s => a + d * kb,
            // A corner where two breaks meet is knocked in a little.
            _ if broken && prev_broken => a * (1.0 - r.range(0.0, 0.14)),
            _ => a,
        };
        let end = match cut {
            Some((c, ka, _)) if c == (s + 1) % 4 => b - d * ka,
            _ => b,
        };
        let crumb = if broken || prev_broken || matches!(cut, Some((c, _, _)) if c == s) { r.range(0.2, 0.5) } else { 0.0 };
        pts.push((start, broken, crumb));
        if broken {
            ragged(&mut r, &mut pts, start, end, n, 0.24);
            edges.push(Edge { mid: (start + end) * 0.5, n, half: (end - start).length() * 0.5 });
        }
        if let Some((c, _, kb)) = cut {
            if c == (s + 1) % 4 {
                let dn = (corners[(s + 2) % 4] - b) * 0.5;
                let q = b + dn * kb;
                let nc = -(q - end).normalize().perp();
                pts.push((end, true, 0.3));
                ragged(&mut r, &mut pts, end, q, nc, 0.16);
                edges.push(Edge { mid: (end + q) * 0.5, n: nc, half: (q - end).length() * 0.5 });
            }
        }
    }
    let (mut v, mut idx) = (Vec::new(), Vec::new());
    let mut tri = |a: Vec3, b: Vec3, c: Vec3, out: Vec3, col: [f32; 4]| {
        let mut nrm = (b - a).cross(c - a).normalize_or_zero();
        let (b, c) = if nrm.dot(out) < 0.0 {
            nrm = -nrm;
            (c, b)
        } else {
            (b, c)
        };
        let base = v.len() as u32;
        for q in [a, b, c] {
            v.push(MeshVertex { pos: q.into(), nrm: nrm.into(), col });
        }
        idx.extend([base, base + 1, base + 2]);
    };
    let n = pts.len();
    let top = |p: &(Vec2, bool, f32)| Vec3::new(p.0.x, 1.0 - p.2 * 1.1, p.0.y);
    let bot = |p: &(Vec2, bool, f32)| Vec3::new(p.0.x, -1.0 + p.2 * 0.7, p.0.y);
    let (ct, cb) = (Vec3::Y, -Vec3::Y);
    for i in 0..n {
        let (p, q) = (&pts[i], &pts[(i + 1) % n]);
        tri(ct, top(p), top(q), Vec3::Y, [1.0, 1.0, 1.0, 0.0]);
        tri(cb, bot(p), bot(q), -Vec3::Y, [1.0, 1.0, 1.0, 0.0]);
        let out = (p.0 + q.0).extend(0.0);
        let out = Vec3::new(out.x, 0.0, out.y);
        let col = if p.1 {
            let sh = r.range(0.7, 0.95);
            [sh, sh, sh * 0.97, 1.0]
        } else {
            [1.0, 1.0, 1.0, 0.0]
        };
        tri(top(p), bot(p), bot(q), out, col);
        tri(top(p), bot(q), top(q), out, col);
    }
    (v, idx, edges)
}

pub fn slab_meshes() -> Vec<(Vec<MeshVertex>, Vec<u32>)> {
    (0..N_SLAB).map(|s| {
        let (v, i, _) = slab_mesh(s);
        (v, i)
    }).collect()
}

/// Bars left sticking out of a broken edge: two layers of rods along +x from
/// inside the concrete, spaced along z over -1..1, bent and some snapped
/// short (to be drawn at one even scale, so the rods stay round).
fn rebar_mesh(seed: u32) -> (Vec<MeshVertex>, Vec<u32>) {
    let mut r = Rng(0x51_7CC1_B727_220A ^ (seed as u64 + 7).wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
    for _ in 0..4 {
        r.f();
    }
    let (mut v, mut idx) = (Vec::new(), Vec::new());
    let col = [1.0, 1.0, 1.0, 2.0];
    let rad = 0.02;
    for layer in [-0.045f32, 0.045] {
        for j in 0..10 {
            if r.f() < 0.3 {
                continue;
            }
            let z = -0.9 + 0.2 * j as f32 + r.range(-0.04, 0.04);
            let len = if r.f() < 0.25 { r.range(0.08, 0.2) } else { r.range(0.25, 0.75) };
            let droop = r.range(-0.6, 0.15) * len;
            let side = r.range(-0.25, 0.25) * len;
            let path = [
                Vec3::new(-0.35, layer, z),
                Vec3::new(r.range(0.0, 0.12), layer + droop * 0.1, z + side * 0.1),
                Vec3::new(len * 0.6, layer + droop * 0.45, z + side * 0.5),
                Vec3::new(len, layer + droop, z + side),
            ];
            // A three-sided tube along the path.
            let ring = |k: usize| {
                let d = if k + 1 < path.len() { path[k + 1] - path[k] } else { path[k] - path[k - 1] }.normalize();
                let a = d.any_orthonormal_vector();
                let b = d.cross(a);
                (0..3).map(|i| {
                    let t = i as f32 * std::f32::consts::TAU / 3.0;
                    let o = a * t.cos() + b * t.sin();
                    (path[k] + o * rad, o)
                }).collect::<Vec<_>>()
            };
            let rings: Vec<_> = (0..path.len()).map(ring).collect();
            for k in 0..path.len() - 1 {
                for i in 0..3 {
                    let i2 = (i + 1) % 3;
                    let (a, b, c, d) = (rings[k][i], rings[k][i2], rings[k + 1][i2], rings[k + 1][i]);
                    let base = v.len() as u32;
                    for (p, n) in [a, b, c, d] {
                        v.push(MeshVertex { pos: p.into(), nrm: n.into(), col });
                    }
                    idx.extend([base, base + 1, base + 2, base, base + 2, base + 3]);
                }
            }
        }
    }
    (v, idx)
}

pub fn rebar_meshes() -> Vec<(Vec<MeshVertex>, Vec<u32>)> {
    (0..crate::cars::N_REBAR).map(rebar_mesh).collect()
}

struct FallTree {
    pos: DVec3,
    mesh: u32,
    inst: TreeInst,
    dir: Vec2,
    ang: f32,
    vel: f32,
    delay: f32,
    rest: f32,
    height: f32,
    crown_r: f32,
    started: bool,
    landed: bool,
    done: bool,
}

/// What is under a falling piece: looked up again only once it has moved a
/// few metres, as the queries are not cheap and pieces are many.
#[derive(Clone, Copy)]
struct Under {
    at: DVec3,
    ground: f64,
    water: bool,
    /// roof of a standing building (or bridge deck) under it, and which one
    roof: f64,
    hit: Option<(TileId, usize, f32)>,
}

impl Under {
    const NONE: Under = Under { at: DVec3::ZERO, ground: 0.0, water: false, roof: -1e9, hit: None };
}

const NO_GROUP: u32 = u32::MAX;

#[derive(Clone)]
struct Chunk {
    pos: DVec3,
    half: Vec3,
    rot: DQuat,
    vel: DVec3,
    omega: DVec3,
    col: [f32; 3],
    mesh: u32,
    /// 1 bare rubble, 2 a piece of facade (shows windows)
    mat: f32,
    hold: f32,
    /// 0 still part of the standing building, 1 falling, 2 at rest
    state: u8,
    bounces: u8,
    /// the structure it belongs to while standing
    group: u32,
    /// part of the block above the impact, which moves as one
    upper: bool,
    home: DVec3,
    home_rot: DQuat,
    /// top of the block above the structure's base (m)
    z1: f32,
    /// distance from the point of impact
    dimp: f32,
    /// outward horizontal normal for pieces of the outer wall, else zero
    out: Vec3,
    /// a small fragment: no window, no pile slide, dropped first
    bit: bool,
    under: Under,
    /// which broken shape it takes once it lets go (standing, it is a plain box)
    var: u32,
    /// half its height while it stands (pieces of wall are thin along local y)
    vh: f32,
    /// how much building it stands for (m³), for how hard it hits and how
    /// much it adds to the heap
    mass: f32,
    /// reinforced concrete: bars stick out of its broken edges
    rebar: bool,
    /// a piece of facade: along-wall coordinate and height of its middle,
    /// height of the building (negative when local x runs against the wall's
    /// coordinate), packed style (inst.wgsl); zero for anything else
    face: [f32; 4],
    /// burst to nothing (taken out at the end of the step)
    gone: bool,
    /// the dark inside of the aircraft's hole: gone once the floors give
    void: bool,
    /// its slot in its structure's mesh from the 3D tiles (skin.rs), if it has one
    skin: u32,
    /// a piece of outer wall or roof
    outer: bool,
}

impl Chunk {
    /// A loose piece flying free.
    fn loose(pos: DVec3, half: Vec3, rot: DQuat, vel: DVec3, omega: DVec3, col: [f32; 3], mesh: u32) -> Chunk {
        Chunk {
            pos,
            half,
            rot,
            vel,
            omega,
            col,
            mesh,
            mat: 1.0,
            hold: 0.0,
            state: 1,
            bounces: 0,
            group: NO_GROUP,
            upper: false,
            home: pos,
            home_rot: rot,
            z1: 0.0,
            dimp: 0.0,
            out: Vec3::ZERO,
            bit: true,
            under: Under::NONE,
            var: 0,
            vh: half.y,
            mass: 8.0 * half.x * half.y * half.z,
            rebar: false,
            face: [0.0; 4],
            gone: false,
            void: false,
            skin: NO_SKIN,
            outer: false,
        }
    }
}

/// Facade glass knocked out (a bit of the style packed in `Chunk::face`).
const SMASHED: f32 = (128 * 256) as f32;

/// Size of the pieces a wall is cut into around the hole the aircraft makes.
const FINE: f32 = 0.32;

/// What went into a building, as the hole it leaves: the aircraft's attitude
/// and what it still had of its wings, stabiliser and fin.
#[derive(Clone, Copy)]
pub struct Hitter {
    pub right: DVec3,
    pub up: DVec3,
    /// span station each wing ends at (left, right); `Spec::span` = whole
    pub span: [f32; 2],
    /// the aircraft's head-on outline and full span
    pub shape: &'static crate::aircraft::Hole,
    pub full: f32,
    /// stabiliser halves (1 or 0) and how much of the fin is left
    pub stab: [f32; 2],
    pub fin: f32,
}

impl Hitter {
    /// Where a point lies seen along `dir`, the way the aircraft came: across
    /// and up on its lateral plane through `at`, and how far along `dir`.
    fn place(&self, p: DVec3, at: DVec3, dir: DVec3) -> (Vec2, f32) {
        let f = self.up.cross(self.right);
        let q = p - at;
        let df = dir.dot(f);
        let q2 = if df.abs() > 0.3 { q - dir * (q.dot(f) / df) } else { q - dir * q.dot(dir) };
        (Vec2::new(q2.dot(self.right) as f32, q2.dot(self.up) as f32), q.dot(dir) as f32)
    }

    /// Signed distance (m) from the aircraft's head-on outline, negative
    /// inside: fuselage, high wing, stabiliser and fin, a little larger than
    /// the aircraft (it tears more than its own shape on the way in).
    fn sdf(&self, p: Vec2) -> f32 {
        const K: f32 = 1.2;
        let h = self.shape;
        let (x, y) = (p.x / K, p.y / K);
        let ax = x.abs();
        let side = (x > 0.0) as usize;
        // Fuselage, engine and all: a rounded box, the widest part of the hole.
        let [fw, fh, fy, fr] = h.fus;
        let q = Vec2::new(ax, y - fy).abs() - Vec2::new(fw, fh) + Vec2::splat(fr);
        let fus = q.max(Vec2::ZERO).length() + q.x.max(q.y).min(0.0) - fr;
        // Wing: thin, a slight dihedral, tapering towards the tip, and any engine pods under it.
        let end = self.span[side];
        let [wy, dih, t0, t1] = h.wing;
        let mut wing = if end > 0.13 * self.full {
            let t = (ax / self.full).min(1.0);
            ((y - wy - dih * ax).abs() - (t0 + (t1 - t0) * t)).max(ax - end)
        } else {
            1e3
        };
        for [px, py, pr] in h.pods {
            if pr > 0.0 && px < end {
                wing = wing.min(Vec2::new(ax - px, y - py).length() - pr);
            }
        }
        let [sy, st, ss] = h.stab;
        let stab = if self.stab[side] > 0.5 { ((y - sy).abs() - st).max(ax - ss) } else { 1e3 };
        let [f0, fh2, fw0, fw1] = h.fin;
        let ft = f0 + fh2 * self.fin;
        let fin = if self.fin > 0.1 { (ax - (fw0 + (fw1 - fw0) * ((y - f0) / fh2).clamp(0.0, 1.0))).max((y - (f0 + ft) * 0.5).abs() - (ft - f0) * 0.5) } else { 1e3 };
        fus.min(wing).min(stab).min(fin) * K
    }
}

/// One building (with any parts standing on it) coming down.
struct Structure {
    /// centre of the footprint at the base, and up
    base: DVec3,
    u: DVec3,
    height: f32,
    ext: f32,
    tall: bool,
    /// height of the impact above the base: the block above it goes as one
    cut: f32,
    /// the hinge on the intact side the upper block leans on, and its axis
    pivot: DVec3,
    axis: DVec3,
    tilt: f32,
    tilt_v: f32,
    tilt_max: f32,
    q: DQuat,
    /// how far the upper block has come down, and how fast
    drop: f32,
    drop_v: f32,
    /// top of what still stands below the crush front
    top_lower: f32,
    /// seconds until the damaged floors give
    delay: f32,
    hole: f32,
    t: f32,
    gave: bool,
    grounded: bool,
    left: usize,
    /// seconds since the last piece let go (negative while some stand)
    done_t: f32,
    /// its own mesh from the 3D tiles, on its pieces (skin.rs)
    skin: Option<Group>,
}

pub struct Destruction {
    falling: Vec<FallTree>,
    chunks: Vec<Chunk>,
    structs: Vec<Structure>,
    rng: Rng,
    /// rubble height field: top altitude per 1.5 m cell on a local grid
    field: HashMap<(i32, i32), f32>,
    f_org: DVec3,
    f_e: DVec3,
    f_n: DVec3,
    crush_cool: f32,
    /// dust jets, glass and landing dust still allowed (a budget per second)
    fx_acc: [f32; 3],
    /// loudness of things currently collapsing, for sound
    pub rumble: f32,
    /// big structural moments (the floors giving, the block hitting the ground)
    pub booms: Vec<f32>,
    /// standing buildings hit hard enough by falling pieces to come down:
    /// (tile, footprint, where, velocity of what hit it)
    pub crush: Vec<(TileId, usize, DVec3, DVec3)>,
    /// the broken edges of each slab shape, where rebar sticks out
    edges: Vec<Vec<Edge>>,
}

const CELL: f64 = 1.5;
const MAX_CHUNKS: usize = 7000;

/// Smooth value noise in -1..1, about one bump per unit.
fn vnoise(p: Vec2, seed: u32) -> f32 {
    let h = |x: i32, y: i32| {
        let mut n = (x as u32).wrapping_mul(0x27d4_eb2d) ^ (y as u32).wrapping_mul(0x1656_67b1) ^ seed.wrapping_mul(0x9e37_79b9);
        n ^= n >> 15;
        n = n.wrapping_mul(0x2c1b_3c6d);
        n ^= n >> 12;
        (n & 0xffff) as f32 / 32767.5 - 1.0
    };
    let (i, f) = (p.floor(), p - p.floor());
    let (x, y) = (i.x as i32, i.y as i32);
    let w = f * f * (Vec2::splat(3.0) - 2.0 * f);
    let a = h(x, y) + (h(x + 1, y) - h(x, y)) * w.x;
    let b = h(x, y + 1) + (h(x + 1, y + 1) - h(x, y + 1)) * w.x;
    a + (b - a) * w.y
}

fn ring_area(r: &[Vec2]) -> f32 {
    r.iter().enumerate().map(|(i, p)| p.perp_dot(r[(i + 1) % r.len()])).sum::<f32>().abs() * 0.5
}

fn inside(ring: &[Vec2], p: Vec2) -> bool {
    let mut c = false;
    let mut j = ring.len() - 1;
    for i in 0..ring.len() {
        let (p0, p1) = (ring[i], ring[j]);
        if (p0.y > p.y) != (p1.y > p.y) && p.x < (p1.x - p0.x) * (p.y - p0.y) / (p1.y - p0.y) + p0.x {
            c = !c;
        }
        j = i;
    }
    c
}

impl Destruction {
    pub fn new(seed: u64) -> Self {
        Self {
            falling: Vec::new(),
            chunks: Vec::new(),
            structs: Vec::new(),
            rng: Rng(seed | 1),
            field: HashMap::new(),
            f_org: DVec3::ZERO,
            f_e: DVec3::X,
            f_n: DVec3::Y,
            crush_cool: 0.0,
            fx_acc: [0.0; 3],
            rumble: 0.0,
            booms: Vec::new(),
            crush: Vec::new(),
            edges: (0..N_SLAB).map(|s| slab_mesh(s).2).collect(),
        }
    }

    /// Knock one tree over. `push` is the direction it is shoved in.
    pub fn knock(&mut self, world: &mut World, queue: &wgpu::Queue, t: &TreeRef, push: DVec3, power: f32, delay: f32, scorch: f32) {
        world.remove_tree(queue, t);
        let (e, n, _) = geo::enu(t.pos);
        let mut d = Vec2::new(push.dot(e) as f32, push.dot(n) as f32);
        d = (d.normalize_or_zero() + Vec2::new(self.rng.range(-0.35, 0.35), self.rng.range(-0.35, 0.35))).normalize_or_zero();
        if d == Vec2::ZERO {
            d = Vec2::X;
        }
        let mut inst = t.inst;
        for c in inst.tint.iter_mut().take(3) {
            *c = (*c as f32 * (1.0 - 0.75 * scorch)) as u8;
        }
        // It comes to rest on its crown, not flat on the trunk.
        let rest = std::f32::consts::FRAC_PI_2 - ((t.crown_r * 0.55) / t.height.max(1.0)).atan() as f32 + self.rng.range(-0.05, 0.05);
        self.falling.push(FallTree { pos: t.pos, mesh: t.variant as u32 * crate::trees::LODS as u32, inst, dir: d, ang: 0.0, vel: 0.2 + power * self.rng.range(0.6, 1.1), delay, rest, height: t.height as f32, crown_r: t.crown_r as f32, started: false, landed: false, done: false });
        if self.falling.len() > 3600 {
            self.falling.drain(0..600);
        }
    }

    /// A blast: trees inside `gone` are shredded, the rest lean away and fall
    /// in a wave that spreads out from the centre.
    pub fn blast_trees(&mut self, world: &mut World, queue: &wgpu::Queue, center: DVec3, radius: f64, gone: f64, emit: &mut Vec<(DVec3, Particle)>) {
        let mut found = Vec::new();
        world.trees_near(center, radius, &mut found);
        for t in found {
            let d = (t.pos - center).length();
            if d < gone {
                world.remove_tree(queue, &t);
                self.leaves(t.pos + t.pos.normalize() * t.height * 0.4, t.crown_r as f32, 14, emit);
                continue;
            }
            let k = (1.0 - d / radius) as f32;
            if self.rng.f() > k.powf(0.6) * 1.15 {
                continue;
            }
            self.knock(world, queue, &t, t.pos - center, 0.3 + 1.5 * k, (d / 70.0) as f32, (k * 1.6 - 0.7).clamp(0.0, 1.0));
        }
    }

    fn leaves(&mut self, at: DVec3, spread: f32, n: usize, emit: &mut Vec<(DVec3, Particle)>) {
        let up = at.normalize().as_vec3();
        for _ in 0..n {
            let d = self.rng.dir();
            emit.push((at + (d * spread * self.rng.f()).as_dvec3(), Particle { vel: (d * self.rng.range(1.0, 7.0) + up * self.rng.range(0.0, 5.0)).into(), life: self.rng.range(2.5, 6.0), size0: self.rng.range(0.1, 0.22), size1: 0.12, kind: LEAF, drag: 1.4, seed: self.rng.f(), alpha: 1.0, ..Default::default() }));
        }
    }

    fn dust(&mut self, at: DVec3, size: f32, n: usize, kind: f32, emit: &mut Vec<(DVec3, Particle)>) {
        let up = at.normalize().as_vec3();
        for _ in 0..n {
            let d = self.rng.dir();
            let flat = (d - up * d.dot(up)).normalize_or_zero();
            emit.push((at + (d * size * 0.4).as_dvec3(), Particle { vel: (flat * self.rng.range(1.0, 6.0) + up * self.rng.range(0.3, 2.5)).into(), life: self.rng.range(3.0, 7.5), size0: size * self.rng.range(0.5, 0.9), size1: size * self.rng.range(2.2, 3.6), temp: 0.2, kind, drag: 1.4, buoy: 0.5, seed: self.rng.f(), alpha: self.rng.range(0.25, 0.45), ..Default::default() }));
        }
    }

    /// A jet of masonry dust blown out of a floor as it is crushed: a few
    /// overlapping puffs at different speeds, so it streams out as a plume.
    fn jet(&mut self, at: DVec3, out: Vec3, carry: Vec3, size: f32, emit: &mut Vec<(DVec3, Particle)>) {
        let up = at.normalize().as_vec3();
        let r = &mut self.rng;
        let d = (out + r.dir() * 0.3).normalize_or_zero();
        let sp = r.range(9.0, 20.0);
        for k in 0..3 {
            let f = 0.35 + 0.33 * k as f32;
            emit.push((at + (r.dir() * size * 0.5).as_dvec3(), Particle { vel: (d * sp * f + r.dir() * 1.5 + carry * 0.25 + up * r.range(-0.5, 1.5)).into(), life: r.range(7.0, 13.0), size0: size * r.range(0.6, 0.9), size1: size * r.range(3.0, 5.0), temp: 0.1, kind: GRIT, drag: r.range(0.7, 1.0), buoy: 0.25, seed: r.f(), alpha: r.range(0.14, 0.24), ..Default::default() }));
        }
    }

    /// Window glass bursting out of a facade.
    fn glass(&mut self, at: DVec3, spread: f32, out: Vec3, carry: Vec3, n: usize, emit: &mut Vec<(DVec3, Particle)>) {
        let up = at.normalize().as_vec3();
        for _ in 0..n {
            let r = &mut self.rng;
            let d = (out * 0.8 + r.dir()).normalize_or_zero();
            emit.push((at + (r.dir() * spread).as_dvec3(), Particle { vel: (d * r.range(2.0, 14.0) + carry + up * r.range(0.0, 4.0)).into(), life: r.range(4.0, 11.0), size0: r.range(0.12, 0.35), size1: r.range(0.1, 0.3), kind: GLASS, drag: r.range(0.15, 0.45), seed: r.f(), alpha: r.range(0.6, 1.0), ..Default::default() }));
        }
    }

    fn fkey(&self, p: DVec3) -> (i32, i32) {
        let d = p - self.f_org;
        ((d.dot(self.f_e) / CELL).floor() as i32, (d.dot(self.f_n) / CELL).floor() as i32)
    }

    /// Heap rubble up to `top` (altitude) over a disc around `p`.
    fn raise(&mut self, p: DVec3, rad: f32, top: f64) {
        let (kx, ky) = self.fkey(p);
        let n = (rad as f64 / CELL).ceil() as i32;
        let r2 = (rad as f64 / CELL + 0.5).powi(2);
        for dy in -n..=n {
            for dx in -n..=n {
                if (dx * dx + dy * dy) as f64 > r2 {
                    continue;
                }
                let h = self.field.entry((kx + dx, ky + dy)).or_insert(-1e9);
                *h = h.max(top as f32);
            }
        }
    }

    /// Where the camera should look while a tall building comes down:
    /// (middle of it, its height).
    pub fn focus(&self) -> Option<(DVec3, f64)> {
        self.structs.iter().filter(|s| s.tall && s.done_t < 6.0).max_by(|a, b| a.height.total_cmp(&b.height)).map(|s| (s.base + s.u * (s.height as f64 * 0.42), s.height as f64))
    }

    /// Bring a building down. `parts`: the building that was hit first, then
    /// any parts standing on it. `impact`/`vel`: where it was hit, how hard.
    /// `hit`: the aircraft itself went in here, so the wall shows its outline.
    /// `photo`: the building as the 3D tiles had it (cut out of them), which
    /// its pieces then carry.
    pub fn collapse(&mut self, parts: &[Wreck], impact: DVec3, vel: DVec3, hit: Option<Hitter>, photo: Option<&SkinSrc>, emit: &mut Vec<(DVec3, Particle)>) {
        let Some(w0) = parts.first() else { return };
        if w0.ring.len() < 3 {
            return;
        }
        let n = w0.ring.len() as f64;
        let c_hit = w0.ring.iter().fold(DVec3::ZERO, |a, p| a + *p / n);
        let (e, nn, u) = geo::enu(c_hit);
        // The middle of the whole structure (the part hit may be one of several).
        let (mut lo_c, mut hi_c) = (Vec2::splat(1e9), Vec2::splat(-1e9));
        for p in parts.iter().flat_map(|w| w.ring.iter()) {
            let q = Vec2::new((*p - c_hit).dot(e) as f32, (*p - c_hit).dot(nn) as f32);
            lo_c = lo_c.min(q);
            hi_c = hi_c.max(q);
        }
        let mc = (lo_c + hi_c) * 0.5;
        let centre0 = c_hit + e * mc.x as f64 + nn * mc.y as f64;
        let base_alt = parts.iter().map(|w| w.base).fold(f64::MAX, f64::min);
        let top_alt = parts.iter().map(|w| w.top).fold(f64::MIN, f64::max);
        let height = (top_alt - base_alt).max(1.0) as f32;
        let base = centre0 + u * (base_alt - w0.base);
        if self.field.is_empty() || (self.f_org - base).length() > 4000.0 {
            self.field.clear();
            self.f_org = base;
            self.f_e = e;
            self.f_n = nn;
        }
        let to2 = |p: DVec3| Vec2::new((p - base).dot(e) as f32, (p - base).dot(nn) as f32);
        let rings: Vec<Vec<Vec2>> = parts.iter().map(|w| w.ring.iter().map(|p| to2(*p)).collect()).collect();
        // Rubble lying on its roof (from a neighbour that came down first) falls with it.
        let on_roof = |p: DVec3| {
            let q = to2(p);
            (p - base).dot(u) > 1.0 && rings.iter().any(|r| r.len() >= 3 && inside(r, q))
        };
        for c in self.chunks.iter_mut().filter(|c| c.state == 2 && on_roof(c.pos)) {
            c.state = 1;
            c.bounces = 0;
            c.under = Under::NONE;
        }
        let (f_org, f_e, f_n) = (self.f_org, self.f_e, self.f_n);
        self.field.retain(|k, _| {
            let p = f_org + f_e * ((k.0 as f64 + 0.5) * CELL) + f_n * ((k.1 as f64 + 0.5) * CELL);
            let q = to2(p);
            !rings.iter().any(|r| r.len() >= 3 && inside(r, q))
        });
        // The pieces of each part, a tower's setbacks each a part of their own.
        struct Part {
            ring: Vec<Vec2>,
            base: f64,
            top: f64,
            wall: [f32; 3],
            roof: [f32; 3],
            /// facade style packed for inst.wgsl, building height, and the
            /// altitude the facade's height counts from
            style: f32,
            h: f32,
            gref: f64,
        }
        let lin = |c: u8| (c as f32 / 255.0).powf(2.2);
        let mut pg: Vec<Part> = Vec::new();
        for (w, ring) in parts.iter().zip(&rings) {
            if ring.len() < 3 {
                continue;
            }
            let style = (w.sty[0] as u32 + 256 * (w.sty[1] as u32 & 15) + 65536 * w.sty[2] as u32) as f32;
            let mk = |ring: Vec<Vec2>, base: f64, top: f64| Part { ring, base, top, wall: w.col.map(lin), roof: w.roof_col.map(lin), style, h: w.sty[3] as f32, gref: w.gref };
            let (mut r, mut z0) = (ring.clone(), w.base);
            for &(d, zt) in &w.tiers {
                let Some(next) = crate::buildings::inset(&r, d) else { break };
                pg.push(mk(r, z0, zt as f64));
                r = next;
                z0 = zt as f64;
            }
            pg.push(mk(r, z0, w.top));
        }
        let areas: Vec<f32> = pg.iter().map(|p| ring_area(&p.ring)).collect();
        // Cut along the building's own grain: the longest wall of its largest part.
        let big = (0..pg.len()).max_by(|a, b| areas[*a].total_cmp(&areas[*b])).unwrap_or(0);
        let mut a = Vec2::X;
        let mut best = 0.0;
        if let Some(rb) = pg.get(big).map(|p| &p.ring) {
            for i in 0..rb.len() {
                let d = rb[(i + 1) % rb.len()] - rb[i];
                if d.length() > best {
                    best = d.length();
                    a = d.normalize();
                }
            }
        }
        let b = a.perp();
        let vol: f32 = pg.iter().zip(&areas).map(|(p, ar)| ar * (p.top - p.base) as f32).sum();
        let tall = height >= 28.0 && vol > 6000.0;
        // A storey or a few per piece; small buildings in smaller pieces.
        let storey = |cs: f32| if cs < 2.6 { (cs * 1.4).max(1.8) } else { 3.4 * (cs / 3.4).round().max(1.0) };
        // Finer pieces than a cube per house, but a bounded count per building:
        // floor slabs over the area and panels round the walls, per storey.
        let budget = if tall { 2800.0 } else { 1000.0 };
        let count = |cs: f32| -> f32 {
            pg.iter().zip(&areas).map(|(p, ar)| {
                let per: f32 = p.ring.iter().enumerate().map(|(i, q)| q.distance(p.ring[(i + 1) % p.ring.len()])).sum();
                let nz = ((p.top - p.base) as f32 / storey(cs)).round().max(1.0);
                nz * (ar / (cs * cs) + per / cs + 1.0)
            }).sum()
        };
        let mut cs = (vol / (budget * 0.7)).cbrt().clamp(1.5, 12.0);
        while cs < 14.0 && count(cs) > budget {
            cs *= 1.08;
        }
        let lh_t = storey(cs);
        let ext = pg.iter().flat_map(|p| p.ring.iter()).map(|p| p.length()).fold(0.0f32, f32::max).max(3.0);
        // Wall panels and floor slabs: concrete over a few storeys has steel in it.
        let rebar = height > 10.0;
        let t_wall = (cs * 0.06).clamp(0.25, 0.45);
        let t_slab = if height > 10.0 { 0.32 } else { 0.24 };

        let hi = ((impact - base).dot(u) as f32).clamp(0.0, height);
        let ih = to2(impact);
        let speed = vel.length() as f32;
        let dirv = vel.normalize_or_zero().as_vec3();
        let hole = if hit.is_some() { 4.0 + speed * 0.05 } else { 4.0 + cs * 1.3 + speed * 0.06 };
        let dir64 = dirv.as_dvec3();
        // How far into the building the aircraft's outline is cut through the floors.
        let tunnel = 8.0 + speed * 0.12;
        let nseed = (self.rng.f() * 1000.0) as u32;
        let h3 =|v: Vec2| (e * v.x as f64 + nn * v.y as f64).as_vec3();
        let (ax, bx) = (h3(a), h3(b));
        let rot = DQuat::from_mat3(&glam::DMat3::from_cols(ax.as_dvec3(), u, ax.as_dvec3().cross(u))).normalize();

        // The block above the impact leans towards the hole, on a hinge at the far side.
        let mut dh2 = ih.normalize_or_zero();
        if dh2 == Vec2::ZERO {
            dh2 = -Vec2::new(vel.dot(e) as f32, vel.dot(nn) as f32).normalize_or_zero();
        }
        if dh2 == Vec2::ZERO {
            dh2 = Vec2::X;
        }
        let dh = e * dh2.x as f64 + nn * dh2.y as f64;
        let far = pg.iter().flat_map(|p| p.ring.iter()).map(|p| -p.dot(dh2)).fold(0.0f32, f32::max);
        let cut_lo = (lh_t * 1.5).min(height * 0.3);
        let cut = hi.clamp(cut_lo, (height - lh_t).max(cut_lo));
        let si = self.structs.len() as u32;
        let mut st = Structure {
            base,
            u,
            height,
            ext,
            tall,
            cut,
            pivot: base + u * cut as f64 - dh * far as f64,
            axis: u.cross(dh).normalize(),
            tilt: 0.0,
            tilt_v: 0.0,
            tilt_max: self.rng.range(0.12, 0.42),
            q: DQuat::IDENTITY,
            drop: 0.0,
            drop_v: 0.0,
            top_lower: cut,
            delay: if tall { self.rng.range(1.4, 2.6) } else { 0.0 },
            // (an aircraft-shaped hole keeps its shape until the floors give)
            hole: if hit.is_some() { 0.0 } else { hole },
            t: 0.0,
            gave: false,
            grounded: false,
            left: 0,
            done_t: -1.0,
            skin: None,
        };

        // Every piece of the standing building, before the hole is taken out:
        // (centre, half size, orientation, top above the base, outward normal,
        // colour, material, facade, how much building it stands for)
        struct Piece {
            pos: DVec3,
            half: Vec3,
            rot: DQuat,
            z1: f32,
            out: Vec3,
            col: [f32; 3],
            mat: f32,
            face: [f32; 4],
            vh: f32,
            mass: f32,
            /// outer wall or roof, and its colour before any soot
            outer: bool,
            col0: [f32; 3],
            /// inside the aircraft's outline: blown out
            cut: bool,
            /// a small piece of wall from around the hole
            fine: bool,
        }
        let mut pieces: Vec<Piece> = Vec::new();
        for p in &pg {
            let ring = &p.ring;
            let n = ring.len();
            let hp = (p.top - p.base).max(1.0) as f32;
            let z_off = (p.base - base_alt) as f32;
            let nz = (hp / lh_t).round().max(1.0) as usize;
            let lh = hp / nz as f32;
            let ccw = ring.iter().enumerate().map(|(i, q)| q.perp_dot(ring[(i + 1) % n])).sum::<f32>() > 0.0;

            // The outer wall, in panels following the walls as they run (the
            // same way the walls' facade coordinate runs, from the first corner).
            let mut s_at = vec![0.0f32; n + 1];
            for i in 0..n {
                let l = ring[i].distance(ring[(i + 1) % n]);
                s_at[i + 1] = s_at[i] + if l < 0.05 { 0.0 } else { l };
            }
            let at = |s: f32| -> Vec2 {
                for i in 0..n {
                    if s <= s_at[i + 1] && s_at[i + 1] > s_at[i] {
                        let t = ((s - s_at[i]) / (s_at[i + 1] - s_at[i])).clamp(0.0, 1.0);
                        return ring[i] + (ring[(i + 1) % n] - ring[i]) * t;
                    }
                }
                ring[0]
            };
            // Runs of wall between corners (a turn of more than 25 degrees).
            let mut breaks: Vec<f32> = (0..n)
                .filter(|&i| {
                    let d0 = (ring[i] - ring[(i + n - 1) % n]).normalize_or_zero();
                    let d1 = (ring[(i + 1) % n] - ring[i]).normalize_or_zero();
                    i == 0 || d0.dot(d1) < 0.906
                })
                .map(|i| s_at[i])
                .collect();
            breaks.push(s_at[n]);
            for w in breaks.windows(2) {
                let (s0, s1) = (w[0], w[1]);
                let len = s1 - s0;
                if len < 0.3 {
                    continue;
                }
                let np = (len / cs).round().max(1.0) as usize;
                let cuts: Vec<f32> = (0..=np).map(|k| s0 + len * (k as f32 + if k == 0 || k == np { 0.0 } else { self.rng.range(-0.3, 0.3) }) / np as f32).collect();
                for k in 0..np {
                    let (sa, sb) = (cuts[k], cuts[k + 1]);
                    let (pa, pb) = (at(sa), at(sb));
                    let lc = pa.distance(pb);
                    if lc < 0.3 {
                        continue;
                    }
                    let dir = (pb - pa) / lc;
                    let out2 = if ccw { -dir.perp() } else { dir.perp() };
                    let c2 = (pa + pb) * 0.5 - out2 * (t_wall * 0.5);
                    // Local frame: x = up x out, y = out (thin), z = down.
                    let out3 = h3(out2);
                    let x3 = u.as_vec3().cross(out3);
                    let xs = if Vec2::new(-out2.y, out2.x).dot(dir) >= 0.0 { 1.0 } else { -1.0 };
                    let prot = DQuat::from_mat3(&glam::DMat3::from_cols(x3.as_dvec3(), out3.as_dvec3(), -u)).normalize();
                    for kz in 0..nz {
                        let zc = z_off + (kz as f32 + 0.5) * lh;
                        let pos = base + h3(c2).as_dvec3() + u * zc as f64;
                        let face = [(sa + sb) * 0.5, (base_alt + zc as f64 - p.gref) as f32, xs * p.h.max(1.0), p.style];
                        // A piece of wall (local x0..x1 across the panel, dz down from its middle, ch high).
                        let cell = |x0: f32, x1: f32, dz: f32, ch: f32| {
                            let dx = (x0 + x1) * 0.5;
                            let w = x1 - x0;
                            Piece {
                                pos: pos + prot * DVec3::new(dx as f64, 0.0, dz as f64),
                                half: Vec3::new(w * 0.5, t_wall * 0.5, ch * 0.5),
                                rot: prot,
                                z1: zc - dz + ch * 0.5,
                                out: out3,
                                col: p.wall,
                                mat: 2.0,
                                face: [face[0] + xs * dx, face[1] - dz, face[2], face[3]],
                                vh: ch * 0.5,
                                mass: w * ch * t_wall * 3.0,
                                outer: true,
                                col0: p.wall,
                                cut: false,
                                fine: false,
                            }
                        };
                        // Facing the aircraft near where it went in: cut finer, and its outline taken out.
                        let facing = -out3.dot(dirv);
                        if let Some(h) = hit.filter(|_| facing > 0.15) {
                            let (c, t) = h.place(pos, impact, dir64);
                            let reach = lc.max(lh) * 0.75 / facing.min(1.0).max(0.25) + 3.0;
                            if t.abs() < 12.0 + lc.max(lh) && h.sdf(c) < reach {
                                let nx = (lc / FINE).round().max(1.0) as usize;
                                let ny = (lh / FINE).round().max(1.0) as usize;
                                let (cw, ch) = (lc / nx as f32, lh / ny as f32);
                                for j in 0..ny {
                                    let dz = -lh * 0.5 + (j as f32 + 0.5) * ch;
                                    // Distance of each cell from the outline, a little ragged.
                                    let row: Vec<f32> = (0..nx)
                                        .map(|i| {
                                            let dx = -lc * 0.5 + (i as f32 + 0.5) * cw;
                                            let (c, t) = h.place(pos + prot * DVec3::new(dx as f64, 0.0, dz as f64), impact, dir64);
                                            if t.abs() > 10.0 { 1e3 } else { h.sdf(c) + 0.3 * vnoise(c * 0.9, nseed) + self.rng.range(-0.06, 0.06) }
                                        })
                                        .collect();
                                    // Untouched cells in a row go back together as one piece.
                                    let mut run: Option<usize> = None;
                                    for i in 0..=nx {
                                        let s = row.get(i).copied().unwrap_or(-1.0);
                                        if i < nx && s >= 2.0 {
                                            run.get_or_insert(i);
                                            continue;
                                        }
                                        if let Some(i0) = run.take() {
                                            let mut pc = cell(-lc * 0.5 + i0 as f32 * cw, -lc * 0.5 + i as f32 * cw, dz, ch);
                                            pc.fine = true;
                                            pieces.push(pc);
                                        }
                                        if i == nx {
                                            break;
                                        }
                                        let x0 = -lc * 0.5 + i as f32 * cw;
                                        let mut pc = cell(x0, x0 + cw, dz, ch);
                                        pc.fine = true;
                                        if s < 0.0 {
                                            pc.cut = true;
                                        } else {
                                            // Round the hole: sooted, the glass out, the
                                            // edge bent in the way the aircraft went.
                                            let c = h.place(pc.pos, impact, dir64).0;
                                            let k = ((s + 0.6 * vnoise(c * 0.6, nseed + 7)) / 2.0).clamp(0.0, 1.0);
                                            let dark = 0.4 + 0.6 * k * k * (3.0 - 2.0 * k);
                                            pc.col = pc.col.map(|v| v * dark);
                                            if s < 0.7 + 0.5 * vnoise(c * 1.3, nseed + 3) && ((pc.face[3] as u32 >> 8) & 128) == 0 {
                                                pc.face[3] += SMASHED;
                                            }
                                            if s < 0.9 {
                                                let b = 1.0 - s / 0.9;
                                                pc.pos += dir64 * (b * self.rng.range(0.1, 0.8)) as f64;
                                                pc.rot = (DQuat::from_axis_angle(self.rng.dir().as_dvec3(), (b * self.rng.range(0.05, 0.4)) as f64) * pc.rot).normalize();
                                            }
                                        }
                                        pieces.push(pc);
                                    }
                                }
                                continue;
                            }
                        }
                        pieces.push(cell(-lc * 0.5, lc * 0.5, 0.0, lh));
                    }
                }
            }

            // Floor slabs inside, a grid of them along the building's grain,
            // pulled in at the edges to stay inside the walls.
            let (mut lo, mut hi2) = (Vec2::splat(1e9), Vec2::splat(-1e9));
            for q in ring {
                let q = Vec2::new(q.dot(a), q.dot(b));
                lo = lo.min(q);
                hi2 = hi2.max(q);
            }
            let (na, nb) = ((((hi2.x - lo.x) / cs).round().max(1.0)) as usize, (((hi2.y - lo.y) / cs).round().max(1.0)) as usize);
            let (sa, sb) = ((hi2.x - lo.x) / na as f32, (hi2.y - lo.y) / nb as f32);
            // Uneven cuts, so pieces come in different sizes and the pile is not a lattice.
            let cuts = |r: &mut Rng, n: usize, s: f32, lo: f32| -> Vec<f32> { (0..=n).map(|i| lo + s * (i as f32 + if i == 0 || i == n { 0.0 } else { r.range(-0.32, 0.32) })).collect() };
            let ea = cuts(&mut self.rng, na, sa, lo.x);
            let eb = cuts(&mut self.rng, nb, sb, lo.y);
            let margin = t_wall + 0.1;
            for ia in 0..na {
                for ib in 0..nb {
                    let (mut x0, mut x1, mut y0, mut y1) = (ea[ia], ea[ia + 1], eb[ib], eb[ib + 1]);
                    let mid = a * (x0 + x1) * 0.5 + b * (y0 + y1) * 0.5;
                    if !inside(ring, mid) {
                        continue;
                    }
                    let (w0, h0) = (x1 - x0, y1 - y0);
                    let mut ok = true;
                    for _ in 0..6 {
                        let mut bad = false;
                        for (sx, sy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                            let (cx, cy) = (if sx == 1 { x1 } else { x0 }, if sy == 1 { y1 } else { y0 });
                            let c = a * cx + b * cy;
                            let away = (c - mid).normalize_or_zero() * margin;
                            if !inside(ring, c + away) {
                                bad = true;
                                if sx == 1 { x1 -= 0.16 * w0 } else { x0 += 0.16 * w0 }
                                if sy == 1 { y1 -= 0.16 * h0 } else { y0 += 0.16 * h0 }
                            }
                        }
                        if !bad {
                            break;
                        }
                        if x1 - x0 < 0.3 * w0 || y1 - y0 < 0.3 * h0 {
                            ok = false;
                            break;
                        }
                    }
                    if !ok {
                        continue;
                    }
                    let (wa, wb) = (x1 - x0, y1 - y0);
                    let q = Vec2::new((x0 + x1) * 0.5, (y0 + y1) * 0.5);
                    let p2 = a * q.x + b * q.y;
                    // At the edge: which way is out.
                    let mut o2 = Vec2::ZERO;
                    for d in [a * (w0 + sa) * 0.5, -a * (w0 + sa) * 0.5, b * (h0 + sb) * 0.5, -b * (h0 + sb) * 0.5] {
                        if !inside(ring, p2 + d) {
                            o2 += d.normalize();
                        }
                    }
                    let out = h3(o2).normalize_or_zero();
                    let shade = self.rng.range(0.85, 1.05);
                    for kz in 1..=nz {
                        let roof = kz == nz;
                        // The roof a little under the top of the walls (the parapet).
                        let ztop = if roof { hp - (0.6f32).min(lh * 0.3) } else { kz as f32 * lh };
                        let zc = z_off + ztop - t_slab * 0.5;
                        let col = if roof {
                            let k = self.rng.range(0.97, 1.03);
                            p.roof.map(|c| c * k)
                        } else {
                            let g = 0.3 * shade;
                            [g * 1.04, g, g * 0.94]
                        };
                        pieces.push(Piece {
                            pos: base + (ax * q.x + bx * q.y).as_dvec3() + u * zc as f64,
                            half: Vec3::new(wa * 0.5, t_slab * 0.5, wb * 0.5),
                            rot,
                            z1: z_off + ztop,
                            out,
                            col,
                            mat: 1.0,
                            face: [0.0; 4],
                            vh: t_slab * 0.5,
                            mass: wa * wb * lh,
                            outer: roof,
                            col0: col,
                            cut: false,
                            fine: false,
                        });
                    }
                }
            }
        }

        // Floors on the aircraft's way in: how close they come to its outline.
        let through = |pc: &Piece| -> f32 {
            let Some(h) = hit else { return 1e3 };
            [(0.0, 0.0), (0.7, 0.7), (-0.7, 0.7), (0.7, -0.7), (-0.7, -0.7)]
                .iter()
                .map(|&(sx, sz)| {
                    let q = pc.pos + pc.rot * DVec3::new((pc.half.x * sx) as f64, 0.0, (pc.half.z * sz) as f64);
                    let (c, t) = h.place(q, impact, dir64);
                    if t > -2.0 && t < tunnel + 4.0 { h.sdf(c) } else { 1e3 }
                })
                .fold(1e3, f32::min)
        };
        // In the zone of the 3D tiles the building's own mesh goes onto its
        // pieces: each piece of wall or roof moved onto the mesh it carries,
        // and coloured like it.
        let skin = photo.map(|src| {
            let refs: Vec<PieceRef> = pieces.iter().map(|p| PieceRef { pos: p.pos, rot: p.rot, half: p.half, outer: p.outer }).collect();
            let a = crate::skin::assign(src, &refs, hit.map(|_| (impact, 11.0)), (cs * 0.25).clamp(0.8, 2.5));
            for (i, pc) in pieces.iter_mut().enumerate() {
                if !pc.outer {
                    continue;
                }
                pc.pos += pc.rot * DVec3::Y * a.shift[i] as f64;
                let c = a.col[i].unwrap_or(a.mean);
                // (soot from the impact stays on: the colour as a share of what it was)
                let k = [0, 1, 2].map(|j| (pc.col[j] / pc.col0[j].max(1e-3)).min(1.0));
                pc.col0 = c;
                pc.col = [c[0] * k[0], c[1] * k[1], c[2] * k[2]];
            }
            log::info!("3D tiles: {} triangles on {} of {} pieces", a.n_tris(), a.area.iter().filter(|&&w| w > 0.0).count(), refs.len());
            (Group::new(src), a)
        });
        let frames: Vec<(DVec3, DQuat)> = pieces.iter().map(|p| (p.pos, p.rot)).collect();
        let mut slot_of = vec![NO_SKIN; pieces.len()];
        let (mut sgroup, sassign) = match skin {
            Some((g, a)) => (Some(g), Some(a)),
            None => (None, None),
        };
        // How deep the wall the aircraft went through lies along its way.
        let (mut t_in, mut n_in) = (0.0f32, 0);
        let mut blown: Vec<(DVec3, Vec3, f32, Vec3, bool, bool)> = Vec::new();
        let mut n_bits = 0;
        for (pi, pc) in pieces.into_iter().enumerate() {
            let d = (pc.pos - impact).length() as f32;
            let facade = pc.mat > 1.5;
            let near_t = if facade { 1e3 } else { through(&pc) };
            let out = if hit.is_some() { pc.cut || near_t < 0.6 } else { d < hole };
            if let (Some(h), true) = (hit, pc.cut) {
                t_in += h.place(pc.pos, impact, dir64).1;
                n_in += 1;
            }
            let mut pc = pc;
            if near_t < 3.5 {
                // Burnt round where it went through.
                let k = 0.2 + 0.8 * (near_t / 3.5).max(0.0);
                pc.col = pc.col.map(|v| v * k);
            }
            if out {
                blown.push((pc.pos, pc.half, d, pc.out, facade, pc.fine));
                // Shattered: a few smaller pieces thrown on through the building
                // (fewer, and smaller, from the small pieces round the hole).
                let near = (1.0 - d / hole).max(0.0);
                for _ in 0..2 {
                    if n_bits > 700 || (pc.fine && self.rng.f() < 0.75) {
                        break;
                    }
                    n_bits += 1;
                    let r = &mut self.rng;
                    let s = if pc.fine { pc.half.x.max(pc.half.z) * r.range(0.5, 0.9) } else { cs * 0.5 * r.range(0.2, 0.5) };
                    let flat = r.f() < 0.6;
                    let h = if flat { Vec3::new(s, s * r.range(0.15, 0.35), s * r.range(0.6, 1.0)) } else { Vec3::splat(s * 0.6) };
                    let v = dirv * speed * r.range(0.12, 0.45) * (0.4 + near) + (pc.pos - impact).normalize_or_zero().as_vec3() * r.range(5.0, 18.0) * (0.3 + near) + r.dir() * 3.0 + u.as_vec3() * r.range(0.0, 4.0);
                    let tone = r.range(0.85, 1.1);
                    let mesh = if flat { M_SLAB + (r.f() * N_SLAB as f32) as u32 % N_SLAB } else { M_ROCK + (r.f() * 3.0) as u32 % 3 };
                    let mut c = Chunk::loose(pc.pos + (r.dir() * pc.half * 0.5).as_dvec3(), h, DQuat::from_axis_angle(r.dir().as_dvec3(), r.range(0.0, 6.0) as f64) * rot, v.as_dvec3(), (r.dir() * r.range(1.0, 7.0)).as_dvec3(), [0.33 * tone, 0.32 * tone, 0.3 * tone], mesh);
                    c.dimp = d;
                    self.chunks.push(c);
                }
                continue;
            }
            let upper = tall && (pc.z1 - pc.vh) > cut;
            st.left += 1;
            let sk = match (&mut sgroup, &sassign) {
                (Some(g), Some(a)) if pc.outer || a.area[pi] > 0.0 => g.alloc(pc.col0),
                _ => NO_SKIN,
            };
            slot_of[pi] = sk;
            self.chunks.push(Chunk {
                pos: pc.pos,
                half: pc.half,
                rot: pc.rot,
                vel: DVec3::ZERO,
                omega: DVec3::ZERO,
                col: pc.col,
                mesh: M_CUBE,
                mat: pc.mat,
                hold: if tall { 0.0 } else { 0.08 + d / 24.0 + self.rng.range(0.0, 0.35) },
                state: 0,
                bounces: 0,
                group: si,
                upper,
                home: pc.pos,
                home_rot: pc.rot,
                z1: pc.z1,
                dimp: d,
                out: pc.out,
                bit: false,
                under: Under::NONE,
                var: (self.rng.f() * N_SLAB as f32) as u32 % N_SLAB,
                vh: pc.vh,
                mass: pc.mass,
                rebar: rebar && !pc.fine && !(facade && ((pc.face[3] as u32 >> 8) & 15) == 3),
                face: pc.face,
                gone: false,
                void: false,
                skin: sk,
                outer: pc.outer,
            });
        }
        if let (Some(mut g), Some(a)) = (sgroup, sassign) {
            g.build(&a, &slot_of, &frames);
            st.skin = Some(g);
        }
        // Inside the hole: soot-black and dark, not the far side of the
        // building in daylight. A burnt wall a few metres in, across the
        // aircraft's outline, kept inside the building.
        if let (Some(h), true) = (hit, n_in > 0) {
            let depth = t_in / n_in as f32 + 4.0;
            let fwd = h.up.cross(h.right).normalize();
            let at_c = |x: f32, y: f32| impact + dir64 * depth as f64 + h.right * x as f64 + h.up * y as f64;
            let (y0, y1) = (-1.5f32, 2.5f32);
            let inb1 = |q: DVec3| {
                let q2 = to2(q);
                pg.iter().any(|p| inside(&p.ring, q2) && (q - base).dot(u) >= p.base - base_alt && (q - base).dot(u) <= p.top - base_alt)
            };
            // (with room to spare: its corners well inside the walls)
            let inb = |x: f32| [y0, y1].iter().all(|&y| [-0.8, 0.8].iter().all(|&dd| inb1(at_c(x, y) + dir64 * dd)));
            // How far it reaches each way, staying inside.
            let reach = |sgn: f32| (1..=20).map(|k| k as f32 * 0.5 * sgn).take_while(|&x| inb(x)).last().unwrap_or(0.0);
            let (xl, xr) = (reach(-1.0), reach(1.0));
            if xr - xl > 1.0 && inb(0.0) {
                let c = at_c((xl + xr) * 0.5, (y0 + y1) * 0.5);
                let prot = DQuat::from_mat3(&glam::DMat3::from_cols(h.right, fwd, h.right.cross(fwd))).normalize();
                let zc = (c - base).dot(u) as f32;
                let half = Vec3::new((xr - xl) * 0.5, 0.15, (y1 - y0) * 0.5);
                st.left += 1;
                let mut ch = Chunk::loose(c, half, prot, DVec3::ZERO, DVec3::ZERO, [0.012, 0.011, 0.01], M_CUBE);
                ch.state = 0;
                ch.bit = false;
                ch.group = si;
                ch.hold = 0.1;
                ch.z1 = zc + half.z;
                ch.vh = half.z;
                ch.dimp = 0.0;
                ch.void = true;
                self.chunks.push(ch);
            }
        }
        // The hole: glass and dust blown out of it.
        let mut n_glass = 0;
        for (pos, half, d, out, facade, fine) in blown {
            let near = (1.0 - d / hole).max(0.0);
            if facade && n_glass < 900 {
                let o = if out == Vec3::ZERO { dirv } else { out };
                let n = if fine { 3 } else { 14 };
                self.glass(pos, half.max_element(), o, dirv * speed * 0.15 * near, n, emit);
                n_glass += n;
            }
            if self.rng.f() < if fine { 0.06 } else { 0.6 } {
                let o = (pos - impact).normalize_or_zero().as_vec3();
                self.jet(pos, o, dirv * speed * 0.4, half.max_element().clamp(1.5, 4.0), emit);
            }
        }
        if tall {
            // Pale dust boiling out of the hole for the first seconds.
            let up = u.as_vec3();
            for _ in 0..40 {
                let r = &mut self.rng;
                let d = (r.dir() + dh.as_vec3() * 0.8).normalize();
                emit.push((impact + (r.dir() * hole * 0.5).as_dvec3(), Particle { vel: (d * r.range(4.0, 16.0) + up * r.range(0.0, 3.0)).into(), life: r.range(8.0, 16.0), size0: r.range(3.0, 6.0), size1: r.range(14.0, 26.0), temp: 0.1, kind: GRIT, drag: 0.9, buoy: 0.25, seed: r.f(), alpha: r.range(0.3, 0.5), ..Default::default() }));
            }
        } else {
            self.rumble += 0.3;
        }
        if st.left == 0 {
            st.done_t = 0.0;
        }
        self.structs.push(st);
        self.trim();
    }

    /// Rock and earth thrown out of a fresh crater.
    pub fn eject(&mut self, site: DVec3, radius: f32, vel: DVec3, emit: &mut Vec<(DVec3, Particle)>) {
        let up = site.normalize();
        let along = (vel - up * vel.dot(up)) * 0.12;
        for k in 0..(10.0 + radius * 2.2) as u32 {
            let d = (self.rng.dir() + up.as_vec3() * 1.3).normalize();
            let r = self.rng.range(0.25, 0.5 + radius * 0.09);
            let tone = self.rng.range(0.6, 1.2);
            let mut c = Chunk::loose(site + up * 0.5 + (self.rng.dir() * radius * 0.5).as_dvec3(), Vec3::splat(r), DQuat::from_axis_angle(self.rng.dir().as_dvec3(), self.rng.range(0.0, 6.0) as f64), (d * self.rng.range(7.0, 26.0)).as_dvec3() + along, self.rng.dir().as_dvec3() * self.rng.range(2.0, 9.0) as f64, [0.17 * tone, 0.14 * tone, 0.11 * tone], M_ROCK + k % 3);
            c.home = site;
            self.chunks.push(c);
        }
        // A fountain of dirt.
        for _ in 0..(26.0 + radius * 3.0) as usize {
            let d = (self.rng.dir() + up.as_vec3() * 1.6).normalize();
            emit.push((site + up, Particle { vel: (d * self.rng.range(6.0, 30.0)).into(), life: self.rng.range(3.0, 7.0), size0: self.rng.range(1.0, 2.0), size1: self.rng.range(5.0, 10.0), temp: 0.2, kind: DUST, drag: 1.5, buoy: 0.3, seed: self.rng.f(), alpha: self.rng.range(0.35, 0.6), ..Default::default() }));
        }
        self.trim();
    }

    /// Loose pieces knocked off something (a laser or a wrecking ball taking
    /// a bite out of a wall, a meteor shot to bits): `n` of them about `size`
    /// metres across, thrown out along `out` (plus `vel`), broken concrete
    /// of colour `col` or (`rock`) lumps of rock.
    #[allow(clippy::too_many_arguments)]
    pub fn fling(&mut self, at: DVec3, out: DVec3, vel: DVec3, n: u32, size: f32, col: [f32; 3], rock: bool, emit: &mut Vec<(DVec3, Particle)>) {
        let o = out.normalize_or_zero().as_vec3();
        for k in 0..n {
            let d = (o * 1.3 + self.rng.dir()).normalize_or_zero();
            let s = self.rng.range(0.35, 1.0) * size;
            let half = if rock { Vec3::splat(s) } else { Vec3::new(s, s * self.rng.range(0.35, 0.7), s * self.rng.range(0.6, 1.0)) };
            let tone = self.rng.range(0.7, 1.1);
            let mesh = if rock { M_ROCK + k % 3 } else { M_SLAB + (self.rng.f() * N_SLAB as f32) as u32 % N_SLAB };
            let rot = DQuat::from_axis_angle(self.rng.dir().as_dvec3(), self.rng.range(0.0, 6.0) as f64);
            let v = (d * self.rng.range(3.0, 14.0)).as_dvec3() + vel;
            let spin = self.rng.dir().as_dvec3() * self.rng.range(2.0, 10.0) as f64;
            let c = Chunk::loose(at + out.normalize_or_zero() * 0.4, half, rot, v, spin, [col[0] * tone, col[1] * tone, col[2] * tone], mesh);
            self.chunks.push(c);
        }
        // A puff of the same dust.
        if rock || self.rng.f() < 0.3 {
            let d = (o + self.rng.dir() * 0.8).normalize_or_zero();
            emit.push((at, Particle { vel: (d * self.rng.range(2.0, 6.0)).into(), life: self.rng.range(1.5, 3.0), size0: self.rng.range(0.3, 0.7), size1: self.rng.range(1.5, 3.5), kind: if rock { DUST } else { GRIT }, drag: 1.4, buoy: 0.2, seed: self.rng.f(), alpha: self.rng.range(0.2, 0.35), ..Default::default() }));
        }
        self.trim();
    }

    /// Keep the piece count bounded: small fragments at rest go first, then
    /// the oldest slabs at rest. Standing pieces are never dropped.
    fn trim(&mut self) {
        for pass in 0..2 {
            if self.chunks.len() <= MAX_CHUNKS {
                return;
            }
            let mut extra = self.chunks.len() - MAX_CHUNKS;
            self.chunks.retain(|c| {
                if extra > 0 && c.state == 2 && (pass == 1 || c.bit) {
                    extra -= 1;
                    false
                } else {
                    true
                }
            });
        }
    }

    pub fn update(&mut self, dt: f32, world: &World, emit: &mut Vec<(DVec3, Particle)>) {
        self.rumble *= (-dt * 2.5).exp();
        self.crush_cool -= dt;
        self.update_trees(dt, emit);

        // The structures: when the damaged floors give, how far the block above has come down.
        let mut grounded: Vec<usize> = Vec::new();
        for (i, s) in self.structs.iter_mut().enumerate() {
            if s.done_t >= 0.0 {
                s.done_t += dt;
                continue;
            }
            s.t += dt;
            if !s.tall {
                continue;
            }
            // It starts to lean a moment before it lets go.
            if s.t > s.delay * 0.6 {
                let hu = (s.height - s.cut).max(6.0);
                let k = if s.t > s.delay { 1.0 } else { 0.12 };
                s.tilt_v += 2.2 * 9.81 / hu * s.tilt.max(0.06).sin() * dt * k;
                s.tilt += s.tilt_v * dt;
                if s.tilt >= s.tilt_max {
                    s.tilt = s.tilt_max;
                    s.tilt_v = 0.0;
                }
                s.q = DQuat::from_axis_angle(s.axis, s.tilt as f64);
            }
            if s.t > s.delay {
                if !s.gave {
                    s.gave = true;
                    // The side with the hole goes first: a lurch towards it.
                    s.tilt_v += 0.05 + 0.05 * (40.0 / (s.height - s.cut).max(10.0)).min(1.0);
                    self.booms.push(0.8 + (s.height / 150.0).min(1.0));
                    self.rumble += 0.8;
                }
                // Each floor it lands on slows it a little: well under free fall.
                let acc = if s.top_lower > 0.0 { 0.55 } else { 0.8 } * 9.81;
                s.drop_v = (s.drop_v + acc * dt).min(55.0);
                s.drop += s.drop_v * dt;
                s.top_lower = (s.cut - 0.7 * s.drop).max(0.0);
                if s.top_lower <= 0.0 && !s.grounded {
                    s.grounded = true;
                    grounded.push(i);
                    self.booms.push(1.2 + (s.height / 120.0).min(1.5));
                }
                self.rumble = self.rumble.max((s.drop_v / 30.0).min(1.5));
            }
        }
        for i in grounded {
            self.surge(i, emit);
        }

        // The pieces.
        let mut released: Vec<(DVec3, Vec3, f32, Vec3, bool)> = Vec::new();
        let mut landed: Vec<(DVec3, f32, f32)> = Vec::new();
        let mut rests: Vec<(DVec3, f32, f64)> = Vec::new();
        let mut left_dec: Vec<(u32, usize)> = Vec::new();
        let n_chunks = self.chunks.len();
        let Self { chunks, structs, rng, field, f_org, f_e, f_n, crush, crush_cool, rumble, .. } = self;
        let fkey = |p: DVec3| {
            let d = p - *f_org;
            ((d.dot(*f_e) / CELL).floor() as i32, (d.dot(*f_n) / CELL).floor() as i32)
        };
        let fat = |field: &HashMap<(i32, i32), f32>, p: DVec3| field.get(&fkey(p)).map(|h| *h as f64).unwrap_or(-1e9);
        let mut bits: Vec<Chunk> = Vec::new();
        let mut shatter: Vec<(DVec3, Vec3, Vec3, Vec3)> = Vec::new();
        let mut mounds: Vec<(DVec3, DVec3, f32, [f32; 3])> = Vec::new();
        // pieces that broke in two: (the new half in `bits`, structure, slot,
        // axis, where it broke, how far each half moved)
        let mut splits: Vec<(usize, u32, u32, usize, f32, f32, f32)> = Vec::new();
        for c in chunks.iter_mut() {
            if c.state == 2 {
                continue;
            }
            if c.state == 0 {
                let s = &structs[c.group as usize];
                let dribble = !s.gave && c.dimp < s.hole * 1.8 && rng.f() < dt * 0.35;
                if s.tall {
                    if c.upper {
                        let prev = c.pos;
                        c.pos = s.pivot - s.u * s.drop as f64 + s.q * (c.home - s.pivot);
                        c.rot = (s.q * c.home_rot).normalize();
                        let v = if dt > 0.0 { (c.pos - prev) / dt as f64 } else { DVec3::ZERO };
                        let hb = (c.pos - s.base).dot(s.u) as f32 - c.vh;
                        if !(s.gave && hb < s.top_lower + 0.3) && !dribble {
                            continue;
                        }
                        let sp = v.length() as f32;
                        // Crushed out sideways: the faster it comes down, the harder.
                        c.vel = v + (c.out * (2.5 + 0.32 * sp) * rng.range(0.4, 1.3) + rng.dir() * (1.0 + sp * 0.08)).as_dvec3();
                        c.omega = (rng.dir() * rng.range(0.4, 2.5)).as_dvec3();
                    } else {
                        if !(s.gave && c.z1 > s.top_lower) && !dribble {
                            continue;
                        }
                        let dv = s.drop_v;
                        c.vel = (c.out * (2.0 + 0.3 * dv) * rng.range(0.4, 1.2) - s.u.as_vec3() * (dv * 0.3) + rng.dir() * (1.2 + dv * 0.05)).as_dvec3();
                        c.omega = (rng.dir() * rng.range(0.3, 2.0)).as_dvec3();
                    }
                } else {
                    c.hold -= dt;
                    if c.hold > 0.0 {
                        continue;
                    }
                    c.vel = (c.out * rng.range(0.5, 2.8) + rng.dir() * 0.6).as_dvec3();
                    c.omega = (rng.dir() * rng.range(0.2, 1.5)).as_dvec3();
                }
                c.state = 1;
                left_dec.push((c.group, 1));
                if c.void {
                    c.gone = true;
                    continue;
                }
                // (the dust a floor throws out scales with how wide the building is)
                released.push((c.pos, c.out, c.half.max_element().max(s.ext * 0.12), c.vel.as_vec3(), c.mat > 1.5));
                *rumble += 0.012 * c.half.max_element().min(5.0);
                // Broken now: no longer a clean box but a ragged slab or panel,
                // and the glass of a piece of facade is out.
                c.mesh = M_SLAB + c.var;
                if c.mat > 1.5 {
                    if ((c.face[3] as u32 >> 8) & 15) == 3 {
                        // Curtain wall: most panels burst into glass and bent
                        // mullions; the rest come away whole, glass and all.
                        if rng.f() < 0.7 {
                            c.gone = true;
                            shatter.push((c.pos, c.half, c.out, c.vel.as_vec3()));
                            continue;
                        }
                    } else {
                        c.face[3] += SMASHED;
                    }
                }
                // Each block that lets go sheds a few lumps of concrete.
                if n_chunks + bits.len() < MAX_CHUNKS - 400 && rng.f() < 0.55 {
                    for k in 0..1 + (rng.f() * 2.0) as u32 {
                        let tone = rng.range(0.7, 1.1);
                        let mut lump = Chunk::loose(c.pos + (rng.dir() * c.half).as_dvec3(), Vec3::splat(rng.range(0.25, 0.4 + c.half.x.min(c.half.z) * 0.12)), DQuat::from_axis_angle(rng.dir().as_dvec3(), rng.range(0.0, 6.0) as f64), c.vel + (rng.dir() * 3.0).as_dvec3(), (rng.dir() * rng.range(2.0, 8.0)).as_dvec3(), [0.27 * tone, 0.26 * tone, 0.245 * tone], M_ROCK + k % 3);
                        lump.home = c.pos;
                        bits.push(lump);
                    }
                }
            }

            // Falling.
            let up = c.pos.normalize();
            c.vel -= up * 9.81 * dt as f64;
            c.vel *= 1.0 - (if c.bit { 0.08 } else { 0.02 }) * dt as f64;
            c.pos += c.vel * dt as f64;
            let w = c.omega * dt as f64;
            if w.length_squared() > 1e-12 {
                c.rot = (DQuat::from_scaled_axis(w) * c.rot).normalize();
            }
            let (lat, lon, h) = geo::pos_to_lla(c.pos);
            if (c.pos - c.under.at).length_squared() > 9.0 {
                let Some(g) = world.ground(lat, lon) else { continue };
                let mut un = Under { at: c.pos, ground: g.h, water: g.water, roof: -1e9, hit: None };
                if let Some(b) = world.building(lat, lon) {
                    if !b.bridge || h > b.base {
                        un.roof = b.top;
                        if !b.bridge {
                            un.hit = Some((b.tile, b.idx, (b.top - b.base) as f32));
                        }
                    }
                }
                c.under = un;
            }
            let un = c.under;
            let slab = !c.bit && (c.mesh == M_CUBE || is_slab(c.mesh));
            let r = if is_rock(c.mesh) { c.half.x * 0.7 } else { c.half.min_element() } as f64;
            let fh = fat(field, c.pos);
            let mut floor = un.ground.max(fh);
            if un.roof > floor && h > un.roof - 4.0 {
                // A standing building underneath: a heavy, fast piece goes through it.
                let vn = -c.vel.dot(up) as f32;
                if let Some((t, i, bh)) = un.hit {
                    let vol = c.mass;
                    let sp2 = c.vel.length_squared() as f32;
                    if !c.bit && vn > 6.0 && h < un.roof + r + 1.0 && vol * sp2 > 3000.0 * bh * (bh / 10.0).max(1.0) && *crush_cool <= 0.0 && crush.len() < 4 && !crush.iter().any(|x| x.0 == t && x.1 == i) {
                        crush.push((t, i, c.pos, c.vel));
                        *crush_cool = 0.2;
                        c.under = Under::NONE;
                        continue;
                    }
                }
                floor = floor.max(un.roof);
            }
            if h < floor + r {
                c.pos += up * (floor + r - h);
                let vn = c.vel.dot(up);
                let tang = c.vel - up * vn;
                if un.water && floor <= un.ground + 0.01 {
                    c.state = 2;
                    c.pos -= up * (r * 3.0);
                    continue;
                }
                if vn < -3.0 && c.bounces < 3 {
                    c.bounces += 1;
                    c.vel = tang * 0.55 - up * vn * if c.bit { 0.3 } else { 0.15 };
                    c.omega *= 0.5;
                    if !c.bit || rng.f() < 0.3 {
                        landed.push((c.pos - up * r, c.half.max_element(), -vn as f32));
                    }
                    *rumble += 0.03;
                    // A big piece hitting hard breaks in two, unevenly, along its longest side.
                    if slab && c.bounces == 1 && -vn > 6.0 && n_chunks + bits.len() < MAX_CHUNKS - 600 && rng.f() < 0.8 {
                        let i = if c.half.x >= c.half.y.max(c.half.z) { 0 } else if c.half.y >= c.half.z { 1 } else { 2 };
                        let hl = c.half[i];
                        if hl > 1.2 {
                            let f = rng.range(0.3, 0.7);
                            let axw = c.rot * [DVec3::X, DVec3::Y, DVec3::Z][i];
                            let (h1, h2) = (hl * f, hl * (1.0 - f));
                            let mut o = c.clone();
                            o.half[i] = h2;
                            o.pos = c.pos + axw * (hl - h2) as f64;
                            c.half[i] = h1;
                            c.pos -= axw * (hl - h1) as f64;
                            let sep = axw * rng.range(1.0, 3.5) as f64;
                            c.vel -= sep;
                            o.vel += sep + (rng.dir() * 1.5).as_dvec3();
                            c.omega += (rng.dir() * rng.range(0.5, 2.5)).as_dvec3();
                            o.omega = (rng.dir() * rng.range(0.5, 3.0)).as_dvec3();
                            o.var = (rng.f() * N_SLAB as f32) as u32 % N_SLAB;
                            o.mesh = M_SLAB + o.var;
                            if c.skin != NO_SKIN {
                                splits.push((bits.len(), c.group, c.skin, i, -hl + 2.0 * h1, hl - h1, hl - h2));
                            }
                            let k = h2 / hl;
                            o.mass = c.mass * k;
                            c.mass *= 1.0 - k;
                            // A piece of facade keeps its place on the wall.
                            if c.face[2] != 0.0 {
                                let xs = c.face[2].signum();
                                if i == 0 {
                                    o.face[0] += xs * (hl - h2);
                                    c.face[0] -= xs * (hl - h1);
                                } else if i == 2 {
                                    o.face[1] -= hl - h2;
                                    c.face[1] += hl - h1;
                                }
                            }
                            bits.push(o);
                        }
                    }
                } else if tang.length() > 1.5 && is_rock(c.mesh) && c.bounces < 12 {
                    // Lumps roll on a bit.
                    c.bounces += 1;
                    c.vel = tang * 0.8;
                } else {
                    // On a heap: a piece that would perch on a steep spot slides down it.
                    if slab && fh > un.ground + 0.5 && c.bounces < 10 {
                        let reach = (c.half.x.max(c.half.z) as f64 * 1.6).max(2.0);
                        let (e, n, _) = geo::enu(c.pos);
                        let mut low = (floor, DVec3::ZERO);
                        for d in [e, -e, n, -n, (e + n) * 0.7071, (e - n) * 0.7071, (-e + n) * 0.7071, (-e - n) * 0.7071] {
                            let hn = fat(field, c.pos + d * reach).max(un.ground);
                            if hn < low.0 {
                                low = (hn, d);
                            }
                        }
                        if low.0 < floor - 0.9 * r {
                            c.bounces += 1;
                            c.vel = low.1 * (2.0 + rng.f() as f64 * 2.0);
                            continue;
                        }
                    }
                    c.state = 2;
                    c.vel = DVec3::ZERO;
                    // A building is mostly air: its pieces settle into the heap
                    // rather than stacking up solid.
                    let on_heap = floor > un.ground + 0.3;
                    if on_heap && !c.bit {
                        c.pos -= up * (r * rng.range(0.4, 1.4) as f64);
                    }
                    if slab {
                        // Slabs end up lying roughly flat, at odd angles.
                        let yaw = DQuat::from_axis_angle(up, rng.range(0.0, 6.28) as f64);
                        let tilt = DQuat::from_axis_angle(rng.dir().as_dvec3(), rng.range(-0.45, 0.45) as f64);
                        let (e, n, _) = geo::enu(c.pos);
                        // Shortest side up.
                        let flat = if c.half.y <= c.half.x.min(c.half.z) { DQuat::from_mat3(&glam::DMat3::from_cols(e, up, e.cross(up))) } else if c.half.x <= c.half.z { DQuat::from_mat3(&glam::DMat3::from_cols(up, n, up.cross(n))) } else { DQuat::from_mat3(&glam::DMat3::from_cols(e, e.cross(up).cross(e).normalize(), up)) };
                        // Either face up (a piece of facade shows its windows or its inside).
                        let flip = if rng.f() < 0.5 { DQuat::from_axis_angle(DVec3::X, std::f64::consts::PI) } else { DQuat::IDENTITY };
                        c.rot = (tilt * yaw * flat * flip).normalize();
                    }
                    // Covered in the dust it fell through.
                    let k = if c.bit { 0.3 } else { 0.45 };
                    for (j, d) in [0.42f32, 0.4, 0.37].iter().enumerate() {
                        c.col[j] += (d * 0.8 - c.col[j]) * k;
                    }
                    // The heap grows by a fraction of each piece and no higher than
                    // a few storeys: past that, pieces sink into it.
                    // (a thin slab stands for the storey of building around it)
                    let rb = if slab { (0.5 * c.mass.cbrt()).max(r as f32) as f64 } else { r };
                    let grow = if slab { 2.0 * rb * if on_heap { 0.12 } else { 0.5 } } else { r * 0.5 };
                    let top = (floor + grow).min(un.ground + 28.0).max(floor.min(un.ground + 28.0));
                    rests.push((c.pos, c.half.max_element() * if slab { 0.85 } else { 0.6 }, top));
                    if slab && rng.f() < 0.8 {
                        mounds.push((c.pos - up * r, up, c.half.max_element(), c.col));
                    }
                }
            }
        }
        chunks.retain(|c| !c.gone);
        // What is left of a curtain-wall panel that burst: its frame, a few
        // bent lengths of mullion.
        for &(p, half, out, v) in &shatter {
            if rng.f() < 0.6 && chunks.len() + bits.len() < MAX_CHUNKS - 300 {
                let len = (half.x.max(half.z) * rng.range(0.3, 0.6)).min(1.8);
                let mut m = Chunk::loose(p + (rng.dir() * half).as_dvec3(), Vec3::new(len, 0.035, 0.07), DQuat::from_axis_angle(rng.dir().as_dvec3(), rng.range(0.0, 6.0) as f64), (v + out * rng.range(1.0, 4.0) + rng.dir() * 2.0).as_dvec3(), (rng.dir() * rng.range(1.0, 5.0)).as_dvec3(), [0.2, 0.21, 0.22], M_CUBE);
                m.bit = true;
                bits.push(m);
            }
        }
        for (b, g, slot, axis, cut, sc, so) in splits {
            if let Some(gr) = structs.get_mut(g as usize).and_then(|s| s.skin.as_mut()) {
                bits[b].skin = gr.split(slot, axis, cut, sc, so);
            }
        }
        chunks.extend(bits);
        // Under every slab that comes to rest, a low mound of the smaller stuff
        // (broken block, plaster, dust) it lies in, so a pile is a heap with
        // slabs in it rather than blocks stacked on the street.
        for (p, up, s, col) in mounds {
            if chunks.len() >= MAX_CHUNKS - 200 {
                break;
            }
            let w = s * rng.range(0.8, 1.25);
            let half = Vec3::new(w * rng.range(0.9, 1.3), w * rng.range(0.12, 0.25), w * rng.range(0.9, 1.3));
            let rot = DQuat::from_rotation_arc(DVec3::Y, up) * DQuat::from_axis_angle(DVec3::Y, rng.range(0.0, 6.28) as f64);
            let tone = rng.range(0.8, 1.05);
            let mut m = Chunk::loose(p + up * (half.y * 0.25) as f64, half, rot, DVec3::ZERO, DVec3::ZERO, [(col[0] * 0.4 + 0.12) * tone, (col[1] * 0.4 + 0.115) * tone, (col[2] * 0.4 + 0.11) * tone], M_ROCK + (rng.f() * 3.0) as u32 % 3);
            m.state = 2;
            m.bit = false;
            chunks.push(m);
        }
        for (g, n) in left_dec {
            if let Some(s) = structs.get_mut(g as usize) {
                s.left = s.left.saturating_sub(n);
                if s.left == 0 && s.done_t < 0.0 {
                    s.done_t = 0.0;
                }
            }
        }
        for (p, rad, top) in rests {
            self.raise(p, rad, top);
        }
        if !self.structs.is_empty() && self.structs.iter().all(|s| s.done_t > 15.0) && !self.chunks.iter().any(|c| c.state == 0) {
            self.structs.clear();
        }

        // Dust and glass from what let go and what landed, within a budget per frame.
        // (a rate per second, not per frame, so it reads the same at any frame rate)
        let n_out = released.iter().filter(|r| r.1 != Vec3::ZERO).count() as f32;
        self.fx_acc[0] = (self.fx_acc[0] + dt * 45.0 + n_out * 0.35).min(10.0);
        self.fx_acc[1] = (self.fx_acc[1] + dt * 40.0).min(8.0);
        self.fx_acc[2] = (self.fx_acc[2] + dt * 30.0).min(8.0);
        let nr = released.len();
        let start = (self.rng.f() * nr as f32) as usize;
        for j in 0..nr {
            let (p, out, size, v, facade) = released[(start + j) % nr];
            if out != Vec3::ZERO && self.fx_acc[0] >= 1.0 {
                self.fx_acc[0] -= 1.0;
                self.jet(p, out, v, size.clamp(1.5, 8.0), emit);
            }
            if facade && self.fx_acc[1] >= 1.0 {
                self.fx_acc[1] -= 1.0;
                self.glass(p, size, out, v, 6, emit);
            }
        }
        for &(p, half, out, v) in shatter.iter().take(30) {
            self.glass(p, half.max_element() * 0.6, out, v, 12, emit);
        }
        landed.sort_by(|a, b| (b.1 * b.2).total_cmp(&(a.1 * a.2)));
        for (p, s, vn) in landed {
            if self.fx_acc[2] < 1.0 {
                break;
            }
            self.fx_acc[2] -= 1.0;
            let up = p.normalize().as_vec3();
            let r = &mut self.rng;
            let sz = s.clamp(1.0, 6.0) * (0.6 + (vn / 25.0).min(1.2));
            {
                let d = r.dir();
                let flat = (d - up * d.dot(up)).normalize_or_zero();
                emit.push((p, Particle { vel: (flat * r.range(3.0, 10.0) * (0.5 + vn / 30.0) + up * r.range(0.5, 3.0)).into(), life: r.range(6.0, 12.0), size0: sz * r.range(0.6, 1.0), size1: sz * r.range(3.0, 5.0), temp: 0.1, kind: GRIT, drag: 0.9, buoy: 0.3, seed: r.f(), alpha: r.range(0.3, 0.5), ..Default::default() }));
            }
        }
    }

    /// The crush front reached the street: a wall of dust rolls out from the
    /// foot of the building, through the streets around it.
    fn surge(&mut self, i: usize, emit: &mut Vec<(DVec3, Particle)>) {
        let (base, u, ext, height) = {
            let s = &self.structs[i];
            (s.base, s.u, s.ext, s.height)
        };
        let e = u.any_orthonormal_vector();
        let n = u.cross(e);
        let k = (height / 100.0).clamp(0.6, 2.5);
        let count = (110.0 + ext * 2.5 * k).min(320.0) as usize;
        for j in 0..count {
            let r = &mut self.rng;
            let a = j as f64 * 2.39996 + r.f() as f64;
            let d = e * a.cos() + n * a.sin();
            let at = base + d * (ext as f64 * r.range(0.5, 1.0) as f64) + u * r.range(1.0, 8.0) as f64;
            emit.push((at, Particle { vel: (d.as_vec3() * r.range(10.0, 26.0) * k.sqrt() + u.as_vec3() * r.range(0.5, 5.0)).into(), life: r.range(14.0, 26.0), size0: r.range(5.0, 9.0), size1: r.range(22.0, 40.0) * k.sqrt(), temp: 0.08, kind: GRIT, drag: r.range(0.3, 0.45), buoy: 0.12, seed: r.f(), alpha: r.range(0.35, 0.55), ..Default::default() }));
        }
        // And a column of it standing where the building was.
        for _ in 0..(24.0 * k) as usize {
            let r = &mut self.rng;
            let d = r.dir();
            let at = base + (d * ext).as_dvec3() + u * (r.f() * r.f() * height * 0.3) as f64;
            let flat = (d - u.as_vec3() * d.dot(u.as_vec3())).normalize_or_zero();
            emit.push((at, Particle { vel: (flat * r.range(3.0, 8.0) + u.as_vec3() * r.range(1.0, 4.0)).into(), life: r.range(12.0, 20.0), size0: r.range(6.0, 10.0), size1: r.range(22.0, 38.0), temp: 0.1, kind: GRIT, drag: 0.6, buoy: 0.2, seed: r.f(), alpha: r.range(0.18, 0.32), ..Default::default() }));
        }
        self.rumble += 1.0;
    }

    fn update_trees(&mut self, dt: f32, emit: &mut Vec<(DVec3, Particle)>) {
        // An inverted pendulum that falls faster the further it leans.
        let mut puffs: Vec<(DVec3, f32, bool)> = Vec::new();
        for t in self.falling.iter_mut().filter(|t| !t.done) {
            if t.delay > 0.0 {
                t.delay -= dt;
                continue;
            }
            if !t.started {
                t.started = true;
                puffs.push((t.pos + t.pos.normalize() * (t.height * 0.6) as f64, t.crown_r, true));
            }
            t.vel += 1.7 * 9.81 / (t.height * 0.55).max(1.5) * (t.ang.max(0.04)).sin() * dt;
            t.ang += t.vel * dt;
            if t.ang >= t.rest {
                t.ang = t.rest;
                if !t.landed {
                    t.landed = true;
                    let (e, n, _) = geo::enu(t.pos);
                    let tip = t.pos + (e * t.dir.x as f64 + n * t.dir.y as f64) * (t.height * 0.7) as f64;
                    puffs.push((tip, t.crown_r, false));
                    self.rumble += 0.15;
                }
                // The crown gives, so it bounces once or twice before settling.
                if t.vel > 0.5 {
                    t.vel = -t.vel * 0.22;
                } else {
                    t.vel = 0.0;
                    t.done = true;
                }
            }
        }
        for (p, r, start) in puffs {
            self.leaves(p, r, if start { 9 } else { 16 }, emit);
            if !start {
                self.dust(p, r * 0.5, 2, DUST, emit);
            }
        }
    }

    pub fn instances(&self, cam: DVec3, insts: &mut Vec<(u32, Inst)>, trees: &mut Vec<(u32, TreeInst)>) {
        let mut n_bars = 0;
        for c in &self.chunks {
            let o = (c.pos - cam).as_vec3();
            let far = if c.bit { 2500.0 } else { 7000.0 };
            if o.length_squared() > far * far {
                continue;
            }
            // Standing walls and roof of a building from the 3D tiles: its mesh is all there is to see.
            let photo = c.skin != NO_SKIN;
            if photo && c.state == 0 && c.outer {
                continue;
            }
            let r3 = Mat3::from_quat(c.rot.as_quat());
            let m = r3 * Mat3::from_diagonal(c.half);
            // Far off, a broken piece is a few pixels: the plain box will do.
            let d2 = o.length_squared();
            let mesh = if is_slab(c.mesh) && d2 > (160.0 * c.half.max_element()).powi(2) { M_CUBE } else { c.mesh };
            // (its pieces are bare: the mesh on them shows the wall)
            let (mat, face) = if photo { (1.0, [0.0; 4]) } else { (c.mat, c.face) };
            insts.push((mesh, Inst { m: [[m.x_axis.x, m.y_axis.x, m.z_axis.x, o.x], [m.x_axis.y, m.y_axis.y, m.z_axis.y, o.y], [m.x_axis.z, m.y_axis.z, m.z_axis.z, o.z]], tint: [c.col[0], c.col[1], c.col[2], mat], ext: face }));
            // Bars sticking out of its broken edges, close enough to be seen.
            if c.rebar && mesh != M_CUBE && is_slab(mesh) && d2 < 320.0 * 320.0 && n_bars < 3000 {
                let up = r3 * Vec3::Y;
                for (k, ed) in self.edges[c.var as usize % self.edges.len()].iter().enumerate() {
                    let t = ed.n.perp();
                    let along = (r3 * Vec3::new(t.x * c.half.x, 0.0, t.y * c.half.z)).normalize_or_zero();
                    let hl = Vec2::new(t.x * c.half.x, t.y * c.half.z).length() * ed.half;
                    let nw = (r3 * Vec3::new(ed.n.x / c.half.x, 0.0, ed.n.y / c.half.z)).normalize_or_zero();
                    if hl < 0.4 || nw == Vec3::ZERO {
                        continue;
                    }
                    // Tufts of a couple of metres at most, spread along a long edge.
                    let sc = hl.clamp(0.7, 2.2);
                    let nt = (hl / sc).floor().clamp(1.0, 3.0) as i32;
                    let mid = o + r3 * Vec3::new(ed.mid.x * c.half.x, 0.0, ed.mid.y * c.half.z);
                    let z = nw.cross(up);
                    for j in 0..nt {
                        let off = (j as f32 - (nt - 1) as f32 * 0.5) * 2.0 * sc;
                        let p = mid + along * off;
                        let (x, y, z) = (nw * sc, up * sc, z * sc);
                        let var = (c.var + k as u32 + j as u32) % N_REBAR;
                        insts.push((M_REBAR + var, Inst { m: [[x.x, y.x, z.x, p.x], [x.y, y.y, z.y, p.y], [x.z, y.z, z.z, p.z]], tint: [c.col[0], c.col[1], c.col[2], 1.0], ext: [0.0; 4] }));
                        n_bars += 1;
                    }
                }
            }
        }
        for t in &self.falling {
            let o = (t.pos - cam).as_vec3();
            if o.length_squared() > 3200.0 * 3200.0 {
                continue;
            }
            let mut i = t.inst;
            i.a = [o.x, o.y, o.z, t.inst.a[3]];
            // A hair of lean even before it moves, so the shader treats it as loose.
            let a = t.ang.max(0.012);
            i.b = [t.dir.x * a, t.dir.y * a, t.inst.b[2], t.inst.b[3]];
            trees.push((t.mesh, i));
        }
    }

    /// Where every piece carrying a destroyed building's mesh is this frame
    /// (skin.rs); meshes none of whose pieces are left are let go.
    pub fn skin_frame(&mut self, cam: DVec3) {
        for s in &mut self.structs {
            if let Some(g) = &mut s.skin {
                g.xf.clear();
                g.xf.resize(g.len(), [[0.0; 4]; 4]);
            }
        }
        let mut live = vec![false; self.structs.len()];
        for c in &self.chunks {
            if c.skin == NO_SKIN {
                continue;
            }
            let Some(g) = self.structs.get_mut(c.group as usize).and_then(|s| s.skin.as_mut()) else { continue };
            if (c.skin as usize) < g.xf.len() {
                let o = (c.pos - cam).as_vec3();
                g.xf[c.skin as usize] = g.slot_xf(c.skin, o, c.rot, c.col);
                live[c.group as usize] = true;
            }
        }
        for (s, l) in self.structs.iter_mut().zip(live) {
            if !l {
                s.skin = None;
            }
        }
    }

    /// Copies new meshes, and triangles that changed pieces, to the GPU.
    pub fn upload_skins(&mut self, r: &crate::render::Renderer) {
        for g in self.structs.iter_mut().filter_map(|s| s.skin.as_mut()) {
            g.upload(r);
        }
    }

    /// This frame's slots, all meshes' one after the other, and each mesh with
    /// where its slots start.
    pub fn skin_draws(&self) -> (Vec<[[f32; 4]; 4]>, Vec<(&crate::render::SkinGpu, u32)>) {
        let (mut slots, mut draws) = (Vec::new(), Vec::new());
        for g in self.structs.iter().filter_map(|s| s.skin.as_ref()) {
            // (each mesh's slots start on a 256-byte boundary)
            while slots.len() % 4 != 0 {
                slots.push([[0.0; 4]; 4]);
            }
            let first = slots.len() as u32;
            slots.extend_from_slice(&g.xf);
            for gpu in g.gpus() {
                draws.push((gpu, first));
            }
        }
        (slots, draws)
    }

    pub fn clear(&mut self) {
        self.falling.clear();
        self.chunks.clear();
        self.structs.clear();
        self.field.clear();
    }
}
