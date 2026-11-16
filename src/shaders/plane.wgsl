// The aircraft (and its wreckage): paint scheme, panel seams and rivets,
// weathering, the cockpit's instruments and radios, and the transparent pass
// (windows and the propeller blur). Every vertex carries its body-frame
// position at rest (L), so whatever is painted here stays put on moving parts.
// Material, pattern, light and instrument ids follow airframe.rs / cabin.rs.

// tear: up to two breaks through the body frame (plane normal, offset); dmg:
// the mode of each (see damage.rs), scrapes 0..1, raggedness of the edge (m);
// dmg2: oil film on the windshield, crack centre z and y, crack size.
struct Obj { model: mat4x4f, tint: vec4f, glow: vec4f, params: array<vec4f, 6>, tear: array<vec4f, 2>, dmg: vec4f, dmg2: vec4f };
@group(1) @binding(0) var<uniform> O: Obj;
@group(2) @binding(0) var<uniform> CS: Cascade;
// The glass pass only: the lit scene before the glass went on, and its depth.
@group(2) @binding(1) var t_scene: texture_2d<f32>;
@group(2) @binding(2) var t_gdepth: texture_depth_2d;

// params[0] airspeed kt, altitude ft, vertical speed fpm, heading deg
// params[1] pitch deg, bank deg, turn rate deg/s, slip
// params[2] rpm, manifold pressure, fuel left, fuel right (gal)
// params[3] oil temp F, oil pressure psi, EGT 0..1, amps
// params[4] suction, clock (s), gear position, flaps deg
// params[5] nav lights (on: 1000 + the compass card's heading), strobe flash, beacon flash, landing light

const M_PAINT = 0u; const M_LIVERY = 1u; const M_ALU = 2u; const M_STEEL = 3u; const M_RUBBER = 4u;
const M_PLASTIC = 5u; const M_FABRIC = 6u; const M_GLASS = 7u; const M_LAMP = 8u; const M_GAUGE = 9u;
const M_BLACK = 10u; const M_EXHAUST = 11u; const M_SCREW = 12u; const M_PROP = 13u; const M_DISPLAY = 14u;
const M_SATIN = 15u; const M_PROPDISC = 16u; const M_PLUME = 17u;
const P_FUSE = 1u; const P_WING = 2u; const P_CTRL = 3u; const P_COWL = 4u; const P_TAIL = 5u; const P_TREAD = 6u;
const P_LINING = 7u; const P_SEAT = 8u; const P_CARPET = 9u; const P_PANEL = 10u; const P_PLACARD = 11u; const P_KNURL = 12u;
const P_NACELLE = 13u; const P_LEGEND = 14u; const P_FLEECE = 15u; const P_SPINNER = 16u; const P_HORNET = 17u;
// The Hornet's cockpit (aircraft/fa18/pit.rs, pit18.wgsl).
const P_PIT = 18u; const P_MESH = 19u; const P_WEBBING = 20u; const P_NONSLIP = 21u; const P_STRIPE = 22u; const P_HOSE = 23u;

// The Hornet's paint (assets/fa18_paint.jpg): it seen from above, from below
// and from the left, one layer each.
@group(1) @binding(1) var t_paint: texture_2d_array<f32>;

// Which aircraft is being drawn (aircraft::Spec::id, carried in glow.y): 0 the
// light single, 1 the 747, 2 the Hornet, 3 and 4 the 747 five and twenty times
// the size (aircraft/b747big.rs), drawn as the 747 with its body frame (L)
// shrunk back by `kscale`. Their windows, wells, paint and panel lines differ. tint.y and
// tint.z carry the engines: how far the afterburners are lit, and how hard
// they are running dry (0 idle .. 1).
fn kind() -> u32 { let k = u32(O.glow.y + 0.5); return select(k, 1u, k == 3u || k == 4u); }
// How many times its model's size the aircraft is (aircraft::Spec::size_k,
// carried in glow.z; 0 for everything else).
fn kscale() -> f32 { return max(O.glow.z, 1.0); }
// The engines, of the aircraft being flown only (wreckage and parts that have come off carry a glow.x).
fn burner() -> f32 { return O.tint.y * step(O.glow.x, 0.001); }
fn dry_power() -> f32 { return O.tint.z * step(O.glow.x, 0.001); }

const WHITE_PAINT = vec3f(0.80, 0.80, 0.78);
const NAVY = vec3f(0.010, 0.024, 0.075);
const ACCENT = vec3f(0.34, 0.018, 0.012);

struct VOut {
  @builtin(position) clip: vec4f,
  @location(0) P: vec3f,
  @location(1) N: vec3f,
  @location(2) col: vec4f,
  @location(3) uv: vec2f,
  @location(4) @interpolate(flat) mat: vec4f,
  @location(5) wear: f32,
  @location(6) L: vec3f,
  @location(7) Nl: vec3f,
};
struct GOut { @location(0) c: vec4f, @location(1) n: vec4f, @location(2) amb: f32 };
struct SOut { @builtin(position) clip: vec4f, @location(0) L: vec3f, @location(1) @interpolate(flat) mat: vec4f };

@vertex fn vs(@location(0) pos: vec3f, @location(1) nrm: vec3f, @location(2) col: vec4f, @location(3) uv: vec2f, @location(4) mat: vec4f) -> VOut {
  let P = (O.model * vec4f(pos, 1.0)).xyz;
  let N = (O.model * vec4f(nrm, 0.0)).xyz;
  return VOut(G.view_proj * vec4f(P, 1.0), P, N, col, uv, mat, mat.z, pos / kscale(), nrm);
}
@vertex fn vs_shadow(@location(0) pos: vec3f, @location(4) mat: vec4f) -> SOut {
  return SOut(CS.vp * vec4f((O.model * vec4f(pos, 1.0)).xyz, 1.0), pos / kscale(), mat);
}
@fragment fn fs_shadow(in: SOut) {
  // Sunlight comes in through the windows, the gear wells stay open, and
  // nothing is cast by what has been torn off.
  if (cut_here(in.L, u32(in.mat.y + 0.5)) || u32(in.mat.x + 0.5) == M_GLASS || u32(in.mat.x + 0.5) == M_PROPDISC || u32(in.mat.x + 0.5) == M_PLUME || tear_d(in.L) > 0.0) { discard; }
}

// ---------------------------------------------------------------- helpers

fn sat(x: f32) -> f32 { return clamp(x, 0.0, 1.0); }
fn nz(p: vec3f) -> vec4f { return textureSampleLevel(t_noise, s_repeat, p, 0.0); }
fn seg_d(p: vec2f, a: vec2f, b: vec2f) -> f32 {
  let pa = p - a;
  let ba = b - a;
  let h = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-9), 0.0, 1.0);
  return length(pa - ba * h);
}
fn line(d: f32, w: f32) -> f32 { return exp(-d * d / (w * w)); }
fn rot2(p: vec2f, a: f32) -> vec2f { let c = cos(a); let s = sin(a); return vec2f(c * p.x - s * p.y, s * p.x + c * p.y); }
// A row of round-headed rivets along `along`, `across` from the row: 0..1 dome.
fn rivets(along: f32, across: f32, pitch: f32, r: f32) -> f32 {
  let a = (fract(along / pitch + 0.5) - 0.5) * pitch;
  return max(0.0, 1.0 - (a * a + across * across) / (r * r));
}

// Exact signed distance to a polygon of n <= 5 points (negative inside).
fn sd_poly(p: vec2f, vin: array<vec2f, 5>, n: i32) -> f32 {
  var v = vin;
  var d = dot(p - v[0], p - v[0]);
  var s = 1.0;
  var j = n - 1;
  for (var i = 0; i < n; i++) {
    let e = v[j] - v[i];
    let w = p - v[i];
    let b = w - e * clamp(dot(w, e) / dot(e, e), 0.0, 1.0);
    d = min(d, dot(b, b));
    let c1 = p.y >= v[i].y;
    let c2 = p.y < v[j].y;
    let c3 = e.x * w.y > e.y * w.x;
    if ((c1 && c2 && c3) || (!c1 && !c2 && !c3)) { s = -s; }
    j = i;
  }
  return s * sqrt(d);
}
fn quad(a: vec2f, b: vec2f, c: vec2f, d: vec2f) -> array<vec2f, 5> { return array<vec2f, 5>(a, b, c, d, a); }

// Window openings (negative inside); matches c172::window / b747::window.
fn window(p: vec3f) -> f32 {
  if (kind() == 1u) { return window747(p); }
  // The Hornet's canopy is one blown piece: all of it is window.
  if (kind() == 2u) { return -1.0; }
  let zy = vec2f(p.z, p.y);
  var d = sd_poly(zy, array<vec2f, 5>(vec2f(-1.72, 0.31), vec2f(-1.52, 0.07), vec2f(-1.03, 0.64), vec2f(-1.0, 0.80), vec2f(-1.75, 0.80)), 5) + 0.01;
  if (abs(p.x) > 0.3 && p.y < 0.6) {
    d = min(d, sd_poly(zy, quad(vec2f(-1.36, 0.12), vec2f(-0.60, 0.12), vec2f(-0.60, 0.48), vec2f(-0.98, 0.48)), 4) - 0.04);
    d = min(d, sd_poly(zy, quad(vec2f(-0.42, 0.12), vec2f(0.62, 0.12), vec2f(0.42, 0.48), vec2f(-0.42, 0.48)), 4) - 0.04);
  }
  if (p.y > 0.35) {
    d = min(d, sd_poly(vec2f(p.z, p.x), quad(vec2f(0.84, -0.30), vec2f(1.48, -0.17), vec2f(1.48, 0.17), vec2f(0.84, 0.30)), 4) - 0.04);
  }
  return d;
}
// Landing gear wells (only the skin has them); matches c172::well / b747::well.
fn well(p: vec3f) -> f32 {
  if (kind() == 1u) { return well747(p); }
  var d = 1.0;
  if (p.y < -0.40) {
    d = min(d, sd_poly(vec2f(p.z, p.x), quad(vec2f(-3.0, -0.135), vec2f(-1.915, -0.135), vec2f(-1.915, 0.135), vec2f(-3.0, 0.135)), 4) + 0.005);
    let q = vec2f(p.z, abs(p.x));
    let trough = sd_poly(q, quad(vec2f(0.215, 0.37), vec2f(1.15, 0.183), vec2f(1.15, 0.427), vec2f(0.215, 0.55)), 4);
    let bay = sd_poly(q, array<vec2f, 5>(vec2f(1.14, 0.03), vec2f(1.545, 0.03), vec2f(1.545, 0.43), vec2f(1.27, 0.505), vec2f(1.14, 0.44)), 5);
    d = min(d, min(trough, bay) + 0.005);
  }
  return d;
}
fn cut_here(L: vec3f, pt: u32) -> bool {
  // The 747's wing gear passes through the lower wing skin as well.
  let wing_well = kind() == 1u && pt == P_WING && well(L) < 0.0;
  return wing_well || ((pt == P_FUSE || pt == P_COWL || pt == P_LINING) && (window(L) < 0.0 || (pt != P_LINING && well(L) < 0.0)));
}
fn torn() -> bool { return O.dmg.x != 0.0 || O.dmg.y != 0.0; }
// Signed distance past the piece's break(s): positive where it has been torn
// away. The edge wanders by dmg.w, the same way on both sides of the break,
// so the stump and the part that left share one ragged line.
fn tear_d(L0: vec3f) -> f32 {
  if (!torn()) { return -1e3; }
  // (the breaks are in the full-size body frame)
  let L = L0 * kscale();
  let jag = O.dmg.w;
  // A few big jags, torn skin between them, and fine shredding along the edge.
  let n1 = (nz(L * 1.4).r - 0.5) * 1.6 + (nz(L * 4.5).b - 0.5) * 0.7;
  let n2 = nz(L * 16.0).g - 0.5;
  var s = -1e3;
  for (var i = 0; i < 2; i++) {
    let m = select(O.dmg.x, O.dmg.y, i == 1);
    if (m == 0.0) { continue; }
    let t = O.tear[i];
    var q = L;
    if (abs(m) > 1.5) { q.x = abs(q.x); }
    var v = dot(q, t.xyz) - t.w + n1 * jag + n2 * jag * 0.35;
    if (m < 0.0) { v = -v; }
    s = max(s, v);
  }
  return s;
}
// Cabin door outline in (z, y).
fn door(L: vec3f) -> f32 {
  return sd_poly(vec2f(L.z, L.y), array<vec2f, 5>(vec2f(-1.47, -0.42), vec2f(-0.47, -0.42), vec2f(-0.47, 0.59), vec2f(-0.99, 0.59), vec2f(-1.47, 0.03)), 5);
}
fn baggage_door(L: vec3f) -> f32 {
  let q = vec2f(L.z - 1.25, L.y + 0.14);
  let b = abs(q) - vec2f(0.20, 0.17) + 0.03;
  return length(max(b, vec2f(0.0))) + min(max(b.x, b.y), 0.0) - 0.03;
}

// ---------------------------------------------------------------- lettering

// Stroke font: digits, N E S W and a point (glyphs 0-14; 15 is a space).
// (const, not var<private>: Metal copies a private array into every
// invocation, which made the aircraft spill registers on every pixel.)
const GSEG = array<vec4f, 237>(
  vec4f(0.25, 0.0, 0.75, 0.0),
  vec4f(0.75, 0.0, 1.0, 0.4),
  vec4f(1.0, 0.4, 1.0, 1.6),
  vec4f(1.0, 1.6, 0.75, 2.0),
  vec4f(0.75, 2.0, 0.25, 2.0),
  vec4f(0.25, 2.0, 0.0, 1.6),
  vec4f(0.0, 1.6, 0.0, 0.4),
  vec4f(0.0, 0.4, 0.25, 0.0),
  vec4f(0.5, 0.0, 0.5, 2.0),
  vec4f(0.5, 2.0, 0.2, 1.65),
  vec4f(0.0, 1.6, 0.25, 2.0),
  vec4f(0.25, 2.0, 0.75, 2.0),
  vec4f(0.75, 2.0, 1.0, 1.6),
  vec4f(1.0, 1.6, 1.0, 1.25),
  vec4f(1.0, 1.25, 0.0, 0.0),
  vec4f(0.0, 0.0, 1.0, 0.0),
  vec4f(0.0, 1.7, 0.25, 2.0),
  vec4f(0.25, 2.0, 0.75, 2.0),
  vec4f(0.75, 2.0, 1.0, 1.65),
  vec4f(1.0, 1.65, 1.0, 1.35),
  vec4f(1.0, 1.35, 0.7, 1.05),
  vec4f(0.7, 1.05, 0.35, 1.05),
  vec4f(0.7, 1.05, 1.0, 0.75),
  vec4f(1.0, 0.75, 1.0, 0.35),
  vec4f(1.0, 0.35, 0.75, 0.0),
  vec4f(0.75, 0.0, 0.25, 0.0),
  vec4f(0.25, 0.0, 0.0, 0.3),
  vec4f(0.75, 0.0, 0.75, 2.0),
  vec4f(0.75, 2.0, 0.0, 0.6),
  vec4f(0.0, 0.6, 1.0, 0.6),
  vec4f(1.0, 2.0, 0.05, 2.0),
  vec4f(0.05, 2.0, 0.0, 1.1),
  vec4f(0.0, 1.1, 0.65, 1.2),
  vec4f(0.65, 1.2, 1.0, 0.85),
  vec4f(1.0, 0.85, 1.0, 0.35),
  vec4f(1.0, 0.35, 0.7, 0.0),
  vec4f(0.7, 0.0, 0.25, 0.0),
  vec4f(0.25, 0.0, 0.0, 0.3),
  vec4f(0.9, 1.9, 0.6, 2.0),
  vec4f(0.6, 2.0, 0.3, 2.0),
  vec4f(0.3, 2.0, 0.0, 1.5),
  vec4f(0.0, 1.5, 0.0, 0.4),
  vec4f(0.0, 0.4, 0.3, 0.0),
  vec4f(0.3, 0.0, 0.7, 0.0),
  vec4f(0.7, 0.0, 1.0, 0.35),
  vec4f(1.0, 0.35, 1.0, 0.8),
  vec4f(1.0, 0.8, 0.7, 1.15),
  vec4f(0.7, 1.15, 0.3, 1.15),
  vec4f(0.3, 1.15, 0.0, 0.85),
  vec4f(0.0, 2.0, 1.0, 2.0),
  vec4f(1.0, 2.0, 0.35, 0.0),
  vec4f(0.5, 1.1, 0.15, 1.3),
  vec4f(0.15, 1.3, 0.1, 1.75),
  vec4f(0.1, 1.75, 0.35, 2.0),
  vec4f(0.35, 2.0, 0.65, 2.0),
  vec4f(0.65, 2.0, 0.9, 1.75),
  vec4f(0.9, 1.75, 0.85, 1.3),
  vec4f(0.85, 1.3, 0.5, 1.1),
  vec4f(0.5, 1.1, 0.05, 0.85),
  vec4f(0.05, 0.85, 0.0, 0.35),
  vec4f(0.0, 0.35, 0.3, 0.0),
  vec4f(0.3, 0.0, 0.7, 0.0),
  vec4f(0.7, 0.0, 1.0, 0.35),
  vec4f(1.0, 0.35, 0.95, 0.85),
  vec4f(0.95, 0.85, 0.5, 1.1),
  vec4f(0.1, 0.1, 0.4, 0.0),
  vec4f(0.4, 0.0, 0.7, 0.0),
  vec4f(0.7, 0.0, 1.0, 0.5),
  vec4f(1.0, 0.5, 1.0, 1.6),
  vec4f(1.0, 1.6, 0.7, 2.0),
  vec4f(0.7, 2.0, 0.3, 2.0),
  vec4f(0.3, 2.0, 0.0, 1.65),
  vec4f(0.0, 1.65, 0.0, 1.2),
  vec4f(0.0, 1.2, 0.3, 0.85),
  vec4f(0.3, 0.85, 0.7, 0.85),
  vec4f(0.7, 0.85, 1.0, 1.15),
  vec4f(0.0, 0.0, 0.0, 2.0),
  vec4f(0.0, 2.0, 1.0, 0.0),
  vec4f(1.0, 0.0, 1.0, 2.0),
  vec4f(1.0, 0.0, 0.0, 0.0),
  vec4f(0.0, 0.0, 0.0, 2.0),
  vec4f(0.0, 2.0, 1.0, 2.0),
  vec4f(0.0, 1.0, 0.75, 1.0),
  vec4f(1.0, 1.7, 0.75, 2.0),
  vec4f(0.75, 2.0, 0.25, 2.0),
  vec4f(0.25, 2.0, 0.0, 1.7),
  vec4f(0.0, 1.7, 0.0, 1.35),
  vec4f(0.0, 1.35, 0.25, 1.05),
  vec4f(0.25, 1.05, 0.75, 0.95),
  vec4f(0.75, 0.95, 1.0, 0.65),
  vec4f(1.0, 0.65, 1.0, 0.3),
  vec4f(1.0, 0.3, 0.75, 0.0),
  vec4f(0.75, 0.0, 0.25, 0.0),
  vec4f(0.25, 0.0, 0.0, 0.3),
  vec4f(0.0, 2.0, 0.25, 0.0),
  vec4f(0.25, 0.0, 0.5, 1.2),
  vec4f(0.5, 1.2, 0.75, 0.0),
  vec4f(0.75, 0.0, 1.0, 2.0),
  vec4f(0.45, 0.08, 0.55, 0.08),
  vec4f(0.0, 0.0, 0.5, 2.0),
  vec4f(0.5, 2.0, 1.0, 0.0),
  vec4f(0.2, 0.8, 0.8, 0.8),
  vec4f(0.0, 0.0, 0.0, 2.0),
  vec4f(0.0, 2.0, 0.7, 2.0),
  vec4f(0.7, 2.0, 0.95, 1.75),
  vec4f(0.95, 1.75, 0.95, 1.3),
  vec4f(0.95, 1.3, 0.7, 1.05),
  vec4f(0.7, 1.05, 0.0, 1.05),
  vec4f(0.7, 1.05, 1.0, 0.75),
  vec4f(1.0, 0.75, 1.0, 0.3),
  vec4f(1.0, 0.3, 0.7, 0.0),
  vec4f(0.7, 0.0, 0.0, 0.0),
  vec4f(1.0, 1.7, 0.7, 2.0),
  vec4f(0.7, 2.0, 0.3, 2.0),
  vec4f(0.3, 2.0, 0.0, 1.6),
  vec4f(0.0, 1.6, 0.0, 0.4),
  vec4f(0.0, 0.4, 0.3, 0.0),
  vec4f(0.3, 0.0, 0.7, 0.0),
  vec4f(0.7, 0.0, 1.0, 0.3),
  vec4f(0.0, 0.0, 0.0, 2.0),
  vec4f(0.0, 2.0, 0.6, 2.0),
  vec4f(0.6, 2.0, 1.0, 1.5),
  vec4f(1.0, 1.5, 1.0, 0.5),
  vec4f(1.0, 0.5, 0.6, 0.0),
  vec4f(0.6, 0.0, 0.0, 0.0),
  vec4f(1.0, 2.0, 0.0, 2.0),
  vec4f(0.0, 2.0, 0.0, 0.0),
  vec4f(0.0, 1.05, 0.75, 1.05),
  vec4f(1.0, 1.7, 0.7, 2.0),
  vec4f(0.7, 2.0, 0.3, 2.0),
  vec4f(0.3, 2.0, 0.0, 1.6),
  vec4f(0.0, 1.6, 0.0, 0.4),
  vec4f(0.0, 0.4, 0.3, 0.0),
  vec4f(0.3, 0.0, 0.7, 0.0),
  vec4f(0.7, 0.0, 1.0, 0.3),
  vec4f(1.0, 0.3, 1.0, 0.9),
  vec4f(1.0, 0.9, 0.55, 0.9),
  vec4f(0.0, 0.0, 0.0, 2.0),
  vec4f(1.0, 0.0, 1.0, 2.0),
  vec4f(0.0, 1.05, 1.0, 1.05),
  vec4f(0.5, 0.0, 0.5, 2.0),
  vec4f(0.2, 0.0, 0.8, 0.0),
  vec4f(0.2, 2.0, 0.8, 2.0),
  vec4f(1.0, 2.0, 1.0, 0.4),
  vec4f(1.0, 0.4, 0.7, 0.0),
  vec4f(0.7, 0.0, 0.3, 0.0),
  vec4f(0.3, 0.0, 0.0, 0.4),
  vec4f(0.0, 0.0, 0.0, 2.0),
  vec4f(1.0, 2.0, 0.0, 0.8),
  vec4f(0.35, 1.15, 1.0, 0.0),
  vec4f(0.0, 2.0, 0.0, 0.0),
  vec4f(0.0, 0.0, 1.0, 0.0),
  vec4f(0.0, 0.0, 0.0, 2.0),
  vec4f(0.0, 2.0, 0.5, 0.9),
  vec4f(0.5, 0.9, 1.0, 2.0),
  vec4f(1.0, 2.0, 1.0, 0.0),
  vec4f(0.3, 0.0, 0.7, 0.0),
  vec4f(0.7, 0.0, 1.0, 0.4),
  vec4f(1.0, 0.4, 1.0, 1.6),
  vec4f(1.0, 1.6, 0.7, 2.0),
  vec4f(0.7, 2.0, 0.3, 2.0),
  vec4f(0.3, 2.0, 0.0, 1.6),
  vec4f(0.0, 1.6, 0.0, 0.4),
  vec4f(0.0, 0.4, 0.3, 0.0),
  vec4f(0.0, 0.0, 0.0, 2.0),
  vec4f(0.0, 2.0, 0.7, 2.0),
  vec4f(0.7, 2.0, 1.0, 1.7),
  vec4f(1.0, 1.7, 1.0, 1.2),
  vec4f(1.0, 1.2, 0.7, 0.9),
  vec4f(0.7, 0.9, 0.0, 0.9),
  vec4f(0.3, 0.0, 0.7, 0.0),
  vec4f(0.7, 0.0, 1.0, 0.4),
  vec4f(1.0, 0.4, 1.0, 1.6),
  vec4f(1.0, 1.6, 0.7, 2.0),
  vec4f(0.7, 2.0, 0.3, 2.0),
  vec4f(0.3, 2.0, 0.0, 1.6),
  vec4f(0.0, 1.6, 0.0, 0.4),
  vec4f(0.0, 0.4, 0.3, 0.0),
  vec4f(0.6, 0.4, 1.05, -0.1),
  vec4f(0.0, 0.0, 0.0, 2.0),
  vec4f(0.0, 2.0, 0.7, 2.0),
  vec4f(0.7, 2.0, 1.0, 1.7),
  vec4f(1.0, 1.7, 1.0, 1.2),
  vec4f(1.0, 1.2, 0.7, 0.9),
  vec4f(0.7, 0.9, 0.0, 0.9),
  vec4f(0.5, 0.9, 1.0, 0.0),
  vec4f(0.0, 2.0, 1.0, 2.0),
  vec4f(0.5, 2.0, 0.5, 0.0),
  vec4f(0.0, 2.0, 0.0, 0.4),
  vec4f(0.0, 0.4, 0.3, 0.0),
  vec4f(0.3, 0.0, 0.7, 0.0),
  vec4f(0.7, 0.0, 1.0, 0.4),
  vec4f(1.0, 0.4, 1.0, 2.0),
  vec4f(0.0, 2.0, 0.5, 0.0),
  vec4f(0.5, 0.0, 1.0, 2.0),
  vec4f(0.0, 0.0, 1.0, 2.0),
  vec4f(0.0, 2.0, 1.0, 0.0),
  vec4f(0.0, 2.0, 0.5, 1.0),
  vec4f(0.5, 1.0, 1.0, 2.0),
  vec4f(0.5, 1.0, 0.5, 0.0),
  vec4f(0.0, 2.0, 1.0, 2.0),
  vec4f(1.0, 2.0, 0.0, 0.0),
  vec4f(0.0, 0.0, 1.0, 0.0),
  vec4f(0.05, 0.0, 0.95, 2.0),
  vec4f(0.15, 1.0, 0.85, 1.0),
  vec4f(0.1, 1.0, 0.9, 1.0),
  vec4f(0.5, 0.6, 0.5, 1.4),
  vec4f(0.5, 0.45, 0.5, 0.5),
  vec4f(0.5, 1.5, 0.5, 1.55),
  vec4f(0.25, 1.55, 0.6, 1.55),
  vec4f(0.6, 1.55, 0.6, 1.95),
  vec4f(0.6, 1.95, 0.25, 1.95),
  vec4f(0.25, 1.95, 0.25, 1.55),
  vec4f(1.0, 1.8, 0.0, 1.0),
  vec4f(0.0, 1.0, 1.0, 0.2),
  vec4f(0.0, 1.8, 1.0, 1.0),
  vec4f(1.0, 1.0, 0.0, 0.2),
  vec4f(0.75, 2.1, 0.35, 1.6),
  vec4f(0.35, 1.6, 0.35, 0.4),
  vec4f(0.35, 0.4, 0.75, -0.1),
  vec4f(0.25, 2.1, 0.65, 1.6),
  vec4f(0.65, 1.6, 0.65, 0.4),
  vec4f(0.65, 0.4, 0.25, -0.1),
  vec4f(0.75, 2.1, 0.3, 2.1),
  vec4f(0.3, 2.1, 0.3, -0.1),
  vec4f(0.3, -0.1, 0.75, -0.1),
  vec4f(0.25, 2.1, 0.7, 2.1),
  vec4f(0.7, 2.1, 0.7, -0.1),
  vec4f(0.7, -0.1, 0.25, -0.1),
  vec4f(0.5, 0.5, 0.5, 1.5),
  vec4f(0.1, 0.75, 0.9, 1.25),
  vec4f(0.1, 1.25, 0.9, 0.75),
  vec4f(0.5, 2.0, 0.45, 1.5),
  vec4f(0.05, 0.05, 0.95, 0.05),
  vec4f(0.95, 0.05, 0.95, 1.95),
  vec4f(0.95, 1.95, 0.05, 1.95),
  vec4f(0.05, 1.95, 0.05, 0.05)
);
const GSTART = array<u32, 53>(0u, 8u, 10u, 16u, 27u, 30u, 38u, 49u, 51u, 65u, 76u, 79u, 83u, 94u, 98u, 99u, 99u, 102u, 112u, 119u, 125u, 128u, 137u, 140u, 143u, 147u, 150u, 152u, 156u, 164u, 170u, 179u, 186u, 188u, 193u, 195u, 197u, 200u, 203u, 204u, 205u, 207u, 209u, 213u, 215u, 217u, 220u, 223u, 226u, 229u, 232u, 233u, 237u);

fn glyph_d(g: u32, p: vec2f) -> f32 {
  var d = 1e3;
  let s = GSTART[g];
  let e = GSTART[g + 1u];
  for (var i = s; i < e; i++) {
    let q = GSEG[i];
    d = min(d, seg_d(p, q.xy, q.zw));
  }
  return d;
}
// A packed string (4 bits a glyph, first lowest) at p in glyph units: glyphs
// are 1 wide, 2 tall, 1.45 apart. Returns ink coverage.
fn text(p: vec2f, code: u32, n: u32, stroke: f32, aa: f32) -> f32 {
  if (p.y < -0.6 || p.y > 2.6 || p.x < -0.6) { return 0.0; }
  let k = i32(floor((p.x + 0.225) / 1.45));
  if (k < 0 || k >= i32(n)) { return 0.0; }
  let g = (code >> (4u * u32(k))) & 15u;
  let d = glyph_d(g, vec2f(p.x - f32(k) * 1.45, p.y));
  return 1.0 - smoothstep(stroke - aa, stroke + aa, d);
}
fn num_code(v: u32) -> vec2u {
  var digs = array<u32, 6>(0u, 0u, 0u, 0u, 0u, 0u);
  var x = v;
  var n = 0u;
  loop {
    digs[n] = x % 10u;
    x = x / 10u;
    n = n + 1u;
    if (x == 0u || n >= 6u) { break; }
  }
  var code = 0u;
  for (var i = 0u; i < n; i++) { code = code | (digs[n - 1u - i] << (4u * i)); }
  return vec2u(code, n);
}
// A string centred on c, glyph height h (same units as p).
fn label_code(p: vec2f, c: vec2f, code: u32, n: u32, h: f32, aa: f32) -> f32 {
  let s = h * 0.5;
  let w = (f32(n) * 1.45 - 0.45) * s;
  return text((p - c + vec2f(w * 0.5, h * 0.5)) / s, code, n, 0.17, aa / s + 0.02);
}
fn pack(a: u32, b: u32, c: u32, d: u32, e: u32, f: u32) -> u32 {
  return a | (b << 4u) | (c << 8u) | (d << 12u) | (e << 16u) | (f << 20u);
}

// ---------------------------------------------------------------- paint scheme

fn livery(L: vec3f, aa: f32, pt: u32) -> vec3f {
  if (kind() == 1u) { return livery747(L, aa, pt); }
  var c = WHITE_PAINT;
  let ax = abs(L.x);
  let fin = ax < 0.15 && L.y > 0.33 && L.z > 2.8;
  // Cheat line along the fuselage, sweeping up the tail cone.
  if (!fin && ax < 0.62 && L.y < 0.62 && L.z > -3.16 && L.z < 4.7) {
    let zz = max(L.z - 1.0, 0.0);
    let yc = -0.18 + zz * zz * 0.045;
    let dy = L.y - yc;
    let taper = smoothstep(-3.16, -2.85, L.z);
    let hw = 0.065 * taper;
    let navy = smoothstep(hw + aa, hw - aa, abs(dy));
    let red = smoothstep(0.009 * taper + aa, 0.009 * taper - aa, abs(dy - 0.095 * taper));
    c = mix(c, NAVY, navy);
    c = mix(c, ACCENT, red);
  }
  if (fin) {
    let band = smoothstep(aa, -aa, abs(L.y - 1.09) - 0.07);
    let pin = smoothstep(aa, -aa, abs(L.y - 1.205) - 0.014);
    c = mix(c, NAVY, max(band, smoothstep(1.69 - aa, 1.69 + aa, L.y)));
    c = mix(c, ACCENT, pin);
  }
  // Wing and stabiliser tips, spinner.
  if (ax > 5.26 && L.y > 0.5) { c = NAVY; }
  if (ax > 1.53 && L.z > 3.3 && L.y > -0.1 && L.y < 0.3) { c = NAVY; }
  if (L.z < -3.178) { c = NAVY; }
  // Registration on both sides of the tail cone, reading nose-to-tail on the
  // left and tail-to-nose on the right as painted.
  if (ax > 0.1 && L.z > 1.6 && L.z < 3.0 && L.y > 0.0 && L.y < 0.3) {
    let tx = (2.25 - L.z) * sign(L.x) + 0.44;
    let reg = pack(10u, 4u, 1u, 9u, 6u, 12u);
    let ink = text(vec2f(tx, L.y - 0.045) / 0.09, reg, 6u, 0.13, aa / 0.09 + 0.01);
    c = mix(c, NAVY, ink);
  }
  return c;
}

// Grime, exhaust soot, oil and fuel stains, fading: everything a few hundred
// hours of flying leaves on the paint. Returns (colour, extra roughness).
fn weather(L: vec3f, N: vec3f, base: vec3f) -> vec4f {
  var c = base;
  let n = nz(L * 0.35);
  let belly = sat(-N.y * 1.5 + 0.2);
  c *= 1.0 - (0.03 + 0.07 * belly) * n.g;
  if (kind() == 1u) { return weather747(L, N, c, belly, n); }
  // Soot from the exhaust stack streaking back along the belly.
  let d = L.z + 2.1;
  if (d > 0.0 && L.y < -0.2) {
    let cx = 0.335 - min(d, 1.8) * 0.12;
    let spread = 0.05 + d * 0.07;
    let streak = nz(vec3f(L.x * 7.0, L.y * 7.0, L.z * 0.35)).r;
    let soot = exp(-pow((L.x - cx) / spread, 2.0)) * exp(-d / 1.6) * sat(belly * 1.5) * (0.55 + 0.6 * streak);
    c = mix(c, vec3f(0.035, 0.03, 0.026), sat(soot * 1.2));
  }
  // Oil mist from the breather along the belly centreline.
  if (L.y < -0.45 && L.z > -2.2 && L.z < 1.5) {
    let oil = exp(-L.x * L.x / 0.03) * exp(-(L.z + 2.1) / 1.4) * (0.4 + 0.6 * nz(vec3f(L.x * 5.0, 0.0, L.z * 0.5)).b);
    c = mix(c, c * vec3f(0.62, 0.52, 0.36), sat(oil * 0.6));
  }
  // Blue streaks of spilt avgas behind the fuel caps.
  if (L.y > 0.6 && abs(abs(L.x) - 1.55) < 0.12 && L.z > -0.5 && L.z < 0.3) {
    let f = exp(-pow((abs(L.x) - 1.55) / 0.035, 2.0)) * smoothstep(-0.47, -0.42, L.z) * exp(-(L.z + 0.42) / 0.25) * nz(vec3f(L.x * 18.0, 0.0, L.z * 2.0)).r;
    c = mix(c, c * vec3f(0.72, 0.84, 1.0), sat(f * 1.6));
  }
  return vec4f(c, 0.08 * belly * n.b);
}

// ---------------------------------------------------------------- surface relief

fn chord_at(s: f32) -> f32 { return select(1.65 - 0.5 * (s - 2.55) / 2.95, 1.65, s <= 2.55); }

// Height (m), seam dirt, rivet coverage, for the surface pattern.
fn relief(pt: u32, uv: vec2f, L: vec3f, Nl: vec3f, px: f32) -> vec3f {
  if (kind() == 1u) { return relief747(pt, uv, L, Nl, px); }
  var h = 0.0;
  var dirt = 0.0;
  var rv = 0.0;
  let gw = max(0.0007, px * 0.9);
  let ga = 0.00045 * min(1.0, 0.0012 / gw);
  let rr = 0.0024;
  let rf = 1.0 - smoothstep(rr * 0.5, rr * 1.6, px);
  switch pt {
    case 1u: { // fuselage
      let z = uv.x;
      let s = abs(uv.y);
      var st = array<f32, 8>(-1.98, 0.92, 1.52, 2.12, 2.72, 3.32, 3.92, 4.40);
      for (var i = 0; i < 8; i++) {
        let dz = z - st[i];
        if (abs(dz) < 0.04) {
          h -= ga * line(dz, gw);
          dirt += line(dz, gw * 2.5);
          rv = max(rv, rivets(s, abs(abs(dz) - 0.012), 0.027, rr) * rf);
        }
      }
      if (L.y < -0.4 && z > -1.9 && z < 0.9) {
        for (var i = 0; i < 3; i++) {
          let dz = z - (-1.25 + f32(i) * 0.75);
          h -= ga * line(dz, gw);
          dirt += line(dz, gw * 2.5);
          rv = max(rv, rivets(s, abs(abs(dz) - 0.012), 0.027, rr) * rf);
        }
      }
      if (z > 0.92) {
        for (var i = 0; i < 2; i++) {
          let ds = s - select(1.25, 0.6, i == 0);
          h -= ga * line(ds, gw);
          dirt += line(ds, gw * 2.0);
          rv = max(rv, rivets(z, abs(ds - 0.011), 0.027, rr) * rf);
        }
        // Single rows of rivets on the stringers between the laps.
        let k = (fract(s / 0.21) - 0.5) * 0.21;
        rv = max(rv, rivets(z, k, 0.04, rr * 0.9) * rf * 0.8);
      }
      if (abs(L.x) > 0.35) {
        let dd = door(L);
        h -= ga * 1.8 * line(dd, gw * 1.3);
        dirt += line(dd, gw * 3.0);
        if (L.x < 0.0) {
          let bd = baggage_door(L);
          h -= ga * 1.6 * line(bd, gw * 1.2);
          dirt += line(bd, gw * 2.5);
        }
      }
      h += rv * 0.0006;
    }
    case 2u, 5u: { // wing, tail surfaces
      let s = abs(uv.x);
      let av = abs(uv.y);
      let tail = pt == 5u;
      let c = select(chord_at(s), 1.0, tail);
      let xc = av / c;
      let rib_p = select(0.305, 0.21, tail);
      let rib0 = select(0.62, 0.15, tail);
      if (s > rib0 && s < select(5.24, 1.6, tail)) {
        let d = (fract((s - rib0) / rib_p + 0.5) - 0.5) * rib_p;
        rv = max(rv, rivets(av, d, 0.03, rr) * rf);
      }
      let spar = select(0.25 * c, 0.20, tail);
      rv = max(rv, rivets(s, av - spar, 0.03, rr) * rf);
      if (!tail) {
        rv = max(rv, rivets(s, av - 0.66 * c, 0.03, rr) * rf);
        let le = av - 0.13 * c;
        h -= ga * line(le, gw);
        dirt += line(le, gw * 2.0);
        rv = max(rv, rivets(s, abs(le + 0.012), 0.025, rr) * rf);
        for (var i = 0; i < 3; i++) {
          let ds = s - select(select(5.28, 3.9, i == 1), 2.55, i == 0);
          h -= ga * line(ds, gw);
          dirt += line(ds, gw * 2.0);
          if (i < 2) { rv = max(rv, rivets(av, abs(ds) - 0.012, 0.027, rr) * rf); }
        }
      }
      h += rv * 0.0006;
    }
    case 3u: { // control surfaces: stiffening beads across the span
      let d = (fract(abs(uv.x) / 0.075) - 0.5) * 0.075;
      h += 0.0009 * line(d, max(0.0045, px));
      dirt += 0.3 * line(abs(d) - 0.007, gw * 2.0);
    }
    case 4u: { // cowling
      let lines = array<f32, 2>(L.z + 2.975, L.y + 0.085);
      for (var i = 0; i < 2; i++) {
        let d = lines[i];
        h -= ga * 1.4 * line(d, gw);
        dirt += line(d, gw * 2.5);
      }
      if (L.y > 0.15) {
        let q = vec2f(L.z + 2.36, L.x);
        let b = abs(q) - vec2f(0.12, 0.10);
        let od = length(max(b, vec2f(0.0))) + min(max(b.x, b.y), 0.0);
        h -= ga * 1.5 * line(od, gw);
        dirt += line(od, gw * 2.0);
      }
    }
    case 7u: { // cabin lining: door gaps, headliner seams, grain
      if (abs(L.x) > 0.35) {
        let dd = door(L);
        h -= 0.0015 * line(dd, max(0.0015, px));
        dirt += line(dd, max(0.003, px * 2.0));
      }
      if (L.y > 0.46) {
        let d = (fract((L.z + 0.6) / 0.42) - 0.5) * 0.42;
        h -= 0.001 * line(d, max(0.004, px));
        dirt += 0.5 * line(d, 0.006);
      }
      h += 0.00015 * nz(L * 40.0).r * (1.0 - smoothstep(0.0005, 0.002, px));
    }
    case 8u: { // pleated upholstery
      if (abs(Nl.x) < 0.7) {
        let t = fract(L.x / 0.065 + 0.5) * 2.0 - 1.0;
        h += 0.004 * sqrt(max(1.0 - t * t, 0.0));
        dirt += smoothstep(0.85, 1.0, abs(t));
      }
      h += 0.0004 * nz(L * 60.0).g * (1.0 - smoothstep(0.0005, 0.002, px));
    }
    case 9u: { h += 0.0015 * nz(L * 45.0).r * (1.0 - smoothstep(0.001, 0.004, px)); }
    case 18u: { // cockpit paint: orange peel
      h += 0.00010 * nz(L * 110.0).r * (1.0 - smoothstep(0.0002, 0.0009, px));
    }
    case 19u: { // open-weave mesh fabric: raised threads round small holes
      let q = uv / 0.0026;
      let f = abs(fract(q) - 0.5) * 2.0;
      let thread = max(smoothstep(0.55, 0.9, f.x), smoothstep(0.55, 0.9, f.y));
      let fade = 1.0 - smoothstep(0.0003, 0.0012, px);
      h += 0.0004 * thread * fade + 0.0012 * nz(L * 18.0).g;
      dirt += (1.0 - thread) * 0.7 * fade;
    }
    case 20u: { // webbing: a twill weave, stitching along both edges
      let tw = sin((uv.y + uv.x * 0.6) * 2400.0);
      let fade = 1.0 - smoothstep(0.0002, 0.0008, px);
      h += 0.00012 * tw * fade;
      let st = step(0.5, fract(uv.y / 0.004)) * line(abs(abs(uv.x) - 0.017), 0.0006);
      h -= 0.0002 * st * fade;
      dirt += st * 0.5 * fade;
    }
    case 21u: { // anti-skid grit
      h += 0.0005 * nz(L * 190.0).r * (1.0 - smoothstep(0.0003, 0.0012, px));
    }
    case 23u: { // corrugated hose
      let r = sin(uv.y * 6.2831853 / 0.0065);
      h += 0.0011 * r * (1.0 - smoothstep(0.0006, 0.0025, px));
      dirt += smoothstep(0.6, 1.0, -r) * 0.4;
    }
    case 10u: { h += select(0.0002, 0.00006, kind() == 2u) * nz(L * 160.0).r * (1.0 - smoothstep(0.0003, 0.0012, px)); }
    case 12u: { h += 0.00025 * sin((L.x + L.y * 1.3 + L.z * 0.7) * 2600.0) * (1.0 - smoothstep(0.0002, 0.0008, px)); }
    case 6u: { // tyre tread: circumferential grooves
      var c = vec3f(sign(L.x) * 1.25, -1.31, 0.30);
      var r = 0.19;
      var g = vec2f(0.022, 0.048);
      if (L.z < -1.0) { c = vec3f(0.0, -1.335, -2.87); r = 0.165; g = vec2f(0.018, 0.038); }
      let ax = abs(L.x - c.x);
      let rad = length(L.yz - c.yz);
      if (rad > r * 0.9) {
        let gd = min(abs(ax - g.x), abs(ax - g.y));
        h -= 0.002 * line(gd, max(0.003, px));
        dirt += line(gd, max(0.004, px));
      }
    }
    default: {}
  }
  // Fine seams fade with distance rather than widen into dark bands.
  return vec3f(h, sat(dirt) * min(1.0, 0.0022 / gw), rv);
}

// ---------------------------------------------------------------- instruments
// Each instrument only sets up what is on its face below; the graduations,
// arcs, needles and lettering are then drawn once, so the shader stays small.

fn ang_of(uv: vec2f) -> f32 { return atan2(uv.x, uv.y); }
fn unwrap(th: f32, a0: f32) -> f32 { return a0 + fract((th - a0) / 6.2831853) * 6.2831853; }
fn wrapa(a: f32) -> f32 { return a - round(a / 6.2831853) * 6.2831853; }
// Graduations: n intervals between angles a0 and a1, every `per`th long.
fn ticks(uv: vec2f, a0: f32, a1: f32, n: f32, per: f32, px: f32) -> f32 {
  let r = length(uv);
  let th = unwrap(ang_of(uv), min(a0, a1) - 0.05);
  let k = clamp(round((th - a0) / (a1 - a0) * n), 0.0, n);
  let ak = a0 + (a1 - a0) * k / n;
  let major = abs(k - round(k / per) * per) < 0.5;
  let len = select(0.085, 0.17, major);
  let w = select(0.011, 0.022, major);
  let d = abs(wrapa(th - ak)) * r;
  let inr = smoothstep(0.95 - len - px, 0.95 - len + px, r) * smoothstep(0.95 + px, 0.95 - px, r);
  return inr * (1.0 - smoothstep(w - px, w + px, d));
}
fn arc(uv: vec2f, a0: f32, a1: f32, r0: f32, r1: f32, px: f32) -> f32 {
  let r = length(uv);
  let th = unwrap(ang_of(uv), a0 - 0.3);
  let inside = step(a0, th) * step(th, a1);
  return inside * smoothstep(r0 - px, r0 + px, r) * smoothstep(r1 + px, r1 - px, r);
}
fn needle(uv: vec2f, a: f32, len: f32, w: f32, tail: f32, px: f32) -> f32 {
  let d = vec2f(sin(a), cos(a));
  let t = clamp(dot(uv, d), -tail, len);
  let r = w * mix(1.0, 0.3, sat(t / len));
  return 1.0 - smoothstep(-px, px, length(uv - d * t) - r);
}
fn dial(v: f32, vmin: f32, vmax: f32, a0: f32, a1: f32) -> f32 { return a0 + sat((v - vmin) / (vmax - vmin)) * (a1 - a0); }
fn rad(d: f32) -> f32 { return d * 0.017453293; }
// Compass card lettering: N, 3, 6, E, 12 ... for card index 0..11.
fn card_code(k: i32) -> vec2u {
  let ki = u32((k % 12 + 12) % 12);
  if (ki == 0u) { return vec2u(10u, 1u); }
  if (ki == 3u) { return vec2u(11u, 1u); }
  if (ki == 6u) { return vec2u(12u, 1u); }
  if (ki == 9u) { return vec2u(13u, 1u); }
  return num_code(ki * 3u);
}

// The face of an instrument: rgb albedo, a = how much of it is luminous paint.
fn gauge(id: u32, uv: vec2f, px: f32) -> vec4f {
  let P0 = O.params[0];
  let P1 = O.params[1];
  let P2 = O.params[2];
  let P3 = O.params[3];
  let P4 = O.params[4];
  let r = length(uv);
  var bg = vec3f(0.012, 0.012, 0.013);
  var ink = 0.0;
  var col = vec3f(0.0);
  var cov = 0.0;
  var ndl = 0.0;
  // Dial: angles, value range, graduations.
  var a0 = 0.0; var a1 = 0.0; var vmin = 0.0; var vmax = 1.0; var tn = 0.0; var tper = 1.0;
  var q = uv;                       // where the dial is drawn (small gauges sit low)
  // Numbers round the dial: count, first value, step, divisor, radius, height.
  var ln = 0.0; var lfirst = 0.0; var lstep = 1.0; var ldiv = 1.0; var lr = 0.6; var lh = 0.15;
  // Coloured arcs (value from, to; colour) and a red line.
  var arcs = array<vec4f, 3>(vec4f(0.0), vec4f(0.0), vec4f(0.0));
  var arc_col = array<vec3f, 3>(vec3f(0.0), vec3f(0.0), vec3f(0.0));
  var redline = -1e9;
  // Needles: angle, length, width, tail.
  var nd = array<vec4f, 3>(vec4f(0.0), vec4f(0.0), vec4f(0.0));
  var nn = 0;
  // One more piece of lettering: point in its frame, centre, code, glyphs, height.
  var tp = uv; var tc = vec2f(9.0); var tcode = 0u; var tlen = 0u; var th = 0.1;
  var hub = true;
  let green = vec3f(0.05, 0.4, 0.08);
  switch id {
    case 0u: { // airspeed
      a0 = 0.0; a1 = rad(330.0); vmax = 200.0; tn = 40.0; tper = 4.0;
      ln = 7.0; lfirst = 40.0; lstep = 20.0; lr = 0.62;
      // A 172's markings (indicated): white Vso..Vfe, green Vs1..Vno, yellow to Vne.
      arcs[0] = vec4f(50.0, 129.0, 0.86, 0.95); arc_col[0] = green;
      arcs[1] = vec4f(129.0, 163.0, 0.86, 0.95); arc_col[1] = vec3f(0.6, 0.45, 0.02);
      arcs[2] = vec4f(45.0, 85.0, 0.79, 0.85); arc_col[2] = vec3f(0.6);
      redline = 163.0;
      nd[0] = vec4f(dial(P0.x, 0.0, 200.0, a0, a1), 0.86, 0.035, 0.18); nn = 1;
      if (kind() == 1u) {
        // The 747's standby airspeed: 60 to 450 kt, no arcs, the barber pole at Vmo.
        a0 = rad(10.0); vmin = 60.0; vmax = 450.0; tn = 39.0; tper = 3.0;
        ln = 8.0; lfirst = 100.0; lstep = 50.0;
        arcs[0] = vec4f(0.0); arcs[1] = vec4f(0.0); arcs[2] = vec4f(0.0);
        redline = 365.0;
        nd[0] = vec4f(dial(P0.x, 60.0, 450.0, a0, a1), 0.86, 0.035, 0.18);
      }
      if (kind() == 2u) {
        // The Hornet's standby airspeed: 50 to 850 kt.
        a0 = rad(10.0); vmin = 50.0; vmax = 850.0; tn = 32.0; tper = 4.0;
        ln = 8.0; lfirst = 100.0; lstep = 100.0; ldiv = 10.0;
        arcs[0] = vec4f(0.0); arcs[1] = vec4f(0.0); arcs[2] = vec4f(0.0);
        redline = -1e9;
        nd[0] = vec4f(dial(P0.x, vmin, vmax, a0, a1), 0.86, 0.035, 0.18);
      }
    }
    case 2u: { // altimeter: hundreds, thousands, ten thousands, pressure window
      a0 = 0.0; a1 = rad(360.0); vmax = 10.0; tn = 50.0; tper = 5.0;
      ln = 10.0; lfirst = 0.0; lstep = 1.0; lr = 0.64; lh = 0.17;
      let alt = P0.y;
      nd[0] = vec4f(fract(alt / 1000.0) * 6.2831853, 0.8, 0.03, 0.15);
      nd[1] = vec4f(fract(alt / 10000.0) * 6.2831853, 0.5, 0.06, 0.1);
      nd[2] = vec4f(fract(alt / 100000.0) * 6.2831853, 0.93, 0.012, 0.0);
      nn = 3;
      let kb = abs(uv - vec2f(0.52, 0.0)) - vec2f(0.2, 0.07);
      if (max(kb.x, kb.y) < 0.0) { bg = vec3f(0.0); ln = 0.0; tn = 0.0; }
      tc = vec2f(0.52, 0.0); tcode = pack(2u, 9u, 9u, 2u, 0u, 0u); tlen = 4u; th = 0.09;
    }
    case 5u: { // vertical speed
      a0 = rad(-260.0); a1 = rad(80.0); vmin = -2000.0; vmax = 2000.0; tn = 40.0; tper = 5.0;
      ln = 9.0; lfirst = -2000.0; lstep = 500.0; ldiv = 100.0;
      nd[0] = vec4f(dial(P0.z, vmin, vmax, a0, a1), 0.86, 0.035, 0.18); nn = 1;
      if (kind() == 2u) {
        vmin = -6000.0; vmax = 6000.0; lfirst = -6000.0; lstep = 2000.0; ln = 7.0; ldiv = 1000.0; tn = 24.0; tper = 4.0;
        nd[0] = vec4f(dial(P0.z, vmin, vmax, a0, a1), 0.86, 0.035, 0.18);
      }
    }
    case 21u: { // the Hornet's brake accumulator pressure, psi x 1000
      a0 = rad(-120.0); a1 = rad(120.0); vmin = 0.0; vmax = 4.0; tn = 8.0; tper = 2.0;
      ln = 5.0; lfirst = 0.0; lstep = 1.0; lr = 0.58; lh = 0.2;
      arcs[0] = vec4f(2.0, 3.2, 0.88, 0.95); arc_col[0] = vec3f(0.05, 0.4, 0.08);
      nd[0] = vec4f(dial(3.0, vmin, vmax, a0, a1), 0.84, 0.045, 0.15); nn = 1;
    }
    case 6u: { // tachometer with its hour meter
      a0 = rad(-135.0); a1 = rad(135.0); vmax = 3500.0; tn = 35.0; tper = 5.0;
      ln = 8.0; lfirst = 0.0; lstep = 500.0; ldiv = 100.0;
      arcs[0] = vec4f(2100.0, 2700.0, 0.86, 0.95); arc_col[0] = green;
      redline = 2700.0;
      nd[0] = vec4f(dial(P2.x, 0.0, vmax, a0, a1), 0.86, 0.035, 0.18); nn = 1;
      let hb = abs(uv - vec2f(0.0, -0.45)) - vec2f(0.2, 0.065);
      if (max(hb.x, hb.y) < 0.0) { bg = vec3f(0.0); tn = 0.0; ln = 0.0; }
      tc = vec2f(0.0, -0.45); tcode = pack(1u, 8u, 3u, 4u, 0u, 0u); tlen = 4u; th = 0.085;
    }
    case 7u: { // manifold pressure
      a0 = rad(-135.0); a1 = rad(135.0); vmin = 10.0; vmax = 35.0; tn = 25.0; tper = 5.0;
      ln = 6.0; lfirst = 10.0; lstep = 5.0; lh = 0.16;
      arcs[0] = vec4f(15.0, 25.0, 0.86, 0.95); arc_col[0] = green;
      nd[0] = vec4f(dial(P2.y, vmin, vmax, a0, a1), 0.86, 0.04, 0.18); nn = 1;
    }
    case 8u, 9u, 10u, 11u, 12u, 13u, 16u, 18u: { // small engine gauges, dial low on the face
      a0 = rad(-60.0); a1 = rad(60.0); tn = 10.0; tper = 5.0;
      q = (uv + vec2f(0.0, 0.35)) * 0.78;
      var v = 0.0; var g0 = 0.0; var g1 = 0.0;
      if (id == 8u) { v = P2.z; vmax = 26.0; g0 = 2.0; g1 = 26.0; }
      if (id == 9u) { v = P2.w; vmax = 26.0; g0 = 2.0; g1 = 26.0; }
      if (id == 10u) { v = P3.x; vmin = 100.0; vmax = 250.0; g0 = 100.0; g1 = 240.0; }
      if (id == 11u) { v = P3.y; vmax = 115.0; g0 = 60.0; g1 = 90.0; redline = 114.0; }
      if (id == 12u) { v = P3.w; vmin = -60.0; vmax = 60.0; g0 = -5.0; g1 = 60.0; }
      if (id == 13u) { v = P4.x; vmin = 3.0; vmax = 7.0; g0 = 4.5; g1 = 5.5; }
      if (id == 16u) { v = P3.z; g0 = 0.5; g1 = 0.8; }
      if (id == 18u) { v = P3.x * 1.6 + 30.0; vmin = 200.0; vmax = 500.0; g0 = 200.0; g1 = 460.0; }
      arcs[0] = vec4f(g0, g1, 0.86, 0.95); arc_col[0] = green;
      if (id == 8u || id == 9u) {
        arcs[1] = vec4f(vmin, 2.0, 0.86, 0.95); arc_col[1] = vec3f(0.6, 0.02, 0.01);
        tc = vec2f(-0.55, 0.35); tcode = 0u; tlen = 1u; th = 0.16;
      }
      nd[0] = vec4f(dial(v, vmin, vmax, a0, a1), 0.9, 0.05, 0.0); nn = 1;
      ndl = 1.0 - smoothstep(0.06 - px, 0.06 + px, length(q));
      hub = false;
    }
    case 14u: { // clock, with a red sweep second hand
      a0 = 0.0; a1 = rad(360.0); vmax = 12.0; tn = 60.0; tper = 5.0;
      ln = 4.0; lfirst = 3.0; lstep = 3.0; lh = 0.2;
      let t = P4.y;
      nd[0] = vec4f(fract(t / 43200.0) * 6.2831853, 0.5, 0.05, 0.1);
      nd[1] = vec4f(fract(t / 3600.0) * 6.2831853, 0.8, 0.035, 0.1);
      nn = 2;
      let sec = needle(uv, floor(t) * 0.10471976, 0.9, 0.012, 0.2, px);
      col = vec3f(0.6, 0.05, 0.02) * sec;
      cov = sec;
    }
    case 1u: { // attitude indicator
      let roll = rad(P1.y);
      let pitch = P1.x;
      let qq = rot2(uv, roll);
      let hz = -pitch * 0.028;
      bg = mix(vec3f(0.2, 0.09, 0.03), vec3f(0.06, 0.2, 0.5), smoothstep(hz - px, hz + px, qq.y));
      if (r < 0.74) {
        ink = 1.0 - smoothstep(0.0, px * 1.5, abs(qq.y - hz) - 0.01);
        let k = clamp(round((qq.y - hz) / 0.14), -4.0, 4.0);
        let w = select(0.1, 0.2, abs(k - round(k / 2.0) * 2.0) < 0.5);
        let m = (1.0 - smoothstep(0.0, px, abs(qq.y - hz - k * 0.14) - 0.008)) * step(abs(qq.x), w) * step(0.5, abs(k));
        ink = max(ink, m);
      } else {
        // Bank scale on the rotating ring: 10, 20, 30, 45, 60, 90 degrees.
        let a = abs(ang_of(qq)) * 57.29578;
        let mk = select(select(select(round(a / 10.0) * 10.0, 45.0, a > 37.5 && a < 52.5), 60.0, a >= 52.5 && a < 75.0), 90.0, a >= 75.0);
        let d = abs(a - mk) * 0.017453 * r;
        let len = select(0.1, 0.18, mk == 30.0 || mk == 60.0 || mk == 90.0);
        ink = (1.0 - smoothstep(0.012, 0.012 + px, d)) * step(0.97 - len, r) * step(mk, 90.5) * select(0.0, 1.0, mk != 40.0 && mk != 50.0 && mk != 70.0 && mk != 80.0);
      }
      let tri = step(abs(uv.x), (0.74 - uv.y) * 0.5) * step(0.62, uv.y) * step(uv.y, 0.74);
      let wing = (1.0 - smoothstep(0.0, px, abs(uv.y) - 0.022)) * step(0.16, abs(uv.x)) * step(abs(uv.x), 0.5);
      let fixed_ = max(max(tri, wing), 1.0 - smoothstep(0.04, 0.04 + px, r));
      col = vec3f(0.75, 0.35, 0.02) * fixed_;
      cov = fixed_;
      hub = false;
    }
    case 4u: { // heading indicator: the card turns under a fixed lubber line
      let hdg = rad(P0.w);
      a0 = 0.0; a1 = rad(360.0); tn = 72.0; tper = 2.0;
      q = rot2(uv, -hdg);
      let k = round((ang_of(uv) + hdg) / rad(30.0));
      let a = k * rad(30.0) - hdg;
      tp = rot2(uv - vec2f(sin(a), cos(a)) * 0.6, a);
      tc = vec2f(0.0);
      let cc = card_code(i32(k));
      tcode = cc.x; tlen = cc.y; th = 0.17;
      let lub = step(abs(uv.x), (0.98 - uv.y) * 0.4) * step(0.8, uv.y);
      let plane = max((1.0 - smoothstep(0.0, px, abs(uv.x) - 0.02)) * step(abs(uv.y + 0.05), 0.3), (1.0 - smoothstep(0.0, px, abs(uv.y - 0.05) - 0.02)) * step(abs(uv.x), 0.25));
      col = vec3f(0.75, 0.35, 0.02) * lub + vec3f(0.6) * plane;
      cov = max(lub, plane);
      hub = false;
    }
    case 3u: { // turn coordinator
      let qq = rot2(uv, clamp(P1.z / 3.0, -1.6, 1.6) * rad(20.0));
      let wing = (1.0 - smoothstep(0.0, px, abs(qq.y) - 0.025)) * step(abs(qq.x), 0.62);
      let body = 1.0 - smoothstep(0.07, 0.07 + px, length(qq));
      let fin = (1.0 - smoothstep(0.0, px, abs(qq.x) - 0.018)) * step(0.0, qq.y) * step(qq.y, 0.2);
      ink = max(max(wing, body), fin);
      // Level and standard-rate marks either side.
      let a = abs(ang_of(uv));
      let mk = select(rad(110.0), rad(90.0), a < rad(100.0));
      ink = max(ink, (1.0 - smoothstep(0.02, 0.02 + px, abs(a - mk) * r)) * step(0.66, r) * step(r, 0.85));
      // Inclinometer: a ball in a curved glass tube.
      let tube = vec2f(uv.x, uv.y + 0.52 - uv.x * uv.x * 0.25);
      if (abs(tube.y) < 0.09 && abs(tube.x) < 0.45) {
        let lines = 1.0 - smoothstep(0.0, px, abs(abs(tube.x) - 0.1) - 0.008);
        let ball = 1.0 - smoothstep(0.075 - px, 0.075 + px, length(tube - vec2f(clamp(P1.w, -1.0, 1.0) * 0.32, 0.0)));
        bg = mix(vec3f(0.45, 0.45, 0.42), vec3f(0.01), max(ball, lines));
      }
      hub = false;
    }
    case 15u: { // VOR / CDI with the OBS card and a NAV flag
      a0 = 0.0; a1 = rad(360.0); tn = 72.0; tper = 2.0;
      if (r < 0.8) { tn = 0.0; }
      let dev = sin(G.time.x * 0.07) * 0.25;
      ndl = (1.0 - smoothstep(0.0, px, abs(uv.x - dev) - 0.018)) * step(abs(uv.y), 0.6);
      let k = clamp(round(uv.x / 0.13), -4.0, 4.0);
      ink = (1.0 - smoothstep(0.022, 0.022 + px, length(uv - vec2f(k * 0.13, 0.0)))) * step(0.5, abs(k));
      let flag = step(abs(uv.x - 0.35), 0.1) * step(abs(uv.y - 0.35), 0.06);
      col = vec3f(0.6, 0.02, 0.01) * flag;
      cov = flag;
      tc = vec2f(0.0, 0.66); tcode = pack(0u, 9u, 0u, 0u, 0u, 0u); tlen = 3u; th = 0.15;
    }
    case 17u: { // magnetic compass: a card seen edge-on through a window
      let hdg = O.params[5].x % 1000.0;
      let x = uv.x * 32.0 + hdg;
      let k = round(x / 5.0);
      let major = abs(k - round(k / 2.0) * 2.0) < 0.5;
      ink = (1.0 - smoothstep(0.008, 0.008 + px, abs(x - k * 5.0) / 32.0)) * step(select(0.3, 0.05, major), uv.y) * step(uv.y, 0.7);
      let kl = round(x / 30.0);
      tc = vec2f((kl * 30.0 - hdg) / 32.0, -0.35);
      let cc = card_code(i32(kl));
      tcode = cc.x; tlen = cc.y; th = 0.5;
      let lub = 1.0 - smoothstep(0.0, px, abs(uv.x) - 0.02);
      col = vec3f(0.6, 0.05, 0.02) * lub;
      cov = lub;
      bg = vec3f(0.02, 0.02, 0.022) * (1.0 - 0.4 * uv.y * uv.y);
      hub = false;
    }
    case 20u: { // outside air temperature, degrees C (the standard atmosphere's at this height)
      a0 = rad(-135.0); a1 = rad(135.0); vmin = -40.0; vmax = 50.0; tn = 18.0; tper = 2.0;
      ln = 5.0; lfirst = -40.0; lstep = 20.0; lr = 0.58; lh = 0.2;
      arcs[0] = vec4f(-40.0, 0.0, 0.88, 0.95); arc_col[0] = vec3f(0.05, 0.15, 0.5);
      nd[0] = vec4f(dial(15.0 - 1.98 * P0.y / 1000.0, vmin, vmax, a0, a1), 0.84, 0.045, 0.15); nn = 1;
    }
    case 19u: { // fuel selector plate
      let arrow = step(abs(uv.x), (0.8 - uv.y) * 0.35) * step(-0.2, uv.y);
      ink = max(arrow, 1.0 - smoothstep(0.0, px, abs(r - 0.85) - 0.02));
      tp = vec2f(abs(uv.x), uv.y); tc = vec2f(0.6, -0.1); tcode = select(2u, 1u, uv.x < 0.0); tlen = 1u; th = 0.22;
      hub = false;
    }
    default: {}
  }
  // The dial, drawn once whatever the instrument.
  if (tn > 0.0) {
    ink = max(ink, ticks(q, a0, a1, tn, tper, px));
    for (var i = 0; i < 3; i++) {
      if (arcs[i].x != arcs[i].y) {
        let m = arc(q, dial(arcs[i].x, vmin, vmax, a0, a1), dial(arcs[i].y, vmin, vmax, a0, a1), arcs[i].z, arcs[i].w, px);
        col += arc_col[i] * m;
        cov = max(cov, m);
      }
    }
    if (redline > -1e8) {
      let m = arc(q, dial(redline, vmin, vmax, a0, a1) - 0.012, dial(redline, vmin, vmax, a0, a1) + 0.012, 0.78, 0.97, px);
      col += vec3f(0.6, 0.02, 0.01) * m;
      cov = max(cov, m);
    }
  }
  for (var i = 0; i < nn; i++) {
    ndl = max(ndl, needle(q, nd[i].x, nd[i].y, nd[i].z, nd[i].w, px));
  }
  // Lettering: the nearest number round the dial, or the instrument's own.
  if (ln > 0.0) {
    let ang = unwrap(ang_of(q), min(a0, a1) - 0.3);
    let x = vmin + (ang - a0) / (a1 - a0) * (vmax - vmin);
    let k = clamp(round((x - lfirst) / lstep), 0.0, ln - 1.0);
    let v = lfirst + k * lstep;
    let a = a0 + (v - vmin) / (vmax - vmin) * (a1 - a0);
    let c = vec2f(sin(a), cos(a)) * lr;
    if (length(q - c) < length(tp - tc)) {
      let nc = num_code(u32(round(abs(v) / ldiv)));
      tp = q; tc = c; tcode = nc.x; tlen = nc.y; th = lh;
    }
  }
  if (tlen > 0u) { ink = max(ink, label_code(tp, tc, tcode, tlen, th, px)); }
  if (hub) {
    let hb = 1.0 - smoothstep(0.075 - px, 0.075 + px, r);
    ndl = ndl * (1.0 - hb);
    col = mix(col, vec3f(0.02), hb);
    cov = max(cov, hb);
  }
  var c = mix(bg, col / max(cov, 1e-3), sat(cov));
  c = mix(c, vec3f(0.78, 0.78, 0.74), ink * (1.0 - cov * 0.5));
  c = mix(c, vec3f(0.82, 0.82, 0.78), ndl);
  c *= 1.0 - 0.6 * smoothstep(0.8, 1.0, r) * step(r, 1.01) * select(1.0, 0.0, id == 17u);
  return vec4f(c, max(ink, ndl) * 0.8 + cov * 0.2);
}

// Radio readouts (emissive): up to two strings per display.
fn display(id: u32, uv: vec2f, px: f32) -> vec3f {
  var size = vec2f(0.048, 0.0085);
  var c0 = vec2f(9.0); var k0 = 0u; var n0 = 0u; var h0 = 0.0105; var col0 = vec3f(0.0);
  var c1 = vec2f(9.0); var k1 = 0u; var n1 = 0u; var h1 = 0.0105; var col1 = vec3f(0.0);
  var bg = vec3f(0.0);
  switch id {
    case 0u: { // nav-com: active and standby
      c0 = vec2f(-0.023, 0.0); k0 = pack(1u, 2u, 2u, 14u, 8u, 0u); n0 = 6u; col0 = vec3f(0.35, 1.0, 0.45);
      c1 = vec2f(0.024, 0.0); k1 = pack(1u, 2u, 1u, 14u, 5u, 0u); n1 = 6u; col1 = vec3f(0.2, 0.55, 0.25);
    }
    case 1u: { // autopilot altitude
      c0 = vec2f(0.015, 0.0); k0 = pack(4u, 5u, 0u, 0u, 0u, 0u); n0 = 4u; col0 = vec3f(1.0, 0.55, 0.1);
    }
    case 2u: { // transponder: code and pressure altitude
      c0 = vec2f(0.012, 0.0); k0 = pack(1u, 2u, 0u, 0u, 0u, 0u); n0 = 4u; col0 = vec3f(1.0, 0.55, 0.1);
      let a = num_code(u32(max(O.params[0].y, 0.0) / 100.0));
      c1 = vec2f(-0.025, 0.0); k1 = a.x; n1 = a.y; h1 = 0.007; col1 = vec3f(0.6, 0.33, 0.06);
    }
    case 3u: { // GPS: a heading-up moving map, ground speed in the corner
      size = vec2f(0.052, 0.027);
      let m = rot2(uv * vec2f(1.0, 0.52), -rad(O.params[0].w));
      let land = nz(vec3f(m * 0.3, 0.37)).g;
      bg = mix(vec3f(0.02, 0.05, 0.14), vec3f(0.12, 0.2, 0.08), smoothstep(0.45, 0.5, land));
      bg = mix(bg, vec3f(0.9, 0.1, 0.8), (1.0 - smoothstep(0.0, px * 60.0, abs(uv.x) - 0.012)) * step(0.0, uv.y));
      bg = mix(bg, vec3f(1.0), step(abs(uv.x), (0.12 - uv.y) * 0.6) * step(-0.12, uv.y));
      bg = mix(bg, vec3f(0.0), step(0.78, uv.y));
      let gs = num_code(u32(O.params[0].x));
      c0 = vec2f(-0.04, 0.021); k0 = gs.x; n0 = gs.y; h0 = 0.006; col0 = vec3f(1.0);
    }
    case 5u: { // hour meter
      size = vec2f(0.021, 0.0065);
      c0 = vec2f(0.0); k0 = pack(1u, 8u, 3u, 4u, 14u, 6u); n0 = 6u; h0 = 0.008; col0 = vec3f(0.9);
    }
    default: {}
  }
  let q = uv * size;
  var c = bg;
  if (n0 > 0u) { c = mix(c, col0, label_code(q, c0, k0, n0, h0, px)); }
  if (n1 > 0u) { c = mix(c, col1, label_code(q, c1, k1, n1, h1, px)); }
  return c;
}

// ---------------------------------------------------------------- lighting

fn ggx(ndh: f32, a: f32) -> f32 {
  let a2 = a * a;
  let d = ndh * ndh * (a2 - 1.0) + 1.0;
  return a2 / (PI * d * d);
}
fn vis(ndv: f32, ndl: f32, a: f32) -> f32 {
  let k = a * 0.5;
  return 0.25 / ((ndv * (1.0 - k) + k) * (ndl * (1.0 - k) + k));
}

// ---------------------------------------------------------------- the Hornet (aircraft/fa18)

// Its paint: three pictures of it (from above, from below and from the left
// side, the layers of t_paint) projected back onto the skin, each where the
// surface faces that way. The rectangles they cover are GUIDE_* in
// assets/blender/fa18.py. rgb the paint; a 0 where no picture faces the
// surface (the lips of the intakes, the ends of things).
fn hornet_paint(L: vec3f, Nl: vec3f, dLx: vec3f, dLy: vec3f) -> vec4f {
  let n = Nl * inverseSqrt(max(dot(Nl, Nl), 1e-12));
  var w = vec3f(pow(max(n.y, 0.0), 4.0), pow(max(-n.y, 0.0), 4.0), pow(abs(n.x), 4.0));
  let sum = w.x + w.y + w.z;
  if (sum < 0.02) { return vec4f(0.0); }
  w /= sum;
  let v = (L.z + 10.3) / 19.8;
  let dv = vec2f(dLx.z, dLy.z) / 19.8;
  var c = vec3f(0.0);
  if (w.x > 0.004) { c += w.x * textureSampleGrad(t_paint, s_aniso, vec2f((L.x + 6.6) / 13.2, v), 0, vec2f(dLx.x / 13.2, dv.x), vec2f(dLy.x / 13.2, dv.y)).rgb; }
  if (w.y > 0.004) { c += w.y * textureSampleGrad(t_paint, s_aniso, vec2f((6.6 - L.x) / 13.2, v), 1, vec2f(-dLx.x / 13.2, dv.x), vec2f(-dLy.x / 13.2, dv.y)).rgb; }
  if (w.z > 0.004) { c += w.z * textureSampleGrad(t_paint, s_aniso, vec2f((L.y + 5.6) / 13.2, v), 2, vec2f(dLx.y / 13.2, dv.x), vec2f(dLy.y / 13.2, dv.y)).rgb; }
  return vec4f(c, 1.0);
}

// Where the engines' exhaust leaves: the nozzle exits are at z = 8.0, the
// engines' centrelines 0.52 m either side and 0.05 m below the datum.
const NOZZLE_Z = 8.0;
fn plume_r(L: vec3f) -> f32 { return length(vec2f(abs(L.x) - 0.52, L.y + 0.05)); }

// The exhaust plume (material 17, drawn in the transparent pass): shells and
// discs round each engine's jet that glow by how squarely they are looked
// through, so the whole reads as a volume. in.mat.w: 0 the heat haze, 1 the
// flame's sheath, 2 its core, 3 discs across it (what is seen looking up the
// jet). Dry, the jet only makes the air shimmer; with the afterburners lit
// it is a flame five metres long with shock diamonds standing in it.
fn plume(in: VOut) -> vec4f {
  let layer = u32(in.mat.w + 0.5);
  let ab = burner();
  let dry = dry_power();
  let L = in.L;
  let z = L.z - NOZZLE_Z;
  let V = normalize(in.P);
  let ndv = abs(dot(normalize(in.N), V));
  let t = G.time.x;
  let side = sign(L.x);
  // Turbulence streaming aft with the jet.
  let n1 = nz(vec3f(L.x * 1.7 + side * 3.1, L.y * 1.7, z * 0.30 - t * 9.0));
  let n2 = nz(vec3f(L.x * 5.0 + side, L.y * 5.0, z * 0.9 - t * 21.0));
  if (layer == 0u) {
    // Heat haze: what lies behind the jet shimmers through it.
    let k = (0.10 + 0.55 * dry + 0.6 * ab) * (0.3 + 0.7 * smoothstep(0.0, 0.6, ndv)) * smoothstep(-0.05, 0.8, z) * (1.0 - smoothstep(5.0, 11.5, z));
    if (k < 0.01) { discard; }
    let off = (vec2f(n1.g, n1.b) - 0.5 + (vec2f(n2.r, n2.a) - 0.5) * 0.6) * k * 0.05 / max(length(in.P), 2.0);
    let uv = in.clip.xy * G.screen.zw + off;
    var seen = textureLoad(t_scene, vec2i(in.clip.xy), 0).rgb;
    let q = vec2i(clamp(uv * G.screen.xy, vec2f(0.0), G.screen.xy - 1.0));
    if (textureLoad(t_gdepth, q, 0) <= in.clip.z) { seen = textureSampleLevel(t_scene, s_clamp, uv, 0.0).rgb; }
    let a = sat(k * 1.6);
    return vec4f(seen * a, a);
  }
  let l = light_at_nv(in.P);
  let night = 1.0 - smoothstep(-0.08, 0.1, dot(l.upn, G.sun_dir.xyz));
  // Shock diamonds: bright knots where the pressure waves in the jet cross,
  // closer together and fainter aft; and the cone of flame off the nozzle.
  let zd = z * (1.0 + 0.035 * z);
  let dia = pow(0.5 + 0.5 * cos((zd - 0.62) * 6.2832 / 0.88), 3.0) * exp(-max(z, 0.0) / 2.6) * smoothstep(0.15, 0.5, z);
  let cone = exp(-max(z, 0.0) / 0.45);
  let flick = 0.72 + 0.56 * n1.r;
  var col = vec3f(0.0);
  var glow = 0.0;
  if (layer == 2u) {
    // The core: blue-white off the nozzle, the diamonds yellow-white, orange further back.
    let s = z / 4.5;
    col = mix(vec3f(0.55, 0.70, 1.0), vec3f(1.0, 0.72, 0.36), smoothstep(0.0, 0.6, s));
    col = mix(col, vec3f(1.0, 0.90, 0.70), sat(dia * 1.4));
    glow = ab * (0.10 + 1.1 * cone + 2.3 * dia) * pow(ndv, 1.4) * (1.0 - smoothstep(0.5, 1.0, s)) * flick;
  } else if (layer == 1u) {
    // The sheath: a thin orange going to red, with a violet edge near the nozzle.
    let s = z / 6.9;
    col = mix(vec3f(1.0, 0.46, 0.14), vec3f(0.85, 0.18, 0.04), smoothstep(0.1, 0.9, s));
    col = mix(col, vec3f(0.50, 0.40, 1.0), 0.45 * (1.0 - ndv) * (1.0 - smoothstep(0.0, 0.45, s)));
    glow = ab * (0.20 + 0.25 * cone + 0.40 * dia) * pow(ndv, 2.0) * (1.0 - smoothstep(0.25, 1.0, s)) * (0.5 + 1.0 * n2.g);
    // Dry, at full power: a faint glow off the nozzle, only seen at dusk.
    glow += dry * dry * 0.012 * pow(ndv, 2.0) * (1.0 - smoothstep(0.0, 0.3, s));
  } else {
    // Discs across the jet: its glow seen from behind, down its length.
    let s = z / 5.6;
    let r = plume_r(L) / max(0.36 * (1.0 - 0.75 * s), 0.05);
    col = mix(vec3f(0.75, 0.80, 1.0), vec3f(1.0, 0.55, 0.20), sat(smoothstep(0.0, 0.7, s) * 0.6 + 0.6 * r));
    glow = ab * 0.20 * pow(ndv, 3.0) * pow(sat(1.0 - r * r), 1.5) * (1.0 - smoothstep(0.4, 1.0, s)) * (0.6 + 1.2 * dia + 0.6 * cone) * flick;
  }
  glow *= smoothstep(-0.14, 0.06, z);
  if (glow < 0.0005) { discard; }
  // (Turned down at night, when the eye has opened up; it still blooms.)
  return vec4f(col * glow * G.sun_dir.w * mix(0.06, 0.007, night), 0.0);
}

// Lit colour, normal and SSR weight, and whether the fragment lies in an opening.
struct Shaded { c: vec4f, n: vec4f, kill: f32, amb: f32 };

// Pieces with openings cut in them (skin, lining) discard; everything else
// goes through fs_solid, which never does, so early depth testing stays on.
@fragment fn fs(in: VOut) -> GOut {
  let s = surface(in);
  if (s.kill > 0.5) { discard; }
  return GOut(s.c, s.n, s.amb);
}
@fragment fn fs_solid(in: VOut) -> GOut {
  let s = surface(in);
  return GOut(s.c, s.n, s.amb);
}

fn surface(in: VOut) -> Shaded {
  let id = u32(in.mat.x + 0.5);
  let pt = u32(in.mat.y + 0.5);
  // Everything that needs screen derivatives first, while control flow is uniform.
  let dPx = dpdx(in.P);
  let dPy = dpdy(in.P);
  let px = max(length(dPx), length(dPy)) / kscale();
  let duv = max(length(dpdx(in.uv)), length(dpdy(in.uv)));
  let dLx = dpdx(in.L);
  let dLy = dpdy(in.L);
  var rel = relief(pt, in.uv, in.L, in.Nl, px);
  // The Hornet's paint; its panel lines and rivets, darker than the skin
  // round them, are taken to lie a little below it.
  var paint = vec4f(0.0);
  if (kind() == 2u && pt == P_HORNET) {
    paint = hornet_paint(in.L, in.Nl, dLx, dLy);
    rel.x += luma(paint.rgb) * 0.004 * paint.a * (1.0 - smoothstep(0.003, 0.012, px));
  }
  let hx = dpdx(rel.x);
  let hy = dpdy(rel.x);
  let cuts = pt == P_FUSE || pt == P_COWL || pt == P_LINING;
  var wd = 1.0;
  var wl = 1.0;
  if (cuts) { wd = window(in.L); }
  // The 747's lining is cut back round its windows by the width of the trim (b747::deck::lining_window).
  if (kind() == 1u && pt == P_LINING) {
    // Measured on the skin straight outside (radially from 0.3 m up), less the trim.
    let rr = normalize(vec2f(in.L.x, in.L.y - 0.3));
    wd = window(in.L + vec3f(rr * 0.10, 0.0)) - 0.03;
  }
  if (pt == P_FUSE || pt == P_COWL) { wl = well(in.L); }
  // The well outlines are drawn in plan (z, x); on the steep sides of the
  // belly that stretches distances along the skin, so measure the edge there.
  let wl_s = wl * px / max(length(vec2f(dpdx(wl), dpdy(wl))), 1e-7);
  let td = tear_d(in.L);
  let kill = (cuts && (wd < 0.0 || wl < 0.0)) || id == M_GLASS || id == M_PROPDISC || id == M_PLUME || td > 0.0;

  var n0 = in.N;
  if (dot(n0, n0) < 1e-10) { n0 = -in.P; }
  n0 = normalize(n0);
  let r1 = cross(dPy, n0);
  let r2 = cross(n0, dPx);
  let det = dot(dPx, r1);
  var N = n0;
  if (abs(det) > 1e-14) { N = normalize(abs(det) * n0 - sign(det) * (hx * r1 + hy * r2)); }
  let V = normalize(in.P);
  var Ng = n0;
  let back = dot(n0, V) > 0.0;
  if (back) { N = -N; Ng = -n0; }
  let l = light_at(in.P);
  let night = 1.0 - smoothstep(-0.08, 0.1, dot(l.upn, G.sun_dir.xyz));
  let L = in.L;
  let ns = nz(L * 2.1);
  let fine = nz(L * 13.0);

  var albedo = in.col.rgb;
  var rough = 0.5;
  var metal = 0.0;
  var emit = vec3f(0.0);
  var spec_k = 1.0;
  var ssr = 0.0;
  var ao = in.col.a;
  let cav = 1.0 - rel.y * 0.4;

  switch id {
    case 0u, 1u, 15u: {
      if (id == 1u) { albedo = livery(L, px * 1.2, pt); }
      rough = select(0.13 + 0.08 * ns.g, 0.5 + 0.15 * ns.g, id == 15u);
      if (pt == P_HORNET && paint.a > 0.5) {
        // The Hornet's matte tactical greys, as painted; fluid stains (darker and browner) are a little glossier.
        albedo = paint.rgb * 0.85;
        rough = 0.48 + 0.14 * ns.g - 0.12 * smoothstep(0.02, 0.10, paint.r - paint.b);
      }
      // Each skin panel's paint aged a little differently.
      albedo *= 0.985 + 0.03 * fine.a * select(0.4, 1.0, id == 1u);
      if (id == 1u || pt == P_FUSE) {
        let w = weather(L, in.Nl, albedo);
        albedo = w.rgb;
        rough += w.a;
        // Chipped paint where things get knocked: leading edges, around fasteners.
        let chip = smoothstep(0.80, 0.83, nz(L * 7.0).b + in.wear * 0.07) * sat(in.wear * 1.5);
        albedo = mix(albedo, vec3f(0.55, 0.56, 0.58), chip);
        metal = chip;
        rough = mix(rough, 0.35, chip);
        // Bugs on the leading edges.
        let bugs = smoothstep(0.86, 0.9, nz(L * 9.0).a) * smoothstep(0.75, 0.97, in.wear) * sat(-in.Nl.z * 2.0) * (1.0 - smoothstep(0.002, 0.008, px));
        albedo = mix(albedo, vec3f(0.12, 0.09, 0.03), bugs * 0.7);
        ssr = 1.0;
      }
      if (pt == P_PIT) {
        // The Hornet's cockpit grey: satin, worn where boots and elbows go,
        // grimier low down, scuffed through to lighter primer here and there.
        rough = 0.55 + 0.12 * fine.g;
        albedo *= 0.9 + 0.2 * nz(L * 3.1).r;
        albedo *= mix(0.7, 1.0, sat((L.y - FLOOR18) / 0.35));
        let sc = smoothstep(0.80, 0.9, nz(vec3f(L.x * 34.0, L.y * 34.0, L.z * 5.0)).r) * smoothstep(0.45, 0.7, nz(L * 4.3).b);
        albedo = mix(albedo, albedo * 1.8 + vec3f(0.03), sc * 0.6);
        rough = mix(rough, 0.4, sc);
        albedo = mix(albedo, albedo * 0.6, smoothstep(0.62, 0.85, nz(L * 9.0).g) * 0.4);
      }
      if (pt == P_NONSLIP) {
        rough = 0.92;
        spec_k = 0.4;
        albedo *= 0.8 + 0.4 * nz(L * 60.0).r;
        albedo = mix(albedo, vec3f(0.06, 0.055, 0.05), smoothstep(0.5, 0.8, nz(L * 7.0).g) * 0.5);
      }
      if (pt == P_STRIPE) {
        // Yellow and black: diagonal on a face, rings along a tube or strap.
        let w = in.mat.w;
        let t = select(in.uv.x + in.uv.y, in.uv.y, w < 0.0) / max(abs(w), 1e-4);
        let blk = smoothstep(0.47, 0.53, abs(fract(t) - 0.5) * 2.0 + 0.0);
        albedo = mix(albedo, vec3f(0.012), blk) * (0.85 + 0.25 * fine.r);
        rough = 0.35 + 0.15 * fine.g;
      }
      if (kind() == 1u && (id == 1u || pt == P_SPINNER)) {
        let fn_ = skin747(Fin(albedo, rough, metal), pt, L, in.Nl, in.uv, in.wear, px);
        albedo = fn_.c;
        rough = fn_.rough;
        metal = fn_.metal;
      }
      // Window sealant and the edges of the gear wells.
      if (pt == P_FUSE || pt == P_COWL) {
        let gk = smoothstep(0.016 + px, 0.016 - px, wd) * step(-0.001, wd);
        let we = smoothstep(0.004 + px, 0.004 - px, wl_s) * step(-0.001, wl);
        albedo = mix(albedo, vec3f(0.012), max(gk, we * 0.6));
        rough = mix(rough, 0.6, gk);
        ssr *= 1.0 - gk;
        if (pt == P_COWL && L.z < -2.97 && kind() == 0u) {
          // Cooling air inlets in the nose bowl.
          let e1 = length((vec2f(abs(L.x), L.y) - vec2f(0.235, 0.05)) / vec2f(0.105, 0.068));
          let e2 = length((L.xy - vec2f(0.0, -0.33)) / vec2f(0.11, 0.045));
          let inlet = smoothstep(1.0 + px * 10.0, 1.0 - px * 10.0, min(e1, e2));
          albedo = mix(albedo, vec3f(0.006) * (0.4 + 0.6 * smoothstep(0.0, 1.0, min(e1, e2))), inlet);
          rough = mix(rough, 0.9, inlet);
          ao *= 1.0 - 0.6 * inlet;
          ssr *= 1.0 - inlet;
        }
      }
      albedo *= cav;
      // Rivet heads catch a little grime round their edges.
      albedo *= 1.0 - rel.z * (1.0 - rel.z) * 0.35 - rel.z * 0.03;
    }
    case 2u: { // bare aluminium
      metal = 1.0;
      rough = 0.22 + 0.18 * fine.g;
      albedo *= 0.9 + 0.12 * ns.r;
      ssr = 0.6;
    }
    case 3u: { // steel: 0 machined, 1 chrome, 2 cast iron / heat-tinted
      metal = 1.0;
      rough = select(select(0.25, 0.05, in.mat.w > 0.5 && in.mat.w < 1.5), 0.45, in.mat.w > 1.5);
      if (in.mat.w > 1.5) {
        albedo = mix(albedo, vec3f(0.30, 0.16, 0.07), smoothstep(0.4, 0.7, fine.r) * 0.6);
        metal = 0.8;
      }
      ssr = 0.5;
    }
    case 4u: { // rubber: dusty, the tread worn lighter
      rough = 0.85;
      albedo = mix(albedo, vec3f(0.07, 0.065, 0.058), sat(rel.y * 0.0 + smoothstep(0.3, 0.8, ns.g) * 0.4 + select(0.0, 0.45, pt == P_TREAD && rel.y < 0.3)));
      spec_k = 0.4;
      if (kind() == 1u) {
        // An airliner's tyres: near-black, greyed by brake dust and rubber
        // scrubbed off on landing; the sidewalls a touch glossier.
        let tread = select(0.0, 1.0, pt == P_TREAD && abs(in.Nl.x) < 0.5);
        albedo = mix(vec3f(0.022, 0.022, 0.023), vec3f(0.05, 0.049, 0.048), sat(smoothstep(0.3, 0.8, ns.g) * 0.5 + tread * 0.5));
        rough = mix(0.62, 0.85, tread);
      }
    }
    case 5u: {
      rough = 0.5 + 0.2 * fine.b;
      spec_k = 0.6;
      albedo *= cav;
      if (pt == P_LEGEND) {
        // Printed legends, button faces, scales (deck747.wgsl): white paint
        // lit from behind at night; annunciators that are on glow by day too.
        let code = u32(floor(in.mat.w));
        let lg = legend(code % 16u, code / 16u, in.uv, max(in.wear, 0.05), max(duv, 0.002), fract(in.mat.w) * 100.0);
        albedo = mix(albedo, lg.col, lg.a);
        rough = mix(rough, 0.35, lg.a);
        emit = (lg.glow * mix(0.0, 0.004, night) + lg.lit * mix(0.05, 0.012, night)) * G.sun_dir.w;
        // (The Hornet's panels are edge-lit in NVG green.)
        if (kind() == 2u) { emit *= vec3f(0.55, 1.0, 0.62); }
      }
      if (pt == P_LINING && kind() == 1u) {
        // The flight deck's lining: matte, stippled plastic.
        rough = 0.8 + 0.08 * fine.b;
        spec_k = 0.45;
      }
      if (pt == P_PANEL && kind() == 1u) {
        // Boeing's grey-brown: a fine texture, scuffs and grime where hands go.
        rough = 0.55 + 0.15 * fine.g;
        albedo *= 0.92 + 0.12 * nz(L * 37.0).r;
        albedo = mix(albedo, albedo * 0.7, smoothstep(0.62, 0.85, nz(L * 6.0).g) * 0.5);
      }
      if (pt == P_PANEL && kind() == 2u) {
        // Black lacquered panels: satin, dusty where they face up, rubbed shiny here and there.
        rough = 0.45 + 0.12 * fine.g - 0.12 * smoothstep(0.75, 0.92, nz(L * 23.0).r);
        albedo *= 0.94 + 0.12 * nz(L * 41.0).r;
        albedo = mix(albedo, vec3f(0.055, 0.054, 0.052), smoothstep(0.55, 0.85, nz(L * 13.0).g) * sat(in.Nl.y * 2.0) * 0.18);
      }
      if (pt == P_PLACARD) {
        let t = fract(in.uv * vec2f(220.0, 340.0));
        let txt = step(0.3, t.y) * step(t.y, 0.72) * step(0.45, nz(vec3f(in.uv * 300.0, 0.5)).r);
        albedo = mix(albedo, vec3f(0.02), txt * 0.8 * (1.0 - smoothstep(0.0003, 0.001, px)));
      }
    }
    case 6u: { // fabric: rough with a soft sheen at grazing angles
      rough = 0.95;
      spec_k = 0.15;
      albedo *= cav * (0.9 + 0.2 * fine.r);
      if (pt == P_MESH) {
        // Olive mesh: the holes dark, the weave faded by the sun unevenly.
        albedo *= (1.0 - rel.y * 0.55) * (0.85 + 0.3 * nz(L * 3.7).g);
        albedo = mix(albedo, albedo * vec3f(1.15, 1.1, 1.25), smoothstep(0.5, 0.8, nz(L * 1.3).r) * 0.5);
      }
      if (pt == P_WEBBING) {
        albedo *= (1.0 - rel.y * 0.4) * (0.9 + 0.2 * nz(L * 9.0).r);
        spec_k = 0.3;
      }
      if (pt == P_FLEECE) {
        // Sheepskin: tufts of wool, lighter at their tips.
        let tuft = nz(L * 70.0);
        albedo *= 0.65 + 0.6 * tuft.r * tuft.g + 0.25 * nz(L * 9.0).b;
        ao *= 0.75 + 0.25 * tuft.g;
      }
      let ndv = max(dot(N, -V), 0.0);
      albedo *= 1.0 + 0.35 * pow(1.0 - ndv, 3.0);
    }
    case 8u: { // light lens
      rough = 0.08;
      let ch = u32(in.mat.w + 0.5);
      let P4 = O.params[4];
      let P5 = O.params[5];
      var on = 0.0;
      var k = 0.0;
      switch ch {
        case 0u, 1u, 2u: { on = step(999.0, P5.x); k = mix(0.08, 0.6, night); }
        case 3u: { on = P5.y; k = 3.0; }
        case 4u: { on = P5.z; k = mix(0.5, 0.6, night); }
        case 5u: { on = P5.w; k = mix(0.3, 0.8, night); }
        case 6u: { on = step(0.999, P4.z); k = mix(0.05, 0.01, night); }
        case 7u: { on = step(P4.z, 0.001); k = mix(0.05, 0.01, night); }
        // Formation light strips: pale panels by day, a soft green glow with the navigation lights at night.
        case 9u: { on = step(999.0, P5.x); k = 0.012 * night; }
        default: { on = 1.0; k = 0.004; }
      }
      emit = in.col.rgb * on * k * G.sun_dir.w;
      if (ch == 5u) { emit *= 0.6 + 0.4 * smoothstep(1.0, 0.0, length(in.uv)); }
      albedo = in.col.rgb * 0.15;
      if (ch == 9u) { albedo = vec3f(0.40, 0.43, 0.27); rough = 0.4; }
      ssr = 0.5;
    }
    case 9u: { // instrument face under glass
      let g = gauge(u32(in.mat.w + 0.5), in.uv, max(duv * 1.2, 0.004));
      albedo = g.rgb;
      rough = 0.04;
      spec_k = 0.5;
      emit = g.rgb * g.a * G.sun_dir.w * 0.0035 * night;
    }
    case 10u: { // matte black: scuffed
      rough = 0.65 + 0.2 * fine.g;
      albedo *= 0.8 + 0.5 * smoothstep(0.6, 0.9, fine.a);
      spec_k = 0.5;
    }
    case 11u: { // exhaust: heat-tinted, sooted at the outlet
      metal = 0.75;
      if (kind() == 2u) {
        // The Hornet's nozzles. in.mat.w: 0 the petals outside, 1 inside the
        // nozzle, 2 the flame holder at the bottom of it.
        let part = u32(in.mat.w + 0.5);
        let s = sat((L.z - 7.15) / 0.85);  // 0 at the shroud .. 1 at the lip
        if (part == 0u) {
          // Titanium petals: straw and blue heat tints toward the lip, soot streaked along them.
          rough = 0.38 + 0.2 * fine.g;
          metal = 0.9;
          let streak = nz(vec3f(atan2(L.y + 0.05, abs(L.x) - 0.52) * 2.4, 0.3, L.z * 0.6)).r;
          albedo = mix(vec3f(0.36, 0.34, 0.32), vec3f(0.30, 0.23, 0.16), smoothstep(0.05, 0.55, s));
          albedo = mix(albedo, vec3f(0.17, 0.165, 0.20), smoothstep(0.5, 0.9, s) * 0.7);
          albedo *= (0.55 + 0.6 * streak) * (0.8 + 0.3 * fine.r);
          albedo = mix(albedo, vec3f(0.03), smoothstep(0.93, 1.0, s) * 0.7);
        } else {
          // Sooted black inside; glowing when the burners are lit, and a dull
          // red from the turbine at full dry power, deep in.
          rough = 0.85;
          metal = 0.2;
          albedo = vec3f(0.035, 0.032, 0.03) * (0.6 + 0.8 * fine.g);
          let deep = sat((NOZZLE_Z - L.z) / 0.58);
          let fl = 0.8 + 0.4 * nz(vec3f(L.xy * 4.0, G.time.x * 7.0)).r;
          let dry = dry_power();
          var heat = burner() * fl * mix(0.35, 1.3, deep) + dry * dry * 0.03 * deep;
          if (part == 2u) { heat = burner() * fl * 2.4 + dry * dry * 0.07; }
          emit = mix(vec3f(1.0, 0.30, 0.06), vec3f(1.0, 0.72, 0.38), sat(heat * 0.6)) * heat * G.sun_dir.w * mix(0.08, 0.014, night);
          ao = max(ao, 0.3);
        }
        break;
      }
      if (kind() == 1u) {
        // A turbofan's nozzle and plug: heat-blued steel, darker toward the outlet.
        rough = 0.4;
        albedo = mix(vec3f(0.36, 0.33, 0.30), vec3f(0.10, 0.09, 0.10), sat(fine.r * 0.6 + 0.2)) * (0.75 + 0.4 * ns.g);
        albedo = mix(albedo, vec3f(0.30, 0.17, 0.09), smoothstep(0.55, 0.8, ns.b) * 0.5);
        break;
      }
      rough = 0.45;
      let t = sat((L.y + 0.45) / -0.3);
      albedo = mix(vec3f(0.42, 0.33, 0.20), vec3f(0.18, 0.16, 0.20), t) * (0.6 + 0.6 * fine.r);
      albedo = mix(albedo, vec3f(0.35, 0.18, 0.08), smoothstep(0.55, 0.8, ns.b) * 0.6);
      albedo = mix(albedo, vec3f(0.02), smoothstep(-2.2, -2.11, L.z));
    }
    case 12u: { // screw head: a cross recess, paint worn off the rim
      let q = in.uv;
      let cr = min(max(abs(q.x) - 0.15, abs(q.y) - 0.62), max(abs(q.y) - 0.15, abs(q.x) - 0.62));
      let slot = 1.0 - smoothstep(-0.05, 0.05, cr);
      let kind = in.mat.w;
      metal = select(1.0, select(0.0, 0.4, kind > 1.5), kind > 0.5);
      rough = select(0.3, select(0.18, 0.5, kind > 1.5), kind > 0.5);
      if (kind > 0.5 && kind < 1.5) {
        let worn = smoothstep(0.6, 0.85, length(q)) * smoothstep(0.4, 0.7, fine.g) + slot * 0.6;
        albedo = mix(albedo, vec3f(0.5, 0.5, 0.52), sat(worn));
        metal = sat(worn);
      }
      albedo *= 1.0 - slot * 0.9;
      ao *= 1.0 - slot * 0.7;
    }
    case 13u: { // propeller: grey face, black back, painted tips, eroded edge
      let r = in.uv.x;
      let front = in.uv.y > 0.0;
      albedo = select(vec3f(0.018), vec3f(0.30, 0.30, 0.31), front);
      rough = select(0.75, 0.4, front);
      let tip = smoothstep(0.864, 0.866, r);
      let band = smoothstep(0.865, 0.867, r) * smoothstep(0.885, 0.883, r);
      albedo = mix(albedo, vec3f(0.78, 0.78, 0.74), tip);
      albedo = mix(albedo, vec3f(0.5, 0.03, 0.02), band);
      let ero = smoothstep(0.5, 0.95, in.wear) * smoothstep(0.35, 0.9, r) * smoothstep(0.4, 0.7, fine.r);
      albedo = mix(albedo, vec3f(0.6, 0.6, 0.62), ero);
      metal = ero;
    }
    case 14u: { // backlit display under glass
      albedo = vec3f(0.01);
      rough = 0.05;
      let did = u32(in.mat.w + 0.5);
      if (did >= 40u) {
        // The Hornet's DDIs, AMPCD, UFC windows and IFEI (pit18.wgsl).
        emit = display18(did, in.uv, duv) * G.sun_dir.w * mix(0.06, 0.009, night);
        albedo = vec3f(0.010, 0.012, 0.012);
        rough = 0.1;
        ssr = 0.3;
      } else if (did >= 20u) {
        // The 747's CRTs: bright enough by day, turned down at night.
        emit = display747(did, in.uv, duv) * G.sun_dir.w * mix(0.07, 0.01, night);
        albedo = vec3f(0.012, 0.014, 0.016);
        rough = 0.12;
        ssr = 0.3;
      } else {
      emit = display(did, in.uv, max(duv * 0.012, 0.00015)) * G.sun_dir.w * mix(0.09, 0.012, night);
      }
    }
    default: {}
  }

  // The 747's skin seen from the flight deck (slivers at the window cut-outs): insulation blanket.
  if (kind() == 1u && back && pt == P_FUSE && L.z < -22.0 && !torn()) {
    albedo = vec3f(0.035, 0.034, 0.032);
    metal = 0.0;
    rough = 0.9;
    ssr = 0.0;
  }
  // Damage. The inside of a torn skin: primer, stringers and ribs, no paint.
  let exterior = pt == P_WING || pt == P_TAIL || pt == P_CTRL || pt == P_FUSE || pt == P_COWL;
  if (torn() && back && exterior) {
    var inner = vec3f(0.30, 0.36, 0.24) * (0.7 + 0.5 * fine.g);
    let rib = line((fract((abs(L.x) - 0.62) / 0.305 + 0.5) - 0.5) * 0.305, 0.012);
    let str_ = line((fract(L.y / 0.09 + 0.5) - 0.5) * 0.09, 0.006) * select(0.0, 1.0, pt == P_FUSE);
    inner *= 1.0 - 0.5 * max(rib, str_);
    albedo = inner;
    metal = 0.0;
    rough = 0.6;
    spec_k = 0.5;
    ssr = 0.0;
    emit = vec3f(0.0);
    // Open to the sky through the break: more light gets in than the bake knew.
    ao = max(ao, 0.55);
  }
  // The torn edge: bare, crumpled metal curling back from the break.
  if (torn()) {
    let rip = smoothstep(-0.05, -0.004, td);
    let shred = nz(L * 40.0).r;
    albedo = mix(albedo, vec3f(0.36, 0.37, 0.39) * (0.5 + 0.8 * shred), rip);
    metal = mix(metal, 0.85, rip);
    rough = mix(rough, 0.5, rip);
    N = normalize(N + (nz(L * 23.0).gba - 0.5) * rip * 1.2);
    ao *= 1.0 - 0.35 * rip;
  }
  // Scrapes and dents from trees, roofs and the ground, streaked back along
  // the airflow; green smears where foliage went over the leading edges.
  if (O.dmg.z > 0.0 && exterior && (id == M_LIVERY || id == M_PAINT || id == M_SATIN)) {
    let scuff = O.dmg.z;
    // Where it gets knocked: leading edges, the belly, and patches at random.
    let where_ = sat(in.wear * 1.5 + sat(-in.Nl.y) * 0.5) * 0.6 + 0.6 * smoothstep(0.55, 0.8, nz(L * 1.7).a);
    let sc = nz(vec3f(L.x * 9.0, L.y * 9.0, L.z * 1.3));
    let scratch = smoothstep(0.84 - 0.2 * scuff * where_, 0.9 - 0.2 * scuff * where_, sc.r) * scuff * where_;
    albedo = mix(albedo, vec3f(0.5, 0.5, 0.52), scratch * 0.8);
    metal = max(metal, scratch * 0.7);
    rough = mix(rough, 0.4, scratch);
    let sap = smoothstep(0.4, 0.95, in.wear) * scuff * smoothstep(0.45, 0.7, sc.b);
    albedo = mix(albedo, vec3f(0.12, 0.16, 0.05), sap * 0.5);
    // Dents.
    N = normalize(N + (nz(L * 4.0).gba - 0.5) * 0.3 * scuff * where_ * smoothstep(0.5, 0.75, nz(L * 2.3).r));
  }

  // Wreckage: soot and burn-through.
  let cn = nz(L * 0.35 + O.glow.w);
  let char = clamp(O.tint.a * 1.6 - cn.g * 0.7, 0.0, 1.0);
  albedo = mix(albedo, vec3f(0.015, 0.013, 0.012), char);
  rough = mix(rough, 0.9, char);
  metal *= 1.0 - char;
  ssr *= 1.0 - char;

  // Light: sun (with the aircraft's own fine shadow), sky, ground bounce, moon.
  let sh = shadow_at(in.P, Ng);
  let Ls = G.sun_dir.xyz;
  let ndl = max(dot(N, Ls), 0.0);
  let ndv = max(dot(N, -V), 1e-3);
  let H = normalize(Ls - V);
  let a = max(rough * rough, 0.002);
  let F0 = mix(vec3f(0.04), albedo, metal);
  let Fs = F0 + (1.0 - F0) * pow(1.0 - max(dot(-V, H), 0.0), 5.0);
  let spec = l.sun * ggx(max(dot(N, H), 0.0), a) * vis(ndv, ndl, a) * Fs * ndl * sh * spec_k;
  // Inside the cabin, light bounces around off every surface: a fill that the
  // occlusion term alone would miss.
  var inside = step(abs(L.x), 0.58) * step(-1.42, L.z) * step(L.z, 1.5) * step(L.y, 0.62) * select(0.0, 1.0, id != M_LIVERY && pt != P_FUSE && pt != P_COWL);
  if (kind() == 1u) { inside = step(abs(L.x), 2.2) * step(-29.0, L.z) * step(L.z, -22.4) * step(1.9, L.y) * select(0.0, 1.0, id != M_LIVERY && pt != P_FUSE); }
  // The Hornet's cockpit tub, under its canopy.
  if (kind() == 2u) { inside = step(abs(L.x), 0.56) * step(-6.2, L.z) * step(L.z, -3.1) * step(0.0, L.y) * select(0.0, 1.0, pt != P_HORNET); }
  // (The baked occlusion already counts only the sky seen through the
  // windows; the eye adapts to the dimmer cabin, main.rs / cockpit.rs.)
  let amb_ao = mix(ao * ao * (3.0 - 2.0 * ao), 0.06 + 0.94 * ao, inside);
  let fill = (l.amb * 0.2 + l.sun * max(dot(l.upn, Ls), 0.0) * 0.03) * (0.25 + 0.75 * ao) * inside;
  let bounce = l.sun * max(dot(l.upn, Ls), 0.0) * 0.05 * sat(0.5 - 0.5 * dot(N, l.upn)) + l.sun * max(dot(l.upn, Ls), 0.0) * 0.025;
  let moon = moon_light(l.upn, l.h) * max(dot(N, G.moon.xyz), 0.0);
  let diffuse = albedo * (1.0 - metal) * ((l.sun * ndl * sh + moon) / PI + (l.amb * (0.55 + 0.45 * dot(N, l.upn)) + bounce) * amb_ao + fill + point_light(in.P, N));
  let R = reflect(V, N);
  let env_sharp = env_reflect(in.P, R, l.upn, false);
  let env_soft = sky_lut(normalize(mix(R, l.upn, 0.5))) * 0.9;
  let Fe = F0 + (max(vec3f(1.0 - rough), F0) - F0) * pow(1.0 - ndv, 5.0);
  let spec_occ = sat(pow(ndv + ao, 2.0) - 1.0 + ao) * ao;
  let envw = Fe * spec_occ * spec_k;
  var c = diffuse + spec + mix(env_sharp, env_soft, sat(rough * 1.6)) * envw;
  // The cabin has its own occlusion baked in; only the outside takes the screen-space kind.
  g_amb = albedo * (1.0 - metal) * (l.amb * (0.55 + 0.45 * dot(N, l.upn)) + bounce) * amb_ao * (1.0 - inside);
  // Instrument flood lighting at night, warm and dim, inside the cabin.
  c += albedo * vec3f(1.0, 0.62, 0.38) * G.sun_dir.w * 0.0025 * night * inside * ao;
  c += emit;
  // Embers along the charred edges while it burns.
  let ember = smoothstep(0.45, 0.7, cn.b) * smoothstep(0.2, 0.6, char) * O.glow.x;
  c += vec3f(1.0, 0.28, 0.04) * ember * G.sun_dir.w * 1.5;
  // Screen-space reflections (in the composite) only where they would show.
  let rw = dot(envw, vec3f(0.333));
  let reflect_w = select(0.0, rw, ssr > 0.0 && rough < 0.3 && rw > 0.09) * ssr;
  return Shaded(vec4f(c, 1.0), vec4f(N, -reflect_w), select(0.0, 1.0, kill), amb_share(c));
}

// ---------------------------------------------------------------- transparent pass

// How far a window bends the view through it (radians along its two in-plane
// axes). Formed acrylic is never optically flat: broad lens-like swells, faint
// ripples drawn into the sheet, and worse where it is clamped at the frame and
// where the windshield is pulled round its corners.
fn glass_bend(g: vec2f, edge: f32, curve: f32) -> vec2f {
  let a = textureSampleLevel(t_wave, s_repeat, g * 0.8 + vec2f(0.31, 0.77), 0.0).rg * 2.0 - 1.0;
  let b = textureSampleLevel(t_wave, s_repeat, vec2f(g.x * 0.3, g.y * 2.4) + vec2f(0.61, 0.23), 0.0).rg * 2.0 - 1.0;
  let c = textureSampleLevel(t_wave, s_repeat, g * 2.7 + vec2f(0.13, 0.52), 0.0).rg * 2.0 - 1.0;
  let s = a * 0.005 + vec2f(b.x * 0.3, b.y) * 0.0026 + c * 0.0006;
  return s * (1.0 + 2.5 * edge + 1.5 * curve);
}
// What the inside of a window mirrors, seen from the cabin: the glareshield
// in the windshield (lit where the sun falls on it), elsewhere the dim cabin.
fn cabin_mirror(p0: vec3f, r: vec3f, l: Light, ls: vec3f) -> vec3f {
  var t = 0.01;
  for (var i = 0; i < 22; i++) {
    let p = p0 + r * t;
    let g = (GS_Z + 0.025 - p.z) / 0.46;
    if (g >= 0.0 && g <= 1.0 && abs(p.x) < 0.56 && p.y < 0.185 + 0.12 * g * g * g) {
      let up = normalize(O.model[1].xyz);
      let sg = shadow_at((O.model * vec4f(p * kscale(), 1.0)).xyz, up);
      let weave = 0.75 + 0.5 * nz(p * 23.0).g;
      // Black fabric, a little dusty.
      return vec3f(0.055, 0.054, 0.056) * weave / PI * (l.sun * max(ls.y, 0.0) * sg + l.amb * 0.55);
    }
    if (p.y < 0.17 && p.z < GS_Z + 0.03) { break; }
    t += 0.03;
  }
  // Seats, lining, a shirt: lighter and darker patches as the eye moves.
  return l.amb * 0.035 * (0.5 + 1.0 * nz(r * 0.9 + vec3f(0.3, 0.1, 0.6)).r);
}
const GS_Z = -1.30; // cabin::PANEL_Z

@fragment fn fs_glass(in: VOut) -> @location(0) vec4f {
  let px = max(length(dpdx(in.P)), length(dpdy(in.P))) / kscale();
  let id = u32(in.mat.x + 0.5);
  if (id == M_PLUME) { return plume(in); }
  let V = normalize(in.P);
  var N = normalize(in.N);
  let from_inside = dot(N, V) > 0.0;
  if (from_inside) { N = -N; }
  let l = light_at(in.P);
  let sh = shadow_at(in.P, N);
  let Ls = G.sun_dir.xyz;
  if (id == M_PROPDISC) {
    // The blur of the spinning blades: their share of each ring, a slow
    // shutter beat, painted tips showing as a ring.
    let r = length(in.uv) * 0.975;
    if (r < 0.2 || r > 0.975) { discard; }
    let c = max(0.155 - 0.11 * pow((r - 0.55) / 0.42, 2.0), 0.05);
    var cover = 2.0 * c / (6.2831853 * r);
    let ang = atan2(in.L.y + 0.03, in.L.x);
    let beat = 0.65 + 0.35 * cos(2.0 * (ang - G.time.x * 1.9));
    let tip = smoothstep(0.86, 0.87, r);
    // One blade snapped: only the inner part of the disc is swept twice.
    if (torn()) { cover *= select(0.5, 1.0, r < 0.45); }
    let lit = l.sun * max(dot(N, Ls), 0.0) * sh / PI + l.amb * 0.8;
    let back = from_inside == false && dot(in.N, -V) > 0.0;
    var col = select(vec3f(0.30), vec3f(0.02), dot(in.N, V) < 0.0);
    col = mix(col, vec3f(0.78, 0.78, 0.74), tip);
    let alpha = sat(cover * beat * O.tint.x * (1.0 - smoothstep(0.955, 0.975, r)) * 1.3);
    let _b = back;
    return vec4f(col * lit * alpha, alpha);
  }
  // The Hornet's HUD combiner: the symbology, focused at infinity.
  if (kind() == 2u && in.L.y > 0.86 && abs(in.L.x) < 0.1 && in.L.z > -5.3 && in.L.z < -5.15) { return hud18(in, px); }
  let d = window(in.L);
  let cov = 1.0 - smoothstep(-px, px, d);
  if (cov <= 0.0) { discard; }
  let ndv = max(dot(N, -V), 0.0);
  let F = 0.04 + 0.96 * pow(1.0 - ndv, 5.0);
  let R = reflect(V, N);
  // The body frame (rigid): where the eye is, and the window's own axes.
  let ml = transpose(mat3x3f(O.model[0].xyz, O.model[1].xyz, O.model[2].xyz));
  let eye_l = -(ml * O.model[3].xyz) / kscale();
  var in_cabin = from_inside && abs(eye_l.x) < 0.6 && eye_l.y > -0.5 && eye_l.y < 0.85 && eye_l.z > -1.3 && eye_l.z < 1.6;
  if (kind() == 1u) { in_cabin = from_inside && abs(eye_l.x) < 1.6 && eye_l.y > 2.2 && eye_l.y < 4.8 && eye_l.z > -28.5 && eye_l.z < -22.5; }
  if (kind() == 2u) { in_cabin = from_inside && abs(eye_l.x) < 0.6 && eye_l.y > 0.2 && eye_l.y < 1.5 && eye_l.z > -6.3 && eye_l.z < -2.6; }
  let wind = abs(in.Nl.x) < 0.5;
  let tb = select(vec3f(0.0, 0.0, 1.0), vec3f(1.0, 0.0, 0.0), wind);
  let bb = select(vec3f(0.0, 1.0, 0.0), vec3f(0.0, 0.454, 0.891), wind);
  let g2 = vec2f(dot(in.L, tb), dot(in.L, bb));
  // Looking out, the glass mostly mirrors the dim cabin.
  var env = l.amb * 0.035;
  if (!from_inside) { env = env_reflect(in.P, R, l.upn, true); }
  // A sheet has two faces, and from inside both mirror the cabin.
  if (in_cabin) {
    env = select(cabin_mirror(in.L, ml * R, l, ml * Ls), deck_mirror(in.L, ml * R, l, ml * Ls), kind() == 1u) * 2.0 / (1.0 + F);
    if (kind() == 2u) { env = pit_mirror(in.L, ml * R, l, ml * Ls) * 2.0 / (1.0 + F); }
  }
  let H = normalize(Ls - V);
  let ndl = max(dot(N, Ls), 0.0);
  let glint = l.sun * ggx(max(dot(N, H), 0.0), 0.02) * vis(max(ndv, 1e-3), ndl, 0.02) * ndl * sh * (0.04 + 0.96 * pow(1.0 - max(dot(-V, H), 0.0), 5.0));
  // Dust, smudges, grime along the edges and bugs low on the windshield; they
  // light up when you look toward the sun.
  let L = in.L;
  let n1 = nz(L * 2.3);
  let n2 = nz(L * 17.0);
  var haze = 0.012 + 0.03 * smoothstep(0.45, 0.85, n1.r) + 0.02 * smoothstep(0.6, 0.9, n2.g);
  haze += 0.06 * smoothstep(-0.05, 0.0, d);
  // Dried rain: rings of grime left where drops evaporated, in clusters.
  var spots = 0.0;
  let spot_cs = 0.016;
  let fine = 1.0 - smoothstep(spot_cs * 0.06, spot_cs * 0.25, px);
  if (fine > 0.0) {
    let cl = smoothstep(0.5, 0.78, nz(L * 1.6 + vec3f(0.4, 0.0, 0.2)).g);
    if (cl > 0.0) {
      let cell = floor(g2 / spot_cs);
      var sp = 0.0;
      for (var j = -1; j <= 1; j++) { for (var i = -1; i <= 1; i++) {
        let c = cell + vec2f(f32(i), f32(j));
        if (hash2(c + 41.3) > 0.3) { continue; }
        let o = (c + vec2f(hash2(c + 7.7), hash2(c + 13.1))) * spot_cs;
        let rad = spot_cs * (0.15 + 0.5 * hash2(c + 23.9) * hash2(c + 29.3));
        let q = (g2 - o) / rad;
        let rr = length(q * vec2f(1.0, 1.0 + 0.3 * hash2(c + 3.3))) + 0.12 * (nz(vec3f(q * 0.8 + c, 0.5) * 0.2).r - 0.5);
        let ring = 1.0 - smoothstep(0.0, max(0.22, px * 1.2 / rad), abs(rr - 0.88));
        let fill = (1.0 - smoothstep(0.75, 0.95, rr)) * 0.3;
        sp = max(sp, max(ring, fill));
      } }
      spots = sp * cl * fine;
      haze += spots * 0.05;
    }
  }
  let bugs = smoothstep(0.86, 0.9, nz(L * 12.0).a) * step(L.z, -1.25) * smoothstep(0.55, 0.2, L.y) * select(1.0, 0.0, kind() == 1u) * (1.0 - smoothstep(0.002, 0.008, px));
  let toward = pow(max(dot(V, Ls), 0.0), 6.0);
  let lit = l.sun * sh * (0.15 + 2.5 * toward) + l.amb * 0.6;
  var rgb = env * F + glint + haze * lit * 0.35 + vec3f(0.05, 0.04, 0.02) * bugs * (l.amb + l.sun * sh * 0.2);
  // The spots scatter forward: they glow round the sun, and are faint grey otherwise.
  let fwd = max(dot(V, Ls), 0.0);
  rgb += spots * (l.amb * 0.02 + l.sun * sh * (0.003 + 0.06 * pow(fwd, 6.0) + 0.5 * pow(fwd, 60.0)));
  var trans = (1.0 - F) * (1.0 - haze) * (1.0 - bugs * 0.9) * 0.93;
  // Hairline scratches from years of wiping, fixed in the plastic. Each one
  // scatters sunlight only when it lies across the line to the sun, so the
  // lit ones ring the sun and slide round as the head moves.
  let face = dot(V, Ls);
  if (face > 0.2) {
    let dv = V - Ls;
    let d2 = vec2f(dot(dv, normalize((O.model * vec4f(tb, 0.0)).xyz)), dot(dv, normalize((O.model * vec4f(bb, 0.0)).xyz)));
    let dn = d2 / max(length(d2), 1e-5);
    let cs = 0.014;
    let cell = floor(g2 / cs);
    let w = max(px * 0.7, 0.00005);
    var sc = 0.0;
    for (var j = -1; j <= 1; j++) { for (var i = -1; i <= 1; i++) {
      let c = cell + vec2f(f32(i), f32(j));
      if (hash2(c) > 0.85) { continue; }
      // Mostly the circular strokes of a polishing cloth, some random.
      let o = (c + vec2f(hash2(c + 3.1), hash2(c + 5.7))) * cs;
      let swirl = normalize(vec2f(-(o.y - 0.4), o.x + 0.1) + vec2f(1e-4));
      let ra = hash2(c + 17.3) * 3.14159;
      let sd = normalize(mix(vec2f(cos(ra), sin(ra)), swirl, step(0.35, hash2(c + 2.9))));
      let len = cs * (0.8 + 2.2 * hash2(c + 9.2));
      let q = g2 - o;
      let along = dot(q, sd);
      let across = abs(q.x * sd.y - q.y * sd.x);
      let ln = (1.0 - smoothstep(w * 0.5, w * 1.5, across)) * (1.0 - smoothstep(len * 0.3, len * 0.5, abs(along)));
      let m = 1.0 - abs(dot(sd, dn));
      sc = max(sc, ln * pow(m, 14.0) * (0.25 + 0.75 * hash2(c + 1.7)));
    } }
    sc *= sqrt(min(1.0, 0.00015 / w));
    rgb += l.sun * sh * sc * (0.015 + 0.6 * pow(face, 18.0) + 12.0 * pow(face, 120.0)) * 1.6;
    // The ones too fine to see one by one, and the smears a rag leaves: a
    // milky, blotchy glare round the sun.
    let smear = 0.35 + 1.3 * smoothstep(0.3, 0.8, n1.g) + 0.5 * n2.b;
    rgb += l.sun * sh * smear * (0.002 + 0.03 * pow(face, 24.0) + 0.12 * pow(face, 400.0)) * 0.5;
  }
  if (L.z < -0.95 && kind() == 0u) {
    // Engine oil blown back over the windshield: a brown film, thickest low
    // down, drawn up and back by the airflow into streaks.
    let oil = O.dmg2.x;
    if (oil > 0.0) {
      let st = nz(vec3f(L.x * 14.0, L.y * 2.2 - L.z * 1.6, 0.3));
      let low = 1.0 - clamp((L.y - 0.05) / 0.8, 0.0, 1.0);
      let film = clamp((st.r * 0.7 + st.g * 0.45 + low * 0.7 - 1.05 + oil * 1.1) * 2.5, 0.0, 1.0) * smoothstep(0.0, 0.15, oil);
      rgb += vec3f(0.22, 0.13, 0.05) * film * (l.amb * 0.8 + l.sun * sh * 0.25 + l.sun * toward * 0.6);
      trans *= 1.0 - film * 0.92;
    }
    // A crack: radial spokes with rings between, scattering the light.
    let size = O.dmg2.w;
    if (size > 0.0) {
      let q = vec2f(L.z, L.y) - O.dmg2.yz;
      let r = length(q);
      if (r < size * 1.05) {
        let th = atan2(q.y, q.x);
        let k = round(th / 0.8976);
        let ak = k * 0.8976 + 0.25 * sin(k * 4.7) + 0.35 * (nz(vec3f(r * 5.0, k * 0.3, 0.7)).r - 0.5);
        let spoke = (1.0 - smoothstep(0.0, px * 1.5 + 0.0012, abs(sin(th - ak)) * r)) * step(r, size * (0.75 + 0.25 * nz(vec3f(k, 0.2, 0.4)).g));
        let ring = (1.0 - smoothstep(0.0, px * 1.5 + 0.001, abs(fract(r / (size * 0.28) + 0.3 * nz(vec3f(th, 0.5, 0.1)).r) - 0.5) * size * 0.28)) * step(r, size * 0.7) * step(size * 0.08, r);
        let web = max(spoke, ring * 0.7) * (1.0 - smoothstep(size * 0.6, size, r)) + (1.0 - smoothstep(0.0, size * 0.1, r)) * 0.6;
        rgb += web * (l.amb * 0.9 + l.sun * sh * (0.08 + 0.5 * toward));
        trans *= 1.0 - web * 0.7;
      }
    }
  }
  if (in_cabin) {
    // From inside, the glass replaces what is behind it: the view out, bent
    // by the acrylic and tinted by it (darker in a band along the top of the
    // windshield). Whatever lies in front of the glass is not bent.
    var tint = vec3f(0.95, 0.98, 0.965);
    if (wind && L.z < -0.95 && kind() == 0u) { tint *= mix(vec3f(1.0), vec3f(0.62, 0.72, 0.69), smoothstep(0.64, 0.77, L.y)); }
    // The 747's laminated, electrically heated glass: a faint green-blue, deeper toward the frame.
    if (kind() == 1u) { tint = mix(vec3f(0.93, 0.965, 0.955), vec3f(0.80, 0.88, 0.86), 1.0 - smoothstep(0.0, 0.05, -d)); }
    // The Hornet's stretched acrylic canopy: a faint warm tint.
    if (kind() == 2u) { tint = vec3f(0.95, 0.95, 0.91); }
    let edge = 1.0 - smoothstep(0.0, 0.08, -d);
    let curve = select(0.0, smoothstep(0.12, 0.45, abs(in.Nl.x)), wind);
    // (The 747's laminated glass is far flatter than formed acrylic: a faint waviness only.)
    let bend = select(glass_bend(g2, edge, curve), glass_bend(g2 * 0.6, 0.0, 0.0) * 0.18, kind() == 1u);
    let dw = (O.model * vec4f(tb * bend.x + bb * bend.y, 0.0)).xyz;
    var seen = textureLoad(t_scene, vec2i(in.clip.xy), 0).rgb;
    let cp = G.view_proj * vec4f(V + dw, 0.0);
    if (cp.w > 1e-4) {
      let uv = vec2f(0.5 + 0.5 * cp.x / cp.w, 0.5 - 0.5 * cp.y / cp.w);
      let q = vec2i(clamp(uv * G.screen.xy, vec2f(0.0), G.screen.xy - 1.0));
      if (textureLoad(t_gdepth, q, 0) <= in.clip.z) { seen = textureSampleLevel(t_scene, s_clamp, uv, 0.0).rgb; }
    }
    return vec4f((rgb + seen * trans * tint) * cov, cov);
  }
  let alpha = 1.0 - trans;
  return vec4f(rgb * cov, alpha * cov);
}

// ---------------------------------------------------------------- the 747 (aircraft/b747)
// Body frame: the origin 31.6 m behind the nose (d = L.z + 31.6), 5.0 m above the ground.

fn sd_box2(p: vec2f, b: vec4f) -> f32 {
  let c = vec2f((b.x + b.y) * 0.5, (b.z + b.w) * 0.5);
  let h = vec2f((b.y - b.x) * 0.5, (b.w - b.z) * 0.5);
  let q = abs(p - c) - h;
  return length(max(q, vec2f(0.0))) + min(max(q.x, q.y), 0.0);
}
fn sd_rbox(p: vec2f, h: vec2f, r: f32) -> f32 {
  let q = abs(p) - h + vec2f(r);
  return length(max(q, vec2f(0.0))) + min(max(q.x, q.y), 0.0) - r;
}
// The flight deck windows; matches b747::window.
fn window747(p: vec3f) -> f32 {
  let d = p.z + 31.6;
  if (d < 3.9 || d > 6.7 || p.y < 2.0) { return 1.0; }
  var best = 1.0;
  if (d < 5.45) {
    let ws = array<vec2f, 5>(vec2f(0.13, 3.18), vec2f(1.08, 2.94), vec2f(1.20, 3.05), vec2f(0.84, 3.49), vec2f(0.13, 3.81));
    best = max(sd_poly(vec2f(abs(p.x), p.y), ws, 5) - 0.06, d - 5.26);
  }
  let th = atan2(abs(p.x), p.y - 0.3);
  var w = array<vec4f, 2>(vec4f(5.30, 5.86, 0.30, 0.55), vec4f(5.98, 6.40, 0.36, 0.55));
  for (var i = 0; i < 2; i++) {
    let v = w[i];
    var q = vec2f(abs(d - (v.x + v.y) * 0.5) - (v.y - v.x) * 0.5, (abs(th - (v.z + v.w) * 0.5) - (v.w - v.z) * 0.5) * 3.4);
    q += vec2f(0.06);
    best = min(best, length(max(q, vec2f(0.0))) + min(max(q.x, q.y), 0.0) - 0.06);
  }
  return best;
}
// The gear wells; matches b747::well.
fn well747(p: vec3f) -> f32 {
  if (p.y > -1.2) { return 1.0; }
  let q = vec2f(p.z, abs(p.x));
  var d = sd_box2(q, vec4f(-27.35, -22.95, -0.78, 0.78));
  d = min(d, sd_box2(q, vec4f(-0.95, 0.45, 1.15, 4.0)));
  d = min(d, sd_box2(q, vec4f(0.15, 3.05, 1.05, 2.8)));
  d = min(d, sd_box2(q, vec4f(3.0, 5.75, 1.6, 2.24)));
  d = min(d, seg_d(q, vec2f(0.85, 5.75), vec2f(-0.30, 2.75)) - 0.36);
  return d + 0.005;
}
// Door outlines on the fuselage sides in (d, y): the five main deck doors
// each side, the upper deck exit, and the cargo doors on the right.
fn door747(L: vec3f) -> f32 {
  let d = L.z + 31.6;
  var best = 1e3;
  for (var i = 0; i < 5; i++) {
    best = min(best, sd_rbox(vec2f(d - DOORS747[i], L.y - 0.92), vec2f(0.595, 0.965), 0.16));
  }
  best = min(best, sd_rbox(vec2f(d - 15.2, L.y - 3.15), vec2f(0.42, 0.78), 0.14));
  if (L.x > 0.0) {
    best = min(best, sd_rbox(vec2f(d - 13.35, L.y + 1.75), vec2f(1.30, 0.86), 0.12));
    best = min(best, sd_rbox(vec2f(d - 45.2, L.y + 1.70), vec2f(1.30, 0.86), 0.12));
    best = min(best, sd_rbox(vec2f(d - 48.75, L.y + 1.45), vec2f(0.56, 0.58), 0.1));
  }
  return best;
}
// The nearest main deck door: (along, up from its centre, 1 inside it / 0).
fn door_local747(L: vec3f) -> vec3f {
  let d = L.z + 31.6;
  var best = vec2f(1e3);
  for (var i = 0; i < 5; i++) {
    let q = vec2f(d - DOORS747[i], L.y - 0.92);
    if (abs(q.x) < abs(best.x)) { best = q; }
  }
  let inside = abs(best.x) < 0.595 && abs(best.y) < 0.965 && abs(L.x) > 2.0;
  return vec3f(best, select(0.0, 1.0, inside));
}

// The finish of the 747's paint, after the livery: each skin panel's paint a
// shade apart, glazed windows, door seams and their markings, the radome's
// lightning strips, bare leading edges, walkways, grime and soot.
struct Fin { c: vec3f, rough: f32, metal: f32 };
fn skin747(f0: Fin, pt: u32, L: vec3f, Nl: vec3f, uv: vec2f, wear: f32, px: f32) -> Fin {
  var f = f0;
  let d = L.z + 31.6;
  let ax = abs(L.x);
  let side = sign(L.x);
  if (pt == P_FUSE) {
    // Panels between the butt joints and the lap joints, each painted a touch apart.
    let s = abs(uv.y);
    let pi = floor((d - 1.6) / 6.096);
    let pj = step(1.7, s) + step(3.6, s) + step(5.6, s) + step(7.7, s);
    let hp = hash2(vec2f(pi * 1.37 + side * 0.71, pj * 2.91));
    f.c *= 0.982 + 0.036 * hp;
    f.c = mix(f.c, f.c * vec3f(1.0, 0.992, 0.972), hash1(pi * 3.1 + pj * 5.7 + side) * 0.7);
    f.rough += 0.05 * (hash1(pi * 7.3 + pj * 1.9 + side * 2.0) - 0.5);
    // The radome: its own paint, greyer and duller, and the lightning diverter strips (dashed).
    if (d < 2.25) {
      let m = 0.30;
      let a = atan2(L.x, L.y - m);
      let k = (fract(a / 6.2831853 * 8.0 + 0.5) - 0.5) / 8.0 * 6.2831853;
      let r = length(vec2f(L.x, L.y - m));
      let dash = step(0.35, fract(d / 0.06)) * smoothstep(0.35, 0.55, d) * smoothstep(2.18, 2.05, d);
      let strip = aline(k * r, 0.012, px) * dash * step(abs(a), 2.6);
      f.c = mix(f.c * vec3f(0.93, 0.925, 0.90), vec3f(0.30, 0.30, 0.31), strip);
      f.rough = mix(f.rough + 0.12, 0.35, strip);
      f.metal = max(f.metal, strip * 0.8);
      f.c = mix(f.c, vec3f(0.62, 0.62, 0.60), aline(d - 2.2, 0.006, px));
    }
    // Glazing: smoky acrylic in a grey frame, smooth as glass.
    var ws = 1.0;
    if (ax > 1.3 && L.y > -0.2) { ws = cabin_win_sd747(L); }
    if (ws < 0.03) {
      let pane = afill(ws + 0.010, px);
      let frame = afill(ws, px) - pane;
      f.c = mix(f.c, vec3f(0.20, 0.205, 0.21), frame);
      f.c = mix(f.c, vec3f(0.012, 0.014, 0.017), pane);
      f.rough = mix(f.rough, 0.025, pane);
      f.metal *= 1.0 - afill(ws, px);
    }
    // Door gaps, and on each main door its handle flap with a red outline
    // and the OPEN arrow; the cargo doors' latch row.
    if (ax > 1.5) {
      let dd = door747(L);
      f.c *= 1.0 - 0.75 * aline(dd, 0.005, px);
      let dl = door_local747(L);
      if (dl.z > 0.0) {
        let q = dl.xy - vec2f(0.0, 0.08);
        let hd = sd_rbox(q, vec2f(0.16, 0.045), 0.02);
        f.c = mix(f.c, vec3f(0.06), afill(hd + 0.004, px) * 0.35);
        let red = aline(sd_rbox(q, vec2f(0.24, 0.10), 0.03), 0.008, px);
        let qa = vec2f(q.x * side, q.y + 0.17);
        let arrow = afill(max(abs(qa.y) - 0.012, abs(qa.x) - 0.14), px) + afill(max(abs(qa.y) - 0.05 + (qa.x - 0.10) * 0.9, -(qa.x - 0.10)), px);
        f.c = mix(f.c, vec3f(0.55, 0.035, 0.025), sat(red + arrow * 0.9));
      }
      if (L.x > 0.0 && L.y < -0.7) {
        for (var i = 0; i < 2; i++) {
          let c = select(45.2, 13.35, i == 0);
          let qx = d - c;
          if (abs(qx) < 1.2) {
            let lp = (fract(qx / 0.4 + 0.5) - 0.5) * 0.4;
            f.c *= 1.0 - 0.5 * afill(sd_rbox(vec2f(lp, L.y + 2.50), vec2f(0.05, 0.025), 0.01), px);
          }
        }
      }
    }
    // The air conditioning packs' ram air inlets in the belly fairing ahead of
    // the wing (a ramp widening aft into a dark mouth), and their exhaust louvres behind.
    if (L.y < -1.5 && d > 21.5 && d < 29.0) {
      let t = (d - 21.8) / 1.7;
      let w = 0.06 + 0.16 * sat(t);
      let q = vec2f(d - 22.65, ax - 1.25);
      let ramp = max(abs(q.x) - 0.85, abs(q.y) - w);
      let inside = afill(ramp, px);
      f.c = mix(f.c, f.c * mix(0.75, 0.05, smoothstep(0.4, 1.0, t)), inside);
      f.c *= 1.0 - 0.5 * aline(ramp, 0.004, px);
      f.rough = mix(f.rough, 0.6, inside * smoothstep(0.6, 1.0, t));
      let lq = vec2f(d - 27.6, ax - 1.15);
      let louv = sd_rbox(lq, vec2f(0.45, 0.22), 0.03);
      let slat = 0.5 + 0.5 * cos((d - 27.6) / 0.06 * 3.14159);
      f.c = mix(f.c, f.c * (0.25 + 0.5 * slat), afill(louv, px) * (1.0 - smoothstep(0.02, 0.05, px)) + afill(louv, px) * smoothstep(0.02, 0.05, px) * 0.5);
      f.c *= 1.0 - 0.5 * aline(louv, 0.004, px);
    }
    // Static ports on the nose: polished discs in a red no-paint ring.
    if (ax > 1.5 && d > 7.6 && d < 8.6 && abs(L.y + 0.35) < 0.3) {
      let q = vec2f(d - 8.1, L.y + 0.35);
      let sp = length(vec2f(abs(q.x) - 0.12, q.y));
      f.c = mix(f.c, vec3f(0.55, 0.035, 0.025), aline(sd_rbox(q, vec2f(0.24, 0.10), 0.04), 0.008, px));
      let disc = afill(sp - 0.035, px);
      f.c = mix(f.c, vec3f(0.75, 0.76, 0.78), disc);
      f.metal = max(f.metal, disc);
      f.rough = mix(f.rough, 0.15, disc);
    }
  }
  if (pt == P_WING || pt == P_CTRL || (pt == P_TAIL && ax > 0.9)) {
    let s = abs(uv.x);
    let av = abs(uv.y);
    let upper = uv.y > 0.0;
    // Skin planks running spanwise, each a slightly different grey.
    let hp = hash2(vec2f(floor(av / 0.62), floor(s / 6.2) + side * 0.5 + f32(pt) * 3.0));
    f.c *= 0.955 + 0.09 * hp;
    f.rough += 0.06 * (hp - 0.5);
    // Grime drawn back along the chord by the airflow; heavier underneath and
    // on the flaps, sooty behind the engines.
    let g = nz(vec3f(L.x * 2.2, L.y * 2.2, L.z * 0.10)).r;
    let gg = nz(vec3f(L.x * 9.0, L.y * 9.0, L.z * 0.4)).g;
    let dirt = smoothstep(0.35, 0.85, g) * (0.6 + 0.5 * gg) * select(0.16, 0.08, upper);
    f.c *= 1.0 - dirt;
    f.rough += dirt * 0.6;
    if (!upper && pt != P_TAIL && L.z < 12.0) {
      let e = eng747(L);
      let soot = exp(-e.x * e.x / 1.4) * smoothstep(5.0, 7.5, e.z) * (0.5 + 0.5 * gg);
      f.c = mix(f.c, vec3f(0.06, 0.055, 0.05), sat(soot * 0.35));
    }
    // Bare, polished leading edges (the wing's and the stabiliser's).
    if (pt != P_CTRL) {
      let bare = smoothstep(0.08, 0.14, wear) * select(1.0, 0.0, pt == P_TAIL && ax > 30.0);
      let polish = 0.85 + 0.15 * nz(vec3f(L.x * 0.8, L.y * 30.0, L.z * 30.0)).r;
      f.c = mix(f.c, vec3f(0.58, 0.59, 0.61) * polish * (1.0 - 0.25 * dirt), bare);
      f.metal = mix(f.metal, 1.0, bare);
      f.rough = mix(f.rough, 0.38 + 0.12 * gg, bare);
    }
    // The walkway over the wing root, outlined in black, and the no-step
    // lines along the flaps' and spoilers' front edges.
    if (upper && pt == P_WING && s < 13.0 && L.z < 12.0) {
      let c = chord747(s);
      let inr = step(3.9, s) * step(s, 11.6);
      let ina = step(0.17 * c, av) * step(av, 0.56 * c);
      let w = (aline(s - 3.9, 0.02, px) + aline(s - 11.6, 0.02, px)) * ina + (aline(av - 0.17 * c, 0.02, px) + aline(av - 0.56 * c, 0.02, px)) * inr;
      f.c = mix(f.c, vec3f(0.02), sat(w) * 0.85);
      // A non-slip finish inside it.
      f.rough = mix(f.rough, 0.7, inr * ina);
      f.c *= 1.0 - 0.06 * inr * ina;
    }
    if (upper && pt == P_WING && s > 3.4 && s < 24.0 && L.z < 12.0) {
      f.c = mix(f.c, vec3f(0.02), aline(av - 0.575 * chord747(s), 0.012, px) * 0.8);
    }
  }
  if (pt == P_NACELLE) {
    let e = eng747(L);
    let r = length(e.xy);
    if (r < 1.34) {
      // Oil and exhaust grime along the bottom of the cowls, soot at the sleeve's end.
      let bot = sat(-e.y / r);
      let st = nz(vec3f(L.x * 6.0, L.y * 6.0, L.z * 0.25)).r;
      f.c *= 1.0 - 0.18 * bot * bot * smoothstep(0.3, 0.8, st) * smoothstep(0.5, 2.5, e.z);
      f.c *= 1.0 - 0.25 * smoothstep(2.55, 2.95, e.z);
      // Each cowl panel a shade apart.
      f.c *= 0.975 + 0.04 * hash1(step(0.95, e.z) + step(1.9, e.z) * 2.0 + side * 5.0 + e.w * 9.0);
    } else {
      // The pylon: a little grimier toward its aft end.
      f.c *= 1.0 - 0.10 * smoothstep(3.0, 7.0, e.z) * smoothstep(0.3, 0.8, nz(L * vec3f(3.0, 3.0, 0.3)).r);
    }
  }
  if (pt == P_SPINNER) {
    // A dark spinner with a white swirl, so ground crew can see the fan turn.
    let e = eng747(L);
    let r = length(e.xy);
    let a = atan2(e.x, e.y) * side;
    let k = wrapa(a - r * 9.0);
    let band = aline(k * max(r, 0.02), 0.03, px) * smoothstep(0.03, 0.06, r) * step(e.z, 1.13);
    f.c = mix(vec3f(0.035, 0.036, 0.04), vec3f(0.80), band);
    f.rough = 0.22;
    f.metal = 0.0;
  }
  return f;
}

// Cabin windows on the skin (there is no cabin behind them): 10 x 15 in,
// every 20 in along the main deck, and along the upper deck; the small
// window in each door. Signed distance (m) to the nearest pane.
const DOORS747 = array<f32, 5>(9.50, 18.80, 30.61, 40.74, 55.14);
fn cabin_win_sd747(L: vec3f) -> f32 {
  let d = L.z + 31.6;
  var sd = 1.0;
  if (abs(L.x) > 2.3 && abs(L.y - 0.69) < 0.3 && d > 2.4 && d < 57.8) {
    var near_door = false;
    let k = (fract((d - 2.4) / 0.508) - 0.5) * 0.508;
    // (Whole windows are left out beside the doors, not cut through.)
    for (var i = 0; i < 5; i++) { if (abs(d - k - DOORS747[i]) < 0.9) { near_door = true; } }
    if (!near_door) {
      sd = sd_rbox(vec2f(k, L.y - 0.69), vec2f(0.127, 0.19), 0.07);
    }
  }
  if (abs(L.x) > 1.3 && abs(L.y - 3.42) < 0.3 && d > 9.6 && d < 22.3) {
    let k = (fract((d - 9.6) / 0.508) - 0.5) * 0.508;
    if (abs(d - k - 15.2) > 0.6) { sd = min(sd, sd_rbox(vec2f(k, L.y - 3.42), vec2f(0.127, 0.18), 0.07)); }
  }
  if (abs(L.x) > 2.3 && abs(L.y - 1.32) < 0.2) {
    for (var i = 0; i < 5; i++) { sd = min(sd, sd_rbox(vec2f(d - DOORS747[i], L.y - 1.32), vec2f(0.11, 0.15), 0.06)); }
  }
  if (abs(L.x) > 1.3 && abs(L.y - 3.42) < 0.3) { sd = min(sd, sd_rbox(vec2f(d - 15.2, L.y - 3.42), vec2f(0.11, 0.15), 0.06)); }
  return sd;
}
fn cabin_windows747(L: vec3f, aa: f32) -> f32 { return smoothstep(aa, -aa, cabin_win_sd747(L)); }

// A line |d| < hw wide, filtered over a footprint px: it fades to its average
// coverage once it is thinner than a pixel instead of vanishing or aliasing.
fn aline(d: f32, hw: f32, px: f32) -> f32 {
  let w = max(hw, px * 0.5);
  return sat((w - abs(d)) / max(px, 1e-5) + 0.5) * (hw / w);
}
// Filled shape (sd < 0) filtered the same way.
fn afill(sd: f32, px: f32) -> f32 { return sat(0.5 - sd / max(px, 1e-5)); }

// The nearest engine: (x outboard of its axis, y above it, z behind its inlet lip, inboard 1/outboard 0).
fn eng747(L: vec3f) -> vec4f {
  let ib = abs(L.x) < 16.3;
  let ex = select(21.0, 11.68, ib);
  return vec4f(L.x - sign(L.x) * ex, L.y - select(-1.58, -2.5, ib), L.z - select(0.81, -8.36, ib), select(0.0, 1.0, ib));
}

const B_BELLY = vec3f(0.46, 0.48, 0.51);
const B_GREY = vec3f(0.34, 0.36, 0.39);
const B_NAVY = vec3f(0.012, 0.03, 0.095);
const B_RED = vec3f(0.42, 0.025, 0.02);

// The 747's paint: white over a grey belly, a navy cheatline with a red
// pinstripe sweeping up the tail, grey wings, a navy fin with a white emblem.
fn livery747(L: vec3f, aa: f32, pt: u32) -> vec3f {
  let d = L.z + 31.6;
  let ax = abs(L.x);
  let fin = ax < 0.9 && L.y > 3.9 && d > 52.0;
  if (fin) {
    var c = B_NAVY;
    // The emblem: a white ring with three swept bands inside, a red ring outside it.
    let q = vec2f(d - 64.2, L.y - 3.6);
    let r = length(q);
    let ring = smoothstep(aa, -aa, abs(r - 1.68) - 0.09);
    let red = smoothstep(aa, -aa, abs(r - 1.92) - 0.035);
    let band = abs(fract((q.x * 0.55 + q.y) / 0.95 + 0.25) - 0.5) - 0.13;
    let inner = smoothstep(aa, -aa, band) * smoothstep(1.5 + aa, 1.5 - aa, r);
    c = mix(c, WHITE_PAINT, max(ring, inner));
    c = mix(c, B_RED, red);
    return c;
  }
  if (ax > 30.5 && (pt == P_TAIL || L.y > 1.6)) { return B_NAVY; }
  if (pt == P_WING || pt == P_CTRL || pt == P_TAIL) { return B_GREY; }
  if (pt == P_NACELLE) { return WHITE_PAINT; }
  // Unpatterned painted parts: the flap track fairings under the wings are
  // wing grey, the gear doors belly grey.
  if (pt == 0u) { return select(B_GREY, B_BELLY, ax < 6.0); }
  var c = WHITE_PAINT;
  // Grey belly below a line that rises with the tail's upsweep.
  let yb = -1.15 + max(d - 46.0, 0.0) * 0.17 - max(4.0 - d, 0.0) * 0.12;
  c = mix(c, B_BELLY, smoothstep(yb + aa, yb - aa, L.y));
  // The cheatline under the windows, sweeping up over the tail cone.
  if (d > 1.4 && d < 68.5 && ax > 0.2) {
    let rise = max(d - 50.0, 0.0) * 0.16;
    let taper = smoothstep(1.4, 4.0, d) * (1.0 - smoothstep(64.0, 68.5, d));
    let y0 = 0.17 + rise;
    let navy = smoothstep(aa, -aa, abs(L.y - y0) - 0.13 * taper);
    let pin = smoothstep(aa, -aa, abs(L.y - y0 - 0.215 * taper) - 0.022 * taper);
    c = mix(c, B_NAVY, navy);
    c = mix(c, B_RED, pin);
  }
  // Registration on the rear fuselage, reading nose-first on the left.
  if (ax > 0.5 && d > 54.5 && d < 60.5 && L.y > 1.3 && L.y < 2.1) {
    let reg = pack(10u, 7u, 4u, 7u, 12u, 11u);
    let ink = label_code(vec2f((57.5 - d) * sign(L.x), L.y), vec2f(0.0, 1.7), reg, 6u, 0.5, aa);
    c = mix(c, B_NAVY, ink);
  }
  // The cabin windows.
  c = mix(c, vec3f(0.018, 0.02, 0.025), cabin_windows747(L, aa));
  return c;
}

fn weather747(L: vec3f, N: vec3f, base: vec3f, belly: f32, n: vec4f) -> vec4f {
  var c = base;
  // Streaks of grime back along the belly from the drains, and a darker
  // patch behind the APU exhaust.
  let streak = nz(vec3f(L.x * 1.5, L.y * 1.5, L.z * 0.06)).r;
  c *= 1.0 - 0.08 * belly * smoothstep(0.4, 0.8, streak);
  let d = L.z + 31.6;
  let fine = nz(vec3f(L.x * 8.0, L.y * 8.0, L.z * 0.3)).g;
  // Hydraulic fluid and brake dust swept back from the gear wells, brownish.
  if (L.y < -0.6 && L.z > -1.5 && L.z < 24.0) {
    let k = smoothstep(-1.5, 2.0, L.z) * (1.0 - smoothstep(10.0, 24.0, L.z)) * sat(-L.y - 0.6) * smoothstep(0.35, 0.75, streak) * (0.5 + 0.7 * fine);
    c = mix(c, c * vec3f(0.62, 0.55, 0.48), sat(k * 0.55));
  }
  // Drain masts under the belly: a stain trailing from each.
  for (var i = 0; i < 2; i++) {
    let z0 = select(15.4, -15.6, i == 0);
    let t = L.z - z0;
    if (t > 0.0 && L.y < -2.0) {
      let k = exp(-L.x * L.x / (0.02 + t * 0.012)) * exp(-t / 5.0) * (0.5 + 0.6 * fine);
      c = mix(c, c * vec3f(0.55, 0.50, 0.44), sat(k * 0.7));
    }
  }
  // Rain has run down from the window frames and door seams on the sides.
  if (abs(L.x) > 2.0 && L.y > -1.2 && L.y < 3.8) {
    let run = nz(vec3f(L.z * 3.0, L.y * 0.12, L.x * 0.1)).r;
    let below = select(0.0, 1.0, L.y < 0.5) * smoothstep(-1.2, 0.3, L.y);
    c *= 1.0 - 0.035 * smoothstep(0.55, 0.85, run) * (0.4 + 0.6 * below);
  }
  // The wing-to-body fairing's lower edge and the keel beam: scuffed, darker.
  if (L.y < -2.6 && abs(L.z) < 18.0) { c *= 1.0 - 0.06 * smoothstep(0.5, 0.8, nz(L * 1.3).b); }
  if (d > 68.0) { c = mix(c, vec3f(0.03), smoothstep(68.6, 69.8, d) * 0.7); }
  return vec4f(c, 0.08 * belly * n.b);
}

// Chord of the wing at a span station (plan view).
fn chord747(x: f32) -> f32 {
  let le = select(37.086 + 0.841 * (x - 21.0), 18.06 + 0.906 * x, x < 21.0);
  let te = select(39.151 + 0.58 * (x - 13.4), 34.93 + 0.315 * x, x < 13.4);
  return te - le;
}

// The 747's panel seams and rivet rows (height, seam dirt, rivets).
fn relief747(pt: u32, uv: vec2f, L: vec3f, Nl: vec3f, px: f32) -> vec3f {
  var h = 0.0;
  var dirt = 0.0;
  var rv = 0.0;
  let gw = max(0.0012, px * 0.9);
  let ga = 0.0007 * min(1.0, 0.0018 / gw);
  let rr = 0.004;
  let rf = 1.0 - smoothstep(rr * 0.5, rr * 1.6, px);
  switch pt {
    case 1u: {
      let d = L.z + 31.6;
      let s = abs(uv.y);
      // A rivet row on every frame (20 in apart), a butt joint every twelfth.
      let fr = (fract(d / 0.508 + 0.5) - 0.5) * 0.508;
      rv = max(rv, rivets(s, abs(fr), 0.04, rr) * rf * 0.6);
      let bj = (fract((d - 1.6) / 6.096 + 0.5) - 0.5) * 6.096;
      h -= ga * line(bj, gw);
      dirt += line(bj, gw * 2.5);
      var laps = array<f32, 4>(1.7, 3.6, 5.6, 7.7);
      for (var i = 0; i < 4; i++) {
        let ds = s - laps[i];
        h -= ga * line(ds, gw);
        dirt += line(ds, gw * 2.0);
        rv = max(rv, rivets(d, abs(abs(ds) - 0.014), 0.035, rr) * rf);
      }
      if (abs(L.x) > 1.5) {
        let dd = door747(L);
        h -= ga * 2.0 * line(dd, gw * 1.3);
        dirt += line(dd, gw * 3.0);
        // The panes sit a little in from the skin, the edge of the opening bevelled.
        let ws = cabin_win_sd747(L);
        h -= 0.006 * smoothstep(0.004, -0.010, ws) * (1.0 - smoothstep(0.004, 0.012, px));
        dirt += 0.5 * line(ws - 0.002, gw * 1.5);
        // Each door's flush handle flap, and the hinged access panels on the belly.
        let dl = door_local747(L);
        if (dl.z > 0.0) {
          let hd = sd_rbox(dl.xy - vec2f(0.0, 0.08), vec2f(0.16, 0.045), 0.02);
          h -= ga * 1.5 * line(hd, gw);
          dirt += line(hd, gw * 2.0);
        }
      }
      if (L.y < -1.5) {
        let pz = (fract(L.z / 1.524 + 0.5) - 0.5) * 1.524;
        let bp = sd_rbox(vec2f(pz, abs(L.x) - 0.9), vec2f(0.28, 0.20), 0.03);
        let keep = step(0.55, hash1(floor(L.z / 1.524) + sign(L.x) * 0.37));
        h -= ga * line(bp, gw) * keep;
        dirt += line(bp, gw * 2.0) * keep;
        rv = max(rv, rivets(pz + abs(L.x) * 3.0, abs(bp + 0.012), 0.04, rr) * rf * keep);
      }
      h += rv * 0.0008;
    }
    case 2u, 5u: {
      let s = abs(uv.x);
      let av = abs(uv.y);
      let c = select(chord747(s), 6.0, pt == 5u);
      rv = max(rv, rivets(av, (fract(s / 0.6 + 0.5) - 0.5) * 0.6, 0.045, rr) * rf);
      rv = max(rv, rivets(s, av - 0.15 * c, 0.04, rr) * rf);
      rv = max(rv, rivets(s, av - 0.62 * c, 0.04, rr) * rf);
      let le = av - 0.12 * c;
      h -= ga * line(le, gw);
      dirt += line(le, gw * 2.0);
      h += rv * 0.0008;
    }
    case 3u: {
      let dd = (fract(abs(uv.x) / 0.5) - 0.5) * 0.5;
      h -= ga * 0.7 * line(dd, gw);
      dirt += 0.3 * line(dd, gw * 2.0);
    }
    case 7u: { // the flight deck lining: panel joints, a fine grain
      let d = L.z + 31.6;
      let j = (fract((d - 4.6) / 0.62 + 0.5) - 0.5) * 0.62;
      h -= 0.0012 * line(j, max(0.002, px));
      dirt += 0.4 * line(j, max(0.004, px * 2.0));
      h += 0.00012 * nz(L * 50.0).r * (1.0 - smoothstep(0.0005, 0.002, px));
    }
    case 8u: {
      if (abs(Nl.x) < 0.7) {
        let t = fract(L.x / 0.065 + 0.5) * 2.0 - 1.0;
        h += 0.004 * sqrt(max(1.0 - t * t, 0.0));
        dirt += smoothstep(0.85, 1.0, abs(t));
      }
      h += 0.0004 * nz(L * 60.0).g * (1.0 - smoothstep(0.0005, 0.002, px));
    }
    case 9u: { h += 0.0015 * nz(L * 45.0).r * (1.0 - smoothstep(0.001, 0.004, px)); }
    case 10u: { h += 0.00012 * nz(L * 220.0).r * (1.0 - smoothstep(0.0002, 0.0009, px)); }
    case 12u: { h += 0.00025 * sin((L.x + L.y * 1.3 + L.z * 0.7) * 2600.0) * (1.0 - smoothstep(0.0002, 0.0008, px)); }
    case 15u: { h += 0.003 * nz(L * 70.0).r * (1.0 - smoothstep(0.001, 0.004, px)); }
    case 13u: {
      // Inlet cowl / fan cowl / reverser joints round the nacelle, and the
      // fan cowl halves' hinge line on top and latch line underneath.
      let e = eng747(L);
      let r = length(e.xy);
      if (r < 1.34) {
        let a = atan2(e.x, e.y);
        var cj = min(abs(e.z - 0.95), abs(e.z - 1.90));
        h -= ga * 1.4 * line(cj, gw);
        dirt += line(cj, gw * 2.5);
        let ls = min(abs(a), 3.14159 - abs(a)) * r;
        let on = step(0.95, e.z) * step(e.z, 2.95);
        h -= ga * line(ls, gw) * on;
        dirt += line(ls, gw * 2.0) * on;
        rv = max(rv, rivets(e.z, abs(cj - 0.016), 0.03, rr) * rf);
        rv = max(rv, rivets(r * a, abs(e.z - 0.25), 0.035, rr) * rf);
        h += rv * 0.0008;
      } else {
        let sj = uv.y - 0.25;
        h -= ga * line(sj, gw);
        dirt += line(sj, gw * 2.5);
      }
    }
    case 6u: {
      let ax = abs(L.x);
      var cs = array<f32, 5>(0.44, 1.36, 2.48, 4.94, 6.06);
      var c = cs[0];
      for (var i = 1; i < 5; i++) { if (abs(ax - cs[i]) < abs(ax - c)) { c = cs[i]; } }
      if (abs(Nl.x) < 0.6) {
        let o = abs(ax - c);
        let gd = min(abs(o - 0.07), abs(o - 0.16));
        h -= 0.003 * line(gd, max(0.005, px));
        dirt += line(gd, max(0.006, px));
      }
    }
    default: {}
  }
  return vec3f(h, sat(dirt) * min(1.0, 0.003 / gw), rv);
}
