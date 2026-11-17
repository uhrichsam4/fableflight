// offset: xyz tile centre relative to the camera, w = grid cell size (m).
// info: x = zoom. tess: the block of cells drawn tessellated (x0, y0, w, h).
struct Tile { offset: vec4f, info: vec4f, tess: vec4f };
@group(1) @binding(0) var<uniform> T: Tile;
@group(1) @binding(1) var t_img: texture_2d<f32>;
// R water coverage, G water class * 80, B ground class | seed << 4, A field direction.
@group(1) @binding(2) var t_mask: texture_2d<f32>;
// The tile's own vertex buffer (10 floats per vertex), read by the tessellator.
@group(1) @binding(3) var<storage, read> TV: array<f32>;

struct VOut {
  @builtin(position) clip: vec4f,
  @location(0) P: vec3f,
  @location(1) N: vec3f,
  @location(2) uv: vec2f,
  @location(3) depth: f32,
};
struct GOut { @location(0) c: vec4f, @location(1) n: vec4f, @location(2) amb: f32 };

const GRID = 64u;
const GV = 65u;

// ------------------------------------------------------------ ground classes
const GC_WOOD = 1u;
const GC_SCRUB = 2u;
const GC_GRASS = 3u;
const GC_FARM = 4u;
const GC_WET = 5u;
const GC_SAND = 6u;
const GC_ROCK = 7u;
const GC_PAVED = 8u;
const GC_PITCH = 9u;
// Material layers (detail.wgsl).
const L_GRASS = 0;
const L_SOIL = 1;
const L_ROCK = 2;
const L_SAND = 3;
const L_ASPH = 4;
const L_LITTER = 5;
const L_CEREAL = 6;
const L_MOSS = 7;
// Mid-scale layers (aerial photo scans, blank until loaded).
const M_GRASS = 8;
const M_GROUND = 9;
const M_ROCK = 10;
const M_SAND = 11;
const M_PAVED = 12;
// Ground detail is faded out by this distance (m).
const DET_FAR = 1000.0;
const TAU = 6.28318530718;

// Per layer (G.det, set by the renderer): metres per repeat, metres of relief,
// z = for procedural layers the scale of the second, larger (or, at 1, just
// rotated) sample that hides the repeat, for photo scans (materials.rs) how
// much of their colour shifts (rather than brightness) to keep, and w = for
// photo scans how much of their variation to keep (0 = procedural layer).
fn det_size(l: i32) -> f32 { return G.det[l].x; }
fn det_range(l: i32) -> f32 { return G.det[l].y; }
fn det_photo(l: i32) -> f32 { return G.det[l].w; }
fn det_macro(l: i32) -> f32 { return select(G.det[l].z, 1.0, det_photo(l) > 0.0); }
fn det_chroma(l: i32) -> f32 { return G.det[l].z; }
const ROT = mat2x2f(0.8, 0.6, -0.6, 0.8);

// ------------------------------------------------------------ hex tiling
// Photo scans would repeat visibly, so they are hex-tiled (after Mikkelsen,
// "Practical Real-Time Hex-Tiling", JCGT 2022): the plane is cut into
// hexagons about half a texture across, each showing the texture under its own
// random rotation and offset. Three samples are blended across the borders,
// with weights sharpened by height so the seams run along the relief rather
// than along a lattice.
struct Hex { w: vec3f, s0: vec2f, s1: vec2f, s2: vec2f, r0: mat2x2f, r1: mat2x2f, r2: mat2x2f };

fn hx_pcg(v: u32) -> u32 {
  let s = v * 747796405u + 2891336453u;
  let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (w >> 22u) ^ w;
}
fn hex_rot(v: vec2i) -> mat2x2f {
  let a = f32(hx_pcg(bitcast<u32>(v.x) ^ hx_pcg(bitcast<u32>(v.y) + 0x9e3779b9u)) >> 8u) * (TAU / 16777216.0);
  let c = cos(a);
  let s = sin(a);
  return mat2x2f(c, s, -s, c);
}
fn hex_off(v: vec2i) -> vec2f {
  let h = hx_pcg(bitcast<u32>(v.y) ^ hx_pcg(bitcast<u32>(v.x) + 0x7f4a7c15u));
  return vec2f(f32(h & 0xffffu), f32(h >> 16u)) / 65536.0;
}
// Where hexagon centre v shows st (texture repeats).
fn hex_st(v: vec2i, st: vec2f, r: mat2x2f) -> vec2f {
  let cen = vec2f(f32(v.x) + 0.5 * f32(v.y), f32(v.y) * 0.8660254) / 3.4641016;
  return r * (st - cen) + cen + hex_off(v);
}
fn hex(st: vec2f) -> Hex {
  let p = st * 3.4641016;
  let sk = vec2f(p.x - 0.57735027 * p.y, 1.15470054 * p.y);
  let base = vec2i(floor(sk));
  let f = fract(sk);
  let z = 1.0 - f.x - f.y;
  let up = z < 0.0;
  let s1 = select(0.0, 1.0, up);
  let s2 = 2.0 * s1 - 1.0;
  let i = select(0, 1, up);
  let v0 = base + vec2i(i, i);
  let v1 = base + vec2i(i, 1 - i);
  let v2 = base + vec2i(1 - i, i);
  let r0 = hex_rot(v0);
  let r1 = hex_rot(v1);
  let r2 = hex_rot(v2);
  return Hex(vec3f(-z * s2, s1 - f.y * s2, s1 - f.x * s2), hex_st(v0, st, r0), hex_st(v1, st, r1), hex_st(v2, st, r2), r0, r1, r2);
}
// Sharpened shares of the three hexagons before height; a hexagon whose
// share is negligible is not sampled at all (over most of the plane only one
// is), so its share is zeroed.
fn hex_p(w: vec3f) -> vec3f {
  let k = pow(w, vec3f(7.0));
  let p = k / (k.x + k.y + k.z);
  return select(vec3f(0.0), p, p > vec3f(0.004));
}
// Final blend weights from those shares and the three heights.
fn hex_w(p: vec3f, h: vec3f) -> vec3f {
  let k = p * (0.25 + h);
  return k / max(k.x + k.y + k.z, 1e-6);
}
// Height only, for parallax (st and its derivatives in texture repeats).
fn hex_h(l: i32, st: vec2f, dx: vec2f, dy: vec2f) -> f32 {
  let x = hex(st);
  let p = hex_p(x.w);
  var h = vec3f(0.0);
  if (p.x > 0.0) { h.x = textureSampleGrad(t_dalb, s_detail, x.s0, l, x.r0 * dx, x.r0 * dy).a; }
  if (p.y > 0.0) { h.y = textureSampleGrad(t_dalb, s_detail, x.s1, l, x.r1 * dx, x.r1 * dy).a; }
  if (p.z > 0.0) { h.z = textureSampleGrad(t_dalb, s_detail, x.s2, l, x.r2 * dx, x.r2 * dy).a; }
  return dot(hex_w(p, h), h);
}
fn hex_h_lod(l: i32, st: vec2f, lod: f32) -> f32 {
  let x = hex(st);
  let p = hex_p(x.w);
  var h = vec3f(0.0);
  if (p.x > 0.0) { h.x = textureSampleLevel(t_dalb, s_detail, x.s0, l, lod).a; }
  if (p.y > 0.0) { h.y = textureSampleLevel(t_dalb, s_detail, x.s1, l, lod).a; }
  if (p.z > 0.0) { h.z = textureSampleLevel(t_dalb, s_detail, x.s2, l, lod).a; }
  return dot(hex_w(p, h), h);
}
// A layer's height at uv (m) in a vertex shader, filtered for spacing sp (m):
// the large copy for procedural layers, hex-tiled for photo scans.
fn layer_h_lod(l: i32, uv: vec2f, sp: f32) -> f32 {
  let s = det_size(l) * det_macro(l);
  let lod = max(log2(sp * f32(textureDimensions(t_dalb).x) / s), 0.0);
  if (det_photo(l) > 0.0) { return hex_h_lod(l, uv / s, lod); }
  return textureSampleLevel(t_dalb, s_detail, ROT * uv / s + vec2f(0.37, 0.71), l, lod).a;
}

// Low-frequency noise (period s metres), filtered for a footprint of fw metres.
fn wn(wc: vec2f, s: f32, fw: f32) -> f32 {
  return textureSampleLevel(t_wave, s_repeat, wc / s, max(log2(fw * 256.0 / s), 0.0)).a;
}

struct Ground {
  la: i32,          // main layer
  lb: i32,          // second layer
  wb: f32,          // share of the second layer (before height blending)
  rock: f32,        // share of rock (triplanar on steep ground)
  amp: f32,         // relief scale: 0 where nothing may stick up (roads)
  soil: vec3f,      // the second layer's own colour, where it is not the photo's
  soil_abs: f32,    // how much of that colour to use
  plant: vec3f,     // the main layer's colour (the photo, or what is left of it once the soil is taken out)
  hrow: f32,        // crop row / furrow / wheel track relief (m)
  grow: vec2f,      // its gradient (east, north)
  tint: vec3f,
  grassy: f32,      // blades: gaps close up and it gets a sheen at grazing angles
  mid: i32,         // mid-scale layer
  midk: f32,        // and how strongly it shows
};

fn ground_mask(uv: vec2f, wc: vec2f, fw: f32) -> vec4f {
  // Nearest texel, looked up through a wobble so class edges are ragged
  // rather than a staircase of texels.
  let jit = vec2f(wn(wc, 3.1, fw), wn(wc + 1.7, 4.3, fw)) - 0.5;
  let dim = vec2f(textureDimensions(t_mask, 0));
  let tx = vec2i(clamp(floor(uv * dim + jit * 2.5), vec2f(0.0), dim - 1.0));
  return textureLoad(t_mask, tx, 0);
}

// Wheel tracks every 20-24 m across a cereal field.
fn tramlines(g: ptr<function, Ground>, across: f32, seed: u32, fa: f32) {
  let per = 20.0 + 4.0 * f32(seed & 1u);
  let local = (fract(across / per + f32(seed) * 0.13) - 0.5) * per;
  let off = abs(abs(local) - 0.9);
  let w = (1.0 - smoothstep(0.18, 0.3 + fa, off)) * (1.0 - smoothstep(1.5, 4.0, fa));
  (*g).wb = max((*g).wb, w);
  (*g).hrow -= 0.05 * w;
  (*g).soil_abs = max((*g).soil_abs, w);
}

fn ground(uv: vec2f, wc: vec2f, img: vec3f, slope: f32, dx: vec2f, dy: vec2f, V: vec3f) -> Ground {
  let fw = max(length(dx), length(dy));
  var g = Ground(L_GRASS, L_SOIL, 0.0, 0.0, 1.0, vec3f(0.0), 0.0, img, 0.0, vec2f(0.0), vec3f(1.0), 0.0, M_GRASS, 1.0);
  let n1 = wn(wc, 23.0, fw);
  let n2 = wn(wc + 7.3, 7.0, fw);
  let m = ground_mask(uv, wc, fw);
  let b8 = u32(round(m.b * 255.0));
  let cls = b8 & 15u;
  let seed = b8 >> 4u;
  let ang = m.a * (255.0 / 256.0) * PI;
  let dir = vec2f(cos(ang), sin(ang));
  let perp = vec2f(-dir.y, dir.x);
  let across = dot(wc, perp);
  // Footprint of a pixel across the rows.
  let fa = abs(dot(dx, perp)) + abs(dot(dy, perp));
  let sum = img.r + img.g + img.b + 0.02;
  let gex = (2.0 * img.g - img.r - img.b) / sum;
  let yel = (img.r + img.g - 2.0 * img.b) / sum;
  let lum = sum / 3.0;
  let veg = clamp(gex * 9.0 - 0.1, 0.0, 1.0) * (1.0 - smoothstep(0.35, 0.6, lum));
  // Bare earth as it would look here: brown, about as bright as the photo.
  g.soil = vec3f(0.36, 0.28, 0.20) * clamp(lum / 0.26, 0.45, 1.6) * 0.9;
  var rock_thr = 0.2;
  switch cls {
    case GC_WOOD: {
      g.la = L_LITTER; g.lb = L_MOSS;
      g.wb = smoothstep(0.5, 0.75, n1);
      rock_thr = 0.28;
      g.mid = M_GROUND; g.midk = 0.6;
    }
    case GC_SCRUB: {
      g.wb = 0.35 + 0.5 * (n2 - 0.5);
      g.soil_abs = 0.6; g.grassy = 0.7;
      g.tint = vec3f(1.05, 1.0, 0.9);
    }
    case GC_GRASS, GC_PITCH: {
      g.wb = smoothstep(0.62, 0.85, n1) * 0.45 + (1.0 - veg) * 0.25;
      g.soil_abs = 0.7; g.grassy = 1.0;
      if (cls == GC_PITCH) {
        g.wb = 0.04;
        g.midk = 0.3;
        // Mowing stripes: the grass is laid toward or away from the viewer.
        let stripe = select(-1.0, 1.0, fract(across / 10.0) < 0.5);
        let hv = normalize(vec2f(dot(V, G.east.xyz), dot(V, G.north.xyz)) + 1e-4);
        let k = 1.0 - smoothstep(1.5, 4.0, fa);
        // Only on the turf, not on tracks or paths inside the pitch.
        g.tint = vec3f(1.0 + 0.11 * stripe * dot(hv, dir) * k * veg);
      }
    }
    case GC_FARM: {
      g.soil_abs = 1.0;
      g.mid = M_GROUND; g.midk = 0.6;
      if (gex > 0.05 && (seed & 1u) == 0u) {
        // Row crop: leafy plants in rows on low ridges, soil between.
        g.la = L_LITTER; g.lb = L_SOIL; g.grassy = 0.4;
        let sp = 0.6 + 0.1 * f32((seed >> 1u) & 3u);
        let cover = clamp(0.3 + gex * 2.5, 0.3, 0.9);
        let ph = fract(across / sp) - 0.5;
        let x = abs(ph) * 2.0;
        let aa = smoothstep(0.12, 0.35, fa / sp);
        let wp = mix(1.0 - smoothstep(cover - 0.15, cover + 0.1, x), cover, aa);
        g.wb = 1.0 - wp;
        g.hrow = 0.06 * cos(TAU * ph) * (1.0 - aa);
        g.grow = perp * (-0.06 * TAU / sp * sin(TAU * ph) * (1.0 - aa));
        // The photo shows plants and soil together: what is left once the soil is taken out is plant.
        g.plant = max((img - (1.0 - cover) * g.soil) / cover, img * 0.4);
      } else if (gex > 0.05 || (yel > 0.14 && lum > 0.22)) {
        // Cereal, green or ripe, with wheel tracks.
        g.la = L_CEREAL; g.lb = L_SOIL; g.grassy = 0.8;
        g.mid = M_GRASS; g.midk = 0.4;
        g.tint = vec3f(1.0 + 0.1 * (n1 - 0.5));
        tramlines(&g, across, seed, fa);
      } else {
        // Ploughed or tilled: furrows.
        g.la = L_SOIL; g.lb = L_SOIL;
        let sp = 0.3 + 0.05 * f32(seed & 3u);
        let ph = fract(across / sp) - 0.5;
        let aa = smoothstep(0.12, 0.35, fa / sp);
        // Rounded ridges, sharp furrow bottoms.
        let r = 1.0 - 4.0 * ph * ph;
        g.hrow = 0.09 * (r - 0.66) * (1.0 - aa);
        g.grow = perp * (-0.09 * 8.0 * ph / sp * (1.0 - aa));
        g.plant = img;
        g.tint = vec3f(0.9 + 0.2 * r * (1.0 - aa));
      }
    }
    case GC_WET: {
      g.la = L_MOSS; g.lb = L_GRASS;
      g.wb = 0.5 + 0.8 * (n2 - 0.5);
      g.tint = vec3f(0.85); g.grassy = 0.5;
    }
    case GC_SAND: {
      g.la = L_SAND; g.lb = L_SAND;
      g.mid = M_SAND;
      rock_thr = 0.45;
    }
    case GC_ROCK: {
      g.rock = 1.0;
      g.mid = M_GROUND;
    }
    case GC_PAVED: {
      g.la = L_ASPH; g.lb = L_ASPH;
      g.amp = 0.0; rock_thr = 2.0;
      g.mid = M_PAVED;
    }
    default: {
      // Unmapped: lawn where the photo is green, else paving in town
      // (grey) and bare earth in the country (brown).
      let grey = 1.0 - smoothstep(0.03, 0.09, (img.r - img.b) / sum);
      g.lb = select(L_SOIL, L_ASPH, grey > 0.5);
      g.wb = 1.0 - smoothstep(0.15, 0.6, veg);
      g.grassy = veg;
      g.amp = mix(1.0, 0.3, grey);
      g.mid = select(select(M_GROUND, M_PAVED, grey > 0.5), M_GRASS, veg > 0.5);
      g.midk = 0.8;
    }
  }
  g.rock = max(g.rock, smoothstep(rock_thr, rock_thr + 0.12, slope + (n2 - 0.5) * 0.12));
  return g;
}

// ------------------------------------------------------------ material samples
// c: colour relative to the layer's mean, h: height 0..1, hm: metres,
// n: world-space normal tilt, cav: cavity.
struct Det { c: vec3f, h: f32, hm: f32, n: vec3f, cav: f32 };

fn det_top() -> i32 { return i32(textureNumLevels(t_dalb)) - 1; }
fn layer_mean(l: i32) -> vec3f {
  return max(textureLoad(t_dalb, vec2i(0), l, det_top()).rgb, vec3f(0.02));
}
// Mean cavity of a layer as det() combines it.
fn cav_mean(l: i32) -> f32 {
  let c = textureLoad(t_dnrm, vec2i(0), l, det_top()).b;
  return select(c, c * (1.0 + c) * 0.5, det_macro(l) > 1.0);
}

// One layer, sampled twice (as is, and larger or rotated) on the plane
// spanned by tangents t and b, with uv in metres.
// lite (0..1): far away only one sample is taken, the large copy for
// self-similar layers, the base one for the rest (whose detail is long gone).
fn det(l: i32, uv: vec2f, dx: vec2f, dy: vec2f, t: vec3f, b: vec3f, wc: vec2f, fw: f32, lite: f32) -> Det {
  let s = det_size(l);
  let m = det_macro(l);
  let r = det_range(l);
  let mean = layer_mean(l);
  let cm = textureLoad(t_dnrm, vec2i(0), l, det_top()).b;
  let u2 = ROT * uv / (s * m) + vec2f(0.37, 0.71);
  var o: Det;
  let photo = det_photo(l);
  if (photo > 0.0) {
    // Photo scan, hex-tiled. Far away only the colour is sampled: the normal
    // detail has long been filtered flat by then.
    let st = uv / s;
    let gx = dx / s;
    let gy = dy / s;
    let x = hex(st);
    let p = hex_p(x.w);
    var a0 = vec4f(0.0);
    var a1 = vec4f(0.0);
    var a2 = vec4f(0.0);
    if (p.x > 0.0) { a0 = textureSampleGrad(t_dalb, s_detail, x.s0, l, x.r0 * gx, x.r0 * gy); }
    if (p.y > 0.0) { a1 = textureSampleGrad(t_dalb, s_detail, x.s1, l, x.r1 * gx, x.r1 * gy); }
    if (p.z > 0.0) { a2 = textureSampleGrad(t_dalb, s_detail, x.s2, l, x.r2 * gx, x.r2 * gy); }
    let w = hex_w(p, vec3f(a0.a, a1.a, a2.a));
    let a = a0 * w.x + a1 * w.y + a2 * w.z;
    var nn = vec4f(0.5, 0.5, cm, a.a);
    var gg = vec2f(0.0);
    if (lite < 1.0) {
      var n0 = vec4f(0.5);
      var n1 = vec4f(0.5);
      var n2 = vec4f(0.5);
      if (p.x > 0.0) { n0 = textureSampleGrad(t_dnrm, s_detail, x.s0, l, x.r0 * gx, x.r0 * gy); }
      if (p.y > 0.0) { n1 = textureSampleGrad(t_dnrm, s_detail, x.s1, l, x.r1 * gx, x.r1 * gy); }
      if (p.z > 0.0) { n2 = textureSampleGrad(t_dnrm, s_detail, x.s2, l, x.r2 * gx, x.r2 * gy); }
      // Slopes back from each hexagon's rotated frame.
      gg = (transpose(x.r0) * (n0.xy * 2.0 - 1.0) * w.x + transpose(x.r1) * (n1.xy * 2.0 - 1.0) * w.y + transpose(x.r2) * (n2.xy * 2.0 - 1.0) * w.z) * (1.0 - lite);
      nn.z = mix(n0.z * w.x + n1.z * w.y + n2.z * w.z, cm, lite);
    }
    // Mostly brightness: a scan's own colour shifts (grey gravel in green
    // moss) turn odd on top of a different satellite colour.
    let rc = a.rgb / mean;
    let rl = max(dot(rc, vec3f(0.2126, 0.7152, 0.0722)), 1e-3);
    o.c = mix(vec3f(1.0), rl * mix(vec3f(1.0), rc / rl, det_chroma(l)), photo);
    o.h = a.a;
    o.hm = (a.a - 0.5) * r;
    o.n = t * gg.x + b * gg.y;
    o.cav = nn.z;
  } else if (m > 1.0) {
    // Self-similar layers: the large copy adds relief and colour variation on top.
    let a2 = textureSampleGrad(t_dalb, s_detail, u2, l, ROT * dx / (s * m), ROT * dy / (s * m));
    let n2 = textureSampleGrad(t_dnrm, s_detail, u2, l, ROT * dx / (s * m), ROT * dy / (s * m));
    var a1 = vec4f(mean, a2.a);
    var n1 = vec4f(0.5, 0.5, cm, 0.0);
    if (lite < 1.0) {
      a1 = mix(textureSampleGrad(t_dalb, s_detail, uv / s, l, dx / s, dy / s), a1, lite);
      n1 = mix(textureSampleGrad(t_dnrm, s_detail, uv / s, l, dx / s, dy / s), n1, lite);
    }
    // The rotated sample's slopes are in the rotated frame.
    let gg = (n1.xy * 2.0 - 1.0) + transpose(ROT) * (n2.xy * 2.0 - 1.0) * 0.5;
    o.c = (a1.rgb / mean) * mix(vec3f(1.0), a2.rgb / mean, 0.6);
    o.h = clamp(a1.a * 0.6 + a2.a * 0.4, 0.0, 1.0);
    o.hm = (a1.a - 0.5) * r + (a2.a - 0.5) * r * m * 0.5;
    o.n = t * gg.x + b * gg.y;
    o.cav = n1.b * mix(1.0, n2.b, 0.5);
  } else {
    let a1 = textureSampleGrad(t_dalb, s_detail, uv / s, l, dx / s, dy / s);
    let n1 = textureSampleGrad(t_dnrm, s_detail, uv / s, l, dx / s, dy / s);
    var a2 = a1;
    var n2 = n1;
    var k = 0.0;
    if (lite < 1.0) {
      // Same scale: patches of one copy or the other, joined at the low points.
      a2 = textureSampleGrad(t_dalb, s_detail, u2, l, ROT * dx / s, ROT * dy / s);
      n2 = textureSampleGrad(t_dnrm, s_detail, u2, l, ROT * dx / s, ROT * dy / s);
      n2 = vec4f(transpose(ROT) * (n2.xy * 2.0 - 1.0) * 0.5 + 0.5, n2.zw);
      k = smoothstep(0.35, 0.65, wn(wc + vec2f(3.1, 1.3), s * 9.0, fw) + (a2.a - a1.a) * 0.3) * (1.0 - lite);
    }
    o.c = mix(a1.rgb, a2.rgb, k) / mean;
    o.h = mix(a1.a, a2.a, k);
    o.hm = (o.h - 0.5) * r;
    let gg = mix(n1.xy, n2.xy, k) * 2.0 - 1.0;
    o.n = t * gg.x + b * gg.y;
    o.cav = mix(n1.b, n2.b, k);
  }
  return o;
}

// Layer l (rock) projected along the three local axes, so cliffs do not smear.
fn tri_det(l: i32, wc: vec2f, alt: f32, N: vec3f, dwc: mat2x2f, dalt: vec2f, fw: f32, lite: f32) -> Det {
  let e = G.east.xyz;
  let nn = G.north.xyz;
  let u = G.up.xyz;
  var w = pow(abs(vec3f(dot(N, u), dot(N, e), dot(N, nn))), vec3f(4.0));
  w /= w.x + w.y + w.z;
  var o = Det(vec3f(0.0), 0.0, 0.0, vec3f(0.0), 0.0);
  var tw = 0.0;
  if (w.x > 0.02) {
    let d = det(l, wc, dwc[0], dwc[1], e, nn, wc, fw, lite);
    let k = w.x * (0.5 + d.h);
    o.c += d.c * k; o.h += d.h * k; o.hm += d.hm * k; o.n += d.n * k; o.cav += d.cav * k; tw += k;
  }
  if (w.y > 0.02) {
    let q = vec2f(wc.y, alt);
    let d = det(l, q, vec2f(dwc[0].y, dalt.x), vec2f(dwc[1].y, dalt.y), nn, u, wc, fw, lite);
    let k = w.y * (0.5 + d.h);
    o.c += d.c * k; o.h += d.h * k; o.hm += d.hm * k; o.n += d.n * k; o.cav += d.cav * k; tw += k;
  }
  if (w.z > 0.02) {
    let q = vec2f(wc.x, alt);
    let d = det(l, q, vec2f(dwc[0].x, dalt.x), vec2f(dwc[1].x, dalt.y), e, u, wc, fw, lite);
    let k = w.z * (0.5 + d.h);
    o.c += d.c * k; o.h += d.h * k; o.hm += d.hm * k; o.n += d.n * k; o.cav += d.cav * k; tw += k;
  }
  let iw = 1.0 / max(tw, 1e-4);
  return Det(o.c * iw, o.h * iw, o.hm * iw, o.n * iw, o.cav * iw);
}

// Blend toward layer b by share w, with b showing first in a's low points.
fn hblend(ha: f32, hb: f32, w: f32) -> f32 {
  let x = (w - 0.5) * 2.4 + (hb - ha) * 0.9;
  var t = smoothstep(-0.3, 0.3, x);
  t = min(t, w * 10.0);
  return max(t, 1.0 - (1.0 - w) * 10.0);
}

// Parallax occlusion: walk the main layer's height field along the view ray
// (Vt in the tangent frame) and return where the ray first hits it.
fn pom(l: i32, wc: vec2f, Vt: vec3f, depth: f32, dx: vec2f, dy: vec2f) -> vec2f {
  let s = det_size(l);
  let steps = i32(mix(14.0, 5.0, clamp(Vt.z, 0.0, 1.0)));
  let dl = 1.0 / f32(steps);
  let shift = -Vt.xy / max(Vt.z, 0.25) * depth * dl;
  var cur = wc;
  var d = 0.0;
  var prev = wc;
  var prev_gap = 1.0;
  for (var i = 0; i < steps; i++) {
    var h: f32;
    if (det_photo(l) > 0.0) { h = hex_h(l, cur / s, dx / s, dy / s); } else { h = textureSampleGrad(t_dalb, s_detail, cur / s, l, dx / s, dy / s).a; }
    let gap = (1.0 - h) - d;
    if (gap <= 0.0) {
      let t = prev_gap / max(prev_gap - gap, 1e-5);
      return mix(prev, cur, select(t, 1.0, i == 0));
    }
    prev = cur;
    prev_gap = gap;
    d += dl;
    cur += shift;
  }
  return cur;
}

// ------------------------------------------------------------ vertices
// sp: spacing of the vertices around this one (m): the waves are filtered to it.
fn finish(Pin: vec3f, nrm: vec3f, uv: vec2f, depth: f32, skirt: f32, sp: f32) -> VOut {
  var P = Pin;
  let m = textureSampleLevel(t_mask, s_clamp, uv, 1.0);
  // (a tile whose water is drawn as its own fine grid, vs_water, keeps its
  // own vertices still: the land's edge stays where it is)
  if (m.r > 0.5 && skirt < 0.5 && !dense_water()) {
    let d = length(P);
    let fade = 1.0 - smoothstep(2500.0, 6000.0, d);
    if (fade > 0.0) {
      let upn = normalize(P - G.planet.xyz);
      let shore = smoothstep(0.5, 1.0, textureSampleLevel(t_mask, s_clamp, uv, 3.0).r);
      let st = sea_state(m.g * 255.0 / 80.0, upn.z, shore);
      // The FFT ocean: height, and the sideways shift that sharpens the crests.
      let w = ocean_disp(ocean_coord(P), sp, st) * fade;
      P += G.east.xyz * w.x + G.north.xyz * w.y + upn * w.z;
    }
  }
  if (skirt < 0.5) { P += normalize(P - G.planet.xyz) * crater_field(P).w; }
  return VOut(G.view_proj * vec4f(P, 1.0), P, nrm, uv, depth);
}

// Our water seen from above, into the green channel of the 3D tiles' coverage
// mask: the height of its surface, so the mesh can give way to it (photo.wgsl).
@group(2) @binding(0) var<uniform> CS: Cascade;
struct TopOut { @builtin(position) clip: vec4f, @location(0) uv: vec2f, @location(1) h: f32, @location(2) skirt: f32 };
@vertex fn vs_top(@location(0) pos: vec3f, @location(3) skirt: f32, @location(4) uv: vec2f) -> TopOut {
  let P = pos + T.offset.xyz;
  return TopOut(CS.vp * vec4f(P, 1.0), uv, height_of(P), skirt);
}
@fragment fn fs_top(in: TopOut) -> @location(0) vec4f {
  if (in.skirt > 0.5 || textureSampleLevel(t_mask, s_clamp, in.uv, 0.0).r < 0.5) { discard; }
  return vec4f(0.0, in.h, 0.0, 0.0);
}
// Where any ground was drawn at all, into the red channel of the shore mask
// (shore.rs): land there is ground without our water on it.
@fragment fn fs_cover(in: TopOut) -> @location(0) vec4f {
  if (in.skirt > 0.5) { discard; }
  return vec4f(1.0, 0.0, 0.0, 0.0);
}

@vertex fn vs(@location(0) pos: vec3f, @location(1) depth: f32, @location(2) nrm: vec3f, @location(3) skirt: f32, @location(4) uv: vec2f) -> VOut {
  return finish(pos + T.offset.xyz, nrm, uv, depth, skirt, T.offset.w);
}

// Tessellation without a tessellation stage: every coarse grid cell near the
// camera is drawn as a TN x TN patch whose vertices snap to a coarser lattice
// with distance. Each edge picks its level from its own midpoint, so the two
// cells sharing it agree and the surface stays closed.
const TN = 16u;
// Triangles are kept to about TESS_K of their distance across.
const TESS_K = 0.03;

struct CV { P: vec3f, depth: f32, N: vec3f, uv: vec2f };
fn cv(i: u32) -> CV {
  let b = i * 10u;
  return CV(vec3f(TV[b], TV[b + 1u], TV[b + 2u]) + T.offset.xyz, TV[b + 3u], vec3f(TV[b + 4u], TV[b + 5u], TV[b + 6u]), vec2f(TV[b + 8u], TV[b + 9u]));
}
// Written out so that t = 0 and t = 1 give the end points exactly.
fn lerp_cv(a: CV, b: CV, t: f32) -> CV {
  let s = 1.0 - t;
  return CV(a.P * s + b.P * t, a.depth * s + b.depth * t, a.N * s + b.N * t, a.uv * s + b.uv * t);
}
fn tess_level(mid: vec3f) -> f32 {
  let sp = max(length(mid) * TESS_K, T.offset.w / f32(TN));
  return clamp(exp2(floor(log2(max(T.offset.w / sp, 1.0)))), 1.0, f32(TN));
}
fn snapk(k: u32, lv: f32) -> f32 { return round(f32(k) * lv / f32(TN)) / lv; }

// Relief (m, along the normal) added by the tessellated surface.
fn relief(P: vec3f, N: vec3f, uv: vec2f, sp: f32) -> f32 {
  let wc = wave_coord(P);
  let up = normalize(P - G.planet.xyz);
  let img = textureSampleLevel(t_img, s_clamp, uv, 3.0).rgb;
  // Rows only need ~3 vertices per period to show, so filter for a smaller footprint than the spacing.
  let g = ground(uv, wc, img, 1.0 - dot(normalize(N), up), vec2f(sp * 0.4, 0.0), vec2f(0.0, sp * 0.4), -normalize(P));
  // Large-scale relief only: what is smaller than the mesh is left to parallax and normals.
  let ha = layer_h_lod(g.la, wc, sp);
  let hb = layer_h_lod(g.lb, wc, sp);
  var h = mix((ha - 0.5) * det_range(g.la) * det_macro(g.la) * 0.5, (hb - 0.5) * det_range(g.lb) * det_macro(g.lb) * 0.5, g.wb);
  // Tussocks.
  h += (wn(wc, 2.3, sp) - 0.5) * 0.06 * g.grassy;
  if (det_photo(g.mid) > 0.0) { h += (layer_h_lod(g.mid, wc, sp) - 0.5) * det_range(g.mid) * g.midk; }
  h += g.hrow;
  if (g.rock > 0.0) {
    let alt = dot(P, G.up.xyz) + G.up.w;
    let nn = normalize(N);
    var w = pow(abs(vec3f(dot(nn, G.up.xyz), dot(nn, G.east.xyz), dot(nn, G.north.xyz))), vec3f(4.0));
    w /= w.x + w.y + w.z;
    let hr = w.x * layer_h_lod(L_ROCK, wc, sp) + w.y * layer_h_lod(L_ROCK, vec2f(wc.y, alt), sp) + w.z * layer_h_lod(L_ROCK, vec2f(wc.x, alt), sp);
    h = mix(h, (hr - 0.5) * det_range(L_ROCK) * det_macro(L_ROCK) * 0.5, g.rock);
  }
  return h * g.amp;
}

@vertex fn vs_tess(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  let rw = u32(T.tess.z);
  let ci = u32(T.tess.x) + ii % rw;
  let cj = u32(T.tess.y) + ii / rw;
  let a = vi % (TN + 1u);
  let b = vi / (TN + 1u);
  let i00 = cj * GV + ci;
  let c00 = cv(i00);
  let c10 = cv(i00 + 1u);
  let c01 = cv(i00 + GV);
  let c11 = cv(i00 + GV + 1u);
  var v: CV;
  if (b == 0u) {
    v = lerp_cv(c00, c10, snapk(a, tess_level((c00.P + c10.P) * 0.5)));
  } else if (b == TN) {
    v = lerp_cv(c01, c11, snapk(a, tess_level((c01.P + c11.P) * 0.5)));
  } else if (a == 0u) {
    v = lerp_cv(c00, c01, snapk(b, tess_level((c00.P + c01.P) * 0.5)));
  } else if (a == TN) {
    v = lerp_cv(c10, c11, snapk(b, tess_level((c10.P + c11.P) * 0.5)));
  } else {
    let lv = tess_level((c00.P + c11.P) * 0.5);
    let fa = snapk(a, lv);
    let fb = snapk(b, lv);
    // Same diagonal as the coarse mesh.
    if (fa + fb <= 1.0) {
      v = CV(c00.P + (c10.P - c00.P) * fa + (c01.P - c00.P) * fb, c00.depth, c00.N + (c10.N - c00.N) * fa + (c01.N - c00.N) * fb, c00.uv + (c10.uv - c00.uv) * fa + (c01.uv - c00.uv) * fb);
    } else {
      let ga = 1.0 - fa;
      let gb = 1.0 - fb;
      v = CV(c11.P + (c01.P - c11.P) * ga + (c10.P - c11.P) * gb, c11.depth, c11.N + (c01.N - c11.N) * ga + (c10.N - c11.N) * gb, c11.uv + (c01.uv - c11.uv) * ga + (c10.uv - c11.uv) * gb);
    }
  }
  var P = v.P;
  let d = length(P);
  // Edges are back to level 1 (unsplit) beyond this: the relief must be gone
  // by then, so the block joins the plain cells around it without cracks. Also
  // fade at tile borders (the neighbour tile cannot follow).
  let far = T.offset.w / (2.0 * TESS_K);
  let edge = min(min(v.uv.x, 1.0 - v.uv.x), min(v.uv.y, 1.0 - v.uv.y)) * f32(GRID);
  let fade = (1.0 - smoothstep(far * 0.5, far * 0.9, d)) * smoothstep(0.0, 1.0, edge);
  let water = textureSampleLevel(t_mask, s_clamp, v.uv, 0.0).r;
  if (fade > 0.0 && water < 0.05) {
    let sp = max(d * TESS_K, T.offset.w / f32(TN));
    P += normalize(v.N) * relief(P, v.N, v.uv, sp) * fade;
  }
  // The waves follow the finer spacing where the cells are split, back to
  // the plain cells' where the block must join them.
  let wsp = mix(T.offset.w, max(d * TESS_K, T.offset.w / f32(TN)), fade);
  return finish(P, v.N, v.uv, v.depth, 0.0, wsp);
}

// The rest of a tessellated tile, one quad per cell; the cells drawn by
// vs_tess collapse to nothing.
@vertex fn vs_cells(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  let ci = ii % GRID;
  let cj = ii / GRID;
  let t = vec4u(T.tess);
  if (ci >= t.x && ci < t.x + t.z && cj >= t.y && cj < t.y + t.w) {
    return VOut(vec4f(2.0, 2.0, 2.0, 1.0), vec3f(0.0), vec3f(0.0, 0.0, 1.0), vec2f(0.0), 0.0);
  }
  let v = cv(cj * GV + ci + (vi & 1u) + (vi >> 1u) * GV);
  return finish(v.P, v.N, v.uv, v.depth, 0.0, T.offset.w);
}

// ------------------------------------------------------------ water
// The water's G-buffer: its photo (unlit tint) and depth, class and openness
// for the composite (composite.wgsl `water`).
fn water_out(img: vec3f, m: vec4f, shore: f32, d: f32) -> GOut {
  let wclass = round(m.g * 255.0 / 80.0);
  // Bathymetry is coarse (and missing in bays): never shallower than a plausible shore profile.
  let profile = 0.35 + shore * shore * select(select(9.0, 3.5, wclass > 2.5), 14.0, wclass < 1.5);
  var depth = max(mix(min(d, 1.2), d, shore), profile);
  return GOut(vec4f(img, 1.0), vec4f(max(depth, 0.15), wclass, shore, 1.0), 0.0);
}

// Real waves (Water High): the water of the tiles near the camera is drawn a
// second time as its own fine grid, S x S over the tile (S = T.info.y,
// render.rs `water_idx`, picked by distance in world.rs `draw_lists`), and
// the FFT ocean moves every vertex - the swell and the chop stand up as
// relief, on the horizon too. The tile's own pass then leaves its water out
// (fs) and keeps its vertices still. The grid lies on the tile's own
// triangles (the same diagonals), and a pixel is water or land by the same
// mask in both passes, so where the waves die away at the shore the two meet
// without a seam. Vertices past (S + 1)^2 are a skirt hanging just below the
// grid's border, over the hairline cracks against a neighbouring tile's grid.
fn dense_water() -> bool { return T.info.y > 0.5; }
// The waves fade out between these distances (m): world.rs draws the grid for
// every tile with water nearer than the far one.
const DENSE_FAR = vec2f(3500.0, 8000.0);

@vertex fn vs_water(@builtin(vertex_index) vi: u32) -> VOut {
  let S = u32(T.info.y);
  let row = S + 1u;
  var k = vi;
  var skirt = false;
  if (k >= row * row) {
    // around the border: bottom, right, top, left
    let r = k - row * row;
    let o = r % S;
    var ij = vec2u(0u, S - o);
    switch (r / S) {
      case 0u: { ij = vec2u(o, 0u); }
      case 1u: { ij = vec2u(S, o); }
      case 2u: { ij = vec2u(S - o, S); }
      default: {}
    }
    k = ij.y * row + ij.x;
    skirt = true;
  }
  // Where in the tile's cells, and on which of the cell's two triangles.
  let g = vec2f(f32(k % row), f32(k / row)) * (f32(GRID) / f32(S));
  let ci = min(u32(g.x), GRID - 1u);
  let cj = min(u32(g.y), GRID - 1u);
  let fa = g.x - f32(ci);
  let fb = g.y - f32(cj);
  let i00 = cj * GV + ci;
  var v: CV;
  if (fa + fb <= 1.0) {
    let c00 = cv(i00);
    let c10 = cv(i00 + 1u);
    let c01 = cv(i00 + GV);
    v = CV(c00.P + (c10.P - c00.P) * fa + (c01.P - c00.P) * fb, c00.depth + (c10.depth - c00.depth) * fa + (c01.depth - c00.depth) * fb, c00.N, c00.uv + (c10.uv - c00.uv) * fa + (c01.uv - c00.uv) * fb);
  } else {
    let c11 = cv(i00 + GV + 1u);
    let c10 = cv(i00 + 1u);
    let c01 = cv(i00 + GV);
    let ga = 1.0 - fa;
    let gb = 1.0 - fb;
    v = CV(c11.P + (c01.P - c11.P) * ga + (c10.P - c11.P) * gb, c11.depth + (c01.depth - c11.depth) * ga + (c10.depth - c11.depth) * gb, c11.N, c11.uv + (c01.uv - c11.uv) * ga + (c10.uv - c11.uv) * gb);
  }
  var P = v.P;
  let d = length(P);
  let upn = normalize(P - G.planet.xyz);
  let fade = 1.0 - smoothstep(DENSE_FAR.x, DENSE_FAR.y, d);
  let m = textureSampleLevel(t_mask, s_clamp, v.uv, 1.0);
  if (fade > 0.0 && m.r > 0.0) {
    let shore = smoothstep(0.5, 1.0, textureSampleLevel(t_mask, s_clamp, v.uv, 3.0).r);
    let st = sea_state(m.g * 255.0 / 80.0, upn.z, shore);
    // Calm at the water's edge: from the shore distance field where it
    // reaches (over a span that grows with the sea), else from the mask.
    var calm = smoothstep(0.5, 1.0, m.r);
    let sh = dense_shore(P);
    if (sh >= 0.0) { calm = min(calm, smoothstep(1.5, 8.0 + 4.0 * G.ocean[1].y, sh)); }
    // Filtered to the grid's spacing, or a little finer than the pixels far off.
    let sp = max(T.offset.w * f32(GRID) / f32(S), d * 0.0025);
    let w = ocean_disp(ocean_coord(P), sp, st) * (fade * calm);
    P += G.east.xyz * w.x + G.north.xyz * w.y + upn * w.z;
  }
  if (skirt) { P -= upn * (0.5 + 0.004 * d); }
  P += upn * crater_field(P).w;
  return VOut(G.view_proj * vec4f(P, 1.0), P, v.N, v.uv, v.depth);
}
// Distance to land (m) from the shore field (shore.rs), or -1 off its square.
fn dense_shore(P: vec3f) -> f32 {
  if (G.shore.x < 0.5) { return -1.0; }
  let q = G.shore_vp * vec4f(P, 1.0);
  let uv = q.xy * vec2f(0.5, -0.5) + 0.5;
  if (any(uv < vec2f(0.002)) || any(uv > vec2f(0.998))) { return -1.0; }
  return textureSampleLevel(t_shore, s_clamp, uv, 0.0).x;
}
@fragment fn fs_water(in: VOut) -> GOut {
  let img = textureSample(t_img, s_aniso, in.uv).rgb;
  let m = textureSample(t_mask, s_clamp, in.uv);
  if (m.r <= 0.5) { discard; }
  let shore = smoothstep(0.5, 1.0, textureSampleLevel(t_mask, s_clamp, in.uv, 3.0).r);
  return water_out(img, m, shore, in.depth);
}

// ------------------------------------------------------------ surface
@fragment fn fs(in: VOut) -> GOut {
  // Derivatives first, while every pixel of the quad is still running.
  let wc0 = wave_coord(in.P);
  let dx = dpdx(wc0);
  let dy = dpdy(wc0);
  let alt = dot(in.P, G.up.xyz) + G.up.w;
  let dalt = vec2f(dpdx(alt), dpdy(alt));
  let img = textureSample(t_img, s_aniso, in.uv).rgb;
  let m = textureSample(t_mask, s_clamp, in.uv);
  let shore = smoothstep(0.5, 1.0, textureSampleLevel(t_mask, s_clamp, in.uv, 3.0).r);
  if (m.r > 0.5) {
    // (drawn by fs_water instead)
    if (dense_water()) { discard; }
    return water_out(img, m, shore, in.depth);
  }
  // Inside the 3D tiles zone the mesh stands in for the land (tiles3d.rs);
  // our water stays, its waves and reflections over the mesh's still photo.
  if (photo_hides(in.P, in.clip.xy)) { discard; }
  var albedo = img;
  let dist = length(in.P);
  let V = -in.P / max(dist, 1e-3);
  var N = in.N;
  if (dot(N, N) < 1e-8) { N = normalize(in.P - G.planet.xyz); }
  N = normalize(N);
  let upn = normalize(in.P - G.planet.xyz);
  var wc = wc0;
  let fw = max(length(dx), length(dy));
  if (dist < DET_FAR) {
    let fade = 1.0 - smoothstep(DET_FAR * 0.45, DET_FAR, dist);
    let g = ground(in.uv, wc0, img, 1.0 - dot(N, upn), dx, dy, V);
    // Tangent frame along east / north, laid onto the surface.
    let t = normalize(G.east.xyz - N * dot(G.east.xyz, N));
    let b = cross(N, t);
    let graze = 1.0 - abs(dot(V, N));
    // Parallax on the main layer up close.
    let pk = (1.0 - smoothstep(18.0, 40.0, dist)) * (1.0 - g.rock) * g.amp;
    if (pk > 0.0) {
      let Vt = vec3f(dot(V, t), dot(V, b), dot(V, N));
      wc = pom(g.la, wc0, Vt, det_range(g.la) * pk, dx, dy);
    }
    let lite = smoothstep(120.0, 180.0, dist);
    let A = det(g.la, wc, dx, dy, t, b, wc, fw, lite);
    var B = A;
    var wb = g.wb * (1.0 - 0.7 * graze * graze * g.grassy);
    if (g.lb != g.la && wb > 0.001) { B = det(g.lb, wc, dx, dy, t, b, wc, fw, lite); }
    let k = hblend(A.h, B.h, wb);
    let ca = g.plant * A.c;
    let cb = mix(img, g.soil, g.soil_abs) * B.c;
    var c = mix(ca, cb, k) * g.tint;
    var nd = mix(A.n, B.n, k) + t * (-g.grow.x) + b * (-g.grow.y);
    var cav = mix(A.cav, B.cav, k);
    var cavm = mix(cav_mean(g.la), cav_mean(g.lb), k);
    var h = mix(A.h, B.h, k);
    var kr = 0.0;
    if (g.rock > 0.001) {
      let R = tri_det(L_ROCK, wc, alt, N, mat2x2f(dx, dy), dalt, fw, lite);
      kr = hblend(h, R.h, g.rock);
      // Rock takes the photo's brightness, greyed.
      let rc = mix(img, vec3f(luma(img)), 0.55) * R.c;
      c = mix(c, rc, kr);
      nd = mix(nd, R.n, kr);
      cav = mix(cav, R.cav, kr);
      cavm = mix(cavm, cav_mean(L_ROCK), kr);
    }
    // Mid-scale detail: tussocks, bare patches, stones, fractures, finer than
    // a satellite pixel. Rock gets its own, projected like the close-up rock.
    var mk = g.midk * select(0.0, 1.0, det_photo(g.mid) > 0.0);
    let mrk = kr * select(0.0, 1.0, det_photo(M_ROCK) > 0.0);
    if (mk > 0.0 || mrk > 0.0) {
      let mlite = smoothstep(350.0, 500.0, dist);
      var M = Det(vec3f(1.0), 0.5, 0.0, vec3f(0.0), cav_mean(g.mid));
      if (mk > 0.0 && mrk < 0.999) { M = det(g.mid, wc0, dx, dy, t, b, wc0, fw, mlite); }
      M.c = mix(vec3f(1.0), M.c, mk);
      M.n *= mk;
      var mcm = cav_mean(g.mid);
      if (mrk > 0.0) {
        let R = tri_det(M_ROCK, wc0, alt, N, mat2x2f(dx, dy), dalt, fw, mlite);
        M = Det(mix(M.c, R.c, mrk), M.h, M.hm, mix(M.n, R.n, mrk), mix(M.cav, R.cav, mrk));
        mcm = mix(mcm, cav_mean(M_ROCK), mrk);
        mk = max(mk, mrk);
      }
      c *= M.c;
      nd += M.n;
      // Its cavities darken like the close-up ones (relative to its mean).
      c *= mix(1.0, (0.55 + 0.45 * M.cav) / (0.55 + 0.45 * mcm), mk);
    }
    // Blades hide the ground between them at grazing angles, and shine.
    cav = mix(cav, 1.0, graze * graze * g.grassy);
    c *= 1.0 + 0.18 * pow(graze, 4.0) * g.grassy;
    // Sunlit blades are warmer than the hazy photo suggests.
    c *= mix(vec3f(1.0), vec3f(1.03, 1.04, 0.84), g.grassy);
    // Darken the cracks and gaps without changing the average brightness of the photo.
    c *= (0.55 + 0.45 * cav) / (0.55 + 0.45 * mix(cavm, 1.0, graze * graze * g.grassy));
    albedo = mix(img, c, fade);
    N = normalize(N + nd * fade);
  }
  for (var i = 0; i < 8; i++) {
    let s = G.scorch[i];
    if (s.w > 0.0) {
      let d = length(in.P - s.xyz) / s.w;
      if (d < 1.0) {
        let n = textureSampleLevel(t_wave, s_repeat, wc0 / 23.0 + f32(i) * 0.37, 0.0).a;
        let k = smoothstep(1.0, 0.25, d + (n - 0.5) * 0.7);
        albedo = mix(albedo, vec3f(0.012, 0.011, 0.01), k * 0.94);
      }
    }
  }
  // Wet sand along the waterline: darker, and faintly glossy.
  let wet = smoothstep(0.15, 0.45, textureSampleLevel(t_mask, s_clamp, in.uv, 1.0).r) * (1.0 - smoothstep(800.0, 3000.0, dist));
  albedo *= 1.0 - 0.3 * wet;
  // Craters: bent normals, bare earth and rock where the ground was dug out.
  let cr = crater_field(in.P);
  if (dot(cr.xyz, cr.xyz) > 1e-6 || cr.w != 0.0) {
    N = normalize(N - cr.xyz);
    let rk = textureSampleLevel(t_wave, s_repeat, wc0 / 4.3, 0.0).a;
    let rk2 = textureSampleLevel(t_wave, s_repeat, wc0 / 0.8, 0.0).a;
    let soil = mix(vec3f(0.11, 0.085, 0.06), vec3f(0.19, 0.17, 0.15), smoothstep(0.4, 0.65, rk)) * (0.6 + 0.8 * rk2);
    albedo = mix(albedo, soil, clamp(abs(cr.w) * 1.6 + length(cr.xyz) * 1.5, 0.0, 1.0));
  }
  albedo = with_snow(albedo, in.P, N);
  let c = shade(in.P, N, albedo);
  return GOut(vec4f(c, 1.0), vec4f(N, 0.0), amb_share(c));
}
