// Buildings: procedural facades by building type, roofs coloured from the
// satellite image with procedural material detail, bridge decks.
// lod: how far faded in the middle and finest levels of detail are (0..1).
struct BTile { offset: vec4f, ax: vec4f, ay: vec4f, lod: vec4f };
@group(1) @binding(0) var<uniform> T: BTile;
@group(1) @binding(1) var t_atlas: texture_2d<f32>;
@group(2) @binding(0) var<uniform> CS: Cascade;

struct VOut {
  @builtin(position) clip: vec4f,
  @location(0) P: vec3f,
  @location(1) N: vec3f,
  @location(2) uv: vec2f,
  @location(3) col: vec4f,
  @location(4) @interpolate(flat) sty: vec4u,
  @location(5) auv: vec2f,
  // level of detail this geometry belongs to (low 3 bits: 0 always drawn,
  // 1 middle, 2 finest, 3 full geometry going, 4 far stand-in coming) and 4x
  // the distance scale of its height class above them (see `lod_dist`)
  @location(6) @interpolate(flat) lod: u32,
};
struct GOut { @location(0) c: vec4f, @location(1) n: vec4f, @location(2) amb: f32 };

@vertex fn vs(@location(0) pos: vec3f, @location(1) nrm: vec3f, @location(2) uv: vec2f, @location(3) col: vec4f, @location(4) sty: vec4u, @builtin(instance_index) lod: u32) -> VOut {
  let P = pos + T.offset.xyz;
  // The roof atlas is the satellite image of this map tile, so its texture
  // coordinate is (very nearly) a linear function of position.
  let auv = vec2f(dot(pos, T.ax.xyz) + T.ax.w, dot(pos, T.ay.xyz) + T.ay.w);
  var clip = G.view_proj * vec4f(P, 1.0);
  // Detail beyond the distance its level is drawn to is dropped here, before
  // it costs fill (and far stand-ins short of theirs). The margin keeps every
  // corner of a triangle that is cut this way past the end of its dissolve.
  let d = lod_dist(P, lod);
  let l = lod & 7u;
  if ((l == 1u || l == 3u) && d > FAR1 + 250.0) || (l == 2u && d > LOD2_FAR) || (l == 4u && d < FAR0 - 250.0) { clip = vec4f(2.0, 2.0, 2.0, 1.0); }
  return VOut(clip, P, nrm, uv, col, sty, auv, lod);
}
// Distance the levels of detail go by: towers keep theirs out to a multiple of
// the usual distances (buildings::HCLASS; the draw's instance index carries 4x
// the multiple above the level).
fn lod_dist(P: vec3f, lod: u32) -> f32 {
  return length(P) / max(f32(lod >> 3u) * 0.25, 1.0);
}
const LOD2_FAR = 680.0;
// Distances over which buildings dissolve from their full geometry (and its
// middle level of dressing) into their far stand-ins. Must match BFAR in buildings.rs.
const FAR0 = 1000.0;
const FAR1 = 1300.0;
@vertex fn vs_shadow(@location(0) pos: vec3f) -> @builtin(position) vec4f {
  // Under the 3D tiles' mesh, which casts instead (decided for the whole
  // batch, so no triangle is cut in two).
  if (G.photo.z > 0.0 && photo_fade(T.offset.xyz) < 0.5) { return vec4f(2.0, 2.0, 2.0, 1.0); }
  return CS.vp * vec4f(pos + T.offset.xyz, 1.0);
}

// ---------------------------------------------------------------- roofs
// A roof's look: albedo, how far the normal leans along the contour (barrel
// tiles, corrugation), and its specular response.
struct RoofOut { albedo: vec3f, bend: f32, f0: f32, rough: f32, metal: f32 };

// Noise over metres from the tiling wave texture, `scale` metres to a tile and
// turned by `ang` so that scales laid over each other never line up. `fwm` is
// the pixel footprint in metres, for picking the mip.
fn mnoise(uv: vec2f, scale: f32, ang: f32, off: vec2f, fwm: f32) -> f32 {
  let cs = vec2f(cos(ang), sin(ang));
  let q = vec2f(cs.x * uv.x - cs.y * uv.y, cs.y * uv.x + cs.x * uv.y) / scale + off;
  return wave_a(q, fwm / scale * 256.0);
}
// The wave texture's noise channel at a footprint of `texels` (of the 256
// level), mip picked by hand: roof shading branches by material, where screen
// derivatives are not to be had. Once a whole tile of it spans only a few
// pixels its repeats line up into a visible lattice, so it fades to its mean.
fn wave_a(q: vec2f, texels: f32) -> f32 {
  let lod = log2(max(texels, 1.0));
  return mix(textureSampleLevel(t_wave, s_repeat, q, lod).a, 0.5, smoothstep(3.0, 4.5, lod));
}

// The same, centred and stretched to roughly unit spread (the raw channel
// is a sum of many waves, so it hugs 0.5).
fn mnz(uv: vec2f, scale: f32, ang: f32, off: vec2f, fwm: f32) -> f32 {
  return (mnoise(uv, scale, ang, off, fwm) - 0.5) * 3.0;
}

// How much of a repeating pattern to draw: all of it while a period spans
// plenty of pixels, fading to its mean before the fine lines inside it alias.
fn pattern_fade(period: f32, fw: f32) -> f32 { return smoothstep(2.5, 7.0, period / fw); }

fn roof_image(img: vec3f, base: vec3f) -> f32 {
  let s = img.r + img.g + img.b + 1e-4;
  let veg = clamp(((2.0 * img.g - img.r - img.b) / s - 0.02) / 0.08, 0.0, 1.0);
  let ratio = luma(img) / max(luma(base), 0.004);
  let keep = (1.0 - veg) * smoothstep(0.3, 0.6, ratio);
  return mix(1.0, clamp(ratio, 0.6, 1.5), keep);
}

// Pitched roofs. uv: x along the contour, y down the slope (metres from the
// eave line, negative); ridge: where the ridge is in uv.y (0: none).
fn pitched_roof(uv: vec2f, base: vec3f, mat: u32, seed: f32, wealth: f32, fw_in: vec2f, ridge: f32) -> RoofOut {
  let fw = max(fw_in, vec2f(0.0015));
  let fwm = max(fw.x, fw.y);
  let sd = vec2f(seed * 17.3, seed * 9.1);
  var o = RoofOut(base, 0.0, 0.04, 0.85, 0.0);
  // The laid pattern: courses down the slope, tiles across. Each fades to its
  // mean before it would shimmer.
  var pat = 1.0;
  var weather = 1.0;  // how readily it streaks and grows algae
  if (mat == 1u) {
    // Clay barrel tiles: humps that catch the sun on one side.
    let fine = pattern_fade(0.22, fw.x) * pattern_fade(0.34, fw.y);
    let row = floor(uv.y / 0.34);
    let u = uv.x / 0.22 + 0.15 * hash1(row + seed * 41.0);
    let sn = sin(u * PI);
    let lap = 0.72 + 0.28 * smoothstep(-fw.y / 0.34, 0.3 + fw.y / 0.34, fract(uv.y / 0.34));
    pat = mix(0.95, (0.8 + 0.25 * abs(sn)) * lap * (0.86 + 0.28 * hash2(vec2f(floor(u), row))), fine);
    o.bend = 0.55 * cos(u * PI) * sign(sn) * fine;
    o.rough = 0.7;
    weather = 0.6;
  } else if (mat == 2u || mat == 4u) {
    // Sheet metal: standing seams or corrugation, rust where nobody repaints.
    let corr = mat == 4u;
    let period = select(0.45, 0.076, corr);
    let fine = pattern_fade(period, fw.x);
    let sx = fract(uv.x / period) - 0.5;
    if (corr) {
      o.bend = 0.5 * sin(sx * 6.283) * fine;
    } else {
      // Ribs: lit on one side, shaded on the other.
      let rib = box(sx * period, 0.02, fw.x);
      o.bend = 0.6 * (box(sx * period - 0.012, 0.012, fw.x) - box(sx * period + 0.012, 0.012, fw.x)) * fine;
      pat = mix(1.0, 1.0 - 0.08 * rib, fine);
      // Panels from different batches.
      pat *= 0.95 + 0.1 * hash1(floor(uv.x / period) + seed * 13.0);
    }
    let n1 = mnoise(uv, 6.0, 0.3, sd, fwm);
    let n2 = mnoise(uv, 0.9, 1.7, sd.yx, fwm);
    let rust = smoothstep(0.35 + 0.6 * wealth, 0.75 + 0.5 * wealth, n1 * 0.7 + n2 * 0.4);
    // Bare or galvanised sheet is metal; painted sheet is paint over metal.
    let grey = 1.0 - smoothstep(0.03, 0.12, (max(base.r, max(base.g, base.b)) - min(base.r, min(base.g, base.b))) / max(luma(base), 0.01) * 0.3);
    o.albedo = mix(base, vec3f(0.16, 0.07, 0.035) * (0.7 + 0.6 * n2), rust * 0.8);
    o.metal = grey * (1.0 - rust) * 0.8;
    o.rough = mix(mix(0.38, 0.5, n2), 0.9, rust);
    weather = 0.3;
  } else {
    // Asphalt shingles, or slate.
    let slate = mat == 3u;
    let rh = select(0.14, 0.2, slate);
    let tw = select(0.33, 0.28, slate);
    let fine = pattern_fade(tw, fw.x) * pattern_fade(rh, fw.y);
    let row = floor(uv.y / rh);
    let tab = floor(uv.x / tw + row * 0.5 + hash1(row) * 0.3);
    let fy = fract(uv.y / rh);
    let fyw = fw.y / rh;
    let lap = 0.74 + 0.26 * smoothstep(-fyw, 0.22 + fyw, fy) - 0.12 * box(fy - 0.96, 0.04, fyw);
    let gap = 1.0 - 0.3 * box(fract(uv.x / tw + row * 0.5 + hash1(row) * 0.3) - 0.5, 0.035, fw.x / tw);
    pat = mix(0.95, lap * gap * (0.86 + 0.28 * hash2(vec2f(tab, row) + seed)), fine);
    o.rough = select(0.92, 0.55, slate);
    weather = select(1.0, 0.5, slate);
  }
  // Seen from further off: granules and tiles from different batches,
  // broad sun-bleaching, and dark streaks running down from the ridge where
  // rain and algae go.
  let mott = mnz(uv, 1.3, 1.9, sd * 1.7, fwm);
  let patchy = mnz(uv, 4.3, 2.7, sd.yx * 1.3, fwm);
  let blot = mnoise(uv, 11.0, 0.6, sd, fwm);
  let sq = vec2f(uv.x / 1.6, uv.y / 22.0) + sd * 0.37;
  let streak = wave_a(sq, max(fw.x / 1.6, fw.y / 22.0) * 256.0);
  var c = o.albedo * pat * (1.0 + 0.07 * mott + 0.07 * patchy + 0.5 * (blot - 0.5));
  let algae = smoothstep(0.48, 0.72, streak) * smoothstep(0.35, 0.6, blot + 0.2) * weather * (0.3 + 0.6 * (1.0 - wealth));
  c = mix(c, c * vec3f(0.55, 0.58, 0.56), algae * 0.65);
  // Ridge cap: a strip of capping along the top, a shade darker.
  if (ridge < -0.2) {
    let d = uv.y - ridge;
    let cap = box(d - 0.12, 0.16, fw.y) * clamp(1.0 - fw.y / 0.3, 0.0, 1.0);
    let ct = select(1.0, 0.82 + 0.18 * step(0.5, fract(uv.x / 0.3)) * pattern_fade(0.3, fw.x), mat == 0u || mat == 3u);
    c = mix(c, o.albedo * select(0.78, 1.08, mat == 2u || mat == 4u) * ct, cap);
    o.bend *= 1.0 - cap;
  }
  o.albedo = c;
  return o;
}

// Flat roofs (mat: F_* in buildings.rs). uv: metres in the building's frame.
fn flat_roof(uv: vec2f, base: vec3f, mat: u32, seed: f32, wealth: f32, fw_in: vec2f) -> RoofOut {
  let fw = max(fw_in, vec2f(0.002));
  let fwm = max(fw.x, fw.y);
  let sd = vec2f(seed * 13.7, seed * 5.3);
  var o = RoofOut(base, 0.0, 0.04, 0.85, 0.0);
  // Soiling and wear at scales that never line up with each other.
  let big = mnoise(uv, 23.0, 0.5 + seed * 6.0, sd, fwm);
  let mid = mnoise(uv, 5.3, 2.1 + seed * 6.0, sd.yx, fwm);
  let dirt = (big - 0.5) * 1.5 + (mid - 0.5) * 1.1 + mnz(uv, 2.1, 4.0 + seed, sd * 0.7, fwm) * 0.12 + mnz(uv, 0.7, 5.2, sd * 1.3, fwm) * 0.06;
  let worn = 1.0 - wealth;
  var c = base;
  if (mat == 0u) {
    // Single-ply membrane: sheets 1.83 m wide with welded laps, end laps
    // staggered sheet to sheet, grey-brown where water stands.
    let w = 1.83;
    let x = (fract(uv.x / w + 0.5) - 0.5) * w;
    let sheet = floor(uv.x / w + 0.5);
    let ey = (fract(uv.y / 12.0 + hash1(sheet + seed * 31.0)) - 0.5) * 12.0;
    let fine = pattern_fade(w, fwm);
    let lapw = box(x - 0.04, 0.04, fw.x) + box(ey - 0.04, 0.04, fw.y);
    let edge = box(x - 0.085, 0.006, fw.x) * pattern_fade(0.15, fw.x);
    c = base * (1.0 + (0.08 * lapw - 0.3 * edge) * fine) * (1.0 + dirt * mix(0.24, 0.45, worn));
    let pond = smoothstep(0.6, 0.7, big * 0.7 + mid * 0.3);
    c = mix(c, c * vec3f(0.76, 0.72, 0.64), pond * (0.25 + 0.5 * worn));
    o.rough = 0.42;
  } else if (mat == 1u) {
    // Bitumen cap sheet: rolls about a metre wide with darker laps, granules,
    // tar run along cracks, and newer patches.
    let w = 0.94;
    let y = fract(uv.y / w) * w;
    let roll = floor(uv.y / w);
    let fine = pattern_fade(w, fwm);
    let lap = box(y - 0.05, 0.05, fw.y);
    c = base * mix(1.0, (1.0 - 0.28 * lap) * (0.92 + 0.16 * hash1(roll + seed * 7.0)), fine);
    c *= 1.0 + dirt * 0.3 + (mnoise(uv, 0.31, 0.8, sd, fwm) - 0.5) * 0.3;
    let fresh = smoothstep(0.7, 0.72, mid) * step(0.5, hash1(floor(big * 6.0) + seed));
    c *= 1.0 + 0.22 * fresh;
    let tn = mnoise(uv, 3.7, 0.9, sd + 0.5, fwm);
    let aa = max(0.012, fwm / 3.7 * 1.5);
    let tar = (1.0 - smoothstep(0.0, aa, abs(tn - 0.5))) * (0.012 / aa) * smoothstep(0.45, 0.6, big) * (0.4 + 0.8 * worn);
    c = mix(c, vec3f(0.02), clamp(tar, 0.0, 1.0) * 0.9);
    o.rough = mix(0.9, 0.5, tar);
  } else if (mat == 2u) {
    // Gravel ballast: speckled stone, swept thin in places to the black membrane.
    let sp = mnoise(uv, 0.37, 1.3, sd, fwm);
    let sp2 = mnoise(uv, 0.09, 0.4, sd.yx, fwm);
    c = base * (0.78 + 0.3 * sp + 0.35 * (sp2 - 0.5)) * (1.0 + dirt * 0.3);
    let bare = smoothstep(0.62, 0.74, mid * 0.6 + big * 0.4) * (0.3 + 0.7 * worn);
    c = mix(c, vec3f(0.03), bare * 0.7);
    o.rough = 1.0;
  } else {
    // Pavers on pedestals: open joints, each slab its own shade, stains.
    let g = uv / 0.6;
    let f = fract(g) - 0.5;
    let fine = pattern_fade(0.6, fwm);
    let joint = 1.0 - box(f.x, 0.48, fw.x / 0.6) * box(f.y, 0.48, fw.y / 0.6);
    c = base * mix(0.96, (0.9 + 0.2 * hash2(floor(g) + seed * 7.0)) * (1.0 - 0.7 * joint), fine) * (1.0 + dirt * 0.28);
    o.rough = 0.8;
  }
  o.albedo = c;
  return o;
}

// Sun glint off a roof: GGX, specular colour f0.
fn roof_glint(N: vec3f, V: vec3f, f0: vec3f, rough: f32) -> vec3f {
  let L = G.sun_dir.xyz;
  let H = normalize(L - V);
  let ndl = max(dot(N, L), 0.0);
  let ndv = max(dot(N, -V), 1e-3);
  let ndh = max(dot(N, H), 0.0);
  let a = rough * rough;
  let a2 = a * a;
  let d = ndh * ndh * (a2 - 1.0) + 1.0;
  let D = a2 / (PI * d * d);
  let k = a * 0.5;
  let vis = 0.25 / ((ndl * (1.0 - k) + k) * (ndv * (1.0 - k) + k));
  let F = f0 + (1.0 - f0) * pow(1.0 - max(dot(H, -V), 0.0), 5.0);
  return F * D * vis * ndl;
}

// False for the pipeline that draws what cannot be dissolving: level 0 runs
// of a tile done fading in, with no 3D tiles zone (render.rs `solids`). With
// no discard left in it the GPU's hidden-surface removal works on it, and
// the facades behind it go unshaded (FABLE_OPT_OFF=build_solid).
override DISSOLVE: bool = true;

@fragment fn fs(in: VOut) -> GOut {
  // Inside the 3D tiles zone the mesh stands in for this (tiles3d.rs).
  if (DISSOLVE && photo_hides(in.P, in.clip.xy)) { discard; }
  // Each level of detail dissolves in: over a second or so when it arrives,
  // and building by building with distance as the camera approaches.
  let dcam = lod_dist(in.P, in.lod);
  let lod = in.lod & 7u;
  var fade = T.lod.z;
  // Full geometry and far stand-in take complementary pixels of the same noise.
  let near = 1.0 - smoothstep(FAR0, FAR1, dcam);
  let n = ign(in.clip.xy);
  if (lod == 1u) { fade = min(min(fade, T.lod.x), near); }
  else if (lod == 2u) { fade = min(fade, T.lod.y) * (1.0 - smoothstep(LOD2_FAR * 0.6, LOD2_FAR * 0.95, dcam)); }
  else if (lod == 3u) { fade = min(fade, near); }
  else if (DISSOLVE && lod == 4u && n < near) { discard; }
  if (DISSOLVE && fade < 1.0 && n >= fade) { discard; }
  let V = normalize(in.P);
  var N = normalize(in.N);
  let kind = u32(in.col.a * 255.0 / 50.0 + 0.5);
  if (dot(N, V) > 0.0 && kind != 1u && kind != 2u) { N = -N; }
  let base = pow(in.col.rgb, vec3f(2.2));
  let seed = f32(in.sty.x) / 255.0;
  let wealth = f32(in.sty.z) / 255.0;
  let fw = fwidth(in.uv);
  let dPx = dpdx(in.P);
  let dPy = dpdy(in.P);
  let dUx = dpdx(in.uv);
  let dUy = dpdy(in.uv);
  let dist = length(in.P);
  let l = light_at(in.P);
  let night = night_at(l.upn);
  var albedo: vec3f;
  var extra = vec3f(0.0);
  var Nout = N;
  var mirror = 0.0;  // how much of this pixel is a mirror, for screen-space reflections
  if (kind == 0u) {
    // Facade frame: x along the wall (the way uv.x runs).
    var Tw = cross(l.upn, N);
    Tw = select(G.east.xyz, Tw / max(length(Tw), 1e-4), length(Tw) > 1e-3);
    if (dot(dPx, Tw) * dUx.x + dot(dPy, Tw) * dUy.x < 0.0) { Tw = -Tw; }
    let fs = facade_shade(in.P, V, N, Tw, in.uv, fw, base, in.sty.y & 15u, seed, wealth, f32(in.sty.w), false);
    albedo = fs.albedo;
    extra = fs.extra;
    Nout = fs.N;
    mirror = fs.mirror;
  } else if (kind == 1u || kind == 2u) {
    let img = textureSample(t_atlas, s_aniso, in.auv).rgb;
    var ro: RoofOut;
    var Tr = cross(l.upn, N);
    if (kind == 1u) {
      ro = flat_roof(in.uv, base, in.sty.y & 15u, seed, wealth, fw);
      // Helipad on some tall towers.
      if ((in.sty.y & 16u) != 0u) {
        let hp = in.uv;
        let r = length(hp);
        let ring = box(r - 7.0, 0.35, fw.x);
        let hx = box(hp.x, 2.2, fw.x) * box(hp.y, 0.35, fw.y) + (box(hp.x - 1.9, 0.35, fw.x) + box(hp.x + 1.9, 0.35, fw.x)) * box(hp.y, 2.6, fw.y);
        ro.albedo = mix(ro.albedo, vec3f(0.12, 0.13, 0.13), step(r, 9.0) * 0.8);
        ro.albedo = mix(ro.albedo, vec3f(0.75), clamp(ring + hx, 0.0, 1.0));
      }
    } else {
      // Where the ridge is down the slope: its height over the sine of the pitch.
      let sn = length(Tr);
      let ridge = select(0.0, -f32(in.sty.w) * 0.1 / max(sn, 1e-3), sn > 0.1 && in.sty.w > 0u);
      ro = pitched_roof(in.uv, base, in.sty.y & 15u, seed, wealth, fw, ridge);
    }
    // The satellite image, cleaned of canopy and shadow, says which parts are
    // lighter or darker; it is too soft and too loosely placed to say more
    // close up.
    ro.albedo *= mix(1.0, roof_image(img, base), mix(0.25, 0.7, smoothstep(120.0, 900.0, dist)));
    // Tiles and corrugations lean the normal across the slope.
    Tr = select(G.east.xyz, Tr / max(length(Tr), 1e-4), length(Tr) > 1e-3);
    if (dot(dPx, Tr) * dUx.x + dot(dPy, Tr) * dUy.x < 0.0) { Tr = -Tr; }
    N = normalize(N - Tr * ro.bend);
    let fspec = mix(vec3f(ro.f0), ro.albedo, ro.metal);
    let sh = shadow_at(in.P, N);
    extra = l.sun * roof_glint(N, V, fspec, ro.rough) * sh;
    // Smooth sheet mirrors the sky; rough coverings only scatter it, which the ambient term already has.
    let gloss = (1.0 - ro.rough) * (1.0 - ro.rough);
    let Fe = fspec + (1.0 - fspec) * pow(1.0 - max(dot(N, -V), 0.0), 5.0);
    extra += env_reflect(in.P, reflect(V, N), l.upn, false) * Fe * gloss;
    albedo = ro.albedo * (1.0 - 0.75 * ro.metal);
  } else if (kind == 4u) {
    let dk = in.sty.y | (in.sty.z << 8u);
    let rs = road_surface(in.uv.x, in.uv.y, f32(in.sty.x) * 0.25, dk, 1000.0, 0.0, max(fw.x, fw.y));
    albedo = mix(vec3f(0.28, 0.27, 0.25), rs.rgb, rs.a);
    extra = albedo * LAMP_COL * lamp_pool(in.uv.x, in.uv.y, f32(in.sty.x) * 0.25, dk) * night * 0.09 * G.sun_dir.w;
  } else if (kind == 5u) {
    let sub = in.sty.y & 15u;
    if (sub == 1u) {
      // Traffic signal: three lamps up each face, cycling green, amber, red.
      let idx = floor(in.uv.y / 0.35);
      let c2 = vec2f(in.uv.x - 0.18, fract(in.uv.y / 0.35) * 0.35 - 0.175);
      let px = max(max(fw.x, fw.y), 0.002);
      let disc = clamp((0.115 - length(c2)) / px + 0.5, 0.0, 1.0) * min(1.0, 0.23 / px);
      let ph = fract(G.time.x / 46.0 + f32(in.sty.x & 127u) / 128.0 + f32(in.sty.x >> 7u) * 0.5);
      let state = select(select(2.0, 1.0, ph < 0.5), 0.0, ph < 0.44);
      let lc = select(select(vec3f(1.0, 0.03, 0.02), vec3f(1.0, 0.45, 0.02), idx == 1.0), vec3f(0.05, 1.0, 0.35), idx == 0.0);
      albedo = mix(vec3f(0.012), lc * 0.05, disc);
      extra = lc * disc * step(abs(idx - state), 0.5) * G.sun_dir.w * (0.12 + 0.2 * night);
    } else if (sub == 2u) {
      // Aircraft warning light on a tall roof: red, flashing.
      let blink = smoothstep(0.0, 0.05, fract(G.time.x * 0.75 + seed)) * (1.0 - smoothstep(0.25, 0.35, fract(G.time.x * 0.75 + seed)));
      albedo = vec3f(0.3, 0.02, 0.02);
      extra = vec3f(1.0, 0.05, 0.02) * blink * G.sun_dir.w * mix(0.05, 0.8, night);
    } else {
      // Street light: pale housing by day, glowing once it is dark.
      albedo = vec3f(0.45);
      extra = base * night * 0.5 * G.sun_dir.w;
    }
  } else {
    // Rooftop equipment and other fittings, by material.
    let sub = in.sty.y & 15u;
    let n = textureSample(t_wave, s_repeat, in.uv / 1.7 + seed).a;
    albedo = base * (0.85 + 0.3 * n);
    let top = dot(N, l.upn) > 0.7;
    let fwm = max(fw, vec2f(0.003));
    if (sub == 1u) {
      // Air handler: louvred sides, fans on top.
      if (top) {
        let c = (fract(in.uv / 1.6) - 0.5) * 1.6;
        let r = length(c);
        let fan = box(r - 0.4, 0.4, fwm.x) * (0.6 + 0.4 * step(0.5, fract(atan2(c.y, c.x) * 1.27 + G.time.x * 3.0)));
        albedo = mix(albedo, vec3f(0.05), clamp(fan, 0.0, 1.0) * 0.8);
      } else {
        albedo *= 0.75 + 0.25 * step(0.5, fract(in.uv.y / 0.09));
      }
    } else if (sub == 2u) {
      // Timber water tank: staves and steel hoops.
      albedo = vec3f(0.3, 0.2, 0.13) * (0.8 + 0.2 * step(0.5, fract(in.uv.x / 0.14))) * (0.85 + 0.3 * n);
      albedo = mix(albedo, vec3f(0.12, 0.1, 0.09), box(fract(in.uv.y / 0.9) - 0.5, 0.03, fwm.y / 0.9));
      if (top) { albedo = vec3f(0.18, 0.16, 0.14); }
    } else if (sub == 3u || sub == 4u) {
      // Solar panels and skylights: glossy, cells or panes in a frame.
      let cell = select(vec2f(1.0, 1.65), vec2f(0.9, 1.2), sub == 4u);
      let g = fract(in.uv / cell);
      let frame = 1.0 - box(g.x - 0.5, 0.46, fwm.x / cell.x) * box(g.y - 0.5, 0.46, fwm.y / cell.y);
      albedo = select(vec3f(0.02, 0.03, 0.06), vec3f(0.015), sub == 4u);
      albedo = mix(albedo, vec3f(0.5), frame);
      let R = reflect(V, N);
      let F = 0.05 + 0.95 * pow(1.0 - max(dot(-V, N), 0.0), 5.0);
      extra = env_reflect(in.P, R, l.upn, false) * F * (1.0 - frame);
      if (sub == 4u) { extra += vec3f(1.0, 0.8, 0.55) * 0.02 * G.sun_dir.w * night * step(0.5, seed) * (1.0 - frame); }
    } else if (sub == 5u) {
      // Stair and lift head: plain render with a steel door.
      let door = box(in.uv.x - 1.0, 0.45, fwm.x) * step(in.uv.y, 2.1);
      albedo = mix(albedo, vec3f(0.25, 0.26, 0.27), door * f32(!top));
    } else if (sub == 6u) {
      // Awning: striped canvas.
      let stripe = step(0.5, fract(in.uv.x / 0.5));
      albedo = mix(base, mix(base, vec3f(0.85, 0.83, 0.78), 0.8), stripe) * (0.9 + 0.1 * n);
    } else if (sub == 7u) {
      // Window air conditioner: a grille on the front.
      albedo = vec3f(0.6, 0.6, 0.58) * (0.8 + 0.2 * step(0.5, fract(in.uv.y / 0.05)));
    } else if (sub == 8u) {
      // Balcony: concrete slab, and a railing with glass or bars.
      albedo = base * (0.8 + 0.2 * n);
    }
  }
  albedo = with_snow(albedo, in.P, N);
  let c = shade(in.P, N, albedo) + extra;
  return GOut(vec4f(c, 1.0), vec4f(Nout, -mirror), amb_share(c));
}
