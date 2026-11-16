// Instanced trees and undergrowth: per-instance size, turn, tint and a fall
// rotation so they can be knocked over. Each model is a solid part (trunk,
// limbs, masses of foliage) and, close to the camera, sprays of leaves cut out
// of cards by a procedural mask. Past IMP_DIST trees are impostors (at the
// end of this file); the simpler models only cast shadows.
struct BTile { offset: vec4f, ax: vec4f, ay: vec4f };
@group(1) @binding(0) var<uniform> T: BTile;
@group(2) @binding(0) var<uniform> CS: Cascade;

// How far away leaf cards are drawn. Must match CARD_RANGE in trees.rs.
const CARD_RANGE = 150.0;
// Share of a tree's leaf sprays out from 100 m on, and in its impostor.
const SPRAY_FAR = 1.0;

struct VOut {
  @builtin(position) clip: vec4f,
  @location(0) P: vec3f,
  @location(1) N: vec3f,
  @location(2) col: vec4f,
  @location(3) L: vec3f,
  @location(4) mat: vec3f,   // kind of bark, openness to the sky, height up the tree
};
struct CVOut {
  @builtin(position) clip: vec4f,
  @location(0) P: vec3f,
  @location(1) N: vec3f,
  @location(2) col: vec4f,   // rgb, a = leaves along the spray
  @location(3) uv: vec4f,    // along, across, seed, style
  @location(4) aux: vec4f,   // aspect, two shape parameters, openness to the sky
  @location(5) grow: f32,
  @location(6) sun: vec3f,   // sunlight and skylight reaching the spray
  @location(7) amb: vec3f,
};
struct GOut { @location(0) c: vec4f, @location(1) n: vec4f, @location(2) amb: f32 };
struct COut { @location(0) c: vec4f, @location(1) n: vec4f, @location(2) amb: f32, @builtin(sample_mask) mask: u32 };
struct TW { P: vec3f, N: vec3f, L: vec3f };

// 1 where leaf cards are fully out, 0 beyond their range. Trees hand over to
// their impostors (pictures of exactly this, see below) before the cards
// start to go.
fn near_k(d: f32) -> f32 { return 1.0 - smoothstep(CARD_RANGE * 0.9, CARD_RANGE, d); }
// How far the outline of a crown is cut into clumps (vs_near / fs_cut).
const CUT_RANGE = CARD_RANGE;
fn cut_k(d: f32) -> f32 { return 1.0 - smoothstep(CUT_RANGE * 0.9, CUT_RANGE, d); }

// Distances at which a tree changes to its next simpler model, moved by up
// to LOD_JITTER for each tree. Must match LOD_DIST and LOD_JITTER in trees.rs.
const LOD_JITTER = 0.12;
// A tree's own factor on those distances. Must match `tree_jit` in trees.rs.
fn tree_jit(ia: vec4f) -> f32 {
  let h = (bitcast<u32>(ia.x) * 0x9E3779B9u) ^ (bitcast<u32>(ia.z) * 0x85EBCA6Bu);
  let h2 = (h ^ (h >> 15u)) * 0x2C1B3C6Du;
  return 1.0 + LOD_JITTER * (f32((h2 ^ (h2 >> 12u)) & 0xFFFFu) / 32767.5 - 1.0);
}
fn lod_of(d: f32, ia: vec4f) -> f32 {
  let dj = d / tree_jit(ia);
  return step(260.0, dj) + step(600.0, dj) + step(1200.0, dj) + step(2200.0, dj);
}
// Beyond this (moved by the tree's own jitter) a standing tree is drawn as an
// impostor. Must match IMP_DIST in trees.rs.
const IMP_DIST = 120.0;
// Whether a tree (not undergrowth, not a loose tree) is far enough away to be an impostor.
fn imp_far(ia: vec4f, far: f32) -> bool {
  return far >= 250.0 && T.offset.w < 0.5 && length(ia.xyz + T.offset.xyz) / tree_jit(ia) >= IMP_DIST;
}
// Whether this vertex belongs to a model other than the one the instance is
// drawn with. Undergrowth (far < 250) and loose trees (T.offset.w = 1) only
// have the one.
fn other_lod(lod: f32, ia: vec4f, far: f32) -> bool {
  return far >= 250.0 && T.offset.w < 0.5 && abs(lod_of(length(ia.xyz + T.offset.xyz), ia) - lod) > 0.5;
}

// far: distance the instance is drawn to (undergrowth), or >= 250 for a tree.
// flex: how much this vertex rustles on its own, and a phase for it.
fn tree_world(pos: vec3f, nrm: vec3f, leaf: f32, ia: vec4f, ib: vec4f, far: f32, flex: vec2f) -> TW {
  let base = ia.xyz + T.offset.xyz;
  let upn = normalize(base - G.planet.xyz);
  let e = normalize(cross(vec3f(0.0, 0.0, 1.0), upn));
  let n = cross(upn, e);
  let cy = cos(ib.z);
  let sy = sin(ib.z);
  // Shrink into the ground toward the edge of the drawing distance.
  let d = length(base);
  var fade = 1.0 - smoothstep(2700.0, 3200.0, d);
  if (far < 250.0) { fade = 1.0 - smoothstep(far * 0.7, far, d); }
  let sc = vec3f(ib.w, 1.0, ib.w) * ia.w * fade;
  var lp = vec3f(pos.x * cy - pos.z * sy, pos.y, pos.x * sy + pos.z * cy) * sc;
  var ln = normalize(vec3f(nrm.x * cy - nrm.z * sy, nrm.y, nrm.x * sy + nrm.z * cy) + vec3f(0.0, 1e-5, 0.0));
  let L = lp;
  // Wind: the whole tree leans a little, the canopy flutters.
  let th = length(ib.xy);
  if (th < 0.01) {
    let ph = ia.x * 0.07 + ia.z * 0.05;
    let t = G.time.x;
    let bend = lp.y * lp.y * 0.0009;
    lp.x += (sin(t * 1.1 + ph) + 0.5 * sin(t * 2.7 + ph * 1.7)) * bend;
    lp.z += cos(t * 0.9 + ph * 1.3) * bend;
    let fl = leaf * 0.05 * min(lp.y * 0.1, 1.0);
    lp += vec3f(sin(t * 3.1 + lp.y * 1.9 + ph), 0.0, cos(t * 2.6 + lp.x * 2.3 + ph)) * fl;
    let fs = flex.y * 40.0 + ph;
    lp += vec3f(sin(t * 2.3 + fs), 0.6 * sin(t * 3.4 + fs * 1.3), cos(t * 1.9 + fs * 0.7)) * flex.x;
  } else {
    // Fallen or falling: rotate about the base, away from whatever hit it.
    let d2 = ib.xy / th;
    let ax = vec3f(d2.y, 0.0, -d2.x);
    let c = cos(th);
    let s = sin(th);
    lp = lp * c + cross(ax, lp) * s + ax * dot(ax, lp) * (1.0 - c);
    ln = ln * c + cross(ax, ln) * s + ax * dot(ax, ln) * (1.0 - c);
  }
  var o: TW;
  o.P = base + e * lp.x + upn * lp.y + n * lp.z;
  o.N = e * ln.x + upn * ln.y + n * ln.z;
  o.L = L;
  return o;
}

fn solid_vertex(pos: vec3f, nrm: vec3f, col: vec4f, ia: vec4f, ib: vec4f, tint: vec4f, uv: vec4f, aux: vec4f) -> VOut {
  var o: VOut;
  o.clip = vec4f(2.0, 2.0, 0.0, 1.0);
  if (other_lod(uv.x, ia, tint.a * 255.0)) { return o; }
  let leaf = step(0.5, col.a);
  // Close up the foliage masses draw in to become the shaded inside of the
  // crown, and the leaf cards carry the outside.
  // Right next to the camera they draw in further still, out of its way.
  let d = length(ia.xyz + T.offset.xyz);
  let k = near_k(d) * (1.0 + 1.0 * (1.0 - smoothstep(12.0, 45.0, d))) * leaf;
  let w = tree_world(pos + aux.xyz * k, nrm, leaf, ia, ib, tint.a * 255.0, vec2f(0.0));
  // The tint is the canopy colour seen from above: bark only takes its brightness.
  let tn = mix(vec3f(dot(tint.rgb, vec3f(1.0 / 3.0))), tint.rgb, leaf);
  return VOut(G.view_proj * vec4f(w.P, 1.0), w.P, w.N, vec4f(col.rgb * tn * 2.0, leaf), w.L + ia.xyz * 0.37, vec3f(uv.w, aux.w, w.L.y));
}
@vertex fn vs(@location(0) pos: vec3f, @location(1) nrm: vec3f, @location(2) col: vec4f, @location(3) ia: vec4f, @location(4) ib: vec4f, @location(5) tint: vec4f, @location(6) uv: vec4f, @location(7) aux: vec4f) -> VOut {
  return solid_vertex(pos, nrm, col, ia, ib, tint, uv, aux);
}
// Trees close enough not to be impostors, and undergrowth.
@vertex fn vs_near(@location(0) pos: vec3f, @location(1) nrm: vec3f, @location(2) col: vec4f, @location(3) ia: vec4f, @location(4) ib: vec4f, @location(5) tint: vec4f, @location(6) uv: vec4f, @location(7) aux: vec4f) -> VOut {
  var o = solid_vertex(pos, nrm, col, ia, ib, tint, uv, aux);
  if (imp_far(ia, tint.a * 255.0)) { o.clip = vec4f(2.0, 2.0, 0.0, 1.0); }
  return o;
}
@vertex fn vs_shadow(@location(0) pos: vec3f, @location(1) nrm: vec3f, @location(2) col: vec4f, @location(3) ia: vec4f, @location(4) ib: vec4f, @location(5) tint: vec4f, @location(6) uv: vec4f) -> @builtin(position) vec4f {
  if (other_lod(uv.x, ia, tint.a * 255.0) || photo_shadow_gone(ia.xyz + T.offset.xyz)) { return vec4f(2.0, 2.0, 0.0, 1.0); }
  return CS.vp * vec4f(tree_world(pos, nrm, step(0.5, col.a), ia, ib, tint.a * 255.0, vec2f(0.0)).P, 1.0);
}
// Whole batches drawn with the one model they are given (the second cascade).
@vertex fn vs_shadow_fixed(@location(0) pos: vec3f, @location(1) nrm: vec3f, @location(2) col: vec4f, @location(3) ia: vec4f, @location(4) ib: vec4f, @location(5) tint: vec4f) -> @builtin(position) vec4f {
  // (a tree under the 3D tiles' mesh casts nothing: the mesh's own tree does)
  if (photo_shadow_gone(ia.xyz + T.offset.xyz)) { return vec4f(2.0, 2.0, 0.0, 1.0); }
  return CS.vp * vec4f(tree_world(pos, nrm, step(0.5, col.a), ia, ib, tint.a * 255.0, vec2f(0.0)).P, 1.0);
}

// The solid crown stands in for a cloud of leaves. Toward its outline it thins
// out in clumps, so the edge breaks up and sky shows through instead of a
// smooth hull. Returns the fraction of the pixel covered.
// `cl` is the clump noise, `fw` how much it changes across a pixel, `facing`
// how squarely the surface faces the eye.
fn crown_cover(cl: f32, fw: f32, facing: f32, amount: f32) -> f32 {
  // A few gaps right across the crown, opening up toward the outline.
  let cut = (0.12 + 0.88 * smoothstep(0.15, 0.9, 1.0 - facing)) * amount;
  let thr = 0.25 + 0.72 * cut;
  let sharp = clamp((cl - thr) / max(fw, 1e-4) + 0.5, 0.0, 1.0);
  // Once the clumps are smaller than a pixel only their average shows.
  return mix(sharp, 1.0 - 0.9 * cut, smoothstep(0.12, 0.45, fw)) * step(0.001, amount) + (1.0 - step(0.001, amount));
}
@vertex fn vs_card(@location(0) pos: vec3f, @location(1) nrm: vec3f, @location(2) col: vec4f, @location(3) ia: vec4f, @location(4) ib: vec4f, @location(5) tint: vec4f, @location(6) uv: vec4f, @location(7) aux: vec4f) -> CVOut {
  let far = tint.a * 255.0;
  // Sprays come out one by one as a tree gets closer; undergrowth grows as a whole.
  // Further off, the sprays are a little smaller (as baked into the impostors).
  let d = length(ia.xyz + T.offset.xyz);
  var grow = clamp((near_k(d) * mix(1.25, SPRAY_FAR, smoothstep(35.0, 100.0, d)) - uv.z) * 4.0, 0.0, 1.0);
  if (far < 250.0) { grow = 1.0; }
  if (imp_far(ia, far)) { grow = 0.0; }
  let w = tree_world(pos, nrm, 1.0, ia, ib, far, vec2f(0.05 * uv.x, uv.z));
  var clip = G.view_proj * vec4f(w.P, 1.0);
  if (grow <= 0.0) { clip = vec4f(2.0, 2.0, 0.0, 1.0); }
  // Light varies slowly across a spray, so it is worked out here rather than per pixel.
  let l = light_at(w.P);
  return CVOut(clip, w.P, w.N, vec4f(col.rgb * tint.rgb * 2.0, col.a), uv, aux, grow, l.sun, l.amb);
}

// ------------------------------------------------------------ foliage light
// Metres of crown between a point and the sun, read off the shadow map
// (-1 where no cascade that trees cast into covers the point).
fn sun_depth(P: vec3f) -> f32 {
  if (G.csm.w <= 0.0) { return -1.0; }
  let res = vec2f(textureDimensions(t_shadow));
  for (var c = 0; c < 2; c++) {
    let m = G.shadow_vp[c];
    let q = m * vec4f(P, 1.0);
    let uv = q.xy * vec2f(0.5, -0.5) + 0.5;
    if (min(min(uv.x, 1.0 - uv.x), min(uv.y, 1.0 - uv.y)) <= 0.002 || q.z <= 0.0 || q.z >= 1.0) { continue; }
    let z = textureLoad(t_shadow, vec2i(uv * res), c, 0);
    // Light-space depth per metre along the sun's rays.
    let per_m = length(vec3f(m[0][2], m[1][2], m[2][2]));
    return max(q.z - z, 0.0) / per_m;
  }
  return -1.0;
}

// Sun and sky on leaves. Leaves are thin: what the canopy shades it also lets
// a little light through, filtered green, and a leaf lit from behind glows.
// `ao` is openness to the sky, `dense` (0..1) how tightly packed the foliage
// is here.
fn foliage(P: vec3f, N: vec3f, V: vec3f, albedo_in: vec3f, ao: f32, dense: f32, l: Light, sh: f32, glint: f32) -> vec3f {
  let albedo = with_snow(albedo_in, P, N);
  let sun = G.sun_dir.xyz;
  let ndl = dot(N, sun);
  // Sunlight that has come through the crown instead of straight from the
  // sky: the further it travelled through leaves, the dimmer and greener.
  let depth = sun_depth(P);
  var through = vec3f(0.1, 0.16, 0.03) * ao;
  if (depth >= 0.0) { through = exp(-depth * (0.2 + 0.3 * dense) * vec3f(1.15, 0.6, 2.4)); }
  let arrive = mix(through, vec3f(1.0), sh);
  // The lit side: soft wrap, since the leaves face every which way.
  let wrap = max(ndl * 0.6 + 0.4, 0.0);
  // The far side: light passing through the leaf itself, strongest when
  // looking toward the sun.
  let vs = max(dot(V, sun), 0.0);
  let back = (0.22 + 0.78 * max(-ndl, 0.0)) * (0.18 + 1.5 * vs * vs * vs);
  let up = dot(N, l.upn);
  // Skylight: from above, and through the leaves overhead onto the underside
  // of the canopy (filtered green), which never goes black.
  let sky = (0.5 + 0.5 * up) + (0.5 - 0.5 * up) * 0.4 * vec3f(0.8, 1.1, 0.45);
  // Light bounced up off the grass and litter below.
  let mu = max(dot(l.upn, sun), 0.0);
  let bounce = vec3f(0.09, 0.11, 0.05) * (l.sun * mu * 0.8 / PI + l.amb) * (0.5 - 0.5 * up);
  // A waxy glint off the top of the leaf.
  let H = normalize(sun - V);
  let gloss = pow(max(dot(N, H), 0.0), 40.0) * glint * sh;
  // Leaves are shiny: at a glancing angle they take on the sky's sheen.
  let fres = pow(1.0 - abs(dot(N, V)), 5.0);
  let sheen = l.amb * (0.012 + 0.25 * fres) * ao * max(0.3 + 0.7 * up, 0.0);
  let moon = moon_light(l.upn, 0.0) * max(dot(N, G.moon.xyz) * 0.6 + 0.4, 0.0) * ao;
  g_amb = albedo * (l.amb * sky + bounce) * ao + sheen;
  return albedo * ((l.sun * arrive * (wrap + back * vec3f(1.15, 1.25, 0.4)) + moon) / PI + l.amb * sky * ao + bounce * ao + point_light(P, N)) + l.sun * gloss + sheen;
}

// One filtered tap of the finest shadow cascade that covers a point: leaf cards
// overlap many deep, so they cannot afford the full soft shadow.
fn shadow_fast(P: vec3f, N: vec3f) -> f32 {
  if (G.csm.w <= 0.0) { return 1.0; }
  for (var c = 0; c < 3; c++) {
    let tex = G.csm[c];
    let q = G.shadow_vp[c] * vec4f(P + N * tex * 1.8 + G.sun_dir.xyz * tex * 0.8, 1.0);
    let uv = q.xy * vec2f(0.5, -0.5) + 0.5;
    if (min(min(uv.x, 1.0 - uv.x), min(uv.y, 1.0 - uv.y)) <= 0.01 || q.z <= 0.0 || q.z >= 1.0) { continue; }
    return textureSampleCompareLevel(t_shadow, s_shadow, uv, c, q.z - 0.00003);
  }
  return 1.0;
}

struct Solid { c: vec4f, n: vec4f, cover: f32, amb: f32 };
fn solid_frag(in: VOut, cut: bool) -> Solid {
  let V = normalize(in.P);
  var N = normalize(in.N);
  let leaf = in.col.a;
  let dist = length(in.P);
  var albedo = in.col.rgb;
  let near = 1.0 - smoothstep(120.0, 600.0, dist);
  // Leaf clumps, as noise that sticks to the tree (worked out for bark too,
  // since the derivatives need every pixel).
  let n = textureSampleLevel(t_noise, s_repeat, in.L * 0.31, 0.0);
  let cl = n.g * 0.75 + n.r * 0.25;
  let dcl = vec2f(dpdx(cl), dpdy(cl));
  let fw = abs(dcl.x) + abs(dcl.y);
  // Bigger clumps, a metre or two across, which carry the crown's light and
  // shade from further away.
  let m = textureSampleLevel(t_noise, s_repeat, in.L * 0.09, 0.0);
  let cm = m.g * 0.7 + m.r * 0.3;
  let dcm = vec2f(dpdx(cm), dpdy(cm));
  let fwm = abs(dcm.x) + abs(dcm.y);
  let dpx = dpdx(in.P);
  let dpy = dpdy(in.P);
  if (leaf > 0.5) {
    var a = 1.0;
    if (cut) {
      var Ng = cross(dpx, dpy);
      Ng = Ng / max(length(Ng), 1e-12);
      let facing = abs(dot(normalize(N + Ng * sign(dot(Ng, N))), V));
      // Close up the leaf cards make the outline, so the crown under them thins less.
      a = crown_cover(cl, fw, facing, (1.0 - 0.4 * near_k(dist)) * cut_k(dist));
    }
    // Darker inside the canopy than at its sunlit surface, and darker still
    // when the camera is in there looking at the back of it.
    let inside = 1.0 - 0.8 * smoothstep(0.1, 0.5, dot(N, V));
    // Each clump is a bump that catches the light on top and is shaded below
    // (a surface gradient from screen derivatives, faded once clumps get small).
    let mid = 1.0 - smoothstep(0.1, 0.35, fwm);
    let dh = dcl * (0.45 * (1.0 - smoothstep(0.08, 0.3, fw))) + dcm * (1.3 * mid);
    let r1 = cross(dpy, N);
    let r2 = cross(N, dpx);
    let det = dot(dpx, r1);
    if (abs(det) > 1e-12) {
      N = normalize(abs(det) * N - sign(det) * (dh.x * r1 + dh.y * r2));
    }
    let n2 = textureSampleLevel(t_noise, s_repeat, in.L * 1.3, 0.0);
    let clump = mix(0.5, cl * 0.6 + n2.b * 0.4, near);
    albedo *= 0.65 + 0.7 * clump;
    // Under the leaf cards this is the inside of the crown.
    albedo *= 1.0 - 0.5 * near_k(dist);
    albedo *= inside;
    // Shaded deep in the gaps between the big clumps.
    let gap = mix(1.0, 0.6 + 0.4 * smoothstep(0.3, 0.7, cm), mid);
    let ao = in.mat.y * (0.55 + 0.45 * smoothstep(0.3, 0.75, clump)) * inside * gap;
    let c = foliage(in.P, N, V, albedo, ao, clump, light_at(in.P), shadow_at(in.P, N), 0.0);
    return Solid(vec4f(c, 1.0), vec4f(N, 0.0), a, amb_share(c));
  }
  // Bark: furrows running up the trunk.
  let nb = textureSampleLevel(t_noise, s_repeat, in.L * vec3f(1.7, 0.21, 1.7), 0.0);
  let detail = 1.0 - smoothstep(60.0, 300.0, dist);
  albedo *= mix(1.0, 0.5 + 1.0 * (nb.g * 0.6 + nb.b * 0.4), detail);
  if (in.mat.x > 2.5) {
    // Gum: smooth pale bark shedding in long strips, patches of fresh
    // orange-tan and old grey, and a rough dark stocking at the foot.
    let pk = textureSampleLevel(t_noise, s_repeat, in.L * vec3f(0.9, 0.16, 0.9), 0.0);
    albedo = mix(vec3f(0.36, 0.34, 0.30), vec3f(0.58, 0.55, 0.48), smoothstep(0.5, 0.56, pk.g));
    albedo = mix(albedo, vec3f(0.44, 0.30, 0.19), smoothstep(0.62, 0.66, pk.b) * 0.8);
    albedo = mix(albedo, vec3f(0.11, 0.09, 0.07) * (0.6 + 0.8 * nb.g), (1.0 - smoothstep(1.2, 2.6, in.mat.z + nb.r * 1.5)) * detail);
  } else if (in.mat.x > 1.5) {
    // Palm: ringed with old leaf scars.
    albedo *= mix(1.0, 0.72 + 0.4 * smoothstep(0.3, 0.7, fract(in.L.y * 3.1)), detail);
  } else if (in.mat.x > 0.5) {
    // Birch: pale bark slashed with dark marks.
    let mk = textureSampleLevel(t_noise, s_repeat, in.L * vec3f(0.8, 3.3, 0.8), 0.0).g;
    albedo = mix(albedo, vec3f(0.03, 0.028, 0.025), smoothstep(0.6, 0.68, mk) * detail);
  }
  let l = light_at(in.P);
  let ndl = max(dot(N, G.sun_dir.xyz), 0.0);
  let sky = 0.5 + 0.5 * dot(N, l.upn);
  g_amb = albedo * l.amb * sky * in.mat.y;
  let c = albedo * (l.sun * ndl * shadow_at(in.P, N) / PI + l.amb * sky * in.mat.y + point_light(in.P, N));
  return Solid(vec4f(c, 1.0), vec4f(N, 0.0), 1.0, amb_share(c));
}
@fragment fn fs(in: VOut) -> GOut {
  // Inside the 3D tiles zone the mesh stands in for this (tiles3d.rs).
  if (photo_hides(in.P, in.clip.xy)) { discard; }
  let o = solid_frag(in, false);
  return GOut(o.c, o.n, o.amb);
}
// Spreads the crown's coverage over the multisamples (alpha itself has to stay
// 1 for the composite). Writing the sample mask costs the early depth test, so
// only trees close to the camera use this.
@fragment fn fs_cut(in: VOut) -> COut {
  if (photo_hides(in.P, in.clip.xy)) { discard; }
  let o = solid_frag(in, true);
  let cnt = min(u32(o.cover * 4.0 + ign(in.clip.xy)), 4u);
  if (cnt == 0u) { discard; }
  let rot = u32(ign(in.clip.yx + 31.0) * 4.0) & 3u;
  let bits = (1u << cnt) - 1u;
  return COut(o.c, o.n, o.amb, ((bits << rot) | (bits >> (4u - rot))) & 15u);
}

// --------------------------------------------------------------- leaf cards
// Card space: x runs 0..1 along the spray, y across it in the same units, so
// the card spans |y| < aspect. Each mask returns how far inside the nearest
// leaf a point is (in card lengths, negative outside), a random tone for that
// leaf (-1 for the twig), how far from its midrib (0..1), and how far inside
// the outline of the whole spray (for when the leaves are too small to see).

// Outline of a whole spray: narrow where it leaves the branch and at the tip.
fn spray_env(x: f32, aspect: f32) -> f32 {
  return aspect * sqrt(max(sin(PI * clamp(0.06 + 0.9 * x, 0.0, 1.0)), 0.0));
}
// Leaves scattered over a lattice, one to a cell and free to overlap their
// neighbours, each turned away from the twig that runs down the middle.
fn sprig(p: vec2f, n: f32, aspect: f32, width: f32, lobes: f32, seed: f32, grow: f32) -> vec4f {
  let c = p * n;
  let cell = floor(c);
  var o = vec3f(-1.0, 0.0, 0.0);
  for (var j = -1; j <= 1; j++) { for (var i = -1; i <= 1; i++) {
    let id = cell + vec2f(f32(i), f32(j));
    let h1 = hash2(id + vec2f(seed * 61.7, seed * 23.3));
    let h2 = fract(h1 * 57.31 + 0.37);
    let h3 = fract(h1 * 113.7 + 0.71);
    let ctr = id + 0.5 + (vec2f(h2, h3) - 0.5) * 0.6;
    let cc = ctr / n;
    if (abs(cc.y) + 0.3 / n > spray_env(cc.x, aspect) || fract(h1 * 29.3) > 0.9) { continue; }
    let th = sign(cc.y) * (0.25 + 0.65 * h1) + (h2 - 0.5) * 0.6;
    let dir = vec2f(cos(th), sin(th));
    let len = max((0.95 + 0.45 * h2) * grow, 0.001);
    let f = c - ctr;
    let a = dot(f, dir) / len + 0.5;
    let b = (f.y * dir.x - f.x * dir.y) / len;
    // Pointed at both ends, a little fuller toward the stalk; lobed for oaks and maples.
    let prof = 4.0 * a * (1.0 - a) * (1.0 - lobes * abs(sin(a * 12.0 + h1 * 6.0)));
    let half = 0.5 * width * prof * mix(1.2, 0.8, a);
    let d = (half - abs(b)) * len / n;
    if (d > o.x) { o = vec3f(d, h1, abs(b) / max(half, 0.0001)); }
  } }
  let twig = 0.006 * (1.1 - p.x) - abs(p.y);
  if (twig > o.x && p.x < 0.85) { o = vec3f(twig, -1.0, 0.0); }
  return vec4f(o, spray_env(p.x, aspect) * grow - abs(p.y));
}
// Needles or leaflets off both sides of a stem: conifer boughs, fern and palm
// fronds. `duty` is how much of the row they fill, `slant` how far they sweep
// forward.
fn pinnate(p: vec2f, n_in: f32, aspect: f32, duty: f32, slant: f32, grow: f32, px: f32) -> vec4f {
  // Needles too fine to resolve are drawn as fewer, fatter ones.
  let n = max(n_in / exp2(max(ceil(log2(px * n_in / 0.3)), 0.0)), 5.0);
  let ay = abs(p.y);
  let xb = p.x - ay * slant;   // where on the stem this needle is attached
  let env = aspect * pow(max(sin(PI * clamp(0.04 + 0.94 * xb, 0.0, 1.0)), 0.0), 0.6) * grow;
  let s = xb * n + step(0.0, p.y) * 0.5;
  let f = abs(fract(s) - 0.5);
  let taper = clamp(1.0 - ay / max(env, 0.0001), 0.0, 1.0);
  let needle = min((duty * 0.5 * (0.3 + 0.7 * taper) - f) * 0.8 / n, env - ay);
  let stem = 0.006 - ay;
  var o = vec3f(needle, hash1(floor(s) + 3.0), f * 2.0);
  if (stem > needle && p.x < 0.97) { o = vec3f(stem, -1.0, 0.0); }
  return vec4f(o, env - ay);
}
// A fan of grass blades. u runs up the blades, v across the card (-1..1).
fn blades(uv: vec2f, seed: f32, grow: f32) -> vec4f {
  var o = vec3f(-1.0, 0.0, 0.0);
  for (var j = 0; j < 6; j++) {
    let h = hash2(vec2f(f32(j), seed * 31.7));
    let h2 = fract(h * 57.31 + 0.37);
    let x0 = (f32(j) - 2.5) * 0.14 + (h - 0.5) * 0.1;
    let t = uv.x / max((0.5 + 0.5 * h2) * grow, 0.001);
    let xc = x0 + ((h - 0.5) * 1.1 + x0 * 1.2) * t * t;
    let w = 0.085 * (1.0 - t * t);
    let d = (w - abs(uv.y - xc)) * step(t, 1.0) - step(1.0, t);
    if (d > o.x) { o = vec3f(d, h2, abs(uv.y - xc) / max(w, 0.0001)); }
  }
  return vec4f(o.x * 0.5, o.yz, -1.0);
}

@fragment fn fs_card(in: CVOut) -> COut {
  if (photo_hides(in.P, in.clip.xy)) { discard; }
  let aspect = in.aux.x;
  let p = vec2f(in.uv.x, in.uv.y * aspect);
  // Size of a pixel in card lengths, and the card's true facing.
  let px = max(length(vec2f(dpdx(p.x), dpdy(p.x))), length(vec2f(dpdx(p.y), dpdy(p.y)))) + 1e-6;
  var Ng = normalize(cross(dpdx(in.P), dpdy(in.P)));
  var m: vec4f;
  if (in.uv.w < 0.5) { m = sprig(p, in.col.a, aspect, in.aux.y, in.aux.z, in.uv.z, in.grow); }
  else if (in.uv.w < 1.5) { m = pinnate(p, in.col.a, aspect, in.aux.y, in.aux.z, in.grow, px); }
  else { m = blades(in.uv.xy, in.uv.z, in.grow); }
  // Coverage: the leaves themselves, or once they are smaller than a pixel,
  // the outline of the spray as a whole.
  let tiny = smoothstep(0.7, 1.3, px * in.col.a) * step(in.uv.w, 0.5) + smoothstep(0.1, 0.2, px) * step(0.5, in.uv.w);
  let a = clamp(mix(m.x, m.w, tiny) / px + 0.5, 0.0, 1.0);
  // Spread it over the multisamples (alpha itself has to stay 1 for the composite).
  let cnt = min(u32(a * 4.0 + ign(in.clip.xy)), 4u);
  if (cnt == 0u) { discard; }
  let rot = u32(ign(in.clip.yx + 31.0) * 4.0) & 3u;
  let bits = (1u << cnt) - 1u;
  let mask = ((bits << rot) | (bits >> (4u - rot))) & 15u;

  let V = normalize(in.P);
  if (dot(Ng, V) > 0.0) { Ng = -Ng; }
  var albedo = in.col.rgb;
  let tone = m.y;
  if (tone < 0.0) {
    albedo = vec3f(0.035, 0.03, 0.024);
  } else {
    // Every leaf its own shade, a few of them yellowing; paler along the midrib.
    albedo *= 0.72 + 0.56 * tone;
    albedo = mix(albedo, albedo * vec3f(1.35, 1.1, 0.5), smoothstep(0.9, 1.0, tone) * 0.4);
    albedo *= 0.88 + 0.2 * smoothstep(0.0, 0.5, m.z);
  }
  // Shade mostly as part of the crown, a little as the leaf's own tilted face.
  let tilt = vec3f(tone, fract(tone * 7.13), fract(tone * 13.7)) - 0.5;
  let N = normalize(normalize(in.N) * 0.7 + Ng * 0.3 + tilt * 0.35);
  let l = Light(in.sun, in.amb, normalize(in.P - G.planet.xyz), 0.0);
  let c = foliage(in.P, N, V, albedo * (1.0 - 0.2 * tiny), in.aux.w, 0.35, l, shadow_fast(in.P, N), 0.006);
  return COut(vec4f(c, 1.0), vec4f(N, 0.0), amb_share(c), mask);
}

// ---------------------------------------------------------------- impostors
// Beyond IMP_DIST a tree is one camera-facing quad showing pictures of its
// full model (solid crown and leaf cards) taken at start-up from IMP_N x IMP_N
// directions over the upper hemisphere (impostor.rs). The three pictures
// nearest the view are blended and lit like the real tree from the albedo,
// normal, depth and openness to the sky stored for every pixel.
const IMP_N = 8u;          // must match N in impostor.rs
const IMP_CELL = 128.0;    // must match CELL in impostor.rs
const IMP_FRAMES = 64u;    // IMP_N * IMP_N
const IMP_PAD = 0.94;      // part of a picture's width the bounding sphere fills
struct ImpBounds { b: array<vec4f, 40> };   // per model: centre, radius
@group(2) @binding(1) var<uniform> IB: ImpBounds;
@group(2) @binding(2) var t_imp_a: texture_2d_array<f32>;   // albedo, coverage
@group(2) @binding(3) var t_imp_n: texture_2d_array<f32>;   // normal (picture x, y), depth, openness

// Direction to the camera (y up, y >= 0) <-> position on the hemi-octahedral grid (-1..1).
fn hemi_enc(d: vec3f) -> vec2f {
  let p = d.xz / (abs(d.x) + abs(d.y) + abs(d.z));
  return vec2f(p.x + p.y, p.x - p.y);
}
struct Frame { F: vec3f, r: vec3f, u: vec3f };
// A picture's view: F toward the camera, r and u its right and up.
fn frame_from(F: vec3f) -> Frame {
  var u = vec3f(0.0, 1.0, 0.0) - F * F.y;
  if (dot(u, u) < 1e-6) { u = vec3f(0.0, 0.0, -1.0); }
  u = normalize(u);
  return Frame(F, cross(u, F), u);
}
fn frame_of(c: vec2u) -> Frame {
  let o = vec2f(c) / f32(IMP_N - 1u) * 2.0 - 1.0;
  let p = vec2f(o.x + o.y, o.x - o.y) * 0.5;
  return frame_from(normalize(vec3f(p.x, 1.0 - abs(p.x) - abs(p.y), p.y)));
}

// ---- baking: model space, orthographic, one picture per viewport
struct BOut {
  @builtin(position) clip: vec4f,
  @location(0) P: vec3f,
  @location(1) N: vec3f,
  @location(2) col: vec4f,
  @location(3) uv: vec4f,
  @location(4) aux: vec4f,
  @location(5) @interpolate(flat) F: vec3f,
  @location(6) @interpolate(flat) layer: u32,
  @location(7) grow: f32,
};
struct BakeOut { @location(0) a: vec4f, @location(1) n: vec4f, @builtin(sample_mask) mask: u32 };

fn bake_vertex(p: vec3f, nrm: vec3f, col: vec4f, uv: vec4f, aux: vec4f, ii: u32, grow: f32) -> BOut {
  let layer = ii / IMP_FRAMES;
  let f = ii % IMP_FRAMES;
  let fr = frame_of(vec2u(f % IMP_N, f / IMP_N));
  let b = IB.b[layer];
  let q = p - b.xyz;
  let s = b.w / IMP_PAD;
  var clip = vec4f(dot(q, fr.r) / s, dot(q, fr.u) / s, 0.5 + 0.5 * dot(q, fr.F) / s, 1.0);
  if (grow <= 0.0) { clip = vec4f(2.0, 2.0, 0.0, 1.0); }
  return BOut(clip, p, nrm, col, uv, aux, fr.F, layer, grow);
}
@vertex fn vs_bake(@location(0) pos: vec3f, @location(1) nrm: vec3f, @location(2) col: vec4f, @location(6) uv: vec4f, @location(7) aux: vec4f, @builtin(instance_index) ii: u32) -> BOut {
  // As up close: the leaf masses drawn in, the leaf cards carrying the outside.
  return bake_vertex(pos + aux.xyz * step(0.5, col.a), nrm, col, uv, aux, ii, 1.0);
}
@vertex fn vs_bake_card(@location(0) pos: vec3f, @location(1) nrm: vec3f, @location(2) col: vec4f, @location(6) uv: vec4f, @location(7) aux: vec4f, @builtin(instance_index) ii: u32) -> BOut {
  // The sprays a tree has out just before it hands over (vs_card).
  return bake_vertex(pos, nrm, col, uv, aux, ii, clamp((SPRAY_FAR - uv.z) * 4.0, 0.0, 1.0));
}
fn bake_out(in: BOut, albedo: vec3f, N_in: vec3f, ao: f32, cover: f32) -> BakeOut {
  let cnt = min(u32(cover * 4.0 + ign(in.clip.xy)), 4u);
  if (cnt == 0u) { discard; }
  let rot = u32(ign(in.clip.yx + 31.0) * 4.0) & 3u;
  let bits = (1u << cnt) - 1u;
  let fr = frame_from(in.F);
  let N = normalize(N_in);
  let b = IB.b[in.layer];
  let depth = clamp(0.5 + 0.5 * dot(in.P - b.xyz, fr.F) / b.w, 0.0, 1.0);
  let nv = vec2f(dot(N, fr.r), dot(N, fr.u)) * 0.5 + 0.5;
  return BakeOut(vec4f(albedo, 1.0), vec4f(nv, depth, ao), ((bits << rot) | (bits >> (4u - rot))) & 15u);
}
// The solid model as solid_frag shades it close up, without the light.
// Openness is kept above 0.04 on foliage: 0 marks bark.
@fragment fn fs_bake(in: BOut) -> BakeOut {
  let V = -in.F;
  var N = normalize(in.N);
  var albedo = in.col.rgb;
  let L = in.P + vec3f(f32(in.layer) * 7.31, 0.0, f32(in.layer) * 3.17);
  let n = textureSampleLevel(t_noise, s_repeat, L * 0.31, 0.0);
  let cl = n.g * 0.75 + n.r * 0.25;
  let dcl = vec2f(dpdx(cl), dpdy(cl));
  let fw = abs(dcl.x) + abs(dcl.y);
  let m = textureSampleLevel(t_noise, s_repeat, L * 0.09, 0.0);
  let cm = m.g * 0.7 + m.r * 0.3;
  let dcm = vec2f(dpdx(cm), dpdy(cm));
  let fwm = abs(dcm.x) + abs(dcm.y);
  let dpx = dpdx(in.P);
  let dpy = dpdy(in.P);
  if (in.col.a > 0.5) {
    var Ng = cross(dpx, dpy);
    Ng = Ng / max(length(Ng), 1e-12);
    let facing = abs(dot(normalize(N + Ng * sign(dot(Ng, N))), V));
    let a = crown_cover(cl, fw, facing, 0.6);
    let inside = 1.0 - 0.8 * smoothstep(0.1, 0.5, dot(N, V));
    let mid = 1.0 - smoothstep(0.1, 0.35, fwm);
    let dh = dcl * (0.45 * (1.0 - smoothstep(0.08, 0.3, fw))) + dcm * (1.3 * mid);
    let r1 = cross(dpy, N);
    let r2 = cross(N, dpx);
    let det = dot(dpx, r1);
    if (abs(det) > 1e-12) {
      N = normalize(abs(det) * N - sign(det) * (dh.x * r1 + dh.y * r2));
    }
    let n2 = textureSampleLevel(t_noise, s_repeat, L * 1.3, 0.0);
    let clump = cl * 0.6 + n2.b * 0.4;
    albedo *= (0.65 + 0.7 * clump) * 0.5 * inside;
    let gap = mix(1.0, 0.6 + 0.4 * smoothstep(0.3, 0.7, cm), mid);
    let ao = in.aux.w * (0.55 + 0.45 * smoothstep(0.3, 0.75, clump)) * inside * gap;
    return bake_out(in, albedo, N, max(ao, 0.04), a);
  }
  let nb = textureSampleLevel(t_noise, s_repeat, L * vec3f(1.7, 0.21, 1.7), 0.0);
  albedo *= 0.5 + 1.0 * (nb.g * 0.6 + nb.b * 0.4);
  let style = in.uv.w;
  if (style > 2.5) {
    let pk = textureSampleLevel(t_noise, s_repeat, L * vec3f(0.9, 0.16, 0.9), 0.0);
    albedo = mix(vec3f(0.36, 0.34, 0.30), vec3f(0.58, 0.55, 0.48), smoothstep(0.5, 0.56, pk.g));
    albedo = mix(albedo, vec3f(0.44, 0.30, 0.19), smoothstep(0.62, 0.66, pk.b) * 0.8);
    albedo = mix(albedo, vec3f(0.11, 0.09, 0.07) * (0.6 + 0.8 * nb.g), 1.0 - smoothstep(1.2, 2.6, in.P.y + nb.r * 1.5));
  } else if (style > 1.5) {
    albedo *= 0.72 + 0.4 * smoothstep(0.3, 0.7, fract(in.P.y * 3.1));
  } else if (style > 0.5) {
    let mk = textureSampleLevel(t_noise, s_repeat, L * vec3f(0.8, 3.3, 0.8), 0.0).g;
    albedo = mix(albedo, vec3f(0.03, 0.028, 0.025), smoothstep(0.6, 0.68, mk));
  }
  return bake_out(in, albedo, N, 0.0, 1.0);
}
// Leaf cards as fs_card cuts and colours them, without the light.
@fragment fn fs_bake_card(in: BOut) -> BakeOut {
  let aspect = in.aux.x;
  let p = vec2f(in.uv.x, in.uv.y * aspect);
  let px = max(length(vec2f(dpdx(p.x), dpdy(p.x))), length(vec2f(dpdx(p.y), dpdy(p.y)))) + 1e-6;
  var Ng = normalize(cross(dpdx(in.P), dpdy(in.P)));
  var m: vec4f;
  if (in.uv.w < 0.5) { m = sprig(p, in.col.a, aspect, in.aux.y, in.aux.z, in.uv.z, in.grow); }
  else if (in.uv.w < 1.5) { m = pinnate(p, in.col.a, aspect, in.aux.y, in.aux.z, in.grow, px); }
  else { m = blades(in.uv.xy, in.uv.z, in.grow); }
  let tiny = smoothstep(0.7, 1.3, px * in.col.a) * step(in.uv.w, 0.5) + smoothstep(0.1, 0.2, px) * step(0.5, in.uv.w);
  let a = clamp(mix(m.x, m.w, tiny) / px + 0.5, 0.0, 1.0);
  let V = -in.F;
  if (dot(Ng, V) > 0.0) { Ng = -Ng; }
  var albedo = in.col.rgb;
  let tone = m.y;
  if (tone < 0.0) {
    albedo = vec3f(0.035, 0.03, 0.024);
  } else {
    albedo *= 0.72 + 0.56 * tone;
    albedo = mix(albedo, albedo * vec3f(1.35, 1.1, 0.5), smoothstep(0.9, 1.0, tone) * 0.4);
    albedo *= 0.88 + 0.2 * smoothstep(0.0, 0.5, m.z);
  }
  let tilt = vec3f(tone, fract(tone * 7.13), fract(tone * 13.7)) - 0.5;
  let N = normalize(normalize(in.N) * 0.7 + Ng * 0.3 + tilt * 0.35);
  return bake_out(in, albedo * (1.0 - 0.2 * tiny), N, max(in.aux.w, 0.04), a);
}

// ---- drawing
struct IOut {
  @builtin(position) clip: vec4f,
  @location(0) Q: vec3f,                            // on the plane through the tree's centre
  @location(1) uva: vec4f,                          // where the pixel falls in pictures 0 and 1 (-1..1)
  @location(2) uvb: vec2f,                          // ... and in picture 2
  @location(3) @interpolate(flat) w: vec3f,         // their weights
  @location(4) @interpolate(flat) cells: vec4u,     // the three pictures, model
  @location(5) @interpolate(flat) ex: vec4f,        // model axes in the world; w = metres per unit of stored depth
  @location(6) @interpolate(flat) ey: vec3f,
  @location(7) @interpolate(flat) ez: vec3f,
  @location(8) @interpolate(flat) tint: vec4f,
  // How far a pixel's place in each picture moves per unit of stored depth
  // (-1 back .. 1 front of the bounding sphere), along the view ray.
  @location(9) @interpolate(flat) ka: vec4f,
  @location(10) @interpolate(flat) kb: vec2f,
  @location(11) @interpolate(flat) C: vec3f,          // centre of the bounding sphere
};
@vertex fn vs_imp(@location(3) ia: vec4f, @location(4) ib: vec4f, @location(5) tint: vec4f, @builtin(vertex_index) vi: u32) -> IOut {
  var o: IOut;
  o.clip = vec4f(2.0, 2.0, 0.0, 1.0);
  let base = ia.xyz + T.offset.xyz;
  let d = length(base);
  if (ia.w <= 0.0 || d / tree_jit(ia) < IMP_DIST) { return o; }
  let layer = vi / 4u;
  let corner = vec2f(f32(vi & 1u), f32((vi >> 1u) & 1u)) * 2.0 - 1.0;
  let b = IB.b[layer];
  // The tree's axes, as tree_world stands and turns it.
  let upn = normalize(base - G.planet.xyz);
  let e = normalize(cross(vec3f(0.0, 0.0, 1.0), upn));
  let nn = cross(upn, e);
  let cy = cos(ib.z);
  let sy = sin(ib.z);
  let ex = e * cy + nn * sy;
  let ez = nn * cy - e * sy;
  let ey = upn;
  let s = ia.w * (1.0 - smoothstep(2700.0, 3200.0, d));
  let sc = vec3f(ib.w, 1.0, ib.w) * s;
  let C = base + ex * (b.x * sc.x) + ey * (b.y * sc.y) + ez * (b.z * sc.z);
  let R = b.w * s * max(ib.w, 1.0);
  let dc = length(C);
  let V = C / dc;
  // Which pictures: the direction to the camera in the tree's own frame.
  var lv = -vec3f(dot(V, ex), dot(V, ey), dot(V, ez));
  lv.y = max(lv.y, 0.0);
  lv = normalize(lv + vec3f(0.0, 1e-4, 0.0));
  let g = (hemi_enc(lv) * 0.5 + 0.5) * f32(IMP_N - 1u);
  let c0 = min(vec2u(max(floor(g), vec2f(0.0))), vec2u(IMP_N - 2u));
  let fr = clamp(g - vec2f(c0), vec2f(0.0), vec2f(1.0));
  var cs = array<vec2u, 3>(c0, c0 + vec2u(1u, 0u), c0 + vec2u(0u, 1u));
  var w = vec3f(1.0 - fr.x - fr.y, fr.x, fr.y);
  if (fr.x + fr.y > 1.0) {
    cs[0] = c0 + vec2u(1u, 1u);
    w = vec3f(fr.x + fr.y - 1.0, 1.0 - fr.y, 1.0 - fr.x);
  }
  // The quad: square to the view, through the centre, covering the bounding sphere.
  var up = ey - V * dot(ey, V);
  if (dot(up, up) < 1e-4) { up = ez - V * dot(ez, V); }
  up = normalize(up);
  let rt = cross(V, up);
  let Q = C + (rt * corner.x + up * corner.y) * R;
  // Where the ray through this corner crosses each picture's plane, in model units.
  let q = Q - C;
  let qm = vec3f(dot(q, ex) / sc.x, dot(q, ey) / sc.y, dot(q, ez) / sc.z);
  let vm = vec3f(dot(V, ex) / sc.x, dot(V, ey) / sc.y, dot(V, ez) / sc.z);
  var uv: array<vec2f, 3>;
  var k: array<vec2f, 3>;
  let span = b.w / IMP_PAD;
  for (var i = 0; i < 3; i++) {
    let f = frame_of(cs[i]);
    var den = dot(vm, f.F);
    den = select(min(den, -1e-4), max(den, 1e-4), den > 0.0);
    let h = qm - vm * (dot(qm, f.F) / den);
    uv[i] = vec2f(dot(h, f.r), dot(h, f.u)) / span;
    k[i] = vec2f(dot(vm, f.r), dot(vm, f.u)) * (b.w / (den * span));
  }
  // Drawn half its radius in front of its centre, so slopes behind it do not cut into it.
  let push = (dc - 0.5 * R) / dc;
  o.clip = G.view_proj * vec4f(Q * push, 1.0);
  o.Q = Q;
  o.uva = vec4f(uv[0], uv[1]);
  o.uvb = uv[2];
  o.w = w;
  o.cells = vec4u(cs[0].y * IMP_N + cs[0].x, cs[1].y * IMP_N + cs[1].x, cs[2].y * IMP_N + cs[2].x, layer);
  o.ex = vec4f(ex, b.w * s);
  o.ey = ey;
  o.ez = ez;
  o.tint = tint;
  o.C = C;
  o.ka = vec4f(k[0], k[1]);
  o.kb = k[2];
  return o;
}
@fragment fn fs_imp(in: IOut) -> COut {
  if (photo_hides(in.Q, in.clip.xy)) { discard; }
  // Mip level from how big a picture is on screen, a little on the sharp
  // side (the temporal filter takes out the shimmer).
  let lod = clamp(log2(max(length(dpdx(in.uva.xy)), length(dpdy(in.uva.xy))) * 0.5 * IMP_CELL) - 0.5, 0.0, 6.0);
  // Keep the filter inside the picture: half a texel of the coarser level.
  let half = 0.5 * exp2(ceil(lod)) / IMP_CELL;
  // Mostly the nearest picture alone: blends of three are softer, since
  // leaves do not line up exactly between views. They cross-fade only near
  // the hand-over from one picture to the next.
  var ww = pow(in.w, vec3f(12.0));
  ww /= ww.x + ww.y + ww.z;
  let uvs = array<vec2f, 3>(in.uva.xy, in.uva.zw, in.uvb);
  let ks = array<vec2f, 3>(in.ka.xy, in.ka.zw, in.kb);
  let layer = in.cells.w;
  var acc = vec4f(0.0);
  var nm = vec3f(0.0);
  var dep = 0.0;
  var open = 0.0;
  for (var i = 0; i < 3; i++) {
    let wi = ww[i];
    if (wi < 0.002) { continue; }
    let cid = in.cells[i];
    let c = vec2u(cid % IMP_N, cid / IMP_N);
    // The picture was taken from a slightly different direction: find where
    // this pixel's ray meets the surface in it, from the depth stored there.
    var u = uvs[i];
    for (var it = 0; it < 2; it++) {
      let l = vec2f(0.5 + 0.5 * u.x, 0.5 - 0.5 * u.y);
      let a0 = (vec2f(c) + clamp(l, vec2f(half), vec2f(1.0 - half))) / f32(IMP_N);
      let A0 = textureSampleLevel(t_imp_a, s_clamp, a0, layer, lod + 1.0);
      if (A0.a < 0.05) { break; }
      let D0 = textureSampleLevel(t_imp_n, s_clamp, a0, layer, lod + 1.0).z / A0.a;
      u = uvs[i] + ks[i] * (D0 * 2.0 - 1.0);
    }
    let loc0 = vec2f(0.5 + 0.5 * u.x, 0.5 - 0.5 * u.y);
    if (any(loc0 < vec2f(0.0)) || any(loc0 > vec2f(1.0))) { continue; }
    let auv = (vec2f(c) + clamp(loc0, vec2f(half), vec2f(1.0 - half))) / f32(IMP_N);
    let A = textureSampleLevel(t_imp_a, s_clamp, auv, layer, lod);
    if (A.a < 0.002) { continue; }
    let Nn = textureSampleLevel(t_imp_n, s_clamp, auv, layer, lod);
    let f = frame_of(c);
    let nv = Nn.xy / A.a * 2.0 - 1.0;
    nm += (nv.x * f.r + nv.y * f.u + sqrt(max(1.0 - dot(nv, nv), 0.0)) * f.F) * (wi * A.a);
    acc += A * wi;
    dep += Nn.z * wi;
    open += Nn.w * wi;
  }
  let cover = acc.a;
  let cnt = min(u32(cover * 4.0 + ign(in.clip.xy)), 4u);
  if (cnt == 0u) { discard; }
  let rot = u32(ign(in.clip.yx + 31.0) * 4.0) & 3u;
  let bits = (1u << cnt) - 1u;
  let mask = ((bits << rot) | (bits >> (4u - rot))) & 15u;

  let inv = 1.0 / max(cover, 1e-4);
  let ao = open * inv;
  let leaf = smoothstep(0.015, 0.035, ao);
  let nmod = normalize(nm + vec3f(0.0, 1e-6, 0.0));
  var N = normalize(in.ex.xyz * nmod.x + in.ey * nmod.y + in.ez * nmod.z);
  let V = normalize(in.Q);
  let P = in.Q - V * ((dep * inv * 2.0 - 1.0) * in.ex.w);
  // Far off, where single leaves are lost, a crown shades as one rough ball.
  let ball = smoothstep(200.0, 700.0, length(in.Q)) * leaf;
  N = normalize(N + normalize(P - in.C + in.ey * (0.25 * in.ex.w)) * (0.8 * ball));
  let tn = mix(vec3f(dot(in.tint.rgb, vec3f(1.0 / 3.0))), in.tint.rgb, leaf);
  let albedo = acc.rgb * inv * tn * 2.0;
  let l = light_at(P);
  let sh = shadow_at(P, N);
  var c: vec3f;
  if (leaf > 0.5) {
    c = foliage(P, N, V, albedo, ao, 0.45, l, sh, 0.0);
  } else {
    let ndl = max(dot(N, G.sun_dir.xyz), 0.0);
    let sky = 0.5 + 0.5 * dot(N, l.upn);
    g_amb = albedo * l.amb * sky;
    c = albedo * (l.sun * ndl * sh / PI + l.amb * sky + point_light(P, N));
  }
  return COut(vec4f(c, 1.0), vec4f(N, 0.0), amb_share(c), mask);
}
