// Deferred resolve: sky, water (waves, reflections, refraction, sea bed), haze.
@group(1) @binding(0) var t_color: texture_2d<f32>;
@group(1) @binding(1) var t_aux: texture_2d<f32>;
@group(1) @binding(2) var t_depth: texture_depth_2d;
@group(1) @binding(3) var t_ao: texture_2d<f32>;
@group(1) @binding(4) var t_amb: texture_2d<f32>;
// Keep the original samples at water silhouettes: resolved material parameters
// mix water depth/class with foreground normals and cannot be shaded correctly.
@group(1) @binding(5) var t_color_ms: texture_multisampled_2d<f32>;
@group(1) @binding(6) var t_aux_ms: texture_multisampled_2d<f32>;
@group(1) @binding(7) var t_depth_ms: texture_depth_multisampled_2d;
@group(1) @binding(8) var t_amb_ms: texture_multisampled_2d<f32>;

// Wave texture lookup at the mip that matches the pixel footprint (fp, metres).
fn wtex(uv: vec2f, size: f32, fp: f32) -> vec4f {
  return textureSampleLevel(t_wave, s_repeat, uv, log2(max(fp * 256.0 / size, 1.0)));
}

fn ssr(P: vec3f, R: vec3f, jit: f32) -> vec4f {
  let dims = vec2f(textureDimensions(t_depth));
  var step_len = max(length(P) * 0.012, 0.4);
  var t = step_len * (0.5 + jit);
  var prev = 0.0;
  for (var i = 0; i < 30; i++) {
    let q = P + R * t;
    let clip = G.view_proj * vec4f(q, 1.0);
    if (clip.w <= 0.0) { break; }
    let ndc = clip.xyz / clip.w;
    let uv = ndc.xy * vec2f(0.5, -0.5) + 0.5;
    if (any(uv < vec2f(0.0)) || any(uv > vec2f(1.0))) { break; }
    let zs = textureLoad(t_depth, vec2i(uv * dims), 0);
    if (zs > ndc.z) {
      // Behind a surface: is it a thin enough crossing to be a real hit?
      let ds = G.misc.x / zs;
      let dr = G.misc.x / ndc.z;
      if (dr - ds > max(step_len * 2.5, 8.0)) { break; }
      var lo = prev;
      var hi = t;
      var huv = uv;
      for (var k = 0; k < 5; k++) {
        let m = (lo + hi) * 0.5;
        let c2 = G.view_proj * vec4f(P + R * m, 1.0);
        let n2 = c2.xyz / c2.w;
        huv = n2.xy * vec2f(0.5, -0.5) + 0.5;
        if (textureLoad(t_depth, vec2i(huv * dims), 0) > n2.z) { hi = m; } else { lo = m; }
      }
      let px = vec2i(huv * dims);
      let c = textureLoad(t_color, px, 0);
      // (any water share at all: its samples hold the unlit tint, not a colour)
      if (textureLoad(t_aux, px, 0).a > 0.004 || c.a < 0.99 || textureLoad(t_depth, px, 0) == 0.0) { break; }
      let e = min(min(huv.x, 1.0 - huv.x), min(huv.y, 1.0 - huv.y));
      return vec4f(c.rgb, smoothstep(0.0, 0.08, e));
    }
    prev = t;
    step_len *= 1.22;
    t += step_len;
  }
  return vec4f(0.0);
}

// The sun's disc in slope space: a mirror tilted by s turns the reflection by
// 2 s, so the disc (radius about 0.0063, sky.wgsl) spans slopes of half
// that; a uniform disc of radius r has variance r^2 / 4 per axis.
const SUN_SLOPE2 = 2.5e-6;

// Smith shadowing for a Gaussian slope distribution of variance s2 along the
// direction's azimuth (Walter et al. 2007's rational fit), z = cosine to the normal.
fn smith_lambda(z: f32, s2: f32) -> f32 {
  let zc = clamp(z, 1e-3, 0.9999);
  let a = zc / sqrt((1.0 - zc * zc) * 2.0 * max(s2, 1e-7));
  if (a >= 1.6) { return 0.0; }
  return (1.0 - 1.259 * a + 0.396 * a * a) / (3.535 * a + 2.181 * a * a);
}
// Slope variance of covariance cv along direction d's azimuth in the frame Tx, Ty.
fn az_var(d: vec3f, Tx: vec3f, Ty: vec3f, cv: vec3f) -> f32 {
  let x = dot(d, Tx);
  let y = dot(d, Ty);
  return (cv.x * x * x + 2.0 * cv.z * x * y + cv.y * y * y) / max(x * x + y * y, 1e-8);
}
// Light from direction Ls reflected toward the eye (view ray V) by facets
// whose slopes, relative to the resolved normal N (frame Tx, Ty), spread with
// covariance cv (Bruneton et al. 2010): Fresnel x slope density / (4 cos_v
// cos^4 of the half vector), with Smith shadowing. Times irradiance.
fn sea_glint(Ls: vec3f, V: vec3f, N: vec3f, Tx: vec3f, Ty: vec3f, cv: vec3f) -> f32 {
  let H = normalize(Ls - V);
  let zh = dot(H, N);
  if (zh <= 0.01) { return 0.0; }
  let zx = dot(H, Tx) / zh;
  let zy = dot(H, Ty) / zh;
  let det = max(cv.x * cv.y - cv.z * cv.z, 1e-12);
  let q = (cv.y * zx * zx - 2.0 * cv.z * zx * zy + cv.x * zy * zy) / det;
  let p = exp(-0.5 * q) / (2.0 * PI * sqrt(det));
  let zl = max(dot(Ls, N), 0.01);
  let zv = max(dot(-V, N), 0.01);
  let fr = 0.02 + 0.98 * pow(1.0 - max(dot(-V, H), 0.0), 5.0);
  let g = 1.0 + smith_lambda(zl, az_var(Ls, Tx, Ty, cv)) + smith_lambda(zv, az_var(-V, Tx, Ty, cv));
  return fr * p / (4.0 * zv * zh * zh * zh * zh * g);
}

// ------------------------------------------------------------ breakers
// The shore distance field at P (shore.rs): distance to land (m) as it is
// and smoothed, the seaward direction (east, north), how far to trust it, and
// how tall the breakers come in here (m; from how exposed it is, shore.wgsl
// `breaker_height`). Near the edge of its square the nearest land may lie
// outside: a distance shorter than the way to the edge is exact, a longer
// one may not be.
struct ShoreAt { d0: f32, d: f32, dir: vec2f, w: f32, hb: f32 };
fn shore_at(P: vec3f) -> ShoreAt {
  var o = ShoreAt(G.shore.z, G.shore.z, vec2f(0.0), 0.0, 0.0);
  if (G.shore.x < 0.5) { return o; }
  let q = G.shore_vp * vec4f(P, 1.0);
  let uv = q.xy * vec2f(0.5, -0.5) + 0.5;
  let edge = min(min(uv.x, 1.0 - uv.x), min(uv.y, 1.0 - uv.y)) * G.shore.w;
  if (edge <= 0.0) { return o; }
  // Smoothed over some 15 m plus a third of the way out: refraction turns
  // the crests to the shore's general line, not its every step and corner
  // (the water's edge itself comes in steps of the land cover's texels).
  let f0 = textureSampleLevel(t_shore, s_clamp, uv, 0.0);
  let lod = clamp(log2((0.35 * f0.x + 15.0) / G.shore.y), 0.0, 6.0);
  let f = textureSampleLevel(t_shore, s_clamp, uv, lod);
  let sl = length(f.yz);
  o.d0 = f0.x;
  o.d = f.x;
  o.dir = select(vec2f(0.0), f.yz / sl, sl > 1e-3);
  o.w = 1.0 - smoothstep(edge * 0.75, edge, f0.x);
  o.hb = max(f.w, 0.0);
  return o;
}

fn surf_hash(i: i32) -> f32 {
  var h = bitcast<u32>(i) * 747796405u + 2891336453u;
  h = ((h >> ((h >> 28u) + 4u)) ^ h) * 277803737u;
  return f32((h >> 22u) ^ h) / 4294967295.0;
}

// exp(-a x) behind a front at x = 0 (x >= 0, in wave cycles), spread over a
// pixel footprint of b cycles without changing its integral; ahead of the
// front (x < 0) it falls off over the footprint or a sharp edge's width.
fn surf_exp(x: f32, a: f32, b: f32) -> f32 {
  let ae = a / (1.0 + a * b);
  return (ae / a) * select(exp(x / (0.006 + b)), exp(-ae * x), x >= 0.0);
}

struct Surf {
  foam: f32,
  slope: vec2f,   // dh/de, dh/dn of the swell's own shape
  h: f32,         // its height above the mean (m)
  sv: vec3f,      // slope covariance (ee, nn, en) of the swell too fine to resolve
};

// Waves breaking along the shore. The swell runs in shore-parallel crests
// over a beach of slope s (depth h = s d + h0 at distance d), slowing as
// shallow-water waves (c = sqrt(g h)), so the phase from the shoreline is
// 2 w (sqrt(h) - sqrt(h0)) / (s sqrt(g)) and the crests bunch up toward land.
// Each crest breaks where the water is shallower than its height / 0.78 and
// runs in as a white bore, leaving foam behind it that fades; at the water's
// edge the swash spreads a fringe of foam that pulses as each bore arrives.
// Every wave has its own height, and breaks in its own sections along the
// shore. How tall they come in depends on how exposed the shore is (sh.hb):
// the surf of an open coast, a ripple lapping at a harbour wall in light air.
// Small waves spill without much white water, and run up steeper banks.
// fp: pixel footprint (m).
fn breakers(wc: vec2f, sh: ShoreAt, wclass: f32, fp: f32, t: f32) -> Surf {
  var o = Surf(0.0, vec2f(0.0), 0.0, vec3f(0.0));
  let d = sh.d;
  if (sh.w <= 0.0 || min(d, sh.d0) >= G.shore.z - 2.0) { return o; }
  let sdir = sh.dir;
  let wind = G.ocean[1].x;
  // Swell is always running in at sea; wind blowing onshore builds it.
  let onshore = -dot(G.ocean[0].zw, sdir);
  var H0: f32;
  var T: f32;
  var s: f32;
  var fk: f32;
  var wmax: f32;
  if (wclass > 2.5) { H0 = min(0.03 + 0.005 * wind, sh.hb + 0.01); T = 1.3 + 0.08 * wind; s = 0.08; fk = 0.3; wmax = 0.3; }        // river
  else if (wclass > 1.5) { H0 = min(0.05 + 0.014 * wind, sh.hb + 0.01); T = 1.6 + 0.13 * wind; s = 0.06; fk = 0.55; wmax = 0.6; } // lake
  else {
    H0 = sh.hb;
    T = min(2.0 + 6.5 * sqrt(H0), 14.0);
    s = mix(0.06, 0.02, smoothstep(0.15, 0.7, H0));
    fk = 1.0;
    wmax = 0.4 + 4.5 * H0;
  }
  H0 *= 0.85 + 0.15 * onshore;
  if (H0 < 0.015) { return o; }
  fk *= smoothstep(0.04, 0.5, H0);
  // Beaches steeper and shallower along the shore, crests not quite parallel to it.
  let nb = textureSampleLevel(t_noise, s_repeat, vec3f(wc / 2600.0, 0.43), 0.0);
  let na = textureSampleLevel(t_noise, s_repeat, vec3f(wc / 1100.0, 0.77), 0.0);
  s *= mix(0.55, 1.7, smoothstep(0.25, 0.75, nb.r));
  let psi = (na.r - 0.5) * 5.0;

  let h0 = 0.08;
  let h = s * d + h0;
  let w = 2.0 * PI / T;
  let kl = w / sqrt(9.81 * h);                 // local wavenumber
  let lam = 2.0 * PI / kl;
  let phi = 2.0 * w * (sqrt(h) - sqrt(h0)) / (s * sqrt(9.81));
  let cyc = (phi + psi) / (2.0 * PI) + t / T;
  let mf = floor(cyc);
  let u = cyc - mf;                             // 0 at the crest shoreward of us, 1 at the next one out
  let m = i32(mf);
  let b = fp / lam;                             // footprint in cycles
  let far = smoothstep(0.18, 0.55, b);

  // The two crests either side: height, and how far in it breaks.
  let z0 = surf_hash(m);
  let z1 = surf_hash(m + 1);
  let sec0 = textureSampleLevel(t_noise, s_repeat, vec3f(wc / 900.0, z0), 0.0).r;
  let sec1 = textureSampleLevel(t_noise, s_repeat, vec3f(wc / 900.0, z1), 0.0).r;
  let H_0 = H0 * (0.65 + 0.6 * z0) * mix(0.5, 1.2, smoothstep(0.3, 0.7, sec0));
  let H_1 = H0 * (0.65 + 0.6 * z1) * mix(0.5, 1.2, smoothstep(0.3, 0.7, sec1));
  let db0 = max(H_0 / 0.78 - h0, 0.0) / s;
  let db1 = max(H_1 / 0.78 - h0, 0.0) / s;
  // Broken by the time it reaches us; whitest just after it breaks.
  let br0 = smoothstep(db0 * 1.08 + 0.5, db0 * 0.85, d) * (0.8 + 0.5 * exp(-pow((d - db0 * 0.85) / (0.25 * db0 + 1.0), 2.0)));
  let br1 = smoothstep(db1 * 1.08 + 0.5, db1 * 0.85, d) * (0.8 + 0.5 * exp(-pow((d - db1 * 0.85) / (0.25 * db1 + 1.0), 2.0)));
  // The bore: a white front, foam trailing behind it (seaward) that fades.
  let fr = 0.5;
  var foam = br0 * (fr * surf_exp(u, 16.0, b) + (1.0 - fr) * surf_exp(u, 3.0, b)) + br1 * (fr * surf_exp(u - 1.0, 16.0, b) + (1.0 - fr) * surf_exp(u - 1.0, 3.0, b));
  // Far off the pattern is finer than a pixel: its mean over a cycle.
  let dbm = max(H0 * 0.9 / 0.78 - h0, 0.0) / s;
  let mean = smoothstep(dbm * 1.15 + 0.5, dbm * 0.7, d) * 0.22;
  foam = mix(foam, mean, far);
  // Spent foam milky over the whole surf zone.
  foam += 0.18 * smoothstep(dbm * 1.2 + 0.5, dbm * 0.3, d);
  // The swash: a fringe at the water's edge, widest as each bore arrives.
  let us = fract(psi / (2.0 * PI) + t / T);
  let ws = wmax * (0.25 + 0.75 * pow(1.0 - us, 2.0)) * (0.7 + 0.6 * surf_hash(i32(floor(psi / (2.0 * PI) + t / T))));
  let fringe = smoothstep(ws + 0.4 + fp, ws * 0.4, sh.d0);
  foam = max(foam, mix(fringe, smoothstep(wmax + fp, 0.0, sh.d0) * 0.5, smoothstep(0.3, 1.0, fp / wmax)));
  o.foam = clamp(foam, 0.0, 1.0) * fk * sh.w;

  // The swell's shape: a long back and a short, steep face toward the shore
  // (trough 70% of the way from one crest to the next), as tall as the wave
  // or the water allows, fading out to sea where the open-ocean waves take over.
  let Hb = mix(H_0, H_1, smoothstep(0.55, 0.85, u));
  let env = (1.0 - smoothstep(min(4.0 * dbm + 40.0, 300.0), min(8.0 * dbm + 90.0, 0.9 * G.shore.z), d)) * sh.w;
  let Hl = min(Hb, 0.78 * h) * env;
  let back = u < 0.7;
  let wu = select(0.5 + 0.5 * (u - 0.7) / 0.3, 0.5 * u / 0.7, back);
  let dw = select(0.5 / 0.3, 0.5 / 0.7, back);
  let amp = 0.5 * Hl * kl;                      // slope scale: d(height)/d(distance)
  let fine = 1.0 - smoothstep(0.12, 0.4, b);
  o.slope = sdir * (-amp * sin(2.0 * PI * wu) * dw * fine);
  o.h = 0.5 * Hl * cos(2.0 * PI * wu) * fine;
  let var_s = 0.6 * amp * amp * (1.0 - fine);
  o.sv = vec3f(sdir.x * sdir.x, sdir.y * sdir.y, sdir.x * sdir.y) * var_s;
  return o;
}

fn water(P: vec3f, V: vec3f, dist: f32, tint: vec3f, aux: vec4f, jit: f32) -> vec3f {
  // Keep the water parameters in their physical ranges.
  let wd = max(aux.x, 0.0);
  let wclass = clamp(aux.y, 0.0, 3.0);
  let shore = clamp(aux.z, 0.0, 1.0);
  var l = light_at(P);
  let upn = l.upn;
  // (looked up a little above our waves inside the 3D tiles' zone: the
  // mesh's own still water, a metre or two above ours, casts too)
  l.sun *= shadow_at(P + upn * select(0.0, 4.0, photo_covered(P)), upn);
  let e = normalize(G.east.xyz - upn * dot(G.east.xyz, upn));
  let n = cross(upn, e);
  let wc = wave_coord(P);
  let t = G.time.x;
  let cosv = max(dot(-V, upn), 0.03);
  let fp = dist * G.misc.w / cosv;                // pixel footprint along the view direction, metres
  var st = sea_state(wclass, upn.z, shore);
  let wind = G.ocean[1].x;

  // Wind (from the weather): it blows in gusts, ruffling some patches of the
  // surface and leaving others glassy between (cat's paws), more so in light air.
  let wdir = G.ocean[0].zw;
  let wn = vec2f(-wdir.y, wdir.x);
  let ww = vec2f(dot(wc, wdir), dot(wc, wn));
  let g1 = textureSampleLevel(t_noise, s_repeat, vec3f(ww.x / 520.0 - t * 0.012, ww.y / 170.0, 0.37), 0.0);
  let g2 = textureSampleLevel(t_noise, s_repeat, vec3f(ww.x / 95.0 - t * 0.035, ww.y / 38.0, 0.81), 0.0);
  // The noise volume has no mips: blend each layer to its mean before its
  // cells shrink under a pixel, or the gust pattern turns to sparkle far off.
  let g1r = mix(g1.r, 0.5, smoothstep(10.0, 45.0, fp));
  let g2g = mix(g2.g, 0.5, smoothstep(2.5, 10.0, fp));
  let calm = 1.0 - smoothstep(1.5, 7.0, wind);
  let ruffle = smoothstep(0.28, 0.72, g1r * 0.65 + g2g * 0.45 - 0.02 + 0.2 * (1.0 - st.z) - 0.3 * calm);
  // Gusts raise the short waves most.
  st.z *= mix(0.3, 1.2, ruffle);
  st.y *= mix(0.8, 1.1, ruffle);
  // Breakers along the shore, from the distance to it.
  let surf = breakers(wc, shore_at(P), wclass, fp, t);

  // The FFT ocean (ocean.wgsl). This pixel is where the sideways shift
  // carried a surface point to: find that point first (two fixed-point steps
  // over the cascades that move it most), then read its slopes.
  let xo = ocean_coord(P);
  var lod = vec3f(0.0);
  for (var c = 0; c < 3; c++) { lod[c] = log2(max(fp * 256.0 / ocean_size(c), 1.0)); }
  var x0 = xo;
  var hgt = 0.0;
  for (var it = 0; it < 2; it++) {
    var ds = vec3f(0.0);
    for (var c = 0; c < 2; c++) {
      if (lod[c] < 8.0) { ds += textureSampleLevel(t_ocean, s_repeat, x0 / ocean_size(c), c, lod[c]).xyz * st[c]; }
    }
    x0 = xo - ds.xy;
    hgt = ds.z;
  }
  var sl = vec2f(0.0);   // dh/dx, dh/dy over the undisplaced surface
  var jd = vec3f(0.0);   // Dxx, Dyy, Dxy
  var jd01 = vec3f(0.0); // the same for the two longer cascades (whitecaps)
  // Roughness from the spectrum (Bruneton et al. 2010): the slope covariance
  // (east-east, north-north, east-north) of every wave the mips read here no
  // longer hold - worked out from the spectrum per cascade and level
  // (ocean.rs `lost_slopes`) - plus the ripples shorter than any cascade.
  // It widens the BRDF by just what the resolved normal lost, so near and far
  // water light alike and a pixel spanning many waves shows their mean glitter.
  var cv = G.ocean[2].xyz * st.z * st.z;
  for (var c = 0; c < 3; c++) {
    let w = st[c];
    if (w <= 0.0) { continue; }
    let L = ocean_size(c);
    if (lod[c] < 8.0) {
      let a = textureSampleLevel(t_ocean, s_repeat, x0 / L, c, lod[c]);
      let b = textureSampleLevel(t_ocean, s_repeat, x0 / L, c + 3, lod[c]);
      sl += b.xy * w;
      jd += vec3f(b.zw, a.w) * w;
      if (c < 2) { jd01 += vec3f(b.zw, a.w) * w; }
    }
    cv += ocean_lost(c, lod[c]) * (w * w);
  }
  // The swell running in to the shore, where it is finer than a pixel.
  cv += surf.sv;
  // The sun's disc, seen in a mirror, spans slopes of about its half radius.
  cv += vec3f(SUN_SLOPE2, SUN_SLOPE2, 0.0);
  // Surface normal through the sideways shift: slopes over the moved surface.
  let jxx = 1.0 + jd.x;
  let jyy = 1.0 + jd.y;
  let J = jxx * jyy - jd.z * jd.z;
  var slope = vec2f(sl.x * jyy - sl.y * jd.z, sl.y * jxx - sl.x * jd.z) / max(J, 0.25) + surf.slope;
  let smag = length(slope);
  if (smag > 1.5) { slope *= 1.5 / smag; }
  let J01 = (1.0 + jd01.x) * (1.0 + jd01.y) - jd01.z * jd01.z;
  let N = normalize(upn - e * slope.x - n * slope.y);
  // (the march for reflected scenery leaves out the swell's long, even
  // crests: whole crest lines would catch the beach behind them at once)
  let Nw = normalize(upn - e * (slope.x - surf.slope.x) - n * (slope.y - surf.slope.y));
  // Tangent frame on the resolved surface, east and north as near as may be.
  let Tx = normalize(e - N * dot(e, N));
  let Ty = cross(N, Tx);

  // Slope variance along the direction we look and across it.
  let vh = normalize(vec2f(dot(V, e), dot(V, n)) + vec2f(1e-6, 0.0));
  let vc = vec2f(-vh.y, vh.x);
  let sv2 = cv.x * vh.x * vh.x + 2.0 * cv.z * vh.x * vh.y + cv.y * vh.y * vh.y;
  let sc2 = cv.x * vc.x * vc.x + 2.0 * cv.z * vc.x * vc.y + cv.y * vc.y * vc.y;
  let sv = sqrt(max(sv2, 0.0));
  let Ta = Tx * vh.x + Ty * vh.y;
  let Tb = Tx * vc.x + Ty * vc.y;

  // Reflection: screen-space first, sky (with cloud glints) as the fallback.
  var R = reflect(V, N);
  let ru = dot(R, upn);
  if (ru < 0.03) { R = normalize(R + upn * (0.03 - ru)); }
  let ndv = max(dot(-V, N), 0.0);
  // Fresnel averaged over the unresolved facets (Bruneton's fit).
  let F = 0.02 + 0.98 * pow(1.0 - ndv, 5.0 * exp(-2.69 * sv)) / (1.0 + 22.7 * pow(sv, 1.5));
  // The sky averaged over the facets' slope distribution: the Gaussian's
  // mean and second moment matched by five reflections (centre 1/3; +-sqrt(3)
  // sigma along and across the view 1/6 each). Along the view the lobe
  // reaches up the sky (a grazing reflection's streak) and down to the
  // horizon, which is where the lifted rays then look.
  let dv = sqrt(3.0 * max(sv2, 0.0));
  let dc = sqrt(3.0 * max(sc2, 0.0));
  var refl = sky_lut(R) / 3.0;
  for (var k = 0; k < 4; k++) {
    let o = select(Tb * select(dc, -dc, k == 3), Ta * select(dv, -dv, k == 1), k < 2);
    var Rk = reflect(V, normalize(N - o));
    Rk = normalize(Rk + upn * max(0.03 - dot(Rk, upn), 0.0));
    refl += sky_lut(Rk) / 6.0;
  }
  let mid = G.cloud.y + 400.0;
  if (l.h < mid) {
    // The cloud layer is far away along a grazing reflection: a wave tilt of
    // a few degrees moves the reflected point by kilometres, so neighbouring
    // pixels would see unrelated clouds (sparkling noise). Tame the tilt so
    // the reflected point moves only a few hundred metres - the clouds then
    // reflect as smooth, wave-wobbled shapes, as they do on real water.
    let hc = max(mid - l.h, 100.0);
    let kc = clamp(1500.0 * cosv * cosv / hc, 0.0, 1.0);
    let Nc = normalize(upn + (N - upn) * kc);
    // The unresolved facets smear the reflection along the view, into a
    // vertical streak (as the sky above): three reflections across that lobe
    // (centre 2/3, +-sqrt(3) sigma 1/6 each). The haze between the water and
    // a reflected cloud fades it toward the sky colour behind it, as it does
    // for the clouds seen directly.
    let ext = atm_ext(atm_density(0.5 * (l.h + mid)));
    let kext = dot(ext, vec3f(0.2126, 0.7152, 0.0722));
    var cov_c = 0.0;
    var tc0 = 0.0;
    var Rc0 = reflect(V, Nc);
    for (var k = 0; k < 3; k++) {
      let off = select(select(0.0, dv, k == 1), -dv, k == 2);
      var Rc = reflect(V, normalize(Nc - Ta * off));
      Rc = normalize(Rc + upn * max(0.03 - dot(Rc, upn), 0.0));
      let tc = hc / max(dot(Rc, upn), 0.03);
      if (k == 0) { tc0 = tc; Rc0 = Rc; }
      if (tc < 90000.0) {
        let cd = cloud_density(P - G.planet.xyz + Rc * tc, mid, true);
        cov_c += select(1.0 / 6.0, 2.0 / 3.0, k == 0) * (1.0 - exp(-cd * 7.0)) * (1.0 - smoothstep(40000.0, 90000.0, tc)) * exp(-tc * kext);
      }
    }
    if (cov_c > 0.0) {
      let lc = light_at(P + Rc0 * min(tc0, 90000.0));
      refl = mix(refl, (lc.sun * 0.05 + lc.amb * 1.3), cov_c);
    }
  }
  if (G.misc.z > 0.5) {
    // Same idea for reflected scenery: per-pixel wave normals make the march
    // hit and miss on alternate pixels. Less tilt the further off (where a
    // pixel spans many waves) keeps the wobble while the reflection holds together.
    let ks = mix(0.5, 0.12, smoothstep(250.0, 2500.0, dist));
    var Rs = reflect(V, normalize(upn + (Nw - upn) * ks));
    Rs = normalize(Rs + upn * max(0.03 - dot(Rs, upn), 0.0));
    let hit = ssr(P, Rs, 0.5);
    refl = mix(refl, hit.rgb, hit.a);
  }

  // Sun and moon glitter: the facets' slope distribution (anisotropic
  // Gaussian, the same covariance) as the BRDF.
  var spec = sea_glint(G.sun_dir.xyz, V, N, Tx, Ty, cv) * l.sun;
  if (G.moon.w > 0.0) {
    spec += sea_glint(G.moon.xyz, V, N, Tx, Ty, cv) * moon_light(upn, l.h);
  }

  // What kind of water is this? Latitude, wclass and the satellite colour decide.
  let alat = abs(upn.z);
  let trop = 1.0 - smoothstep(0.33, 0.58, alat);
  let polar = smoothstep(0.78, 0.93, alat);
  var turb = mix(0.8, 0.06, trop);
  var body = mix(vec3f(0.004, 0.024, 0.046), vec3f(0.003, 0.05, 0.075), trop);
  body = mix(body, vec3f(0.006, 0.022, 0.03), polar);
  var weedy = mix(1.0, 0.82, trop);
  if (wclass > 2.5) { turb = 3.2; body = vec3f(0.045, 0.048, 0.022); weedy = 0.9; }
  else if (wclass > 1.5) { turb = 1.5; body = vec3f(0.012, 0.03, 0.022); weedy = 1.1; }
  let tl = max(luma(tint), 0.015);
  body = mix(body, tint / tl * luma(body), 0.4);
  let ext = vec3f(0.36, 0.052, 0.024) + vec3f(0.22, 0.2, 0.3) * turb;

  // Refraction down to the bed.
  let Rf = refract(V, N, 0.75);
  let path = wd / max(dot(-Rf, upn), 0.25);
  let Tv = exp(-ext * path);
  let mu = max(dot(upn, G.sun_dir.xyz), 0.0);
  var under = body * (l.sun * (0.3 + 0.7 * mu) / PI + l.amb) * 3.2 * (vec3f(1.0) - Tv);
  if (luma(Tv) > 0.01) {
    let bp = wc + vec2f(dot(Rf, e), dot(Rf, n)) * path;
    let near = 1.0 - smoothstep(300.0, 2500.0, dist);
    // Organic bed detail from the 3D noise volume (no visible repeat).
    let nA = textureSampleLevel(t_noise, s_repeat, vec3f(bp / 140.0, 0.37), 0.0);
    let nB = textureSampleLevel(t_noise, s_repeat, vec3f(bp / 23.0, 0.71), 0.0);
    let n1 = nA.r;
    let n2 = nB.r;
    var bed = vec3f(0.58, 0.51, 0.37) * (0.75 + 0.5 * n2);
    bed = mix(bed, vec3f(0.14, 0.13, 0.12), smoothstep(0.55, 0.75, n1 * 0.7 + nB.g * 0.4) * 0.85 * near);
    // Seaweed and kelp beds that sway with the swell.
    let sway = vec2f(sin(t * 0.8 + bp.y * 0.35), cos(t * 0.65 + bp.x * 0.3)) * 0.35;
    let wA = textureSampleLevel(t_noise, s_repeat, vec3f((bp + sway) / 61.0, 0.13), 0.0);
    let wB = textureSampleLevel(t_noise, s_repeat, vec3f((bp + sway * 0.6) / 9.0, 0.53), 0.0);
    let weed = smoothstep(0.52, 0.66, wA.r * weedy) * smoothstep(0.35, 0.6, wB.g + wB.b * 0.4) * smoothstep(0.6, 2.5, wd) * near;
    bed = mix(bed, vec3f(0.02, 0.07, 0.022) * (0.5 + 1.0 * wB.b), weed * 0.92);
    bed = mix(bed, tint * 2.4, 0.4 * (1.0 - weed));
    // Caustics.
    let c1 = wtex(bp / 6.3 + vec2f(0.03, 0.02) * t, 6.3, fp).b;
    let c2 = wtex(bp / 4.9 - vec2f(0.025, 0.04) * t, 4.9, fp).b;
    let ca = pow(1.0 - abs(c1 - c2), 9.0) * 3.0 * exp(-wd * 0.12) * near * smoothstep(0.1, 0.8, wd);
    let Td = exp(-ext * wd / max(mu, 0.35));
    under += bed * (l.sun * mu * Td * (1.0 + ca) / PI + l.amb * Td) * Tv;
  }

  // Light through the crests: looking toward the sun, the wave between the
  // eye and it glows green where it is thin and lit from behind.
  let hs = max(G.ocean[1].y * (st.x + st.y) * 0.5, 0.2);
  let crest_h = clamp((hgt + surf.h) / hs + 0.45, 0.0, 1.0);
  let toward = pow(max(dot(V, G.sun_dir.xyz), 0.0), 4.0);
  let sss = toward * crest_h * (0.3 + 0.7 * (1.0 - ndv)) * max(dot(G.sun_dir.xyz, upn), 0.0);
  under += (body * 8.0 + vec3f(0.01, 0.06, 0.05)) * l.sun * sss * 0.4 / PI;

  var col = mix(under, refl, clamp(F, 0.0, 1.0)) + spec + point_light(P, upn) * body * 4.0;

  // Foam: breakers along the shore, whitecaps where the crests fold over.
  let fa = textureSampleLevel(t_noise, s_repeat, vec3f(wc / 37.0 + vec2f(0.012, 0.007) * t, t * 0.02), 0.0);
  let fb = textureSampleLevel(t_noise, s_repeat, vec3f(wc / 5.3 - vec2f(0.02, 0.03) * t, 0.4 + t * 0.03), 0.0);
  // The noise volume has no mips: fade each layer to its mean before the
  // pixel footprint outgrows its features, or it sparkles as white speckle.
  let lace = mix(fb.g * 0.6 + fb.b * 0.5, 0.55, smoothstep(0.12, 0.7, fp));
  let far_a = smoothstep(1.5, 6.0, fp);
  let fa_g = mix(fa.g, 0.5, far_a);
  let fnear = 1.0 - smoothstep(2500.0, 9000.0, dist);
  // The breakers' foam, laced: solid on a fresh bore, broken up as it thins.
  var foam = clamp(surf.foam * (0.45 + lace + (fa_g - 0.5) * 0.3) * 1.2, 0.0, 1.0);
  // Whitecaps (ocean.wgsl `foam`): foam put down where the crests fold (the
  // Jacobian of the sideways shift), built up, drifted downwind and faded,
  // its cover steered to what the wind whitens (Monahan). It is kept in the
  // surface's own coordinates, so it rides the waves' orbits with the water.
  let fz = textureSampleLevel(t_ocean, s_repeat, x0 / ocean_size(0), 6, lod[0]);
  let capw = step(wclass, 1.5) * smoothstep(0.35, 0.95, 0.5 * (st.x + st.y));
  // Spent foam breaks up into lace and streaks drawn out downwind: a
  // threshold over noise stretched along the wind (in the surface's own
  // coordinates too) that keeps the cover on average. Far off, where the
  // pattern would sparkle, the mips' mean cover shows instead.
  let wl = wc + (x0 - xo);
  let wwl = vec2f(dot(wl, wdir) - 0.03 * wind * t, dot(wl, wn));
  let sA = textureSampleLevel(t_noise, s_repeat, vec3f(wwl.x / 40.0, wwl.y / 12.0, 0.29), 0.0);
  let sB = textureSampleLevel(t_noise, s_repeat, vec3f(wwl.x / 12.0, wwl.y / 4.0, 0.63), 0.0);
  // (the noise volume has no mips: each layer goes to its mean before its
  // texels shrink under a pixel)
  let nc = (sA.r - 0.5) * 0.7 + (mix(sB.g, 0.55, smoothstep(0.04, 0.2, fp)) - 0.55) * 0.5;
  let spent = mix(smoothstep(-0.1, 0.1, fz.r - 0.5 - nc * 2.2), fz.r, smoothstep(0.15, 0.8, fp));
  // The breaking itself: solid white right on the folding crest, sharp at
  // this pixel's own fold, where the foam is fresh.
  let crest = fz.g * smoothstep(fz.b + 0.12, fz.b - 0.08, J01);
  foam += max(spent, crest) * capw * mix(0.95, 0.75, far_a);
  foam = clamp(foam * fnear, 0.0, 1.0);
  // White water churned up where a crash went in (main.rs, Crash::foam):
  // a solid boil at first that soon breaks up into lace and streaks as it
  // spreads and fades, over water turned pale green by the air in it.
  var aer = 0.0;
  for (var i = 0; i < 8; i++) {
    let f = G.foam[2 * i];
    if (f.w <= 0.0) { continue; }
    let fg = G.foam[2 * i + 1];
    let dp = P - f.xyz;
    let d = length(dp - upn * dot(dp, upn)) / f.w;
    if (d >= 1.4) { continue; }
    let q = vec2f(dot(dp, e), dot(dp, n)) + fg.z * 131.0;
    let nA = textureSampleLevel(t_noise, s_repeat, vec3f(q / max(f.w * 0.7, 3.0), 0.23 + fg.y * 0.003), 0.0);
    let nB = textureSampleLevel(t_noise, s_repeat, vec3f(q / 5.3, 0.57 + fg.y * 0.006), 0.0);
    let nC = textureSampleLevel(t_noise, s_repeat, vec3f(q / 1.7, 0.11 + fg.y * 0.01), 0.0);
    let body = smoothstep(1.0, 0.35, d + (nA.r - 0.5) * 0.9);
    let lc = mix(nB.g * 0.55 + nC.b * 0.45 + (nA.g - 0.5) * 0.5, 0.5, smoothstep(0.15, 0.9, fp));
    let fresh = exp(-fg.y / 2.5);
    let k = body * fg.x * (0.42 + 0.58 * fresh);
    aer = max(aer, body * fg.x);
    foam = max(foam, clamp(k * (1.0 + 0.9 * fresh) - (1.0 - lc) * (1.0 - k) * 1.8, 0.0, 1.0) * fnear);
  }
  // Foam is bubbles, not paint: holes and lace within it, close up.
  let bub = textureSampleLevel(t_noise, s_repeat, vec3f(wc / 2.1, 0.2 + t * 0.04), 0.0);
  let bub2 = textureSampleLevel(t_noise, s_repeat, vec3f(wc / 0.55, 0.6), 0.0);
  let holes = smoothstep(0.3, 0.65, bub.r * 0.6 + bub2.g * 0.5);
  foam *= mix(1.0, 0.45 + 0.7 * holes, 1.0 - smoothstep(0.04, 0.2, fp));
  // (bubbles just under the surface: the water pale and milky green)
  col = mix(col, vec3f(0.12, 0.26, 0.25) * (l.sun * (0.3 + 0.7 * mu) / PI + l.amb), aer * 0.55 * fnear);
  col = mix(col, vec3f(0.78, 0.8, 0.8) * (0.85 + 0.3 * bub2.b) * (l.sun * (0.4 + 0.6 * mu) / PI + l.amb), foam);
  return max(col, vec3f(0.0));
}

// Whether this pipeline shades water. The land pass (render.rs, "composite"
// with FABLE_OPT_OFF=comp_split unset) is built with it false, so the water
// shading and the per-sample loop are compiled out of it: with them in, the
// registers they need cost every pixel of the screen occupancy (some 7 ms at
// High fullscreen). Water pixels then get the "composite water" pass, and
// with MSAA the pixels that mix water and not "composite edge".
override WATER: bool = true;

// Whether composite_sample takes the water path: the same test as inside it.
fn is_water(z: f32, c0: vec4f, a0: vec4f) -> bool {
  return !(z == 0.0 || c0.a < 0.004) && a0.a / c0.a > 0.996;
}

// Falling snow (G.alto.w how hard): four layers of flakes 4, 8, 16 and 32 m
// out, hidden behind anything nearer. Each layer is a grid of cells over the
// directions around the camera, in log-polar angles about the direction the
// flakes stream from (G.snow.xyz): moving through them is then sliding the
// grid outward, by G.snow.w over the layer's distance. The wrap of G.snow.w
// (SNOW_WRAP in main.rs) is a whole number of cells in each layer, and the
// pattern repeats over that many, so it never jumps.
const SNOW_N = 96.0;  // cells around the axis
fn snowfall(col: vec3f, rd: vec3f, dist: f32) -> vec3f {
  let amt = G.alto.w;
  if (amt <= 0.001) { return col; }
  let ax = normalize(G.snow.xyz);
  let ct = clamp(dot(rd, ax), -0.9999, 0.9999);
  let st = sqrt(1.0 - ct * ct);
  // log(tan(theta / 2)), and the angle around the axis
  let u = log(st / (1.0 + ct));
  var e1 = cross(ax, vec3f(0.0, 0.0, 1.0));
  if (dot(e1, e1) < 1e-4) { e1 = cross(ax, vec3f(1.0, 0.0, 0.0)); }
  e1 = normalize(e1);
  let e2 = cross(ax, e1);
  let v = atan2(dot(rd, e2), dot(rd, e1));
  let cu = 6.2831853 / SNOW_N;
  let l = light_at_nv(vec3f(0.0));
  let lit = (l.sun * max(dot(l.upn, G.sun_dir.xyz), 0.0) * 0.6 / PI + l.amb) * 0.85;
  var c = col;
  var d = 4.0;
  var period = 64.0;
  // (fewer toward the point they stream from, where the cells crowd together)
  let thin = smoothstep(0.04, 0.35, st);
  for (var i = 0; i < 4; i++) {
    if (dist > d) {
      // Each layer's grid turned and shifted its own way, so no two line up.
      let g = vec2f((u - G.snow.w / d) / cu + f32(i) * 0.37, (v + 3.14159265) / cu + f32(i) * 0.61);
      let cell = floor(g);
      let cx = cell.x - period * floor(cell.x / period);
      let cy = cell.y - SNOW_N * floor(cell.y / SNOW_N);
      let h = vec3f(hash2(vec2f(cx, cy) + f32(i) * 17.0), hash2(vec2f(cy, cx) + f32(i) * 31.0), hash2(vec2f(cx + 0.5, cy * 1.7) + f32(i) * 5.0));
      // One cell in so many holds a flake, more the harder it snows.
      if (h.z < (0.12 + 0.3 * amt) * thin) {
        let f = (cell + 0.12 + 0.76 * h.xy - g) * cu * st;  // angle to the flake (rad)
        let r = (0.008 + 0.012 * fract(h.z * 13.7)) / d;
        let rp = max(r, G.misc.w * 0.8);
        let a = (1.0 - smoothstep(rp * 0.4, rp, length(f))) * (r * r) / (rp * rp) * min(amt * 2.0, 1.0) * 0.9;
        c = mix(c, lit, a);
      }
    }
    d *= 2.0;
    period *= 0.5;
  }
  return c;
}

fn composite_sample(in: FsIn, z: f32, c0: vec4f, a0: vec4f, amb: f32) -> vec3f {
  let px = vec2i(in.pos.xy);
  let ray = view_ray(in.uv);
  let rd = normalize(ray);
  let jit = ign(in.pos.xy);
  let oc = -G.planet.xyz;
  // Alpha is how much of the pixel geometry covers (the anti-aliased edge).
  let cov = c0.a;
  if (z == 0.0 || cov < 0.004) {
    return clamp(snowfall(sky_color(rd, jit), rd, 1e9), vec3f(0.0), vec3f(3000.0));
  }
  let P = ray / z;
  let dist = length(P);
  var col = c0.rgb / cov;
  let aux = a0 / cov;
  if (aux.a <= 0.004) {
    // Ambient occlusion takes away from the sky and bounce light only.
    let share = clamp(amb / cov, 0.0, 1.0);
    col *= 1.0 - share * (1.0 - textureLoad(t_ao, px, 0).r);
  }
  if (aux.a > 0.996) {
    if (!WATER) { return vec3f(0.0); }
    col = water(P, rd, dist, col, aux, jit);
  }
  else if (aux.a < -0.01 && G.misc.z > 0.5) {
    // Glass: swap the sky the building shader reflected for whatever stands
    // in that direction on screen - the neighbouring towers, the street.
    let Ng = normalize(aux.xyz);
    let R = reflect(rd, Ng);
    let hit = ssr(P + Ng * 0.25, R, jit);
    if (hit.a > 0.0) {
      let upn = normalize(P - G.planet.xyz);
      let env = env_reflect(P, R, upn, dist < 2500.0);
      col = max(col + (hit.rgb - env) * min(-aux.a, 1.0) * hit.a, vec3f(0.0));
    }
  }
  if (dist > 150.0) {
    let top = ray_sphere(oc, rd, G.planet.w + ATM_H);
    let t0 = max(top.x, 0.0);
    let t1 = min(dist, top.y);
    if (t1 > t0) {
      let sc = atm_march(oc, rd, t0, t1, atm_steps(12), fract(jit + G.time.w * 0.618034));
      col = col * sc.T + sc.L + SKYGLOW * 0.004 * G.misc.y * night_cam() * (1.0 - luma(sc.T));
    }
  }
  if (cov < 0.996) { col = mix(sky_color(rd, jit), col, cov); }
  col = snowfall(col, rd, dist);
  return clamp(col, vec3f(0.0), vec3f(3000.0));
}

// How many of the pixel's samples are water, where the resolved marker allows
// a mix of water and not: the resolved marker already tells a pixel with no
// water (0, or near it) or all water (1) from a mix of them (in between, or
// below with glass), and only those read their samples. (G.vol_n.w = 1:
// FABLE_OPT_OFF=water_edge reads them everywhere, as before)
fn water_samples(px: vec2i) -> i32 {
  var n = 0;
  let ra = textureLoad(t_aux, px, 0).a;
  let mixed = !((ra >= -0.01 && ra <= 0.004) || ra > 0.996) || G.vol_n.w > 0.5;
  for (var i = 0; i < 4; i++) {
    if (mixed && textureLoad(t_aux_ms, px, i).a > 0.996) { n++; }
  }
  return n;
}
// Shade before resolving: each water sample needs its own tint, parameters
// and depth, even when no neighbouring pixel is entirely water. Foreground
// samples retain their own lighting and haze rather than water's colour.
fn per_sample(in: FsIn, px: vec2i) -> vec4f {
  var col = vec3f(0.0);
  for (var i = 0; i < 4; i++) {
    col += composite_sample(in, textureLoad(t_depth_ms, px, i),
      textureLoad(t_color_ms, px, i), textureLoad(t_aux_ms, px, i),
      textureLoad(t_amb_ms, px, i).r);
  }
  return vec4f(col * 0.25, 1.0);
}
fn single(in: FsIn, px: vec2i) -> vec4f {
  return vec4f(composite_sample(in, textureLoad(t_depth, px, 0),
    textureLoad(t_color, px, 0), textureLoad(t_aux, px, 0),
    textureLoad(t_amb, px, 0).r), 1.0);
}

// All in one pass (FABLE_OPT_OFF=comp_split).
@fragment fn fs(in: FsIn) -> @location(0) vec4f {
  let px = vec2i(in.pos.xy);
  let n = water_samples(px);
  if (n > 0 && n < 4) { return per_sample(in, px); }
  return single(in, px);
}
// MSAA off (Medium and Low): one sample a pixel, and no multisampled inputs
// bound. Built with WATER false it is also the land pass, MSAA or not: its
// water pixels (and the mixed ones) are then left to fs_water(1).
@fragment fn fs1(in: FsIn) -> @location(0) vec4f {
  return single(in, vec2i(in.pos.xy));
}
// After the land pass, over it: the water pixels (fs_water1, MSAA or not),
// then with MSAA those whose samples mix water and not (fs_edge). The rest
// keep what the passes before gave them, which is what fs gives them. Three
// passes, not one: each keeps only the registers its own work needs.
@fragment fn fs_water1(in: FsIn) -> @location(0) vec4f {
  let px = vec2i(in.pos.xy);
  if (!is_water(textureLoad(t_depth, px, 0), textureLoad(t_color, px, 0), textureLoad(t_aux, px, 0))) { discard; }
  return single(in, px);
}
@fragment fn fs_edge(in: FsIn) -> @location(0) vec4f {
  let px = vec2i(in.pos.xy);
  let n = water_samples(px);
  if (n == 0 || n == 4) { discard; }
  return per_sample(in, px);
}
