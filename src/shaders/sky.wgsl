// Sky as seen through a pixel with no geometry: atmosphere, sun, stars, moon.
// Shared by the composite (sky pixels) and the cloud upsample (anti-aliased edges).

// The star field turns with the sun (near enough to sidereal for a flight).
fn celestial(rd: vec3f) -> vec3f {
  let s = normalize(G.sun_dir.xy + vec2f(1e-6, 0.0));
  return vec3f(s.x * rd.x + s.y * rd.y, -s.y * rd.x + s.x * rd.y, rd.z);
}
// One layer of stars: at most one per cell of a 3D grid of n cells per radian,
// each drawn as a small gaussian at least a pixel wide so it never aliases.
fn star_layer(q: vec3f, n: f32, dens: f32, seed: f32) -> vec3f {
  let p = q * n + seed;
  let c = floor(p);
  if (hash3(c) > dens) { return vec3f(0.0); }
  let off = vec3f(hash3(c + 1.3), hash3(c + 2.7), hash3(c + 4.1)) * 0.6 + 0.2;
  let d = length(fract(p) - off) / n;
  let sig = G.misc.w * 0.7;
  let g = exp(-d * d / (2.0 * sig * sig));
  if (g < 0.002) { return vec3f(0.0); }
  let hb = hash3(c + 9.0);
  let ht = hash3(c + 5.5);
  // Mostly faint ones, a few bright; blue-white through to orange giants.
  let b = exp2(-hb * 7.5);
  let col = mix(vec3f(0.62, 0.76, 1.0), vec3f(1.0, 0.78, 0.52), ht * ht) / mix(0.75, 0.82, ht * ht);
  let tw = 1.0 + 0.35 * sin(G.time.x * (7.0 + 9.0 * ht) + hb * 60.0) * sin(G.time.x * (3.1 + 4.0 * hb) + ht * 30.0);
  return col * b * g * tw;
}
fn stars(rd: vec3f) -> vec3f {
  let q = celestial(rd);
  var c = star_layer(q, 58.0, 0.075, 0.0) + star_layer(q.zxy, 41.0, 0.06, 17.3) * 0.8;
  // The Milky Way: a faint, blotchy band with dark dust lanes down its middle.
  let gp = normalize(vec3f(-0.42, 0.55, 0.72));
  let lat = dot(q, gp);
  let band = exp(-lat * lat / 0.03);
  if (band > 0.01) {
    let n1 = textureSampleLevel(t_noise, s_repeat, q * 1.7 + vec3f(0.2, 0.5, 0.1), 0.0);
    let n2 = textureSampleLevel(t_noise, s_repeat, q * 6.5 + vec3f(0.7, 0.3, 0.9), 0.0);
    let core = exp(-pow(length(q - normalize(vec3f(0.62, 0.71, -0.33))), 2.0) / 0.35);
    let dust = smoothstep(0.45, 0.75, n2.g * 0.6 + n1.b * 0.5) * exp(-lat * lat / 0.004);
    let glow = band * (0.45 + 0.9 * n1.r * n2.r + 1.4 * core) * (1.0 - 0.75 * dust);
    c += vec3f(0.85, 0.88, 1.0) * glow * 0.0065;
    // Unresolved stars crowd the band.
    c += star_layer(q.yzx, 97.0, 0.12 * band, 5.1) * 0.35;
  }
  return c;
}
// Deep space for the start screen, where the planet hangs in the dark: just
// stars, denser than the night sky shows, with no Milky Way or nebulae.
// Display-ready (the night sky scales `stars` down behind the atmosphere).
fn space(rd: vec3f) -> vec3f {
  let q = celestial(rd);
  var c = (star_layer(q, 58.0, 0.075, 0.0) + star_layer(q.zxy, 41.0, 0.06, 17.3) * 0.8) * 0.2;
  // Faint dust of stars too far to pick out one by one.
  c += star_layer(q.yxz, 150.0, 0.14, 31.7) * 0.05 + star_layer(q.zyx, 240.0, 0.12, 47.1) * 0.025;
  return c;
}
// The moon's disc: lit by the sun so it shows its phase, with darker maria.
fn moon_disc(rd: vec3f) -> vec3f {
  let m = G.moon.xyz;
  let R = 0.0058;
  if (dot(rd, m) < 1.0 - R * R * 0.75) { return vec3f(0.0); }
  let ex = normalize(cross(vec3f(0.0, 0.0, 1.0), m) + vec3f(1e-6, 0.0, 0.0));
  let ey = cross(m, ex);
  let q = vec2f(dot(rd, ex), dot(rd, ey)) / R;
  let r = length(q);
  let edge = clamp((1.0 - r) * R / G.misc.w + 0.5, 0.0, 1.0);
  if (edge <= 0.0) { return vec3f(0.0); }
  let z = sqrt(max(1.0 - r * r, 0.0));
  let nrm = ex * q.x + ey * q.y - m * z;
  let lit = smoothstep(-0.02, 0.1, dot(nrm, G.sun_dir.xyz));
  let mp = vec3f(q, z);
  let a = textureSampleLevel(t_noise, s_repeat, mp * 0.38 + vec3f(0.31, 0.12, 0.71), 0.0);
  let b = textureSampleLevel(t_noise, s_repeat, mp * 1.9 + vec3f(0.2, 0.6, 0.4), 0.0);
  let maria = smoothstep(0.52, 0.66, a.r * 0.8 + a.g * 0.3);
  let alb = (1.0 - 0.42 * maria) * (0.86 + 0.28 * b.g) * (0.82 + 0.18 * z);
  return vec3f(1.0, 0.97, 0.92) * alb * (lit + 0.012) * edge * G.moon.w * 8.0;
}
// How dark it is at the camera (1 = full night).
fn night_cam() -> f32 { return 1.0 - smoothstep(-0.2, 0.05, dot(G.up.xyz, G.sun_dir.xyz)); }

fn sky_color(rd: vec3f, jit: f32) -> vec3f {
  let oc = -G.planet.xyz;
  let sun = G.sun_dir.xyz;
  // Fresh jitter every frame: cloud shadows cross the march (light shafts), and the
  // temporal AA smooths what the steps miss.
  let sc = sky_march(rd, atm_steps(24), fract(jit + G.time.w * 0.618034));
  var col = sc.L;
  let gnd = ray_sphere(oc, rd, G.planet.w);
  if (gnd.x > 0.0) {
    // Bare planet where no tile is drawn yet: a plain lit ocean.
    let p = oc + rd * gnd.x;
    let mu = dot(normalize(p), sun);
    let mum = dot(normalize(p), G.moon.xyz);
    col += sc.T * vec3f(0.012, 0.035, 0.07) * ((sun_trans(0.0, mu) * max(mu, 0.0) + 0.02) * G.sun_dir.w + moon_light(normalize(p), 0.0) * max(mum, 0.0)) / PI;
  } else {
    let cs = dot(rd, sun);
    col += sc.T * smoothstep(0.999978, 0.999990, cs) * G.sun_dir.w * 600.0;
    col += sc.T * (select(stars(rd) * 0.11, space(rd), G.start.x > 0.0) + moon_disc(rd));
    // City light scattered by the haze: an orange glow low over the horizon.
    let el = dot(rd, G.up.xyz);
    col += SKYGLOW * 0.0022 * G.misc.y * night_cam() * exp(-max(el, 0.0) * 9.0) * (1.0 + G.cloud.x);
    // Airglow and scattered starlight: even a moonless sky is deep blue, not black.
    col += vec3f(0.0004, 0.0008, 0.0019) * night_cam() * (1.0 - 0.4 * max(el, 0.0));
  }
  return col;
}
