// Shared by every pass: frame globals, atmosphere, clouds, waves, lighting.
struct Globals {
  view_proj: mat4x4f,  // includes this frame's sub-pixel offset (see taa.wgsl)
  inv_view_proj: mat4x4f,
  shadow_vp: array<mat4x4f, 3>,
  planet: vec4f,     // xyz planet centre relative to camera, w radius
  sun_dir: vec4f,    // xyz, w illuminance
  east: vec4f,       // xyz, w wave offset x
  north: vec4f,      // xyz, w wave offset y
  up: vec4f,         // xyz, w camera altitude
  screen: vec4f,     // w, h, 1/w, 1/h
  time: vec4f,       // seconds, dt, exposure, frame
  cloud: vec4f,      // coverage, base altitude, thickness, ceiling (top including storm towers)
  light_pos: vec4f,  // fire light, xyz relative to camera, w radius of the glowing ball
  light_col: vec4f,
  vol_o: vec4f,      // explosion volume corner relative to camera, w cell size
  vol_u: vec4f,      // volume up axis, w = 1 while it holds smoke
  misc: vec4f,       // near plane, city glow 0..1, water quality, pixel angle
  csm: vec4f,        // world size of a shadow texel in each cascade, w = shadows on
  scorch: array<vec4f, 8>,
  crater: array<vec4f, 8>,  // xyz relative to camera, w radius
  prev_vp: mat4x4f,  // last frame's view-projection (without its sub-pixel offset), relative to last frame's camera
  cam_delta: vec4f,  // xyz camera movement since last frame, w = 1 if history is usable
  moon: vec4f,       // xyz direction to the moon, w illuminance
  vol_e: vec4f,      // volume east axis; w (set by the renderer) the share of the
                     // atmosphere march's steps taken (`atm_steps`): 1, or 0.5 below High clouds
  vol_n: vec4f,      // volume north axis; w = 1 under FABLE_OPT_OFF=water_edge (composite.wgsl fs)
  pshadow_vp: mat4x4f, // the aircraft's own shadow map
  pshadow: vec4f,    // texel size (m), depth bias, share of the layer used, w = 1 when in use
  taa: vec4f,        // xy this frame's sub-pixel offset (uv), z = 1 if the frame history is usable, w = 1 if the aircraft is drawn
  plane_inv: mat4x4f,  // camera-relative position to the aircraft's body frame
  plane_prev: mat4x4f, // body frame to last frame's clip space
  plane_box: array<vec4f, 2>, // centre and half size of the aircraft in its body frame
  weather: vec4f,    // xyz how far the wind has carried the clouds (planet frame, m), w haze (1 = clear day)
  storm: vec4f,      // xyz direction the wind aloft blows towards (planet frame), w number of storm cells
  storm_cells: array<vec4f, 32>, // per cell: centre (km on the sphere, cloud space) + tower radius (km); maturity
  det: array<vec4f, 16>, // ground material layers: m per repeat, m of relief, scale of the second sample, photo contrast (0 = procedural)
  photo_vp: mat4x4f, // photorealistic 3D tiles: camera-relative to coverage mask
  photo: vec4f,      // zone radius (m, 0 = off), dissolve band (m), mask valid
  photo_c: vec4f,    // zone centre on the ground, relative to the camera
  photo_sun: vec4f,  // the sun the photographs were taken under, w how sure (-1 = light them flat)
  // Destroyed buildings cut out of the 3D tiles (tiles3d.rs, Cut), G.photo.w of
  // them: floor middle (camera-relative) + radius; east axis + vertex count;
  // north axis + first vertex. Their outlines, two vertices (east, north) a row.
  cut: array<vec4f, 48>,
  cut_v: array<vec4f, 128>,
  // Cloud shadow maps (render.rs CloudShadow, see cloud_sun): the one lookups read
  // (t_cshadow), then the one being drawn. Each, six vec4s: anchor centre direction
  // + inner half size; east axis + outer half size; north axis + first layer; base
  // and top of its bands, centre (m, east and north of the anchor on its tangent
  // plane); the sun it was drawn for + valid; the cloud drift (G.weather.xyz) when
  // it was begun. The looked-up one is a cross-fade of two maps, with their sun and
  // drift blended the same way.
  cs: array<vec4f, 12>,
  alto: vec4f,       // mid-level cloud: amount 0..1, height (m); z snow lying 0..1 (with_snow), w snow falling 0..1
  // The ocean (ocean.rs): offset of its coordinates + direction the wind blows
  // toward (east, north); wind (m/s), significant wave height (m), whitecap
  // cover, 1; slope covariance (east-east, north-north, east-north) of the
  // ripples shorter than any cascade + the finest mip level the waves are
  // written to (0, or 1 at Low water quality: `ocean_lod`); then at 3 + 9 c + m the slope covariance
  // cascade c's mip level m no longer holds (see `ocean_lost`).
  ocean: array<vec4f, 30>,
  shore_vp: mat4x4f, // camera-relative to the shore distance field (shore.rs)
  shore: vec4f,      // field valid, metres per texel, farthest distance kept (m), side of the square (m)
  // Churned white water where a crash went in, two each: centre (camera-relative)
  // + radius (m, 0 = none); strength 0..1, age (s), seed.
  foam: array<vec4f, 16>,
  // The start screen: x = 1 while it shows (deep-space sky, no clouds). Set by
  // the renderer: y the cloud quality 0..2 (clouds.wgsl cloud_div), z = 1 with bloom off,
  // w the shadow filter: 0 the 3x3 PCF in four lookups (pcf9); with
  // FABLE_OPT_OFF=shadow_taps as before it, 1 four lookups half a texel out
  // (Low and Medium shadows) or 2 nine lookups (shadow_world).
  start: vec4f,
  snow: vec4f,       // falling snow: xyz direction the flakes stream from, w metres moved through them (composite.wgsl snowfall)
};
@group(0) @binding(0) var<uniform> G: Globals;
@group(0) @binding(1) var t_trans: texture_2d<f32>;
@group(0) @binding(2) var t_ms: texture_2d<f32>;
@group(0) @binding(3) var t_sky: texture_2d<f32>;
@group(0) @binding(4) var t_noise: texture_3d<f32>;
@group(0) @binding(5) var t_wave: texture_2d<f32>;
@group(0) @binding(6) var s_clamp: sampler;
@group(0) @binding(7) var s_repeat: sampler;
@group(0) @binding(8) var t_shadow: texture_depth_2d_array;
@group(0) @binding(9) var s_shadow: sampler_comparison;
@group(0) @binding(10) var s_aniso: sampler;
// Lighting of the smoke volume (see vol.wgsl): x = optical depth toward the
// sun, y = fire light, z = optical depth upward, w = density.
@group(0) @binding(11) var t_vlight: texture_3d<f32>;
// Ground material library (detail.wgsl): albedo + height, normal + cavity.
@group(0) @binding(12) var t_dalb: texture_2d_array<f32>;
@group(0) @binding(13) var t_dnrm: texture_2d_array<f32>;
@group(0) @binding(14) var s_detail: sampler;
// Where the photorealistic 3D tiles are drawn, seen from above (tiles3d.rs).
@group(0) @binding(15) var t_photo: texture_2d<f32>;
// Cloud shadow map, the two cascades (see cloud_sun).
@group(0) @binding(16) var t_cshadow: texture_2d_array<f32>;
// The ocean's three wave cascades (ocean.wgsl): layer c = (Dx, Dy, height,
// Dxy), layer c + 3 = (dh/dx, dh/dy, Dxx, Dyy); layer 6 the whitecaps on
// cascade 0's grid (foam cover, fresh foam, fold threshold on the Jacobian, 0).
@group(0) @binding(17) var t_ocean: texture_2d_array<f32>;
// Distance to the shore (shore.rs): metres to the nearest land, seaward
// direction (east, north), 1.
@group(0) @binding(18) var t_shore: texture_2d<f32>;

// GPU particles (explosions, smoke, dust, spray) and their simulation constants.
struct Particle {
  pos: vec3f, age: f32,
  vel: vec3f, life: f32,
  size0: f32, size1: f32, temp: f32, kind: f32,
  drag: f32, buoy: f32, seed: f32, alpha: f32,
};
// ve/vn: east and north axes of the smoke volume (w = cell size, lowest altitude).
// shift: how far the frame moved this step (every live particle is carried along).
struct PSim { anchor: vec4f, upv: vec4f, cam_r: vec4f, cam_u: vec4f, wind: vec4f, ve: vec4f, vn: vec4f, shift: vec4f };

// The light matrix of whichever shadow cascade is being rendered.
struct Cascade { vp: mat4x4f };

const PI = 3.14159265;
const ATM_H = 100000.0;
const RAYLEIGH = vec3f(5.802e-6, 13.558e-6, 33.1e-6);
const MIE_S = 3.996e-6;
const MIE_A = 4.4e-6;
const OZONE = vec3f(0.650e-6, 1.881e-6, 0.085e-6);
// Moonlight is sunlight, but night vision sees it blue (Purkinje shift).
const MOON_TINT = vec3f(0.72, 0.95, 1.45);
// Sodium and LED city light scattered back down by the haze.
const SKYGLOW = vec3f(1.0, 0.62, 0.36);

struct FsIn { @builtin(position) pos: vec4f, @location(0) uv: vec2f };
@vertex fn vs_full(@builtin(vertex_index) i: u32) -> FsIn {
  let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
  return FsIn(vec4f(p * 2.0 - 1.0, 0.0, 1.0), vec2f(p.x, 1.0 - p.y));
}

fn ign(p: vec2f) -> f32 { return fract(52.9829189 * fract(dot(p, vec2f(0.06711056, 0.00583715)))); }
fn hash1(p: f32) -> f32 { return fract(sin(p * 127.1) * 43758.5453); }
fn hash2(p: vec2f) -> f32 { return fract(sin(dot(p, vec2f(127.1, 311.7))) * 43758.5453); }
fn hash3(p: vec3f) -> f32 { return fract(sin(dot(p, vec3f(127.1, 311.7, 74.7))) * 43758.5453); }
fn remap(v: f32, a: f32, b: f32, c: f32, d: f32) -> f32 { return c + (v - a) / (b - a) * (d - c); }
fn luma(c: vec3f) -> f32 { return dot(c, vec3f(0.2126, 0.7152, 0.0722)); }
// The sky and bounce light in the colour being shaded, set by the lighting
// functions, so the G-buffer can record how much of a pixel ambient occlusion
// should darken (sunlight, lamps and glow stay as they are).
var<private> g_amb: vec3f;
fn amb_share(c: vec3f) -> f32 { return clamp(luma(g_amb) / max(luma(c), 1e-12), 0.0, 1.0); }

// Camera ray for a screen uv, and the point at a given reverse-Z depth.
fn view_ray(uv: vec2f) -> vec3f {
  let p = G.inv_view_proj * vec4f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 1.0, 1.0);
  return p.xyz / p.w;
}

// Numerically careful ray/sphere: oc = ray origin - centre. Returns (-1,-1) on a miss.
fn ray_sphere(oc: vec3f, rd: vec3f, r: f32) -> vec2f {
  let b = dot(oc, rd);
  let dp = length(oc - rd * b);
  if (dp > r) { return vec2f(-1.0); }
  let s = sqrt((r - dp) * (r + dp));
  return vec2f(-b - s, -b + s);
}

// ------------------------------------------------------------- atmosphere
fn atm_density(h: f32) -> vec3f {
  let hh = max(h, 0.0);
  // Haze scales the aerosol (Mie) layer only; it lives in the lowest kilometre or two.
  return vec3f(exp(-hh / 8000.0), exp(-hh / 1200.0) * G.weather.w, max(0.0, 1.0 - abs(hh - 25000.0) / 15000.0));
}
fn atm_ext(d: vec3f) -> vec3f { return RAYLEIGH * d.x + vec3f((MIE_S + MIE_A) * d.y) + OZONE * d.z; }

fn trans_uv(h_in: f32, mu: f32) -> vec2f {
  let rg = G.planet.w;
  let h = clamp(h_in, 0.0, ATM_H);
  let hh = sqrt(ATM_H * (2.0 * rg + ATM_H));
  let rho = sqrt(h * (2.0 * rg + h));
  let r = rg + h;
  let rt = rg + ATM_H;
  let disc = rt * rt - r * r * (1.0 - mu * mu);
  let d = max(0.0, -r * mu + sqrt(max(disc, 0.0)));
  let dmin = ATM_H - h;
  let dmax = rho + hh;
  return vec2f(clamp((d - dmin) / max(dmax - dmin, 1.0), 0.0, 1.0), rho / hh) * vec2f(255.0 / 256.0, 63.0 / 64.0) + vec2f(0.5 / 256.0, 0.5 / 64.0);
}
// Sunlight colour reaching a point at altitude h where the sun's zenith cosine is mu.
fn sun_trans(h: f32, mu: f32) -> vec3f {
  let rg = G.planet.w;
  let hc = max(h, 0.0);
  let mu_h = -sqrt(hc * (2.0 * rg + hc)) / (rg + hc);
  let vis = smoothstep(mu_h - 0.006, mu_h + 0.006, mu);
  return textureSampleLevel(t_trans, s_clamp, trans_uv(hc, mu), 0.0).rgb * vis;
}
fn ms_lut(h: f32, mu: f32) -> vec3f {
  return textureSampleLevel(t_ms, s_clamp, vec2f(mu * 0.5 + 0.5, clamp(h / ATM_H, 0.0, 1.0)) * (31.0 / 32.0) + 0.5 / 32.0, 0.0).rgb;
}
fn phase_r(c: f32) -> f32 { return 3.0 / (16.0 * PI) * (1.0 + c * c); }
fn phase_m(c: f32, g: f32) -> f32 {
  let g2 = g * g;
  return 3.0 / (8.0 * PI) * (1.0 - g2) * (1.0 + c * c) / ((2.0 + g2) * pow(1.0 + g2 - 2.0 * g * c, 1.5));
}

struct Scatter { L: vec3f, T: vec3f };
// Single scattering plus the multi-scatter LUT along a ray segment. oc is the
// ray origin relative to the planet centre.
// Steps for an atmosphere march that takes `n` at full quality.
fn atm_steps(n: i32) -> i32 { return max(i32(f32(n) * G.vol_e.w + 0.5), 4); }

fn atm_march(oc: vec3f, rd: vec3f, tmin: f32, tmax: f32, steps: i32, jit: f32) -> Scatter {
  let sun = G.sun_dir.xyz;
  let c = dot(rd, sun);
  let pr = phase_r(c);
  let pm = phase_m(c, 0.8);
  let moon = G.moon.xyz;
  let cm = dot(rd, moon);
  let prm = phase_r(cm);
  let pmm = phase_m(cm, 0.8);
  let mk = MOON_TINT * (G.moon.w / G.sun_dir.w);
  let moon_on = G.moon.w > 0.0;
  var T = vec3f(1.0);
  var L = vec3f(0.0);
  let len = max(tmax - tmin, 0.0);
  var tp = tmin;
  for (var i = 0; i < steps; i++) {
    let f = f32(i + 1) / f32(steps);
    let tn = tmin + len * f * f;
    let dt = tn - tp;
    let p = oc + rd * mix(tp, tn, jit);
    let r = length(p);
    let h = r - G.planet.w;
    let mu = dot(p / r, sun);
    let d = atm_density(h);
    let ext = atm_ext(d);
    let sr = RAYLEIGH * d.x;
    let sm = MIE_S * d.y;
    // Air in the shadow of a cloud gets no sunlight (light shafts through the gaps);
    // under a closed deck the light that remains is the deck's dull grey.
    let cl = cloud_sun(p, false);
    var msl = ms_lut(h, mu);
    msl = mix(msl, vec3f(luma(msl)), cl.closed * 0.75) * (1.0 - 0.45 * cl.closed);
    var S = sun_trans(h, mu) * cl.vis * (sr * pr + vec3f(sm * pm)) + msl * (sr + vec3f(sm));
    if (moon_on) {
      let mum = dot(p / r, moon);
      S += (sun_trans(h, mum) * (sr * prm + vec3f(sm * pmm)) + ms_lut(h, mum) * (sr + vec3f(sm))) * mk;
    }
    let st = exp(-ext * dt);
    L += T * S * (vec3f(1.0) - st) / max(ext, vec3f(1e-12));
    T *= st;
    tp = tn;
  }
  // 1.35: artistic lift so the sky and haze read as bright as they do in photographs.
  return Scatter(L * G.sun_dir.w * 1.35, T);
}

// Sky along a ray from the camera to space or the ground.
fn sky_march(rd: vec3f, steps: i32, jit: f32) -> Scatter {
  let oc = -G.planet.xyz;
  let top = ray_sphere(oc, rd, G.planet.w + ATM_H);
  if (top.y <= 0.0) { return Scatter(vec3f(0.0), vec3f(1.0)); }
  var tmax = top.y;
  let gnd = ray_sphere(oc, rd, G.planet.w);
  if (gnd.x > 0.0) { tmax = gnd.x; }
  return atm_march(oc, rd, max(top.x, 0.0), tmax, steps, jit);
}

fn sky_uv(rd: vec3f) -> vec2f {
  let el = asin(clamp(dot(rd, G.up.xyz), -1.0, 1.0));
  let az = atan2(dot(rd, G.north.xyz), dot(rd, G.east.xyz));
  let v = 0.5 + 0.5 * sign(el) * sqrt(abs(el) / (PI * 0.5));
  return vec2f(az / (2.0 * PI) + 0.5, clamp(1.0 - v, 0.5 / 128.0, 1.0 - 0.5 / 128.0));
}
// Cheap sky lookup (reflections, ambient).
fn sky_lut(rd: vec3f) -> vec3f { return textureSampleLevel(t_sky, s_repeat, sky_uv(rd), 0.0).rgb; }

// ------------------------------------------------------------------ clouds
// The layer spans G.cloud.y .. G.cloud.y + G.cloud.z. A weather field decides,
// per column, how much cloud there is, what kind (stratus, stratocumulus,
// cumulus) and how tall it grows; 3D noise carves the actual shapes. Rare storm
// cells (cumulonimbus) tower on up to G.cloud.w and spread into anvils there.
const CLOUD_SHAPE = 7600.0;   // tile size of the base shape noise, metres
const CLOUD_DETAIL = 1350.0;  // tile size of the erosion noise, metres
const CLOUD_FINE = 310.0;     // tile size of the close-up erosion octave, metres
const CIRRUS_H = 8800.0;

// cb: storm cell strength in this column, 0 outside, 1 at the core of the tower.
// sc: the storm cell nearest this column, xyz centre (km on the sphere, planet frame),
// w radius of its tower (km, 0 = none); sm its maturity (young towers have no anvil).
// cell: stratocumulus cell pattern, 1 at the heart of a cell, low between cells.
struct Weather { cov: f32, typ: f32, top: f32, cb: f32, sc: vec4f, sm: f32, cell: f32 };

// Storm cells (cumulonimbus) are picked on the CPU (Weather::storm_cells): a short
// list, nearest first. mask selects which of them to consider (bit i = cell i);
// the cloud march passes only those its ray can meet.
const ALL_STORMS = 0xffffffffu;
fn storm_find(q: vec3f, mask: u32) -> Weather {
  var w = Weather(0.0, 0.0, 0.0, 0.0, vec4f(0.0), 0.0, 0.0);
  let n = u32(G.storm.w);
  var best = -1.5;
  for (var i = 0u; i < n; i++) {
    if ((mask & (1u << i)) == 0u) { continue; }
    let c = G.storm_cells[2u * i];
    let val = 1.0 - distance(q, c.xyz) / (c.w * 2.6);
    if (val > best) { best = val; w.sc = c; w.sm = G.storm_cells[2u * i + 1u].x; }
  }
  return w;
}
// Bit i set when the ray from the camera (planet-centred oc, direction rd, up to
// tmax metres) passes within reach of storm cell i: the tower, its lobes and the
// anvil streaming downwind.
fn storm_mask(oc: vec3f, rd: vec3f, tmax: f32) -> u32 {
  var m = 0u;
  let n = u32(G.storm.w);
  let w = G.weather.xyz * 0.001;
  for (var i = 0u; i < n; i++) {
    let c = G.storm_cells[2u * i];
    let up = normalize(c.xyz - w);
    let r0 = c.w * 2.6 * 1.37 * 1000.0;
    // Bound: shifted downwind by half the anvil's stretch, plus the tower's lean.
    let cen = up * G.planet.w + G.storm.xyz * (r0 * 0.5);
    let v = cen - oc;
    let vh = v - up * dot(v, up);
    let rh = rd - up * dot(rd, up);
    let t = clamp(dot(vh, rh) / max(dot(rh, rh), 1e-6), 0.0, tmax);
    if (length(vh - rh * t) < r0 * 1.5 + 1200.0) { m |= 1u << i; }
  }
  return m;
}
// Strength (0..1) of a storm cell at a point hb metres above the cloud base: 1 at
// the centre, 0.6 at the tower wall, 0 at the edge of its anvil (2.6 radii out).
// The tower leans a little downwind; at anvil level the cell stretches out downwind.
fn storm_cb(pp: vec3f, hb: f32, sc: vec4f, sm: f32) -> f32 {
  if (sc.w <= 0.0) { return 0.0; }
  let hc = hb / (G.cloud.w - G.cloud.y);
  let q = normalize(pp) * 6371.0 + G.weather.xyz * 0.001 - G.storm.xyz * (hb * 0.0001);
  // Lobed rather than round: a few bulges around the tower that twist with height,
  // so it reads as a cluster of updrafts, not a cylinder. Less so in the anvil.
  var dv = q - sc.xyz;
  let dw = dot(dv, G.storm.xyz);
  dv -= G.storm.xyz * (max(dw, 0.0) * 0.5 * smoothstep(0.5, 0.8, sm) * smoothstep(0.6, 0.9, hc));
  let ang = atan2(dot(dv, G.north.xyz), dot(dv, G.east.xyz));
  let lob = 1.0 + (0.24 * sin(2.0 * ang + sm * 40.0 + hc * 1.7) + 0.13 * sin(3.0 * ang + sm * 73.0 - hc * 2.3)) * (1.0 - smoothstep(0.7, 0.9, hc) * 0.3);
  return clamp(1.0 - length(dv) / (sc.w * 2.6 * lob), 0.0, 1.0);
}
fn cloud_weather(pp: vec3f) -> Weather { return cloud_weather_m(pp, ALL_STORMS); }
fn cloud_weather_m(pp: vec3f, mask: u32) -> Weather {
  let q = normalize(pp) * 6371.0;  // km on the sphere
  let w = G.weather.xyz * 0.001;
  let hb = max(length(pp) - G.planet.w - G.cloud.y, 0.0);
  var st = Weather(0.0, 0.0, 0.0, 0.0, vec4f(0.0), 0.0, 0.0);
  if (mask != 0u && G.storm.w > 0.0) { st = storm_find(q + w, mask); }
  let cb = storm_cb(pp, hb, st.sc, st.sm);
  // Above the ordinary layer only storm towers reach (light probes from here only go up).
  if (hb >= G.cloud.z) { return Weather(0.0, 0.0, 0.0, cb, st.sc, st.sm, 0.0); }
  let big = textureSampleLevel(t_noise, s_repeat, q / 1700.0 + vec3f(0.13, 0.71, 0.37), 0.0);
  let mid = textureSampleLevel(t_noise, s_repeat, (q + w) / 173.0, 0.0);
  let cell = textureSampleLevel(t_noise, s_repeat, (q + w) / 29.0 + vec3f(0.53, 0.21, 0.87), 0.0);
  // Regional coverage: the slider, pushed around by weather systems and clearings.
  let c = G.cloud.x + (big.r - 0.5) * 0.4 + (mid.r - 0.5) * 0.45 + (cell.g - 0.62) * 0.2;
  let cov = clamp(c, 0.0, 1.0);
  // Broken skies grow cumulus; as it closes in the cloud flattens into stratocumulus, then stratus.
  let typ = clamp(1.0 - smoothstep(0.68, 0.98, cov) + (big.g - 0.62) * 0.6, 0.0, 1.0);
  // How tall the cumulus in this area grow: mostly fair-weather puffs, some towers.
  // Towers need broken sky around them; a closed deck caps the convection (storm
  // cells are separate and still punch through).
  let tower = smoothstep(0.25, 0.95, cov * 0.75 + cell.r * 0.55 - 0.1) * (1.0 - 0.75 * smoothstep(0.7, 0.92, cov));
  let cu_top = 0.28 + 0.72 * tower * tower;
  let top = mix(mix(0.1, 0.22, smoothstep(0.0, 0.5, typ)), cu_top, smoothstep(0.4, 1.0, typ));
  // Stratocumulus decks are a mosaic of cells 1-2 km across (Worley, columnar so each
  // cell is one lump). Looked up per column, so light probes reuse it.
  var cellv = 0.0;
  if (typ < 0.9) { cellv = textureSampleLevel(t_noise, s_repeat, (pp + G.weather.xyz - normalize(pp) * (hb * 0.85)) / 16000.0 + vec3f(0.41, 0.77, 0.19), 0.0).g; }
  return Weather(cov, typ, top, cb, st.sc, st.sm, cellv);
}
// How high a storm cell grows, as a share of base..ceiling: young towers stop short.
fn storm_top(sm: f32) -> f32 { return mix(0.3, 1.0, smoothstep(0.0, 0.55, sm)); }
// Height of a column's cloud above the base, metres: storm cores reach the ceiling.
fn column_top(wx: Weather) -> f32 {
  return mix(G.cloud.z * wx.top, (G.cloud.w - G.cloud.y) * storm_top(wx.sm), smoothstep(0.55, 0.8, wx.cb));
}
// Cloud density (0..1) given the weather of its column. detail = erode the edges with
// fine noise; fine = how much of the close-up octave to add (0 when far away).
fn cloud_shape_f(pp: vec3f, h: f32, wx: Weather, detail: bool, fine: f32) -> f32 {
  let hb = h - G.cloud.y;
  if (hb <= 0.0 || h >= G.cloud.w) { return 0.0; }
  let hn = hb / (G.cloud.z * wx.top);  // 0 at the base, 1 at the top of this column
  // The storm profile is re-evaluated here, not taken from the column: towers are
  // only a few km across, so light probes leave them quickly.
  let cb = storm_cb(pp, hb, wx.sc, wx.sm);
  // Storm envelope at this height (see below); skip the noise where even the biggest
  // turret bulge cannot reach it.
  var storm = false;
  var reach = 0.0;
  var thr = 1.0;
  var hc = 0.0;
  let mature = smoothstep(0.5, 0.8, wx.sm);
  if (cb > 0.0) {
    hc = hb / ((G.cloud.w - G.cloud.y) * storm_top(wx.sm));  // 0 at the base, 1 at this cell's top
    reach = pow(smoothstep(0.52, 0.93, hc), 1.6) * mature;
    // Young towers round off into a cauliflower dome; mature ones overshoot the anvil at the core.
    thr = mix(mix(0.6, 0.72, hc) + smoothstep(0.6, 1.0, hc) * (1.0 - mature) * 0.35, 0.06, reach);
    storm = hc < 1.0 && cb + 0.28 * mix(1.0, 0.55, reach) > thr;
  }
  let cumulus = hn < 1.0 && wx.cov > 0.02;
  if (!storm && !cumulus) { return 0.0; }
  let wind = G.weather.xyz;
  // Seen from far above, single puffs are sub-pixel: a smooth field is enough.
  let far = smoothstep(40000.0, 400000.0, G.up.w);
  let smooth_d = wx.cov * wx.cov * 0.5 * smoothstep(0.0, 0.2, hn) * smoothstep(1.0, 0.6, hn);
  if (far >= 1.0) { return smooth_d; }
  // For cumulus the base shape is stretched upwards (half the vertical frequency), so a
  // puff keeps its footprint as it climbs and grows into a tower instead of leaning
  // off into a flat smear.
  let n = textureSampleLevel(t_noise, s_repeat, (pp + wind - normalize(pp) * (hb * 0.5 * wx.typ)) / CLOUD_SHAPE, 0.0);
  let fbm = n.g * 0.625 + n.b * 0.25 + n.a * 0.125;
  let base = remap(n.r, fbm - 1.0, 1.0, 0.0, 1.0);
  var d = 0.0;
  if (cumulus) {
    // Flat, sharp bases (sharper for cumulus); tops that round off into domes as only
    // the strongest cores reach them. Stratus stays a slab with a soft top.
    // The profile scales the coverage, so the threshold rises with height: only the
    // strongest cores reach the column top and every puff keeps a crisp, dense edge
    // all the way up (scaling the density instead stopped them at half height).
    // Stratocumulus: each cell of the deck is thickest (darkest from below) in the
    // middle and thins towards its edges, where gaps open as the cover drops.
    let stw = 1.0 - smoothstep(0.45, 0.9, wx.typ);
    let tl = mix(1.0, mix(0.55, 1.0, smoothstep(0.25, 0.8, wx.cell)), stw);
    let bv = mix(base, base * 0.5 + wx.cell * 0.55 + 0.05, stw);
    let bottom = smoothstep(0.0, mix(150.0, 70.0, wx.typ), hb);
    let dome = 1.0 - pow(hn, mix(3.0, 1.6, wx.typ));
    let prof = bottom * mix(smoothstep(tl, tl * 0.7, hn), dome, wx.typ);
    let cv = mix(min(wx.cov * 1.15, 1.0), wx.cov, wx.typ);
    d = remap(bv, min(1.0 - cv * prof, 0.999), 1.0, 0.0, 1.0);
  }
  // Storm cell: a fat tower of stacked billows that narrows a little with height,
  // overshoots above the anvil at its core, and spreads into a flat-topped anvil
  // whose underside slopes back into the tower.
  var anvil = 0.0;
  if (storm) {
    anvil = reach * (1.0 - smoothstep(0.55, 0.85, cb));
    // Billowy turrets: Worley bulges (~2.5 km and ~1 km) push the tower wall in and out.
    let tn = textureSampleLevel(t_noise, s_repeat, (pp + wind) / (CLOUD_SHAPE * 2.6) + vec3f(0.71, 0.33, 0.12), 0.0);
    let turret = tn.g * 0.55 + n.g * 0.3 + n.b * 0.15;
    let cbp = cb + (turret - 0.5) * mix(0.55, 0.3, reach);
    let top = mix(1.0, 0.93 + 0.07 * smoothstep(0.65, 1.0, cb), mature);
    let env = smoothstep(thr, thr + 0.16, cbp) * smoothstep(0.0, 70.0, hb) * smoothstep(top, top - 0.03, hc);
    // Anvils are smooth ice sheets; the tower is solid and keeps the noise.
    let body = mix(0.6 + base * 0.6, 0.85 + base * 0.15, anvil);
    d = max(d, remap(body * env, 0.2, 1.0, 0.0, 1.0));
  }
  if (d <= 0.0) { return smooth_d * far; }
  if (detail) {
    // Curl-like warp so the erosion swirls instead of looking like stamped noise.
    let wv = textureSampleLevel(t_noise, s_repeat, (pp + wind * 1.15) / 2900.0 + vec3f(0.31, 0.62, 0.17), 0.0).gba - 0.62;
    let pe = pp + wind * 1.3 + wv * 420.0;
    let e = textureSampleLevel(t_noise, s_repeat, pe / CLOUD_DETAIL, 0.0);
    var ef = e.g * 0.625 + e.b * 0.25 + e.a * 0.125;
    if (fine > 0.0) {
      // Close up, a finer octave keeps the billows crisp instead of going soft.
      let e2 = textureSampleLevel(t_noise, s_repeat, pe / CLOUD_FINE + vec3f(0.17, 0.59, 0.43), 0.0);
      ef = mix(ef, ef * 0.72 + (e2.g * 0.7 + e2.b * 0.3) * 0.28, fine);
    }
    // Worley is brightest at cell centres. Eroding by its inverse keeps round
    // bulges (cauliflower billows); eroding by it directly leaves ragged, wispy
    // strands, which suits the undersides and the anvil's fibrous ice.
    let low = select(1.0 - smoothstep(0.02, 0.13, hn), 1.0 - smoothstep(80.0, 600.0, hb), storm);
    let wisp = max(low, anvil);
    let ed = mix(1.0 - ef, ef, wisp);
    d = remap(d, ed * mix(0.5, 0.62, wx.typ) * mix(1.0, 0.35, anvil), 1.0, 0.0, 1.0);
  }
  // Real cloud edges are abrupt: a few tens of metres from clear air to dense.
  d = smoothstep(0.0, mix(0.22, 0.16, wx.typ), d) * mix(0.55, 1.0, max(wx.typ, min(cb * 4.0, 1.0)));
  return mix(d, smooth_d, far);
}
fn cloud_shape(pp: vec3f, h: f32, wx: Weather, detail: bool) -> f32 { return cloud_shape_f(pp, h, wx, detail, 0.0); }
fn cloud_density(pp: vec3f, h: f32, cheap: bool) -> f32 {
  if (h <= G.cloud.y || h >= G.cloud.w) { return 0.0; }
  return cloud_shape(pp, h, cloud_weather(pp), !cheap);
}
// Sunlight through the clouds at a point (planet-centred), from the cloud shadow map.
// Its texels are rays along the sun from points on the cloud base, so any point
// below or inside the layer finds its ray by sliding along the sun to the base.
// Each holds the optical depth of cloud from the base up to 1/9, 4/9 and all of
// the way (base..top, bands squeezed towards the base where the cumulus are), and
// in w how closed the deck is there (1 = overcast).
// vis: the share of direct sunlight left (with the forward-scattered glow that
// thinner cloud lets through); closed: the deck overhead, for points under it.
// hq: smooth (cubic B-spline) filtering, for surfaces; the haze marches, which
// call it many times a pixel, take one bilinear fetch.
struct CloudSun { vis: f32, closed: f32 };
const CS_RES = 1024.0;
fn cs_fetch(uv: vec2f, layer: i32, hq: bool) -> vec4f {
  if (!hq) { return textureSampleLevel(t_cshadow, s_clamp, uv, layer, 0.0); }
  // Cubic B-spline from four bilinear taps: no texel corners in the shadow's outline.
  let p = uv * CS_RES - 0.5;
  let i = floor(p);
  let f = p - i;
  let f2 = f * f;
  let f3 = f2 * f;
  let w0 = (1.0 - 3.0 * f + 3.0 * f2 - f3) / 6.0;
  let w1 = (4.0 - 6.0 * f2 + 3.0 * f3) / 6.0;
  let w2 = (1.0 + 3.0 * f + 3.0 * f2 - 3.0 * f3) / 6.0;
  let w3 = f3 / 6.0;
  let g0 = w0 + w1;
  let g1 = w2 + w3;
  let c0 = (i - 0.5 + w1 / g0) / CS_RES;
  let c1 = (i + 1.5 + w3 / g1) / CS_RES;
  return g0.y * (g0.x * textureSampleLevel(t_cshadow, s_clamp, c0, layer, 0.0) + g1.x * textureSampleLevel(t_cshadow, s_clamp, vec2f(c1.x, c0.y), layer, 0.0))
       + g1.y * (g0.x * textureSampleLevel(t_cshadow, s_clamp, vec2f(c0.x, c1.y), layer, 0.0) + g1.x * textureSampleLevel(t_cshadow, s_clamp, c1, layer, 0.0));
}
fn cloud_sun(pc: vec3f, hq: bool) -> CloudSun {
  var o = CloudSun(1.0, 0.0);
  let base = G.cs[3].x;
  let top = G.cs[3].y;
  let r = length(pc);
  let h = r - G.planet.w;
  if (G.cs[4].w < 0.5 || G.start.x > 0.0 || h >= top) { return o; }
  // The sun the maps were drawn for (it differs from this frame's by a fraction
  // of a degree while the clock runs fast).
  let sun = G.cs[4].xyz;
  let mu = dot(pc / r, sun);
  let R = G.planet.w + base;
  // Exactly along the sun (a low sun's rays cross the base tens of km away): forward
  // to the base from below it, back down to it from inside the layer.
  var ts = ray_sphere(pc, sun, R).y;
  if (h >= base) {
    let s = ray_sphere(pc, -sun, R);
    ts = select((base - h) / max(mu, 0.04), -s.x, s.x > 0.0);
  }
  // The clouds have drifted on with the wind since the maps were drawn.
  let qn = normalize(pc + sun * ts + (G.weather.xyz - G.cs[5].xyz));
  let loc = vec2f(dot(qn, G.cs[1].xyz), dot(qn, G.cs[2].xyz)) / max(dot(qn, G.cs[0].xyz), 0.5) * R - G.cs[3].zw;
  let ui = loc / G.cs[0].w * 0.5 + 0.5;
  let uo = loc / G.cs[1].w * 0.5 + 0.5;
  let ei = max(abs(ui.x - 0.5), abs(ui.y - 0.5)) * 2.0;
  let eo = max(abs(uo.x - 0.5), abs(uo.y - 0.5)) * 2.0;
  if (eo >= 1.0) { return o; }
  var v: vec4f;
  if (ei < 0.8) {
    v = cs_fetch(ui, 0, hq);
  } else {
    v = cs_fetch(uo, 1, hq);
    if (ei < 1.0) { v = mix(cs_fetch(ui, 0, hq), v, smoothstep(0.8, 0.98, ei)); }
  }
  // Optical depth still above this point.
  let f = sqrt(clamp((h - base) / (top - base), 0.0, 1.0)) * 3.0;
  let below = select(select(mix(v.y, v.z, f - 2.0), mix(v.x, v.y, f - 1.0), f < 2.0), v.x * f, f < 1.0);
  let od = max(v.z - below, 0.0);
  let fade = smoothstep(0.85, 1.0, eo);
  let t = exp(-od);
  o.vis = mix(t + (1.0 - t) * 0.22 / (1.0 + 0.12 * od), 1.0, fade);
  o.closed = v.w * (1.0 - fade) * (1.0 - smoothstep(base, base + 600.0, h));
  return o;
}
// High cirrus: a thin streaky sheet at CIRRUS_H. Returns optical thickness.
fn cirrus(pp: vec3f) -> f32 {
  let q = pp + G.weather.xyz * 2.2;
  let m = textureSampleLevel(t_noise, s_repeat, q / vec3f(140000.0, 60000.0, 60000.0) + vec3f(0.7, 0.1, 0.4), 0.0);
  let amount = clamp((m.r - 0.42) * 2.6 + (m.g - 0.6) + G.cloud.x * 0.35, 0.0, 1.0);
  if (amount <= 0.0) { return 0.0; }
  let a = textureSampleLevel(t_noise, s_repeat, q / vec3f(42000.0, 5200.0, 5200.0) + (m.gba - 0.6) * 0.25, 0.0);
  let b = textureSampleLevel(t_noise, s_repeat, q / vec3f(9000.0, 1500.0, 1500.0) + (a.gba - 0.6) * 0.18, 0.0);
  let v = a.r * 0.7 + b.g * 0.18 + b.a * 0.12;
  // Wide threshold: soft veils and streaks, never isolated specks.
  return smoothstep(0.82 - amount * 0.35, 1.05 - amount * 0.15, v) * amount * 0.8;
}

// ------------------------------------------------------------------- waves
// Position in metres on the camera's local tangent plane, continuous as the camera moves.
fn wave_coord(p: vec3f) -> vec2f {
  return vec2f(dot(p, G.east.xyz) + G.east.w, dot(p, G.north.xyz) + G.north.w);
}
// Position in the ocean's own coordinates (m, east and north), which wrap at a
// multiple of every cascade's size (ocean.rs WRAP).
fn ocean_coord(p: vec3f) -> vec2f {
  return vec2f(dot(p, G.east.xyz), dot(p, G.north.xyz)) + G.ocean[0].xy;
}
// Slope covariance (east-east, north-north, east-north) of the waves that
// cascade c loses when read at mip level lv (ocean.rs `lost_slopes`).
fn ocean_lost(c: i32, lv: f32) -> vec3f {
  let l = clamp(lv, 0.0, 8.0);
  let i = i32(floor(l));
  let j = min(i + 1, 8);
  return mix(G.ocean[3 + c * 9 + i].xyz, G.ocean[3 + c * 9 + j].xyz, l - f32(i));
}
// A level to read the waves at, no finer than the grid the GPU runs (Low
// water quality writes a 128 grid into level 1 and leaves level 0 alone).
fn ocean_lod(lv: f32) -> f32 { return max(lv, G.ocean[2].w); }
fn ocean_size(c: i32) -> f32 { return select(select(6.0, 250.0 / 6.0, c == 1), 250.0, c == 0); }
// Sea state: how much of each wave cascade (about 250 m, 40 m and 6 m across)
// a stretch of water shows, by its class, latitude and how open it is (shore:
// 0 by the coast, 1 out on open water). Short fetch - bays, lakes, rivers -
// keeps the short chop and loses the long waves and the swell. ocean.rs
// `sea_state` mirrors it for the floating aircraft.
fn sea_state(wclass: f32, sinlat: f32, shore: f32) -> vec3f {
  if (wclass > 2.5) { return vec3f(0.0, 0.1, 0.45); }   // river
  if (wclass > 1.5) { return vec3f(0.05, 0.45, 0.85); } // lake
  let rough = 1.0 + 0.3 * smoothstep(0.55, 0.85, abs(sinlat));
  return vec3f(rough * mix(0.3, 1.0, shore), mix(0.75, 1.0, shore), 1.0);
}
// Sideways (x = east, y = north) and vertical displacement of the sea surface
// above ocean coordinates x, filtered for a vertex spacing of sp metres;
// w = sea_state.
fn ocean_disp(x: vec2f, sp: f32, w: vec3f) -> vec3f {
  var o = vec3f(0.0);
  for (var c = 0; c < 3; c++) {
    let L = ocean_size(c);
    let lod = ocean_lod(log2(max(2.0 * sp * 256.0 / L, 1.0)));
    if (lod < 8.0 && w[c] > 0.0) {
      o += textureSampleLevel(t_ocean, s_repeat, x / L, c, lod).xyz * w[c];
    }
  }
  return o;
}

// ---------------------------------------------------------------- lighting
// Cascaded sun shadows: each cascade covers a wider area around the aircraft
// at a coarser resolution; they cross-fade so there is no visible boundary.
// The aircraft's own shadow: a fine map a few metres across (drawn into the
// top-left corner of its layer, pshadow.z of it), used everywhere
// (so its shadow on the ground is as sharp as on its own wing).
// Nine bilinear compare lookups a whole texel apart (a 3x3 PCF) weigh the
// 4x4 texels under them 1-f, 1, 1, f along each axis (f: where `uv` falls
// between texels). Two lookups an axis, each placed between a pair of those
// texels by its share, weigh them exactly the same: the same filter in four
// lookups instead of nine. Returns the sum (out of 9, as the nine would).
fn pcf9(uv: vec2f, layer: i32, z: f32, dim: f32) -> f32 {
  let p = uv * dim - 0.5;
  let b = floor(p);
  let f = p - b;
  let wa = 2.0 - f;
  let wb = 1.0 + f;
  let ua = (b - 0.5 + 1.0 / wa) / dim;
  let ub = (b + 1.5 + f / wb) / dim;
  return wa.x * wa.y * textureSampleCompareLevel(t_shadow, s_shadow, ua, layer, z)
       + wb.x * wa.y * textureSampleCompareLevel(t_shadow, s_shadow, vec2f(ub.x, ua.y), layer, z)
       + wa.x * wb.y * textureSampleCompareLevel(t_shadow, s_shadow, vec2f(ua.x, ub.y), layer, z)
       + wb.x * wb.y * textureSampleCompareLevel(t_shadow, s_shadow, ub, layer, z);
}
fn plane_shadow(P: vec3f, N: vec3f) -> f32 {
  if (G.pshadow.w <= 0.0) { return 1.0; }
  let tex = G.pshadow.x;
  let q = G.pshadow_vp * vec4f(P + N * tex * 1.5 + G.sun_dir.xyz * tex, 1.0);
  let uv = q.xy * vec2f(0.5, -0.5) + 0.5;
  if (any(uv <= vec2f(0.002)) || any(uv >= vec2f(0.998)) || q.z <= 0.0 || q.z >= 1.0) { return 1.0; }
  let dim = f32(textureDimensions(t_shadow).x);
  if (G.start.w < 0.5) { return pcf9(uv * G.pshadow.z, 3, q.z - G.pshadow.y, dim) / 9.0; }
  let texel = 1.0 / dim;
  var sh = 0.0;
  for (var j = -1; j <= 1; j++) { for (var i = -1; i <= 1; i++) {
    sh += textureSampleCompareLevel(t_shadow, s_shadow, uv * G.pshadow.z + vec2f(f32(i), f32(j)) * texel, 3, q.z - G.pshadow.y);
  } }
  return sh / 9.0;
}
fn shadow_at(P: vec3f, N: vec3f) -> f32 {
  if (G.csm.w <= 0.0) { return 1.0; }
  return shadow_world(P, N) * plane_shadow(P, N);
}
fn shadow_world(P: vec3f, N: vec3f) -> f32 {
  var acc = 0.0;
  var rem = 1.0;
  // (the map is SHADOW_RES or half of it on a side: render.rs set_shadow_res)
  let texel = 1.0 / f32(textureDimensions(t_shadow).x);
  for (var c = 0; c < 3; c++) {
    let tex = G.csm[c];
    let q = G.shadow_vp[c] * vec4f(P + N * tex * 1.8 + G.sun_dir.xyz * tex * 0.8, 1.0);
    let uv = q.xy * vec2f(0.5, -0.5) + 0.5;
    let e = min(min(uv.x, 1.0 - uv.x), min(uv.y, 1.0 - uv.y));
    if (e <= 0.0 || q.z <= 0.0 || q.z >= 1.0) { continue; }
    var sh = 0.0;
    if (G.start.w < 0.5) {
      sh = pcf9(uv, c, q.z - 0.00003, 1.0 / texel);
    } else if (G.start.w < 1.5) {
      // (FABLE_OPT_OFF=shadow_taps, Low and Medium shadows: four filtered
      // lookups half a texel out, nearly the same 3x3 footprint)
      for (var j = 0; j < 2; j++) { for (var i = 0; i < 2; i++) {
        sh += 2.25 * textureSampleCompareLevel(t_shadow, s_shadow, uv + (vec2f(f32(i), f32(j)) - 0.5) * texel, c, q.z - 0.00003);
      } }
    } else {
      for (var j = -1; j <= 1; j++) { for (var i = -1; i <= 1; i++) {
        sh += textureSampleCompareLevel(t_shadow, s_shadow, uv + vec2f(f32(i), f32(j)) * texel, c, q.z - 0.00003);
      } }
    }
    let w = smoothstep(0.0, 0.07, e);
    acc += rem * w * sh / 9.0;
    rem *= 1.0 - w;
    if (rem < 0.01) { break; }
  }
  return acc + rem;
}

// ---------------------------------------------------------------- craters
// Ground knocked out by impacts. Returns the height change in w and the
// world-space slope (to bend normals) in xyz, for a point relative to camera.
fn crater_field(P: vec3f) -> vec4f {
  var o = vec4f(0.0);
  for (var i = 0; i < 8; i++) {
    let c = G.crater[i];
    if (c.w <= 0.0) { continue; }
    let v = P - c.xyz;
    if (abs(v.x) > c.w * 2.0 || abs(v.y) > c.w * 2.0 || abs(v.z) > c.w * 2.0) { continue; }
    let upn = normalize(c.xyz - G.planet.xyz);
    let hv = v - upn * dot(v, upn);
    let r = length(hv);
    let d = r / c.w;
    if (d > 1.8) { continue; }
    let D = c.w * 0.34;
    let k = (d - 1.1) / 0.24;
    let rim = 0.22 * exp(-k * k);
    var h = rim;
    var dh = rim * (-2.0 * k / 0.24);
    if (d < 1.0) {
      let b = 1.0 - d * d;
      h -= b * b;
      dh += 4.0 * d * b;
    }
    o += vec4f(hv / max(r, 0.001) * (dh * D / c.w), h * D);
  }
  return o;
}

// ------------------------------------------------------------------ roads
fn stripe(x: f32, w: f32, fw: f32) -> f32 {
  return clamp((w * 0.5 - abs(x)) / fw + 0.5, 0.0, 1.0) * min(1.0, w / fw);
}
// Procedural road surface. across/along in metres, half = half width,
// kind = rclass | oneway<<4 | yellow<<5 | sidewalk<<6 | lanes<<8 | parking<<12 | keep-left<<13 | lit<<14,
// endd = distance to the nearest junction, measured so the junction's edge is at
// `half`; app = +1 where `along` runs away from that junction, -1 toward it;
// fw = pixel size in metres. Returns albedo and coverage.
fn road_surface(across: f32, along: f32, half: f32, kind: u32, endd: f32, app: f32, fw_in: f32) -> vec4f {
  let rclass = kind & 15u;
  let oneway = (kind & 16u) != 0u;
  let yellow = (kind & 32u) != 0u;
  let walk = (kind & 64u) != 0u;
  let lanes = f32((kind >> 8u) & 15u);
  let parking = (kind & 4096u) != 0u;
  let keep_left = (kind & 8192u) != 0u;
  let fw = max(fw_in, 0.002);
  let uv = vec2f(across, along);
  let n1 = textureSampleLevel(t_wave, s_repeat, uv / 31.0, 0.0).a;
  let n2 = textureSampleLevel(t_wave, s_repeat, uv / 3.7, log2(max(fw * 256.0 / 3.7, 1.0))).a;
  let n3 = textureSampleLevel(t_wave, s_repeat, uv / 0.9, log2(max(fw * 256.0 / 0.9, 1.0))).a;
  let ax = abs(across);
  var edge = clamp((half - ax) / max(fw, 0.12) + 0.5, 0.0, 1.0);
  if (rclass == 9u) {
    // Railway: ballast, sleepers, two rails.
    var col = vec3f(0.16, 0.14, 0.12) * (0.7 + 0.6 * n3);
    let sl = stripe(fract(along / 0.65) - 0.5, 0.36, fw / 0.65) * step(ax, 1.25);
    col = mix(col, vec3f(0.09, 0.075, 0.06), sl);
    col = mix(col, vec3f(0.42, 0.40, 0.38), stripe(ax - 0.72, 0.09, fw));
    return vec4f(col, edge * 0.9);
  }
  if (rclass == 7u) {
    // Dirt track: two worn ruts, ragged edges.
    let rut = stripe(ax - 0.8, 0.5, fw);
    let col = mix(vec3f(0.20, 0.16, 0.11), vec3f(0.27, 0.22, 0.16), rut) * (0.75 + 0.5 * n2);
    return vec4f(col, clamp((half - ax + (n2 - 0.5) * 0.9) / max(fw, 0.25), 0.0, 1.0) * 0.7);
  }
  if (rclass == 8u) {
    let col = vec3f(0.30, 0.29, 0.27) * (0.8 + 0.4 * n2);
    return vec4f(col, edge * 0.8);
  }
  // Asphalt, newer and darker on the big roads.
  var tone = 0.050;
  if (rclass <= 1u) { tone = 0.036; } else if (rclass >= 5u) { tone = 0.062; }
  if (rclass == 6u) { tone = 0.075; }
  var col = vec3f(tone, tone, tone * 1.04) * (0.72 + 0.35 * n1 + 0.25 * n3);
  // Patches and cracks close up.
  col *= 1.0 - 0.25 * smoothstep(0.62, 0.7, n2) * (1.0 - smoothstep(0.02, 0.2, fw));
  let close = 1.0 - smoothstep(0.03, 0.25, fw);
  if (close > 0.0 && rclass <= 6u) {
    // Trench repairs: long strips of newer, darker asphalt, sealed along their edges.
    let pid = floor(along / 23.0);
    let ph = hash2(vec2f(pid, half * 7.0));
    if (ph < 0.3) {
      let px = across - (hash1(pid + 3.0) - 0.5) * half;
      let py = fract(along / 23.0) * 23.0 - 4.0;
      let pl = 4.0 + 13.0 * hash1(pid + 9.0);
      let inside = stripe(px, 1.3, fw) * stripe(py - pl * 0.5, pl, fw);
      let seal = inside - stripe(px, 1.14, fw) * stripe(py - pl * 0.5, pl - 0.16, fw);
      col = mix(col, col * 0.72, inside * close);
      col = mix(col, vec3f(0.012), clamp(seal, 0.0, 1.0) * close * 0.8);
    }
    // Tar-sealed cracks wandering along the lanes, oil down the middle of each one.
    let wob = (textureSampleLevel(t_wave, s_repeat, vec2f(along / 19.0, floor(across / 3.3) * 0.37), 0.0).a - 0.5) * 2.2;
    let crack = stripe(fract(across / 3.3 + 0.5) * 3.3 - 1.65 - wob, 0.045, fw) * smoothstep(0.52, 0.6, n1);
    col = mix(col, vec3f(0.012), crack * close * 0.75);
    // Manhole covers.
    let mid = floor(along / 41.0);
    let mc = vec2f(across - (hash1(mid * 1.3 + half) - 0.5) * half, fract(along / 41.0) * 41.0 - 20.0);
    let ring = clamp((0.34 - length(mc)) / fw + 0.5, 0.0, 1.0) * step(hash1(mid + 17.0), 0.55);
    col = mix(col, vec3f(0.035, 0.032, 0.03) * (0.7 + 0.6 * step(0.5, fract(length(mc) / 0.11))), ring * close);
  }
  var road_half = half;
  if (walk) {
    road_half = half - 1.9;
    // Kerb and paved sidewalk.
    let sw = smoothstep(road_half - fw, road_half + fw, ax);
    let joint = 1.0 - 0.35 * stripe(fract(along / 1.5) - 0.5, 0.03, fw / 1.5);
    let pave = vec3f(0.30, 0.29, 0.27) * (0.8 + 0.3 * n2) * joint;
    col = mix(col, pave, sw);
    col = mix(col, vec3f(0.38, 0.37, 0.35), stripe(ax - road_half, 0.16, fw));
  }
  if (rclass <= 5u && lanes > 0.5) {
    let fade = smoothstep(half + 5.6, half + 6.6, endd);
    var mk = 0.0;
    var mcol = vec3f(0.62);
    let park = select(0.0, 2.2, parking);
    let drive_half = road_half - park;
    let lw = min(2.0 * drive_half / lanes, 3.7);
    let x0 = -lw * lanes * 0.5;
    let dash = step(fract(along / 12.0), 0.25);
    if (rclass <= 4u) {
      // Lane lines.
      let li = (across - x0) / lw;
      let near = round(li);
      if (near > 0.5 && near < lanes - 0.5) {
        let x = (li - near) * lw;
        let centre = !oneway && abs(near - lanes * 0.5) < 0.25;
        if (centre) {
          if (yellow) {
            mk = max(stripe(x - 0.12, 0.11, fw), stripe(x + 0.12, 0.11, fw));
            mcol = vec3f(0.62, 0.42, 0.04);
          } else {
            mk = stripe(x, 0.14, fw);
          }
        } else {
          mk = stripe(x, 0.12, fw) * dash;
        }
      }
      // Edge lines on the bigger roads.
      if (rclass <= 3u) { mk = max(mk, stripe(ax - (lw * lanes * 0.5 + 0.15), 0.12, fw) * select(1.0, 0.0, parking)); }
    } else if (!oneway && lanes > 1.5) {
      mk = stripe(across, 0.11, fw) * dash * 0.8;
      if (yellow) { mcol = vec3f(0.62, 0.42, 0.04); }
    }
    mk *= fade;
    // Where streets meet: a stop line for arriving traffic, and a zebra
    // crossing from kerb to kerb if people walk here.
    if (endd < half + 6.0) {
      let side = select(1.0, -1.0, keep_left) * app;
      let mine = select(clamp(-across * side / fw + 0.5, 0.0, 1.0), select(0.0, 1.0, app < 0.0), oneway);
      let stop = stripe(endd - (half + 5.2), 0.4, fw) * mine * step(ax, drive_half + park);
      var zebra = 0.0;
      if (walk) { zebra = stripe(endd - (half + 2.5), 3.0, fw) * stripe(fract(across / 1.1) - 0.5, 0.5, fw / 1.1) * step(ax, road_half - 0.2); }
      let jm = max(stop, zebra);
      mk = max(mk * (1.0 - jm), jm);
      mcol = mix(mcol, vec3f(0.62), jm);
    }
    // Paint wears.
    mk *= 0.55 + 0.45 * smoothstep(0.3, 0.6, n2);
    col = mix(col, mcol, mk);
  }
  return vec4f(col, edge * 0.9);
}

// Street lights: must match `lamp_spec` in roads.rs. Returns the light landing
// on the road surface from the lamps up and down the street (1 right under one).
fn lamp_pool(across: f32, along: f32, half: f32, kind: u32) -> f32 {
  if ((kind & 16384u) == 0u) { return 0.0; }
  let big = (kind & 15u) <= 1u;
  let sp = select(36.0, 48.0, big);
  let h = select(8.0, 11.0, big);
  let off = half - select(3.3, 2.9, big);
  let k0 = floor(along / sp);
  var o = 0.0;
  for (var i = -1; i <= 1; i++) {
    let k = k0 + f32(i);
    let sgn = 1.0 - 2.0 * abs(k - 2.0 * floor(k * 0.5 + 0.25));
    let d = vec2f(across + sgn * off, along - (k + 0.5) * sp);
    let r2 = dot(d, d) + h * h;
    o += h * h * h / (r2 * sqrt(r2));
  }
  return o;
}
const LAMP_COL = vec3f(1.0, 0.72, 0.40);
// 1 once it is dark enough for the lights to be on.
// ------------------------------------------------------------ 3D tiles zone
// Inside the zone of photorealistic 3D tiles our own ground, buildings, trees
// and cars give way to the mesh wherever it has been drawn into the coverage
// mask. Across the band at the edge of the zone the two are dithered into
// each other pixel by pixel, the same way on both sides (TAA smooths it).
fn photo_fade(P: vec3f) -> f32 {
  let q = P - G.photo_c.xyz;
  let up = normalize(G.photo_c.xyz - G.planet.xyz);
  let d = length(q - up * dot(q, up));
  return smoothstep(G.photo.x - G.photo.y, G.photo.x, d);
}
// The mesh is drawn at this pixel (P on it).
fn photo_keep(P: vec3f, px: vec2f) -> bool {
  return ign(px + 5.588238 * (G.time.w % 64.0)) >= photo_fade(P);
}
// The mask at P: r = the mesh is there, g = height of our water surface
// there (far below everything where there is none).
fn photo_mask(P: vec3f) -> vec2f {
  let q = G.photo_vp * vec4f(P, 1.0);
  let uv = q.xy * vec2f(0.5, -0.5) + 0.5;
  if (any(uv <= vec2f(0.0)) || any(uv >= vec2f(1.0))) { return vec2f(0.0, -1e4); }
  return textureSampleLevel(t_photo, s_clamp, uv, 0.0).rg;
}
fn photo_covered(P: vec3f) -> bool {
  if (G.photo.z <= 0.0) { return false; }
  return photo_mask(P).r > 0.5;
}
// Height above the planet of a camera-relative point, good to a few cm
// within tens of km (a straight length() loses half a metre in f32).
fn height_of(P: vec3f) -> f32 {
  let v = dot(P, G.up.xyz);
  return G.up.w + v + (dot(P, P) - v * v) / (2.0 * G.planet.w);
}
// Whether P lies inside a destroyed building's outline (and, with `floor`,
// above its floor): the mesh is not drawn there and our own scenery is.
fn cut_vert(k: u32) -> vec2f {
  let r = G.cut_v[k / 2u];
  return select(r.xy, r.zw, (k & 1u) == 1u);
}
fn photo_cut_at(P: vec3f, floor: bool) -> bool {
  let n = u32(G.photo.w);
  for (var i = 0u; i < n; i++) {
    let h0 = G.cut[i * 3u];
    let d = P - h0.xyz;
    let e = G.cut[i * 3u + 1u];
    let nn = G.cut[i * 3u + 2u];
    let q = vec2f(dot(d, e.xyz), dot(d, nn.xyz));
    if (dot(q, q) > h0.w * h0.w) { continue; }
    if (floor && dot(d, cross(e.xyz, nn.xyz)) < 0.0) { continue; }
    let first = u32(nn.w);
    let count = u32(e.w);
    var c = false;
    var pj = cut_vert(first + count - 1u);
    for (var k = 0u; k < count; k++) {
      let pk = cut_vert(first + k);
      if ((pk.y > q.y) != (pj.y > q.y) && q.x < (pj.x - pk.x) * (q.y - pk.y) / (pj.y - pk.y) + pk.x) { c = !c; }
      pj = pk;
    }
    if (c) { return true; }
  }
  return false;
}
fn photo_cut(P: vec3f) -> bool { return photo_cut_at(P, true); }
// Our own scenery at P stands under the mesh, so casts no shadow (the mesh
// casts its own): the shadow passes see the coverage mask; the band at the
// edge of the zone switches over in its middle.
fn photo_shadow_gone(P: vec3f) -> bool {
  return G.photo.z > 0.0 && photo_fade(P) < 0.5 && photo_mask(P).r > 0.5 && !photo_cut_at(P, false);
}
// False for pipelines drawn only while there is no 3D tiles zone (G.photo.z
// is 0, render.rs): their discard for it compiled out, the GPU's hidden-surface
// removal works on them (FABLE_OPT_OFF=photo_solid).
override PHOTO: bool = true;
// Our own scenery at P (pixel px) is hidden by the mesh.
fn photo_hides(P: vec3f, px: vec2f) -> bool {
  return PHOTO && photo_covered(P) && photo_keep(P, px) && !photo_cut_at(P, false);
}

fn night_at(upn: vec3f) -> f32 { return 1.0 - smoothstep(-0.08, 0.1, dot(upn, G.sun_dir.xyz)); }
// Moonlight arriving at a point (dimmed by the atmosphere near the horizon and
// by the cloud deck as a whole).
fn moon_light(upn: vec3f, h: f32) -> vec3f {
  if (G.moon.w <= 0.0) { return vec3f(0.0); }
  return sun_trans(h, dot(upn, G.moon.xyz)) * G.moon.w * MOON_TINT * mix(1.0, 0.35, G.cloud.x);
}

// ------------------------------------------------------------ smoke volume
const VOL_N = vec3f(40.0, 40.0, 48.0);
fn vol_uvw(P: vec3f) -> vec3f {
  let q = P - G.vol_o.xyz;
  return vec3f(dot(q, G.vol_e.xyz), dot(q, G.vol_n.xyz), dot(q, G.vol_u.xyz)) / (G.vol_o.w * VOL_N);
}
// Smoke lighting at a point relative to camera; zero outside the volume.
fn vol_light(P: vec3f) -> vec4f {
  if (G.vol_u.w <= 0.0) { return vec4f(0.0); }
  let uvw = vol_uvw(P);
  let m = min(uvw, 1.0 - uvw);
  let e = min(m.x, min(m.y, m.z));
  if (e <= 0.0) { return vec4f(0.0); }
  return textureSampleLevel(t_vlight, s_clamp, uvw, 0.0) * smoothstep(0.0, 0.05, e);
}

struct Light { sun: vec3f, amb: vec3f, upn: vec3f, h: f32 };
// Sun and sky light at a point, shaded by the smoke of any fire nearby.
fn light_at(P: vec3f) -> Light {
  var l = light_at_nv(P);
  if (G.vol_u.w > 0.0) {
    let v = vol_light(P);
    l.sun *= exp(-v.x);
    l.amb *= 0.55 + 0.45 * exp(-v.z * 0.7);
  }
  return l;
}
// Sun and sky light arriving at a point (relative to camera).
fn light_at_nv(P: vec3f) -> Light {
  let pp = P - G.planet.xyz;
  let r = length(pp);
  let upn = pp / r;
  let h = r - G.planet.w;
  let mu = dot(upn, G.sun_dir.xyz);
  let sun = sun_trans(h, mu) * G.sun_dir.w * cloud_sun(pp, true).vis;
  // Skylight falls off steeply once the sun is down: about half a decade per
  // 3 degrees of twilight (the exp term), so a street at blue hour is far
  // darker than the glowing sky above it rather than as bright as overcast day.
  let day = min(smoothstep(-0.12, 0.3, mu), 0.06 * exp(20.0 * mu));
  let dusk = smoothstep(-0.2, 0.0, mu) * (1.0 - smoothstep(0.0, 0.3, mu));
  // Skylight is a pale blue, not a saturated one (the sky is bright near the
  // horizon and around the sun, which whitens what it adds up to), and the
  // white undersides of clouds grey it further as the cover grows. In
  // twilight it turns the deep blue-violet of the blue hour.
  var skyc = mix(vec3f(0.30, 0.43, 0.70), vec3f(0.47, 0.49, 0.52), clamp(G.cloud.x, 0.0, 1.0) * 0.7);
  skyc = mix(skyc, vec3f(0.38, 0.36, 0.62), dusk * 0.6);
  var amb = skyc * 0.11 * day * G.sun_dir.w;
  // Night: the moonlit sky, starlight and airglow, and the city's own glow
  // reflected back down off the haze. It takes over as the twilight fades.
  let mum = dot(upn, G.moon.xyz);
  let night = 1.0 - smoothstep(-0.12, 0.02, mu);
  amb += vec3f(0.30, 0.45, 0.85) * 0.16 * smoothstep(-0.1, 0.35, mum) * G.moon.w * mix(1.0, 0.7, G.cloud.x);
  amb += (vec3f(0.0016, 0.0024, 0.0048) + SKYGLOW * 0.0022 * G.misc.y * (0.6 + 0.6 * G.cloud.x)) * night;
  return Light(sun, amb, upn, h);
}
// What a mirror at P sees along R: the sky (with clouds), or below the
// horizon the hazy ground and city.
fn env_reflect(P: vec3f, R: vec3f, upn: vec3f, clouds: bool) -> vec3f {
  let ru = dot(R, upn);
  var sky = sky_lut(normalize(R + upn * max(0.015 - ru, 0.0)));
  if (clouds && ru > 0.02) {
    let pp = P - G.planet.xyz;
    let h = length(pp) - G.planet.w;
    let mid = G.cloud.y + 400.0;
    if (h < mid) {
      let tc = (mid - h) / ru;
      if (tc < 60000.0) {
        let cd = cloud_density(pp + R * tc, mid, true);
        let mu = dot(upn, G.sun_dir.xyz);
        let cl = sun_trans(mid, mu) * G.sun_dir.w * 0.05 + sky_lut(upn) * 1.2;
        sky = mix(sky, cl, (1.0 - exp(-cd * 7.0)) * (1.0 - smoothstep(25000.0, 60000.0, tc)));
      }
    }
  }
  if (ru > 0.01) { return sky; }
  let hz = R - upn * ru;
  let hl = length(hz);
  let hzn = select(G.east.xyz, hz / max(hl, 1e-4), hl > 1e-3);
  let mu = dot(upn, G.sun_dir.xyz);
  let gnd = vec3f(0.06, 0.06, 0.055) * (sun_trans(0.0, mu) * max(mu, 0.0) * G.sun_dir.w / PI + sky_lut(upn) * 0.6);
  // Near the horizon the ground fades into the haze.
  let below = mix(sky_lut(normalize(hzn + upn * 0.01)), gnd, smoothstep(0.0, 0.3, -ru));
  return mix(sky, below, smoothstep(0.01, -0.05, ru));
}
// Light of a fire: a glowing ball of radius light_pos.w whose light falls off
// with the square of the distance beyond it.
fn point_light(P: vec3f, N: vec3f) -> vec3f {
  let d = G.light_pos.xyz - P;
  let r2 = dot(d, d);
  let rad = G.light_pos.w;
  if (rad <= 0.0) { return vec3f(0.0); }
  let att = rad * rad / (rad * rad + r2);
  return G.light_col.rgb * att * max(dot(N, d * inverseSqrt(max(r2, 1e-6))) * 0.7 + 0.3, 0.0);
}
// Smooth 3D value noise for breaking up the snow cover.
fn snow_vn(p: vec3f) -> f32 {
  let i = floor(p);
  let f = fract(p);
  let u = f * f * (3.0 - 2.0 * f);
  let a = mix(mix(hash3(i), hash3(i + vec3f(1.0, 0.0, 0.0)), u.x), mix(hash3(i + vec3f(0.0, 1.0, 0.0)), hash3(i + vec3f(1.0, 1.0, 0.0)), u.x), u.y);
  let b = mix(mix(hash3(i + vec3f(0.0, 0.0, 1.0)), hash3(i + vec3f(1.0, 0.0, 1.0)), u.x), mix(hash3(i + vec3f(0.0, 1.0, 1.0)), hash3(i + vec3f(1.0, 1.0, 1.0)), u.x), u.y);
  return mix(a, b, u.z);
}
// Snow lying on the ground, roofs and tree crowns (G.alto.z the amount):
// on what faces up, more of it higher up. Light snow whitens only flat
// ground in the mountains; deep snow all but the steep slopes everywhere.
// It changes the albedo only, so the sun, shadows and relief still show;
// steep rock, dark ground (rock, forest) and noisy drifts break it up, and
// what lies thin lets the ground below show through.
fn with_snow(albedo: vec3f, P: vec3f, N: vec3f) -> vec3f {
  let s = G.alto.z;
  if (s <= 0.001) { return albedo; }
  let r = P - G.planet.xyz;
  let h = length(r) - G.planet.w;
  let amt = clamp(s * (0.55 + h / 1600.0), 0.0, 1.0);
  let slope = dot(N, r * inverseSqrt(max(dot(r, r), 1.0)));
  // (in ground coordinates: P is camera-relative and would swim)
  let q = vec3f(wave_coord(P), h);
  let n = snow_vn(q / 140.0) * 0.6 + snow_vn(q / 31.0) * 0.4;
  let lum = dot(albedo, vec3f(0.3, 0.59, 0.11));
  let dark = 1.0 - smoothstep(0.04, 0.2, lum);
  // Steep ground sheds it: well short of vertical even in deep snow.
  let th = mix(0.97, 0.72, amt) + (n - 0.5) * 0.12 + dark * 0.08;
  let on_slope = smoothstep(th, th + 0.1, slope);
  let cover = amt * 1.15 + (n - 0.5) * 0.8 - dark * 0.55;
  let k = on_slope * smoothstep(0.35, 0.75, cover) * smoothstep(0.05, 0.35, amt) * 0.94;
  // Snow ~0.8, a touch blue, a little uneven, keeping a trace of what is under it.
  let snow = vec3f(0.74, 0.78, 0.84) * (0.9 + 0.2 * n) * mix(1.0, 0.75 + 1.25 * lum, 0.25);
  return mix(albedo, snow, k);
}
fn shade(P: vec3f, N: vec3f, albedo: vec3f) -> vec3f {
  let l = light_at(P);
  let ndl = max(dot(N, G.sun_dir.xyz), 0.0);
  let sh = shadow_at(P, N);
  let sky = 0.55 + 0.45 * dot(N, l.upn);
  let moon = moon_light(l.upn, l.h) * max(dot(N, G.moon.xyz), 0.0);
  // Light bounced off the ground: what keeps walls and overhangs from going
  // flat blue-black in shade.
  let mu = max(dot(l.upn, G.sun_dir.xyz), 0.0);
  let bounce = vec3f(0.15, 0.14, 0.12) * (l.sun * mu * 0.8 / PI + l.amb) * (0.5 - 0.5 * dot(N, l.upn));
  g_amb = albedo * (l.amb * sky + bounce);
  return albedo * ((l.sun * ndl * sh + moon) / PI + l.amb * sky + bounce + point_light(P, N));
}
