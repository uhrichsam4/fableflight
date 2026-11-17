// Wind-driven ocean (ocean.rs). Three tiling cascades (about 250 m, 40 m and
// 6 m across), each a 256 x 256 inverse FFT of a JONSWAP wind sea with
// Donelan-Banner directional spread, plus a long swell, rebuilt every frame.
//
// `rows` evaluates the spectrum at this frame's time and transforms each row;
// `cols` transforms the columns and writes the result into one 6-layer
// texture: layer c = (Dx, Dy, height, Dxy), layer c + 3 = (dh/dx, dh/dy, Dxx, Dyy)
// for cascade c, the D terms sideways displacement (already scaled by the
// choppiness) and its derivatives. `foam` keeps the whitecaps (layer 6, see
// there) and `foam_ctl` steers how much of the sea they cover. `mip` builds
// the mip chain.
//
// The grid is NN x NN (the Water quality: 256 at High, 128 at Low, set per
// pipeline, ocean.rs `OceanGpu`). The workgroups and the shared FFT buffer
// stay sized for 256, the most; at 128 half the threads sit out. A 128 grid
// is written into level 1 of the texture (128 x 128), so the shaders that
// read it see the same tile in the same coordinates, one level coarser.
override NN: u32 = 256u;
override LOGN: u32 = 8u;
// stride of the four rows in `buf`
const B = 256u;
const GRAV = 9.81;
const PI = 3.14159265;

struct OU {
  // per cascade: x = tile size (m), y = k from, z = k up to (rad/m), w = choppiness
  casc: array<vec4f, 3>,
  // wind sea: x = wind (m/s), y = peak angular frequency, z = alpha, w = gamma
  sea: vec4f,
  // swell: x = peak angular frequency, y = alpha, z = gamma, w = spread
  swell: vec4f,
  // xy direction the wind blows toward, zw the swell's (east, north)
  dir: vec4f,
  // x = time (s, wrapped at the repeat period), y = frequency quantum (2 pi /
  // period), z = how much of the short-wave tail the wind raises (1 = all of
  // it), w = seconds since the last frame (0 while paused)
  t: vec4f,
};
@group(0) @binding(0) var<uniform> U: OU;
// Complex unit Gaussian per texel and cascade (ocean.rs `noise`, shared with the CPU copy).
@group(0) @binding(1) var<storage, read> noise: array<vec2f>;
// Rows transformed: [cascade][field][y][x].
@group(0) @binding(2) var<storage, read_write> spec: array<vec2f>;
@group(0) @binding(3) var out_t: texture_storage_2d_array<rgba16float, write>;
// Mip chain: the level above, and the one being written. (`foam` reads this
// frame's waves, level 0, through mip_src too.)
@group(0) @binding(5) var mip_src: texture_2d_array<f32>;
@group(0) @binding(6) var mip_dst: texture_storage_2d_array<rgba16float, write>;

var<workgroup> buf: array<vec2f, 1024>;

fn cmul(a: vec2f, b: vec2f) -> vec2f { return vec2f(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); }
fn ci(a: vec2f) -> vec2f { return vec2f(-a.y, a.x); }
fn cconj(a: vec2f) -> vec2f { return vec2f(a.x, -a.y); }

fn jonswap(w: f32, wp: f32, alpha: f32, gamma: f32) -> f32 {
  let sig = select(0.09, 0.07, w <= wp);
  let d = (w - wp) / (sig * wp);
  let r = exp(-0.5 * d * d);
  let x = wp / w;
  let x2 = x * x;
  return alpha * GRAV * GRAV / pow(w, 5.0) * exp(-1.25 * x2 * x2) * pow(gamma, r);
}
// Donelan-Banner sech^2 spread (integrates to 1 over the circle).
fn spread(th: f32, beta: f32) -> f32 {
  let b = max(beta, 0.05);
  let c = cosh(min(b * abs(th), 40.0));
  return b / (2.0 * tanh(b * PI) * c * c);
}
fn db_beta(wr: f32) -> f32 {
  if (wr < 0.95) { return 2.61 * pow(wr, 1.3); }
  if (wr < 1.6) { return 2.28 * pow(wr, -1.3); }
  return pow(10.0, -0.4 + 0.8393 * exp(-0.567 * log(wr * wr)));
}
fn angle(a: vec2f, b: vec2f) -> f32 { return atan2(a.x * b.y - a.y * b.x, dot(a, b)); }

// Wave energy per unit kx, ky (m^4).
fn energy(kv: vec2f, k: f32) -> f32 {
  let w = sqrt(GRAV * k);
  let kh = kv / k;
  var e = 0.0;
  if (U.sea.z > 0.0) {
    e += jonswap(w, U.sea.y, U.sea.z, U.sea.w) * spread(angle(U.dir.xy, kh), db_beta(w / U.sea.y));
  }
  if (U.swell.y > 0.0) {
    e += jonswap(w, U.swell.x, U.swell.y, U.swell.z) * spread(angle(U.dir.zw, kh), U.swell.w);
  }
  // Light air leaves the short wavelets (under a metre or so) weak.
  let tail = mix(1.0, U.t.z, smoothstep(2.0, 20.0, k));
  return e * tail * GRAV / (2.0 * w) / k;
}

struct Z4 { a: vec2f, b: vec2f, c: vec2f, d: vec2f };

fn spectrum(c: u32, x: u32, y: u32) -> Z4 {
  var o = Z4(vec2f(0.0), vec2f(0.0), vec2f(0.0), vec2f(0.0));
  // (the Nyquist row and column have no mirror image: left out)
  if (x == 0u || y == 0u) { return o; }
  let L = U.casc[c].x;
  let dk = 2.0 * PI / L;
  let kv = vec2f(f32(i32(x) - i32(NN / 2u)), f32(i32(y) - i32(NN / 2u))) * dk;
  let k = length(kv);
  if (k <= 0.0 || k < U.casc[c].y || k >= U.casc[c].z) { return o; }
  let a = dk * sqrt(0.5);
  let base = c * NN * NN;
  let h0p = noise[base + y * NN + x] * (sqrt(energy(kv, k)) * a);
  let h0m = noise[base + ((NN - y) % NN) * NN + ((NN - x) % NN)] * (sqrt(energy(-kv, k)) * a);
  // Deep water with surface tension; frequencies quantised so the sea repeats exactly.
  let w = floor(sqrt(GRAV * k * (1.0 + k * k / 136900.0)) / U.t.y) * U.t.y;
  let ph = w * U.t.x;
  let e = vec2f(cos(ph), sin(ph));
  let h = cmul(h0p, e) + cmul(cconj(h0m), cconj(e));
  let ih = ci(h);
  let dx = ih * (kv.x / k);
  let dy = ih * (kv.y / k);
  let dxx = h * (-kv.x * kv.x / k);
  let dyy = h * (-kv.y * kv.y / k);
  let dxy = h * (-kv.x * kv.y / k);
  o.a = dx + ci(dy);
  o.b = h + ci(dxy);
  o.c = ih * kv.x + ci(ih * kv.y);
  o.d = dxx + ci(dyy);
  return o;
}

// ------------------------------------------------------------------ foam
// Whitecaps live on cascade 0's grid (250 m, about 1 m a texel), in the
// surface's own (undisplaced) coordinates, so a patch rides the waves'
// orbits as the water does. Each frame:
//  - the crests that fold - where the Jacobian of the sideways shift of the
//    two long cascades drops below a threshold - put foam down, building up
//    over a few frames. The crest runs on downwind while the foam stays, so a
//    breaker leaves a patch drawn out along its path;
//  - fresh foam (the breaking itself) fades within a second, spent foam over
//    several, drifting downwind with the wind's surface drift;
//  - the threshold is the quantile of this frame's Jacobians that folds the
//    share q of the sea; `foam_ctl` turns q until the foam's mean cover meets
//    Monahan & O'Muircheartaigh's W = 3.84e-6 U^3.41.
// Layer 6 then holds (foam cover 0..1, fresh foam 0..1, threshold, 0).
const FOAM_BINS = 128u;
const J_LO = -1.5;
const J_HI = 1.5;
struct Ctl {
  hist: array<atomic<u32>, 128>,
  // sum of the foam cover over the grid, x 255
  sum: atomic<u32>,
  // the fold threshold on the Jacobian, and log of the share it folds
  bias: f32,
  logq: f32,
  pad: f32,
};
// Foam state (cover, fresh): last frame's and this frame's.
@group(0) @binding(7) var<storage, read> foam_src: array<vec2f>;
@group(0) @binding(8) var<storage, read_write> foam_dst: array<vec2f>;
@group(0) @binding(9) var<storage, read_write> ctl: Ctl;
@group(0) @binding(4) var foam_t: texture_storage_2d_array<rgba16float, write>;

var<workgroup> wh: array<atomic<u32>, 128>;
var<workgroup> wsum: atomic<u32>;

fn foam_at(p: vec2i) -> vec2f {
  let q = (p + vec2i(i32(NN))) % vec2i(i32(NN));
  return foam_src[u32(q.y) * NN + u32(q.x)];
}
// Last frame's foam at texel coordinates p (bilinear, wrapping).
fn foam_bilin(p: vec2f) -> vec2f {
  let f = floor(p);
  let t = p - f;
  let i = vec2i(f);
  return mix(mix(foam_at(i), foam_at(i + vec2i(1, 0)), t.x), mix(foam_at(i + vec2i(0, 1)), foam_at(i + vec2i(1, 1)), t.x), t.y);
}

@compute @workgroup_size(16, 16, 1)
fn foam(@builtin(global_invocation_id) id: vec3u, @builtin(local_invocation_index) li: u32) {
  if (li < FOAM_BINS) { atomicStore(&wh[li], 0u); }
  if (li == 0u) { atomicStore(&wsum, 0u); }
  workgroupBarrier();
  let x = id.x;
  let y = id.y;
  let dt = clamp(U.t.w, 0.0, 0.1);
  let bias = ctl.bias;
  // Cascade 0 here; cascade 1 (exactly six of its tiles across one of
  // cascade 0's) over all 6 x 6 of its texels that this one covers, so the
  // foam put down is the share of the cell that folds.
  let a0 = textureLoad(mip_src, vec2u(x, y), 0, 0);
  let b0 = textureLoad(mip_src, vec2u(x, y), 3, 0);
  var inj = 0.0;
  for (var k = 0u; k < 36u; k++) {
    let p1 = vec2u(x * 6u + k % 6u, y * 6u + k / 6u) % vec2u(NN);
    let a1 = textureLoad(mip_src, p1, 1, 0);
    let b1 = textureLoad(mip_src, p1, 4, 0);
    let dxx = b0.z + b1.z;
    let dyy = b0.w + b1.w;
    let dxy = a0.w + a1.w;
    let J = (1.0 + dxx) * (1.0 + dyy) - dxy * dxy;
    inj += smoothstep(bias + 0.06, bias - 0.06, J) / 36.0;
    let bin = u32(clamp((J - J_LO) / (J_HI - J_LO) * f32(FOAM_BINS), 0.0, f32(FOAM_BINS) - 1.0));
    atomicAdd(&wh[bin], 1u);
  }
  // The foam drifts downwind with the surface (about 3 % of the wind).
  let L0 = U.casc[0].x;
  let drift = U.dir.xy * (0.03 * U.sea.x) * dt * f32(NN) / L0;
  var f = foam_bilin(vec2f(f32(x), f32(y)) - drift);
  // Spent foam lasts longer in a stronger wind (bigger breakers, more air).
  let tau = mix(3.5, 7.0, smoothstep(6.0, 18.0, U.sea.x));
  let cover = f.x * exp(-dt / tau) + (1.0 - f.x) * (1.0 - exp(-dt * 8.0 * inj));
  let fresh = max(f.y * exp(-dt / 0.6), inj);
  f = clamp(vec2f(cover, fresh), vec2f(0.0), vec2f(1.0));
  foam_dst[y * NN + x] = f;
  textureStore(foam_t, vec2u(x, y), 0, vec4f(f, bias, 0.0));
  atomicAdd(&wsum, u32(f.x * 255.0 + 0.5));
  workgroupBarrier();
  if (li < FOAM_BINS) {
    let h = atomicLoad(&wh[li]);
    if (h > 0u) { atomicAdd(&ctl.hist[li], h); }
  }
  if (li == 0u) { atomicAdd(&ctl.sum, atomicLoad(&wsum)); }
}

@compute @workgroup_size(1)
fn foam_ctl() {
  let n = f32(NN * NN);
  let cover = f32(atomicLoad(&ctl.sum)) / 255.0 / n;
  let u = max(U.sea.x, 0.0);
  let W = min(3.84e-6 * pow(u, 3.41), 0.3);
  let dt = clamp(U.t.w, 0.0, 0.1);
  var logq = ctl.logq;
  if (dt > 0.0) {
    let err = clamp(log((W + 1e-4) / (cover + 1e-4)), -2.0, 2.0);
    logq = clamp(logq + err * dt * 0.15, log(1e-5), log(0.3));
  }
  ctl.logq = logq;
  // Threshold for next frame: the Jacobian below which the share q lies.
  let want = exp(logq) * n * 36.0;
  var acc = 0.0;
  var bias = J_LO;
  let bw = (J_HI - J_LO) / f32(FOAM_BINS);
  for (var i = 0u; i < FOAM_BINS; i++) {
    let h = f32(atomicLoad(&ctl.hist[i]));
    if (acc + h >= want) {
      bias = J_LO + (f32(i) + (want - acc) / max(h, 1.0)) * bw;
      break;
    }
    acc += h;
  }
  // Only a surface that is really squeezed breaks.
  ctl.bias = min(bias, 0.7);
  for (var i = 0u; i < FOAM_BINS; i++) { atomicStore(&ctl.hist[i], 0u); }
  atomicStore(&ctl.sum, 0u);
}

fn bitrev(x: u32) -> u32 { return reverseBits(x) >> (32u - LOGN); }

// In-place inverse radix-2 FFT of the four NN-point rows in `buf`
// (loaded in bit-reversed order).
fn fft(t: u32) {
  for (var s = 0u; s < LOGN; s++) {
    workgroupBarrier();
    if (t < NN / 2u) {
      let hw = 1u << s;
      let pos = t & (hw - 1u);
      let i0 = ((t >> s) << (s + 1u)) + pos;
      let i1 = i0 + hw;
      let a = PI * f32(pos) / f32(hw);
      let w = vec2f(cos(a), sin(a));
      for (var f = 0u; f < 4u; f++) {
        let p = buf[f * B + i0];
        let q = cmul(buf[f * B + i1], w);
        buf[f * B + i0] = p + q;
        buf[f * B + i1] = p - q;
      }
    }
  }
  workgroupBarrier();
}

@compute @workgroup_size(256)
fn rows(@builtin(local_invocation_id) lid: vec3u, @builtin(workgroup_id) wid: vec3u) {
  let x = lid.x;
  let y = wid.x;
  let c = wid.y;
  if (x < NN) {
    let z = spectrum(c, x, y);
    let r = bitrev(x);
    buf[r] = z.a;
    buf[B + r] = z.b;
    buf[2u * B + r] = z.c;
    buf[3u * B + r] = z.d;
  }
  fft(x);
  if (x < NN) {
    for (var f = 0u; f < 4u; f++) {
      spec[((c * 4u + f) * NN + y) * NN + x] = buf[f * B + x];
    }
  }
}

@compute @workgroup_size(256)
fn cols(@builtin(local_invocation_id) lid: vec3u, @builtin(workgroup_id) wid: vec3u) {
  let y = lid.x;
  let x = wid.x;
  let c = wid.y;
  if (y < NN) {
    let r = bitrev(y);
    for (var f = 0u; f < 4u; f++) {
      buf[f * B + r] = spec[((c * 4u + f) * NN + y) * NN + x];
    }
  }
  fft(y);
  if (y >= NN) { return; }
  // Wavenumbers run from -NN/2: that shifts every other texel's sign.
  let sg = select(1.0, -1.0, ((x + y) & 1u) == 1u);
  let a = buf[y] * sg;
  let b = buf[B + y] * sg;
  let s = buf[2u * B + y] * sg;
  let d = buf[3u * B + y] * sg;
  let lam = U.casc[c].w;
  textureStore(out_t, vec2u(x, y), c, vec4f(a * lam, b.x, b.y * lam));
  textureStore(out_t, vec2u(x, y), c + 3u, vec4f(s, d * lam));
}

@compute @workgroup_size(8, 8, 1)
fn mip(@builtin(global_invocation_id) id: vec3u) {
  let s = textureDimensions(mip_dst);
  if (id.x >= s.x || id.y >= s.y) { return; }
  let p = vec2i(id.xy * 2u);
  let l = i32(id.z);
  let v = textureLoad(mip_src, p, l, 0) + textureLoad(mip_src, p + vec2i(1, 0), l, 0) + textureLoad(mip_src, p + vec2i(0, 1), l, 0) + textureLoad(mip_src, p + vec2i(1, 1), l, 0);
  textureStore(mip_dst, id.xy, l, v * 0.25);
}
