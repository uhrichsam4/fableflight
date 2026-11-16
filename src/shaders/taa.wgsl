// Temporal anti-aliasing: every frame is drawn a fraction of a pixel off the
// last (G.taa.xy), and the finished HDR frame is blended into a history of
// the frames before it, found again by reprojecting each pixel's depth.
@group(1) @binding(0) var t_cur: texture_2d<f32>;
@group(1) @binding(1) var t_hist: texture_2d<f32>;
@group(1) @binding(2) var t_depth: texture_depth_2d;

// Share of the new frame in the result: for a pixel at rest, and for one
// crossing the screen. Every frame a moving pixel's history is read from
// between the pixels of the last, which softens it a little each time, so
// once it moves at all the history counts for less (even a twentieth of a
// pixel a frame, as distant scenery drifts in flight, adds up).
const TAA_BLEND = 0.1;
const TAA_BLEND_MOVING = 0.3;
// Width of the colour box the history is held to, in standard deviations of
// the pixel's neighbourhood. Tighter ghosts less, wider flickers less.
const TAA_CLIP = 1.25;

// Blend in a compressed range (undone afterwards) so one very bright sample
// - a glint, a lamp - does not outweigh the rest and flicker.
fn taa_k() -> f32 { return G.time.z * 1.6; }
fn squash(c: vec3f) -> vec3f { return c / (1.0 + luma(c) * taa_k()); }
fn unsquash(c: vec3f) -> vec3f { return c / max(1.0 - luma(c) * taa_k(), 1e-4); }
fn to_ycc(c: vec3f) -> vec3f {
  return vec3f(0.25 * c.r + 0.5 * c.g + 0.25 * c.b, 0.5 * c.r - 0.5 * c.b, -0.25 * c.r + 0.5 * c.g - 0.25 * c.b);
}
fn from_ycc(c: vec3f) -> vec3f { return vec3f(c.x + c.y - c.z, c.x + c.z, c.x - c.y - c.z); }

// Catmull-Rom lookup in nine bilinear taps: a plain bilinear lookup of the
// history would blur it a little more every frame.
fn history(uv: vec2f) -> vec3f {
  let dims = vec2f(textureDimensions(t_hist));
  let p = uv * dims;
  let c = floor(p - 0.5) + 0.5;
  let f = p - c;
  let w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
  let w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
  let w2 = f * (0.5 + f * (2.0 - 1.5 * f));
  let w3 = f * f * (-0.5 + 0.5 * f);
  let w12 = w1 + w2;
  let t0 = (c - 1.0) / dims;
  let t12 = (c + w2 / w12) / dims;
  let t3 = (c + 2.0) / dims;
  var s = textureSampleLevel(t_hist, s_clamp, vec2f(t0.x, t0.y), 0.0).rgb * (w0.x * w0.y);
  s += textureSampleLevel(t_hist, s_clamp, vec2f(t12.x, t0.y), 0.0).rgb * (w12.x * w0.y);
  s += textureSampleLevel(t_hist, s_clamp, vec2f(t3.x, t0.y), 0.0).rgb * (w3.x * w0.y);
  s += textureSampleLevel(t_hist, s_clamp, vec2f(t0.x, t12.y), 0.0).rgb * (w0.x * w12.y);
  s += textureSampleLevel(t_hist, s_clamp, vec2f(t12.x, t12.y), 0.0).rgb * (w12.x * w12.y);
  s += textureSampleLevel(t_hist, s_clamp, vec2f(t3.x, t12.y), 0.0).rgb * (w3.x * w12.y);
  s += textureSampleLevel(t_hist, s_clamp, vec2f(t0.x, t3.y), 0.0).rgb * (w0.x * w3.y);
  s += textureSampleLevel(t_hist, s_clamp, vec2f(t12.x, t3.y), 0.0).rgb * (w12.x * w3.y);
  s += textureSampleLevel(t_hist, s_clamp, vec2f(t3.x, t3.y), 0.0).rgb * (w3.x * w3.y);
  return max(s, vec3f(0.0));
}

// Where the surface seen at `uvn` with depth `z` was in last frame's clip space.
fn prev_clip(uvn: vec2f, z: f32) -> vec4f {
  let ray = view_ray(uvn);
  // Sky: infinitely far, it only turns with the camera.
  if (z == 0.0) { return G.prev_vp * vec4f(ray, 0.0); }
  let P = ray / z;
  let b = (G.plane_inv * vec4f(P, 1.0)).xyz;
  if (G.taa.w > 0.5 && all(abs(b - G.plane_box[0].xyz) < G.plane_box[1].xyz)) {
    // The aircraft flies along with the camera rather than standing in the world.
    return G.plane_prev * vec4f(b, 1.0);
  }
  return G.prev_vp * vec4f(P + G.cam_delta.xyz, 1.0);
}

// Where the farthest surface around moved, as a uv offset: where it moves
// quite unlike the nearest (the aircraft's outline against the ground racing
// past), the history behind the edge belongs to neither, and the pixel is
// treated as moving.
fn edge_split(uvf: vec2f, zf: f32, huv: vec2f, uv: vec2f) -> f32 {
  let pf = prev_clip(uvf, zf);
  if (pf.w <= 0.0) { return 1.0; }
  let hf = uv + (pf.xy / pf.w * vec2f(0.5, -0.5) + 0.5) - (uvf - G.taa.xy);
  return length(hf - huv);
}

@fragment fn fs_taa(in: FsIn) -> @location(0) vec4f {
  let dims = vec2i(textureDimensions(t_cur));
  let px = vec2i(in.pos.xy);
  let cur = textureLoad(t_cur, px, 0).rgb;
  if (G.taa.z < 0.5) { return vec4f(cur, 0.0); }
  // The colours around this pixel bound what its history may plausibly be;
  // the nearest surface around it says how it moved (so an edge keeps the
  // motion of the thing in front).
  let yc = to_ycc(squash(cur));
  var m1 = vec3f(0.0);
  var m2 = vec3f(0.0);
  var mn = yc;
  var mx = yc;
  var zc = 0.0;
  var off = vec2i(0);
  var zf = 2.0;
  var offf = vec2i(0);
  for (var y = -1; y <= 1; y++) {
    for (var x = -1; x <= 1; x++) {
      let q = clamp(px + vec2i(x, y), vec2i(0), dims - 1);
      let s = to_ycc(squash(textureLoad(t_cur, q, 0).rgb));
      m1 += s;
      m2 += s * s;
      mn = min(mn, s);
      mx = max(mx, s);
      let z = textureLoad(t_depth, q, 0);
      if (z > zc) { zc = z; off = q - px; }
      if (z < zf) { zf = z; offf = q - px; }
    }
  }
  // Where that surface was on screen last frame.
  let uvn = (vec2f(px + off) + 0.5) * G.screen.zw;
  let pc = prev_clip(uvn, zc);
  if (pc.w <= 0.0) { return vec4f(cur, 0.0); }
  // The history is of the image without the sub-pixel offset, so take the offset out of the motion.
  let huv = in.uv + (pc.xy / pc.w * vec2f(0.5, -0.5) + 0.5) - (uvn - G.taa.xy);
  if (any(huv < vec2f(0.0)) || any(huv > vec2f(1.0))) { return vec4f(cur, 0.0); }
  let mean = m1 / 9.0;
  let sd = sqrt(max(m2 / 9.0 - mean * mean, vec3f(0.0)));
  let lo = max(mn, mean - sd * TAA_CLIP);
  let hi = min(mx, mean + sd * TAA_CLIP);
  let h = clamp(to_ycc(squash(history(huv))), lo, hi);
  let split = edge_split((vec2f(px + offf) + 0.5) * G.screen.zw, zf, huv, in.uv) * G.screen.x;
  let speed = max(length((huv - in.uv) * G.screen.xy), split);
  let moving = smoothstep(0.02, 0.4, speed);
  let a = mix(TAA_BLEND, TAA_BLEND_MOVING, moving);
  let o = unsquash(max(from_ycc(mix(h, yc, a)), vec3f(0.0)));
  // Alpha: how much the pixel is moving, for the sharpening that follows (post.wgsl).
  return vec4f(clamp(o, vec3f(0.0), vec3f(3000.0)), moving);
}

// Catmull-Rom in five bilinear taps: `history` less its four corners (they
// weigh little, and at the window's resolution every tap counts).
fn history5(uv: vec2f) -> vec3f {
  let dims = vec2f(textureDimensions(t_hist));
  let p = uv * dims;
  let c = floor(p - 0.5) + 0.5;
  let f = p - c;
  let w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
  let w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
  let w2 = f * (0.5 + f * (2.0 - 1.5 * f));
  let w3 = f * f * (-0.5 + 0.5 * f);
  let w12 = w1 + w2;
  let t0 = (c - 1.0) / dims;
  let t12 = (c + w2 / w12) / dims;
  let t3 = (c + 2.0) / dims;
  var s = textureSampleLevel(t_hist, s_clamp, vec2f(t12.x, t0.y), 0.0).rgb * (w12.x * w0.y);
  s += textureSampleLevel(t_hist, s_clamp, vec2f(t0.x, t12.y), 0.0).rgb * (w0.x * w12.y);
  s += textureSampleLevel(t_hist, s_clamp, vec2f(t12.x, t12.y), 0.0).rgb * (w12.x * w12.y);
  s += textureSampleLevel(t_hist, s_clamp, vec2f(t3.x, t12.y), 0.0).rgb * (w3.x * w12.y);
  s += textureSampleLevel(t_hist, s_clamp, vec2f(t12.x, t3.y), 0.0).rgb * (w12.x * w3.y);
  let wsum = w12.x * w0.y + w0.x * w12.y + w12.x * w12.y + w3.x * w12.y + w12.x * w3.y;
  return max(s / wsum, vec3f(0.0));
}

// Temporal upscaling (the scene drawn smaller than the window, `Renderer::scale`
// under 1): the history is kept at the window's resolution, and each frame's
// jittered samples are splatted into it where they actually fell, so over a
// few frames it gathers detail finer than one frame's pixels hold. The jitter
// sequence is longer the smaller the scene (main.rs), to cover the finer grid.
// Each of the 3x3 scene pixels around this output pixel is weighed by how
// close its sample (its centre less this frame's offset) lies to it: in scene
// pixels for a smooth stand-in (no history yet), in output pixels for what
// goes into the history (a sample counts where it fell, so detail builds up).
fn taau_w(d: vec2f) -> f32 { return exp(-2.29 * dot(d, d)); }
// (wider than TAA_CLIP: weighted moments of an edge spread less)
const TAAU_CLIP = 1.6;
const TAAU_BLEND_MOVING = 0.2;
@fragment fn fs_taa_up(in: FsIn) -> @location(0) vec4f {
  let dims = vec2i(textureDimensions(t_cur));
  let fd = vec2f(dims);
  // This output pixel in scene pixels, in the unjittered image: sample k sits
  // at k + 0.5 - offset there.
  let p = in.uv * fd + G.taa.xy * fd - 0.5;
  let base = vec2i(floor(p + 0.5));
  // output pixels per scene pixel
  let ratio = f32(textureDimensions(t_hist).x) / fd.x;
  var acc = vec3f(0.0);
  var wsum = 0.0;
  var accn = vec3f(0.0);
  var wsumn = 0.0;
  var wmax = 0.0;
  var m1 = vec3f(0.0);
  var m2 = vec3f(0.0);
  var zc = 0.0;
  var off = vec2i(0);
  var zf = 2.0;
  var offf = vec2i(0);
  for (var y = -1; y <= 1; y++) {
    for (var x = -1; x <= 1; x++) {
      let q = clamp(base + vec2i(x, y), vec2i(0), dims - 1);
      let c = squash(textureLoad(t_cur, q, 0).rgb);
      let w = taau_w(vec2f(q) - p);
      acc += c * w;
      wsum += w;
      let wn = taau_w((vec2f(q) - p) * ratio);
      accn += c * wn;
      wsumn += wn;
      wmax = max(wmax, wn);
      // (moments weighed the same way, so the box the history is held to
      // moves smoothly with this pixel rather than in steps of scene pixels)
      let s = to_ycc(c);
      m1 += s * w;
      m2 += s * s * w;
      // (the nearest surface from the centre and its four sides)
      if (x * y == 0) {
        let z = textureLoad(t_depth, q, 0);
        if (z > zc) { zc = z; off = q - base; }
        if (z < zf) { zf = z; offf = q - base; }
      }
    }
  }
  let curs = acc / max(wsum, 1e-5);
  let cur = unsquash(curs);
  if (G.taa.z < 0.5) { return vec4f(cur, 0.0); }
  let ycn = to_ycc(accn / max(wsumn, 1e-6));
  // How the nearest surface around moved (as fs_taa).
  let uvn = (vec2f(base + off) + 0.5) / fd;
  let pc = prev_clip(uvn, zc);
  if (pc.w <= 0.0) { return vec4f(cur, 0.0); }
  let huv = in.uv + (pc.xy / pc.w * vec2f(0.5, -0.5) + 0.5) - (uvn - G.taa.xy);
  if (any(huv < vec2f(0.0)) || any(huv > vec2f(1.0))) { return vec4f(cur, 0.0); }
  let mean = m1 / max(wsum, 1e-5);
  let sd = sqrt(max(m2 / max(wsum, 1e-5) - mean * mean, vec3f(0.0)));
  let lo = mean - sd * TAAU_CLIP;
  let hi = mean + sd * TAAU_CLIP;
  let h = clamp(to_ycc(squash(history5(huv))), lo, hi);
  // (speed in scene pixels, as fs_taa)
  let split = edge_split((vec2f(base + offf) + 0.5) / fd, zf, huv, in.uv) * fd.x;
  let speed = max(length((huv - in.uv) * fd), split);
  let moving = smoothstep(0.02, 0.4, speed);
  // A sample right on this pixel counts in full, one an output pixel off
  // hardly: the history fills in what this frame's samples did not land on.
  // Moving, the history is clipped often and would take in this frame's
  // nearest samples (stair-stepped edges): take the smooth reconstruction,
  // at a steady share, instead.
  let yc = mix(ycn, to_ycc(curs), moving);
  let a = mix(TAA_BLEND * clamp(wmax, 0.02, 1.0), TAAU_BLEND_MOVING, moving);
  let o = unsquash(max(from_ycc(mix(h, yc, a)), vec3f(0.0)));
  return vec4f(clamp(o, vec3f(0.0), vec3f(3000.0)), moving);
}

// FSR (render.rs, the Graphics setting Upscaler: FSR) does its own blending;
// it is handed how each pixel moved: where its surface was on screen last
// frame less where it is now, in uv, without this frame's sub-pixel offset
// (the same reprojection as the history lookups above: the aircraft through
// its own frame, the sky by the camera's turn alone).
@fragment fn fs_motion(in: FsIn) -> @location(0) vec4f {
  let px = vec2i(in.pos.xy);
  let uvn = (vec2f(px) + 0.5) * G.screen.zw;
  let pc = prev_clip(uvn, textureLoad(t_depth, px, 0));
  // (behind the camera last frame: no history to find anyway)
  if (pc.w <= 0.0) { return vec4f(0.0); }
  return vec4f(pc.xy / pc.w * vec2f(0.5, -0.5) + 0.5 - (uvn - G.taa.xy), 0.0, 0.0);
}

// FSR's reactive mask: how much the particles and the fire changed each
// pixel (t_cur the scene after them, t_hist before), compared as the eye
// sees them. Where they did, FSR leans on this frame rather than on a
// history that moved with the surface behind them (smears, trails).
const REACTIVE_GAIN = 3.0;
const REACTIVE_MAX = 0.9;
@fragment fn fs_reactive(in: FsIn) -> @location(0) vec4f {
  let px = vec2i(in.pos.xy);
  let k = taa_k();
  let a = squash(textureLoad(t_hist, px, 0).rgb) * k;
  let b = squash(textureLoad(t_cur, px, 0).rgb) * k;
  let d = abs(a - b);
  return vec4f(min(max(d.r, max(d.g, d.b)) * REACTIVE_GAIN, REACTIVE_MAX), 0.0, 0.0, 0.0);
}
