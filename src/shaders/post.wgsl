// Bloom chain, tonemap.
@group(1) @binding(0) var t_src: texture_2d<f32>;
@group(1) @binding(1) var t_bloom: texture_2d<f32>;
@group(1) @binding(2) var t_depth: texture_depth_2d;

@fragment fn fs_down(in: FsIn) -> @location(0) vec4f {
  let tx = 1.0 / vec2f(textureDimensions(t_src));
  var c = textureSampleLevel(t_src, s_clamp, in.uv, 0.0).rgb * 4.0;
  c += textureSampleLevel(t_src, s_clamp, in.uv + tx * vec2f(-1.0, -1.0), 0.0).rgb;
  c += textureSampleLevel(t_src, s_clamp, in.uv + tx * vec2f(1.0, -1.0), 0.0).rgb;
  c += textureSampleLevel(t_src, s_clamp, in.uv + tx * vec2f(-1.0, 1.0), 0.0).rgb;
  c += textureSampleLevel(t_src, s_clamp, in.uv + tx * vec2f(1.0, 1.0), 0.0).rgb;
  c = clamp(c / 8.0, vec3f(0.0), vec3f(3000.0));
  // Soft clamp so a single hot pixel cannot flood the screen.
  return vec4f(c / (1.0 + luma(c) * 0.002), 1.0);
}
@fragment fn fs_up(in: FsIn) -> @location(0) vec4f {
  let tx = 1.0 / vec2f(textureDimensions(t_src));
  var c = textureSampleLevel(t_src, s_clamp, in.uv, 0.0).rgb * 4.0;
  c += (textureSampleLevel(t_src, s_clamp, in.uv + tx * vec2f(-1.0, 0.0), 0.0).rgb + textureSampleLevel(t_src, s_clamp, in.uv + tx * vec2f(1.0, 0.0), 0.0).rgb
      + textureSampleLevel(t_src, s_clamp, in.uv + tx * vec2f(0.0, -1.0), 0.0).rgb + textureSampleLevel(t_src, s_clamp, in.uv + tx * vec2f(0.0, 1.0), 0.0).rgb) * 2.0;
  c += textureSampleLevel(t_src, s_clamp, in.uv + tx * vec2f(-1.0, -1.0), 0.0).rgb + textureSampleLevel(t_src, s_clamp, in.uv + tx * vec2f(1.0, -1.0), 0.0).rgb
     + textureSampleLevel(t_src, s_clamp, in.uv + tx * vec2f(-1.0, 1.0), 0.0).rgb + textureSampleLevel(t_src, s_clamp, in.uv + tx * vec2f(1.0, 1.0), 0.0).rgb;
  return vec4f(c / 16.0, 1.0);
}

// AgX (Troy Sobotka's filmic transform): rolls bright colours off towards
// white the way film and camera sensors do instead of skewing their hue, and
// keeps a long, soft toe so shadows hold detail. Returns display-encoded RGB.
fn agx(x: vec3f) -> vec3f {
  let inset = mat3x3f(0.842479062253094, 0.0423282422610123, 0.0423756549057051,
                      0.0784335999999992, 0.878468636469772, 0.0784336,
                      0.0792237451477643, 0.0791661274605434, 0.879142973793104);
  let outset = mat3x3f(1.19687900512017, -0.0528968517574562, -0.0529716355144438,
                       -0.0980208811401368, 1.15190312990417, -0.0980434501171241,
                       -0.0990297440797205, -0.0989611768448433, 1.15107367264116);
  let lo = -12.47393;
  let hi = 4.026069;
  var v = inset * max(x, vec3f(1e-10));
  v = (clamp(log2(v), vec3f(lo), vec3f(hi)) - lo) / (hi - lo);
  let v2 = v * v;
  let v4 = v2 * v2;
  v = 15.5 * v4 * v2 - 40.14 * v4 * v + 31.96 * v4 - 6.868 * v2 * v + 0.4298 * v2 + 0.1191 * v - 0.00232;
  // A gentle 'punchy' look: a little more contrast and colour than plain AgX,
  // closer to what a camera's picture profile does. The S-curve pivots on
  // mid-grey, so it deepens shadows and lifts highlights without darkening
  // the midtones the way a plain power would.
  v = clamp(v, vec3f(0.0), vec3f(1.0));
  v = mix(v, v * v * (3.0 - 2.0 * v), AGX_CONTRAST);
  let l = dot(v, vec3f(0.2126, 0.7152, 0.0722));
  v = l + (v - l) * AGX_SAT;
  v = outset * v;
  return clamp(v, vec3f(0.0), vec3f(1.0));
}
const AGX_CONTRAST = 0.45;
const AGX_SAT = 1.2;
const AGX_EXPOSURE = 1.6;

// Bloom, exposure, vignette and tonemap of a scene colour.
fn finish(scene: vec3f, in: FsIn) -> vec4f {
  let uv = in.uv;
  var c = scene;
  // Lights bloom more at night, when the eye is dark-adapted. (Bloom off in
  // the Graphics settings: G.start.z = 1, and the bloom texture is not drawn.)
  if (G.start.z < 0.5) {
    let bloom = textureSampleLevel(t_bloom, s_clamp, uv, 0.0).rgb;
    c = mix(c, bloom, mix(0.055, 0.13, smoothstep(1.0, 9.0, G.time.z)));
  }
  c *= G.time.z;
  // Gentle vignette.
  let q = in.uv - 0.5;
  c *= 1.0 - dot(q, q) * 0.35;
  c = agx(c * AGX_EXPOSURE);
  c += (ign(in.pos.xy + 17.0) - 0.5) / 255.0;
  return vec4f(c, 1.0);
}
@fragment fn fs_final(in: FsIn) -> @location(0) vec4f {
  return finish(textureSampleLevel(t_src, s_clamp, in.uv, 0.0).rgb, in);
}
// After temporal anti-aliasing (taa.wgsl), whose alpha says how much each
// pixel is moving: what moves has been softened by the blend, so sharpen it
// back, never past the brightest or darkest of its neighbours (no halos).
const TAA_SHARPEN = 0.8;
@fragment fn fs_final_taa(in: FsIn) -> @location(0) vec4f {
  let px = vec2i(in.pos.xy);
  let dmax = vec2i(textureDimensions(t_src)) - 1;
  let c = textureLoad(t_src, px, 0);
  let n = textureLoad(t_src, clamp(px + vec2i(0, -1), vec2i(0), dmax), 0).rgb;
  let s = textureLoad(t_src, clamp(px + vec2i(0, 1), vec2i(0), dmax), 0).rgb;
  let w = textureLoad(t_src, clamp(px + vec2i(-1, 0), vec2i(0), dmax), 0).rgb;
  let e = textureLoad(t_src, clamp(px + vec2i(1, 0), vec2i(0), dmax), 0).rgb;
  return finish(sharpen(c.rgb, n, s, w, e, TAA_SHARPEN * c.a), in);
}
// The same for a scene drawn smaller than the window (--render-scale): read
// filtered, its own pixels apart, and sharpened a little even where still
// (the stretch softens everything).
const UP_SHARPEN = 0.5;
@fragment fn fs_final_taa_up(in: FsIn) -> @location(0) vec4f {
  let tx = 1.0 / vec2f(textureDimensions(t_src));
  let c = textureSampleLevel(t_src, s_clamp, in.uv, 0.0);
  let n = textureSampleLevel(t_src, s_clamp, in.uv + vec2f(0.0, -tx.y), 0.0).rgb;
  let s = textureSampleLevel(t_src, s_clamp, in.uv + vec2f(0.0, tx.y), 0.0).rgb;
  let w = textureSampleLevel(t_src, s_clamp, in.uv + vec2f(-tx.x, 0.0), 0.0).rgb;
  let e = textureSampleLevel(t_src, s_clamp, in.uv + vec2f(tx.x, 0.0), 0.0).rgb;
  return finish(sharpen(c.rgb, n, s, w, e, max(TAA_SHARPEN * c.a, UP_SHARPEN)), in);
}
// After temporal upscaling (taa.wgsl fs_taa_up): the history is already at
// the window's resolution, softened a little by the upscale; sharpen it
// everywhere, more where it moves (the same halo-free limit as above).
const TAAU_SHARPEN = 0.45;
@fragment fn fs_final_taau(in: FsIn) -> @location(0) vec4f {
  let px = vec2i(in.pos.xy);
  let dmax = vec2i(textureDimensions(t_src)) - 1;
  let c = textureLoad(t_src, px, 0);
  let n = textureLoad(t_src, clamp(px + vec2i(0, -1), vec2i(0), dmax), 0).rgb;
  let s = textureLoad(t_src, clamp(px + vec2i(0, 1), vec2i(0), dmax), 0).rgb;
  let w = textureLoad(t_src, clamp(px + vec2i(-1, 0), vec2i(0), dmax), 0).rgb;
  let e = textureLoad(t_src, clamp(px + vec2i(1, 0), vec2i(0), dmax), 0).rgb;
  return finish(sharpen(c.rgb, n, s, w, e, max(TAA_SHARPEN * c.a, TAAU_SHARPEN)), in);
}
fn sharpen(c: vec3f, n: vec3f, s: vec3f, w: vec3f, e: vec3f, k: f32) -> vec3f {
  let lo = min(min(min(n, s), min(w, e)), c);
  let hi = max(max(max(n, s), max(w, e)), c);
  return clamp(c + (c - (n + s + w + e) * 0.25) * k, lo, hi);
}

// Final copy to the window.
@fragment fn fs_blit(in: FsIn) -> @location(0) vec4f {
  return vec4f(textureSampleLevel(t_src, s_clamp, in.uv, 0.0).rgb, 1.0);
}
@fragment fn fs_blit_srgb(in: FsIn) -> @location(0) vec4f {
  return vec4f(pow(textureSampleLevel(t_src, s_clamp, in.uv, 0.0).rgb, vec3f(2.2)), 1.0);
}
// The start screen's title (logo.rs), over the finished frame: the source is
// its coverage, and the blend (src x (1 - dst) + dst x (1 - src)) turns each
// pixel it covers into the inverse of the colour already there.
@fragment fn fs_logo(in: FsIn) -> @location(0) vec4f {
  let m = textureSampleLevel(t_src, s_clamp, in.uv, 0.0).r;
  return vec4f(m, m, m, 0.0);
}
