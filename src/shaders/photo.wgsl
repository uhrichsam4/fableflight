// Photorealistic 3D tiles (tiles3d.rs): Google's photogrammetry meshes, one
// texture each. The textures are photographs: relight.rs has lifted their
// shadows out as they loaded and left in alpha how lit each texel was, so
// they are lit by our own sun. Where it stands as the photographer's did the
// photographed shadows are kept (sharper and finer than ours); elsewhere the
// mesh casts its own, into our cascades (vs_shadow).
// At night they take the light of our street lamps (from the coverage mask)
// and lit windows on the walls of our map's buildings (photonight.rs).
// offset: xyz tile origin relative to the camera, w its geometric error (m).
// lod: x how far it has faded in (or, leaving, how much of it is left), y 1
// while leaving: a tile and the parent or children it replaces are dithered
// into each other (tiles3d.rs, `crossfade`).
struct PhotoTile { offset: vec4f, lod: vec4f };
@group(1) @binding(0) var<uniform> PT: PhotoTile;
@group(1) @binding(1) var t_ph: texture_2d<f32>;
@group(2) @binding(0) var<uniform> CS: Cascade;

struct VOut {
  @builtin(position) clip: vec4f,
  @location(0) P: vec3f,
  @location(1) uv: vec2f,
  @location(2) N: vec3f,
  @location(3) @interpolate(flat) tag: f32,
};
struct GOut { @location(0) c: vec4f, @location(1) n: vec4f, @location(2) amb: f32 };

@vertex fn vs(@location(0) pos: vec3f, @location(1) uv: vec2f, @location(2) nrm: vec4f) -> VOut {
  let P = pos + PT.offset.xyz;
  return VOut(G.view_proj * vec4f(P, 1.0), P, uv, nrm.xyz, round(nrm.w * 127.0));
}

// The coverage mask: the tiles seen from straight above.
@vertex fn vs_mask(@location(0) pos: vec3f) -> @builtin(position) vec4f {
  return CS.vp * vec4f(pos + PT.offset.xyz, 1.0);
}
@fragment fn fs_mask() -> @location(0) vec4f { return vec4f(1.0); }

// Into a shadow cascade: what is drawn casts (not past the middle of the
// band at the edge of the zone, nor the mesh's still water, nor cut-out
// buildings). Tiles clear of the band and the cuts take vs_shadow alone.
struct SOut { @builtin(position) clip: vec4f, @location(0) P: vec3f };
@vertex fn vs_shadow(@location(0) pos: vec3f) -> SOut {
  let P = pos + PT.offset.xyz;
  return SOut(CS.vp * vec4f(P, 1.0), P);
}
@fragment fn fs_shadow(in: SOut) {
  if (photo_fade(in.P) >= 0.5) { discard; }
  if (height_of(in.P) < photo_mask(in.P).g + 2.0 + 0.12 * PT.offset.w) { discard; }
  if (photo_cut(in.P)) { discard; }
}

// A tile changing level of detail: the one coming in shows where the noise is
// below its fade, the one leaving where it is above (1 - its fade = the same
// threshold), so each pixel shows exactly one of them. TAA smooths it.
fn lod_keep(px: vec2f) -> bool {
  if (PT.lod.x >= 1.0) { return true; }
  let n = ign(px + vec2f(17.0, 41.0) + 5.588238 * (G.time.w % 64.0));
  return select(n, 1.0 - n, PT.lod.y > 0.5) < PT.lod.x;
}

@fragment fn fs(in: VOut) -> GOut {
  // Dissolves into our own world across the band at the edge of the zone.
  if (!photo_keep(in.P, in.clip.xy)) { discard; }
  if (!lod_keep(in.clip.xy)) { discard; }
  // Our water (with its waves) shows over the mesh's still photograph of it:
  // the mesh gives way near our water surface (coarse tiles, whose water is
  // only roughly level, further above it). Piers, hulls and bridges stand above.
  if (height_of(in.P) < photo_mask(in.P).g + 2.0 + 0.12 * PT.offset.w) { discard; }
  // A destroyed building: its pieces carry this part of the mesh (vs_skin).
  if (photo_cut(in.P)) { discard; }
  return photo_shade(in.P, textureSample(t_ph, s_aniso, in.uv), in.N, in.tag, false);
}

// The relit mesh's share of our skylight by day (see photo_shade).
const PHOTO_AMB = 0.65;

// How much our own light, rather than the photograph's, shapes the mesh at P:
// 0 where our sun stands within some 14 degrees of the photographer's (its
// shadows kept), 1 from 35 degrees off (ours only), always 1 for falling pieces.
fn photo_relight(own: bool) -> f32 {
  if (own) { return 1.0; }
  return 1.0 - smoothstep(0.82, 0.97, dot(G.sun_dir.xyz, G.photo_sun.xyz)) * G.photo_sun.w;
}

// `tex`: the de-lit photograph, alpha how lit it was there. `own`: a falling
// piece, whose photographed light means nothing any more.
fn photo_shade(P: vec3f, tex: vec4f, Nv: vec3f, tag: f32, own: bool) -> GOut {
  let l = light_at(P);
  var N = Nv;
  if (dot(N, N) < 1e-6) { N = l.upn; }
  N = normalize(N);
  if (dot(N, P) > 0.0) { N = -N; }
  let albedo = with_snow(tex.rgb, P, N);
  var Nl: vec3f;
  var sh: f32;
  var amb_k = 1.0;
  var bounce = vec3f(0.0);
  if (G.photo_sun.w < 0.0) {
    // The photographs as they come (FABLE_NO_RELIGHT): facing mostly up,
    // the photo already holds the shading of walls and the shadows in the
    // streets; shadows from the aircraft only.
    Nl = normalize(l.upn * 0.7 + N * 0.3);
    sh = plane_shadow(P, N);
  } else {
    // Relit: the photographed shadows, or ours, or between the two. The
    // photo's lit walls keep the shading of the photographer's sun, so
    // only with our own light does the true normal take over (meshes are
    // lumpy: never all of it).
    let w = photo_relight(own);
    var ours = 1.0;
    if (w > 0.001 && G.csm.w > 0.0) { ours = shadow_world(P, N); }
    sh = mix(tex.a, ours, w) * plane_shadow(P, N);
    Nl = normalize(mix(l.upn, N, mix(0.3, 0.75, w)));
    // Shade as deep as in the photographs (some 6:1 against the sun at
    // midday, where our sky alone gives 4:1: the mesh has none of the
    // clutter that shades a real street), by day; warmed by the light off
    // the sunlit ground, as our own buildings are (common.wgsl, `shade`).
    amb_k = mix(1.0, PHOTO_AMB, 1.0 - night_at(l.upn));
    let mu = max(dot(l.upn, G.sun_dir.xyz), 0.0);
    bounce = vec3f(0.15, 0.14, 0.12) * (l.sun * mu * 0.8 / PI + l.amb) * (0.5 - 0.5 * dot(Nl, l.upn));
  }
  let ndl = max(dot(Nl, G.sun_dir.xyz), 0.0);
  let moon = moon_light(l.upn, l.h) * max(dot(Nl, G.moon.xyz), 0.0);
  let sky = (0.55 + 0.45 * dot(Nl, l.upn)) * amb_k;
  g_amb = albedo * (l.amb * sky + bounce);
  var c = albedo * ((l.sun * ndl * sh + moon) / PI + l.amb * sky + bounce + point_light(P, N));
  // After dusk: street lamps on what lies along our roads, lit rooms behind
  // the walls of buildings.
  let night = night_at(l.upn);
  if (night > 0.0) {
    // (a third of what our roads take, roads.wgsl: the photographed street
    // is sunlit concrete, three times as light as the asphalt we draw)
    var glow = albedo * LAMP_COL * photo_lamps(P, N, l.upn) * 0.032;
    if (tag > 0.5) { glow += photo_windows(P, N, l.upn, u32(tag), albedo); }
    c += glow * night * G.sun_dir.w;
  }
  return GOut(vec4f(c, 1.0), vec4f(N, 0.0), amb_share(c));
}

// Light from our street lamps at P: render.rs draws the lamp pools of our
// roads (which the mesh hides) into the coverage mask from above, with the
// road's height. A soft 4x4-texel footprint (texels are about 5.6 m) spills
// it over the kerbs onto pavements and the foot of walls; it fades with
// height above the road, so roofs and upper floors stay dark.
fn photo_lamps(P: vec3f, N: vec3f, upn: vec3f) -> f32 {
  let q = G.photo_vp * vec4f(P, 1.0);
  let uv = q.xy * vec2f(0.5, -0.5) + 0.5;
  if (any(uv <= vec2f(0.0)) || any(uv >= vec2f(1.0))) { return 0.0; }
  let dim = vec2f(textureDimensions(t_photo));
  let t = uv * dim;
  let base = floor(t - 0.5);
  let h = height_of(P);
  var sum = 0.0;
  var wsum = 0.0;
  for (var by = -1; by <= 1; by += 2) {
    for (var bx = -1; bx <= 1; bx += 2) {
      // The 2x2 block of texels from b (the four blocks span base - 1 .. base + 2).
      let b = base + vec2f(f32(bx), f32(by));
      let g = (b + 1.0) / dim;
      let lb = textureGather(2, t_photo, s_clamp, g);
      let la = textureGather(3, t_photo, s_clamp, g);
      // Gather order: (0,1), (1,1), (1,0), (0,0) texels from the block's corner.
      let cx = vec4f(0.5, 1.5, 1.5, 0.5) + b.x;
      let cy = vec4f(1.5, 1.5, 0.5, 0.5) + b.y;
      let dx = cx - t.x;
      let dy = cy - t.y;
      let w = exp(-(dx * dx + dy * dy) / (2.0 * 0.7 * 0.7));
      let dh = h - la;
      let fall = smoothstep(vec4f(14.0), vec4f(5.0), dh) * smoothstep(vec4f(-6.0), vec4f(-1.5), dh) * step(vec4f(-5000.0), la);
      sum += dot(w, lb * fall);
      wsum += dot(w, vec4f(1.0));
    }
  }
  // Facing up takes the light from overhead best; walls a little less.
  return sum / max(wsum, 1e-4) * (0.55 + 0.45 * max(dot(N, upn), 0.0));
}

// Lit windows on a wall of one of our buildings (tag: photonight.rs): a grid
// square to the building, a storey high, in world-fixed coordinates (the
// camera's travel is in G.east.w / G.north.w), so it stays put on the walls.
fn photo_windows(P: vec3f, N: vec3f, upn: vec3f, tag: u32, albedo: vec3f) -> vec3f {
  let wall = 1.0 - smoothstep(0.3, 0.55, abs(dot(N, upn)));
  // Tree crowns that stray into a footprint are no walls.
  let lum = dot(albedo, vec3f(0.3, 0.59, 0.11));
  let veg = clamp(((2.0 * albedo.g - albedo.r - albedo.b) / max(lum, 0.02) - 0.05) / 0.15, 0.0, 1.0);
  if (wall * (1.0 - veg) <= 0.0) { return vec3f(0.0); }
  let tall = tag > 64u;
  let ang = f32((tag - 1u) & 63u) * (0.5 * PI / 63.0);
  let ca = cos(ang);
  let sa = sin(ang);
  let wx = dot(P, G.east.xyz) + G.east.w;
  let wy = dot(P, G.north.xyz) + G.north.w;
  let sx = wx * ca + wy * sa;
  let sy = wy * ca - wx * sa;
  let ax = G.east.xyz * ca + G.north.xyz * sa;
  let ay = cross(upn, ax);
  // A wall facing along one axis runs along the other.
  let along_y = abs(dot(N, ax)) > abs(dot(N, ay));
  let s = select(sx, sy, along_y);
  let d = select(sy, sx, along_y);
  let side = select(sign(dot(N, ay)), sign(dot(N, ax)), along_y);
  let face = (floor(d / 12.0 + 0.5) % 1024.0) * 2.0 + side + select(0.0, 7.0, along_y);
  let fh = select(3.0, 3.7, tall);
  let cw = select(2.6, 1.6, tall);
  let h = height_of(P);
  let g = vec2f(s / cw, h / fh);
  let idf = floor(g);
  let f = g - idf;
  // (kept small for the hashes, whose sin() loses its way on large numbers)
  let id = idf - 4096.0 * floor(idf / 4096.0);
  let fw = max(fwidth(g), vec2f(1e-4));
  // The glass in each cell: ribbons on offices, single windows on houses.
  let hx = select(0.25, 0.44, tall);
  let y0 = select(0.3, 0.25, tall);
  let y1 = select(0.8, 0.9, tall);
  let bx = clamp((hx - abs(f.x - 0.5)) / fw.x + 0.5, 0.0, 1.0);
  let byv = clamp((min(f.y - y0, y1 - f.y)) / fw.y + 0.5, 0.0, 1.0);
  let glass = bx * byv;
  let winfrac = 2.0 * hx * (y1 - y0);
  // Who is in tonight: some buildings nearly dark, some blazing; offices
  // light whole floors at a time, for the cleaners.
  let blk = floor(vec2f(sx, sy) / 45.0) % 512.0;
  let occ = hash2(blk + 0.37);
  let hw = hash3(vec3f(id, face));
  let h2 = hash3(vec3f(id.yx * 1.7, face + 3.0));
  var litp = 0.32 * mix(0.1, 1.7, occ * occ);
  var on = step(1.0 - litp, hw);
  var bright = 0.4 + 0.9 * h2;
  var lc = mix(vec3f(1.0, 0.6, 0.28), vec3f(1.0, 0.82, 0.6), smoothstep(0.3, 0.75, hash3(vec3f(id * 2.31, face))));
  var far_c = vec3f(1.0, 0.72, 0.42);
  if (tall) {
    litp = 0.2 * mix(0.1, 2.2, occ * occ);
    on = step(1.0 - litp, hash3(vec3f(id.y * 3.1, face, occ * 17.0))) * step(0.1, hw);
    bright *= 0.6;
    lc = mix(vec3f(0.82, 0.9, 1.0), vec3f(1.0, 0.85, 0.65), step(0.72, h2));
    far_c = vec3f(1.0, 0.86, 0.68);
  }
  // Glass in the photograph is darker than the wall round it: lean on that.
  let dark = clamp(1.4 - lum * 3.0, 0.5, 1.2);
  let aa = clamp(1.0 - max(fw.x, fw.y) * 1.7, 0.0, 1.0);
  let near = lc * on * bright * glass * dark;
  let far = far_c * litp * winfrac * select(0.36, 0.26, tall);
  // (as bright as our own facades' rooms, facade.wgsl)
  return mix(far * 0.025, near * 0.015, aa) * wall * (1.0 - veg);
}

// ------------------------------------------------------------ destroyed buildings
// A destroyed building's own mesh, cut into the pieces it came down in
// (skin.rs): each vertex sits on its piece's axes; the piece's slot holds
// where it is now (rotation rows, camera-relative offset in w) and how much
// soot and dust it has taken on (tint, w = 0 once it is gone).
struct Slot { m0: vec4f, m1: vec4f, m2: vec4f, tint: vec4f };
@group(1) @binding(2) var<storage, read> SL: array<Slot>;

struct VSkin {
  @builtin(position) clip: vec4f,
  @location(0) P: vec3f,
  @location(1) uv: vec2f,
  @location(2) N: vec3f,
  @location(3) tint: vec3f,
};

@vertex fn vs_skin(@location(0) pos: vec3f, @location(1) uv: vec2f, @location(2) nrm: vec4f, @location(3) slot: u32) -> VSkin {
  let s = SL[slot];
  if (s.tint.w < 0.5) { return VSkin(vec4f(0.0, 0.0, -2.0, 1.0), vec3f(0.0), uv, vec3f(0.0), vec3f(0.0)); }
  let P = vec3f(dot(s.m0.xyz, pos) + s.m0.w, dot(s.m1.xyz, pos) + s.m1.w, dot(s.m2.xyz, pos) + s.m2.w);
  let N = vec3f(dot(s.m0.xyz, nrm.xyz), dot(s.m1.xyz, nrm.xyz), dot(s.m2.xyz, nrm.xyz));
  return VSkin(G.view_proj * vec4f(P, 1.0), P, uv, N, s.tint.rgb);
}

@fragment fn fs_skin(in: VSkin) -> GOut {
  let tex = textureSample(t_ph, s_aniso, in.uv);
  return photo_shade(in.P, vec4f(tex.rgb * in.tint, 1.0), in.N, 0.0, true);
}
