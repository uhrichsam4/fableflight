// Instanced solid objects: cars, rubble, rocks.
@group(2) @binding(0) var<uniform> CS: Cascade;

struct VOut {
  @builtin(position) clip: vec4f,
  @location(0) P: vec3f,
  @location(1) N: vec3f,
  @location(2) col: vec4f,
  @location(3) L: vec3f,
  @location(4) @interpolate(flat) tint: vec4f,
  @location(5) Ln: vec3f,
  // pieces of facade: wall coordinates and style (see destruct.rs), and the
  // direction along the wall in which they grow
  @location(6) @interpolate(flat) ext: vec4f,
  @location(7) @interpolate(flat) T: vec3f,
};
struct GOut { @location(0) c: vec4f, @location(1) n: vec4f, @location(2) amb: f32 };

fn world(p: vec3f, w: f32, m0: vec4f, m1: vec4f, m2: vec4f) -> vec3f {
  let v = vec4f(p, w);
  return vec3f(dot(m0, v), dot(m1, v), dot(m2, v));
}

@vertex fn vs(@location(0) pos: vec3f, @location(1) nrm: vec3f, @location(2) col: vec4f,
              @location(3) m0: vec4f, @location(4) m1: vec4f, @location(5) m2: vec4f, @location(6) tint: vec4f, @location(7) ext: vec4f) -> VOut {
  let P = world(pos, 1.0, m0, m1, m2);
  // Scale may be uneven (rubble slabs): normals need the inverse scale.
  let sx = vec3f(length(vec3f(m0.x, m1.x, m2.x)), length(vec3f(m0.y, m1.y, m2.y)), length(vec3f(m0.z, m1.z, m2.z)));
  let N = world(nrm / max(sx * sx, vec3f(1e-6)), 0.0, m0, m1, m2);
  let T = normalize(vec3f(m0.x, m1.x, m2.x)) * sign(ext.z);
  return VOut(G.view_proj * vec4f(P, 1.0), P, N, col, pos * sx, tint, nrm, ext, T);
}
@vertex fn vs_shadow(@location(0) pos: vec3f, @location(3) m0: vec4f, @location(4) m1: vec4f, @location(5) m2: vec4f, @location(6) tint: vec4f, @location(7) ext: vec4f) -> @builtin(position) vec4f {
  // Street traffic hidden under the 3D tiles casts nothing (decided per car).
  if (tint.a < 0.5 && ext.w > 0.5 && photo_shadow_gone(vec3f(m0.w, m1.w, m2.w))) { return vec4f(2.0, 2.0, 0.0, 1.0); }
  return CS.vp * vec4f(world(pos, 1.0, m0, m1, m2), 1.0);
}

@fragment fn fs(in: VOut) -> GOut {
  // Street traffic (ext.w = 1) gives way to the 3D tiles (tiles3d.rs).
  if (in.tint.a < 0.5 && in.ext.w > 0.5 && photo_hides(in.P, in.clip.xy)) { discard; }
  // Light itself (destroy.rs: laser bolts, a meteor's white-hot front):
  // tint.a 4, its colour the radiance, a little brighter seen face on.
  if (in.tint.a > 3.5) {
    let Vg = normalize(in.P);
    let Ng = normalize(select(-Vg, in.N, dot(in.N, in.N) > 1e-10));
    let core = 0.65 + 0.35 * abs(dot(Ng, Vg));
    return GOut(vec4f(in.tint.rgb * core, 1.0), vec4f(-Vg, 0.0), 0.0);
  }
  var N = in.N;
  if (dot(N, N) < 1e-10) { N = -normalize(in.P); }
  N = normalize(N);
  let V = normalize(in.P);
  if (dot(N, V) > 0.0) { N = -N; }
  let l = light_at(in.P);
  let sh = shadow_at(in.P, N);
  let m = in.col.a;
  // Facade coordinates of a piece of outer wall: along the wall from the
  // building's own origin, and height above its ground (local -z is up).
  let fuv = vec2f(in.ext.x + sign(in.ext.z) * in.L.x, in.ext.y - in.L.z);
  let ffw = fwidth(fuv);
  var c: vec3f;
  var Nout = N;
  var mirror = 0.0;
  if (in.tint.a > 0.5) {
    // Matte rubble and rock, tinted per piece, with grit.
    let n = textureSampleLevel(t_noise, s_repeat, in.L * 0.23 + in.tint.rgb * 3.0, 0.0);
    let n2 = textureSampleLevel(t_noise, s_repeat, in.L * 1.1, 0.0);
    var albedo = in.col.rgb * in.tint.rgb * (0.55 + 0.9 * n.g) * (0.8 + 0.4 * n2.b);
    // Broken faces show darker insides.
    albedo = mix(albedo, albedo * 0.6, smoothstep(0.5, 0.85, n.b));
    // Faces of a fresh break (col.a = 1 on the broken-concrete meshes): grey
    // concrete whatever the facade was, coarse, with darker pits.
    let broken = m > 0.5 && m < 1.5;
    if (broken) {
      let n3 = textureSampleLevel(t_noise, s_repeat, in.L * 0.6 + 0.37, 0.0);
      let conc = vec3f(0.36, 0.35, 0.33) * mix(vec3f(1.0), in.tint.rgb / max(max(in.tint.r, in.tint.g), max(in.tint.b, 0.05)), 0.25);
      albedo = mix(albedo, in.col.rgb * conc * (0.6 + 0.8 * n3.r) * (0.75 + 0.5 * n2.g), 0.7);
      albedo *= 1.0 - 0.45 * smoothstep(0.62, 0.8, n3.g);
    }
    if (m > 1.5) {
      // Reinforcing bar: rusty steel, dusted.
      let n4 = textureSampleLevel(t_noise, s_repeat, in.P * 0.9, 0.0);
      albedo = mix(vec3f(0.085, 0.052, 0.036), vec3f(0.16, 0.15, 0.14), 0.25 * n4.r) * (0.75 + 0.5 * n4.g);
    }
    c = shade(in.P, N, albedo);
    // Rock glowing hot (tint.a 3, a meteor): molten through its cracks, and
    // all over where it faces its flight (ext.x: how hot).
    if (in.tint.a > 2.5) {
      let n5 = textureSampleLevel(t_noise, s_repeat, in.L * 0.7 + 0.21, 0.0);
      let crack = smoothstep(0.52, 0.7, n5.r) * (0.6 + 0.4 * n5.g);
      c = c * 0.5 + vec3f(1.0, 0.33, 0.07) * (2.0 + 26.0 * crack) * in.ext.x;
    }
    // The outer face of a piece of facade (local +y) is drawn as the building drew it.
    if (in.tint.a > 1.5 && m < 0.5 && in.Ln.y > 0.5 && in.ext.z != 0.0) {
      let st = u32(in.ext.w);
      let fty = (st >> 8u) & 255u;
      let fs = facade_shade(in.P, V, N, in.T, fuv, ffw, in.col.rgb * in.tint.rgb, fty & 15u, f32(st & 255u) / 255.0, f32((st >> 16u) & 255u) / 255.0, abs(in.ext.z), (fty & 128u) != 0u);
      c = shade(in.P, fs.N, fs.albedo) + fs.extra;
      Nout = fs.N;
      mirror = fs.mirror;
    }
  } else {
    // Vehicle: m = 0 paint, 0.25 headlight, 0.5 glass, 0.75 tail light, 1 trim / tyre.
    let paint = step(m, 0.1);
    let glass = step(0.4, m) * step(m, 0.6);
    let head = step(0.15, m) * step(m, 0.35);
    let tail = step(0.65, m) * step(m, 0.85);
    var albedo = mix(in.col.rgb, in.tint.rgb, paint);
    let R = reflect(V, N);
    let ndv = max(dot(-V, N), 0.0);
    let f0 = paint * 0.05 + glass * 0.08 + head * 0.08;
    let fres = f0 + (1.0 - f0) * pow(1.0 - ndv, 5.0) * (paint * 0.6 + glass + head * 0.5);
    let Hh = normalize(G.sun_dir.xyz - V);
    let spec = pow(max(dot(N, Hh), 0.0), 220.0) * (paint + glass) * 2.5 * sh;
    c = shade(in.P, N, albedo * (1.0 - glass * 0.5));
    c += sky_lut(normalize(R + l.upn * 0.12)) * fres + l.sun * spec * 0.3;
    // Lights come on at dusk.
    let night = 1.0 - smoothstep(-0.05, 0.12, dot(l.upn, G.sun_dir.xyz));
    c += (head * vec3f(1.0, 0.95, 0.8) * 3.0 + tail * vec3f(1.0, 0.04, 0.02) * 0.9) * G.sun_dir.w * 0.06 * (0.15 + night);
  }
  return GOut(vec4f(c, 1.0), vec4f(Nout, -mirror), amb_share(c));
}
