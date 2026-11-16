// The 747-400's flight deck (aircraft/b747/deck.rs): printed panel legends,
// button faces and knob scales, and the six CRT displays (PFD, ND and the two
// EICAS screens), the CDUs and the small readouts, all drawn from the
// instrument readings. Appended to plane.wgsl, with the string table (STRS,
// from deck.rs) in front of it.
//
// params[0] IAS kt, altitude ft, vertical speed fpm, heading deg
// params[1] pitch deg, bank deg, turn rate, slip
// params[2] N1 (fraction), EGT C, fuel flow kg/h an engine, fuel on board kg
// params[3] Mach, ground speed kt, radio altitude ft, TAT C
// params[4] throttle 0..1, local time s, gear 0..1, flaps deg

fn str_len(id: u32) -> u32 { return STRS[id].w >> 24u; }
fn str_glyph(id: u32, k: u32) -> u32 { return (STRS[id][k / 4u] >> (8u * (k % 4u))) & 255u; }

// Ink of one glyph cell: q in glyph units (1 wide, 2 tall), stroke half width w.
fn glyph_ink(g: u32, q: vec2f, w: f32, aa: f32) -> f32 {
  if (g == 15u || q.y < -0.6 || q.y > 2.6) { return 0.0; }
  return 1.0 - smoothstep(w - aa, w + aa, glyph_d(g, q));
}
// String `id` at c, glyph height h; align -1 left, 0 centred, 1 right.
fn str_at(p: vec2f, c: vec2f, id: u32, h: f32, aa: f32, align: f32) -> f32 {
  let n = str_len(id);
  let s = h * 0.5;
  let w = (f32(n) * 1.45 - 0.45) * s;
  let q = (p - c + vec2f(w * 0.5 * (1.0 + align), h * 0.5)) / s;
  if (q.x < -0.6 || q.x > f32(n) * 1.45) { return 0.0; }
  let k = i32(floor((q.x + 0.225) / 1.45));
  if (k < 0 || k >= i32(n)) { return 0.0; }
  return glyph_ink(str_glyph(id, u32(k)), vec2f(q.x - f32(k) * 1.45, q.y), 0.19, aa / s + 0.03);
}
// Made-up text, n glyphs (letters, a few digits, gaps between words).
fn rtext(p: vec2f, c: vec2f, n: u32, seed: f32, h: f32, aa: f32, align: f32) -> f32 {
  let s = h * 0.5;
  let w = (f32(n) * 1.45 - 0.45) * s;
  let q = (p - c + vec2f(w * 0.5 * (1.0 + align), h * 0.5)) / s;
  if (q.x < -0.6 || q.x > f32(n) * 1.45) { return 0.0; }
  let k = i32(floor((q.x + 0.225) / 1.45));
  if (k < 0 || k >= i32(n)) { return 0.0; }
  let r = hash1(seed * 7.13 + f32(k) * 1.618);
  var g = 16u + u32(hash1(seed * 3.7 + f32(k) * 9.1) * 22.0);
  if (r < 0.14 && k > 0 && k < i32(n) - 1) { g = 15u; }
  if (r > 0.93) { g = u32(hash1(seed + f32(k)) * 10.0); }
  return glyph_ink(g, vec2f(q.x - f32(k) * 1.45, q.y), 0.19, aa / s + 0.03);
}
// A number with `dec` decimals (and a leading '.' when it is below 1 and lead is false), padded to `pad` digits.
fn num_at(p: vec2f, c: vec2f, v: f32, dec: u32, pad: u32, h: f32, aa: f32, align: f32) -> f32 {
  var g = array<u32, 10>(15u, 15u, 15u, 15u, 15u, 15u, 15u, 15u, 15u, 15u);
  var n = 0u;
  let neg = v < 0.0;
  var x = u32(round(abs(v) * pow(10.0, f32(dec))));
  var digs = array<u32, 10>(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
  var nd = 0u;
  loop {
    digs[nd] = x % 10u;
    x = x / 10u;
    nd = nd + 1u;
    if ((x == 0u && nd > dec && nd >= pad) || nd >= 9u) { break; }
  }
  if (neg) { g[n] = 39u; n = n + 1u; }
  for (var i = 0u; i < nd; i++) {
    let k = nd - 1u - i;
    if (dec > 0u && k == dec - 1u) { g[n] = 14u; n = n + 1u; }
    g[n] = digs[k];
    n = n + 1u;
    if (n >= 10u) { break; }
  }
  let s = h * 0.5;
  let w = (f32(n) * 1.45 - 0.45) * s;
  let q = (p - c + vec2f(w * 0.5 * (1.0 + align), h * 0.5)) / s;
  if (q.x < -0.6 || q.x > f32(n) * 1.45 || q.y < -0.6 || q.y > 2.6) { return 0.0; }
  let k = i32(floor((q.x + 0.225) / 1.45));
  if (k < 0 || k >= i32(n)) { return 0.0; }
  return glyph_ink(g[k], vec2f(q.x - f32(k) * 1.45, q.y), 0.19, aa / s + 0.03);
}

fn aa_fill(d: f32, px: f32) -> f32 { return 1.0 - smoothstep(-px, px, d); }
fn aa_line(d: f32, w: f32, px: f32) -> f32 { return 1.0 - smoothstep(w - px, w + px, abs(d)); }
fn sd_box(p: vec2f, c: vec2f, h: vec2f) -> f32 {
  let q = abs(p - c) - h;
  return length(max(q, vec2f(0.0))) + min(max(q.x, q.y), 0.0);
}
fn box_line(p: vec2f, c: vec2f, h: vec2f, w: f32, px: f32) -> f32 { return aa_line(sd_box(p, c, h), w, px); }
fn seg_line(p: vec2f, a: vec2f, b: vec2f, w: f32, px: f32) -> f32 { return 1.0 - smoothstep(w - px, w + px, seg_d(p, a, b)); }
fn put(c: vec3f, col: vec3f, a: f32) -> vec3f { return mix(c, col, sat(a)); }

const C_WHITE = vec3f(0.92, 0.94, 0.96);
const C_GREEN = vec3f(0.15, 1.0, 0.28);
const C_MAG = vec3f(1.0, 0.28, 1.0);
const C_CYAN = vec3f(0.25, 0.9, 1.0);
const C_AMBER = vec3f(1.0, 0.62, 0.08);
const C_SKY = vec3f(0.06, 0.32, 0.95);
const C_GND = vec3f(0.50, 0.26, 0.07);
const C_TAPE = vec3f(0.16, 0.16, 0.20);

// ---------------------------------------------------------------- legends

struct Legend { col: vec3f, a: f32, glow: vec3f, lit: vec3f };

// A printed legend, a button's face or a knob's scale. uv is in units of the
// quad's half height (v -1..1, u -asp..asp).
fn legend(style: u32, s: u32, uv: vec2f, asp: f32, px: f32, seed: f32) -> Legend {
  var r = Legend(vec3f(0.0), 0.0, vec3f(0.0), vec3f(0.0));
  let aa = max(px, 0.002);
  let ink = vec3f(0.80, 0.80, 0.77);
  let fits = u32(max(floor(asp * 2.0 / 0.58 * 0.9), 1.0));
  switch style {
    case 0u, 9u: { // a line of text
      var a = 0.0;
      if (s > 0u) {
        let n = f32(str_len(s));
        let h = min(1.4, asp * 2.0 / ((n * 1.45 - 0.45) * 0.5) * 0.92);
        a = str_at(uv, vec2f(0.0), s, h, aa, 0.0);
      } else {
        a = rtext(uv, vec2f(0.0), min(fits, 14u), seed, 1.3, aa, 0.0);
      }
      r.col = select(ink, vec3f(0.55, 0.04, 0.03), style == 9u);
      r.a = a;
      r.glow = r.col * a;
    }
    case 1u: { // two lines
      var a = 0.0;
      if (s > 0u) { a = str_at(uv, vec2f(0.0, 0.45), s, 0.75, aa, 0.0); }
      else { a = rtext(uv, vec2f(0.0, 0.45), min(u32(f32(fits) * 1.6), 12u), seed, 0.7, aa, 0.0); }
      a = max(a, rtext(uv, vec2f(0.0, -0.5), min(u32(f32(fits) * 1.4), 10u), seed + 5.0, 0.7, aa, 0.0));
      r.col = ink;
      r.a = a;
      r.glow = ink * a;
    }
    case 2u, 3u, 4u, 10u: { // a push button's face: its legend; lit or not
      let top = vec2f(0.0, 0.4);
      var a = 0.0;
      let h = min(0.62, asp * 2.0 / 3.4);
      if (s > 0u) {
        let n = f32(str_len(s));
        let hh = min(0.62, asp * 1.8 / max((n * 1.45 - 0.45) * 0.5, 0.5));
        a = str_at(uv, top, s, hh, aa, 0.0);
      } else { a = rtext(uv, top, 3u + u32(hash1(seed) * 2.0), seed, h, aa, 0.0); }
      r.col = vec3f(0.62, 0.62, 0.60);
      r.a = a * 0.9;
      r.glow = r.col * a * 0.6;
      // The lower half: a legend that lights (OFF, VALVE, ON...), dark grey until it does.
      let lo = vec2f(0.0, -0.45);
      let words = array<u32, 4>(S_OFF, S_VALVE, S_FAULT, S_ON);
      let w = words[u32(hash1(seed * 1.3) * 3.99)];
      let b = str_at(uv, lo, select(w, S_ON, style == 4u), h * 0.9, aa, 0.0);
      if (style == 3u || style == 4u) {
        let lit = select(C_AMBER, vec3f(0.85, 0.95, 1.0), style == 4u);
        r.col = mix(r.col, lit, b);
        r.a = max(r.a, b);
        r.lit += lit * b;
      } else if (style == 10u) {
        // An active mode: the bar under the legend lights.
        let bar = aa_fill(sd_box(uv, lo, vec2f(asp * 0.62, 0.17)), aa);
        let lit = vec3f(1.0, 0.42, 0.12);
        r.col = mix(r.col, lit, bar);
        r.a = max(r.a, bar);
        r.lit += lit * bar;
      } else {
        r.col = mix(r.col, vec3f(0.10), b);
        r.a = max(r.a, b * 0.5);
      }
    }
    case 5u: { // a knob's scale: ticks round it, a word at each position
      let n = 3.0 + f32(s % 5u);
      let ang = atan2(uv.x, uv.y);
      let rr = length(uv);
      let span = 2.4;
      let st = span / (n - 1.0);
      let k = clamp(round((ang + span * 0.5) / st), 0.0, n - 1.0);
      let a0 = -span * 0.5 + k * st;
      let d = abs(wrapa(ang - a0)) * rr;
      var a = (1.0 - smoothstep(0.035, 0.035 + aa * 2.0, d)) * step(0.44, rr) * step(rr, 0.58);
      let lp = vec2f(sin(a0), cos(a0)) * 0.78;
      a = max(a, rtext(uv, lp, 2u + u32(hash1(k + f32(s)) * 2.0), f32(s) * 3.0 + k, 0.13, aa, 0.0));
      r.col = ink;
      r.a = a;
      r.glow = ink * a;
    }
    case 6u: { // a schematic: flow lines between the switches (the fuel panel)
      let g = uv * vec2f(4.2, 2.6);
      let cell = floor(g);
      let f = fract(g) - 0.5;
      var a = 0.0;
      if (hash2(cell + 3.1) > 0.35) { a = max(a, aa_line(f.y, 0.035, aa * 4.0)); }
      if (hash2(cell + 7.7) > 0.6) { a = max(a, aa_line(f.x, 0.035, aa * 4.0)); }
      a *= step(abs(uv.x), asp - 0.05) * step(abs(uv.y), 0.95);
      r.col = ink;
      r.a = a * 0.8;
      r.glow = ink * a;
    }
    case 7u: { // a printed card: black text lines on white
      let lnum = floor((uv.y + 1.0) / 0.09);
      let ly = (uv.y + 1.0) - (lnum + 0.5) * 0.09;
      let len = 0.4 + 0.5 * hash1(lnum * 3.7);
      let a = rtext(vec2f(uv.x, ly), vec2f(-asp * 0.85, 0.0), u32(len * 18.0), lnum + seed, 0.055, aa, -1.0) * step(abs(uv.y), 0.9);
      let title = step(0.78, uv.y) * aa_fill(sd_box(uv, vec2f(0.0, 0.86), vec2f(asp * 0.9, 0.06)), aa);
      r.col = vec3f(0.03);
      r.a = max(a, title * 0.6);
    }
    case 8u: { // a CDU key's legend
      var a = 0.0;
      if (s > 0u) {
        let n = f32(str_len(s));
        let hh = min(0.9, asp * 1.7 / max((n * 1.45 - 0.45) * 0.5, 0.5));
        a = str_at(uv, vec2f(0.0), s, hh, aa, 0.0);
      } else { a = rtext(uv, vec2f(0.0), 1u, seed, 0.9, aa, 0.0); }
      r.col = vec3f(0.85);
      r.a = a;
      r.glow = vec3f(0.85) * a;
    }
    default: {}
  }
  return r;
}

// ---------------------------------------------------------------- displays

// The attitude, speed, altitude and heading of the primary flight display.
fn pfd(p: vec2f, px: f32) -> vec3f {
  let P0 = O.params[0];
  let P1 = O.params[1];
  let P3 = O.params[3];
  let ias = P0.x;
  let alt = P0.y;
  let vs = P0.z;
  let hdg = P0.w;
  let pitch = P1.x;
  let bank = P1.y;
  var c = vec3f(0.0);
  let aa = px;
  // Attitude: sky and ground behind a rounded window, the pitch ladder, the bank scale.
  let ac = vec2f(0.0, 0.06);
  let k = 0.021;
  let win = sd_box(p, ac, vec2f(0.40, 0.44)) - 0.06;
  if (win < 0.0) {
    let q = rot2(p - ac, rad(bank));
    let hy = -pitch * k;
    c = select(C_GND, C_SKY, q.y > hy);
    c = put(c, C_WHITE, aa_line(q.y - hy, 0.006, aa));
    // Ladder every 2.5 degrees, numbered every 10.
    let rel = q.y / k + pitch;
    let step_ = round(rel / 2.5) * 2.5;
    let dy = (rel - step_) * k;
    let m10 = abs(step_ % 10.0) < 0.01;
    let m5 = abs(step_ % 5.0) < 0.01;
    let hw = select(select(0.035, 0.07, m5), 0.14, m10);
    if (abs(step_) > 0.1 && abs(q.y) < 0.30) {
      c = put(c, C_WHITE, aa_line(dy, 0.0045, aa) * step(abs(q.x), hw));
      if (m10) {
        let a = num_at(q, vec2f(-hw - 0.06, step_ * k - pitch * k), abs(step_), 0u, 1u, 0.05, aa, 0.0);
        let b = num_at(q, vec2f(hw + 0.06, step_ * k - pitch * k), abs(step_), 0u, 1u, 0.05, aa, 0.0);
        c = put(c, C_WHITE, max(a, b));
      }
    }
  }
  c = put(c, vec3f(0.0), aa_line(win, 0.004, aa));
  // Bank scale: an arc of ticks, the pointer under it with the slip brick.
  let br = 0.40;
  let rp = p - ac;
  let ang = atan2(rp.x, rp.y);
  let rl = length(rp);
  for (var i = 0; i < 11; i++) {
    let marks = array<f32, 11>(-60.0, -45.0, -30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0, 45.0, 60.0);
    let a0 = rad(marks[i]);
    let l = select(0.035, 0.06, abs(marks[i]) == 30.0 || abs(marks[i]) == 60.0);
    let d = vec2f(sin(a0), cos(a0));
    if (marks[i] != 0.0) { c = put(c, C_WHITE, seg_line(rp, d * br, d * (br + l), 0.005, aa)); }
  }
  // The fixed index at the top, the moving pointer under it.
  let tri = sd_poly(rp - vec2f(0.0, br + 0.005), array<vec2f, 5>(vec2f(-0.03, 0.05), vec2f(0.03, 0.05), vec2f(0.0, 0.0), vec2f(-0.03, 0.05), vec2f(-0.03, 0.05)), 3);
  c = put(c, C_WHITE, aa_line(tri, 0.004, aa));
  let rq = rot2(rp, rad(bank));
  let ptr = sd_poly(rq - vec2f(0.0, br - 0.005), array<vec2f, 5>(vec2f(-0.03, -0.05), vec2f(0.0, 0.0), vec2f(0.03, -0.05), vec2f(-0.03, -0.05), vec2f(-0.03, -0.05)), 3);
  c = put(c, C_WHITE, aa_line(ptr, 0.004, aa));
  let slip = clamp(P1.w, -1.0, 1.0) * 0.05;
  c = put(c, C_WHITE, box_line(rq, vec2f(slip, br - 0.075), vec2f(0.032, 0.012), 0.004, aa));
  // The aeroplane symbol: two L-shaped wings and a square, black outlined.
  let wl = min(sd_box(abs(p - ac) * vec2f(1.0, 1.0), vec2f(0.20, 0.0), vec2f(0.075, 0.012)), sd_box(abs(p - ac), vec2f(0.135, -0.03), vec2f(0.012, 0.035)) * select(1.0, 1e3, p.y > ac.y + 0.01));
  let sq = sd_box(p, ac, vec2f(0.014, 0.014));
  let sym = min(wl, sq);
  c = put(c, vec3f(0.0), aa_fill(sym - 0.006, aa));
  c = put(c, C_WHITE * 0.95, aa_fill(sym, aa));
  // Flight director bars.
  c = put(c, C_MAG, aa_line(p.x - ac.x - 0.01, 0.006, aa) * step(abs(p.y - ac.y), 0.2));
  c = put(c, C_MAG, aa_line(p.y - ac.y + 0.012, 0.006, aa) * step(abs(p.x - ac.x), 0.2));

  // Speed tape.
  let sx0 = -0.86;
  let sx1 = -0.58;
  let sc = 0.0072;
  if (p.x > sx0 && p.x < sx1 && abs(p.y - 0.04) < 0.56) {
    c = C_TAPE;
    let sv = ias + (p.y - 0.04) / sc;
    let t10 = round(sv / 10.0) * 10.0;
    let ty = (t10 - ias) * sc + 0.04;
    c = put(c, C_WHITE, aa_line(p.y - ty, 0.004, aa) * step(sx1 - 0.035, p.x));
    let t20 = round(sv / 20.0) * 20.0;
    if (t20 >= 0.0) {
      c = put(c, C_WHITE, num_at(p, vec2f(sx1 - 0.05, (t20 - ias) * sc + 0.04), t20, 0u, 1u, 0.058, aa, 1.0));
    }
    // The speed limits: the red and black barber pole near the top.
    let vmo = 365.0;
    let y_vmo = (vmo - ias) * sc + 0.04;
    if (p.y > y_vmo && p.x > sx1 - 0.02) { c = select(vec3f(0.0), vec3f(0.9, 0.05, 0.03), fract((p.y - y_vmo) / 0.06) < 0.5); }
  }
  // Its readout.
  let sb = sd_box(p, vec2f(-0.75, 0.04), vec2f(0.12, 0.055));
  if (sb < 0.0) { c = vec3f(0.0); }
  c = put(c, C_WHITE, aa_line(sb, 0.004, aa));
  c = put(c, C_WHITE, num_at(p, vec2f(-0.65, 0.04), floor(ias), 0u, 1u, 0.075, aa, 1.0));
  c = put(c, C_MAG, num_at(p, vec2f(-0.72, 0.68), round(max(ias, 140.0) / 10.0) * 10.0, 0u, 1u, 0.07, aa, 0.0));
  if (P3.x > 0.4) { c = put(c, C_WHITE, num_at(p, vec2f(-0.72, -0.62), P3.x * 1000.0, 0u, 3u, 0.065, aa, 0.0) * step(-0.83, p.x)); c = put(c, C_WHITE, aa_fill(length(p - vec2f(-0.81, -0.652)) - 0.007, aa)); }

  // Altitude tape and readout, selected altitude over it.
  let ax0 = 0.56;
  let ax1 = 0.84;
  let asc = 0.0011;
  if (p.x > ax0 && p.x < ax1 && abs(p.y - 0.04) < 0.56) {
    c = C_TAPE;
    let av = alt + (p.y - 0.04) / asc;
    let t1 = round(av / 100.0) * 100.0;
    c = put(c, C_WHITE, aa_line(p.y - ((t1 - alt) * asc + 0.04), 0.004, aa) * step(p.x, ax0 + 0.035));
    let t2 = round(av / 200.0) * 200.0;
    c = put(c, C_WHITE, num_at(p, vec2f(ax0 + 0.05, (t2 - alt) * asc + 0.04), t2, 0u, 1u, 0.05, aa, -1.0));
  }
  let ab = sd_box(p, vec2f(0.72, 0.04), vec2f(0.15, 0.055));
  if (ab < 0.0) { c = vec3f(0.0); }
  c = put(c, C_WHITE, aa_line(ab, 0.004, aa));
  c = put(c, C_WHITE, num_at(p, vec2f(0.85, 0.04), round(alt / 20.0) * 20.0, 0u, 1u, 0.07, aa, 1.0));
  c = put(c, C_MAG, num_at(p, vec2f(0.70, 0.68), (floor(alt / 1000.0) + 1.0) * 1000.0, 0u, 1u, 0.07, aa, 0.0));
  c = put(c, C_GREEN, str_at(p, vec2f(0.70, -0.62), S_STD, 0.06, aa, 0.0));

  // Vertical speed: a scale on the right, the needle from the far edge.
  if (p.x > 0.88 && abs(p.y - 0.04) < 0.42) {
    c = put(c, C_TAPE, 0.9);
    let marks = array<f32, 7>(-0.36, -0.24, -0.12, 0.0, 0.12, 0.24, 0.36);
    for (var i = 0; i < 7; i++) { c = put(c, C_WHITE, aa_line(p.y - 0.04 - marks[i], 0.004, aa) * step(p.x, 0.92)); }
    let v = clamp(vs / 1000.0, -6.0, 6.0);
    let vy = sign(v) * select(abs(v) * 0.12, 0.24 + (abs(v) - 2.0) * 0.03, abs(v) > 2.0) + 0.04;
    c = put(c, C_WHITE, seg_line(p, vec2f(0.89, vy), vec2f(1.05, 0.04), 0.005, aa));
  }
  if (abs(vs) > 400.0) { c = put(c, C_WHITE, num_at(p, vec2f(0.94, select(-0.47, 0.55, vs > 0.0)), round(abs(vs) / 50.0) * 50.0, 0u, 1u, 0.045, aa, 0.0)); }

  // Heading: the top of a compass rose under the attitude.
  let rc = vec2f(0.0, -1.38);
  let rr = 0.66;
  let hp = p - rc;
  let hl = length(hp);
  if (hl < rr && p.y > -0.98) {
    c = put(c, C_TAPE, 0.95);
    let a = atan2(hp.x, hp.y);
    let deg = hdg + a * 57.29578;
    let t5 = round(deg / 5.0) * 5.0;
    let ta = rad(t5 - hdg);
    let d = abs(wrapa(a - ta)) * hl;
    let long_ = abs(t5 % 10.0) < 0.1;
    c = put(c, C_WHITE, (1.0 - smoothstep(0.004, 0.004 + aa, d)) * step(rr - select(0.03, 0.05, long_), hl));
    let t10 = round(deg / 10.0) * 10.0;
    let la = rad(t10 - hdg);
    let lp = vec2f(sin(la), cos(la)) * (rr - 0.1);
    let ld = rot2(hp - lp, la);
    let lab = (u32(round(t10 + 360.0)) % 360u) / 10u;
    if (abs(t10 % 30.0) < 0.1) { c = put(c, C_WHITE, num_at(ld, vec2f(0.0), f32(lab), 0u, 1u, 0.06, aa, 0.0)); }
  }
  c = put(c, C_WHITE, aa_line(sd_poly(p - vec2f(0.0, rc.y + rr), array<vec2f, 5>(vec2f(-0.025, 0.04), vec2f(0.025, 0.04), vec2f(0.0, 0.0), vec2f(-0.025, 0.04), vec2f(-0.025, 0.04)), 3), 0.004, aa));
  let hb = sd_box(p, vec2f(0.0, -0.65), vec2f(0.075, 0.04));
  // Mode annunciations across the top: autothrottle, roll, pitch; the autopilot status.
  c = put(c, C_WHITE, aa_line(p.x + 0.27, 0.003, aa) * step(0.80, p.y) * step(p.y, 0.97));
  c = put(c, C_WHITE, aa_line(p.x - 0.27, 0.003, aa) * step(0.80, p.y) * step(p.y, 0.97));
  c = put(c, C_GREEN, str_at(p, vec2f(-0.55, 0.88), S_SPD, 0.075, aa, 0.0));
  c = put(c, C_GREEN, str_at(p, vec2f(0.0, 0.88), S_LNAV, 0.075, aa, 0.0));
  c = put(c, C_GREEN, str_at(p, vec2f(0.55, 0.88), S_VNAV_PTH, 0.075, aa, 0.0));
  c = put(c, C_GREEN, str_at(p, vec2f(0.0, 0.62), S_CMD, 0.09, aa, 0.0));
  let _hb = hb;
  // Radio altitude when low.
  if (P3.z < 2500.0) { c = put(c, C_WHITE, num_at(p, vec2f(0.0, -0.36), round(P3.z / 10.0) * 10.0, 0u, 1u, 0.08, aa, 0.0)); }
  return c;
}

// The navigation display, map mode.
fn nd(p: vec2f, px: f32) -> vec3f {
  let P0 = O.params[0];
  let P3 = O.params[3];
  let hdg = P0.w;
  var c = vec3f(0.0);
  let aa = px;
  let ac = vec2f(0.0, -0.66);
  let R = 1.30;
  let q = p - ac;
  let rl = length(q);
  let a = atan2(q.x, q.y);
  // The compass arc with its ticks and numbers.
  c = put(c, C_WHITE, aa_line(rl - R, 0.005, aa) * step(abs(a), 0.95));
  let deg = hdg + a * 57.29578;
  let t5 = round(deg / 5.0) * 5.0;
  let ta = rad(t5 - hdg);
  let d = abs(wrapa(a - ta)) * rl;
  let long_ = abs(t5 % 10.0) < 0.1;
  c = put(c, C_WHITE, (1.0 - smoothstep(0.005, 0.005 + aa, d)) * step(R - select(0.03, 0.06, long_), rl) * step(rl, R) * step(abs(a), 0.95));
  let t30 = round(deg / 30.0) * 30.0;
  let la = rad(t30 - hdg);
  if (abs(la) < 0.85) {
    let lp = vec2f(sin(la), cos(la)) * (R - 0.12);
    let lab = (u32(round(t30 + 360.0)) % 360u) / 10u;
    c = put(c, C_WHITE, num_at(rot2(q - lp, la), vec2f(0.0), f32(lab), 0u, 1u, 0.08, aa, 0.0));
  }
  // Half range ring, dashed.
  c = put(c, C_WHITE * 0.8, aa_line(rl - R * 0.5, 0.004, aa) * step(0.5, fract(a * 9.0)) * step(abs(a), 1.3));
  // The route: a magenta line along the active leg through the next waypoints.
  let crs = rad(292.0 - hdg);
  let dir = vec2f(sin(crs), cos(crs));
  let off = vec2f(cos(crs), -sin(crs)) * 0.05;
  let rp = q - off;
  let along = dot(rp, dir);
  let across = rp.x * dir.y - rp.y * dir.x;
  c = put(c, C_MAG, aa_line(across, 0.005, aa) * step(-0.25, along) * step(rl, R));
  // Waypoints: four-pointed stars with their names.
  for (var i = 0; i < 3; i++) {
    let wd = 0.35 + f32(i) * 0.55;
    let wp = off + dir * wd;
    let sp = q - wp;
    let star = min(abs(sp.x) + abs(sp.y) * 4.0, abs(sp.x) * 4.0 + abs(sp.y)) - 0.06;
    let col = select(C_WHITE, C_MAG, i == 0);
    if (length(wp) < R) {
      c = put(c, col, aa_line(star, 0.005, aa));
      c = put(c, col, str_at(p, ac + wp + vec2f(0.08, -0.02), select(S_CARPE, S_DERBY, i == 1), 0.06, aa, -1.0));
    }
  }
  // An airport off to the side.
  let ap = rot2(vec2f(-0.55, 0.62), rad(hdg - 280.0) * 0.0);
  c = put(c, C_CYAN, aa_line(length(q - ap) - 0.035, 0.005, aa));
  c = put(c, C_CYAN, str_at(p, ac + ap + vec2f(0.06, -0.05), S_SFO, 0.055, aa, -1.0));
  // The aeroplane.
  let tri = sd_poly(q, array<vec2f, 5>(vec2f(-0.045, -0.11), vec2f(0.0, 0.0), vec2f(0.045, -0.11), vec2f(-0.045, -0.11), vec2f(-0.045, -0.11)), 3);
  c = put(c, C_WHITE, aa_line(tri, 0.005, aa));
  // The heading box at the top, the labels round it.
  let hb = sd_box(p, vec2f(0.0, 0.83), vec2f(0.12, 0.065));
  if (hb < 0.0) { c = vec3f(0.0); }
  c = put(c, C_WHITE, aa_line(hb, 0.005, aa));
  c = put(c, C_WHITE, num_at(p, vec2f(0.0, 0.83), floor(hdg + 0.5) % 360.0, 0u, 3u, 0.09, aa, 0.0));
  c = put(c, C_GREEN, str_at(p, vec2f(-0.15, 0.82), S_HDG, 0.06, aa, 1.0));
  c = put(c, C_GREEN, str_at(p, vec2f(0.15, 0.82), S_MAG, 0.06, aa, -1.0));
  // Ground speed and true airspeed, the wind.
  c = put(c, C_WHITE, str_at(p, vec2f(-0.93, 0.86), S_GS, 0.05, aa, -1.0));
  c = put(c, C_WHITE, num_at(p, vec2f(-0.85, 0.86), round(P3.y), 0u, 1u, 0.07, aa, -1.0));
  c = put(c, C_WHITE, str_at(p, vec2f(-0.6, 0.86), S_TAS, 0.05, aa, -1.0));
  c = put(c, C_WHITE, num_at(p, vec2f(-0.49, 0.86), round(P0.x * (1.0 + P0.y * 0.000017)), 0u, 1u, 0.07, aa, -1.0));
  // The active waypoint, its time and distance.
  c = put(c, C_MAG, str_at(p, vec2f(0.93, 0.88), S_CARPE, 0.065, aa, 1.0));
  c = put(c, C_WHITE, str_at(p, vec2f(0.93, 0.78), S_2219_0Z, 0.058, aa, 1.0));
  c = put(c, C_WHITE, str_at(p, vec2f(0.93, 0.68), S_35_0NM, 0.058, aa, 1.0));
  c = put(c, C_GREEN, str_at(p, vec2f(-0.92, -0.90), S_VOR_L, 0.06, aa, -1.0));
  c = put(c, C_CYAN, str_at(p, vec2f(-0.93, -0.70), S_TFC, 0.05, aa, -1.0));
  c = put(c, C_CYAN, str_at(p, vec2f(-0.93, -0.78), S_ARPT, 0.05, aa, -1.0));
  return c;
}

// One engine's vertical tape: value 0..1 of full scale, the box with the reading over it.
fn eng_tape(p: vec2f, x: f32, y0: f32, y1: f32, v: f32, red: f32, aa: f32) -> vec3f {
  var c = vec3f(0.0);
  let w = 0.028;
  let fy = y0 + (y1 - y0) * sat(v);
  let o = box_line(p, vec2f(x, (y0 + y1) * 0.5), vec2f(w, (y1 - y0) * 0.5), 0.004, aa);
  c = put(c, C_WHITE, o);
  c = put(c, C_WHITE * 0.92, aa_fill(sd_box(p, vec2f(x, (y0 + fy) * 0.5), vec2f(w - 0.009, max((fy - y0) * 0.5 - 0.004, 0.0))), aa));
  let ry = y0 + (y1 - y0) * red;
  c = put(c, vec3f(1.0, 0.1, 0.05), aa_line(p.y - ry, 0.006, aa) * step(abs(p.x - x), w + 0.02));
  return c;
}

// Upper EICAS: EPR, N1 and EGT for each engine, the gear and flaps, fuel and temperature.
fn eicas(p: vec2f, px: f32) -> vec3f {
  let P2 = O.params[2];
  let P3 = O.params[3];
  let P4 = O.params[4];
  var c = vec3f(0.0);
  let aa = px;
  let n1 = P2.x;
  for (var i = 0; i < 4; i++) {
    let x = -0.80 + f32(i) * 0.24;
    if (abs(p.x - x) > 0.12) { continue; }
    let jit = (hash1(f32(i) * 3.3) - 0.5) * 0.006;
    let e_n1 = max(n1 + jit, 0.0);
    let epr = 1.0 + 0.62 * pow(sat((e_n1 - 0.2) / 0.85), 1.6);
    let egt = P2.y + (hash1(f32(i) + 7.0) - 0.5) * 18.0;
    // EPR
    c += eng_tape(p, x, 0.30, 0.66, (epr - 0.9) / 0.8, 0.95, aa);
    c = put(c, C_WHITE, box_line(p, vec2f(x, 0.75), vec2f(0.085, 0.045), 0.004, aa));
    c = put(c, C_WHITE, num_at(p, vec2f(x, 0.75), epr, 2u, 1u, 0.06, aa, 0.0));
    c = put(c, C_GREEN, num_at(p, vec2f(x, 0.86), 1.58, 2u, 1u, 0.05, aa, 0.0));
    c = put(c, C_GREEN, aa_line(p.y - (0.30 + 0.36 * (1.58 - 0.9) / 0.8), 0.005, aa) * step(abs(p.x - x), 0.045));
    // N1
    c += eng_tape(p, x, -0.24, 0.12, e_n1 / 1.15, 1.115 / 1.15, aa);
    c = put(c, C_WHITE, box_line(p, vec2f(x, 0.20), vec2f(0.085, 0.045), 0.004, aa));
    c = put(c, C_WHITE, num_at(p, vec2f(x, 0.20), e_n1 * 100.0, 1u, 1u, 0.06, aa, 0.0));
    // EGT
    c += eng_tape(p, x, -0.78, -0.42, egt / 700.0, 650.0 / 700.0, aa);
    c = put(c, C_WHITE, box_line(p, vec2f(x, -0.34), vec2f(0.085, 0.045), 0.004, aa));
    c = put(c, C_WHITE, num_at(p, vec2f(x, -0.34), round(egt), 0u, 1u, 0.06, aa, 0.0));
  }
  c = put(c, C_CYAN, str_at(p, vec2f(-0.44, 0.48), S_EPR, 0.05, aa, 0.0));
  c = put(c, C_CYAN, str_at(p, vec2f(-0.44, -0.06), S_N1, 0.05, aa, 0.0));
  c = put(c, C_CYAN, str_at(p, vec2f(-0.44, -0.60), S_EGT, 0.05, aa, 0.0));
  // TAT at the top left.
  c = put(c, C_CYAN, str_at(p, vec2f(-0.92, 0.92), S_TAT, 0.045, aa, -1.0));
  c = put(c, C_WHITE, num_at(p, vec2f(-0.78, 0.92), round(P3.w), 0u, 1u, 0.05, aa, -1.0));
  // Gear: a green DOWN box when down and locked, crosshatch in transit, UP after retraction.
  let g = P4.z;
  let gb = sd_box(p, vec2f(0.62, -0.18), vec2f(0.11, 0.045));
  if (g > 0.999) {
    c = put(c, C_GREEN, aa_line(gb, 0.005, aa));
    c = put(c, C_GREEN, str_at(p, vec2f(0.62, -0.18), S_DOWN, 0.055, aa, 0.0));
  } else if (g > 0.001) {
    c = put(c, C_WHITE, aa_line(gb, 0.005, aa));
    c = put(c, C_WHITE, aa_line(fract((p.x + p.y) * 18.0) - 0.5, 0.08, aa * 18.0) * step(gb, 0.0));
  }
  if (g < 0.999) { c = put(c, C_WHITE, str_at(p, vec2f(0.62, -0.18), S_UP, 0.055, aa, 0.0) * step(g, 0.001)); }
  c = put(c, C_CYAN, str_at(p, vec2f(0.62, -0.27), S_GEAR, 0.045, aa, 0.0));
  // Flaps, while they are out: a scale with the position.
  let fl = P4.w;
  if (fl > 0.5) {
    let fy = 0.40 - 0.42 * sat(fl / 30.0);
    c = put(c, C_WHITE, box_line(p, vec2f(0.88, 0.19), vec2f(0.022, 0.21), 0.004, aa));
    c = put(c, C_GREEN, aa_fill(sd_box(p, vec2f(0.88, (0.40 + fy) * 0.5), vec2f(0.013, (0.40 - fy) * 0.5)), aa));
    c = put(c, C_MAG, aa_line(p.y - fy, 0.005, aa) * step(abs(p.x - 0.88), 0.04));
    c = put(c, C_GREEN, num_at(p, vec2f(0.80, fy), round(fl), 0u, 1u, 0.05, aa, 1.0));
    c = put(c, C_CYAN, str_at(p, vec2f(0.88, 0.46), S_FLAPS, 0.045, aa, 0.0));
  }
  // Fuel and weight along the bottom.
  c = put(c, C_CYAN, str_at(p, vec2f(-0.15, -0.88), S_TOTAL_FUEL, 0.045, aa, -1.0));
  c = put(c, C_WHITE, num_at(p, vec2f(0.42, -0.88), P2.w / 1000.0, 1u, 1u, 0.065, aa, 0.0));
  c = put(c, C_CYAN, str_at(p, vec2f(0.55, -0.88), S_KGS_X_1000, 0.04, aa, -1.0));
  return c;
}

// Lower EICAS: the secondary engine page (N2, fuel flow, oil, vibration).
fn eicas2(p: vec2f, px: f32) -> vec3f {
  let P2 = O.params[2];
  var c = vec3f(0.0);
  let aa = px;
  let n1 = P2.x;
  let xs = array<f32, 4>(-0.72, -0.40, 0.40, 0.72);
  let labels = array<u32, 6>(S_N2, S_FF, S_OIL_P, S_OIL_T, S_OIL_Q, S_VIB);
  let ys = array<f32, 6>(0.62, 0.30, 0.0, -0.30, -0.55, -0.80);
  for (var r = 0; r < 6; r++) {
    c = put(c, C_CYAN, str_at(p, vec2f(0.0, ys[r]), labels[r], 0.05, aa, 0.0));
  }
  for (var i = 0; i < 4; i++) {
    let x = xs[i];
    if (abs(p.x - x) > 0.2) { continue; }
    let j = hash1(f32(i) * 5.1);
    let n2 = 0.6 + 0.4 * sat(n1 * 1.05) + (j - 0.5) * 0.01;
    let ff = P2.z * (0.98 + 0.04 * j) / 1000.0;
    let vals = array<f32, 6>(n2 * 100.0, ff, 40.0 + 30.0 * n1 + j * 3.0, 80.0 + 50.0 * n1 + j * 6.0, 17.0 + j * 2.0, 0.3 + 0.6 * n1 * j);
    let decs = array<u32, 6>(1u, 1u, 0u, 0u, 0u, 1u);
    for (var r = 0; r < 6; r++) {
      let bx = vec2f(x, ys[r]);
      c = put(c, C_WHITE, box_line(p, bx, vec2f(0.085, 0.045), 0.004, aa) * step(f32(r), 1.5));
      c = put(c, C_WHITE, num_at(p, bx, vals[r], decs[r], 1u, 0.06, aa, 0.0));
    }
  }
  return c;
}

// A CDU's screen: a page of the flight plan.
fn cdu_screen(p: vec2f, px: f32, centre: bool) -> vec3f {
  var c = vec3f(0.0);
  let aa = px;
  let lh = 2.0 / 14.0;
  let ln = floor((1.0 - p.y) / lh);
  let ly = 1.0 - (ln + 0.5) * lh;
  if (ln < 0.0 || ln > 13.0) { return c; }
  let h = select(0.075, 0.095, ln == 0.0 || (ln % 2.0 == 1.0));
  if (ln == 0.0) {
    c = put(c, C_WHITE, str_at(p, vec2f(0.0, ly), select(S_ACT_RTE_1_LEGS, S_MENU, centre), 0.09, aa, 0.0));
    return c;
  }
  let seed = ln * 13.1 + select(0.0, 50.0, centre);
  let lc = select(select(C_GREEN, C_WHITE, ln % 2.0 == 0.0), C_MAG, ln == 1.0);
  c = put(c, lc, rtext(p, vec2f(-0.93, ly), 5u + u32(hash1(seed) * 4.0), seed, h, aa, -1.0));
  c = put(c, select(C_WHITE, C_GREEN, ln % 2.0 == 0.0), rtext(p, vec2f(0.93, ly), 3u + u32(hash1(seed + 2.0) * 6.0), seed + 9.0, h, aa, 1.0) * step(0.3, hash1(seed + 4.0)));
  return c;
}

// The 747's displays and readouts (DISPLAY extra 20 and up).
fn display747(id: u32, uv: vec2f, px: f32) -> vec3f {
  // CRT faces have rounded corners inside the mask.
  let corner = sd_box(uv, vec2f(0.0), vec2f(0.88)) - 0.12;
  var c = vec3f(0.0);
  let P0 = O.params[0];
  let P4 = O.params[4];
  let pu = max(px, 0.003);
  switch id {
    case 20u: { c = pfd(uv, pu); }
    case 21u: { c = nd(uv, pu); }
    case 22u: { c = eicas(uv, pu); }
    case 23u: { c = eicas2(uv, pu); }
    case 24u: { c = cdu_screen(uv, pu, false); }
    case 25u: { c = cdu_screen(uv, pu, true); }
    case 26u: { // radio frequency, green
      c = put(c, C_GREEN, num_at(uv * vec2f(2.6, 1.0), vec2f(0.0), 118.3 + floor(hash1(floor(uv.x + 3.0)) * 400.0) * 0.025, 2u, 1u, 0.95, pu * 2.6, 0.0) * 0.9);
      return c;
    }
    case 27u: { return C_WHITE * num_at(uv * vec2f(2.6, 1.0), vec2f(0.0), round(max(P0.x, 140.0) / 10.0) * 10.0, 0u, 3u, 1.25, pu * 2.6, 0.0) * 0.8; }
    case 28u: { return C_WHITE * num_at(uv * vec2f(2.35, 1.0), vec2f(0.0), (round(P0.w) + 360.0) % 360.0, 0u, 3u, 1.25, pu * 2.35, 0.0) * 0.8; }
    case 29u: { return vec3f(0.0); }
    case 30u: { return C_WHITE * num_at(uv * vec2f(3.06, 1.0), vec2f(0.0), (floor(P0.y / 1000.0) + 1.0) * 1000.0, 0u, 4u, 1.25, pu * 3.06, 0.0) * 0.8; }
    case 31u: { return C_GREEN * num_at(uv * vec2f(3.1, 1.0), vec2f(0.0), 2000.0, 0u, 4u, 1.2, pu * 3.1, 0.0) * 0.8; }
    case 32u: { // clock: hours and minutes
      let t = P4.y;
      let hm = floor(t / 3600.0) * 100.0 + floor((t % 3600.0) / 60.0);
      return vec3f(0.85, 0.9, 0.8) * num_at(uv * vec2f(2.6, 1.0), vec2f(0.0), hm, 0u, 4u, 1.2, pu * 2.6, 0.0) * 0.7;
    }
    case 33u: { return C_GREEN * num_at(uv * vec2f(2.6, 1.0), vec2f(0.0), 200.0, 0u, 3u, 1.2, pu * 2.6, 0.0) * 0.7; }
    default: { return vec3f(0.0); }
  }
  // Scan lines and the phosphor's slight glow at small sizes.
  return c * (1.0 - smoothstep(-0.01, 0.01, corner));
}

// What the inside of a flight deck window mirrors: the glareshield's dark top
// (lit where the sun falls on it) low in the windshields, the MCP's lights,
// elsewhere the dim deck.
fn deck_mirror(p0: vec3f, r: vec3f, l: Light, ls: vec3f) -> vec3f {
  var t = 0.02;
  let ey = 3.40;
  let ez = -25.95;
  for (var i = 0; i < 24; i++) {
    let p = p0 + r * t;
    let top = ey - 0.17 - 0.22 * max(ez - 0.70 - p.z, 0.0);
    if (p.z < ez - 0.68 && abs(p.x) < 1.0 && p.y < top) {
      let sg = shadow_at((O.model * vec4f(p * kscale(), 1.0)).xyz, normalize(O.model[1].xyz));
      let weave = 0.75 + 0.5 * nz(p * 23.0).g;
      return vec3f(0.032, 0.03, 0.029) * weave / PI * (l.sun * max(ls.y, 0.0) * sg + l.amb * 0.5);
    }
    if (p.y < ey - 0.25 && p.z > ez - 0.75) {
      // The panel below and behind: the displays' glow, dark paint.
      let night = 1.0 - smoothstep(-0.08, 0.1, dot(l.upn, G.sun_dir.xyz));
      return l.amb * 0.025 + vec3f(0.03, 0.05, 0.06) * G.sun_dir.w * 0.004 * (0.3 + night);
    }
    t += 0.04;
  }
  return l.amb * 0.03 * (0.5 + 1.0 * nz(r * 0.9 + vec3f(0.3, 0.1, 0.6)).r);
}
