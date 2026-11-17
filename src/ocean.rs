//! Wind-driven ocean waves. The weather's wind (speed, direction) and the
//! fetch shape a JONSWAP wind sea with Donelan-Banner directional spread, plus
//! a long swell; three tiling cascades of it (about 250 m, 42 m and 6 m across,
//! 256 x 256 each, 128 x 128 at Low water quality) are rebuilt every frame on the GPU by inverse FFT
//! (ocean.wgsl) into height, sideways displacement (sharp crests) and slopes.
//!
//! Whitecaps are kept on the GPU too (ocean.wgsl `foam`): foam goes down where
//! the crests fold (the Jacobian of the sideways shift), builds up, drifts
//! downwind and decays, and its cover is steered to Monahan's W(U).
//!
//! Which cascades a stretch of water shows depends on its fetch (common.wgsl
//! `sea_state`): the open ocean gets all three, the long swell included; bays
//! (ocean near the shore), lakes and rivers lose the long waves and keep the
//! short chop.
//!
//! The CPU keeps its own copy of the two longer cascades (what matters for
//! floating: the 6 m one is finer than any hull), the same spectrum from the
//! same random numbers, so a ditched aircraft or the boat rides the waves that
//! are drawn (`Ocean::height_w`).
use bytemuck::{Pod, Zeroable};
use glam::{DVec2, DVec3};
use std::f64::consts::PI;

/// Texels across each cascade at most (High water quality); Low has half.
pub const N: usize = 256;
/// Texels across each cascade by water quality (0 Low, 1 High).
pub fn grid(quality: u8) -> usize {
    if quality == 0 { N / 2 } else { N }
}
/// Tile sizes (m). Cascade 1 fits six times into cascade 0 exactly, so the
/// whitecaps (kept on cascade 0's grid, from both) land on its crests in
/// every tile (common.wgsl `ocean_size`).
pub const SIZES: [f64; 3] = [250.0, 250.0 / 6.0, 6.0];
/// Band edges (wavelength, m): cascade 0 holds the waves longer than the
/// first, cascade 1 those down to the second, cascade 2 the rest.
const BAND: [f64; 2] = [25.0, 4.0];
/// Sideways displacement as a share of the linear (Gerstner) amount.
const CHOP: [f64; 3] = [0.85, 0.9, 0.6];
/// The ocean's coordinates wrap at a multiple of every tile size.
pub const WRAP: f64 = 3000.0;
/// Seconds after which the sea repeats (wave frequencies are quantised to it,
/// so the time handed to the GPU can stay small).
const PERIOD: f64 = 256.0;
const GRAV: f64 = 9.81;
/// Fetch of the open ocean (m).
const FETCH: f64 = 200_000.0;
/// Wavelength of the long swell (m).
const SWELL_L: f64 = 150.0;
/// Resolution of the CPU copy of cascades 0 and 1 (holds all of their bands).
const M: usize = 64;
/// Cascades kept on the CPU.
const CPU_C: usize = 2;

/// ocean.wgsl `OU`.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Default, Debug)]
pub struct OceanU {
    pub casc: [[f32; 4]; 3],
    pub sea: [f32; 4],
    pub swell: [f32; 4],
    pub dir: [f32; 4],
    pub t: [f32; 4],
}

#[derive(Clone, Copy, Debug)]
struct Spec {
    /// wind sea: peak angular frequency, alpha, gamma
    wp: f64,
    alpha: f64,
    gamma: f64,
    /// swell: peak angular frequency, alpha, gamma, spread
    swp: f64,
    salpha: f64,
    sgamma: f64,
    sbeta: f64,
    /// direction the wind and the swell run toward (east, north)
    dir: DVec2,
    sdir: DVec2,
    /// strength of the short-wave tail: the wind raises the wavelets
    tail: f64,
}

/// Complex unit Gaussians, n x n per cascade: the same numbers on GPU and CPU.
pub fn noise(n: usize) -> Vec<[f32; 2]> {
    let mut s: u64 = 0x9e37_79b9_7f4a_7c15;
    let mut next = || {
        s = s.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^= z >> 31;
        ((z >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    };
    (0..3 * n * n)
        .map(|_| {
            let (u, v) = (next(), next());
            let r = (-2.0 * u.ln()).sqrt() * std::f64::consts::FRAC_1_SQRT_2;
            let a = 2.0 * PI * v;
            [(r * a.cos()) as f32, (r * a.sin()) as f32]
        })
        .collect()
}

fn smoothstep(a: f64, b: f64, x: f64) -> f64 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn jonswap(w: f64, wp: f64, alpha: f64, gamma: f64) -> f64 {
    if w <= 0.0 {
        return 0.0;
    }
    let sig = if w <= wp { 0.07 } else { 0.09 };
    let d = (w - wp) / (sig * wp);
    let x = (wp / w).powi(4);
    alpha * GRAV * GRAV / w.powi(5) * (-1.25 * x).exp() * gamma.powf((-0.5 * d * d).exp())
}
fn spread(th: f64, beta: f64) -> f64 {
    let b = beta.max(0.05);
    let c = (b * th.abs()).min(40.0).cosh();
    b / (2.0 * (b * PI).tanh() * c * c)
}
fn db_beta(wr: f64) -> f64 {
    if wr < 0.95 {
        2.61 * wr.powf(1.3)
    } else if wr < 1.6 {
        2.28 * wr.powf(-1.3)
    } else {
        10f64.powf(-0.4 + 0.8393 * (-0.567 * (wr * wr).ln()).exp())
    }
}
fn angle(a: DVec2, b: DVec2) -> f64 {
    (a.x * b.y - a.y * b.x).atan2(a.dot(b))
}
/// JONSWAP's energy for its peak enhancement, relative to Pierson-Moskowitz.
fn gamma_gain(g: f64) -> f64 {
    1.0 / (1.0 - 0.287 * g.ln())
}

impl Spec {
    fn new(wind: f64, dir: DVec2, sdir: DVec2, swell_hs: f64) -> Self {
        let u = wind.max(0.5);
        // Fetch-limited (Hasselmann et al.), never past fully developed.
        let wp = (22.0 * (GRAV * GRAV / (u * FETCH)).cbrt()).max(0.855 * GRAV / u);
        let alpha = (0.076 * (u * u / (FETCH * GRAV)).powf(0.22)).max(0.0081);
        let swp = (GRAV * 2.0 * PI / SWELL_L).sqrt();
        let sgamma = 5.0;
        let m0 = (swell_hs / 4.0).powi(2);
        let salpha = 5.0 * swp.powi(4) * m0 / (GRAV * GRAV * gamma_gain(sgamma));
        let tail = (0.1 + u / 9.0).clamp(0.15, 1.3);
        Self { wp, alpha, gamma: 3.3, swp, salpha, sgamma, sbeta: 5.0, dir, sdir, tail }
    }
    /// Energy per unit kx, ky (m^4), as ocean.wgsl `energy`.
    fn energy(&self, kv: DVec2, k: f64) -> f64 {
        let w = (GRAV * k).sqrt();
        let kh = kv / k;
        let e = jonswap(w, self.wp, self.alpha, self.gamma) * spread(angle(self.dir, kh), db_beta(w / self.wp))
            + jonswap(w, self.swp, self.salpha, self.sgamma) * spread(angle(self.sdir, kh), self.sbeta);
        e * self.tail_at(k) * GRAV / (2.0 * w) / k
    }
    /// Omnidirectional energy per unit k.
    fn omni(&self, k: f64) -> f64 {
        let w = (GRAV * k).sqrt();
        (jonswap(w, self.wp, self.alpha, self.gamma) + jonswap(w, self.swp, self.salpha, self.sgamma)) * self.tail_at(k) * GRAV / (2.0 * w)
    }
    fn tail_at(&self, k: f64) -> f64 {
        1.0 + (self.tail - 1.0) * smoothstep(2.0, 20.0, k)
    }
}

/// Wavenumber band of cascade `c` (rad/m).
fn band(c: usize) -> (f64, f64) {
    let edge = |l: f64| 2.0 * PI / l;
    match c {
        0 => (0.0, edge(BAND[0])),
        1 => (edge(BAND[0]), edge(BAND[1])),
        _ => (edge(BAND[1]), 1e9),
    }
}

/// In-place inverse FFT (positive exponent, unnormalised) of a power-of-two run.
fn ifft(a: &mut [[f64; 2]]) {
    let n = a.len();
    let bits = n.trailing_zeros();
    for i in 0..n {
        let j = i.reverse_bits() >> (usize::BITS - bits);
        if j > i {
            a.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        let ang = 2.0 * PI / len as f64;
        for s in (0..n).step_by(len) {
            for k in 0..len / 2 {
                let (wc, ws) = ((ang * k as f64).cos(), (ang * k as f64).sin());
                let p = a[s + k];
                let q = a[s + k + len / 2];
                let q = [q[0] * wc - q[1] * ws, q[0] * ws + q[1] * wc];
                a[s + k] = [p[0] + q[0], p[1] + q[1]];
                a[s + k + len / 2] = [p[0] - q[0], p[1] - q[1]];
            }
        }
        len *= 2;
    }
}

/// The sea as the weather panel sets it (weather.rs `Sea`): the wind sea's
/// significant height (None: from the wind), the swell's height and the
/// compass direction it comes from (None: the usual one), how much of it
/// whitecaps (×), the water's clarity (×, 1 normal) and calm lakes.
#[derive(Clone, Copy, Debug)]
pub struct SeaSet {
    pub hs: Option<f64>,
    pub swell: Option<(f64, f64)>,
    pub foam: f64,
    pub clarity: f64,
    pub calm_lakes: bool,
}
impl Default for SeaSet {
    fn default() -> Self {
        Self { hs: None, swell: None, foam: 1.0, clarity: 1.0, calm_lakes: false }
    }
}

/// Zeroth moment (variance, m²) of the JONSWAP spectrum.
fn jonswap_m0(wp: f64, alpha: f64, gamma: f64) -> f64 {
    let (a, b) = ((0.3 * wp).ln(), (10.0 * wp).ln());
    let n = 400;
    (0..n).map(|i| {
        let w = (a + (b - a) * (i as f64 + 0.5) / n as f64).exp();
        jonswap(w, wp, alpha, gamma) * w * (b - a) / n as f64
    }).sum()
}

pub struct Ocean {
    /// the sea's settings (weather panel)
    pub set: SeaSet,
    /// texels across each cascade (`grid`)
    n: usize,
    /// smoothed wind (m/s) and the direction it blows toward (east, north)
    wind: f64,
    dir: DVec2,
    /// direction the swell runs toward: turns only slowly with the wind
    sdir: DVec2,
    started: bool,
    spec: Spec,
    /// game seconds since the last update (0 while paused)
    dt: f64,
    /// time (s), wrapped at PERIOD
    t: f64,
    /// the random numbers of the cascades kept on the CPU, one after the other
    noise0: Vec<[f32; 2]>,
    /// those cascades on an M x M grid each: Dx, Dy, height
    field: [Vec<[f32; 3]>; CPU_C],
    /// this frame's camera, its east and north, and the ocean's coordinates of it
    cam: DVec3,
    east: DVec3,
    north: DVec3,
    off: DVec2,
    /// significant wave height of the open ocean (m)
    pub hs: f64,
    /// Slope covariance (east-east, north-north, east-north) of the waves
    /// each cascade's mip level no longer holds (`lost_slopes`), and of the
    /// ripples too short for any cascade
    lost: [[[f32; 4]; MIPS as usize]; 3],
    ripple: [f32; 4],
    /// the wind, directions and swell `lost` was worked out for
    lost_for: [f64; 6],
}

impl Default for Ocean {
    fn default() -> Self {
        Self::new()
    }
}

impl Ocean {
    pub fn new() -> Self {
        let mut noise0 = noise(N);
        noise0.truncate(CPU_C * N * N);
        let dir = DVec2::new(1.0, 0.0);
        Self {
            set: SeaSet::default(),
            n: N,
            wind: 5.0,
            dir,
            sdir: dir,
            started: false,
            spec: Spec::new(5.0, dir, dir, 0.8),
            dt: 0.0,
            t: 0.0,
            noise0,
            field: std::array::from_fn(|_| vec![[0.0; 3]; M * M]),
            cam: DVec3::ZERO,
            east: DVec3::X,
            north: DVec3::Y,
            off: DVec2::ZERO,
            hs: 0.0,
            lost: [[[0.0; 4]; MIPS as usize]; 3],
            ripple: [0.0; 4],
            lost_for: [f64::NAN; 6],
        }
    }

    /// The water quality (0 Low, 1 High): the grid the GPU runs, which the
    /// CPU copy and the roughness follow (its random numbers come with it).
    pub fn set_quality(&mut self, quality: u8) {
        let n = grid(quality);
        if n != self.n {
            self.n = n;
            self.noise0 = noise(n);
            self.noise0.truncate(CPU_C * n * n);
            self.lost_for = [f64::NAN; 6];
        }
    }

    /// Advance to game time `time` with the weather's wind (m/s, compass
    /// direction it blows from); `sinlat` where the camera is; the camera
    /// frame the GPU draws this frame with, `off` the ocean's coordinates of
    /// the camera (wrapped at WRAP).
    #[allow(clippy::too_many_arguments)]
    pub fn update(&mut self, dt: f64, time: f64, wind: f64, wind_from: f64, sinlat: f64, cam: DVec3, east: DVec3, north: DVec3, off: DVec2) {
        let to = (wind_from + 180.0).to_radians();
        let target = DVec2::new(to.sin(), to.cos());
        // Waves set by hand: the wind that raises a sea that high (fully
        // developed, Pierson-Moskowitz: Hs = 0.21 U² / g) shapes it, and its
        // energy is then scaled to the height asked for.
        let wind = match self.set.hs {
            Some(h) => (h.max(0.0) * GRAV / 0.21).sqrt().clamp(0.5, 40.0),
            None => wind,
        };
        let swell_to = self.set.swell.map(|(_, from)| {
            let a = (from + 180.0).to_radians();
            DVec2::new(a.sin(), a.cos())
        });
        if !self.started {
            self.started = true;
            self.wind = wind;
            self.dir = target;
            self.sdir = swell_to.unwrap_or_else(|| DVec2::from_angle(0.6).rotate(target));
        } else {
            // The sea takes a while to answer the wind.
            let k = (dt / 20.0).clamp(0.0, 1.0);
            self.wind += (wind - self.wind) * k;
            self.dir = (self.dir + (target - self.dir) * k).try_normalize().unwrap_or(target);
            let ks = (dt / if swell_to.is_some() { 10.0 } else { 600.0 }).clamp(0.0, 1.0);
            let st = swell_to.unwrap_or_else(|| DVec2::from_angle(0.6).rotate(target));
            self.sdir = (self.sdir + (st - self.sdir) * ks).try_normalize().unwrap_or(st);
        }
        // The southern ocean and the far north run a bigger swell.
        let swell_hs = match self.set.swell {
            Some((h, _)) => h.max(0.0),
            None => 0.7 + 0.6 * smoothstep(0.55, 0.85, sinlat.abs()) + 0.02 * self.wind,
        };
        self.spec = Spec::new(self.wind, self.dir, self.sdir, swell_hs);
        if let Some(h) = self.set.hs {
            // (the smoothed wind's sea, brought to the height asked for)
            let want = (h.max(0.0) / 4.0).powi(2);
            let m0 = jonswap_m0(self.spec.wp, self.spec.alpha, self.spec.gamma).max(1e-9);
            self.spec.alpha *= want / m0;
        }
        self.t = time.rem_euclid(PERIOD);
        self.dt = dt.max(0.0);
        self.cam = cam;
        self.east = east;
        self.north = north;
        self.off = off;
        self.stats();
        // The spectrum drifts slowly (the sea answers the wind over tens of
        // seconds): redo the roughness only once it has moved.
        let key = [self.wind, self.dir.x, self.dir.y, self.sdir.x, self.sdir.y, swell_hs];
        let moved = key.iter().zip(&self.lost_for).enumerate().any(|(i, (a, b))| {
            let tol = if i == 0 || i == 5 { 0.03 * a.abs().max(1.0) } else { 0.01 };
            !((a - b).abs() <= tol)
        });
        if moved {
            self.lost_for = key;
            self.lost_slopes();
        }
        for c in 0..CPU_C {
            self.cpu_field(c);
        }
    }

    /// Significant wave height, integrated over the cascades' bands.
    fn stats(&mut self) {
        let mut m0 = 0.0;
        for c in 0..3 {
            let (a, b) = band(c);
            let (lo, hi) = (a.max(2.0 * PI / SIZES[c]), b.min((self.n / 2 - 1) as f64 * 2.0 * PI / SIZES[c]));
            let steps = 96;
            let (la, lb) = (lo.ln(), hi.ln());
            for i in 0..steps {
                let k = (la + (lb - la) * (i as f64 + 0.5) / steps as f64).exp();
                m0 += self.spec.omni(k) * k * (lb - la) / steps as f64;
            }
        }
        self.hs = 4.0 * m0.sqrt();
    }

    /// Roughness from the spectrum (Bruneton, Neyret & Holzschuch 2010, "Real-
    /// time Realistic Ocean Lighting using Seamless Transitions from Geometry to
    /// BRDF"): the slope variance of the waves a pixel can no longer resolve
    /// becomes the width of the BRDF, so near water (sharp waves, a fine
    /// glitter) and far water (a smooth sheet, a broad sun path) light alike.
    ///
    /// The shader reads each cascade's slopes at the mip that matches the
    /// pixel footprint. Mip m is the mean of 2^m x 2^m texels, read back
    /// bilinearly; a wave of phase step u (rad per texel, along each axis)
    /// keeps the share of its slope variance
    ///   D_m(u)^2 (2 + cos(2^m u)) / 3,  D_m(u) = sin(2^m u / 2) / (2^m sin(u / 2))
    /// (box average, then the mean square of a linear blend between two of
    /// its samples). What each level loses, summed over the cascade's modes as
    /// the FFT holds them, is its slope covariance, per cascade and level.
    /// The last level (one texel) keeps nothing, and the shader stops reading
    /// a cascade there. Ripples shorter than the finest cascade's grid are
    /// always lost. Levels count as the texture's: a 128 grid starts at level
    /// 1, and level 0 (never read then) repeats it.
    fn lost_slopes(&mut self) {
        const ML: usize = MIPS as usize;
        let l0 = (N / self.n).trailing_zeros() as usize;
        let keep = |u: f64, m: usize| {
            let s = (1u32 << m) as f64;
            let h = (0.5 * u).sin();
            let d = if m == 0 || h.abs() < 1e-9 { 1.0 } else { (0.5 * s * u).sin() / (s * h) };
            d * d * (2.0 + (s * u).cos()) / 3.0
        };
        let h = self.n as i64 / 2;
        // keep(u) for every grid step along an axis (the same in every
        // cascade), by the texture's level
        let kt: Vec<[f64; ML]> = (1 - h..h).map(|n| std::array::from_fn(|m| keep(2.0 * PI * n as f64 / self.n as f64, m.saturating_sub(l0)))).collect();
        for c in 0..3 {
            let dk = 2.0 * PI / SIZES[c];
            let (klo, khi) = band(c);
            let mut tot = [0.0f64; 3];
            let mut kept = [[0.0f64; 3]; ML];
            for ny in 1 - h..h {
                for nx in 1 - h..h {
                    // Every mode near the centre of the spectrum; further out,
                    // where it is smooth, every 2nd, 4th, 8th one standing in
                    // for its neighbours (the mips that would tell them apart
                    // hold almost none of them anyway).
                    let r = nx.abs().max(ny.abs()) << l0;
                    let st = ((if r <= 16 { 1 } else if r <= 32 { 2 } else if r <= 64 { 4 } else { 8 }) >> l0).max(1);
                    if nx % st != 0 || ny % st != 0 {
                        continue;
                    }
                    let kv = DVec2::new(nx as f64, ny as f64) * dk;
                    let k = kv.length();
                    if k <= 0.0 || k < klo || k >= khi {
                        continue;
                    }
                    // A mode's height variance is energy x dk^2 (h0(k) and
                    // h0(-k) together, see ocean.wgsl `spectrum`).
                    let e = self.spec.energy(kv, k) * dk * dk * (st * st) as f64;
                    let v = [kv.x * kv.x * e, kv.y * kv.y * e, kv.x * kv.y * e];
                    let (kx, ky) = (&kt[(nx + h - 1) as usize], &kt[(ny + h - 1) as usize]);
                    for i in 0..3 {
                        tot[i] += v[i];
                    }
                    for (m, kp) in kept.iter_mut().enumerate().take(ML - 1) {
                        let f = kx[m] * ky[m];
                        for i in 0..3 {
                            kp[i] += v[i] * f;
                        }
                    }
                }
            }
            for m in 0..ML {
                let l = if m == ML - 1 { [0.0; 3] } else { kept[m] };
                self.lost[c][m] = [(tot[0] - l[0]).max(0.0) as f32, (tot[1] - l[1]).max(0.0) as f32, (tot[2] - l[2]) as f32, 0.0];
            }
        }
        // Ripples: everything outside the finest grid, up to capillary waves of
        // a few millimetres.
        let edge = (h as f64 - 0.5) * 2.0 * PI / SIZES[2];
        let (la, lb) = (edge.ln(), 1200f64.ln());
        let (nk, nt) = (48, 48);
        let mut rp = [0.0f64; 3];
        for i in 0..nk {
            let k = (la + (lb - la) * (i as f64 + 0.5) / nk as f64).exp();
            let dk = k * (lb - la) / nk as f64;
            for j in 0..nt {
                let th = 2.0 * PI * (j as f64 + 0.5) / nt as f64;
                let kv = DVec2::new(th.cos(), th.sin()) * k;
                if kv.x.abs().max(kv.y.abs()) <= edge {
                    continue;
                }
                let e = self.spec.energy(kv, k) * k * dk * 2.0 * PI / nt as f64;
                rp[0] += kv.x * kv.x * e;
                rp[1] += kv.y * kv.y * e;
                rp[2] += kv.x * kv.y * e;
            }
        }
        self.ripple = [rp[0] as f32, rp[1] as f32, rp[2] as f32, 0.0];
    }

    /// Cascade `c`'s waves at this time, as ocean.wgsl `spectrum` has them
    /// (the modes the CPU grid holds): grid cell, wave vector and complex
    /// amplitude h0(k) e^(iwt) + conj(h0(-k)) e^(-iwt).
    fn modes(&self, c: usize) -> Vec<(usize, DVec2, [f64; 2])> {
        let l = SIZES[c];
        let dk = 2.0 * PI / l;
        let (klo, khi) = band(c);
        let base = c * self.n * self.n;
        let wq = 2.0 * PI / PERIOD;
        let h = M as i64 / 2;
        let mut out = Vec::new();
        for cy in 0..M {
            for cx in 0..M {
                let n = (cx as i64 - h, cy as i64 - h);
                let kv = DVec2::new(n.0 as f64, n.1 as f64) * dk;
                let k = kv.length();
                if k < klo || k >= khi || k <= 0.0 {
                    continue;
                }
                let (gx, gy) = ((n.0 + self.n as i64 / 2) as usize, (n.1 + self.n as i64 / 2) as usize);
                let a = dk * 0.5f64.sqrt();
                let np = self.noise0[base + gy * self.n + gx];
                let nm = self.noise0[base + ((self.n - gy) % self.n) * self.n + ((self.n - gx) % self.n)];
                let ap = self.spec.energy(kv, k).sqrt() * a;
                let am = self.spec.energy(-kv, k).sqrt() * a;
                let w = ((GRAV * k * (1.0 + k * k / 136900.0)).sqrt() / wq).floor() * wq;
                let (sn, cs) = (w * self.t).sin_cos();
                let hp = [np[0] as f64 * ap, np[1] as f64 * ap];
                let hm = [nm[0] as f64 * am, -(nm[1] as f64) * am];
                let hr = hp[0] * cs - hp[1] * sn + hm[0] * cs + hm[1] * sn;
                let hi = hp[0] * sn + hp[1] * cs - hm[0] * sn + hm[1] * cs;
                out.push((cy * M + cx, kv, [hr, hi]));
            }
        }
        out
    }

    /// Cascade `c` on the CPU grid, from the same spectrum and noise as the GPU.
    fn cpu_field(&mut self, c: usize) {
        let mut fd = vec![[0f64; 2]; M * M];
        let mut fh = vec![[0f64; 2]; M * M];
        for (i, kv, [hr, hi]) in self.modes(c) {
            // i k/|k| h for the displacement, packed as Dx + i Dy.
            let k = kv.length();
            let (ix, iy) = (kv.x / k, kv.y / k);
            let dx = [-hi * ix, hr * ix];
            let dy = [-hi * iy, hr * iy];
            fd[i] = [dx[0] - dy[1], dx[1] + dy[0]];
            fh[i] = [hr, hi];
        }
        for f in [&mut fd, &mut fh] {
            for r in f.chunks_mut(M) {
                ifft(r);
            }
            let mut col = vec![[0f64; 2]; M];
            for x in 0..M {
                for y in 0..M {
                    col[y] = f[y * M + x];
                }
                ifft(&mut col);
                for y in 0..M {
                    f[y * M + x] = col[y];
                }
            }
        }
        for y in 0..M {
            for x in 0..M {
                let sg = if (x + y) & 1 == 1 { -1.0 } else { 1.0 };
                let d = fd[y * M + x];
                self.field[c][y * M + x] = [(d[0] * sg * CHOP[c]) as f32, (d[1] * sg * CHOP[c]) as f32, (fh[y * M + x][0] * sg) as f32];
            }
        }
    }

    /// Cascade `c` at ocean coordinates `p` (m), bilinear, sampled as the
    /// GPU's texture is (its texel centres sit half a texel off the grid).
    fn sample(&self, c: usize, p: DVec2) -> [f64; 3] {
        let s = M as f64 / SIZES[c];
        let u = (p - DVec2::splat(SIZES[c] / (2.0 * self.n as f64))) * s;
        let (x0, y0) = (u.x.floor(), u.y.floor());
        let (fx, fy) = (u.x - x0, u.y - y0);
        let f = &self.field[c];
        let at = |i: f64, j: f64| f[(j.rem_euclid(M as f64) as usize % M) * M + (i.rem_euclid(M as f64) as usize % M)];
        let (a, b, c, d) = (at(x0, y0), at(x0 + 1.0, y0), at(x0, y0 + 1.0), at(x0 + 1.0, y0 + 1.0));
        let mut o = [0.0; 3];
        for i in 0..3 {
            o[i] = (a[i] as f64 * (1.0 - fx) + b[i] as f64 * fx) * (1.0 - fy) + (c[i] as f64 * (1.0 - fx) + d[i] as f64 * fx) * fy;
        }
        o
    }

    /// Height of the sea surface at a point, with `w` of each of the two
    /// longer cascades (common.wgsl `sea_state`): what the GPU draws there
    /// (terrain.wgsl `vs_water`, but for the 6 m chop): the surface point
    /// above it, found through the sideways displacement.
    pub fn height_w(&self, p: DVec3, w: [f64; 2]) -> f64 {
        if !self.started {
            return 0.0;
        }
        let x = self.coord(p);
        let disp = |x0: DVec2| {
            let mut o = [0.0; 3];
            for (c, wc) in w.iter().enumerate() {
                if *wc > 0.0 {
                    let d = self.sample(c, x0);
                    for i in 0..3 {
                        o[i] += d[i] * wc;
                    }
                }
            }
            o
        };
        let mut x0 = x;
        for _ in 0..5 {
            let d = disp(x0);
            x0 = x - DVec2::new(d[0], d[1]);
        }
        disp(x0)[2]
    }

    /// The ocean's coordinates (common.wgsl `ocean_coord`) of a point.
    fn coord(&self, p: DVec3) -> DVec2 {
        let v = p - self.cam;
        DVec2::new(v.dot(self.east), v.dot(self.north)) + self.off
    }

    /// Sideways shift and height (m) of the sea surface point that starts out
    /// above ocean coordinates of `p` (no search: the raw displacement field),
    /// with `w` of each cascade. For tests against the GPU's `ocean_disp`.
    #[cfg(test)]
    pub fn displacement(&self, p: DVec3, w: [f64; 2]) -> [f64; 3] {
        let x = self.coord(p);
        let mut o = [0.0; 3];
        for (c, wc) in w.iter().enumerate() {
            let d = self.sample(c, x);
            for i in 0..3 {
                o[i] += d[i] * wc;
            }
        }
        o
    }

    /// The GPU's constants for this frame.
    pub fn uniform(&self) -> OceanU {
        let s = &self.spec;
        let mut casc = [[0f32; 4]; 3];
        for (c, o) in casc.iter_mut().enumerate() {
            let (lo, hi) = band(c);
            *o = [SIZES[c] as f32, lo as f32, hi.min(1e6) as f32, CHOP[c] as f32];
        }
        OceanU {
            casc,
            sea: [self.wind as f32, s.wp as f32, s.alpha as f32, s.gamma as f32],
            swell: [s.swp as f32, s.salpha as f32, s.sgamma as f32, s.sbeta as f32],
            dir: [s.dir.x as f32, s.dir.y as f32, s.sdir.x as f32, s.sdir.y as f32],
            t: [self.t as f32, (2.0 * PI / PERIOD) as f32, s.tail as f32, self.dt as f32],
        }
    }

    /// Globals `ocean` (common.wgsl): offset and wind direction; wind,
    /// significant wave height, whitecap cover (what ocean.wgsl `foam_ctl`
    /// aims the foam at); the ripples' slope covariance; then, per cascade
    /// and mip level, the slope covariance that level loses (`lost_slopes`).
    pub fn globals(&self) -> [[f32; 4]; OCEAN_G] {
        let u = self.wind;
        // Whitecap cover (Monahan & O'Muircheartaigh 1980).
        let wc = ((3.84e-6 * u.max(0.0).powf(3.41)).min(0.3) * self.set.foam).min(0.45);
        let mut g = [[0f32; 4]; OCEAN_G];
        g[0] = [self.off.x as f32, self.off.y as f32, self.dir.x as f32, self.dir.y as f32];
        // (w: the water's clarity, negative with the lakes calm)
        g[1] = [u as f32, self.hs as f32, wc as f32, self.set.clarity.max(0.05) as f32 * if self.set.calm_lakes { -1.0 } else { 1.0 }];
        // (w: the finest level the GPU writes, which the shaders read no finer than)
        g[2] = [self.ripple[0], self.ripple[1], self.ripple[2], (N / self.n).trailing_zeros() as f32];
        for c in 0..3 {
            for m in 0..MIPS as usize {
                g[3 + c * MIPS as usize + m] = self.lost[c][m];
            }
        }
        g
    }
}

/// common.wgsl `sea_state`: how much of each cascade a stretch of water shows.
pub fn sea_state(wclass: f32, sinlat: f32, shore: f32) -> [f32; 3] {
    let ss = |a: f32, b: f32, x: f32| {
        let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
        t * t * (3.0 - 2.0 * t)
    };
    if wclass > 2.5 {
        [0.0, 0.1, 0.45]
    } else if wclass > 1.5 {
        [0.05, 0.45, 0.85]
    } else {
        let rough = 1.0 + 0.3 * ss(0.55, 0.85, sinlat.abs());
        [rough * (0.3 + 0.7 * shore), 0.75 + 0.25 * shore, 1.0]
    }
}

// ------------------------------------------------------------------ GPU

/// The FFT and whitecap passes for one grid size (ocean.wgsl `NN`).
struct GridPasses {
    n: usize,
    p_rows: wgpu::ComputePipeline,
    p_cols: wgpu::ComputePipeline,
    p_foam: wgpu::ComputePipeline,
    p_ctl: wgpu::ComputePipeline,
    bg: wgpu::BindGroup,
    bg_cols: wgpu::BindGroup,
    /// foam update, one per direction of the ping-pong between the two state buffers
    bg_foam: [wgpu::BindGroup; 2],
    bg_ctl: wgpu::BindGroup,
}

pub struct OceanGpu {
    ub: wgpu::Buffer,
    noise_buf: wgpu::Buffer,
    foam_state: [wgpu::Buffer; 2],
    p_mip: wgpu::ComputePipeline,
    bg_mips: Vec<wgpu::BindGroup>,
    /// Low (128) and High (256) water quality
    grids: [GridPasses; 2],
    quality: usize,
    flip: std::cell::Cell<usize>,
    /// the seven-layer texture the shaders sample (common.wgsl `t_ocean`):
    /// six of waves, then the foam
    pub view: wgpu::TextureView,
}

/// Texture layers: six of waves, one of foam.
const LAYERS: u32 = 7;

pub const MIPS: u32 = 9;
/// Rows of the globals' `ocean` (common.wgsl): three, then the lost slope
/// covariance of each cascade at each mip level.
pub const OCEAN_G: usize = 3 + 3 * MIPS as usize;

impl OceanGpu {
    pub fn new(d: &wgpu::Device, q: &wgpu::Queue) -> Self {
        let m = d.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("ocean"), source: wgpu::ShaderSource::Wgsl(include_str!("shaders/ocean.wgsl").into()) });
        let pipe = |e: &str, n: usize| {
            let k = [("NN", n as f64), ("LOGN", n.trailing_zeros() as f64)];
            d.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some(e), layout: None, module: &m, entry_point: Some(e), compilation_options: wgpu::PipelineCompilationOptions { constants: &k, ..Default::default() }, cache: None })
        };
        let p_mip = pipe("mip", N);
        let ub = d.create_buffer(&wgpu::BufferDescriptor { label: Some("ocean u"), size: std::mem::size_of::<OceanU>() as u64, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let noise_buf = d.create_buffer(&wgpu::BufferDescriptor { label: Some("ocean noise"), size: (3 * N * N * 8) as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        q.write_buffer(&noise_buf, 0, bytemuck::cast_slice(&noise(N)));
        let spec = d.create_buffer(&wgpu::BufferDescriptor { label: Some("ocean spectrum"), size: (3 * 4 * N * N * 8) as u64, usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false });
        let tex = d.create_texture(&wgpu::TextureDescriptor {
            label: Some("ocean"),
            size: wgpu::Extent3d { width: N as u32, height: N as u32, depth_or_array_layers: LAYERS },
            mip_level_count: MIPS,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let level = |m: u32| tex.create_view(&wgpu::TextureViewDescriptor { dimension: Some(wgpu::TextureViewDimension::D2Array), base_mip_level: m, mip_level_count: Some(1), ..Default::default() });
        let view = tex.create_view(&wgpu::TextureViewDescriptor { dimension: Some(wgpu::TextureViewDimension::D2Array), ..Default::default() });
        fn buf(b: u32, buf: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
            wgpu::BindGroupEntry { binding: b, resource: buf.as_entire_binding() }
        }
        use crate::render::tex_res as tv;
        let layers = |m: u32, base: u32, n: u32| tex.create_view(&wgpu::TextureViewDescriptor { dimension: Some(wgpu::TextureViewDimension::D2Array), base_mip_level: m, mip_level_count: Some(1), base_array_layer: base, array_layer_count: Some(n), ..Default::default() });
        let state = |l: &str| d.create_buffer(&wgpu::BufferDescriptor { label: Some(l), size: (N * N * 8) as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let foam_state = [state("foam a"), state("foam b")];
        // ocean.wgsl `Ctl`: histogram, sum, threshold, log share.
        let ctl = d.create_buffer(&wgpu::BufferDescriptor { label: Some("foam ctl"), size: 528, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        q.write_buffer(&ctl, 516, bytemuck::cast_slice(&[0.3f32, (0.003f32).ln()]));
        // A grid of n texels writes texture level log2(256 / n): the 128 grid
        // level 1, which then reads as the 256 one's would, a level coarser.
        let grid_passes = |n: usize| {
            let lv = (N / n).trailing_zeros();
            let (p_rows, p_cols, p_foam, p_ctl) = (pipe("rows", n), pipe("cols", n), pipe("foam", n), pipe("foam_ctl", n));
            let bg = d.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("ocean"), layout: &p_rows.get_bind_group_layout(0), entries: &[buf(0, &ub), buf(1, &noise_buf), buf(2, &spec)] });
            let bg_cols = d.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("ocean cols"), layout: &p_cols.get_bind_group_layout(0), entries: &[buf(0, &ub), buf(2, &spec), tv(3, &level(lv))] });
            // Foam: this frame's waves (layers 0-5) in, layer 6 out.
            let (waves, foam_out) = (layers(lv, 0, 6), layers(lv, 6, 1));
            let foam_bg = |src: &wgpu::Buffer, dst: &wgpu::Buffer| {
                d.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("ocean foam"), layout: &p_foam.get_bind_group_layout(0), entries: &[buf(0, &ub), tv(4, &foam_out), tv(5, &waves), buf(7, src), buf(8, dst), buf(9, &ctl)] })
            };
            let bg_foam = [foam_bg(&foam_state[0], &foam_state[1]), foam_bg(&foam_state[1], &foam_state[0])];
            let bg_ctl = d.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("foam ctl"), layout: &p_ctl.get_bind_group_layout(0), entries: &[buf(0, &ub), buf(9, &ctl)] });
            GridPasses { n, p_rows, p_cols, p_foam, p_ctl, bg, bg_cols, bg_foam, bg_ctl }
        };
        let grids = [grid_passes(grid(0)), grid_passes(grid(1))];
        let mut bg_mips = Vec::new();
        for m in 1..MIPS {
            let (src, dst) = (level(m - 1), level(m));
            bg_mips.push(d.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("ocean mip"), layout: &p_mip.get_bind_group_layout(0), entries: &[tv(5, &src), tv(6, &dst)] }));
        }
        Self { ub, noise_buf, foam_state, p_mip, bg_mips, grids, quality: 1, flip: std::cell::Cell::new(0), view }
    }

    /// The water quality (0 Low: 128 x 128, 1 High: 256 x 256), as
    /// `Ocean::set_quality`: the random numbers for its grid, and the
    /// whitecaps begun again on it.
    pub fn set_quality(&mut self, q: &wgpu::Queue, quality: u8) {
        let i = (quality > 0) as usize;
        if i == self.quality {
            return;
        }
        self.quality = i;
        let n = self.grids[i].n;
        q.write_buffer(&self.noise_buf, 0, bytemuck::cast_slice(&noise(n)));
        let zero = vec![0u8; N * N * 8];
        for b in &self.foam_state {
            q.write_buffer(b, 0, &zero);
        }
    }

    /// Rebuild the waves for this frame.
    pub fn compute(&self, q: &wgpu::Queue, enc: &mut wgpu::CommandEncoder, u: &OceanU, prof: Option<&crate::gpuprof::GpuProf>) {
        q.write_buffer(&self.ub, 0, bytemuck::bytes_of(u));
        let g = &self.grids[self.quality];
        let n = g.n as u32;
        let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("ocean"), timestamp_writes: prof.and_then(|p| p.compute("ocean")) });
        cp.set_pipeline(&g.p_rows);
        cp.set_bind_group(0, &g.bg, &[]);
        cp.dispatch_workgroups(n, 3, 1);
        cp.set_pipeline(&g.p_cols);
        cp.set_bind_group(0, &g.bg_cols, &[]);
        cp.dispatch_workgroups(n, 3, 1);
        // Whitecaps from this frame's folds, then the threshold for the next.
        let f = self.flip.get();
        self.flip.set(f ^ 1);
        cp.set_pipeline(&g.p_foam);
        cp.set_bind_group(0, &g.bg_foam[f], &[]);
        cp.dispatch_workgroups(n / 16, n / 16, 1);
        cp.set_pipeline(&g.p_ctl);
        cp.set_bind_group(0, &g.bg_ctl, &[]);
        cp.dispatch_workgroups(1, 1, 1);
        // The mips below the level the grid wrote.
        cp.set_pipeline(&self.p_mip);
        for m in (N as u32 / n).trailing_zeros() + 1..MIPS {
            let s = ((N as u32) >> m).max(1);
            cp.set_bind_group(0, &self.bg_mips[m as usize - 1], &[]);
            cp.dispatch_workgroups(s.div_ceil(8), s.div_ceil(8), LAYERS);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The CPU copy of the long waves carries the energy its band should, and
    /// the sea grows with the wind.
    #[test]
    fn wave_heights() {
        let mut last = 0.0;
        for wind in [2.0, 7.0, 14.0] {
            let mut o = Ocean::new();
            o.update(0.0, 10.0, wind, 270.0, 0.6, DVec3::ZERO, DVec3::X, DVec3::Y, DVec2::ZERO);
            // Variance the band should hold, against what the grid holds.
            let (lo, hi) = band(0);
            let (lo, hi) = (lo.max(2.0 * PI / SIZES[0]), hi);
            let mut m0 = 0.0;
            let steps = 400;
            for i in 0..steps {
                let k = lo + (hi - lo) * (i as f64 + 0.5) / steps as f64;
                m0 += o.spec.omni(k) * (hi - lo) / steps as f64;
            }
            let var = o.field[0].iter().map(|f| (f[2] as f64).powi(2)).sum::<f64>() / o.field[0].len() as f64;
            let hmax = o.field[0].iter().map(|f| f[2].abs()).fold(0.0, f32::max);
            eprintln!("wind {wind}: Hs {:.2} m, long-wave rms {:.2} m (band {:.2}), max {hmax:.2}", o.hs, var.sqrt(), m0.sqrt());
            assert!(o.hs > last);
            assert!(var > m0 * 0.3 && var < m0 * 3.0);
            last = o.hs;
            // Floating: the height found through the displacement stays near the field's.
            let h = o.height_w(DVec3::new(10.0, 20.0, 0.0), [1.0, 0.0]);
            assert!(h.abs() < 4.0 * m0.sqrt() + 0.5);
        }
    }

    /// The CPU copy of the two longer cascades is the sum of their waves
    /// (as the GPU's FFT is), and the height found through the sideways
    /// shift is that of the surface point carried there.
    #[test]
    fn cpu_field_matches_waves() {
        let mut o = Ocean::new();
        o.update(0.0, 37.0, 12.0, 250.0, 0.6, DVec3::ZERO, DVec3::X, DVec3::Y, DVec2::ZERO);
        for c in 0..CPU_C {
            let modes = o.modes(c);
            let rms = (o.field[c].iter().map(|f| (f[2] as f64).powi(2)).sum::<f64>() / (M * M) as f64).sqrt();
            assert!(rms > 0.01, "cascade {c} empty");
            for (x, y) in [(0, 0), (5, 17), (33, 2), (63, 41)] {
                let p = DVec2::new(x as f64, y as f64) * (SIZES[c] / M as f64);
                let (mut h, mut dx) = (0.0, 0.0);
                for (_, kv, [hr, hi]) in &modes {
                    let (sn, cs) = kv.dot(p).sin_cos();
                    h += hr * cs - hi * sn;
                    // i k/|k| h, real part
                    dx += -(hr * sn + hi * cs) * kv.x / kv.length();
                }
                let f = o.field[c][y * M + x];
                assert!((f[2] as f64 - h).abs() < 1e-3 * rms.max(1.0), "cascade {c} ({x}, {y}): grid {} sum {h}", f[2]);
                assert!((f[0] as f64 - dx * CHOP[c]).abs() < 1e-3 * rms.max(1.0), "cascade {c} ({x}, {y}): dx {} sum {}", f[0], dx * CHOP[c]);
            }
        }
        // Floating: wherever a surface point is carried, height_w finds it.
        let w = [1.0, 0.8];
        for (x, y) in [(3.0, 4.0), (-71.5, 12.25), (140.0, -33.0)] {
            let p0 = DVec3::new(x, y, 0.0);
            let d = o.displacement(p0, w);
            let p = p0 + DVec3::new(d[0], d[1], 0.0);
            let h = o.height_w(p, w);
            assert!((h - d[2]).abs() < 0.05, "at {p0}: {h} vs {}", d[2]);
        }
    }

    /// The roughness the mips hand to the BRDF grows level by level up to
    /// the cascade's whole slope variance, matches the spectrum's, and the
    /// total stays near Cox & Munk's measured mean square slope.
    #[test]
    fn lost_slopes() {
        for wind in [2.0, 7.0, 14.0] {
            let mut o = Ocean::new();
            o.update(0.0, 10.0, wind, 270.0, 0.6, DVec3::ZERO, DVec3::X, DVec3::Y, DVec2::ZERO);
            let t0 = std::time::Instant::now();
            o.lost_slopes();
            let dt = t0.elapsed();
            let mut total = (o.ripple[0] + o.ripple[1]) as f64;
            for c in 0..3 {
                let l = &o.lost[c];
                for m in 1..MIPS as usize {
                    assert!(l[m][0] + l[m][1] >= (l[m - 1][0] + l[m - 1][1]) * 0.999, "cascade {c} level {m}: {:?}", l);
                }
                // Against the omnidirectional spectrum over the cascade's band.
                let (a, b) = band(c);
                let (lo, hi) = (a.max(2.0 * PI / SIZES[c]), b.min(127.5 * 2.0 * PI / SIZES[c]));
                let mut mss = 0.0;
                for i in 0..2000 {
                    let k = lo + (hi - lo) * (i as f64 + 0.5) / 2000.0;
                    mss += o.spec.omni(k) * k * k * (hi - lo) / 2000.0;
                }
                let all = (l[8][0] + l[8][1]) as f64;
                eprintln!("wind {wind} cascade {c}: lost {:?} / all {all:.5} (spectrum {mss:.5})", l.iter().map(|v| format!("{:.5}", v[0] + v[1])).collect::<Vec<_>>());
                assert!(all > mss * 0.6 && all < mss * 1.6);
                total += all;
            }
            let cox = 0.003 + 0.00512 * wind;
            eprintln!("wind {wind}: ripples {:?}, total mss {total:.4} (Cox-Munk {cox:.4}), {dt:?}", o.ripple);
            assert!(total > cox * 0.3 && total < cox * 3.0);
        }
    }
}
