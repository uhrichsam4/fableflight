//! Procedural sound, synthesised live in stereo (the one exception: the
//! fighter's engines, which are recordings, see `Fighter`).
//!
//! The aircraft: a four-cylinder engine (each cylinder's exhaust pulse rung
//! through the pipe's resonances, so it lopes at idle and roars at full
//! power, misfires and knocks when damaged, and pops when the throttle is
//! chopped), the propeller's blade-pass buzz that turns raspy as its tips
//! near the speed of sound, wind, gear and flap motors, tyres, airframe
//! creaks and groans when overstressed, turbulence thumps and rattles, stall
//! buffet and flutter. Its sounds fade and dull with the camera's distance;
//! from the cockpit they are heard through the cabin: the engine as a boom
//! through the firewall, the wind as a muffled roar with a whistle at the
//! door seals, creaks and rattles close by, in a small boxy room.
//!
//! The cockpit warnings: stall horn, terrain / pull-up, nose-dive, too-low
//! and sink-rate beepers, the overspeed clacker and the gear-up horn.
//!
//! And the world: explosions and fire.
use parking_lot::Mutex;
use std::f32::consts::{FRAC_PI_2, PI, TAU};
use std::num::NonZero;
use std::sync::{Arc, OnceLock};
use std::time::Duration;

/// The cockpit warnings that are sounding.
#[derive(Clone, Copy, Default, Debug, PartialEq)]
pub struct Warn {
    /// stall horn, 0..1 as the wing nears the stall
    pub stall: f32,
    /// terrain dead ahead, or the ground coming up fast
    pub pull_up: bool,
    /// nose well down and going down fast, 0..1 by how steep
    pub dive: f32,
    /// low with the gear up and descending, or terrain a little way ahead
    pub too_low: bool,
    /// descending fast near the ground
    pub sink: bool,
    /// past the never-exceed speed, 0..1
    pub overspeed: f32,
    /// throttle back near the ground with the gear up
    pub gear: bool,
}

/// What the sound is made from, set once a frame.
#[derive(Clone, Copy, Default, Debug)]
pub struct Params {
    /// 1 while flying; 0 paused, in the menu or crashed (the aircraft falls silent)
    pub live: f32,
    /// overall volume (0 = muted)
    pub master: f32,
    /// engine revolutions per minute
    pub rpm: f32,
    /// power being made, 0..1 of rated, and where the throttle is
    pub power: f32,
    pub throttle: f32,
    /// engine and propeller condition, 0 dead .. 1 sound
    pub health: f32,
    pub prop: f32,
    /// m/s through the air, and the speed where the airframe starts to howl (0: a light single's)
    pub airspeed: f32,
    pub vne: f32,
    /// the engine is a turbofan (1) rather than a piston single (0), and its fan speed (fraction of N1)
    pub jet: f32,
    pub n1: f32,
    /// the jet is a fighter's pair of afterburning engines (1), played from recordings; how far the afterburners are lit, 0..1
    pub fighter: f32,
    pub burner: f32,
    pub g_load: f32,
    /// RMS vertical turbulence, m/s
    pub rough: f32,
    /// sudden change in load just now (g), for the airframe shaking
    pub jolt: f32,
    /// stall buffet 0..1, control-surface flutter 0..
    pub buffet: f32,
    pub flutter: f32,
    /// gear 0 up .. 1 down; whether the gear / flap motors are running
    pub gear: f32,
    pub gear_moving: f32,
    pub flaps: f32,
    pub flaps_moving: f32,
    /// speed over the ground with the wheels on it, m/s; wheel brakes on
    pub rolling: f32,
    pub braking: f32,
    /// metal grinding along the ground 0..1, foliage brushing past 0..1
    pub scrape: f32,
    pub foliage: f32,
    /// windshield cracked (wind whistles in) 0..1
    pub cracked: f32,
    /// 1 in the cockpit, 0 outside
    pub cabin: f32,
    /// camera's distance from the aircraft, m
    pub dist: f32,
    /// fire burning near the camera 0..1
    pub fire: f32,
    pub warn: Warn,
}

impl Params {
    fn vne_or(&self) -> f32 {
        if self.vne > 0.0 { self.vne } else { 83.9 }
    }
}

/// One-off sounds.
#[derive(Clone, Copy, Debug)]
pub enum Event {
    /// an explosion (power ~0.2 .. 2)
    Boom(f32),
    /// something on the airframe breaking off
    Snap(f32),
    /// a turbulence jolt through the airframe (strength in g)
    Bump(f32),
    /// tyres meeting the runway (sink rate, m/s)
    Touch(f32),
    /// gear locking up or down
    Clunk(f32),
    /// ploughing through a tree
    Crunch(f32),
    /// something big going into the water (~0.2 .. 2)
    Splash(f32),
    /// the Destruction mode's lasers firing (~1)
    Laser(f32),
}

#[derive(Default)]
struct Shared {
    params: Mutex<Params>,
    events: Mutex<Vec<Event>>,
}

pub struct Audio {
    shared: Arc<Shared>,
    _sink: Option<rodio::MixerDeviceSink>,
    /// a synth with no sound device, pulled by `take` (for recording videos)
    offline: Option<Mutex<Synth>>,
}

impl Audio {
    pub fn new() -> Option<Self> {
        let sink = match rodio::DeviceSinkBuilder::open_default_sink() {
            Ok(s) => s,
            Err(e) => {
                log::warn!("no sound: {e}");
                return None;
            }
        };
        let shared = Arc::new(Shared::default());
        sink.mixer().add(Synth::new(shared.clone()));
        Some(Self { shared, _sink: Some(sink), offline: None })
    }
    /// No sound device: samples are generated on demand with `take`.
    pub fn offline() -> Self {
        let shared = Arc::new(Shared::default());
        let synth = Synth::new(shared.clone());
        Self { shared, _sink: None, offline: Some(Mutex::new(synth)) }
    }
    pub const RATE: u32 = RATE;
    /// The next `n` stereo samples of an offline synth (empty with a device).
    pub fn take(&self, n: usize) -> Vec<[f32; 2]> {
        let Some(s) = &self.offline else { return Vec::new() };
        let mut s = s.lock();
        (0..n).map(|_| [s.next().unwrap_or(0.0), s.next().unwrap_or(0.0)]).collect()
    }
    pub fn set(&self, p: Params) {
        *self.shared.params.lock() = p;
    }
    pub fn event(&self, e: Event) {
        self.shared.events.lock().push(e);
    }
    pub fn boom(&self, power: f32) {
        self.event(Event::Boom(power));
    }
}

/// Never-exceed speed, m/s.
const RATE: u32 = 44100;
const DT: f32 = 1.0 / RATE as f32;
/// Parameters are smoothed and filters retuned every this many samples.
const TICK: u32 = 64;
const TICK_DT: f32 = TICK as f32 * DT;
const MAX_VOICES: usize = 32;

fn smooth(a: f32, b: f32, x: f32) -> f32 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}
/// One-pole coefficient for a cutoff (Hz).
fn kf(f: f32) -> f32 {
    1.0 - (-TAU * f * DT).exp()
}
/// Per-tick smoothing coefficient for a time constant (s).
fn kt(tau: f32) -> f32 {
    1.0 - (-TICK_DT / tau).exp()
}
/// Per-sample decay factor for a time constant (s).
fn decay(tau: f32) -> f32 {
    (-DT / tau).exp()
}
fn pan(y: f32, p: f32) -> (f32, f32) {
    (y * ((1.0 - p) * 0.5).sqrt(), y * ((1.0 + p) * 0.5).sqrt())
}

struct Rng(u32);
impl Rng {
    /// -1..1
    fn f(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        (self.0 as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
    /// 0..1
    fn u(&mut self) -> f32 {
        self.f() * 0.5 + 0.5
    }
    fn range(&mut self, a: f32, b: f32) -> f32 {
        a + (b - a) * self.u()
    }
}

/// A biquad filter (RBJ cookbook), transposed direct form II.
#[derive(Clone, Copy, Default)]
struct Bq {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    z1: f32,
    z2: f32,
}

impl Bq {
    fn coefs(&mut self, b: [f32; 3], a: [f32; 3]) {
        let n = 1.0 / a[0];
        (self.b0, self.b1, self.b2, self.a1, self.a2) = (b[0] * n, b[1] * n, b[2] * n, a[1] * n, a[2] * n);
    }
    fn w(f: f32, q: f32) -> (f32, f32) {
        let w = TAU * f.clamp(5.0, RATE as f32 * 0.45) * DT;
        let (s, c) = w.sin_cos();
        (s / (2.0 * q.max(0.05)), c)
    }
    fn lp(&mut self, f: f32, q: f32) {
        let (al, c) = Self::w(f, q);
        self.coefs([(1.0 - c) * 0.5, 1.0 - c, (1.0 - c) * 0.5], [1.0 + al, -2.0 * c, 1.0 - al]);
    }
    fn hp(&mut self, f: f32, q: f32) {
        let (al, c) = Self::w(f, q);
        self.coefs([(1.0 + c) * 0.5, -(1.0 + c), (1.0 + c) * 0.5], [1.0 + al, -2.0 * c, 1.0 - al]);
    }
    /// Band-pass with unity gain at the centre.
    fn bp(&mut self, f: f32, q: f32) {
        let (al, c) = Self::w(f, q);
        self.coefs([al, 0.0, -al], [1.0 + al, -2.0 * c, 1.0 - al]);
    }
    /// Band-pass scaled so a unit impulse rings at about unit amplitude.
    fn ring(&mut self, f: f32, q: f32) {
        let (al, c) = Self::w(f, q);
        self.coefs([0.5, 0.0, -0.5], [1.0 + al, -2.0 * c, 1.0 - al]);
    }
    fn peak(&mut self, f: f32, q: f32, db: f32) {
        let (al, c) = Self::w(f, q);
        let a = 10f32.powf(db / 40.0);
        self.coefs([1.0 + al * a, -2.0 * c, 1.0 - al * a], [1.0 + al / a, -2.0 * c, 1.0 - al / a]);
    }
    fn new(kind: fn(&mut Self, f32, f32), f: f32, q: f32) -> Self {
        let mut b = Self::default();
        kind(&mut b, f, q);
        b
    }
    #[inline]
    fn run(&mut self, x: f32) -> f32 {
        let y = self.b0 * x + self.z1;
        self.z1 = self.b1 * x - self.a1 * y + self.z2;
        self.z2 = self.b2 * x - self.a2 * y;
        y
    }
}

/// Smoothed noise for slow random modulation, roughly unit RMS.
#[derive(Default)]
struct Slow {
    a: f32,
    b: f32,
}
impl Slow {
    fn run(&mut self, n: f32, k: f32) -> f32 {
        self.a += (n - self.a) * k;
        self.b += (self.a - self.b) * k;
        self.b * (1.7 / k.sqrt()).min(60.0)
    }
}

/// Which mix a one-shot sound goes into.
#[derive(Clone, Copy, PartialEq)]
enum Bus {
    /// airframe-borne: heard loud inside, faintly outside
    Struct,
    /// the aircraft's outside noises: full outside, dulled inside
    Mid,
}

/// A one-shot sound: a gliding tone, gliding filtered noise, and pulses
/// (periodic or random) rung through two resonators. Covers thumps,
/// clunks, tyre chirps, creaks, groans, snapping metal and splintering wood.
#[derive(Clone, Copy)]
struct Voice {
    t: f32,
    dur: f32,
    pan: f32,
    bus: Bus,
    tf: [f32; 2],
    tamp: f32,
    tdec: f32,
    tcur: f32,
    ph: f32,
    nf: [f32; 2],
    nq: f32,
    namp: f32,
    ndec: f32,
    nbq: Bq,
    pr: [f32; 2],
    prand: bool,
    pamp: f32,
    pcur: f32,
    pph: f32,
    res: [Bq; 2],
    rg: [f32; 2],
    strike: f32,
}

impl Voice {
    fn new(dur: f32, pan: f32, bus: Bus) -> Self {
        Self { t: 0.0, dur, pan, bus, tf: [50.0; 2], tamp: 0.0, tdec: 0.1, tcur: 50.0, ph: 0.0, nf: [1000.0; 2], nq: 1.0, namp: 0.0, ndec: 0.1, nbq: Bq::default(), pr: [0.0; 2], prand: false, pamp: 0.0, pcur: 0.0, pph: 0.0, res: [Bq::default(); 2], rg: [0.0; 2], strike: 0.0 }
    }
    fn tone(mut self, f0: f32, f1: f32, amp: f32, dec: f32) -> Self {
        (self.tf, self.tamp, self.tdec, self.tcur) = ([f0, f1], amp, dec, f0);
        self
    }
    fn noise(mut self, f0: f32, f1: f32, q: f32, amp: f32, dec: f32) -> Self {
        (self.nf, self.nq, self.namp, self.ndec) = ([f0, f1], q, amp, dec);
        self.nbq.bp(f0, q);
        self
    }
    fn pulses(mut self, r0: f32, r1: f32, random: bool, amp: f32) -> Self {
        (self.pr, self.prand, self.pamp, self.pcur) = ([r0, r1], random, amp, r0);
        self
    }
    fn res(mut self, f: [f32; 2], q: [f32; 2], g: [f32; 2]) -> Self {
        self.res[0].ring(f[0], q[0]);
        self.res[1].ring(f[1], q[1]);
        self.rg = g;
        self
    }
    fn strike(mut self, s: f32) -> Self {
        self.strike = s;
        self
    }
    fn delay(mut self, d: f32) -> Self {
        self.t = -d;
        self
    }
    fn done(&self) -> bool {
        self.t > self.dur
    }
    fn control(&mut self) {
        if self.t < 0.0 {
            return;
        }
        let u = (self.t / self.dur).min(1.0);
        let glide = |f: [f32; 2]| f[0] * (f[1] / f[0]).powf(u);
        self.tcur = glide(self.tf);
        if self.namp > 0.0 {
            self.nbq.bp(glide(self.nf), self.nq);
        }
        self.pcur = self.pr[0] + (self.pr[1] - self.pr[0]) * u;
    }
    fn sample(&mut self, rng: &mut Rng) -> f32 {
        let t = self.t;
        self.t += DT;
        if t < 0.0 {
            return 0.0;
        }
        let att = (t / 0.0015).min(1.0);
        let mut y = 0.0;
        if self.tamp > 0.0 {
            self.ph = (self.ph + self.tcur * DT).fract();
            y += (TAU * self.ph).sin() * self.tamp * att * (-t / self.tdec).exp();
        }
        if self.namp > 0.0 {
            y += self.nbq.run(rng.f()) * self.namp * att * (-t / self.ndec).exp();
        }
        if self.rg[0] > 0.0 {
            let mut x = if t < DT { self.strike } else { 0.0 };
            if self.pamp > 0.0 && t < self.dur {
                let fire = if self.prand {
                    rng.u() < self.pcur * DT
                } else {
                    self.pph += self.pcur * DT;
                    let f = self.pph >= 1.0;
                    self.pph = self.pph.fract();
                    f
                };
                if fire {
                    x += self.pamp * (PI * t / self.dur).sin() * rng.range(0.5, 1.0);
                }
            }
            y += self.res[0].run(x) * self.rg[0] + self.res[1].run(x) * self.rg[1];
        }
        y
    }
}

/// Firing strength of each cylinder (no two are quite alike: the lope at idle).
const CYL: [f32; 4] = [1.0, 0.84, 1.1, 0.93];

struct Engine {
    /// 0..1 through the four-stroke cycle (two turns of the crank)
    cyc: f32,
    pulse: f32,
    res: [Bq; 4],
    body: Bq,
    hp: Bq,
    tick: f32,
    mech: Bq,
    knock: Bq,
    acc_ph: f32,
    prop_ph: f32,
    prop_gain: [f32; 7],
    prop_bright: f32,
    prop_nz: Bq,
    wob_ph: f32,
    pulse_k: f32,
    tick_k: f32,
}

impl Engine {
    fn new() -> Self {
        Self {
            cyc: 0.0,
            pulse: 0.0,
            // The exhaust stack's resonances, and the bark of the open pipe.
            res: [Bq::new(Bq::bp, 95.0, 1.4), Bq::new(Bq::bp, 230.0, 2.2), Bq::new(Bq::bp, 520.0, 3.0), Bq::new(Bq::bp, 1350.0, 2.0)],
            body: Bq::new(Bq::lp, 2400.0, 0.7),
            hp: Bq::new(Bq::hp, 28.0, 0.7),
            tick: 0.0,
            mech: Bq::new(Bq::bp, 3200.0, 1.3),
            knock: Bq::new(Bq::ring, 1750.0, 9.0),
            acc_ph: 0.0,
            prop_ph: 0.0,
            prop_gain: [0.0; 7],
            prop_bright: 0.0,
            prop_nz: Bq::new(Bq::bp, 1100.0, 0.7),
            wob_ph: 0.0,
            pulse_k: decay(0.0018),
            tick_k: decay(0.004),
        }
    }
    fn control(&mut self, p: &Params) {
        // Tip speed: 0.95 m radius. Near Mach 0.8 the blades rasp.
        let mach = p.rpm * TAU / 60.0 * 0.95 / 340.0;
        self.prop_bright = smooth(0.42, 0.8, mach);
        let ex = 2.1 - 1.3 * self.prop_bright;
        for (h, g) in self.prop_gain.iter_mut().enumerate() {
            *g = ((h + 1) as f32).powf(-ex);
        }
    }
    /// Exhaust, propeller, mechanical.
    fn sample(&mut self, p: &Params, rng: &mut Rng) -> (f32, f32, f32) {
        let crank = p.rpm / 60.0;
        let before = (self.cyc * 4.0) as u32;
        self.cyc += crank * 0.5 * DT;
        if self.cyc >= 1.0 {
            self.cyc -= 1.0;
        }
        let after = (self.cyc * 4.0) as u32;
        let firing = p.health > 0.0;
        let mut kin = 0.0;
        if after != before && crank > 1.0 {
            let k = after as usize & 3;
            // A windmilling engine still pushes air through: faint compression chuffs.
            let load = if firing { 0.3 + 0.7 * p.power } else { 0.1 };
            let miss = if firing { 0.003 + 0.35 * (1.0 - p.health) } else { 0.0 };
            // Fewer, fuller pulses at low revs: the idle's distinct thud.
            let mut a = load * CYL[k] * (1.0 + 0.1 * rng.f()) * (1.0 + 1.5 * (1.0 - crank / 40.0).max(0.0));
            if rng.u() < miss {
                a *= 0.12;
            }
            // Throttle chopped at high revs: unburnt mixture pops in the exhaust.
            let decel = (p.power - p.throttle).max(0.0);
            if firing && decel > 0.12 && crank > 25.0 && rng.u() < decel * 0.3 {
                a *= rng.range(2.5, 4.5);
            }
            self.pulse += a;
            self.tick = 1.0;
            if firing && rng.u() < (1.0 - p.health) * 0.6 {
                kin = rng.range(0.5, 1.0);
            }
        }
        let x = self.pulse * rng.range(0.6, 1.4);
        self.pulse *= self.pulse_k;
        let ex = self.res[0].run(x) * 1.0 + self.res[1].run(x) * 0.75 + self.res[2].run(x) * 0.5 + self.res[3].run(x) * 0.3 * (0.4 + p.power) + self.body.run(x) * 0.35;
        let ex = self.hp.run(ex) * 0.55;

        // Valve train clatter and the accessory drive's whine.
        self.tick *= self.tick_k;
        let rev = (crank / 45.0).min(1.3);
        let mut mech = self.mech.run(rng.f()) * (0.2 + self.tick) * rev * 0.1;
        self.acc_ph = (self.acc_ph + crank * 16.0 * DT).fract();
        mech += (TAU * self.acc_ph).sin() * 0.01 * rev;
        mech += self.knock.run(kin) * 0.2;

        // Propeller: two blades, so its tone sits at twice the crank speed.
        self.prop_ph = (self.prop_ph + crank * 2.0 * DT).fract();
        let mut tone = 0.0;
        for (h, g) in self.prop_gain.iter().enumerate() {
            let h = (h + 1) as f32;
            tone += g * (TAU * h * self.prop_ph + h * 1.3).sin();
        }
        let blade = 0.5 + 0.5 * (TAU * self.prop_ph).cos();
        let whop = self.prop_nz.run(rng.f()) * blade * blade;
        let load = if firing { 0.3 + 0.7 * p.power } else { 0.15 };
        let level = (crank / 40.0).powi(2).min(1.6) * load;
        // An out-of-balance propeller beats once a turn.
        self.wob_ph = (self.wob_ph + crank * DT).fract();
        let wob = 1.0 + (1.0 - p.prop).clamp(0.0, 1.0) * 1.4 * (TAU * self.wob_ph).sin();
        let prop = (tone * 0.1 + whop * (0.28 + 0.6 * self.prop_bright)) * level * wob;
        (ex, prop, mech)
    }
}

/// A turbofan (all four together): the fan's tone at its blade-passing
/// frequency (38 blades), the buzz-saw of shock waves off the blade tips near
/// full power (a rasping chord on the shaft's harmonics), the exhaust's roar,
/// and the core's high whine.
struct Jet {
    shaft_ph: f32,
    core_ph: f32,
    roar: [Bq; 2],
    hiss: Bq,
    saw: [f32; 20],
    saw_gain: f32,
}

impl Jet {
    fn new() -> Self {
        // Each harmonic of the buzz-saw starts at its own phase (no two blades are alike).
        let mut saw = [0.0; 20];
        for (h, s) in saw.iter_mut().enumerate() {
            *s = (h as f32 * 2.399).fract() * TAU;
        }
        Self { shaft_ph: 0.0, core_ph: 0.0, roar: [Bq::new(Bq::lp, 240.0, 0.7), Bq::new(Bq::bp, 650.0, 0.7)], hiss: Bq::new(Bq::bp, 3000.0, 0.9), saw, saw_gain: 0.0 }
    }
    fn control(&mut self, p: &Params) {
        self.saw_gain = smooth(0.80, 0.97, p.n1);
        self.hiss.bp(1200.0 + 3400.0 * p.n1, 0.9);
    }
    fn sample(&mut self, p: &Params, rng: &mut Rng) -> f32 {
        let n1 = p.n1.max(0.0);
        // 3600 rpm at 100% N1.
        let shaft = n1 * 60.0;
        self.shaft_ph = (self.shaft_ph + shaft * DT).fract();
        let fan = TAU * (self.shaft_ph * 38.0).fract();
        let tone = fan.sin() * 0.6 + (2.0 * fan + 0.7).sin() * 0.25;
        self.core_ph = (self.core_ph + (4200.0 + 3600.0 * n1) * DT).fract();
        let whine = (TAU * self.core_ph).sin();
        let mut saw = 0.0;
        if self.saw_gain > 0.0 {
            for (h, ph) in self.saw.iter().enumerate() {
                let k = (h + 4) as f32;
                saw += (TAU * (self.shaft_ph * k).fract() + ph).sin() / k.sqrt();
            }
        }
        let nz = rng.f();
        let thrust = p.power.max(0.0);
        let roar = self.roar[0].run(nz) * 1.3 + self.roar[1].run(nz) * 0.8;
        let hiss = self.hiss.run(nz);
        let run = if p.health > 0.0 { 1.0 } else { 0.25 };
        (tone * (0.03 + 0.10 * n1) + whine * 0.012 * n1 + saw * 0.018 * self.saw_gain + hiss * (0.05 + 0.25 * n1) + roar * (0.10 + 1.6 * thrust.powf(1.3))) * run * smooth(0.02, 0.2, n1)
    }
}

/// A recorded sound (assets/audio): stereo, at its own sample rate.
struct Clip {
    l: Vec<f32>,
    r: Vec<f32>,
    /// samples of the clip per sample of output
    step: f64,
}

impl Clip {
    /// A 16-bit PCM WAV file. A loop has its end cross-faded into its start,
    /// so it comes round without a seam, and is held to one loudness, an RMS
    /// of 1; a one-off is faded in and out and levelled to a peak of 1.
    fn wav(b: &[u8], looped: bool) -> Clip {
        let u16_at = |i: usize| u16::from_le_bytes([b[i], b[i + 1]]) as usize;
        let u32_at = |i: usize| u32::from_le_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]]) as usize;
        assert!(b.len() > 12 && &b[..4] == b"RIFF" && &b[8..12] == b"WAVE", "not a WAV file");
        let (mut ch, mut rate, mut data) = (2, RATE as usize, &b[..0]);
        let mut at = 12;
        while at + 8 <= b.len() {
            let n = u32_at(at + 4).min(b.len() - at - 8);
            match &b[at..at + 4] {
                b"fmt " => {
                    assert!(u16_at(at + 8) == 1 && u16_at(at + 22) == 16, "WAV must be 16-bit PCM");
                    (ch, rate) = (u16_at(at + 10).max(1), u32_at(at + 12));
                }
                b"data" => data = &b[at + 8..at + 8 + n],
                _ => {}
            }
            at += 8 + n + (n & 1);
        }
        let s = |i: usize| i16::from_le_bytes([data[i * 2], data[i * 2 + 1]]) as f32 / 32768.0;
        let frames = data.len() / 2 / ch;
        let mut l: Vec<f32> = (0..frames).map(|i| s(i * ch)).collect();
        let mut r: Vec<f32> = if ch > 1 { (0..frames).map(|i| s(i * ch + 1)).collect() } else { l.clone() };
        if looped {
            // A tenth of a second of the end under the start, at equal power (the two are unrelated noise).
            let n = (rate / 10).min(frames / 4);
            for v in [&mut l, &mut r] {
                let len = v.len() - n;
                for i in 0..n {
                    let t = i as f32 / n as f32 * FRAC_PI_2;
                    v[i] = v[i] * t.sin() + v[len + i] * t.cos();
                }
                v.truncate(len);
            }
            // Held to one loudness all the way round: a recording that swells and
            // fades (as generated ones do) is turned up and down against its own
            // envelope, half a second wide, by up to 12 dB.
            let len = l.len();
            let mut sum = vec![0.0f64; len + 1];
            for i in 0..len {
                sum[i + 1] = sum[i] + (l[i] * l[i] + r[i] * r[i]) as f64 * 0.5;
            }
            let mean = (sum[len] / len.max(1) as f64).max(1e-12);
            let half = (rate / 4).clamp(1, len.max(2) / 2);
            for i in 0..len {
                let (a, b) = ((i + len - half) % len, (i + half) % len);
                let around = if a < b { sum[b] - sum[a] } else { sum[len] - sum[a] + sum[b] } / (2 * half) as f64;
                let g = ((mean / around.max(1e-12)).sqrt() as f32).clamp(0.25, 4.0);
                l[i] *= g;
                r[i] *= g;
            }
            let rms = (l.iter().chain(&r).map(|x| x * x).sum::<f32>() / (2 * l.len()).max(1) as f32).sqrt().max(1e-6);
            l.iter_mut().chain(r.iter_mut()).for_each(|x| *x /= rms);
        } else {
            let (rise, fall) = ((rate / 400).max(1) as f32, (rate * 2 / 5).max(1) as f32);
            let peak = l.iter().chain(&r).fold(1e-6f32, |m, x| m.max(x.abs()));
            let len = l.len();
            for v in [&mut l, &mut r] {
                for (i, x) in v.iter_mut().enumerate() {
                    *x *= (i as f32 / rise).min((len - i) as f32 / fall).min(1.0) / peak;
                }
            }
        }
        Clip { l, r, step: rate as f64 / RATE as f64 }
    }
    fn len(&self) -> f64 {
        self.l.len() as f64
    }
    /// The clip at `pos` (in its samples), coming round again past its end.
    fn at(&self, pos: f64) -> (f32, f32) {
        let n = self.l.len();
        if n == 0 {
            return (0.0, 0.0);
        }
        let i = pos as usize % n;
        let j = if i + 1 == n { 0 } else { i + 1 };
        let f = pos.fract() as f32;
        (self.l[i] + (self.l[j] - self.l[i]) * f, self.r[i] + (self.r[j] - self.r[i]) * f)
    }
}

/// The fighter's recordings (generated with ElevenLabs, see assets/audio).
struct Clips {
    idle: Clip,
    mil: Clip,
    /// the afterburner: its tearing crackle, and the rumble under it
    burn: Clip,
    rumble: Clip,
    light: Clip,
    cabin: Clip,
}

fn clips() -> &'static Clips {
    static CLIPS: OnceLock<Clips> = OnceLock::new();
    CLIPS.get_or_init(|| Clips {
        idle: Clip::wav(include_bytes!("../assets/audio/fa18_idle_loop.wav"), true),
        mil: Clip::wav(include_bytes!("../assets/audio/fa18_mil_loop.wav"), true),
        burn: Clip::wav(include_bytes!("../assets/audio/fa18_afterburner_loop.wav"), true),
        rumble: Clip::wav(include_bytes!("../assets/audio/fa18_afterburner_rumble_loop.wav"), true),
        light: Clip::wav(include_bytes!("../assets/audio/fa18_afterburner_ignite.wav"), false),
        cabin: Clip::wav(include_bytes!("../assets/audio/fa18_cockpit_loop.wav"), true),
    })
}

/// A fighter's pair of afterburning turbofans, from recordings: the whine at
/// idle gives way to the roar of military power as the engines spool up (each
/// loop following the fan speed in pitch), the afterburners light with a
/// thump and tear and crackle over the top, and in the cockpit all of it is a
/// rumble under the hiss of the air conditioning. `Jet` adds the tones that
/// follow the fan exactly.
struct Fighter {
    c: Option<&'static Clips>,
    /// where each loop has got to: idle, military power, the afterburner's crackle and rumble, cockpit
    pos: [f64; 5],
    /// the afterburners lighting: how far through the thump
    light: Option<f64>,
    lit: bool,
    /// set each control tick: the loops' levels, and the idle and military loops' pitch
    gain: [f32; 5],
    pitch: [f64; 2],
}

impl Fighter {
    fn new() -> Self {
        // (The loops start apart, so two of them never beat against each other from the same place.)
        Self { c: None, pos: [0.0, 31_000.0, 97_000.0, 203_000.0, 11_000.0], light: None, lit: false, gain: [0.0; 5], pitch: [1.0; 2] }
    }
    fn control(&mut self, p: &Params, lit: bool) {
        if p.fighter < 0.5 {
            self.gain = [0.0; 5];
            self.lit = false;
            return;
        }
        self.c.get_or_insert_with(clips);
        let n1 = p.n1.max(0.0);
        // 0 at flight idle .. 1 flat out
        let x = ((n1 - 0.6) / 0.4).clamp(0.0, 1.0);
        let run = smooth(0.02, 0.3, n1) * if p.health > 0.0 { 1.0 } else { 0.2 } * p.live;
        // (The loops are levelled to an RMS of 1: these set them against the rest of the mix,
        // the afterburner well into the master limiter, which is how loud it is.)
        self.gain = [0.07 * (1.0 - smooth(0.3, 0.95, x)) * run, (0.05 + 0.95 * x.powf(1.6)) * 0.31 * run, 0.42 * p.burner * run, 0.50 * p.burner * run, (0.3 + 0.7 * x + 0.5 * p.burner) * 0.13 * run];
        self.pitch = [(1.0 + 0.9 * (n1 - 0.62)).clamp(0.4, 1.5) as f64, 0.86 + 0.14 * x as f64];
        if lit && !self.lit && run > 0.5 {
            self.light = Some(0.0);
        }
        self.lit = lit;
    }
    /// Left and right as heard outside, and in the cockpit.
    fn sample(&mut self) -> ([f32; 2], [f32; 2]) {
        let Some(c) = self.c else { return ([0.0; 2], [0.0; 2]) };
        let mut out = [0.0f32; 2];
        for (k, clip) in [&c.idle, &c.mil, &c.burn, &c.rumble].into_iter().enumerate() {
            self.pos[k] = (self.pos[k] + clip.step * if k < 2 { self.pitch[k] } else { 1.0 }) % clip.len().max(1.0);
            let g = self.gain[k];
            if g > 1e-4 {
                let (l, r) = clip.at(self.pos[k]);
                out[0] += l * g;
                out[1] += r * g;
            }
        }
        if let Some(t) = self.light {
            // A thump that dies away into the roar it started.
            let (l, r) = c.light.at(t);
            let g = 0.9 * (-(t / c.light.step.max(1e-9) / RATE as f64) as f32 / 0.5).exp();
            out[0] += l * g;
            out[1] += r * g;
            self.light = (t + c.light.step < c.light.len() - 1.0).then_some(t + c.light.step);
        }
        self.pos[4] = (self.pos[4] + c.cabin.step) % c.cabin.len().max(1.0);
        let (l, r) = c.cabin.at(self.pos[4]);
        (out, [l * self.gain[4], r * self.gain[4]])
    }
}

/// Air rushing past: outside, and as heard through the cabin.
struct Wind {
    ext: [Bq; 4],
    hiss: [Bq; 2],
    roar: Bq,
    cab: [Bq; 2],
    seal: [Bq; 2],
    crack: Bq,
    howl: [Bq; 2],
    gust: Slow,
    gust_v: f32,
}

impl Wind {
    fn new() -> Self {
        Self { ext: [Bq::default(); 4], hiss: [Bq::new(Bq::hp, 4500.0, 0.7); 2], roar: Bq::new(Bq::bp, 180.0, 0.8), cab: [Bq::default(); 2], seal: [Bq::default(); 2], crack: Bq::new(Bq::hp, 2500.0, 0.7), howl: [Bq::default(); 2], gust: Slow::default(), gust_v: 0.0 }
    }
    fn control(&mut self, p: &Params) {
        let v = p.airspeed;
        let fc = 220.0 + 28.0 * v;
        for b in &mut self.ext {
            b.lp(fc, 0.6);
        }
        let cab = (280.0 + 6.0 * v).min(900.0);
        for b in &mut self.cab {
            b.lp(cab, 0.7);
        }
        self.seal[0].bp(1850.0 + 9.0 * v, 9.0);
        self.seal[1].bp(2900.0 + 11.0 * v, 12.0);
        self.howl[0].bp(520.0 + 8.0 * v, 16.0);
        self.howl[1].bp(780.0 + 10.0 * v, 20.0);
    }
    /// Outside (l, r), cabin (l, r).
    fn sample(&mut self, p: &Params, rng: &mut Rng) -> [f32; 4] {
        let v = p.airspeed;
        let (nc, nl, nr) = (rng.f(), rng.f(), rng.f());
        let (l, r) = (nc * 0.6 + nl * 0.4, nc * 0.6 + nr * 0.4);
        // Gusts swell and fade the rush.
        self.gust_v = self.gust.run(rng.f(), 0.0004);
        let gust = 1.0 + self.gust_v * (0.06 + 0.25 * smooth(0.2, 3.0, p.rough));
        let q = (v / 60.0).powi(2).min(2.6) * gust.max(0.2);
        let a = self.ext[0].run(l);
        let mut ext_l = self.ext[1].run(a) * q * 0.48;
        let a = self.ext[2].run(r);
        let mut ext_r = self.ext[3].run(a) * q * 0.48;
        let hiss = (v / 80.0).powi(3).min(3.0) * 0.09;
        ext_l += self.hiss[0].run(nl) * hiss;
        ext_r += self.hiss[1].run(nr) * hiss;
        // Gear legs and flaps hanging in the airflow.
        let roar = self.roar.run(nc) * (p.gear * 0.8 + p.flaps * 0.6) * (v / 50.0).powi(2).min(2.5) * 0.5;
        ext_l += roar;
        ext_r += roar;
        // Inside: a muffled roar, a whistle at the door seals, and in through a cracked windshield.
        let a = self.cab[0].run(nc);
        let cab = self.cab[1].run(a) * q * 0.4;
        let s = smooth(25.0, 95.0, v).powi(2) * gust.max(0.0);
        let seal = (self.seal[0].run(nl) * 0.7 + self.seal[1].run(nr) * 0.45) * s * (1.0 + 4.0 * p.cracked);
        let crack = self.crack.run(nr) * q * 0.12 * p.cracked;
        // Past the never-exceed speed the airframe howls.
        let vne = p.vne_or();
        let over = smooth(0.85 * vne, vne + 15.0, v).powi(2);
        let howl = (self.howl[0].run(nl) + self.howl[1].run(nr) * 0.7) * over * 2.0;
        [ext_l + howl * 0.5, ext_r + howl * 0.5, cab + seal + crack + howl, cab + seal * 0.8 + crack + howl * 0.9]
    }
}

/// Airframe and ground noises that run continuously (not one-shots).
struct Body {
    rumble: [Bq; 2],
    buffet: Bq,
    buffet_mid: Bq,
    buffet_mod: Slow,
    flutter_ph: f32,
    flutter_bp: Bq,
    rat: [Bq; 2],
    roll: Bq,
    roll_mod: Slow,
    squeal: Bq,
    scrape: [Bq; 2],
    scrape_mod: Slow,
    rustle: Bq,
    rustle_mod: Slow,
    gear_ph: f32,
    flap_ph: f32,
    motor: Bq,
}

impl Body {
    fn new() -> Self {
        Self {
            rumble: [Bq::new(Bq::lp, 65.0, 0.8); 2],
            buffet: Bq::new(Bq::lp, 45.0, 0.9),
            buffet_mid: Bq::new(Bq::bp, 160.0, 1.2),
            buffet_mod: Slow::default(),
            flutter_ph: 0.0,
            flutter_bp: Bq::new(Bq::bp, 420.0, 1.5),
            rat: [Bq::new(Bq::ring, 2300.0, 7.0), Bq::new(Bq::ring, 1150.0, 6.0)],
            roll: Bq::default(),
            roll_mod: Slow::default(),
            squeal: Bq::new(Bq::bp, 2450.0, 30.0),
            scrape: [Bq::new(Bq::bp, 1200.0, 1.5), Bq::new(Bq::bp, 3600.0, 4.0)],
            scrape_mod: Slow::default(),
            rustle: Bq::new(Bq::hp, 1600.0, 0.7),
            rustle_mod: Slow::default(),
            gear_ph: 0.0,
            flap_ph: 0.0,
            motor: Bq::new(Bq::bp, 900.0, 3.0),
        }
    }
    fn control(&mut self, p: &Params) {
        self.roll.lp(70.0 + 5.0 * p.rolling, 0.8);
    }
    /// Structure-borne (l, r) and outside noises (l, r).
    fn sample(&mut self, p: &Params, rng: &mut Rng) -> [f32; 4] {
        let (mut sl, mut sr, mut ml, mut mr) = (0.0, 0.0, 0.0, 0.0);
        // Rough air: a low rumble through the airframe, and things rattling in the panel and doors.
        let rough = smooth(0.25, 3.0, p.rough) * smooth(10.0, 50.0, p.airspeed) + p.jolt.min(2.0) * 0.5;
        if rough > 0.001 {
            sl += self.rumble[0].run(rng.f()) * rough * 0.9;
            sr += self.rumble[1].run(rng.f()) * rough * 0.9;
        }
        let rate = rough * 25.0 + p.buffet * 40.0 + p.flutter.min(1.0) * 60.0 + smooth(2.0, 20.0, p.rolling) * 6.0;
        let mut c = [0.0; 2];
        for (i, ci) in c.iter_mut().enumerate() {
            if rng.u() < rate * DT {
                *ci = rng.range(0.2, 1.0);
            }
            let y = self.rat[i].run(*ci) * 0.05;
            *ci = y;
        }
        sl += c[0] + c[1] * 0.4;
        sr += c[1] + c[0] * 0.4;
        // Stall buffet: the wing shaking in its own wake.
        if p.buffet > 0.001 {
            let m = (0.5 + 0.5 * self.buffet_mod.run(rng.f(), 0.004)).max(0.0);
            let b = (self.buffet.run(rng.f()) * 2.5 + self.buffet_mid.run(rng.f()) * 0.5) * p.buffet * m;
            sl += b;
            sr += b;
        }
        // Flutter: a control surface buzzing on its hinges.
        if p.flutter > 0.001 {
            self.flutter_ph = (self.flutter_ph + 21.0 * DT).fract();
            let saw = 1.0 - self.flutter_ph;
            let f = (self.flutter_bp.run(rng.f()) * saw * saw * saw * 3.0 + (TAU * self.flutter_ph).sin() * 0.4) * p.flutter.min(1.5) * 0.7;
            ml += f;
            mr += f;
        }
        // Tyres on the runway, brakes, belly scraping, branches.
        if p.rolling > 0.3 {
            let m = 1.0 + 0.4 * self.roll_mod.run(rng.f(), 0.002);
            let r = self.roll.run(rng.f()) * (p.rolling / 30.0).min(1.2) * 0.5 * m;
            ml += r;
            mr += r;
            let sq = self.squeal.run(rng.f()) * p.braking * smooth(3.0, 25.0, p.rolling) * 1.2;
            ml += sq;
            mr += sq;
        }
        if p.scrape > 0.001 {
            let m = (0.6 + 0.5 * self.scrape_mod.run(rng.f(), 0.02)).max(0.0);
            let s = (self.scrape[0].run(rng.f()) * 1.4 + self.scrape[1].run(rng.f()) * 1.0) * p.scrape * m;
            ml += s;
            mr += s;
        }
        if p.foliage > 0.001 {
            let m = (self.rustle_mod.run(rng.f(), 0.03)).max(0.0);
            let s = self.rustle.run(rng.f()) * p.foliage * m * 0.25;
            ml += s;
            mr += s;
        }
        // The gear's hydraulic pump and the flaps' electric motor.
        if p.gear_moving > 0.001 {
            self.gear_ph = (self.gear_ph + 370.0 * DT).fract();
            let w = ((TAU * self.gear_ph).sin() + 0.5 * (2.0 * TAU * self.gear_ph).sin()) * 0.03 + self.motor.run(rng.f()) * 0.08;
            sl += w * p.gear_moving;
            sr += w * p.gear_moving;
        }
        if p.flaps_moving > 0.001 {
            self.flap_ph = (self.flap_ph + 610.0 * (1.0 + 0.01 * rng.f()) * DT).fract();
            let w = ((TAU * self.flap_ph).sin() + 0.3 * (3.0 * TAU * self.flap_ph).sin()) * 0.025 * p.flaps_moving;
            sl += w * 0.6;
            sr += w;
        }
        [sl, sr, ml, mr]
    }
}

/// The cockpit's warning sounds.
struct Alarms {
    t: f32,
    beep_ph: f32,
    beep_f: f32,
    beep_env: f32,
    horn_ph: f32,
    horn_env: f32,
    vib: f32,
    clack_ph: f32,
    clack: [Bq; 2],
    gear_ph: f32,
    gear_env: f32,
    env_k: f32,
}

impl Alarms {
    fn new() -> Self {
        Self { t: 0.0, beep_ph: 0.0, beep_f: 1000.0, beep_env: 0.0, horn_ph: 0.0, horn_env: 0.0, vib: 0.0, clack_ph: 0.0, clack: [Bq::new(Bq::ring, 2400.0, 22.0), Bq::new(Bq::ring, 950.0, 14.0)], gear_ph: 0.0, gear_env: 0.0, env_k: kf(80.0) }
    }
    fn sample(&mut self, w: &Warn) -> f32 {
        self.t = (self.t + DT) % 240.0;
        let t = self.t;
        // One beeper, the most urgent warning wins.
        let (f, on) = if w.pull_up {
            let c = t % 0.24;
            if c < 0.09 { (1450.0, true) } else { (1050.0, (0.12..0.21).contains(&c)) }
        } else if w.dive > 0.0 {
            let rate = 3.0 + 5.0 * w.dive.min(1.0);
            (1250.0 + 250.0 * w.dive.min(1.0), (t * rate).fract() < 0.5)
        } else if w.too_low {
            let c = t % 0.6;
            (900.0, c < 0.13 || (0.2..0.33).contains(&c))
        } else if w.sink {
            (760.0, t % 0.8 < 0.25)
        } else {
            (self.beep_f, false)
        };
        self.beep_f = f;
        self.beep_env += (on as u8 as f32 - self.beep_env) * self.env_k;
        let mut out = 0.0;
        if self.beep_env > 1e-4 {
            self.beep_ph = (self.beep_ph + f * DT).fract();
            let x = TAU * self.beep_ph;
            out += (x.sin() + 0.3 * (3.0 * x).sin()) * self.beep_env * 0.5;
        }
        // Stall horn: a reedy note that comes in a few knots above the stall.
        let target = if w.stall > 0.02 { 0.45 + 0.55 * w.stall.min(1.0) } else { 0.0 };
        self.horn_env += (target - self.horn_env) * self.env_k * 0.5;
        if self.horn_env > 1e-4 {
            self.vib = (self.vib + 5.5 * DT).fract();
            self.horn_ph = (self.horn_ph + 1150.0 * (1.0 + 0.012 * (TAU * self.vib).sin()) * DT).fract();
            let x = TAU * self.horn_ph;
            out += (x.sin() + 0.5 * (2.0 * x).sin() + 0.33 * (3.0 * x).sin() + 0.2 * (4.0 * x).sin()) * self.horn_env * 0.32;
        }
        // Overspeed clacker.
        let mut x = 0.0;
        if w.overspeed > 0.0 {
            self.clack_ph += 13.0 * DT;
            if self.clack_ph >= 1.0 {
                self.clack_ph -= 1.0;
                x = 0.6 + 0.4 * w.overspeed.min(1.0);
            }
        }
        out += (self.clack[0].run(x) + self.clack[1].run(x) * 0.6) * 0.9;
        // Gear-up horn: on and off, twice a second.
        let g = w.gear && t % 0.5 < 0.3;
        self.gear_env += (g as u8 as f32 - self.gear_env) * self.env_k;
        if self.gear_env > 1e-4 {
            self.gear_ph = (self.gear_ph + 480.0 * DT).fract();
            let x = TAU * self.gear_ph;
            out += (x.sin() + 0.33 * (3.0 * x).sin() + 0.2 * (5.0 * x).sin()) * self.gear_env * 0.3;
        }
        out
    }
}

/// Explosions and fire.
struct World {
    lp_boom: f32,
    lp_boom2: f32,
    lp_wind: f32,
    booms: Vec<(f32, f32)>,
    crackle: f32,
}

impl World {
    fn sample(&mut self, fire: f32, rng: &mut Rng) -> f32 {
        let mut out = 0.0;
        // A sub thump, a long low roar, and a crack at the front.
        let mut i = 0;
        while i < self.booms.len() {
            let (t, p) = self.booms[i];
            let nz = rng.f();
            let cutoff = 0.004 + 0.07 * (-t * 2.2).exp();
            self.lp_boom += (nz - self.lp_boom) * cutoff;
            self.lp_boom2 += (self.lp_boom - self.lp_boom2) * 0.2;
            let roar = self.lp_boom2 * 9.0 * (-t * 1.1).exp();
            let thump = (TAU * (30.0 * t + 22.0 * (1.0 - (-t * 6.0).exp()) / 6.0)).sin() * (-t * 3.5).exp() * 1.3;
            let crack = nz * (-t * 38.0).exp() * 0.9;
            out += (roar + thump + crack) * p.min(2.0) * 0.55;
            self.booms[i].0 += DT;
            if t > 6.0 {
                self.booms.swap_remove(i);
            } else {
                i += 1;
            }
        }
        // Fire: a soft roar and random pops.
        if fire > 0.01 {
            let w = rng.f();
            self.lp_wind += (w - self.lp_wind) * 0.02;
            if rng.u() > 0.99965 {
                self.crackle = 1.0;
            }
            self.crackle *= 0.995;
            out += (rng.f() * self.crackle * 0.25 + self.lp_wind * 0.5) * fire;
        }
        out
    }
}

/// How the aircraft's sound reaches the ear: outside through the air, or inside the cabin.
struct Cabin {
    /// outside: air absorption, two poles per channel
    abs: [f32; 4],
    abs_k: f32,
    /// inside: the engine through the firewall, and the cabin's boom
    eng_lp: [Bq; 2],
    eng_boom: Bq,
    mid_lp: [f32; 2],
    mid_k: f32,
    /// a small, boxy room: a few early reflections
    buf: Vec<f32>,
    pos: usize,
}

const ROOM: usize = 1024;
const TAPS: [(usize, f32, f32); 6] = [(97, 0.22, -0.7), (131, 0.2, 0.7), (241, 0.15, -0.4), (283, 0.14, 0.5), (389, 0.1, -0.2), (433, 0.09, 0.3)];

impl Cabin {
    fn new() -> Self {
        let mut eng_lp = [Bq::default(); 2];
        eng_lp[0].lp(650.0, 0.7);
        eng_lp[1].lp(900.0, 0.6);
        let mut eng_boom = Bq::default();
        eng_boom.peak(105.0, 1.3, 5.0);
        Self { abs: [0.0; 4], abs_k: 1.0, eng_lp, eng_boom, mid_lp: [0.0; 2], mid_k: kf(1800.0), buf: vec![0.0; ROOM], pos: 0 }
    }
}

struct Synth {
    s: Arc<Shared>,
    n: u32,
    rng: Rng,
    /// what the game asked for, and the smoothed values the sound follows
    t: Params,
    p: Params,
    eng: Engine,
    jet: Jet,
    fighter: Fighter,
    wind: Wind,
    body: Body,
    voices: Vec<Voice>,
    alarms: Alarms,
    world: World,
    cab: Cabin,
    dist_gain: f32,
    lim: f32,
    lim_k: f32,
    dc: [Bq; 2],
    right: Option<f32>,
    /// sums of squares of each part of the mix, for tuning the levels in the tests
    #[cfg(test)]
    meter: [f64; 10],
}

impl Synth {
    fn new(s: Arc<Shared>) -> Self {
        Self {
            s,
            n: 0,
            rng: Rng(0x1234567),
            t: Params::default(),
            p: Params::default(),
            eng: Engine::new(),
            jet: Jet::new(),
            fighter: Fighter::new(),
            wind: Wind::new(),
            body: Body::new(),
            voices: Vec::with_capacity(MAX_VOICES),
            alarms: Alarms::new(),
            world: World { lp_boom: 0.0, lp_boom2: 0.0, lp_wind: 0.0, booms: Vec::new(), crackle: 0.0 },
            cab: Cabin::new(),
            dist_gain: 1.0,
            lim: 0.0,
            lim_k: decay(0.15),
            dc: [Bq::new(Bq::hp, 18.0, 0.7); 2],
            right: None,
            #[cfg(test)]
            meter: [0.0; 10],
        }
    }

    fn push(&mut self, v: Voice) {
        if self.voices.len() >= MAX_VOICES {
            // The one furthest through goes.
            if let Some(i) = (0..self.voices.len()).max_by(|&a, &b| (self.voices[a].t / self.voices[a].dur).total_cmp(&(self.voices[b].t / self.voices[b].dur))) {
                self.voices.swap_remove(i);
            }
        }
        self.voices.push(v);
    }

    fn event(&mut self, e: Event) {
        let r = &mut self.rng;
        match e {
            Event::Boom(p) => self.world.booms.push((0.0, p)),
            Event::Snap(s) => {
                let s = s.min(2.0);
                let pn = r.range(-0.6, 0.6);
                let f = [r.range(500.0, 900.0), r.range(1300.0, 2600.0)];
                let v = Voice::new(0.45, pn, Bus::Mid).strike(0.9 * s).res(f, [35.0, 45.0], [0.35, 0.25]).pulses(300.0, 30.0, true, 0.35 * s).noise(2800.0, 1100.0, 0.8, 1.6 * s, 0.1).tone(55.0, 38.0, 0.7 * s, 0.12);
                self.push(v);
            }
            Event::Bump(s) => {
                let s = s.min(3.0);
                let v = Voice::new(0.3, r.range(-0.3, 0.3), Bus::Struct).tone(r.range(50.0, 70.0), 40.0, 0.35 * s, 0.07).noise(260.0, 180.0, 1.0, 0.9 * s, 0.03).pulses(70.0, 10.0, true, 0.6 * s.min(1.5)).res([r.range(1900.0, 2600.0), r.range(3000.0, 3800.0)], [7.0, 8.0], [0.05, 0.035]);
                self.push(v);
            }
            Event::Touch(s) => {
                let s = (s / 2.5).clamp(0.15, 2.0);
                for (k, pn) in [(0.0, -0.4), (r.range(0.01, 0.06), 0.4)] {
                    let f0 = self.rng.range(2200.0, 2700.0);
                    let v = Voice::new(0.28, pn, Bus::Mid).delay(k).noise(f0, f0 * 0.6, 3.5, 2.5 * s, 0.08).tone(62.0, 42.0, 0.45 * s, 0.09);
                    self.push(v);
                }
            }
            Event::Clunk(s) => {
                let v = Voice::new(0.2, r.range(-0.2, 0.2), Bus::Struct).tone(88.0, 70.0, 0.4 * s, 0.05).noise(900.0, 700.0, 1.2, 1.0 * s, 0.012).strike(0.5 * s).res([r.range(600.0, 700.0), r.range(1350.0, 1500.0)], [10.0, 12.0], [0.08, 0.05]);
                self.push(v);
            }
            Event::Crunch(s) => {
                let s = s.min(2.0);
                let v = Voice::new(0.6, r.range(-0.5, 0.5), Bus::Mid).pulses(160.0, 15.0, true, 0.9 * s).res([r.range(800.0, 1100.0), r.range(2200.0, 3000.0)], [3.0, 4.0], [0.4, 0.25]).noise(4200.0, 3000.0, 0.7, 0.8 * s, 0.25).tone(70.0, 45.0, 0.5 * s, 0.06);
                self.push(v);
            }
            Event::Laser(s) => {
                // The snap of a structural failure, made over into a laser's
                // "pew": its gliding tone swept down fast from high up, its
                // noise a short bright hiss, no ringing.
                let s = s.min(2.0);
                let f0 = r.range(2300.0, 2700.0);
                let v = Voice::new(0.2, r.range(-0.2, 0.2), Bus::Mid).tone(f0, f0 * 0.16, 0.3 * s, 0.07).noise(6000.0, 2000.0, 1.2, 0.25 * s, 0.025);
                self.push(v);
            }
            Event::Splash(s) => {
                let s = s.min(2.0);
                let pn = r.range(-0.3, 0.3);
                // The slap: a dull thump with a burst of torn water.
                let slap = Voice::new(1.4, pn, Bus::Mid).tone(r.range(42.0, 55.0), 28.0, 0.9 * s, 0.2).noise(1100.0, 240.0, 0.6, 2.4 * s, 0.3);
                // The sheet thrown up roaring as it breaks into spray.
                let sheet = Voice::new(2.6, -pn, Bus::Mid).noise(2400.0, 650.0, 0.5, 1.3 * s, 0.8);
                // Then it rains back down: a hiss and a patter of drops.
                let rain = Voice::new(3.6, pn * 0.5, Bus::Mid).delay(r.range(0.7, 1.1) * s.sqrt()).noise(5200.0, 2600.0, 0.6, 0.55 * s, 1.6).pulses(900.0, 60.0, true, 0.5 * s).res([r.range(2600.0, 3400.0), r.range(5000.0, 6500.0)], [2.5, 3.0], [0.12, 0.08]);
                for v in [slap, sheet, rain] {
                    self.push(v);
                }
            }
        }
    }

    /// The airframe creaks and groans under load.
    fn creak(&mut self, intensity: f32) {
        let r = &mut self.rng;
        let pn = r.range(-0.8, 0.8);
        let v = if r.u() < 0.6 {
            let f0 = r.range(40.0, 180.0);
            Voice::new(r.range(0.12, 0.6), pn, Bus::Struct).pulses(f0, f0 * r.range(0.6, 1.6), false, intensity).res([r.range(450.0, 1500.0), r.range(1600.0, 2600.0)], [r.range(10.0, 15.0), 12.0], [0.09, 0.04])
        } else {
            let f0 = r.range(12.0, 35.0);
            let fr = r.range(150.0, 420.0);
            Voice::new(r.range(0.4, 1.2), pn, Bus::Struct).pulses(f0, f0 * r.range(0.7, 1.4), false, intensity * 1.3).res([fr, fr * 2.4], [7.0, 9.0], [0.16, 0.07])
        };
        self.push(v);
    }

    fn control(&mut self) {
        if let Some(p) = self.s.params.try_lock() {
            self.t = *p;
        }
        let events: Vec<Event> = match self.s.events.try_lock() {
            Some(mut e) => e.drain(..).collect(),
            None => Vec::new(),
        };
        for e in events {
            self.event(e);
        }
        let (t, p) = (&self.t, &mut self.p);
        let sm = |a: &mut f32, b: f32, tau: f32| *a += (b - *a) * kt(tau);
        sm(&mut p.live, t.live, 0.12);
        sm(&mut p.master, t.master, 0.05);
        sm(&mut p.rpm, t.rpm, 0.04);
        sm(&mut p.power, t.power, 0.05);
        sm(&mut p.throttle, t.throttle, 0.05);
        sm(&mut p.health, t.health, 0.1);
        sm(&mut p.prop, t.prop, 0.1);
        sm(&mut p.airspeed, t.airspeed, 0.08);
        sm(&mut p.g_load, t.g_load, 0.05);
        sm(&mut p.rough, t.rough, 0.3);
        sm(&mut p.jolt, t.jolt, 0.04);
        sm(&mut p.buffet, t.buffet, 0.06);
        sm(&mut p.flutter, t.flutter, 0.06);
        sm(&mut p.gear, t.gear, 0.05);
        sm(&mut p.gear_moving, t.gear_moving, 0.08);
        sm(&mut p.flaps, t.flaps, 0.05);
        sm(&mut p.flaps_moving, t.flaps_moving, 0.06);
        sm(&mut p.rolling, t.rolling, 0.05);
        sm(&mut p.braking, t.braking, 0.08);
        sm(&mut p.scrape, t.scrape, 0.06);
        sm(&mut p.foliage, t.foliage, 0.08);
        sm(&mut p.cracked, t.cracked, 0.2);
        sm(&mut p.cabin, t.cabin, 0.05);
        sm(&mut p.dist, t.dist, 0.08);
        sm(&mut p.fire, t.fire, 0.3);
        sm(&mut p.n1, t.n1, 0.12);
        sm(&mut p.burner, t.burner, 0.05);
        (p.jet, p.fighter, p.vne) = (t.jet, t.fighter, t.vne);
        p.health = if t.health <= 0.0 { 0.0 } else { p.health };
        p.warn = t.warn;
        if p.live < 0.01 {
            p.warn = Warn::default();
        }
        let p = self.p;

        self.eng.control(&p);
        self.jet.control(&p);
        self.fighter.control(&p, self.t.burner > 0.5);
        self.wind.control(&p);
        self.body.control(&p);
        for v in &mut self.voices {
            v.control();
        }
        // Distance: about -6 dB per doubling past the chase camera's usual
        // spot, and the air soaks up the highs.
        let d = p.dist.max(1.0);
        self.dist_gain = (24.0 / (d + 8.0)).min(1.25);
        self.cab.abs_k = kf(20000.0 / (1.0 + d / 70.0));

        // Creaks: more the harder the airframe is worked.
        if p.live > 0.5 {
            let v = p.airspeed;
            let vne = p.vne_or();
            let over = smooth(0.82 * vne, vne + 15.0, v);
            let g = smooth(2.8, 5.5, p.g_load.abs()) + smooth(-0.5, -2.5, p.g_load);
            let rough = smooth(0.5, 4.0, p.rough) * smooth(20.0, 60.0, v);
            let stress = over * 2.5 + g * 2.5 + rough * 0.5 + p.jolt.min(2.0) * 2.0 + p.buffet * 0.8 + p.flutter.min(1.0) * 2.0;
            if self.rng.u() < stress * 1.6 * TICK_DT {
                let k = self.rng.range(0.5, 1.0);
                self.creak((0.3 + 0.35 * stress).min(1.2) * k);
            }
        }
    }
}

impl Iterator for Synth {
    type Item = f32;
    fn next(&mut self) -> Option<f32> {
        if let Some(r) = self.right.take() {
            return Some(r);
        }
        if self.n % TICK == 0 {
            self.control();
        }
        self.n = self.n.wrapping_add(1);
        let p = self.p;
        let rng = &mut self.rng;

        // The aircraft's own sources.
        let jet = p.jet > 0.5;
        let fighter = p.fighter > 0.5;
        let (ex, prop, mech) = if p.live > 0.001 && p.rpm > 1.0 {
            if fighter {
                // The recordings carry the roar: of the synthesised engine only the fan's tones are wanted.
                (self.jet.sample(&Params { power: p.power.min(1.0) * 0.12, ..p }, rng) * 0.3, 0.0, 0.0)
            } else if jet {
                (self.jet.sample(&p, rng), 0.0, 0.0)
            } else {
                self.eng.sample(&p, rng)
            }
        } else {
            (0.0, 0.0, 0.0)
        };
        let (fx, fc) = if fighter && p.live > 0.001 { self.fighter.sample() } else { ([0.0; 2], [0.0; 2]) };
        let w = self.wind.sample(&p, rng);
        let b = self.body.sample(&p, rng);
        let (mut sl, mut sr, mut ml, mut mr) = (b[0], b[1], b[2], b[3]);
        let mut i = 0;
        while i < self.voices.len() {
            let v = &mut self.voices[i];
            let y = v.sample(rng);
            let (l, r) = pan(y, v.pan);
            if v.bus == Bus::Struct {
                sl += l;
                sr += r;
            } else {
                ml += l;
                mr += r;
            }
            if v.done() {
                self.voices.swap_remove(i);
            } else {
                i += 1;
            }
        }

        // Outside: everything, fading and dulling with distance; the
        // airframe's creaks barely carry.
        let c = &mut self.cab;
        let dg = self.dist_gain;
        let wg = dg.sqrt().min(1.0);
        let eng = (ex + prop * 1.1 + mech * 0.6) * if jet { 3.2 } else { 1.0 };
        let mut el = (eng + fx[0] + ml + sl * 0.3) * dg + w[0] * wg;
        let mut er = (eng + fx[1] + mr + sr * 0.3) * dg + w[1] * wg;
        let k = c.abs_k;
        c.abs[0] += (el - c.abs[0]) * k;
        c.abs[1] += (c.abs[0] - c.abs[1]) * k;
        c.abs[2] += (er - c.abs[2]) * k;
        c.abs[3] += (c.abs[2] - c.abs[3]) * k;
        (el, er) = (c.abs[1], c.abs[3]);

        // Inside: the engine through the firewall with the cabin booming
        // along, the propeller's beat strong, the outside noises dulled,
        // the airframe right there.
        let (mut il, mut ir) = (0.0, 0.0);
        if p.cabin > 0.001 {
            // (An airliner's flight deck is far ahead of its engines, and well insulated.)
            // (A fighter's engines are right behind the seat: more of them comes through the airframe.)
            let ie = (ex * 1.0 + prop * 1.5 + mech * 0.3) * if jet { 0.18 } else { 1.0 } + (fx[0] + fx[1]) * 0.10;
            let ie = c.eng_lp[0].run(ie);
            let ie = c.eng_lp[1].run(ie);
            let ie = c.eng_boom.run(ie) * 0.75 + ex * 0.03;
            c.mid_lp[0] += (ml - c.mid_lp[0]) * c.mid_k;
            c.mid_lp[1] += (mr - c.mid_lp[1]) * c.mid_k;
            il = ie + w[2] + c.mid_lp[0] * 0.8 + sl + fc[0];
            ir = ie + w[3] + c.mid_lp[1] * 0.8 + sr + fc[1];
            let m = (il + ir) * 0.5;
            c.buf[c.pos] = m;
            for (d, g, side) in TAPS {
                let e = c.buf[(c.pos + ROOM - d) % ROOM] * g;
                il += e * (1.0 - side) * 0.5;
                ir += e * (1.0 + side) * 0.5;
            }
            c.pos = (c.pos + 1) % ROOM;
        }
        let x = p.cabin;
        let mut l = (el * (1.0 - x) + il * x) * p.live;
        let mut r = (er * (1.0 - x) + ir * x) * p.live;

        // The world, and the warnings (louder in the cockpit, where the speaker is).
        let world = self.world.sample(p.fire, rng);
        l += world;
        r += world;
        let alarm = self.alarms.sample(&p.warn) * (0.55 + 0.45 * x) * 0.32;
        l += alarm;
        r += alarm;

        #[cfg(test)]
        for (m, v) in self.meter.iter_mut().zip([ex, prop, mech, w[0], w[2], sl, ml, il - w[2] - sl, alarm, l]) {
            *m += (v * v) as f64;
        }
        // Master: a gentle limiter, then a soft clip.
        let (l, r) = (self.dc[0].run(l) * p.master * 0.8, self.dc[1].run(r) * p.master * 0.8);
        let pk = l.abs().max(r.abs());
        self.lim = if pk > self.lim { pk } else { self.lim * self.lim_k };
        let g = if self.lim > 0.8 { 0.8 / self.lim } else { 1.0 };
        let (l, r) = ((l * g).tanh(), (r * g).tanh());
        self.right = Some(r);
        Some(l)
    }
}

impl rodio::Source for Synth {
    fn current_span_len(&self) -> Option<usize> {
        None
    }
    fn channels(&self) -> rodio::ChannelCount {
        NonZero::new(2).unwrap()
    }
    fn sample_rate(&self) -> rodio::SampleRate {
        NonZero::new(RATE).unwrap()
    }
    fn total_duration(&self) -> Option<Duration> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flying() -> Params {
        Params { live: 1.0, master: 1.0, rpm: 2300.0, power: 0.65, throttle: 0.65, health: 1.0, prop: 1.0, airspeed: 60.0, g_load: 1.0, rough: 0.15, dist: 16.0, ..Default::default() }
    }

    /// Render `secs` of sound with these parameters (after a second to settle).
    fn render(p: Params, events: &[Event], secs: f32) -> Vec<[f32; 2]> {
        let shared = Arc::new(Shared::default());
        *shared.params.lock() = p;
        let mut s = Synth::new(shared.clone());
        s.p = p;
        for _ in 0..RATE * 2 {
            s.next();
        }
        shared.events.lock().extend_from_slice(events);
        s.meter = [0.0; 10];
        let x: Vec<[f32; 2]> = (0..(secs * RATE as f32) as usize).map(|_| [s.next().unwrap(), s.next().unwrap()]).collect();
        if std::env::var("AUDIO_METER").is_ok() {
            let n = x.len() as f64;
            let names = ["exh", "prop", "mech", "windX", "windI", "struct", "mid", "intEng", "alarm", "pre"];
            println!("    {}", names.iter().zip(s.meter).map(|(k, m)| format!("{k} {:.3}", (m / n).sqrt())).collect::<Vec<_>>().join("  "));
        }
        x
    }
    fn rms(x: &[[f32; 2]]) -> f32 {
        (x.iter().map(|s| s[0] * s[0] + s[1] * s[1]).sum::<f32>() / (2 * x.len()) as f32).sqrt()
    }
    /// RMS of the first difference: a rough measure of how much treble there is.
    fn bright(x: &[[f32; 2]]) -> f32 {
        (x.windows(2).map(|w| (w[1][0] - w[0][0]).powi(2)).sum::<f32>() / x.len() as f32).sqrt() / rms(x).max(1e-6)
    }
    fn db(a: f32, b: f32) -> f32 {
        20.0 * (a / b).log10()
    }

    #[test]
    fn stays_finite_and_in_range() {
        let mut p = flying();
        p.airspeed = 120.0;
        p.rough = 6.0;
        p.jolt = 3.0;
        p.buffet = 1.0;
        p.flutter = 2.0;
        p.scrape = 1.0;
        p.rolling = 40.0;
        p.braking = 1.0;
        p.fire = 1.0;
        p.cracked = 1.0;
        p.warn = Warn { stall: 1.0, pull_up: true, overspeed: 1.0, gear: true, ..Default::default() };
        for cabin in [0.0, 1.0] {
            p.cabin = cabin;
            let ev = [Event::Boom(2.0), Event::Snap(2.0), Event::Bump(3.0), Event::Touch(8.0), Event::Clunk(1.0), Event::Crunch(2.0), Event::Splash(2.0)];
            let x = render(p, &ev, 1.5);
            assert!(x.iter().all(|s| s[0].is_finite() && s[1].is_finite() && s[0].abs() <= 1.0 && s[1].abs() <= 1.0));
        }
    }

    #[test]
    fn quieter_and_duller_far_away() {
        let near = render(flying(), &[], 1.0);
        let far = render(Params { dist: 400.0, ..flying() }, &[], 1.0);
        let d = db(rms(&far), rms(&near));
        assert!(d < -14.0, "400 m is only {d:.1} dB down");
        assert!(bright(&far) < bright(&near) * 0.8);
    }

    #[test]
    fn cabin_is_muffled() {
        let out = render(flying(), &[], 1.0);
        let inside = render(Params { cabin: 1.0, ..flying() }, &[], 1.0);
        assert!(bright(&inside) < bright(&out) * 0.75, "inside {} outside {}", bright(&inside), bright(&out));
        let d = db(rms(&inside), rms(&out));
        assert!(d.abs() < 8.0, "cabin level off by {d:.1} dB");
    }

    #[test]
    fn engine_follows_the_throttle() {
        let idle = render(Params { rpm: 700.0, power: 0.0, throttle: 0.0, airspeed: 0.0, ..flying() }, &[], 1.0);
        let full = render(Params { rpm: 2400.0, power: 1.0, throttle: 1.0, airspeed: 0.0, ..flying() }, &[], 1.0);
        let d = db(rms(&full), rms(&idle));
        assert!(d > 6.0, "full power only {d:.1} dB over idle");
    }

    #[test]
    fn jet_spools_up_louder_than_the_single() {
        let jet = |n1: f32, power: f32, dist: f32, cabin: f32| Params { jet: 1.0, n1, rpm: n1 * 3600.0, power, throttle: power, airspeed: 0.0, dist, cabin, vne: 85.0, ..flying() };
        let idle = rms(&render(jet(0.25, 0.035, 85.0, 0.0), &[], 1.0));
        let full = rms(&render(jet(1.0, 1.0, 85.0, 0.0), &[], 1.0));
        let deck = rms(&render(jet(1.0, 1.0, 85.0, 1.0), &[], 1.0));
        let single = rms(&render(Params { rpm: 2400.0, power: 1.0, throttle: 1.0, airspeed: 0.0, ..flying() }, &[], 1.0));
        println!("jet idle {:.1} dB, full {:.1} dB, on the flight deck {:.1} dB, the single flat out {:.1} dB (rms re 1)", db(idle, 1.0), db(full, 1.0), db(deck, 1.0), db(single, 1.0));
        assert!(db(full, idle) > 8.0);
        assert!(full > single * 0.8);
        assert!(deck < full);
    }

    #[test]
    fn fighter_roars_and_lights_its_burners() {
        let f18 = |n1: f32, power: f32, burner: f32, cabin: f32| Params { jet: 1.0, fighter: 1.0, burner, n1, rpm: n1 * 13_200.0, power, throttle: power, airspeed: 0.0, dist: 26.0, cabin, vne: 170.0, ..flying() };
        let airliner = rms(&render(Params { jet: 1.0, n1: 1.0, rpm: 3600.0, power: 1.0, throttle: 1.0, airspeed: 0.0, dist: 26.0, vne: 85.0, ..flying() }, &[], 1.0));
        let idle = rms(&render(f18(0.62, 0.06, 0.0, 0.0), &[], 1.0));
        let mil = rms(&render(f18(1.0, 1.0, 0.0, 0.0), &[], 1.0));
        let wet = rms(&render(f18(1.0, 1.8, 1.0, 0.0), &[], 1.0));
        let seat = rms(&render(f18(1.0, 1.0, 0.0, 1.0), &[], 1.0));
        let seat_wet = rms(&render(f18(1.0, 1.8, 1.0, 1.0), &[], 1.0));
        println!("fighter idle {:.1} dB, military {:.1} dB, afterburner {:.1} dB, in the seat {:.1} / {:.1} dB; an airliner flat out at the same distance {:.1} dB (rms re 1)", db(idle, 1.0), db(mil, 1.0), db(wet, 1.0), db(seat, 1.0), db(seat_wet, 1.0), db(airliner, 1.0));
        assert!(db(mil, idle) > 8.0, "military power only {:.1} dB over idle", db(mil, idle));
        assert!(db(wet, mil) > 2.0, "the afterburner adds only {:.1} dB", db(wet, mil));
        assert!(seat < mil && seat_wet > seat);
        // The thump as they light: louder than the settled roar for a moment.
        let shared = Arc::new(Shared::default());
        *shared.params.lock() = f18(1.0, 1.0, 0.0, 0.0);
        let mut s = Synth::new(shared.clone());
        s.p = f18(1.0, 1.0, 0.0, 0.0);
        let mut take = |s: &mut Synth, secs: f32| -> Vec<[f32; 2]> { (0..(secs * RATE as f32) as usize).map(|_| [s.next().unwrap(), s.next().unwrap()]).collect() };
        take(&mut s, 1.0);
        *shared.params.lock() = f18(1.0, 1.8, 1.0, 0.0);
        let x = take(&mut s, 2.5);
        assert!(x.iter().all(|v| v[0].is_finite() && v[1].is_finite() && v[0].abs() <= 1.0 && v[1].abs() <= 1.0));
        assert!(s.fighter.light.is_none() && s.fighter.lit, "the light-off thump plays once and ends");
    }

    #[test]
    fn warnings_sound() {
        let base = Params { airspeed: 40.0, rpm: 0.0, ..flying() };
        let quiet = rms(&render(base, &[], 1.0));
        for w in [Warn { stall: 1.0, ..Default::default() }, Warn { pull_up: true, ..Default::default() }, Warn { dive: 0.5, ..Default::default() }, Warn { too_low: true, ..Default::default() }, Warn { sink: true, ..Default::default() }, Warn { overspeed: 1.0, ..Default::default() }, Warn { gear: true, ..Default::default() }] {
            let loud = rms(&render(Params { warn: w, ..base }, &[], 1.0));
            assert!(db(loud, quiet) > 3.0, "{w:?} barely audible: {:.1} dB", db(loud, quiet));
        }
    }

    /// `AUDIO_DEMO=/path/demo.wav cargo test audio_demo -- --nocapture` writes a tour of the sounds.
    #[test]
    fn audio_demo() {
        let Ok(path) = std::env::var("AUDIO_DEMO") else { return };
        let f = flying();
        let scenes: Vec<(&str, Params, Vec<Event>, f32)> = vec![
            ("idle, outside", Params { rpm: 700.0, power: 0.05, throttle: 0.0, airspeed: 0.0, ..f }, vec![], 3.0),
            ("idle, cockpit", Params { rpm: 700.0, power: 0.05, throttle: 0.0, airspeed: 0.0, cabin: 1.0, ..f }, vec![], 3.0),
            ("takeoff roll, outside", Params { rpm: 2400.0, power: 1.0, throttle: 1.0, airspeed: 25.0, rolling: 25.0, gear: 1.0, ..f }, vec![], 3.0),
            ("cruise, outside 16 m", f, vec![], 3.0),
            ("cruise, outside 80 m", Params { dist: 80.0, ..f }, vec![], 3.0),
            ("cruise, outside 400 m", Params { dist: 400.0, ..f }, vec![], 3.0),
            ("cruise, cockpit", Params { cabin: 1.0, ..f }, vec![], 3.0),
            ("throttle chop pops", Params { power: 0.9, throttle: 0.1, ..f }, vec![], 2.0),
            ("rough engine, cockpit", Params { health: 0.4, prop: 0.6, cabin: 1.0, ..f }, vec![], 3.0),
            ("storm turbulence, cockpit", Params { rough: 4.0, jolt: 0.6, cabin: 1.0, ..f }, vec![Event::Bump(1.5), Event::Bump(0.8)], 3.0),
            ("overspeed dive, cockpit", Params { airspeed: 105.0, rpm: 2600.0, cabin: 1.0, g_load: 0.6, warn: Warn { dive: 0.8, overspeed: 0.8, ..Default::default() }, ..f }, vec![], 3.0),
            ("overspeed dive, outside", Params { airspeed: 105.0, rpm: 2600.0, warn: Warn { dive: 0.8, overspeed: 0.8, ..Default::default() }, ..f }, vec![], 3.0),
            ("pull up", Params { warn: Warn { pull_up: true, ..Default::default() }, cabin: 1.0, ..f }, vec![], 2.0),
            ("too low", Params { warn: Warn { too_low: true, ..Default::default() }, cabin: 1.0, ..f }, vec![], 2.0),
            ("stall buffet + horn", Params { airspeed: 28.0, buffet: 0.8, rpm: 1500.0, power: 0.3, cabin: 1.0, warn: Warn { stall: 0.9, ..Default::default() }, ..f }, vec![], 3.0),
            ("gear down + clunk", Params { gear: 0.6, gear_moving: 1.0, cabin: 1.0, ..f }, vec![Event::Clunk(1.0)], 2.0),
            ("touchdown", Params { rpm: 900.0, power: 0.1, throttle: 0.0, airspeed: 30.0, rolling: 30.0, gear: 1.0, ..f }, vec![Event::Touch(2.0)], 2.5),
            ("structural failure", Params { cabin: 1.0, ..f }, vec![Event::Snap(1.0)], 2.0),
            ("tree strike", f, vec![Event::Crunch(1.0)], 2.0),
            ("crash", Params { live: 0.0, fire: 0.8, ..f }, vec![Event::Boom(1.5)], 5.0),
            ("crash into water", Params { live: 0.0, ..f }, vec![Event::Splash(1.5)], 5.0),
        ];
        let mut out: Vec<i16> = Vec::new();
        for (name, p, ev, secs) in scenes {
            let x = render(p, &ev, secs);
            println!("{name:28} rms {:6.3}  peak {:5.3}  bright {:5.3}", rms(&x), x.iter().map(|s| s[0].abs().max(s[1].abs())).fold(0.0, f32::max), bright(&x));
            for (k, s) in x.iter().enumerate() {
                // Short fades so scenes do not click into each other.
                let fade = (k as f32 / 2000.0).min(1.0).min((x.len() - k) as f32 / 2000.0);
                out.push((s[0] * fade * 32767.0) as i16);
                out.push((s[1] * fade * 32767.0) as i16);
            }
            out.extend(std::iter::repeat_n(0i16, RATE as usize));
        }
        let mut w = Vec::new();
        let data = (out.len() * 2) as u32;
        w.extend_from_slice(b"RIFF");
        w.extend_from_slice(&(36 + data).to_le_bytes());
        w.extend_from_slice(b"WAVEfmt ");
        w.extend_from_slice(&16u32.to_le_bytes());
        w.extend_from_slice(&1u16.to_le_bytes());
        w.extend_from_slice(&2u16.to_le_bytes());
        w.extend_from_slice(&RATE.to_le_bytes());
        w.extend_from_slice(&(RATE * 4).to_le_bytes());
        w.extend_from_slice(&4u16.to_le_bytes());
        w.extend_from_slice(&16u16.to_le_bytes());
        w.extend_from_slice(b"data");
        w.extend_from_slice(&data.to_le_bytes());
        for s in out {
            w.extend_from_slice(&s.to_le_bytes());
        }
        std::fs::write(&path, w).unwrap();
    }
}
