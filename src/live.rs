//! Real air traffic: the aircraft that ADS-B receivers report around the
//! player right now, from adsb.lol's open data (ODbL). A background thread
//! asks for them every few seconds; in between, each one flies on along its
//! reported track, speed and climb rate, and a new report blends in rather
//! than making it jump. They are drawn with the airport traffic's models
//! (the nearest of five sizes) and carry their navigation lights.

use crate::airfield::{Light, L_GREEN, L_RED, L_STROBE, L_WHITE};
use crate::airmodels::{self as am, TYPES};
use crate::render::Inst;
use crate::{geo, sources::Sources};
use glam::{DVec3, Vec3};
use parking_lot::Mutex;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

/// How far around the player to ask (nautical miles), and how often.
const RADIUS_NM: u32 = 60;
const PERIOD: Duration = Duration::from_secs(5);
/// An aircraft no longer reported is dropped after this long (s).
const FORGET: f64 = 20.0;
/// How long past its last report it is flown on (s); then it waits.
const COAST: f64 = 25.0;
/// Drawn up to this far from the camera (m), lights and labels nearer.
pub const DRAW_DIST: f64 = 60_000.0;
const LIGHT_DIST: f64 = 20_000.0;

/// One aircraft as reported.
#[derive(Clone, Debug, PartialEq)]
struct Report {
    hex: String,
    call: String,
    ty: String,
    size: u8,
    lat: f64,
    lon: f64,
    alt_m: f64,
    /// ground speed (m/s), track (deg), vertical speed (m/s)
    gs: f64,
    trk: f64,
    vs: f64,
    /// how old the position was when it was sent (s)
    age: f64,
}

pub struct Plane {
    pub hex: String,
    /// callsign ("AFR77BJ"), else the registration or the ICAO address
    pub call: String,
    /// ICAO type code ("A20N"), may be empty
    pub ty: String,
    pub size: u8,
    /// position at the last report, velocity (m/s, world axes), and when
    fix: DVec3,
    vel: DVec3,
    at: Instant,
    /// what is left of the jump at the last report, faded out
    off: DVec3,
    seen: Instant,
    livery: (u8, u8),
    /// where it is drawn now, and its nose direction (smoothed)
    pub pos: DVec3,
    pub fwd: DVec3,
    pub alt_ft: f64,
    pub gs_kt: f64,
    pub vs_fpm: f64,
}

#[derive(Default)]
struct Shared {
    /// where to look (lat, lon in degrees); None while in the menus
    want: Option<(f64, f64)>,
    /// the latest answer: when it came, and what it said
    got: Option<(Instant, Vec<Report>)>,
    /// the last error, for the settings page
    error: String,
}

pub struct Live {
    shared: Arc<Mutex<Shared>>,
    on: Arc<AtomicBool>,
    thread: bool,
    last_got: Option<Instant>,
    last_step: Option<Instant>,
    pub planes: Vec<Plane>,
}

impl Live {
    pub fn new(on: bool) -> Self {
        Self { shared: Arc::default(), on: Arc::new(AtomicBool::new(on)), thread: false, last_got: None, last_step: None, planes: Vec::new() }
    }

    pub fn enabled(&self) -> bool {
        self.on.load(Ordering::Relaxed)
    }

    pub fn set_enabled(&mut self, on: bool) {
        self.on.store(on, Ordering::Relaxed);
        if !on {
            self.planes.clear();
            self.shared.lock().got = None;
        }
    }

    /// The last problem fetching, if the last fetch failed.
    pub fn error(&self) -> String {
        self.shared.lock().error.clone()
    }

    /// Each frame: where the player is (None in the menus), then fly the
    /// aircraft on to now, taking in a new report if one has come.
    pub fn step(&mut self, src: &Arc<Sources>, at: Option<DVec3>) {
        let now = Instant::now();
        let dt = self.last_step.map(|t| (now - t).as_secs_f64()).unwrap_or(0.0).min(0.5);
        self.last_step = Some(now);
        if !self.enabled() {
            return;
        }
        let want = at.map(|p| {
            let (lat, lon, _) = geo::pos_to_lla(p);
            (lat.to_degrees(), lon.to_degrees())
        });
        let got = {
            let mut s = self.shared.lock();
            s.want = want;
            s.got.as_ref().filter(|(t, _)| Some(*t) != self.last_got).cloned()
        };
        if !self.thread && want.is_some() {
            self.thread = true;
            spawn(src.clone(), self.shared.clone(), self.on.clone());
        }
        if let Some((t, reports)) = got {
            self.last_got = Some(t);
            self.take(t, &reports);
        }
        self.planes.retain(|p| (now - p.seen).as_secs_f64() < FORGET);
        let fade = (-dt / 1.5).exp();
        let turn = (dt * 1.5).min(1.0);
        for p in &mut self.planes {
            p.off *= fade;
            let coast = (now - p.at).as_secs_f64().min(COAST);
            p.pos = p.fix + p.vel * coast + p.off;
            if p.vel.length() > 5.0 {
                p.fwd = (p.fwd + (p.vel.normalize() - p.fwd) * turn).normalize();
            }
        }
    }

    /// A new set of reports, received at `t`.
    fn take(&mut self, t: Instant, reports: &[Report]) {
        let now = Instant::now();
        for r in reports {
            let at = t.checked_sub(Duration::from_secs_f64(r.age.clamp(0.0, 60.0))).unwrap_or(t);
            let fix = geo::lla_to_pos(r.lat.to_radians(), r.lon.to_radians(), r.alt_m);
            let (e, n, u) = geo::enu(fix);
            let (s, c) = r.trk.to_radians().sin_cos();
            let vel = (e * s + n * c) * r.gs + u * r.vs;
            let predicted = fix + vel * (now - at).as_secs_f64().min(COAST);
            let (alt_ft, gs_kt, vs_fpm) = (r.alt_m * 3.28084, r.gs * 1.94384, r.vs * 196.85);
            if let Some(p) = self.planes.iter_mut().find(|p| p.hex == r.hex) {
                // Keep it where it was drawn and let the difference fade.
                let jump = p.pos - predicted;
                p.off = if jump.length() < 3000.0 { jump } else { DVec3::ZERO };
                (p.fix, p.vel, p.at, p.seen) = (fix, vel, at, now);
                (p.call, p.ty, p.size) = (r.call.clone(), r.ty.clone(), r.size);
                (p.alt_ft, p.gs_kt, p.vs_fpm) = (alt_ft, gs_kt, vs_fpm);
            } else {
                let h = r.hex.bytes().fold(2166136261u32, |h, b| (h ^ b as u32).wrapping_mul(16777619));
                let fwd = if vel.length() > 1.0 { vel.normalize() } else { n };
                self.planes.push(Plane {
                    hex: r.hex.clone(),
                    call: r.call.clone(),
                    ty: r.ty.clone(),
                    size: r.size,
                    fix,
                    vel,
                    at,
                    off: DVec3::ZERO,
                    seen: now,
                    livery: ((h % crate::traffic::BODY.len() as u32) as u8, ((h >> 8) % crate::traffic::FIN.len() as u32) as u8),
                    pos: predicted,
                    fwd,
                    alt_ft,
                    gs_kt,
                    vs_fpm,
                });
            }
        }
    }

    /// The aircraft as instanced models, and their lights.
    pub fn instances(&self, cam: DVec3, out: &mut Vec<(u32, Inst)>, lights: &mut Vec<Light>) {
        for (i, p) in self.planes.iter().enumerate() {
            let d = (p.pos - cam).length();
            if d > DRAW_DIST {
                continue;
            }
            let up = p.pos.normalize();
            let fwd = p.fwd;
            let upv = (up - fwd * fwd.dot(up)).normalize_or_zero();
            if upv == DVec3::ZERO {
                continue;
            }
            let (x, y, z) = (fwd.cross(upv).as_vec3(), upv.as_vec3(), (-fwd).as_vec3());
            let o = (p.pos - cam).as_vec3();
            let inst = |tint: [f32; 3]| Inst { m: [[x.x, y.x, z.x, o.x], [x.y, y.y, z.y, o.y], [x.z, y.z, z.z, o.z]], tint: [tint[0], tint[1], tint[2], 0.0], ext: [0.0; 4] };
            let base = am::M_BODY + 3 * p.size as u32;
            out.push((base, inst(crate::traffic::BODY[p.livery.0 as usize])));
            out.push((base + 1, inst(crate::traffic::FIN[p.livery.1 as usize])));
            let gear = p.alt_ft < 2500.0 || (p.alt_ft < 5000.0 && p.vs_fpm < -300.0);
            if gear {
                out.push((base + 2, inst([0.5; 3])));
            }
            if d > LIGHT_DIST {
                continue;
            }
            let ty = TYPES[p.size as usize];
            let at = |q: Vec3| -> [f32; 3] { (o + x * q.x + y * q.y - z * q.z).into() };
            let mut l = |q: Vec3, inten: f32, col: [u8; 3], fx: [u8; 4]| {
                if lights.len() < crate::render::MAX_DYN_LIGHTS {
                    lights.push(Light { pos: at(q), inten, dir: [0.0; 3], spread: 0.0, col: [col[0], col[1], col[2], 0], fx });
                }
            };
            let steady = [0, 0, 255, 1];
            let ph = ((i * 53) % 255) as u8;
            l(Vec3::new(-ty.tip.x, ty.tip.y, -ty.tip.z), 40.0, L_RED, steady);
            l(Vec3::new(ty.tip.x, ty.tip.y, -ty.tip.z), 40.0, L_GREEN, steady);
            l(Vec3::new(0.0, ty.tail.y * 0.35, -ty.tail.z), 25.0, L_WHITE, steady);
            l(Vec3::new(0.0, ty.cy + ty.rf + 0.1, 0.0), 120.0, L_RED, [12, ph, 25, 1]);
            for sx in [-1.0f32, 1.0] {
                l(Vec3::new(sx * ty.tip.x, ty.tip.y, -ty.tip.z), 1600.0, L_STROBE, [11, ph.wrapping_add(60), 7, 1]);
            }
        }
    }
}

/// The background thread: every few seconds, while switched on and the
/// player is flying, the aircraft around them.
fn spawn(src: Arc<Sources>, shared: Arc<Mutex<Shared>>, on: Arc<AtomicBool>) {
    std::thread::spawn(move || {
        loop {
            let started = Instant::now();
            let want = shared.lock().want;
            if let (true, Some((lat, lon))) = (on.load(Ordering::Relaxed), want) {
                let url = format!("https://api.adsb.lol/v2/point/{lat:.4}/{lon:.4}/{RADIUS_NM}");
                let res = src.fetch.get(&url).map_err(|e| e.to_string()).and_then(|d| parse(&d).ok_or_else(|| "unreadable answer".to_string()));
                let mut s = shared.lock();
                match res {
                    Ok(r) => {
                        s.got = Some((Instant::now(), r));
                        s.error.clear();
                    }
                    Err(e) => {
                        if e != "offline" {
                            log::warn!("live traffic: {e}");
                        }
                        s.error = e;
                    }
                }
            }
            std::thread::sleep(PERIOD.saturating_sub(started.elapsed()).max(Duration::from_millis(500)));
        }
    });
}

/// adsb.lol's v2 answer: `{"ac": [{"hex", "flight", "t", "category", "lat",
/// "lon", "alt_baro", "alt_geom", "gs", "track", "geom_rate", "seen_pos"}…]}`.
/// Aircraft on the ground, or without a position, are left out.
fn parse(data: &[u8]) -> Option<Vec<Report>> {
    let j: serde_json::Value = serde_json::from_slice(data).ok()?;
    let ac = j.get("ac")?.as_array()?;
    Some(
        ac.iter()
            .filter_map(|a| {
                let num = |k: &str| a.get(k).and_then(|v| v.as_f64());
                let text = |k: &str| a.get(k).and_then(|v| v.as_str()).map(|s| s.trim().to_string()).filter(|s| !s.is_empty());
                if a.get("alt_baro").and_then(|v| v.as_str()) == Some("ground") {
                    return None;
                }
                let hex = text("hex")?;
                let ty = text("t").unwrap_or_default();
                let cat = text("category").unwrap_or_default();
                Some(Report {
                    call: text("flight").or_else(|| text("r")).unwrap_or_else(|| hex.to_uppercase()),
                    size: size_of(&ty, &cat),
                    lat: num("lat")?,
                    lon: num("lon")?,
                    alt_m: num("alt_geom").or_else(|| num("alt_baro"))? / 3.28084,
                    gs: num("gs").unwrap_or(0.0) / 1.94384,
                    trk: num("track").or_else(|| num("true_heading")).unwrap_or(0.0),
                    vs: num("geom_rate").or_else(|| num("baro_rate")).unwrap_or(0.0) / 196.85,
                    age: num("seen_pos").unwrap_or(0.0),
                    hex,
                    ty,
                })
            })
            .collect(),
    )
}

/// Which of the five traffic models looks most like it: 0 light single,
/// 1 business jet, 2 regional, 3 narrow-body, 4 wide-body. From the ICAO
/// type code when it is a known family, else the ADS-B emitter category.
fn size_of(ty: &str, cat: &str) -> u8 {
    const WIDE: &[&str] = &["A30", "A31", "A33", "A34", "A35", "A38", "B74", "B76", "B77", "B78", "MD11", "DC10", "IL96", "A124", "A225", "C5M", "KC10"];
    const NARROW: &[&str] = &["A318", "A319", "A32", "A20N", "A21N", "A19N", "B73", "B37M", "B38M", "B39M", "B3XM", "B75", "BCS", "MD8", "MD9", "B71", "C919", "T204", "SU95"];
    const REGIONAL: &[&str] = &["E1", "E2", "E7", "CRJ", "AT4", "AT7", "DH8", "SF34", "SB20", "F70", "F100", "RJ", "B46", "J328", "D328"];
    const BIZ: &[&str] = &["C25", "C5", "C6", "C7", "CL30", "CL35", "CL60", "GLF", "GL5", "GL7", "GLEX", "FA", "LJ", "E50", "E55", "PC24", "H25", "HDJT", "PRM", "SF50", "BE40", "G150", "G280"];
    let has = |l: &[&str]| l.iter().any(|p| ty.starts_with(p));
    // (The narrow-bodies go first: "A318"/"A319" are listed whole so that the
    // A310 and A300 fall through to the wide-bodies.)
    if has(NARROW) {
        3
    } else if has(WIDE) {
        4
    } else if has(REGIONAL) {
        2
    } else if has(BIZ) {
        1
    } else {
        match cat {
            "A5" => 4,
            "A3" | "A4" => 3,
            "A2" => 1,
            _ => 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes() {
        assert_eq!(size_of("A20N", "A3"), 3);
        assert_eq!(size_of("A319", "A3"), 3);
        assert_eq!(size_of("A333", "A5"), 4);
        assert_eq!(size_of("B77W", "A5"), 4);
        assert_eq!(size_of("B38M", "A3"), 3);
        assert_eq!(size_of("E295", "A3"), 2);
        assert_eq!(size_of("CRJ9", "A3"), 2);
        assert_eq!(size_of("C56X", "A2"), 1);
        assert_eq!(size_of("C172", "A1"), 0);
        assert_eq!(size_of("", "A5"), 4);
        assert_eq!(size_of("", ""), 0);
    }

    #[test]
    fn reads_adsb_lol() {
        let d = br#"{"ac":[
            {"hex":"486494","flight":"KLM44T  ","t":"E295","category":"A3","lat":51.1,"lon":3.5,"alt_baro":24050,"alt_geom":23575,"gs":446.3,"track":41.5,"geom_rate":-1632,"seen_pos":0.4},
            {"hex":"4ca1b2","t":"A320","alt_baro":"ground","lat":50.9,"lon":4.48,"gs":3},
            {"hex":"aaaaaa","flight":"NOPOS"},
            {"hex":"3c6444","r":"D-ABCD","alt_baro":3000,"lat":50.0,"lon":4.0}
        ],"msg":"No error"}"#;
        let r = parse(d).unwrap();
        assert_eq!(r.len(), 2);
        assert_eq!((r[0].call.as_str(), r[0].size), ("KLM44T", 2));
        assert!((r[0].alt_m - 23575.0 / 3.28084).abs() < 0.01);
        assert!((r[0].vs + 1632.0 / 196.85).abs() < 0.01);
        assert_eq!(r[1].call, "D-ABCD");
    }
}
