//! Online multiplayer: fly with friends by typing the same room code.
//!
//! The players' games do not talk to each other directly: each opens a
//! WebSocket to a small relay (a Cloudflare Worker, see `server/`), which
//! passes whatever one player sends on to the others in the same room. A
//! background thread keeps that connection (reconnecting after a drop, with
//! a growing pause), so the frame never waits on the network: the main thread
//! only trades messages with it over channels.
//!
//! Each game sends its own aircraft ten times a second as a short JSON text
//! (`State::encode`). The others are drawn a little in the past (`DELAY`),
//! between the two reports around that moment, so they move smoothly however
//! the reports arrive; past the newest one they fly on along its velocity for
//! a moment. A player not heard from in `SILENT` seconds, or reported gone by
//! the room, is dropped.
//!
//! They are drawn with the real model of the aircraft they fly (`Models`:
//! each type's pieces merged into a few meshes, built once on a worker
//! thread), with their navigation lights, a name tag and an orange mark on
//! the maps. One in the speedboat is drawn with the boat's own meshes (the
//! instanced solids), and the nearest such leaves its wake while we are not
//! in the boat ourselves (the shaders have room for one boat's wake).

use crate::airfield::{Light, L_GREEN, L_RED, L_STROBE, L_WHITE};
use crate::i18n::t;
use crate::render::{DrawObj, Inst, Renderer};
use crate::{aircraft, airframe, boat, geo};
use crossbeam_channel::{Receiver, Sender, TryRecvError};
use glam::{DMat4, DQuat, DVec3, Mat4, Vec3};
use std::collections::VecDeque;
use std::net::{TcpStream, ToSocketAddrs};
use std::time::{Duration, Instant};

/// The relay the game uses unless the settings or `--mp-server` name another.
/// PLACEHOLDER: after `npx wrangler deploy` (server/README.md), put the
/// address it prints here, with wss:// in place of https://.
pub const MP_SERVER: &str = "wss://fableflight-mp.fableflight-mp.workers.dev";
/// Own state sent this often (s).
const SEND_EVERY: f64 = 0.1;
/// Others are drawn this far in the past (s), between two reports.
const DELAY: f64 = 0.12;
/// Flown on along the last velocity at most this long past the newest report (s).
const COAST: f64 = 1.0;
/// Dropped when not heard from for this long (s).
const SILENT: f64 = 5.0;
/// Reports kept per player.
const KEEP: usize = 4;
/// Drawn up to this far from the camera (m).
pub const DRAW_DIST: f64 = 120_000.0;
const LIGHT_DIST: f64 = 25_000.0;
/// Room codes: 4 to 8 letters or digits.
pub const CODE_MAX: usize = 8;

/// The vehicle number of the speedboat: the aircraft are `aircraft::ALL`'s
/// indices, the boat comes after them.
pub fn boat_id() -> usize {
    aircraft::ALL.len()
}

/// A vehicle number's name, as shown.
pub fn vehicle_name(ac: usize) -> &'static str {
    if ac >= boat_id() { boat::NAME } else { aircraft::ALL[ac].name }
}

/// How high over a vehicle's centre its name tag sits (m), clear of it however big.
pub fn tag_height(ac: usize) -> f64 {
    if ac >= boat_id() { 3.0 } else { aircraft::ALL[ac].b * 0.3 }
}

/// One player's aircraft at one moment, as sent.
#[derive(Clone, Debug, PartialEq)]
pub struct State {
    pub name: String,
    /// vehicle: index into `aircraft::ALL`, or `boat_id()` for the speedboat
    pub ac: u8,
    /// degrees, degrees, metres above the sphere
    pub lat: f64,
    pub lon: f64,
    pub alt: f64,
    /// body to world rotation (x, y, z, w)
    pub q: [f32; 4],
    /// world velocity (m/s)
    pub vel: [f32; 3],
    /// gear (0 up .. 1 down), flaps (0..1), throttle (0..1)
    pub gear: f32,
    pub flaps: f32,
    pub thr: f32,
    pub crashed: bool,
    /// firing its lasers (Destruction mode), drawn for the others
    pub laser: bool,
    /// the sender's own clock (s since its game started)
    pub t: f64,
}

impl State {
    /// `{"n":"Sam","a":0,"p":[lat,lon,alt],"q":[x,y,z,w],"v":[x,y,z],"g":1,"f":0.33,"h":0.8,"c":0,"ts":12.3}`
    pub fn encode(&self) -> String {
        let name = serde_json::to_string(&self.name).unwrap_or_else(|_| "\"\"".into());
        let [qx, qy, qz, qw] = self.q;
        let [vx, vy, vz] = self.vel;
        format!(
            "{{\"n\":{name},\"a\":{},\"p\":[{:.7},{:.7},{:.2}],\"q\":[{qx:.5},{qy:.5},{qz:.5},{qw:.5}],\"v\":[{vx:.2},{vy:.2},{vz:.2}],\"g\":{:.2},\"f\":{:.2},\"h\":{:.2},\"c\":{},\"l\":{},\"ts\":{:.3}}}",
            self.ac,
            self.lat,
            self.lon,
            self.alt,
            self.gear,
            self.flaps,
            self.thr,
            self.crashed as u8,
            self.laser as u8,
            self.t
        )
    }

    pub fn decode(j: &serde_json::Value) -> Option<State> {
        let arr = |k: &str, n: usize| -> Option<Vec<f64>> {
            let a = j.get(k)?.as_array()?;
            (a.len() == n).then(|| a.iter().map(|x| x.as_f64()).collect::<Option<Vec<f64>>>())?.filter(|v| v.iter().all(|x| x.is_finite()))
        };
        let num = |k: &str| j.get(k).and_then(|v| v.as_f64()).filter(|x| x.is_finite());
        let p = arr("p", 3)?;
        let q = arr("q", 4)?;
        let v = arr("v", 3)?;
        Some(State {
            name: j.get("n").and_then(|v| v.as_str()).map(clean_name).unwrap_or_default(),
            ac: (num("a").unwrap_or(0.0) as usize).min(boat_id()) as u8,
            lat: p[0].clamp(-90.0, 90.0),
            lon: p[1],
            alt: p[2].clamp(-1000.0, 200_000.0),
            q: [q[0] as f32, q[1] as f32, q[2] as f32, q[3] as f32],
            vel: [v[0] as f32, v[1] as f32, v[2] as f32],
            gear: num("g").unwrap_or(1.0).clamp(0.0, 1.0) as f32,
            flaps: num("f").unwrap_or(0.0).clamp(0.0, 1.0) as f32,
            thr: num("h").unwrap_or(0.0).clamp(0.0, 1.0) as f32,
            crashed: num("c").unwrap_or(0.0) > 0.5,
            laser: num("l").unwrap_or(0.0) > 0.5,
            t: num("ts")?,
        })
    }

    fn snap(&self) -> Snap {
        let q = DQuat::from_xyzw(self.q[0] as f64, self.q[1] as f64, self.q[2] as f64, self.q[3] as f64);
        Snap {
            t: self.t,
            pos: geo::lla_to_pos(self.lat.to_radians(), self.lon.to_radians(), self.alt),
            rot: if q.length_squared() > 0.25 { q.normalize() } else { DQuat::IDENTITY },
            vel: DVec3::new(self.vel[0] as f64, self.vel[1] as f64, self.vel[2] as f64),
            gear: self.gear,
            thr: self.thr,
        }
    }
}

/// A name as shown: printable, trimmed, at most 20 characters.
pub fn clean_name(s: &str) -> String {
    s.chars().filter(|c| !c.is_control()).take(20).collect::<String>().trim().to_string()
}

/// A room code as typed: upper case letters and digits only, at most `CODE_MAX`.
pub fn clean_code(s: &str) -> String {
    s.chars().filter(|c| c.is_ascii_alphanumeric()).map(|c| c.to_ascii_uppercase()).take(CODE_MAX).collect()
}

/// A new room's code: four letters (without I and O, which read as digits).
pub fn random_code() -> String {
    const L: &[u8] = b"ABCDEFGHJKLMNPQRSTUVWXYZ";
    let mut h = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(1) ^ (std::process::id() as u64) << 32;
    (0..4)
        .map(|_| {
            h ^= h << 13;
            h ^= h >> 7;
            h ^= h << 17;
            L[(h % L.len() as u64) as usize] as char
        })
        .collect()
}

/// The room's WebSocket address on a server (`wss://host`, `https://host`, `ws://127.0.0.1:8787`...).
pub fn room_url(server: &str, code: &str) -> String {
    let s = server.trim().trim_end_matches('/');
    let s = if let Some(r) = s.strip_prefix("https://") {
        format!("wss://{r}")
    } else if let Some(r) = s.strip_prefix("http://") {
        format!("ws://{r}")
    } else if s.starts_with("ws://") || s.starts_with("wss://") {
        s.to_string()
    } else {
        format!("wss://{s}")
    };
    format!("{s}/room/{code}")
}

/// A report as kept: where, which way, how fast (world frame), at the sender's time.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Snap {
    t: f64,
    pos: DVec3,
    rot: DQuat,
    vel: DVec3,
    gear: f32,
    /// throttle (0..1): the boat's wake
    thr: f32,
}

/// Where a player is at the sender's time `t`, from the reports kept (oldest
/// first): between the two around it (a cubic through both positions with
/// their velocities, the rotation turned evenly), before the first held
/// there, past the last flown on along its velocity (at most `COAST`).
fn sample(snaps: &[Snap], t: f64) -> Option<(DVec3, DQuat, DVec3, f32)> {
    let first = snaps.first()?;
    let last = snaps.last()?;
    if t <= first.t {
        return Some((first.pos, first.rot, first.vel, first.gear));
    }
    if t >= last.t {
        let dt = (t - last.t).min(COAST);
        return Some((last.pos + last.vel * dt, last.rot, last.vel, last.gear));
    }
    let k = snaps.windows(2).position(|w| t >= w[0].t && t <= w[1].t)?;
    let (a, b) = (&snaps[k], &snaps[k + 1]);
    let h = b.t - a.t;
    if h <= 1e-6 {
        return Some((b.pos, b.rot, b.vel, b.gear));
    }
    let s = (t - a.t) / h;
    let (s2, s3) = (s * s, s * s * s);
    let pos = a.pos * (2.0 * s3 - 3.0 * s2 + 1.0) + a.vel * (h * (s3 - 2.0 * s2 + s)) + b.pos * (-2.0 * s3 + 3.0 * s2) + b.vel * (h * (s3 - s2));
    let vel = a.vel.lerp(b.vel, s);
    Some((pos, a.rot.slerp(b.rot, s), vel, a.gear + (b.gear - a.gear) * s as f32))
}

/// Another player in the room.
pub struct Peer {
    /// the room's number for them
    pub id: u32,
    pub name: String,
    /// vehicle: index into `aircraft::ALL`, or `boat_id()` for the speedboat
    pub ac: usize,
    pub crashed: bool,
    /// firing their lasers
    pub laser: bool,
    snaps: Vec<Snap>,
    /// their clock minus ours (s), lowest seen (least delayed report), drifting up slowly
    offset: Option<f64>,
    seen: Instant,
    /// where they are drawn now
    pub pos: DVec3,
    pub rot: DQuat,
    pub vel: DVec3,
    pub gear: f32,
    /// in the speedboat: the track its transom left, on our clock (see `boat::Boat::wake`)
    wake: VecDeque<boat::WakePt>,
}

impl Peer {
    fn new(id: u32) -> Self {
        Peer { id, name: String::new(), ac: 0, crashed: false, laser: false, snaps: Vec::new(), offset: None, seen: Instant::now(), pos: DVec3::ZERO, rot: DQuat::IDENTITY, vel: DVec3::ZERO, gear: 1.0, wake: VecDeque::new() }
    }

    pub fn is_boat(&self) -> bool {
        self.ac >= boat_id()
    }

    /// The boat's transom, on the water.
    fn stern(&self) -> DVec3 {
        let up = self.pos.normalize();
        let s = self.pos + self.rot * DVec3::new(0.0, -0.3, boat::STERN);
        s - up * ((s - self.pos).dot(up) + 0.3)
    }

    /// Has a position to be drawn at.
    pub fn placed(&self) -> bool {
        !self.snaps.is_empty()
    }

    fn take(&mut self, s: &State, now: f64) {
        // (their clock restarted: a new game, start over)
        if self.snaps.last().is_some_and(|l| s.t < l.t - 1.0) {
            self.snaps.clear();
            self.offset = None;
        }
        if self.snaps.last().is_some_and(|l| s.t <= l.t) {
            return;
        }
        self.name = s.name.clone();
        if self.ac != s.ac as usize {
            self.wake.clear();
        }
        self.ac = s.ac as usize;
        self.crashed = s.crashed;
        self.laser = s.laser;
        let o = now - s.t;
        self.offset = Some(match self.offset {
            Some(prev) if o >= prev => prev + (o - prev) * 0.02,
            _ => o,
        });
        self.snaps.push(s.snap());
        if self.snaps.len() > KEEP {
            self.snaps.remove(0);
        }
    }

    fn update(&mut self, now: f64) {
        let Some(off) = self.offset else { return };
        if let Some((pos, rot, vel, gear)) = sample(&self.snaps, now - off - DELAY) {
            (self.pos, self.rot, self.vel, self.gear) = (pos, rot, vel, gear);
        }
        if self.is_boat() && !self.crashed {
            // The wake's track, as the boat lays its own (boat.rs).
            let stern = self.stern();
            let spd = self.vel.dot(self.rot * DVec3::NEG_Z).max(0.0);
            let gap = self.wake.back().map(|w| (w.pos - stern).length()).unwrap_or(f64::MAX);
            let since = self.wake.back().map(|w| now - w.t).unwrap_or(f64::MAX);
            if gap > 2.0 + 0.35 * spd || (since > 1.2 && gap > 0.3) {
                let thr = self.snaps.last().map(|s| s.thr).unwrap_or(0.0) as f64;
                self.wake.push_back(boat::WakePt { pos: stern, t: now, speed: spd, power: thr });
            }
        }
        while self.wake.len() > boat::WAKE_N || self.wake.front().is_some_and(|w| now - w.t > boat::WAKE_LIFE) {
            self.wake.pop_front();
        }
    }
}

/// How the connection is doing.
#[derive(Clone, Debug, PartialEq)]
pub enum Net {
    Off,
    Connecting,
    Connected,
    /// the room has its 8 players
    Full,
    /// lost or refused (why), trying again shortly
    Retrying(String),
}

enum Cmd {
    Join(String),
    Leave,
    Send(String),
}

enum Evt {
    Net(Net),
    Welcome(u32, Vec<u32>),
    Join(u32),
    Leave(u32),
    State(u32, State),
}

pub struct Mp {
    tx: Option<Sender<Cmd>>,
    rx: Option<Receiver<Evt>>,
    /// the room joined (None: not in one)
    pub room: Option<String>,
    pub net: Net,
    /// the room's number for us
    pub me: Option<u32>,
    pub peers: Vec<Peer>,
    t0: Instant,
    last_send: f64,
}

impl Default for Mp {
    fn default() -> Self {
        Self::new()
    }
}

impl Mp {
    pub fn new() -> Self {
        Mp { tx: None, rx: None, room: None, net: Net::Off, me: None, peers: Vec::new(), t0: Instant::now(), last_send: -1.0 }
    }

    pub fn active(&self) -> bool {
        self.room.is_some()
    }

    fn now(&self) -> f64 {
        self.t0.elapsed().as_secs_f64()
    }

    /// Joins room `code` on `server` (leaving any other).
    pub fn join(&mut self, server: &str, code: &str) {
        let code = clean_code(code);
        if code.len() < 4 {
            return;
        }
        if self.tx.is_none() {
            let (tx, crx) = crossbeam_channel::unbounded();
            let (etx, rx) = crossbeam_channel::unbounded();
            if let Err(e) = std::thread::Builder::new().name("multiplayer".into()).spawn(move || run(crx, etx)) {
                log::warn!("multiplayer: {e}");
                return;
            }
            (self.tx, self.rx) = (Some(tx), Some(rx));
        }
        self.peers.clear();
        self.me = None;
        self.net = Net::Connecting;
        let url = room_url(server, &code);
        log::info!("multiplayer: joining {url}");
        self.room = Some(code);
        if let Some(tx) = &self.tx {
            let _ = tx.send(Cmd::Join(url));
        }
    }

    pub fn leave(&mut self) {
        if let Some(tx) = &self.tx {
            let _ = tx.send(Cmd::Leave);
        }
        self.room = None;
        self.net = Net::Off;
        self.me = None;
        self.peers.clear();
    }

    /// Each frame: take in what came, send our own state when one is due
    /// (`own` is asked only then; None in the menus), and place the others.
    pub fn step(&mut self, own: impl FnOnce(f64) -> Option<State>) {
        if !self.active() {
            return;
        }
        let now = self.now();
        let rx = self.rx.clone();
        for e in rx.iter().flat_map(|r| r.try_iter()) {
            match e {
                Evt::Net(n) => {
                    if n != Net::Connected {
                        self.peers.clear();
                        self.me = None;
                    }
                    self.net = n;
                }
                Evt::Welcome(me, others) => {
                    log::info!("multiplayer: in the room as player {me}, with {others:?}");
                    self.me = Some(me);
                    self.peers = others.into_iter().map(Peer::new).collect();
                }
                Evt::Join(id) => {
                    log::info!("multiplayer: player {id} joined");
                    if !self.peers.iter().any(|p| p.id == id) {
                        self.peers.push(Peer::new(id));
                    }
                }
                Evt::Leave(id) => {
                    log::info!("multiplayer: player {id} left");
                    self.peers.retain(|p| p.id != id);
                }
                Evt::State(id, s) => {
                    let i = match self.peers.iter().position(|p| p.id == id) {
                        Some(i) => i,
                        None => {
                            self.peers.push(Peer::new(id));
                            self.peers.len() - 1
                        }
                    };
                    self.peers[i].take(&s, now);
                    self.peers[i].seen = Instant::now();
                }
            }
        }
        // (one who joined but never flew is kept: they may be on the globe)
        self.peers.retain(|p| {
            let keep = !p.placed() || p.seen.elapsed().as_secs_f64() < SILENT;
            if !keep {
                log::info!("multiplayer: player {} silent, dropped", p.id);
            }
            keep
        });
        for p in &mut self.peers {
            p.update(now);
        }
        if self.net == Net::Connected && now - self.last_send >= SEND_EVERY {
            self.last_send = now;
            if let (Some(s), Some(tx)) = (own(now), &self.tx) {
                let _ = tx.send(Cmd::Send(s.encode()));
            }
        }
    }

    /// The others on the maps: lat, lon, heading (degrees), name.
    pub fn map_list(&self) -> Vec<(f64, f64, f64, String)> {
        self.peers
            .iter()
            .filter(|p| p.placed())
            .map(|p| {
                let (la, lo, _) = geo::pos_to_lla(p.pos);
                let (e, n, _) = geo::enu(p.pos);
                let f = p.rot * DVec3::NEG_Z;
                (la.to_degrees(), lo.to_degrees(), f.dot(e).atan2(f.dot(n)).to_degrees(), if p.name.is_empty() { "?".to_string() } else { p.name.clone() })
            })
            .collect()
    }

    /// The nearest other speedboat's wake for the shaders, in the rows the
    /// own boat fills (`boat::Boat::wake_rows`, common.wgsl `Globals.wake`):
    /// for when we are not in the boat ourselves.
    pub fn wake_rows(&self, cam: DVec3, rows: &mut [[f32; 4]]) {
        let now = self.now();
        let near = self
            .peers
            .iter()
            .filter(|p| p.placed() && p.is_boat() && !p.crashed && (p.pos - cam).length() < 6000.0)
            .min_by(|a, b| (a.pos - cam).length().total_cmp(&(b.pos - cam).length()));
        let Some(p) = near else { return };
        let n_max = (rows.len() - 1) / 2;
        let thr = p.snaps.last().map(|s| s.thr).unwrap_or(0.0) as f64;
        let mut pts = Vec::with_capacity(n_max);
        pts.push(boat::WakePt { pos: p.stern(), t: now, speed: p.vel.dot(p.rot * DVec3::NEG_Z).max(0.0), power: thr });
        pts.extend(p.wake.iter().rev().copied());
        pts.truncate(n_max);
        rows.fill([0.0; 4]);
        rows[0][0] = pts.len() as f32;
        for (i, w) in pts.iter().enumerate() {
            let q = (w.pos - cam).as_vec3();
            rows[1 + 2 * i] = [q.x, q.y, q.z, (now - w.t) as f32];
            rows[2 + 2 * i] = [w.speed as f32, w.power as f32, 0.0, 0.0];
        }
    }

    /// The connection's state, for the menus and the HUD.
    pub fn status(&self) -> String {
        match &self.net {
            Net::Off => t!("Not connected", "Non connecté").to_string(),
            Net::Connecting => t!("Connecting…", "Connexion…").to_string(),
            Net::Connected => {
                let n = self.peers.len() + 1;
                if n == 1 { t!("Connected, waiting for others", "Connecté, en attente des autres").to_string() } else { t!("Connected, {} players", "Connecté, {} joueurs", n) }
            }
            Net::Full => t!("Room full (8 players): trying again…", "Salle pleine (8 joueurs) : nouvel essai…").to_string(),
            Net::Retrying(why) => t!("Connection lost ({}): retrying…", "Connexion perdue ({}) : nouvel essai…", why),
        }
    }

    /// The others' aircraft (the real models, once built: see `Models`) and
    /// their lights, and their speedboats (instanced solids, as `boat::Boat::instances`).
    pub fn draw(&self, models: &mut Models, r: &mut Renderer, cam: DVec3, objs: &mut Vec<DrawObj>, insts: &mut Vec<(u32, Inst)>, lights: &mut Vec<Light>) {
        models.poll(r);
        for (i, p) in self.peers.iter().enumerate() {
            if !p.placed() || p.crashed {
                continue;
            }
            let d = (p.pos - cam).length();
            if d > DRAW_DIST {
                continue;
            }
            if p.is_boat() {
                if d < 20_000.0 {
                    let inst = |m: DMat4| {
                        let (x, y, z, o) = (m.x_axis, m.y_axis, m.z_axis, m.w_axis - cam.extend(0.0));
                        let r = |i: usize| [x[i] as f32, y[i] as f32, z[i] as f32, o[i] as f32];
                        Inst { m: [r(0), r(1), r(2)], tint: boat::HULL_TINT, ext: [0.0; 4] }
                    };
                    let base = DMat4::from_rotation_translation(p.rot, p.pos);
                    insts.push((boat::MESH, inst(base)));
                    insts.push((boat::MESH + 1, inst(base * DMat4::from_translation(boat::MOTOR))));
                }
                continue;
            }
            let spec = aircraft::ALL[p.ac.min(aircraft::ALL.len() - 1)];
            let o = (p.pos - cam).as_vec3();
            let rot = p.rot.as_quat();
            if let Some(m) = models.get(p.ac) {
                let model = Mat4::from_rotation_translation(rot, o);
                let glow = [0.0, spec.id as f32, spec.size_k, 0.0];
                let mut put = |part: Option<(usize, bool)>, glass: bool, tint: [f32; 4]| {
                    if let Some((mesh, cut)) = part {
                        objs.push(DrawObj { mesh, model, tint, glow, cast: if glass { 0 } else { 0b11 }, glass, cut, ..DrawObj::DEFAULT });
                    }
                };
                put(m.body, false, [0.0; 4]);
                if p.gear > 0.5 {
                    put(m.gear, false, [0.0; 4]);
                }
                put(m.glass, true, [0.0; 4]);
                // (the propeller as its blur: tint.x is how far it is blurred)
                put(m.prop, m.prop_glass, [1.0, 0.0, 0.0, 0.0]);
            } else {
                models.want(p.ac);
            }
            let Some(m) = models.get(p.ac) else { continue };
            if d > LIGHT_DIST {
                continue;
            }
            // Navigation lights: red on the left wing tip, green on the right, white on the tail, strobes.
            let at = |q: Vec3| -> [f32; 3] { (o + rot * q).into() };
            let k = spec.size_k.max(1.0);
            let mut l = |q: Vec3, inten: f32, col: [u8; 3], fx: [u8; 4]| {
                if lights.len() < crate::render::MAX_DYN_LIGHTS {
                    lights.push(Light { pos: at(q), inten: inten * k, dir: [0.0; 3], spread: 0.0, col: [col[0], col[1], col[2], 0], fx });
                }
            };
            let steady = [0, 0, 255, 1];
            let ph = ((i * 71) % 255) as u8;
            let [lt, rt, tail] = m.marks;
            l(lt, 40.0, L_RED, steady);
            l(rt, 40.0, L_GREEN, steady);
            l(tail, 25.0, L_WHITE, steady);
            for q in [lt, rt] {
                l(q, 1600.0, L_STROBE, [11, ph, 7, 1]);
            }
        }
    }
}

/// A peer aircraft type's meshes on the GPU: (mesh, has cut openings).
pub struct PeerMesh {
    body: Option<(usize, bool)>,
    gear: Option<(usize, bool)>,
    glass: Option<(usize, bool)>,
    prop: Option<(usize, bool)>,
    prop_glass: bool,
    /// the left and right wing tips and the tail's end (body frame), for the lights
    marks: [Vec3; 3],
}

/// Merged meshes of one type, made on a worker thread: body, gear, glass, propeller blur (and whether that is glass).
struct Merged([airframe::Mesh; 4], bool, [Vec3; 3]);

/// The other players' aircraft models, each type built once when first seen.
#[derive(Default)]
pub struct Models {
    ready: Vec<Option<PeerMesh>>,
    building: Vec<(usize, Receiver<Merged>)>,
}

impl Models {
    fn get(&self, ac: usize) -> Option<&PeerMesh> {
        self.ready.get(ac).and_then(|m| m.as_ref())
    }

    fn want(&mut self, ac: usize) {
        if ac >= aircraft::ALL.len() || self.get(ac).is_some() || self.building.iter().any(|(a, _)| *a == ac) {
            return;
        }
        let (tx, rx) = crossbeam_channel::bounded(1);
        let spec = aircraft::ALL[ac];
        let ok = std::thread::Builder::new()
            .name("peer model".into())
            .spawn(move || {
                let t0 = Instant::now();
                let m = merge(&(spec.build)());
                log::info!("multiplayer: {} model ready in {:.0} ms", spec.name, t0.elapsed().as_secs_f64() * 1000.0);
                let _ = tx.send(m);
            })
            .is_ok();
        if ok {
            self.building.push((ac, rx));
        }
    }

    /// Uploads the models that have been built.
    fn poll(&mut self, r: &mut Renderer) {
        let mut i = 0;
        while i < self.building.len() {
            match self.building[i].1.try_recv() {
                Ok(Merged(m, prop_glass, marks)) => {
                    let ac = self.building.swap_remove(i).0;
                    let up = |r: &mut Renderer, m: &airframe::Mesh| (!m.i.is_empty()).then(|| (r.add_mesh(&m.v, &m.i), m.v.iter().any(|v| airframe::cut_by(v.mat[1]))));
                    if self.ready.len() < aircraft::ALL.len() {
                        self.ready.resize_with(aircraft::ALL.len(), || None);
                    }
                    self.ready[ac] = Some(PeerMesh { body: up(r, &m[0]), gear: up(r, &m[1]), glass: up(r, &m[2]), prop: up(r, &m[3]), prop_glass, marks });
                }
                Err(TryRecvError::Empty) => i += 1,
                Err(TryRecvError::Disconnected) => {
                    self.building.swap_remove(i);
                }
            }
        }
    }
}

/// A model's pieces at rest merged by how they are drawn: the airframe, the
/// landing gear (shown while it is down), the glass, and the propeller's
/// blur (its blades are left out: the others are seen with it turning).
fn merge(m: &airframe::Model) -> Merged {
    let mut out: [airframe::Mesh; 4] = Default::default();
    let mut prop_glass = false;
    for p in &m.pieces {
        let k = match p.vis {
            airframe::Vis::SlowProp => continue,
            airframe::Vis::FastProp => {
                prop_glass |= p.glass;
                3
            }
            airframe::Vis::Always if p.glass => 2,
            airframe::Vis::Always if p.group == airframe::G_GEAR => 1,
            airframe::Vis::Always => 0,
        };
        out[k].append(&p.mesh);
    }
    let mut marks = [Vec3::ZERO; 3];
    for v in &out[0].v {
        let q = Vec3::from(v.pos);
        if q.x < marks[0].x {
            marks[0] = q;
        }
        if q.x > marks[1].x {
            marks[1] = q;
        }
        if q.z > marks[2].z {
            marks[2] = q;
        }
    }
    Merged(out, prop_glass, marks)
}

/// The network thread: waits to be told a room, keeps a connection to it
/// (reconnecting after a drop with a growing pause), and trades messages.
fn run(rx: Receiver<Cmd>, tx: Sender<Evt>) {
    let mut url: Option<String> = None;
    loop {
        let Some(u) = url.take() else {
            match rx.recv() {
                Ok(Cmd::Join(u)) => url = Some(u),
                Ok(_) => {}
                Err(_) => return,
            }
            continue;
        };
        let mut pause = 1.0f64;
        let mut next: Option<String> = None;
        'room: loop {
            let _ = tx.send(Evt::Net(Net::Connecting));
            let why = match connect(&u) {
                Ok(mut ws) => {
                    pause = 1.0;
                    let _ = tx.send(Evt::Net(Net::Connected));
                    let why = session(&mut ws, &rx, &tx, &mut next);
                    let _ = ws.close(None);
                    let _ = ws.flush();
                    match why {
                        Ok(()) => break 'room,
                        Err(e) => e,
                    }
                }
                Err(e) => e,
            };
            log::info!("multiplayer: {why}");
            let _ = tx.send(Evt::Net(if why == "full" { Net::Full } else { Net::Retrying(why) }));
            // Wait before trying again, still listening.
            let until = Instant::now() + Duration::from_secs_f64(pause);
            pause = (pause * 2.0).min(30.0);
            loop {
                let left = until.saturating_duration_since(Instant::now());
                if left.is_zero() {
                    break;
                }
                match rx.recv_timeout(left) {
                    Ok(Cmd::Join(n)) => {
                        next = Some(n);
                        break 'room;
                    }
                    Ok(Cmd::Leave) | Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break 'room,
                    Ok(Cmd::Send(_)) | Err(crossbeam_channel::RecvTimeoutError::Timeout) => {}
                }
            }
        }
        url = next.take();
        if url.is_none() {
            let _ = tx.send(Evt::Net(Net::Off));
        }
    }
}

type Ws = tungstenite::WebSocket<tungstenite::stream::MaybeTlsStream<TcpStream>>;

/// Opens the WebSocket: the TCP connection with a time limit, then TLS (for
/// wss://) and the upgrade. Reads then wait at most 50 ms, so the thread can
/// look at its commands in between.
fn connect(url: &str) -> Result<Ws, String> {
    let uri: tungstenite::http::Uri = url.parse().map_err(|_| "bad server address".to_string())?;
    let host = uri.host().ok_or("bad server address")?.trim_start_matches('[').trim_end_matches(']').to_string();
    let tls = uri.scheme_str() == Some("wss");
    let port = uri.port_u16().unwrap_or(if tls { 443 } else { 80 });
    let addrs: Vec<_> = (host.as_str(), port).to_socket_addrs().map_err(|_| "server not found".to_string())?.collect();
    let stream = addrs.iter().find_map(|a| TcpStream::connect_timeout(a, Duration::from_secs(6)).ok()).ok_or("server unreachable")?;
    let _ = stream.set_nodelay(true);
    let _ = stream.set_read_timeout(Some(Duration::from_secs(8)));
    let _ = stream.set_write_timeout(Some(Duration::from_secs(5)));
    let (ws, _) = tungstenite::client_tls(url, stream).map_err(|e| match e {
        tungstenite::HandshakeError::Failure(tungstenite::Error::Http(r)) if r.status().as_u16() == 409 => "full".to_string(),
        tungstenite::HandshakeError::Failure(tungstenite::Error::Http(r)) => format!("HTTP {}", r.status().as_u16()),
        e => e.to_string(),
    })?;
    let short = Some(Duration::from_millis(50));
    match ws.get_ref() {
        tungstenite::stream::MaybeTlsStream::Plain(s) => s.set_read_timeout(short),
        tungstenite::stream::MaybeTlsStream::Rustls(s) => s.sock.set_read_timeout(short),
        _ => Ok(()),
    }
    .map_err(|e| e.to_string())?;
    Ok(ws)
}

/// One connection's life: Ok when told to leave (or to join elsewhere, in
/// `next`), Err with why it ended otherwise.
fn session(ws: &mut Ws, rx: &Receiver<Cmd>, tx: &Sender<Evt>, next: &mut Option<String>) -> Result<(), String> {
    use tungstenite::Message;
    let mut ping = Instant::now();
    loop {
        // Only the newest of our own states is worth sending.
        let mut out = None;
        loop {
            match rx.try_recv() {
                Ok(Cmd::Send(s)) => out = Some(s),
                Ok(Cmd::Join(u)) => {
                    *next = Some(u);
                    return Ok(());
                }
                Ok(Cmd::Leave) | Err(TryRecvError::Disconnected) => return Ok(()),
                Err(TryRecvError::Empty) => break,
            }
        }
        if let Some(s) = out {
            ws.send(Message::text(s)).map_err(|e| e.to_string())?;
        }
        if ping.elapsed() > Duration::from_secs(20) {
            ping = Instant::now();
            ws.send(Message::Ping(Vec::new().into())).map_err(|e| e.to_string())?;
        }
        match ws.read() {
            Ok(Message::Text(t)) => handle(t.as_str(), tx),
            Ok(Message::Close(_)) => return Err("closed by the server".into()),
            Ok(_) => {}
            Err(tungstenite::Error::Io(e)) if matches!(e.kind(), std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut) => {}
            Err(e) => return Err(e.to_string()),
        }
    }
}

/// One message from the room.
fn handle(text: &str, tx: &Sender<Evt>) {
    let Ok(j) = serde_json::from_str::<serde_json::Value>(text) else { return };
    let id = |k: &str| j.get(k).and_then(|v| v.as_u64()).map(|x| x as u32);
    let e = match j.get("t").and_then(|v| v.as_str()) {
        Some("welcome") => id("id").map(|me| Evt::Welcome(me, j.get("peers").and_then(|p| p.as_array()).map(|a| a.iter().filter_map(|x| x.as_u64().map(|x| x as u32)).collect()).unwrap_or_default())),
        Some("join") => id("id").map(Evt::Join),
        Some("leave") => id("id").map(Evt::Leave),
        _ => id("from").and_then(|from| Some(Evt::State(from, State::decode(j.get("d")?)?))),
    };
    if let Some(e) = e {
        let _ = tx.send(e);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state() -> State {
        State { name: "Sam \"Ace\" ✈".into(), ac: 3, lat: 37.6189354, lon: -122.3749812, alt: 1523.37, q: [0.1, -0.2, 0.3, 0.9273618], vel: [120.5, -3.25, 7.0], gear: 0.0, flaps: 0.33, thr: 0.8, crashed: false, laser: true, t: 1234.567 }
    }

    #[test]
    fn state_round_trip() {
        let s = state();
        let text = s.encode();
        assert!(text.len() < 300, "{text}");
        let j: serde_json::Value = serde_json::from_str(&text).unwrap();
        let d = State::decode(&j).unwrap();
        assert_eq!((d.name.as_str(), d.ac, d.crashed, d.laser), (s.name.as_str(), s.ac, s.crashed, s.laser));
        assert!((d.lat - s.lat).abs() < 1e-7 && (d.lon - s.lon).abs() < 1e-7 && (d.alt - s.alt).abs() < 0.01);
        for k in 0..4 {
            assert!((d.q[k] - s.q[k]).abs() < 1e-5);
        }
        for k in 0..3 {
            assert!((d.vel[k] - s.vel[k]).abs() < 0.01);
        }
        assert!((d.gear - s.gear).abs() < 0.01 && (d.flaps - s.flaps).abs() < 0.01 && (d.thr - s.thr).abs() < 0.01);
        assert!((d.t - s.t).abs() < 1e-3);
        // As relayed by the room, with its sender.
        let (tx, rx) = crossbeam_channel::unbounded();
        handle(&format!("{{\"from\":7,\"d\":{text}}}"), &tx);
        handle("{\"t\":\"leave\",\"id\":3}", &tx);
        handle("{\"from\":2,\"d\":{\"n\":\"x\"}}", &tx);
        handle("not json", &tx);
        let got: Vec<Evt> = rx.try_iter().collect();
        assert_eq!(got.len(), 2);
        assert!(matches!(&got[0], Evt::State(7, d) if d.name == s.name));
        assert!(matches!(got[1], Evt::Leave(3)));
        // Odd values are kept in bounds; a vehicle past the list is the speedboat.
        let j = serde_json::json!({ "n": "a\u{7}b", "a": 99, "p": [95.0, 0.0, 1e9], "q": [0, 0, 0, 1], "v": [0, 0, 0], "ts": 1.0 });
        let d = State::decode(&j).unwrap();
        assert_eq!((d.name.as_str(), d.ac as usize, d.lat, d.alt), ("ab", boat_id(), 90.0, 200_000.0));
        assert_eq!(vehicle_name(boat_id()), boat::NAME);
    }

    #[test]
    fn interpolation() {
        let r0 = DQuat::from_rotation_y(0.0);
        let r1 = DQuat::from_rotation_y(0.2);
        let v = DVec3::new(100.0, 0.0, 0.0);
        let a = Snap { t: 10.0, pos: DVec3::new(0.0, 0.0, 0.0), rot: r0, vel: v, gear: 1.0, thr: 0.0 };
        let b = Snap { t: 10.1, pos: DVec3::new(10.0, 0.0, 0.0), rot: r1, vel: v, gear: 0.0, thr: 0.0 };
        let s = [a, b];
        // At the reports themselves, exactly them.
        let (p, q, _, g) = sample(&s, 10.0).unwrap();
        assert!(p.distance(a.pos) < 1e-9 && q.angle_between(r0) < 1e-9 && g == 1.0);
        let (p, q, _, _) = sample(&s, 10.1).unwrap();
        assert!(p.distance(b.pos) < 1e-9 && q.angle_between(r1) < 1e-6);
        // Halfway: on the straight line flown at constant speed, half turned.
        let (p, q, _, g) = sample(&s, 10.05).unwrap();
        assert!(p.distance(DVec3::new(5.0, 0.0, 0.0)) < 1e-9, "{p}");
        assert!((q.angle_between(r0) - 0.1).abs() < 1e-6 && (g - 0.5).abs() < 1e-6);
        // Past the last: flown on along its velocity, but not for ever.
        let (p, _, _, _) = sample(&s, 10.3).unwrap();
        assert!(p.distance(DVec3::new(30.0, 0.0, 0.0)) < 1e-9);
        let (p, _, _, _) = sample(&s, 100.0).unwrap();
        assert!(p.distance(DVec3::new(10.0 + 100.0 * COAST, 0.0, 0.0)) < 1e-9);
        // Before the first: held there.
        assert!(sample(&s, 5.0).unwrap().0.distance(a.pos) < 1e-9);
        assert!(sample(&[], 1.0).is_none());

        // A peer drawn DELAY behind, from reports that came late and unevenly.
        let mut peer = Peer::new(1);
        let st = |t: f64, x: f64| State { lat: 0.0, lon: x, alt: 1000.0, vel: [0.0, 0.0, 0.0], t, ..state() };
        peer.take(&st(50.0, 0.0), 1000.0);
        peer.take(&st(50.1, 0.001), 1000.15); // late: the offset stays the lowest seen
        peer.take(&st(50.05, 0.0005), 1000.16); // older than the newest: ignored
        assert_eq!(peer.snaps.len(), 2);
        assert!((peer.offset.unwrap() - 950.0).abs() < 0.01);
        peer.update(1000.0 + 0.05 + DELAY);
        let (lat, lon, _) = geo::pos_to_lla(peer.pos);
        assert!(lat.abs() < 1e-9 && (lon.to_degrees() - 0.0005).abs() < 1e-4, "{}", lon.to_degrees());

        // A speedboat lays a wake track as it goes, and it shows in the shaders' rows.
        let mut mp = Mp::new();
        let mut b = Peer::new(2);
        let now = mp.now();
        for k in 0..30 {
            let t = k as f64 * 0.1;
            b.take(&State { ac: boat_id() as u8, lat: 0.0, lon: t * 1e-4, alt: 0.0, q: [0.0, 0.0, 0.0, 1.0], vel: [0.0, 0.0, -11.0], thr: 0.9, t, ..state() }, now + t);
            b.update(now + t + DELAY);
        }
        assert!(b.is_boat() && b.wake.len() >= 3, "{}", b.wake.len());
        mp.peers.push(b);
        let mut rows = [[0.0f32; 4]; crate::render::WAKE_ROWS];
        mp.wake_rows(mp.peers[0].pos, &mut rows);
        assert!(rows[0][0] >= 4.0, "{:?}", rows[0]);
    }

    #[test]
    fn codes_and_urls() {
        assert_eq!(clean_code(" ab-c1 "), "ABC1");
        assert_eq!(clean_code("abcdefghijk"), "ABCDEFGH");
        let c = random_code();
        assert!(c.len() == 4 && c.chars().all(|c| c.is_ascii_uppercase()));
        assert_eq!(room_url("ws://127.0.0.1:8787/", "TEST"), "ws://127.0.0.1:8787/room/TEST");
        assert_eq!(room_url("https://x.workers.dev", "ABCD"), "wss://x.workers.dev/room/ABCD");
        assert_eq!(room_url("x.workers.dev", "ABCD"), "wss://x.workers.dev/room/ABCD");
    }
}
