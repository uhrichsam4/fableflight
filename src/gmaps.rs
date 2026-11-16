//! Optional Google Maps satellite imagery (Map Tiles API, 2D tiles), in place
//! of the default imagery. Off by default: it needs the player's own API key,
//! and every tile that is not already on disk is a billed request, so this
//! counts requests and bytes (this session, this calendar month, all time) and
//! keeps the counts in `<config>/fablegame/google_maps.json`.
//!
//! The API: `POST createSession` hands out a session token (valid ~2 weeks),
//! then tiles are `GET 2dtiles/{z}/{x}/{y}?session=..&key=..` in the usual XYZ
//! scheme. `viewport` gives the copyright line that has to be shown with them.
//!
//! Photorealistic 3D Tiles (tiles3d.rs) go through the same key and counts.
//! Google bills those per root tileset request (`3dtiles/root.json`), which
//! opens a session of about three hours for the tiles under it, so roots are
//! counted on their own as well.
use anyhow::{anyhow, Result};
use parking_lot::Mutex;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU8, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

const API: &str = "https://tile.googleapis.com";

/// Request and byte counts. Only real network calls to Google count; tiles
/// read back from the disk cache are free and counted apart.
#[derive(Clone, Default, Debug)]
pub struct Usage {
    pub session_requests: u64,
    pub session_bytes: u64,
    pub session_tiles: u64,
    pub session_cached: u64,
    pub session_errors: u64,
    /// "YYYY-MM" the month counts belong to (Google bills per month)
    pub month: String,
    pub month_requests: u64,
    pub month_bytes: u64,
    pub total_requests: u64,
    pub total_bytes: u64,
    /// 3D tiles: root tileset requests (the billed unit), and tiles fetched
    pub session_roots: u64,
    pub session_3d: u64,
    /// 3D tiles served from the disk cache (still within their max-age)
    pub session_3d_cached: u64,
    pub month_roots: u64,
    pub total_roots: u64,
}

/// How much of Google's 3D mesh is streamed (tiles3d::Params has the
/// numbers): for slower connections and smaller GPUs, or for more.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Detail {
    Low,
    Medium,
    #[default]
    High,
    Ultra,
}

impl Detail {
    pub const ALL: [Detail; 4] = [Detail::Low, Detail::Medium, Detail::High, Detail::Ultra];

    pub fn name(self) -> &'static str {
        match self {
            Detail::Low => "low",
            Detail::Medium => "medium",
            Detail::High => "high",
            Detail::Ultra => "ultra",
        }
    }

    pub fn parse(s: &str) -> Option<Detail> {
        Self::ALL.into_iter().find(|d| d.name().eq_ignore_ascii_case(s.trim()))
    }
}

struct Session {
    token: String,
    /// unix seconds
    expiry: u64,
}

pub struct Google {
    enabled: AtomicBool,
    /// photorealistic 3D tiles near the aircraft (independent of the 2D imagery)
    photoreal: AtomicBool,
    /// how much of the 3D mesh to stream (an index into Detail::ALL)
    detail: AtomicU8,
    key: Mutex<String>,
    session: Mutex<Option<Session>>,
    pub usage: Mutex<Usage>,
    /// what went wrong last, or empty
    pub status: Mutex<String>,
    /// copyright line for the imagery in view
    pub attribution: Mutex<String>,
    attr_busy: AtomicBool,
    attr_at: Mutex<Option<(f64, f64, u8)>>,
    /// bumped whenever the imagery source changes, so loaded terrain can be refreshed
    generation: AtomicU32,
    path: Option<PathBuf>,
    /// where 3D tiles are kept until their max-age runs out (None: not at all)
    cache3d: Option<PathBuf>,
    unsaved: AtomicU32,
    agent: ureq::Agent,
}

fn now_s() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

/// "YYYY-MM" (UTC) for unix seconds.
fn month_of(secs: u64) -> String {
    // civil-from-days, Howard Hinnant
    let z = (secs / 86400) as i64 + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!("{y:04}-{m:02}")
}

pub fn bytes(n: u64) -> String {
    match n {
        n if n >= 1 << 30 => format!("{:.2} GB", n as f64 / (1u64 << 30) as f64),
        n if n >= 1 << 20 => format!("{:.1} MB", n as f64 / (1u64 << 20) as f64),
        n if n >= 1 << 10 => format!("{:.0} KB", n as f64 / 1024.0),
        n => format!("{n} B"),
    }
}

impl Google {
    /// Off, not saved anywhere (tests and tools).
    pub fn disabled() -> Self {
        Self::with(None, String::new(), false, false, Usage::default())
    }

    fn with(path: Option<PathBuf>, key: String, enabled: bool, photoreal: bool, usage: Usage) -> Self {
        let agent = ureq::Agent::config_builder().timeout_global(Some(Duration::from_secs(20))).user_agent("fablegame-flightsim/0.1").http_status_as_error(false).build().into();
        Self {
            enabled: AtomicBool::new(enabled && !key.is_empty()),
            photoreal: AtomicBool::new(photoreal && !key.is_empty()),
            detail: AtomicU8::new(Detail::default() as u8),
            key: Mutex::new(key),
            session: Mutex::new(None),
            usage: Mutex::new(usage),
            status: Mutex::new(String::new()),
            attribution: Mutex::new(String::new()),
            attr_busy: AtomicBool::new(false),
            attr_at: Mutex::new(None),
            generation: AtomicU32::new(0),
            path,
            cache3d: None,
            unsaved: AtomicU32::new(0),
            agent,
        }
    }

    /// The saved settings and counts. `key` (from the command line or the
    /// environment) replaces the saved key when given.
    pub fn load(key: Option<String>) -> Self {
        let path = crate::settings::config_dir().map(|d| d.join("google_maps.json"));
        let j: serde_json::Value = path.as_ref().and_then(|p| std::fs::read(p).ok()).and_then(|d| serde_json::from_slice(&d).ok()).unwrap_or_default();
        let n = |k: &str| j[k].as_u64().unwrap_or(0);
        let mut u = Usage { month: j["month"].as_str().unwrap_or("").to_string(), month_requests: n("month_requests"), month_bytes: n("month_bytes"), total_requests: n("total_requests"), total_bytes: n("total_bytes"), month_roots: n("month_roots"), total_roots: n("total_roots"), ..Default::default() };
        let m = month_of(now_s());
        if u.month != m {
            (u.month, u.month_requests, u.month_bytes, u.month_roots) = (m, 0, 0, 0);
        }
        let key = key.filter(|k| !k.trim().is_empty()).unwrap_or_else(|| j["key"].as_str().unwrap_or("").to_string());
        let mut g = Self::with(path, key.trim().to_string(), j["enabled"].as_bool().unwrap_or(false), j["photoreal"].as_bool().unwrap_or(false), u);
        if let Some(d) = j["photoreal_detail"].as_str().and_then(Detail::parse) {
            g.detail = AtomicU8::new(d as u8);
        }
        if std::env::var_os("FABLE_NO_3D_CACHE").is_none() {
            g.cache3d = dirs::cache_dir().map(|d| d.join("fablegame").join("g3d"));
            if let Some(dir) = g.cache3d.clone() {
                std::thread::spawn(move || prune_cache3d(&dir, now_s(), CACHE3D_MAX));
            }
        }
        g
    }

    pub fn save(&self) {
        let Some(path) = &self.path else { return };
        self.unsaved.store(0, Ordering::Relaxed);
        let u = self.usage.lock().clone();
        let j = serde_json::json!({
            "enabled": self.enabled(),
            "photoreal": self.photoreal(),
            "photoreal_detail": self.detail().name(),
            "key": *self.key.lock(),
            "month": u.month,
            "month_requests": u.month_requests,
            "month_bytes": u.month_bytes,
            "total_requests": u.total_requests,
            "total_bytes": u.total_bytes,
            "month_roots": u.month_roots,
            "total_roots": u.total_roots,
        });
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        let tmp = path.with_extension("json.tmp");
        if std::fs::write(&tmp, serde_json::to_vec_pretty(&j).unwrap_or_default()).is_ok() {
            let _ = std::fs::rename(&tmp, path);
        }
    }

    pub fn enabled(&self) -> bool {
        self.enabled.load(Ordering::Relaxed)
    }

    pub fn photoreal(&self) -> bool {
        self.photoreal.load(Ordering::Relaxed)
    }

    pub fn set_photoreal(&self, on: bool) {
        let on = on && self.has_key();
        if self.photoreal.swap(on, Ordering::Relaxed) != on {
            self.status.lock().clear();
            self.save();
        }
    }

    pub fn detail(&self) -> Detail {
        Detail::ALL[(self.detail.load(Ordering::Relaxed) as usize).min(3)]
    }

    pub fn set_detail(&self, d: Detail) {
        if self.detail.swap(d as u8, Ordering::Relaxed) != d as u8 {
            self.save();
        }
    }

    pub fn has_key(&self) -> bool {
        !self.key.lock().is_empty()
    }

    pub fn key(&self) -> String {
        self.key.lock().clone()
    }

    /// Generation of the imagery source; changes on every switch.
    pub fn generation(&self) -> u32 {
        self.generation.load(Ordering::Relaxed)
    }

    pub fn set_enabled(&self, on: bool) {
        let on = on && self.has_key();
        if self.enabled.swap(on, Ordering::Relaxed) != on {
            self.generation.fetch_add(1, Ordering::Relaxed);
            self.status.lock().clear();
            self.attr_at.lock().take();
            self.save();
        }
    }

    pub fn set_key(&self, key: &str) {
        let key = key.trim().to_string();
        if *self.key.lock() == key {
            return;
        }
        *self.key.lock() = key;
        self.session.lock().take();
        self.status.lock().clear();
        if !self.has_key() {
            self.set_enabled(false);
            self.set_photoreal(false);
        } else if self.enabled() {
            // tiles that fell back to the default imagery under the old key get another go
            self.generation.fetch_add(1, Ordering::Relaxed);
        }
        self.save();
    }

    pub fn reset_session_counts(&self) {
        let mut u = self.usage.lock();
        (u.session_requests, u.session_bytes, u.session_tiles, u.session_cached, u.session_errors, u.session_roots, u.session_3d, u.session_3d_cached) = (0, 0, 0, 0, 0, 0, 0, 0);
    }

    fn count(&self, n: usize, ok: bool) {
        {
            let mut u = self.usage.lock();
            let m = month_of(now_s());
            if u.month != m {
                (u.month, u.month_requests, u.month_bytes, u.month_roots) = (m, 0, 0, 0);
            }
            u.session_requests += 1;
            u.month_requests += 1;
            u.total_requests += 1;
            u.session_bytes += n as u64;
            u.month_bytes += n as u64;
            u.total_bytes += n as u64;
            if !ok {
                u.session_errors += 1;
            }
        }
        if self.unsaved.fetch_add(1, Ordering::Relaxed) >= 50 {
            self.save();
        }
    }

    /// One counted call to the API. Errors carry Google's own message.
    fn call(&self, send: impl FnOnce() -> Result<ureq::http::Response<ureq::Body>, ureq::Error>) -> Result<(u16, Vec<u8>)> {
        self.call_aged(send).map(|(c, d, _)| (c, d))
    }

    /// `call`, plus the reply's `Cache-Control: max-age` in seconds (0 when it
    /// has none, or says no-store / no-cache / private).
    fn call_aged(&self, send: impl FnOnce() -> Result<ureq::http::Response<ureq::Body>, ureq::Error>) -> Result<(u16, Vec<u8>, u64)> {
        let mut res = match send() {
            Ok(r) => r,
            Err(e) => {
                self.count(0, false);
                return Err(e.into());
            }
        };
        let code = res.status().as_u16();
        let age = res.headers().get("cache-control").and_then(|v| v.to_str().ok()).map_or(0, max_age);
        let data = res.body_mut().with_config().limit(16 << 20).read_to_vec().unwrap_or_default();
        self.count(data.len(), code == 200);
        Ok((code, data, age))
    }

    fn fail(&self, code: u16, data: &[u8]) -> anyhow::Error {
        let msg = serde_json::from_slice::<serde_json::Value>(data).ok().and_then(|j| j["error"]["message"].as_str().map(String::from)).unwrap_or_else(|| String::from_utf8_lossy(&data[..data.len().min(200)]).into_owned());
        let msg = format!("HTTP {code}: {}", msg.trim());
        // A missing tile (no imagery there) is no reason to worry the player.
        if code != 404 {
            *self.status.lock() = msg.clone();
        }
        anyhow!(msg)
    }

    fn token(&self) -> Result<String> {
        let mut s = self.session.lock();
        if let Some(t) = s.as_ref().filter(|t| t.expiry > now_s() + 3600) {
            return Ok(t.token.clone());
        }
        // Others queued on the lock while a bad key switched this off.
        if !self.enabled() {
            return Err(anyhow!("google imagery off"));
        }
        let key = self.key();
        let body = serde_json::to_vec(&serde_json::json!({ "mapType": "satellite", "language": "en-US", "region": "US" }))?;
        let (code, data) = self.call(|| self.agent.post(format!("{API}/v1/createSession")).query("key", &key).header("Content-Type", "application/json").send(&body[..]))?;
        if code != 200 {
            // Bad key or the API not enabled: no point asking again for every tile.
            if code == 400 || code == 403 {
                drop(s);
                self.set_enabled(false);
            }
            return Err(self.fail(code, &data));
        }
        let j: serde_json::Value = serde_json::from_slice(&data)?;
        let token = j["session"].as_str().ok_or_else(|| anyhow!("no session in reply"))?.to_string();
        let expiry = j["expiry"].as_str().and_then(|e| e.parse().ok()).or(j["expiry"].as_u64()).unwrap_or(now_s() + 86400);
        *s = Some(Session { token: token.clone(), expiry });
        self.status.lock().clear();
        Ok(token)
    }

    /// The satellite tile, from the disk cache under `root` or from Google.
    pub fn tile(&self, root: &std::path::Path, z: u8, x: u32, y: u32) -> Result<Vec<u8>> {
        let path = root.join(format!("gimg/{z}/{x}/{y}.jpg"));
        if let Ok(d) = std::fs::read(&path) {
            if !d.is_empty() {
                self.usage.lock().session_cached += 1;
                return Ok(d);
            }
        }
        if !self.enabled() {
            return Err(anyhow!("google imagery off"));
        }
        let mut retried = false;
        loop {
            let token = self.token()?;
            let (code, data) = self.call(|| self.agent.get(format!("{API}/v1/2dtiles/{z}/{x}/{y}")).query("session", &token).query("key", self.key()).call())?;
            match code {
                200 if !data.is_empty() => {
                    if let Some(dir) = path.parent() {
                        let _ = std::fs::create_dir_all(dir);
                    }
                    let tmp = path.with_extension(format!("tmp{:?}", std::thread::current().id()).replace(['(', ')'], ""));
                    if std::fs::write(&tmp, &data).is_ok() {
                        let _ = std::fs::rename(&tmp, &path);
                    }
                    self.usage.lock().session_tiles += 1;
                    return Ok(data);
                }
                // the session ran out early: one fresh one
                400 | 401 if !retried => {
                    retried = true;
                    self.session.lock().take();
                }
                _ => return Err(self.fail(code, &data)),
            }
        }
    }

    /// A 3D tiles resource: `path` is the URL path and query from the tileset
    /// (`/v1/3dtiles/...`, with its `session` already in it, except for the
    /// root). Meshes (`.glb`) are kept on disk only as long as their own
    /// `Cache-Control: max-age` allows (the terms allow nothing beyond what the
    /// HTTP headers say), keyed by path without the session, so a later session
    /// within that time reads them back. The root (max-age=0, billed) and the
    /// subtree JSON (its links are tied to a session) always go out.
    pub fn get3d(&self, path: &str) -> Result<Vec<u8>> {
        if !self.photoreal() {
            return Err(anyhow!("google 3d tiles off"));
        }
        let root = path.starts_with("/v1/3dtiles/root.json");
        let file = self.cache3d.as_ref().and_then(|d| cache3d_file(d, path));
        let now = now_s();
        if let Some(d) = file.as_ref().and_then(|f| read_cache3d(f, now)) {
            self.usage.lock().session_3d_cached += 1;
            return Ok(d);
        }
        let url = format!("{API}{path}");
        let (code, data, age) = self.call_aged(|| self.agent.get(&url).query("key", self.key()).call())?;
        if let (Some(f), true) = (&file, code == 200 && !data.is_empty() && age > 0) {
            write_cache3d(f, now + age, &data);
        }
        if root {
            let mut u = self.usage.lock();
            u.session_roots += 1;
            u.month_roots += 1;
            u.total_roots += 1;
            drop(u);
            // a root is billed: keep its count even if the game dies
            self.save();
        }
        if code == 200 && !data.is_empty() {
            if !root {
                self.usage.lock().session_3d += 1;
            }
            return Ok(data);
        }
        // Bad key, or the API not enabled for it: stop asking.
        if root && (code == 400 || code == 403) {
            self.set_photoreal(false);
        }
        Err(self.fail(code, &data))
    }

    /// Ask, in the background, for the copyright line of the imagery around a
    /// spot. Only goes out again once the view has moved on a fair way.
    pub fn update_attribution(self: &std::sync::Arc<Self>, lat: f64, lon: f64, alt: f64) {
        if !self.enabled() || self.attr_busy.load(Ordering::Relaxed) {
            return;
        }
        let zoom = (25.0 - (alt.max(100.0) / 50.0).log2()).clamp(3.0, 20.0) as u8;
        // half-width of the view, roughly, in degrees
        let span = (alt.max(500.0) * 2.0 / 111_000.0).min(60.0);
        {
            let mut at = self.attr_at.lock();
            if let Some((la, lo, z)) = *at {
                if (la - lat).abs() < span * 0.5 && (lo - lon).abs() < span * 0.5 && z.abs_diff(zoom) < 2 {
                    return;
                }
            }
            *at = Some((lat, lon, zoom));
        }
        self.attr_busy.store(true, Ordering::Relaxed);
        let g = self.clone();
        std::thread::spawn(move || {
            let r = (|| -> Result<String> {
                let token = g.token()?;
                let (code, data) = g.call(|| {
                    g.agent
                    .get(format!("{API}/tile/v1/viewport"))
                    .query("session", &token)
                    .query("key", g.key())
                    .query("zoom", zoom.to_string())
                    .query("north", format!("{:.5}", (lat + span).min(85.0)))
                    .query("south", format!("{:.5}", (lat - span).max(-85.0)))
                    .query("east", format!("{:.5}", (lon + span).min(180.0)))
                        .query("west", format!("{:.5}", (lon - span).max(-180.0)))
                        .call()
                })?;
                if code != 200 {
                    return Err(g.fail(code, &data));
                }
                let j: serde_json::Value = serde_json::from_slice(&data)?;
                Ok(j["copyright"].as_str().unwrap_or("").to_string())
            })();
            if let Ok(c) = r {
                *g.attribution.lock() = c;
            }
            g.attr_busy.store(false, Ordering::Relaxed);
        });
    }
}

/// Disk cache of 3D tiles: past this, the oldest-expiring go first.
const CACHE3D_MAX: u64 = 3 << 30;

/// `max-age` from a Cache-Control value, 0 if it forbids keeping the reply.
/// `private` is fine: it rules out shared caches, and this one is the
/// player's own (Google sends `private, max-age=14400` on meshes).
fn max_age(cc: &str) -> u64 {
    let mut age = 0;
    for d in cc.split(',').map(|d| d.trim().to_ascii_lowercase()) {
        if d == "no-store" || d == "no-cache" {
            return 0;
        }
        if let Some(v) = d.strip_prefix("max-age=") {
            age = v.trim_matches('"').parse().unwrap_or(0);
        }
    }
    age
}

/// The cache file for a 3D tiles path: meshes only, named by a hash of the
/// path without its query (session and key change, the content doesn't).
fn cache3d_file(dir: &std::path::Path, path: &str) -> Option<PathBuf> {
    let p = path.split('?').next().unwrap_or(path);
    if !p.ends_with(".glb") {
        return None;
    }
    let h = p.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ b as u64).wrapping_mul(0x100_0000_01b3));
    Some(dir.join(format!("{:02x}/{h:016x}.glb", h >> 56)))
}

/// A cached tile: 8 bytes of expiry (unix seconds, little-endian), then the
/// body. Expired or damaged files are removed and count as missing.
fn read_cache3d(f: &std::path::Path, now: u64) -> Option<Vec<u8>> {
    let mut d = std::fs::read(f).ok()?;
    let exp = d.get(..8).map(|b| u64::from_le_bytes(b.try_into().unwrap()));
    if exp.is_none_or(|e| e <= now) || d.len() <= 8 {
        let _ = std::fs::remove_file(f);
        return None;
    }
    d.drain(..8);
    Some(d)
}

fn write_cache3d(f: &std::path::Path, expiry: u64, data: &[u8]) {
    if let Some(dir) = f.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let mut buf = Vec::with_capacity(data.len() + 8);
    buf.extend_from_slice(&expiry.to_le_bytes());
    buf.extend_from_slice(data);
    let tmp = f.with_extension(format!("tmp{:?}", std::thread::current().id()).replace(['(', ')'], ""));
    if std::fs::write(&tmp, &buf).is_ok() {
        let _ = std::fs::rename(&tmp, f);
    }
}

/// Removes expired tiles (and stray temporaries), then the soonest-expiring
/// until the rest fits in `max` bytes.
fn prune_cache3d(dir: &std::path::Path, now: u64, max: u64) {
    use std::io::Read;
    let mut live = Vec::new();
    for sub in std::fs::read_dir(dir).into_iter().flatten().flatten() {
        for e in std::fs::read_dir(sub.path()).into_iter().flatten().flatten() {
            let p = e.path();
            let mut b = [0u8; 8];
            let exp = (p.extension().is_some_and(|x| x == "glb")).then(|| std::fs::File::open(&p).and_then(|mut f| f.read_exact(&mut b)).ok()).flatten().map(|_| u64::from_le_bytes(b));
            match exp {
                Some(e) if e > now => live.push((e, e_len(&p), p)),
                _ => {
                    let _ = std::fs::remove_file(&p);
                }
            }
        }
    }
    let mut total: u64 = live.iter().map(|l| l.1).sum();
    live.sort_by_key(|l| l.0);
    for (_, n, p) in live {
        if total <= max {
            break;
        }
        let _ = std::fs::remove_file(&p);
        total -= n;
    }
}

fn e_len(p: &std::path::Path) -> u64 {
    std::fs::metadata(p).map_or(0, |m| m.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn months() {
        assert_eq!(month_of(0), "1970-01");
        assert_eq!(month_of(1_794_528_000), "2026-11"); // 2026-11-13
        assert_eq!(month_of(1_798_761_599), "2026-12"); // 2026-12-31 23:59:59
        assert_eq!(month_of(951_782_400), "2000-02"); // 2000-02-29
    }

    #[test]
    fn detail_names() {
        for d in Detail::ALL {
            assert_eq!(Detail::parse(d.name()), Some(d));
        }
        assert_eq!(Detail::parse(" Ultra "), Some(Detail::Ultra));
        assert_eq!(Detail::parse("max"), None);
        assert_eq!(Detail::default(), Detail::High);
    }

    #[test]
    fn cache_control() {
        assert_eq!(max_age("public, max-age=14400"), 14400);
        assert_eq!(max_age("max-age=0"), 0);
        assert_eq!(max_age("private, max-age=14400, must-revalidate, no-transform"), 14400);
        assert_eq!(max_age("private, max-age=0, must-revalidate"), 0);
        assert_eq!(max_age("no-store"), 0);
        assert_eq!(max_age(""), 0);
    }

    #[test]
    fn cache3d_roundtrip() {
        let dir = std::env::temp_dir().join(format!("fable-g3d-test-{}", std::process::id()));
        let a = cache3d_file(&dir, "/v1/3dtiles/datasets/A/files/B.glb?session=S1").unwrap();
        assert_eq!(Some(a.clone()), cache3d_file(&dir, "/v1/3dtiles/datasets/A/files/B.glb?session=S2"));
        assert!(cache3d_file(&dir, "/v1/3dtiles/datasets/A/files/B.json?session=S1").is_none());
        assert!(cache3d_file(&dir, "/v1/3dtiles/root.json").is_none());
        write_cache3d(&a, 1000, b"mesh");
        assert_eq!(read_cache3d(&a, 999).as_deref(), Some(&b"mesh"[..]));
        assert!(read_cache3d(&a, 1000).is_none());
        assert!(!a.exists());
        let b = cache3d_file(&dir, "/x/b.glb").unwrap();
        let c = cache3d_file(&dir, "/x/c.glb").unwrap();
        write_cache3d(&a, 500, b"old");
        write_cache3d(&b, 2000, b"0123456789");
        write_cache3d(&c, 3000, b"0123456789");
        prune_cache3d(&dir, 600, 20);
        assert!(!a.exists() && !b.exists() && c.exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn off_without_key() {
        let g = Google::disabled();
        g.set_enabled(true);
        assert!(!g.enabled());
        g.set_key("abc");
        g.set_enabled(true);
        assert!(g.enabled());
        let generation = g.generation();
        g.set_key("");
        assert!(!g.enabled());
        assert!(g.generation() > generation);
        g.set_key("abc");
        g.set_photoreal(true);
        assert!(g.photoreal());
        g.set_key("");
        assert!(!g.photoreal());
    }
}
