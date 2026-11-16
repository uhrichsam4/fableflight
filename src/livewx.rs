//! Real weather: the current conditions where the player is (or where they
//! are about to start), turned into the weather sliders. The nearest METAR
//! (aviationweather.gov) within 60 km gives the wind, gusts, visibility, the
//! cloud layers and thunderstorms; away from any airport, Open-Meteo's model
//! stands in. A background thread looks again every ten minutes or after a
//! move of 30 km, and the sky eases into each new report.

use crate::geo;
use crate::sources::Sources;
use crate::weather::Weather;
use parking_lot::Mutex;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// How often to look again (s), and how far the player may move first (km).
const REFRESH: f64 = 600.0;
const MOVE_KM: f64 = 30.0;
/// The nearest METAR station is used up to this far away (km).
const STATION_KM: f64 = 60.0;

/// The weather sliders a report comes to.
#[derive(Clone, Debug, PartialEq)]
pub struct Target {
    pub clouds: f32,
    /// None: the report says nothing about it (keep what there is)
    pub base_m: Option<f32>,
    pub depth_m: f32,
    pub alto: f32,
    pub storms: f32,
    pub haze: f32,
    /// snow 0..1 (falling, or freezing with precipitation)
    pub snow: f32,
    /// m/s, and where from (None: variable)
    pub wind: f64,
    pub wind_from: Option<f64>,
    pub gusts: f64,
    /// what it came from, for the weather panel: "EBBR" or "Open-Meteo",
    /// and the report itself (the raw METAR)
    pub station: String,
    pub raw: String,
}

#[derive(Default)]
struct Shared {
    want: Option<(f64, f64)>,
    got: Option<(Instant, Target)>,
    error: String,
}

pub struct LiveWx {
    shared: Arc<Mutex<Shared>>,
    thread: bool,
    last_got: Option<Instant>,
    target: Option<Target>,
    /// ease in (false right after a new place is picked: jump straight there)
    settled: bool,
}

impl LiveWx {
    pub fn new() -> Self {
        Self { shared: Arc::default(), thread: false, last_got: None, target: None, settled: false }
    }

    pub fn error(&self) -> String {
        self.shared.lock().error.clone()
    }

    /// Each frame: where the weather is wanted (degrees), and the time since
    /// the last frame (s; 0 jumps straight to the report, as on the globe).
    pub fn step(&mut self, src: &Arc<Sources>, wx: &mut Weather, at: Option<(f64, f64)>, dt: f64) {
        if !wx.live {
            self.settled = false;
            return;
        }
        let got = {
            let mut s = self.shared.lock();
            s.want = at;
            s.got.as_ref().filter(|(t, _)| Some(*t) != self.last_got).cloned()
        };
        if !self.thread && at.is_some() {
            self.thread = true;
            spawn(src.clone(), self.shared.clone());
        }
        if let Some((t, target)) = got {
            self.last_got = Some(t);
            wx.live_station = target.station.clone();
            wx.live_raw = target.raw.clone();
            self.target = Some(target);
        }
        let Some(t) = &self.target else {
            let e = self.error();
            if !e.is_empty() {
                wx.live_raw = format!("({e})");
            }
            return;
        };
        // About half way in 10 s; at once on the globe or the first time.
        let k = if dt <= 0.0 || !self.settled { 1.0 } else { 1.0 - (-dt / 14.0).exp() };
        self.settled = true;
        let mix = |a: f32, b: f32| a + (b - a) * k as f32;
        wx.clouds = mix(wx.clouds, t.clouds);
        if let Some(b) = t.base_m {
            wx.cloud_base = mix(wx.cloud_base, b);
        }
        wx.cloud_depth = mix(wx.cloud_depth, t.depth_m);
        wx.alto = mix(wx.alto, t.alto);
        wx.storms = mix(wx.storms, t.storms);
        wx.haze = mix(wx.haze, t.haze);
        wx.snow = mix(wx.snow, t.snow);
        wx.wind += (t.wind - wx.wind) * k;
        wx.gusts += (t.gusts - wx.gusts) * k;
        if let Some(from) = t.wind_from {
            let d = (from - wx.wind_from + 540.0).rem_euclid(360.0) - 180.0;
            wx.wind_from = (wx.wind_from + d * k).rem_euclid(360.0);
        }
    }
}

fn spawn(src: Arc<Sources>, shared: Arc<Mutex<Shared>>) {
    std::thread::spawn(move || {
        let mut last: Option<(Instant, (f64, f64))> = None;
        loop {
            let want = shared.lock().want;
            if let Some(p) = want {
                let due = match last {
                    None => true,
                    Some((t, q)) => t.elapsed().as_secs_f64() > REFRESH || dist_km(p, q) > MOVE_KM,
                };
                if due {
                    let res = metar(&src, p).or_else(|e| {
                        log::info!("live weather: {e}; using Open-Meteo");
                        open_meteo(&src, p)
                    });
                    let mut s = shared.lock();
                    match res {
                        Ok(t) => {
                            s.got = Some((Instant::now(), t));
                            s.error.clear();
                            last = Some((Instant::now(), p));
                        }
                        Err(e) => {
                            if e != "offline" {
                                log::warn!("live weather: {e}");
                            }
                            s.error = e;
                            // try again in a minute, not every second
                            last = Some((Instant::now() - Duration::from_secs_f64(REFRESH - 60.0), p));
                        }
                    }
                }
            }
            std::thread::sleep(Duration::from_secs(1));
        }
    });
}

fn dist_km(a: (f64, f64), b: (f64, f64)) -> f64 {
    let p = |q: (f64, f64)| geo::lla_to_pos(q.0.to_radians(), q.1.to_radians(), 0.0);
    (p(a) - p(b)).length() / 1000.0
}

fn get_json(src: &Sources, url: &str) -> Result<serde_json::Value, String> {
    let d = src.fetch.get(url).map_err(|e| e.to_string())?;
    serde_json::from_slice(&d).map_err(|e| e.to_string())
}

/// The nearest METAR station's latest report.
fn metar(src: &Sources, (lat, lon): (f64, f64)) -> Result<Target, String> {
    let dlat = STATION_KM / 111.0;
    let dlon = dlat / lat.to_radians().cos().max(0.2);
    let url = format!("https://aviationweather.gov/api/data/metar?bbox={:.3},{:.3},{:.3},{:.3}&format=json", lat - dlat, lon - dlon, lat + dlat, lon + dlon);
    let j = get_json(src, &url)?;
    let near = j
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|m| Some((dist_km((lat, lon), (m["lat"].as_f64()?, m["lon"].as_f64()?)), m)))
        .filter(|(d, _)| *d < STATION_KM)
        .min_by(|a, b| a.0.total_cmp(&b.0))
        .map(|(_, m)| m)
        .ok_or_else(|| "no METAR station near".to_string())?;
    from_metar(near).ok_or_else(|| "unreadable METAR".to_string())
}

/// A METAR (aviationweather.gov's JSON) as the weather sliders.
fn from_metar(m: &serde_json::Value) -> Option<Target> {
    let raw = m["rawOb"].as_str().unwrap_or("").to_string();
    let station = m["icaoId"].as_str()?.to_string();
    let elev = m["elev"].as_f64().unwrap_or(0.0);
    let kt = |v: &serde_json::Value| v.as_f64();
    let wind = kt(&m["wspd"]).unwrap_or(0.0) / 1.94384;
    let gust = kt(&m["wgst"]).map(|g| g / 1.94384);
    let wind_from = m["wdir"].as_f64();
    // Visibility in statute miles: 6, "6+", "1/2", "1 1/2".
    let vis_km = match &m["visib"] {
        serde_json::Value::Number(n) => n.as_f64(),
        // ("6+": at least six miles, as clear as the sky model goes)
        serde_json::Value::String(s) if s.trim().ends_with('+') => None,
        serde_json::Value::String(s) => sm(s),
        _ => None,
    }
    .map(|v| v * 1.609);
    let tokens: Vec<&str> = raw.split_whitespace().collect();
    let wx = m["wxString"].as_str().unwrap_or("");
    let cavok = tokens.contains(&"CAVOK");
    // The cloud layers: (cover 0..1, base in m above sea level).
    let mut layers: Vec<(f32, f32)> = m["clouds"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|c| {
            let cover = match c["cover"].as_str()? {
                "FEW" => 0.2,
                "SCT" => 0.45,
                "BKN" => 0.75,
                "OVC" | "OVX" | "VV" => 1.0,
                _ => return None,
            };
            Some((cover, (c["base"].as_f64()? / 3.28084 + elev) as f32))
        })
        .collect();
    layers.sort_by(|a, b| a.1.total_cmp(&b.1));
    let cb = tokens.iter().any(|t| t.ends_with("CB"));
    let tcu = tokens.iter().any(|t| t.ends_with("TCU"));
    let storms = if wx.contains("TS") { 0.75 } else if cb { 0.45 } else if tcu { 0.2 } else { 0.0 };
    let (clouds, base_m, alto) = match layers.first() {
        None => (0.03, None, 0.0),
        Some(&(_, base)) => {
            let low = layers.iter().filter(|l| l.1 < base + 1000.0).map(|l| l.0).fold(0.0f32, f32::max);
            let high = layers.iter().filter(|l| l.1 >= base + 1000.0).map(|l| l.0).fold(0.0f32, f32::max);
            (low, Some(base.clamp(300.0, 4000.0)), high)
        }
    };
    let foggy = ["FG", "BR", "HZ", "FU", "DU", "SA"].iter().any(|w| wx.contains(w));
    Some(Target {
        clouds,
        base_m,
        depth_m: depth(clouds, storms),
        alto,
        storms,
        haze: haze(if cavok { None } else { vis_km }, foggy),
        snow: metar_snow(wx, m["temp"].as_f64()),
        wind,
        wind_from,
        gusts: gust.map(|g| ((g - wind) / 8.0).clamp(0.0, 1.0)).unwrap_or(0.1),
        station,
        raw,
    })
}

/// Snow from a METAR's weather: SN (light -SN, heavy +SN), or any
/// precipitation at or below freezing.
fn metar_snow(wx: &str, temp: Option<f64>) -> f32 {
    if wx.contains("SN") {
        return if wx.contains("+SN") { 0.9 } else if wx.contains("-SN") { 0.4 } else { 0.65 };
    }
    let precip = ["RA", "DZ", "PL", "GS", "GR", "SG", "UP"].iter().any(|p| wx.contains(p));
    if precip && temp.is_some_and(|t| t <= 0.0) { 0.5 } else { 0.0 }
}

/// "6+", "1/2", "1 1/2" statute miles.
fn sm(s: &str) -> Option<f64> {
    let s = s.trim().trim_end_matches('+');
    s.split_whitespace().map(|p| match p.split_once('/') {
        Some((a, b)) => Some(a.parse::<f64>().ok()? / b.parse::<f64>().ok()?),
        None => p.parse().ok(),
    }).sum()
}

/// Cloud layer thickness: towering with storms, a deck when overcast.
fn depth(clouds: f32, storms: f32) -> f32 {
    if storms > 0.3 { 6000.0 } else if clouds >= 0.75 { 2500.0 } else { 1800.0 }
}

/// The haze slider for a visibility (km). The sky model's clear air already
/// sees ~180 km; it cannot show less than ~11 km, so anything under ten
/// (mist, fog) takes the thickest haze.
fn haze(vis_km: Option<f64>, foggy: bool) -> f32 {
    match vis_km {
        None => 1.0,
        Some(v) if v >= 9.5 && !foggy => 1.0 + (2.5 * (1.0 - (v - 9.5) / 15.0)).clamp(0.0, 2.5) as f32,
        Some(v) => {
            let h = (3.912 / (v.max(0.1) * 1000.0) - 13.558e-6) / 8.396e-6;
            h.clamp(6.0, 40.0) as f32
        }
    }
}

/// Open-Meteo's model where there is no METAR near.
fn open_meteo(src: &Sources, (lat, lon): (f64, f64)) -> Result<Target, String> {
    let url = format!(
        "https://api.open-meteo.com/v1/forecast?latitude={lat:.3}&longitude={lon:.3}&current=temperature_2m,dew_point_2m,cloud_cover_low,cloud_cover_mid,wind_speed_10m,wind_direction_10m,wind_gusts_10m,visibility,weather_code&wind_speed_unit=ms"
    );
    let j = get_json(src, &url)?;
    from_open_meteo(&j).ok_or_else(|| "unreadable Open-Meteo answer".to_string())
}

fn from_open_meteo(j: &serde_json::Value) -> Option<Target> {
    let c = &j["current"];
    let f = |k: &str| c[k].as_f64();
    let elev = j["elevation"].as_f64().unwrap_or(0.0);
    let code = f("weather_code").unwrap_or(0.0) as u32;
    let low = (f("cloud_cover_low").unwrap_or(0.0) / 100.0) as f32;
    let mid = (f("cloud_cover_mid").unwrap_or(0.0) / 100.0) as f32;
    // Cumulus base from the spread: about 125 m per degree.
    let base = match (f("temperature_2m"), f("dew_point_2m")) {
        (Some(t), Some(d)) => Some(((t - d).max(0.0) * 125.0 + elev).clamp(300.0, 4000.0) as f32),
        _ => None,
    };
    let storms = match code {
        95..=99 => 0.75,
        80..=82 => 0.2,
        _ => 0.0,
    };
    let clouds = if low > 0.05 { low } else { mid * 0.6 }.max(0.03);
    let wind = f("wind_speed_10m")?;
    Some(Target {
        clouds,
        base_m: base,
        depth_m: depth(clouds, storms),
        alto: mid,
        storms,
        haze: haze(f("visibility").map(|v| v / 1000.0), matches!(code, 45 | 48)),
        snow: match code {
            71 | 85 => 0.4,
            73 | 77 => 0.65,
            75 | 86 => 0.9,
            51..=67 | 80..=82 if f("temperature_2m").is_some_and(|t| t <= 0.0) => 0.5,
            _ => 0.0,
        },
        wind,
        wind_from: f("wind_direction_10m"),
        gusts: f("wind_gusts_10m").map(|g| ((g - wind) / 8.0).clamp(0.0, 1.0)).unwrap_or(0.1),
        station: "Open-Meteo".into(),
        raw: String::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metar() {
        let m = serde_json::json!({"icaoId":"EGLL","rawOb":"METAR EGLL 151720Z 24015G28KT 9999 -TSRA FEW012 BKN025CB OVC080 12/10 Q1002","elev":25,"wdir":240,"wspd":15,"wgst":28,"visib":"6+","wxString":"-TSRA","clouds":[{"cover":"FEW","base":1200},{"cover":"BKN","base":2500},{"cover":"OVC","base":8000}]});
        let t = from_metar(&m).unwrap();
        assert_eq!(t.station, "EGLL");
        assert_eq!(t.wind_from, Some(240.0));
        assert!((t.wind - 15.0 / 1.94384).abs() < 1e-6);
        assert!(t.gusts > 0.8);
        assert_eq!(t.storms, 0.75);
        // FEW at 1200 ft and BKN at 2500 ft are one layer; OVC at 8000 ft is mid-level cloud.
        assert_eq!(t.clouds, 0.75);
        assert_eq!(t.alto, 1.0);
        assert!((t.base_m.unwrap() - (1200.0 / 3.28084 + 25.0) as f32).abs() < 0.5);
        assert!(t.haze < 2.0);
        assert_eq!(t.snow, 0.0);
        assert_eq!((metar_snow("-SN", Some(-2.0)), metar_snow("+SN BR", None), metar_snow("-RA", Some(-1.0)), metar_snow("RA", Some(4.0))), (0.4, 0.9, 0.5, 0.0));

        let cavok = serde_json::json!({"icaoId":"EBBR","rawOb":"METAR EBBR 151720Z 19010KT CAVOK 10/09 Q0999","elev":56,"wdir":190,"wspd":10,"visib":"6+","clouds":[]});
        let t = from_metar(&cavok).unwrap();
        assert_eq!((t.clouds, t.base_m, t.storms, t.haze), (0.03, None, 0.0, 1.0));

        let fog = serde_json::json!({"icaoId":"LFPG","rawOb":"METAR LFPG 150600Z VRB02KT 0300 FG VV001 05/05 Q1020","wdir":"VRB","wspd":2,"visib":0.25,"wxString":"FG","clouds":[{"cover":"OVX","base":100}]});
        let t = from_metar(&fog).unwrap();
        assert_eq!(t.wind_from, None);
        assert_eq!(t.haze, 40.0);
        assert_eq!(t.base_m, Some(300.0));
    }

    #[test]
    fn visibility_text() {
        assert_eq!(sm("6+"), Some(6.0));
        assert_eq!(sm("1/2"), Some(0.5));
        assert_eq!(sm("1 1/2"), Some(1.5));
    }

    #[test]
    fn open_meteo_model() {
        let j = serde_json::json!({"elevation":37.0,"current":{"temperature_2m":10.0,"dew_point_2m":8.0,"cloud_cover_low":100,"cloud_cover_mid":40,"wind_speed_10m":6.6,"wind_direction_10m":203,"wind_gusts_10m":13.1,"visibility":19420.0,"weather_code":53}});
        let t = from_open_meteo(&j).unwrap();
        assert_eq!((t.clouds, t.alto, t.storms), (1.0, 0.4, 0.0));
        assert_eq!(t.base_m, Some(300.0));
        assert_eq!(t.wind_from, Some(203.0));
        assert_eq!(t.station, "Open-Meteo");
    }
}
