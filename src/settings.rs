//! Settings kept between runs, in `<config dir>/fablegame/settings.json`
//! (`~/.config/fablegame` on Linux): the language, sound, camera, HUD and
//! graphics. Command-line flags still win over them for one run.

use crate::i18n::Lang;
use std::path::PathBuf;

#[derive(Clone, Debug, PartialEq)]
pub struct Settings {
    /// None: follow the system's language
    pub lang: Option<Lang>,
    pub sound_on: bool,
    pub volume: f32,
    /// 0 chase, 1 cockpit
    pub cam_mode: u8,
    pub cockpit_fov: f32,
    pub look_sens: f64,
    pub show_hud: bool,
    /// the minimap in the HUD's lower-right corner
    pub minimap: bool,
    pub taa: bool,
    /// real air traffic (see live.rs)
    pub live_traffic: bool,
    /// real weather (see livewx.rs)
    pub live_weather: bool,
    /// the clock follows the real time and date
    pub real_time: bool,
    /// None: never set, so the default preset for the GPU (`Graphics::default_for`)
    pub graphics: Option<Graphics>,
    /// multiplayer: the name shown to the others, and the relay's address
    /// (empty: the game's own, `mp::MP_SERVER`)
    pub mp_name: String,
    pub mp_server: String,
}

/// The quality presets of the Graphics settings. Each sets every option of
/// `Graphics` but the display ones (vsync, frame rate limit, fps counter).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Preset {
    Low,
    Medium,
    High,
    Ultra,
}

impl Preset {
    pub const ALL: [Preset; 4] = [Preset::Low, Preset::Medium, Preset::High, Preset::Ultra];
    pub fn parse(s: &str) -> Option<Preset> {
        Self::ALL.into_iter().find(|p| format!("{p:?}").eq_ignore_ascii_case(s))
    }
}

/// How a scene drawn smaller than the window (or jittered for anti-aliasing)
/// becomes the window's image: the game's own temporal AA and upscaling
/// (render.rs taau), or AMD's FSR 3 (fsr/), where the GPU can run it.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Upscaler {
    Taau,
    Fsr,
}

impl Upscaler {
    pub const ALL: [Upscaler; 2] = [Upscaler::Taau, Upscaler::Fsr];
    /// "taau" or "fsr" (settings file, --graphics ...,upscaler=fsr)
    pub fn code(self) -> &'static str {
        match self {
            Upscaler::Taau => "taau",
            Upscaler::Fsr => "fsr",
        }
    }
    pub fn parse(s: &str) -> Option<Upscaler> {
        Self::ALL.into_iter().find(|u| u.code().eq_ignore_ascii_case(s))
    }
}

/// Graphics options (the display ones: vsync, frame rate limit, fps counter,
/// fullscreen, and the dynamic resolution's target, are not part of presets). Levels count up from the cheapest: clouds 0..=2 (low,
/// medium, high), shadows 0..=3 (off, low, medium, high), terrain 0..=3 (low,
/// medium, high, ultra), water 0..=1 (low, high). Anti-aliasing stays `Settings::taa`.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Graphics {
    /// the 3D scene's resolution, a fraction of the window's (MIN_SCALE..1); with
    /// `dyn_res` the most it may rise to
    pub scale: f32,
    /// lower the resolution (down to DYN_MIN_SCALE) while the frame rate is under `target_fps`
    pub dyn_res: bool,
    pub target_fps: u32,
    pub clouds: u8,
    pub shadows: u8,
    pub ao: bool,
    pub bloom: bool,
    pub terrain: u8,
    /// the ocean's wave grids (ocean.rs `grid`): 0 Low (128 x 128, Low and
    /// Medium), 1 High (256 x 256, High and Ultra)
    pub water: u8,
    /// multisampled edges (MSAA 4x) under the temporal AA: sharper edges, at
    /// a high price on Apple's GPUs (all presets but Low)
    pub msaa: bool,
    /// what turns the jittered (and maybe smaller) scene into the window's
    /// image while the temporal AA is on
    pub upscaler: Upscaler,
    pub vsync: bool,
    /// frames a second at most, 0 = no limit
    pub fps_limit: u32,
    /// the frame rate in the top right corner (F4)
    pub show_fps: bool,
    /// borderless fullscreen (F11)
    pub fullscreen: bool,
}

/// The lowest render scale: below about half, the temporal upscaling
/// (render.rs `taau`) gathers the window's detail back over a few frames.
pub const MIN_SCALE: f32 = 0.4;
/// The lowest dynamic resolution may go (a third of the window's pixels a side).
pub const DYN_MIN_SCALE: f32 = 0.33;

/// The render scale of High quality on an integrated GPU (Apple silicon and
/// the like): it shares memory bandwidth and power with the CPU, and the
/// full-resolution scene is too much for it. The temporal AA and its
/// sharpening hide most of it.
pub const INTEGRATED_SCALE: f32 = 0.75;

/// What each terrain detail level splits tiles at (world.rs `lod_k`: lower = finer).
pub const TERRAIN_LOD: [f64; 4] = [0.95, 0.78, 0.62, 0.50];

impl Graphics {
    /// A preset's options; High is the game as it always looked (a smaller
    /// scene on an integrated GPU).
    pub fn preset(p: Preset, integrated: bool) -> Self {
        let (scale, clouds, shadows, fx, terrain) = match p {
            // (Low and Medium are sized for a fullscreen Retina window on an
            // M4 Pro, about 60 and 45 fps: the scene's pixels cost the most,
            // so they draw well under half of them and upscale temporally,
            // render.rs taau. Medium also lowers its resolution, down to
            // DYN_MIN_SCALE, while under 60 fps. Medium keeps MSAA: without
            // it the aircraft's edges stair-step in flight, where the
            // temporal AA's history does not hold on them.)
            Preset::Low => (MIN_SCALE, 0, 1, false, 0),
            Preset::Medium => (0.45, 1, 2, true, 1),
            Preset::High => (if integrated { INTEGRATED_SCALE } else { 1.0 }, 2, 3, true, 2),
            Preset::Ultra => (1.0, 2, 3, true, 3),
        };
        let msaa = p != Preset::Low;
        let dyn_res = p == Preset::Medium;
        let water = matches!(p, Preset::High | Preset::Ultra) as u8;
        Self { scale, dyn_res, target_fps: 60, clouds, shadows, ao: fx, bloom: fx, terrain, water, msaa, upscaler: Upscaler::Taau, vsync: true, fps_limit: 0, show_fps: false, fullscreen: false }
    }

    /// Before any are saved: Medium on an integrated GPU, High on others.
    pub fn default_for(integrated: bool) -> Self {
        Self::preset(if integrated { Preset::Medium } else { Preset::High }, integrated)
    }

    /// Sets a preset's options, keeping the display ones.
    pub fn apply(&mut self, p: Preset, integrated: bool) {
        *self = Self { vsync: self.vsync, fps_limit: self.fps_limit, show_fps: self.show_fps, fullscreen: self.fullscreen, target_fps: self.target_fps, ..Self::preset(p, integrated) };
    }

    /// The preset these options (and anti-aliasing) are, if any: anything else is "Custom".
    pub fn preset_of(&self, taa: bool, integrated: bool) -> Option<Preset> {
        Preset::ALL.into_iter().find(|&p| {
            let mut q = *self;
            q.apply(p, integrated);
            taa && q == *self
        })
    }

    /// Field by field as `Settings::keep_run`.
    pub fn keep_run(&self, start: &Graphics, saved: &Graphics) -> Graphics {
        Graphics {
            scale: kept(self.scale, start.scale, saved.scale),
            dyn_res: kept(self.dyn_res, start.dyn_res, saved.dyn_res),
            target_fps: kept(self.target_fps, start.target_fps, saved.target_fps),
            clouds: kept(self.clouds, start.clouds, saved.clouds),
            shadows: kept(self.shadows, start.shadows, saved.shadows),
            ao: kept(self.ao, start.ao, saved.ao),
            bloom: kept(self.bloom, start.bloom, saved.bloom),
            terrain: kept(self.terrain, start.terrain, saved.terrain),
            water: kept(self.water, start.water, saved.water),
            msaa: kept(self.msaa, start.msaa, saved.msaa),
            upscaler: kept(self.upscaler, start.upscaler, saved.upscaler),
            vsync: kept(self.vsync, start.vsync, saved.vsync),
            fps_limit: kept(self.fps_limit, start.fps_limit, saved.fps_limit),
            show_fps: kept(self.show_fps, start.show_fps, saved.show_fps),
            fullscreen: kept(self.fullscreen, start.fullscreen, saved.fullscreen),
        }
    }

    fn from_json(j: &serde_json::Value) -> Self {
        let d = Self::preset(Preset::Medium, false);
        let b = |k: &str, v: bool| j[k].as_bool().unwrap_or(v);
        let n = |k: &str, v: u32| j[k].as_u64().map(|x| x as u32).unwrap_or(v);
        let g = Self {
            scale: j["scale"].as_f64().filter(|x| x.is_finite()).map(|x| x as f32).unwrap_or(d.scale).clamp(MIN_SCALE, 1.0),
            dyn_res: b("dynamic_resolution", d.dyn_res),
            target_fps: n("target_fps", d.target_fps).clamp(20, 240),
            clouds: n("clouds", d.clouds as u32).min(2) as u8,
            shadows: n("shadows", d.shadows as u32).min(3) as u8,
            ao: b("ambient_occlusion", d.ao),
            bloom: b("bloom", d.bloom),
            terrain: n("terrain_detail", d.terrain as u32).min(3) as u8,
            // (before it was an option the waves were always the High ones)
            water: n("water", 1).min(1) as u8,
            msaa: b("msaa", true),
            upscaler: j["upscaler"].as_str().and_then(Upscaler::parse).unwrap_or(d.upscaler),
            vsync: b("vsync", d.vsync),
            fps_limit: n("fps_limit", d.fps_limit).min(1000),
            show_fps: b("show_fps", d.show_fps),
            fullscreen: b("fullscreen", d.fullscreen),
        };
        // Saved before MSAA or the water quality were options: a preset's
        // own setting for them if the rest is that preset, so it stays that preset.
        if j["msaa"].is_null() || j["water"].is_null() {
            for integrated in [true, false] {
                for p in Preset::ALL {
                    let pp = Self::preset(p, integrated);
                    let h = Self { msaa: if j["msaa"].is_null() { pp.msaa } else { g.msaa }, water: if j["water"].is_null() { pp.water } else { g.water }, ..g };
                    if h.preset_of(true, integrated) == Some(p) {
                        return h;
                    }
                }
            }
        }
        g
    }

    fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "scale": self.scale,
            "dynamic_resolution": self.dyn_res,
            "target_fps": self.target_fps,
            "clouds": self.clouds,
            "shadows": self.shadows,
            "ambient_occlusion": self.ao,
            "bloom": self.bloom,
            "terrain_detail": self.terrain,
            "water": self.water,
            "msaa": self.msaa,
            "upscaler": self.upscaler.code(),
            "vsync": self.vsync,
            "fps_limit": self.fps_limit,
            "show_fps": self.show_fps,
            "fullscreen": self.fullscreen,
        })
    }
}

impl Default for Settings {
    fn default() -> Self {
        Self { lang: None, sound_on: true, volume: 1.0, cam_mode: 0, cockpit_fov: crate::cockpit::FOV, look_sens: 1.0, show_hud: true, minimap: true, taa: true, live_traffic: true, live_weather: true, real_time: true, graphics: None, mp_name: String::new(), mp_server: String::new() }
    }
}

/// Where the game keeps its settings: `$FABLE_CONFIG_DIR` when set (test
/// and benchmark runs point it elsewhere so they never touch the player's
/// own), else `<config dir>/fablegame`.
pub fn config_dir() -> Option<PathBuf> {
    match std::env::var_os("FABLE_CONFIG_DIR") {
        Some(d) if !d.is_empty() => Some(PathBuf::from(d)),
        _ => dirs::config_dir().map(|d| d.join("fablegame")),
    }
}

fn path() -> Option<PathBuf> {
    config_dir().map(|d| d.join("settings.json"))
}

/// One field of `keep_run`: what the player changed since the start, else what was saved.
fn kept<T: PartialEq + Copy>(cur: T, start: T, saved: T) -> T {
    if cur == start { saved } else { cur }
}

/// `kept` for text.
fn kept_s(cur: &str, start: &str, saved: &str) -> String {
    if cur == start { saved } else { cur }.to_string()
}

impl Settings {
    /// The saved settings; the defaults for anything missing or unreadable.
    pub fn load() -> Self {
        let Some(j) = path().and_then(|p| std::fs::read(p).ok()).and_then(|d| serde_json::from_slice::<serde_json::Value>(&d).ok()) else {
            return Self::default();
        };
        Self::from_json(&j)
    }

    fn from_json(j: &serde_json::Value) -> Self {
        let d = Self::default();
        let b = |k: &str, v: bool| j[k].as_bool().unwrap_or(v);
        let f = |k: &str, v: f64| j[k].as_f64().filter(|x| x.is_finite()).unwrap_or(v);
        Self {
            lang: j["language"].as_str().and_then(Lang::parse),
            sound_on: b("sound", d.sound_on),
            volume: f("volume", d.volume as f64).clamp(0.0, 1.0) as f32,
            cam_mode: if j["camera"].as_str() == Some("cockpit") { 1 } else { 0 },
            cockpit_fov: f("cockpit_fov", d.cockpit_fov as f64).clamp(35.0, 85.0) as f32,
            look_sens: f("mouse_look_speed", d.look_sens).clamp(0.25, 3.0),
            show_hud: b("hud", d.show_hud),
            minimap: b("minimap", d.minimap),
            taa: b("taa", d.taa),
            live_traffic: b("live_traffic", d.live_traffic),
            live_weather: b("live_weather", d.live_weather),
            real_time: b("real_time", d.real_time),
            graphics: j["graphics"].is_object().then(|| Graphics::from_json(&j["graphics"])),
            mp_name: j["mp_name"].as_str().map(crate::mp::clean_name).unwrap_or_default(),
            mp_server: j["mp_server"].as_str().unwrap_or("").trim().to_string(),
        }
    }

    fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "language": self.lang.map(|l| l.code()),
            "sound": self.sound_on,
            "volume": self.volume,
            "camera": if self.cam_mode == 1 { "cockpit" } else { "chase" },
            "cockpit_fov": self.cockpit_fov,
            "mouse_look_speed": self.look_sens,
            "hud": self.show_hud,
            "minimap": self.minimap,
            "taa": self.taa,
            "live_traffic": self.live_traffic,
            "live_weather": self.live_weather,
            "real_time": self.real_time,
            "graphics": self.graphics.map(|g| g.to_json()),
            "mp_name": self.mp_name,
            "mp_server": self.mp_server,
        })
    }

    /// What to save of `self` (the settings now) when the run began as `start`
    /// over the saved `saved`: a value the player has not changed since the
    /// start is the saved one, so one-run command-line flags (--no-taa,
    /// --time, --no-vsync, --graphics...) never reach the file.
    pub fn keep_run(&self, start: &Settings, saved: &Settings) -> Settings {
        Settings {
            lang: kept(self.lang, start.lang, saved.lang),
            sound_on: kept(self.sound_on, start.sound_on, saved.sound_on),
            volume: kept(self.volume, start.volume, saved.volume),
            cam_mode: kept(self.cam_mode, start.cam_mode, saved.cam_mode),
            cockpit_fov: kept(self.cockpit_fov, start.cockpit_fov, saved.cockpit_fov),
            look_sens: kept(self.look_sens, start.look_sens, saved.look_sens),
            show_hud: kept(self.show_hud, start.show_hud, saved.show_hud),
            minimap: kept(self.minimap, start.minimap, saved.minimap),
            taa: kept(self.taa, start.taa, saved.taa),
            live_traffic: kept(self.live_traffic, start.live_traffic, saved.live_traffic),
            live_weather: kept(self.live_weather, start.live_weather, saved.live_weather),
            real_time: kept(self.real_time, start.real_time, saved.real_time),
            // (merged by the caller with `Graphics::keep_run`: before any are
            // saved, the run starts from the GPU's default preset)
            graphics: self.graphics,
            mp_name: kept_s(&self.mp_name, &start.mp_name, &saved.mp_name),
            mp_server: kept_s(&self.mp_server, &start.mp_server, &saved.mp_server),
        }
    }

    /// Writes them (through a temporary file, so a crash cannot leave half a file).
    pub fn save(&self) {
        let Some(p) = path() else { return };
        if let Some(dir) = p.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        let tmp = p.with_extension("json.tmp");
        let data = serde_json::to_vec_pretty(&self.to_json()).unwrap_or_default();
        if let Err(e) = std::fs::write(&tmp, data).and_then(|_| std::fs::rename(&tmp, &p)) {
            log::warn!("saving settings to {}: {e}", p.display());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let s = Settings { lang: Some(Lang::Fr), sound_on: false, volume: 0.4, cam_mode: 1, cockpit_fov: 50.0, look_sens: 1.5, show_hud: false, minimap: false, taa: false, live_traffic: false, live_weather: false, real_time: false, graphics: None, mp_name: "Sam".into(), mp_server: "ws://127.0.0.1:8787".into() };
        assert_eq!(Settings::from_json(&s.to_json()), s);
        let g = Graphics { scale: 0.65, dyn_res: true, target_fps: 45, clouds: 0, shadows: 2, ao: false, bloom: true, terrain: 3, water: 0, msaa: true, upscaler: Upscaler::Fsr, vsync: false, fps_limit: 60, show_fps: true, fullscreen: true };
        let s = Settings { graphics: Some(g), ..s };
        assert_eq!(Settings::from_json(&s.to_json()), s);
        // Presets are recognised again from their options, and not past a change.
        for p in Preset::ALL {
            let mut g = Graphics::preset(p, true);
            assert_eq!(g.preset_of(true, true), Some(p));
            assert_eq!(g.preset_of(false, true), None);
            g.bloom = !g.bloom;
            assert_eq!(g.preset_of(true, true), None);
        }
        // Missing or odd values fall back to the defaults.
        let j = serde_json::json!({ "volume": 7.0, "language": "xx" });
        let t = Settings::from_json(&j);
        assert_eq!((t.volume, t.lang, t.taa, t.minimap, t.graphics), (1.0, None, true, true, None));
        // Saved before MSAA was an option: a preset stays that preset.
        let mut j = Graphics::preset(Preset::Medium, true).to_json();
        j.as_object_mut().unwrap().remove("msaa");
        assert_eq!(Graphics::from_json(&j).preset_of(true, true), Some(Preset::Medium));
        // Saved before the water quality was an option: presets stay presets
        // (Low and Medium get the Low waves, High and Ultra the High ones).
        for p in Preset::ALL {
            let mut j = Graphics::preset(p, true).to_json();
            j.as_object_mut().unwrap().remove("water");
            let h = Graphics::from_json(&j);
            assert_eq!((h.preset_of(true, true), h.water), (Some(p), matches!(p, Preset::High | Preset::Ultra) as u8));
        }
        let t = Settings::from_json(&serde_json::json!({ "graphics": { "clouds": 9, "scale": 0.1, "upscaler": "xx" } }));
        assert_eq!(t.graphics.map(|g| (g.clouds, g.scale, g.shadows, g.upscaler)), Some((2, MIN_SCALE, 2, Upscaler::Taau)));
        // One-run flags are not kept; the player's own changes are.
        let saved = Settings::default();
        let start = Settings { taa: false, real_time: false, ..saved.clone() };
        let now = Settings { show_hud: false, ..start.clone() };
        let k = now.keep_run(&start, &saved);
        assert_eq!((k.taa, k.real_time, k.show_hud), (true, true, false));
        // (--mp-name and --mp-server likewise; a name typed in the game is kept)
        let start = Settings { mp_name: "B".into(), mp_server: "ws://127.0.0.1:8787".into(), ..saved.clone() };
        let k = start.keep_run(&start, &saved);
        assert_eq!((k.mp_name.as_str(), k.mp_server.as_str()), ("", ""));
        let k = Settings { mp_name: "Ace".into(), ..start.clone() }.keep_run(&start, &saved);
        assert_eq!((k.mp_name.as_str(), k.mp_server.as_str()), ("Ace", ""));
        let g0 = Graphics { vsync: true, ..g };
        let k = Graphics { clouds: 2, ..g0 }.keep_run(&g0, &g);
        assert_eq!((k.vsync, k.clouds), (false, 2));
    }
}
