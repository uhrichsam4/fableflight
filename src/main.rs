mod airframe;
mod airfield;
mod airmodels;
mod airports;
mod audio;
mod buildings;
mod aircraft;
mod cockpit;
mod cars;
mod crash;
mod damage;
mod destruct;
mod fetch;
mod fluid;
mod fsr;
mod geo;
mod gpuprof;
mod i18n;
mod live;
mod livewx;
mod settings;
mod gauges;
mod gmaps;
mod tiles3d;
mod impostor;
mod materials;
mod minimap;
mod logo;
mod menu;
mod padui;
mod pave;
mod photonight;
mod relight;
mod weather;
mod mvt;
mod plane;
mod render;
mod roads;
mod route;
mod skin;
mod sources;
mod traffic;
mod trees;
mod ocean;
mod shore;
mod world;

#[derive(Copy, Clone, PartialEq)]
enum Mode {
    Menu,
    Fly,
}

/// What the gamepad's buttons work at the moment, besides flying.
#[derive(Copy, Clone, PartialEq, Eq)]
enum Nav {
    Off,
    Globe,
    Pause,
    Panel,
}

/// What was chosen in the in-flight menu, carried out once the UI is drawn.
#[derive(Copy, Clone)]
enum PauseAct {
    Resume,
    Respawn,
    GoTo,
    FlyTo(f64, f64, &'static str),
    /// start at an airport: which, on final approach (else on the runway)
    Airport(u32, bool),
    /// start the planned route: on the departure runway (else above the place)
    RouteStart,
    /// fly the planned route from where the aircraft is
    RouteFly,
    Globe,
    Quit,
}

use crash::Crash;
use damage::Tear;
use glam::{DMat4, DQuat, DVec2, DVec3, Mat4, Vec2, Vec3};
use i18n::{t, tr};
use padui::Hint;
use plane::{Controls, Plane};
use render::{DrawObj, Frame, Globals, PSim, Renderer};
use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

const SUN_E: f32 = 10.0;
/// Falling snow: the distance moved through the flakes wraps at this many
/// metres, a whole number of flake cells in each layer (composite.wgsl `snowfall`).
const SNOW_WRAP: f64 = 256.0 * std::f64::consts::TAU / 96.0;
/// the route's course line and markers
const MAGENTA: egui::Color32 = egui::Color32::from_rgb(255, 80, 230);
/// The aircraft's own shadow map: half width (m), how far back toward the sun
/// it starts, and how deep it reaches.
const PSHADOW_R: f32 = 6.2;
const PSHADOW_BACK: f32 = 40.0;
const PSHADOW_DEPTH: f32 = 1600.0;
/// How deep the aircraft's own shadow map reaches: at least `PSHADOW_DEPTH`,
/// and past the far side of an aircraft too big for that (it starts
/// `PSHADOW_BACK` scaled up sunward of it).
fn pshadow_depth(shadow_r: f32) -> f32 {
    PSHADOW_DEPTH.max(PSHADOW_BACK * shadow_r / PSHADOW_R + 2.0 * shadow_r)
}
/// Moonlight relative to sunlight. Far brighter than the real ratio, so a
/// moonlit night reads like a film's night scene instead of a black screen.
const MOON_E: f32 = SUN_E * 0.0035;
/// What a 3D tiles detail level costs, as measured over San Francisco (see
/// docs/photorealistic-3d-tiles-plan.md).
fn detail_hint(d: gmaps::Detail) -> &'static str {
    match d {
        gmaps::Detail::Low => t!("Low: softer buildings, 4 km around the aircraft. About 45 MB a minute over a city (a 6 Mbit/s connection keeps up), 250 MB of GPU memory.", "Faible : bâtiments plus flous, 4 km autour de l'avion. Environ 45 Mo par minute au-dessus d'une ville (une connexion à 6 Mbit/s suffit), 250 Mo de mémoire GPU."),
        gmaps::Detail::Medium => t!("Medium: 6 km around the aircraft. About 75 MB a minute over a city (10 Mbit/s), 400 MB of GPU memory.", "Moyen : 6 km autour de l'avion. Environ 75 Mo par minute au-dessus d'une ville (10 Mbit/s), 400 Mo de mémoire GPU."),
        gmaps::Detail::High => t!("High (default): 9 km around the aircraft. About 90 MB a minute over a city (12 Mbit/s), 640 MB of GPU memory.", "Élevé (par défaut) : 9 km autour de l'avion. Environ 90 Mo par minute au-dessus d'une ville (12 Mbit/s), 640 Mo de mémoire GPU."),
        gmaps::Detail::Ultra => t!("Ultra: sharpest close up, 9 km around the aircraft. About 240 MB a minute over a city (32 Mbit/s), 1 GB of GPU memory.", "Ultra : le plus net de près, 9 km autour de l'avion. Environ 240 Mo par minute au-dessus d'une ville (32 Mbit/s), 1 Go de mémoire GPU."),
    }
}

fn attribution_default() -> &'static str {
    t!("Imagery © Esri, Maxar, Earthstar Geographics", "Imagerie © Esri, Maxar, Earthstar Geographics")
}
fn attribution_rest() -> &'static str {
    t!(
        "Elevation: Mapzen / AWS Terrain Tiles  ·  Buildings, roads, land cover & water © OpenStreetMap contributors, OpenFreeMap",
        "Relief : Mapzen / AWS Terrain Tiles  ·  Bâtiments, routes, occupation du sol et eau © contributeurs OpenStreetMap, OpenFreeMap"
    )
}

/// Data sizes ("12.3 MB"), with the French units ("12.3 Mo") in French.
fn bytes(n: u64) -> String {
    let s = gmaps::bytes(n);
    if i18n::is_fr() { s.replace(" GB", " Go").replace(" MB", " Mo").replace(" KB", " Ko").replace(" B", " o") } else { s }
}

/// An aircraft's model and its meshes on the GPU.
struct Loaded {
    spec: &'static aircraft::Spec,
    model: airframe::Model,
    piece_mesh: Vec<usize>,
    group_mesh: Vec<usize>,
    piece_cut: Vec<bool>,
    group_pivots: Vec<(Vec3, f32)>,
    /// what the anti-aliasing takes to be flying along with the camera:
    /// everything within the aircraft's bounds, and a little around them for
    /// the gear, flaps and control surfaces as they move
    plane_box: [[f32; 4]; 2],
}

impl Loaded {
    fn build(spec: &'static aircraft::Spec, r: &mut Renderer) -> Loaded {
        Self::upload(spec, Self::make(spec), r)
    }

    /// The CPU half (mesh generation): safe to run off the main thread.
    fn make(spec: &'static aircraft::Spec) -> airframe::Model {
        let t0 = Instant::now();
        let model = (spec.build)();
        log::info!("{}: {} pieces, {} vertices, built in {:.0} ms", spec.name, model.pieces.len(), model.pieces.iter().map(|p| p.mesh.v.len()).sum::<usize>(), t0.elapsed().as_secs_f64() * 1000.0);
        model
    }

    fn upload(spec: &'static aircraft::Spec, model: airframe::Model, r: &mut Renderer) -> Loaded {
        let piece_mesh: Vec<usize> = model.pieces.iter().map(|p| r.add_mesh(&p.mesh.v, &p.mesh.i)).collect();
        let piece_cut: Vec<bool> = model.pieces.iter().map(|p| p.mesh.v.iter().any(|v| airframe::cut_by(v.mat[1]))).collect();
        let mut group_mesh: Vec<usize> = model.groups.iter().map(|m| r.add_mesh(&m.v, &m.i)).collect();
        for m in crash::shards() {
            group_mesh.push(r.add_mesh(&m.v, &m.i));
        }
        // Each group's wreckage tumbles about the middle of its bounds.
        let group_pivots = model
            .groups
            .iter()
            .map(|m| {
                let (lo, hi) = m.v.iter().fold((Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)), |(lo, hi), v| (lo.min(v.pos.into()), hi.max(v.pos.into())));
                if m.v.is_empty() { (Vec3::ZERO, 0.5) } else { ((lo + hi) * 0.5, ((hi - lo).max_element() * 0.35).max(0.3)) }
            })
            .collect();
        let (c, h) = model.bounds();
        let plane_box = [c.extend(0.0).to_array(), (h + Vec3::splat(0.25 * spec.span / 5.5)).extend(0.0).to_array()];
        Loaded { spec, model, piece_mesh, group_mesh, piece_cut, group_pivots, plane_box }
    }
}

/// A graphics preset's name, in the interface language.
fn preset_name(p: settings::Preset) -> &'static str {
    match p {
        settings::Preset::Low => t!("Low", "Faible"),
        settings::Preset::Medium => t!("Medium", "Moyen"),
        settings::Preset::High => t!("High", "Élevé"),
        settings::Preset::Ultra => "Ultra",
    }
}

struct Args {
    lat: f64,
    lon: f64,
    alt: Option<f64>,
    /// starting height above the ground, instead of `alt` above sea level
    agl: Option<f64>,
    heading: f64,
    place: Option<String>,
    max_zoom: u8,
    hour: f64,
    clouds: f32,
    /// cumulonimbus storm cell amount 0..1
    storms: Option<f32>,
    time_rate: Option<f64>,
    alto: Option<f32>,
    /// wind in knots and the direction it blows from, haze, day of year; open the F3 panel
    wind: Option<(f64, f64)>,
    haze: Option<f32>,
    snow: Option<f32>,
    cloud_base: Option<f32>,
    day: Option<f64>,
    weather_panel: bool,
    vsync: bool,
    taa: bool,
    /// the 3D scene's resolution as a fraction of the window's (default: the
    /// Graphics settings', settings::INTEGRATED_SCALE or 1 at High quality)
    render_scale: Option<f32>,
    /// a graphics quality preset for this run only (benchmarks), and options
    /// set over it (`--graphics low,shadows=0,ao=1`)
    graphics: Option<settings::Preset>,
    gfx_set: Vec<(String, f32)>,
    /// borderless fullscreen for this run
    fullscreen: bool,
    /// log the frame rate once a second (average and slowest frame)
    fps_log: bool,
    /// draw without showing anything (benchmarks and scripted shots with the screen off)
    headless: bool,
    offline: bool,
    shots: Vec<(f64, PathBuf, Option<(f64, f64, f64)>)>,
    /// record frames + audio to a directory: (dir, start, seconds), fixed 30 fps steps
    record: Option<(PathBuf, f64, f64)>,
    crash_at: Option<f64>,
    /// from this many seconds on, full throttle with the boost key held (the afterburners), for scripted screenshots and recordings
    boost_at: Option<f64>,
    /// game time stands still for this many wall-clock seconds at the start (tiles stream in)
    hold: f64,
    size: (u32, u32),
    imagery: String,
    /// Google Maps API key (else GOOGLE_MAPS_API_KEY, else the saved one)
    google_key: Option<String>,
    /// switch Google imagery on or off for good from the command line
    google: Option<bool>,
    /// and Google's photorealistic 3D tiles
    photoreal: Option<bool>,
    /// how much of the 3D tiles to stream
    photo_detail: Option<gmaps::Detail>,
    cam: Option<(f64, f64, f64)>,
    cockpit: bool,
    /// cockpit vertical field of view (deg)
    fov: Option<f32>,
    mute: bool,
    /// interface language (else the saved one, else the system's)
    lang: Option<i18n::Lang>,
    /// start on the globe menu (default unless a place or scripted run is given)
    menu: Option<bool>,
    menu_view: Option<(f64, f64, f64)>,
    menu_go: Option<f64>,
    /// start already damaged (see `Damage::apply_spec`), for trying the damage model out
    damage: Option<String>,
    /// open the in-flight menu on this page after this many seconds, for scripted screenshots
    pause_menu: Option<(menu::Page, f64)>,
    /// --route KSFO,KOAK: a planned route (started at its departure unless --airport or --place says otherwise)
    route: Option<String>,
    /// --sim-speed 1|2|4|8: start with time acceleration (for testing)
    sim_speed: u32,
    /// start at an airport (ICAO or IATA code), on a runway or on approach
    airport: Option<String>,
    runway: Option<String>,
    approach: bool,
    /// which aircraft to fly (`aircraft::by_key`)
    aircraft: Option<String>,
    /// start with the HUD hidden (H), for screenshots
    no_hud: bool,
    /// open the large map (M) after this many seconds, for scripted screenshots
    map_open: Option<f64>,
    /// --map-pick KSFO,KOAK: airports "clicked" on the large map as it opens (scripted screenshots)
    map_pick: Vec<String>,
}

fn parse_args() -> Args {
    let mut a = Args {
        lat: 37.8085,
        lon: -122.4650,
        alt: None,
        agl: None,
        heading: 95.0,
        place: None,
        max_zoom: 17,
        hour: 16.6,
        clouds: 0.5,
        storms: None,
        time_rate: None,
        alto: None,
        wind: None,
        haze: None,
        snow: None,
        cloud_base: None,
        day: None,
        weather_panel: false,
        vsync: true,
        taa: true,
        render_scale: None,
        graphics: None,
        gfx_set: Vec::new(),
        fullscreen: false,
        fps_log: false,
        headless: false,
        offline: false,
        shots: Vec::new(),
        record: None,
        crash_at: None,
        boost_at: None,
        hold: 0.0,
        size: (1600, 900),
        imagery: "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}".into(),
        google_key: std::env::var("GOOGLE_MAPS_API_KEY").ok(),
        google: None,
        photoreal: None,
        photo_detail: None,
        cam: None,
        cockpit: false,
        fov: None,
        mute: false,
        lang: None,
        menu: None,
        menu_view: None,
        menu_go: None,
        damage: None,
        pause_menu: None,
        map_open: None,
        map_pick: Vec::new(),
        route: None,
        sim_speed: 1,
        airport: None,
        runway: None,
        approach: false,
        aircraft: None,
        no_hud: false,
    };
    let v: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    let num = |v: &Vec<String>, i: usize| v.get(i + 1).and_then(|s| s.parse::<f64>().ok());
    while i < v.len() {
        let mut used = 2;
        match v[i].as_str() {
            "--lat" => {
                a.lat = num(&v, i).unwrap_or(a.lat);
                a.menu.get_or_insert(false);
            }
            "--lon" => {
                a.lon = num(&v, i).unwrap_or(a.lon);
                a.menu.get_or_insert(false);
            }
            "--menu-go" => a.menu_go = num(&v, i),
            "--menu-view" => {
                // lat,lon,altitude of the menu camera, for scripted screenshots
                let p: Vec<f64> = v.get(i + 1).map(|s| s.split(',').filter_map(|x| x.parse().ok()).collect()).unwrap_or_default();
                if p.len() == 3 {
                    a.menu_view = Some((p[0], p[1], p[2]));
                }
            }
            "--menu" => {
                a.menu = Some(true);
                used = 1;
            }
            "--no-menu" => {
                a.menu = Some(false);
                used = 1;
            }
            "--cockpit" => {
                a.cockpit = true;
                used = 1;
            }
            "--fov" => {
                a.fov = v.get(i + 1).and_then(|s| s.parse().ok());
                used = 2;
            }
            "--airport" => {
                a.airport = v.get(i + 1).cloned();
                a.menu.get_or_insert(false);
            }
            "--runway" => a.runway = v.get(i + 1).cloned(),
            "--sim-speed" => a.sim_speed = num(&v, i).map(|k| (k as u32).clamp(1, 8).next_power_of_two().min(8)).unwrap_or(1),
            "--route" => {
                a.route = v.get(i + 1).cloned();
                a.menu.get_or_insert(false);
                used = 2;
            }
            "--aircraft" => {
                a.aircraft = v.get(i + 1).cloned();
                if let Some(k) = &a.aircraft {
                    if aircraft::by_key(k).is_none() {
                        eprintln!("unknown aircraft {k} (try {})", aircraft::ALL.iter().map(|s| s.keys[0]).collect::<Vec<_>>().join(", "));
                    }
                }
            }
            "--approach" => {
                a.approach = true;
                used = 1;
            }
            "--alt" => a.alt = num(&v, i),
            "--agl" => a.agl = num(&v, i),
            "--heading" => a.heading = num(&v, i).unwrap_or(a.heading),
            "--place" => {
                a.place = v.get(i + 1).cloned();
                a.menu.get_or_insert(false);
            }
            "--max-zoom" => a.max_zoom = num(&v, i).unwrap_or(17.0).clamp(8.0, 19.0) as u8,
            "--time" => a.hour = num(&v, i).unwrap_or(a.hour),
            "--clouds" => a.clouds = num(&v, i).unwrap_or(0.5) as f32,
            "--storms" => a.storms = num(&v, i).map(|x| x as f32),
            "--time-rate" => a.time_rate = num(&v, i),
            "--alto" => a.alto = num(&v, i).map(|x| x as f32),
            "--wind" => {
                // knots,from-degrees
                let p: Vec<f64> = v.get(i + 1).map(|s| s.split(',').filter_map(|x| x.parse().ok()).collect()).unwrap_or_default();
                if p.len() == 2 {
                    a.wind = Some((p[0], p[1]));
                }
            }
            "--haze" => a.haze = num(&v, i).map(|h| h as f32),
            "--snow" => a.snow = num(&v, i).map(|x| x as f32),
            "--cloud-base" => a.cloud_base = num(&v, i).map(|x| x as f32),
            "--day" => a.day = num(&v, i),
            "--weather" => {
                a.weather_panel = true;
                used = 1;
            }
            "--imagery" => a.imagery = v.get(i + 1).cloned().unwrap_or(a.imagery.clone()),
            "--crash-at" => a.crash_at = num(&v, i),
            "--boost-at" => a.boost_at = num(&v, i),
            "--hold" => a.hold = num(&v, i).unwrap_or(0.0),
            "--damage" => a.damage = v.get(i + 1).cloned(),
            "--map-open" => a.map_open = num(&v, i),
            "--map-pick" => a.map_pick = v.get(i + 1).map(|s| s.split(',').map(|c| c.trim().to_string()).collect()).unwrap_or_default(),
            "--pause-menu" => {
                // page[@seconds]: open the flight panel (Tab) or the pause menu (Esc) on that page, for scripted screenshots
                let f: Vec<&str> = v.get(i + 1).map(|s| s.split('@').collect()).unwrap_or_default();
                if let Some(page) = f.first().and_then(|p| menu::Page::parse(p)) {
                    a.pause_menu = Some((page, f.get(1).and_then(|t| t.parse().ok()).unwrap_or(0.0)));
                }
            }
            "--width" => a.size.0 = num(&v, i).unwrap_or(1600.0) as u32,
            "--height" => a.size.1 = num(&v, i).unwrap_or(900.0) as u32,
            "--cam" => {
                // yaw,pitch,distance of the chase camera, for scripted screenshots
                let p: Vec<f64> = v.get(i + 1).map(|s| s.split(',').filter_map(|x| x.parse().ok()).collect()).unwrap_or_default();
                if p.len() == 3 {
                    a.cam = Some((p[0], p[1], p[2]));
                }
            }
            "--shot" => {
                // path@seconds[@yaw,pitch,distance]
                let f: Vec<&str> = v.get(i + 1).map(|s| s.split('@').collect()).unwrap_or_default();
                if f.len() >= 2 {
                    let c: Vec<f64> = f.get(2).map(|s| s.split(',').filter_map(|x| x.parse().ok()).collect()).unwrap_or_default();
                    a.shots.push((f[1].parse().unwrap_or(5.0), PathBuf::from(f[0]), if c.len() == 3 { Some((c[0], c[1], c[2])) } else { None }));
                }
                a.menu.get_or_insert(false);
            }
            "--record" => {
                // dir@start@seconds: every frame steps 1/30 s; PNG frames + audio.wav
                let f: Vec<&str> = v.get(i + 1).map(|s| s.split('@').collect()).unwrap_or_default();
                if f.len() == 3 {
                    a.record = Some((PathBuf::from(f[0]), f[1].parse().unwrap_or(0.0), f[2].parse().unwrap_or(10.0)));
                }
                a.menu.get_or_insert(false);
            }
            "--no-vsync" => {
                a.vsync = false;
                used = 1;
            }
            "--fullscreen" => {
                a.fullscreen = true;
                used = 1;
            }
            "--no-taa" => {
                a.taa = false;
                used = 1;
            }
            "--render-scale" => a.render_scale = num(&v, i).map(|s| s as f32),
            "--graphics" => {
                let f: Vec<&str> = v.get(i + 1).map(|s| s.split(',').collect()).unwrap_or_default();
                a.graphics = f.first().and_then(|s| settings::Preset::parse(s));
                // (upscaler=taau|fsr: its index in Upscaler::ALL; water=low|high: 0 or 1)
                let val = |k: &str, x: &str| match k {
                    "upscaler" => settings::Upscaler::ALL.iter().position(|u| Some(*u) == settings::Upscaler::parse(x)).map(|i| i as f32),
                    "water" => ["low", "high"].iter().position(|w| w.eq_ignore_ascii_case(x)).map(|i| i as f32).or_else(|| x.parse().ok()),
                    _ => x.parse().ok(),
                };
                a.gfx_set = f.iter().skip(1).filter_map(|kv| kv.split_once('=')).filter_map(|(k, x)| Some((k.to_string(), val(k, x)?))).collect();
            }
            "--headless" => {
                a.headless = true;
                used = 1;
            }
            // (read by the renderer itself: per-pass GPU times, gpuprof.rs)
            "--gpu-profile" => used = 1,
            "--fps-log" => {
                a.fps_log = true;
                used = 1;
            }
            "--no-hud" => {
                a.no_hud = true;
                used = 1;
            }
            "--photo-detail" => {
                a.photo_detail = Some(v.get(i + 1).and_then(|s| gmaps::Detail::parse(s)).unwrap_or_else(|| {
                    eprintln!("--photo-detail takes low, medium, high or ultra");
                    std::process::exit(2)
                }));
            }
            "--no-pave" => {
                pave::ON.store(false, std::sync::atomic::Ordering::Relaxed);
                used = 1;
            }
            "--mute" => {
                a.mute = true;
                used = 1;
            }
            "--lang" => {
                a.lang = Some(v.get(i + 1).and_then(|s| i18n::Lang::parse(s)).unwrap_or_else(|| {
                    eprintln!("--lang takes en or fr");
                    std::process::exit(2)
                }));
            }
            "--offline" => {
                a.offline = true;
                used = 1;
            }
            "--google-key" => a.google_key = v.get(i + 1).cloned(),
            "--google" => {
                a.google = Some(true);
                used = 1;
            }
            "--no-google" => {
                a.google = Some(false);
                used = 1;
            }
            "--photoreal" => {
                a.photoreal = Some(true);
                used = 1;
            }
            "--no-photoreal" => {
                a.photoreal = Some(false);
                used = 1;
            }
            "--help" | "-h" => {
                println!("Samiam's Flight Sim\nfablegame [--menu | --no-menu] [--place \"city or address\"] [--lat D --lon D] [--alt M | --agl M] [--heading DEG] [--time HOURS] [--time-rate X] [--day 0..364] [--clouds 0..1] [--storms 0..1] [--alto 0..1] [--wind KT,FROM_DEG] [--haze 0.2..40] [--snow 0..1] [--cloud-base M] [--weather] [--pause-menu flight|places|weather|game|route|settings|graphics|controls[@SECONDS]] [--map-open SECONDS [--map-pick KSFO,KOAK]] [--airport CODE [--runway 28L] [--approach]] [--route KSFO,KOAK,...] [--sim-speed 1|2|4|8] [--max-zoom 8..19] [--cockpit] [--fov DEG] [--no-vsync] [--no-taa] [--render-scale 0.5..1] [--graphics low|medium|high|ultra[,scale=0.5..1,clouds=0..2,shadows=0..3,ao=0|1,bloom=0|1,terrain=0..3,water=low|high,msaa=0|1,dyn=0|1,target=FPS,upscaler=taau|fsr]] [--gpu-profile] [--fullscreen] [--mute] [--lang en|fr] [--offline] [--imagery URL_TEMPLATE] [--google | --no-google] [--photoreal | --no-photoreal] [--photo-detail low|medium|high|ultra] [--no-pave] [--google-key KEY] [--hold SECONDS] [--boost-at SECONDS] [--damage rwing=4.3,laileron=0,rudder=0.5,prop=0.5,engine=0.4,oil=0.6,crack=1,nosegear=0,lsnap=3.4,rfold=3.2,tail=2.0]");
                std::process::exit(0);
            }
            other => {
                eprintln!("unknown argument {other}");
                used = 1;
            }
        }
        i += used;
    }
    a.shots.sort_by(|x, y| x.0.total_cmp(&y.0));
    a
}

fn url_encode(s: &str) -> String {
    s.bytes().map(|b| if b.is_ascii_alphanumeric() || b"-_.~".contains(&b) { (b as char).to_string() } else { format!("%{b:02X}") }).collect()
}

/// Look a place name up with OpenStreetMap's Nominatim. Returns lat, lon (degrees) and a label.
fn geocode(f: &fetch::Fetcher, q: &str) -> Option<(f64, f64, String)> {
    let url = format!("https://nominatim.openstreetmap.org/search?format=json&limit=1&accept-language={}&q={}", i18n::lang().code(), url_encode(q));
    let j: serde_json::Value = serde_json::from_slice(&f.get(&url).ok()?).ok()?;
    let r = j.get(0)?;
    Some((r["lat"].as_str()?.parse().ok()?, r["lon"].as_str()?.parse().ok()?, r["display_name"].as_str().unwrap_or(q).to_string()))
}

fn ground_elev(src: &sources::Sources, lat: f64, lon: f64) -> f64 {
    let id = geo::TileId::at(lat.to_radians(), lon.to_radians(), 11);
    let (u, v) = id.frac(lat.to_radians(), lon.to_radians());
    src.elev(id).map(|e| e.sample(u, v) as f64).unwrap_or(0.0).max(0.0)
}

struct Game {
    window: Arc<Window>,
    r: Renderer,
    world: world::World,
    src: Arc<sources::Sources>,
    /// the Google API key as typed in the settings, and whether it is shown
    gkey: String,
    gkey_show: bool,
    /// Google Maps logo, once it has been shown
    gm_logo: Option<egui::TextureHandle>,
    /// the start screen's aircraft pictures (by index into `aircraft::ALL`), once shown
    stills: Vec<Option<egui::TextureHandle>>,
    /// the aircraft being flown, and its model
    spec: &'static aircraft::Spec,
    model: airframe::Model,
    /// renderer mesh of each model piece, of each breakable group, of the two shard shapes
    piece_mesh: Vec<usize>,
    group_mesh: Vec<usize>,
    /// does the piece have openings cut in it
    piece_cut: Vec<bool>,
    /// each breakable group's centre and size, for the wreckage
    group_pivots: Vec<(Vec3, f32)>,
    /// the models of the other aircraft flown this session, already on the GPU
    hangar: Vec<Loaded>,
    /// aircraft models being built on a background thread (menu selection)
    building: Vec<(&'static aircraft::Spec, std::sync::mpsc::Receiver<airframe::Model>)>,
    plane: Plane,
    crashed: bool,
    crash: Option<Crash>,
    crash_real: f64,
    egui_ctx: egui::Context,
    egui_state: egui_winit::State,
    keys: HashSet<KeyCode>,
    drag: bool,
    last_cursor: Option<(f64, f64)>,
    cam_yaw: f64,
    cam_pitch: f64,
    cam_dist: f64,
    cam_mode: u8,
    cam_pos: DVec3,
    cam_dir: DVec3,
    cam_look: DVec3,
    orbit: f64,
    /// how far the crash camera rises to see past buildings in the way (a
    /// fraction of its distance), and when that was last checked
    cam_rise: f64,
    cam_clear_t: f64,
    /// the crash camera's bearing round what it watches
    cam_ang: Option<f64>,
    cam_hold: f64,
    utc: f64,
    doy: f64,
    /// clouds, haze, wind, clock speed (the F3 panel)
    wx: weather::Weather,
    time: f64,
    real: f64,
    last: Instant,
    acc: f64,
    wave_off: DVec2,
    /// the shore distance field's squares (shore.rs)
    shore: shore::Shore,
    /// the camera in the ocean's coordinates (ocean.rs, wraps at its WRAP)
    ocean_off: DVec2,
    prev_cam: DVec3,
    /// falling snow (composite.wgsl `snowfall`): metres moved through the
    /// flakes (wrapped, see SNOW_WRAP), and the direction they stream from
    snow_s: f64,
    snow_axis: DVec3,
    prev_vp: [[f32; 4]; 4],
    /// temporal anti-aliasing (F2)
    taa: bool,
    /// where the aircraft was drawn last frame, relative to that frame's camera
    prev_plane: Mat4,
    /// centre and half size of the aircraft in its body frame
    plane_box: [[f32; 4]; 2],
    frame: u64,
    fps: f64,
    /// --fps-log: on, start of this second, frames in it, its slowest frame
    fps_log: (bool, Instant, u32, f64),
    paused: bool,
    show_help: bool,
    show_hud: bool,
    goto_open: bool,
    goto_text: String,
    goto_rx: Option<std::sync::mpsc::Receiver<Option<(f64, f64, f64, String)>>>,
    /// the route planner (pause menu, Route page) and the route being flown
    route: route::Route,
    /// route stops being looked up online: (stop, what was typed, the place found)
    route_rx: Vec<std::sync::mpsc::Receiver<(usize, String, Option<route::Wp>)>>,
    /// time acceleration: 1, 2, 4 or 8 times (not kept between runs)
    sim_speed: u32,
    status: String,
    status_until: f64,
    shots: Vec<(f64, PathBuf, Option<(f64, f64, f64)>)>,
    crash_at: Option<f64>,
    boost_at: Option<f64>,
    hold: f64,
    gilrs: Option<gilrs::Gilrs>,
    pad: Controls,
    /// the gamepad last used (the one whose sticks fly the aircraft)
    pad_id: Option<gilrs::GamepadId>,
    /// pads already given the Xbox layout (so a pad it does not take on is not reopened forever)
    pad_fixed: Vec<[u8; 16]>,
    /// set with Y + B on a gamepad: holds the wheel brakes on until released
    parking_brake: bool,
    /// gamepad input for the menus: key events for the next UI pass, a focus
    /// move out of a text box, a d-pad direction held (and when it repeats),
    /// and a request to pick out the screen's first choice
    pad_events: Vec<egui::Event>,
    pad_move: Option<egui::FocusDirection>,
    /// moving on past text boxes: direction, passes since the last move, moves made
    pad_skip: Option<(egui::FocusDirection, u8, u8)>,
    pad_rep: Option<(egui::Key, f64)>,
    pad_home: bool,
    /// the camera look the view starts out at (right stick click goes back to it)
    cam_home: (f64, f64),
    rng: crash::Rng,
    audio: Option<audio::Audio>,
    record: Option<(PathBuf, f64, f64)>,
    rec_audio: Vec<[f32; 2]>,
    rec_n: u32,
    rec_t0: Instant,
    /// sound on (U), and its volume (0..1)
    sound_on: bool,
    volume: f32,
    /// the flight panel (Tab), beside the running game, and the page it shows
    info_open: bool,
    info_page: menu::Page,
    /// the pause menu (Esc): open, the page it shows, and a scripted opening
    pause_open: bool,
    pause_page: menu::Page,
    pause_at: Option<(menu::Page, f64)>,
    /// the minimap and the large map (M), and a scripted opening of the latter
    map: minimap::Map,
    map_at: Option<f64>,
    map_pick: Vec<String>,
    /// mouse-look speed (1 = default)
    look_sens: f64,
    /// the pilot's head in the cockpit view, the cockpit's vertical field of
    /// view (deg), and whether the head was in use last frame
    head: cockpit::Head,
    cockpit_fov: f32,
    /// the instruments' own state: lags, gyros, the compass card
    gauges: gauges::Gauges,
    head_live: bool,
    /// the cockpit warnings sounding, and when each latched one last tripped
    /// (pull up, too low, sink rate, gear) so they do not flicker
    warn: audio::Warn,
    warn_t: [f64; 4],
    /// for the sound: slow-following load factor (jolts stand out from it),
    /// when the last jolt was, last frame's ground contact, vertical speed,
    /// gear and flaps, and metal scraping along the ground this frame
    g_slow: f64,
    last_bump: f64,
    was_ground: bool,
    prev_vs: f64,
    prev_gear: f64,
    prev_flaps: f64,
    quit: bool,
    mode: Mode,
    menu: menu::Menu,
    menu_go: Option<f64>,
    heading: f64,
    destr: destruct::Destruction,
    /// parts that have come off the aircraft and are falling
    sheds: damage::Sheds,
    /// origin of the particle system's local frame, and how far it moved this frame
    fx_anchor: DVec3,
    fx_shift: DVec3,
    smoke_acc: f32,
    /// things that just broke, to flash on the HUD: text, when
    flash: Vec<(String, f64)>,
    damage_spec: Option<String>,
    /// group wreck meshes have been cut down and need restoring on respawn
    wreck_dirty: bool,
    /// how long the aircraft has lain wrecked (unburnt), and when that was last looked at
    wreck_t: f64,
    wreck_seen: f64,
    start_rx: Option<std::sync::mpsc::Receiver<(f64, f64, f64)>>,
    name_rx: Option<std::sync::mpsc::Receiver<String>>,
    last_click: f64,
    n_cars: usize,
    /// how much city is around (0..1), for the orange skyglow at night
    city_glow: f32,
    /// an airport start being prepared: (airport, the airfield if it could be put together, runway, on approach, keep the local time)
    airport_rx: Option<std::sync::mpsc::Receiver<(u32, Option<Arc<airports::Geo>>, Option<String>, bool, bool)>>,
    /// an airport start asked for on the command line
    pending_airport: Option<(u32, Option<String>, bool)>,
    /// the settings as last saved, the language picked in the game (None: the
    /// system's), and when they were last compared with the live ones
    prefs: settings::Settings,
    /// The settings file as this run found it, and the settings as the run
    /// began (one-run flags applied): `keep_settings` saves only what changed since.
    prefs_load: settings::Settings,
    prefs_start: Option<settings::Settings>,
    /// The Graphics options saved (or the GPU's default), and as the run began.
    gfx_base: settings::Graphics,
    gfx0: settings::Graphics,
    prefs_lang: Option<i18n::Lang>,
    prefs_t: f64,
    /// real air traffic around the player, and the last frame's camera
    /// (view-projection, position) to put labels on it
    live: live::Live,
    live_view: Option<(Mat4, DVec3)>,
    /// real weather (sets `wx` while `wx.live` is on)
    livewx: livewx::LiveWx,
    /// waiting on the ground for the terrain to load: where, and until when at most
    hold_spot: Option<(f64, f64, f64)>,
    /// the Graphics settings in effect, the ones last handed to the renderer,
    /// and the ones to save (None: the defaults, or only a command-line
    /// override so far, which is not kept)
    gfx: settings::Graphics,
    gfx_applied: Option<settings::Graphics>,
    gfx_keep: Option<settings::Graphics>,
    /// dynamic resolution: smoothed frame time (ms) and when the scale last changed
    dyn_ms: f64,
    dyn_t: Instant,
    /// the last dynamic-resolution step was up
    dyn_up: bool,
    /// a scale dynamic resolution stepped up to and straight back from, and when
    dyn_ban: (f32, Instant),
    /// when this frame began (the frame rate limit counts from it), and the
    /// smoothed frame time for the fps counter (ms)
    frame_t0: Instant,
    frame_ms: f64,
    /// each sun shadow cascade's matrix as last drawn, and the camera then
    /// (amortised shadows sample the ones not drawn this frame with them);
    /// false when the cascades hold nothing usable
    casc_keep: [([[f32; 4]; 4], DVec3); 3],
    casc_ok: bool,
}

impl Game {
    fn new(window: Arc<Window>, args: &Args) -> anyhow::Result<Self> {
        let prefs = settings::Settings::load();
        let r = pollster::block_on(Renderer::new(window.clone(), args.vsync))?;
        let cache = dirs::cache_dir().unwrap_or_else(|| PathBuf::from(".")).join("fablegame");
        let mut r = r;
        r.headless = args.headless;
        // Graphics: the saved settings (the GPU's default preset before any),
        // then this run's overrides from the command line.
        let gfx_base = prefs.graphics.unwrap_or_else(|| settings::Graphics::default_for(r.integrated));
        let mut gfx = gfx_base;
        if let Some(p) = args.graphics {
            gfx.apply(p, r.integrated);
        }
        for (k, x) in &args.gfx_set {
            let (n, on) = (x.clamp(0.0, 3.0) as u8, *x > 0.5);
            match k.as_str() {
                "scale" => gfx.scale = x.clamp(settings::MIN_SCALE, 1.0),
                "clouds" => gfx.clouds = n.min(2),
                "shadows" => gfx.shadows = n,
                "ao" => gfx.ao = on,
                "bloom" => gfx.bloom = on,
                "terrain" => gfx.terrain = n,
                "water" => gfx.water = n.min(1),
                "msaa" => gfx.msaa = on,
                "dyn" => gfx.dyn_res = on,
                "target" => gfx.target_fps = x.clamp(20.0, 240.0) as u32,
                "upscaler" => gfx.upscaler = settings::Upscaler::ALL[n.min(1) as usize],
                _ => log::warn!("--graphics: no option {k}"),
            }
        }
        if let Some(s) = args.render_scale {
            gfx.scale = s.clamp(settings::MIN_SCALE, 1.0);
        }
        gfx.vsync &= args.vsync;
        gfx.fullscreen |= args.fullscreen;
        // Cached scans load offline too; missing ones keep the procedural layer.
        r.load_ground_photos(fetch::Fetcher::new(cache.clone(), args.offline));
        let google = Arc::new(gmaps::Google::load(args.google_key.clone()));
        if let Some(on) = args.google {
            google.set_enabled(on);
        }
        if let Some(on) = args.photoreal {
            google.set_photoreal(on);
        }
        if let Some(d) = args.photo_detail {
            google.set_detail(d);
        }
        let src = Arc::new(sources::Sources::with_google(fetch::Fetcher::new(cache, args.offline), args.imagery.clone(), google));
        let (mut lat, mut lon, mut status) = (args.lat, args.lon, String::new());
        if let Some(p) = &args.place {
            match geocode(&src.fetch, p) {
                Some((la, lo, name)) => {
                    (lat, lon) = (la, lo);
                    status = name;
                }
                None => status = t!("Could not find \"{}\"", "Introuvable : « {} »", p),
            }
        }
        let spec = args.aircraft.as_deref().and_then(aircraft::by_key).unwrap_or(aircraft::ALL[0]);
        // 450 m up, or two spans for one too big for that (the 747 ×20).
        let alt = args.alt.unwrap_or_else(|| ground_elev(&src, lat, lon) + args.agl.unwrap_or(450f64.max(2.0 * spec.b)));
        let mut plane = Plane::spawn(spec, lat.to_radians(), lon.to_radians(), alt, args.heading.to_radians(), spec.cruise);
        if let Some(spec) = &args.damage {
            plane.dmg.apply_spec(spec);
        }
        let mut world = world::World::new(sources::Loader::new(src.clone(), 14), args.max_zoom);
        let mut r = r;
        let Loaded { spec: _, model, piece_mesh, group_mesh, piece_cut, group_pivots, plane_box } = Loaded::build(spec, &mut r);
        let tm = trees::build_meshes();
        r.set_tree_meshes(&tm.meshes);
        world.tree_dims = tm.dims;
        let mut solids = cars::build_models();
        solids.push(cars::cube());
        solids.extend((0..3).map(|k| cars::rock(k * 17 + 3)));
        solids.extend(destruct::slab_meshes());
        solids.extend(destruct::rebar_meshes());
        debug_assert_eq!(solids.len() as u32, airmodels::BASE);
        solids.extend(airmodels::build());
        r.set_inst_meshes(&solids);
        let mut menu = menu::Menu::new(lat, lon, args.hour);
        menu.aircraft = aircraft::ALL.iter().position(|s| std::ptr::eq(*s, spec)).unwrap_or(0);
        if let Some((la, lo, al)) = args.menu_view {
            menu.look_at(la, lo, al);
            menu.alt = al;
        }
        let mut wx = weather::Weather::new(args.clouds);
        if let Some(s) = args.storms {
            wx.storms = s.clamp(0.0, 1.0);
        }
        if let Some(s) = args.alto {
            wx.alto = s.clamp(0.0, 1.0);
        }
        if let Some(r) = args.time_rate {
            wx.time_rate = r.clamp(0.0, 3600.0);
        }
        if let Some((kt, from)) = args.wind {
            (wx.wind, wx.wind_from) = (kt / 1.94384, from);
        }
        if let Some(h) = args.haze {
            wx.haze = h.clamp(0.2, 40.0);
        }
        if let Some(s) = args.snow {
            wx.snow = s.clamp(0.0, 1.0);
        }
        if let Some(b) = args.cloud_base {
            wx.cloud_base = b.clamp(300.0, 4000.0);
        }
        wx.open = args.weather_panel;
        // Real weather, unless the command line sets the weather itself.
        let scripted = std::env::args().any(|a| matches!(a.as_str(), "--clouds" | "--storms" | "--alto" | "--wind" | "--haze" | "--snow" | "--cloud-base"));
        wx.live = prefs.live_weather && !scripted;
        let set_time = std::env::args().any(|a| matches!(a.as_str(), "--time" | "--day" | "--time-rate"));
        wx.real_time = prefs.real_time && !set_time;
        // The spawn speed is through the air: start out moving with the wind,
        // or a headwind starts it off fast (and climbing it away).
        plane.wind = wx.steady(plane.pos, 300.0);
        plane.vel += plane.wind;
        let egui_ctx = egui::Context::default();
        load_fonts(&egui_ctx);
        menu::apply_theme(&egui_ctx);
        let egui_state = egui_winit::State::new(egui_ctx.clone(), egui::ViewportId::ROOT, &window, Some(window.scale_factor() as f32), None, Some(4096));
        let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0);
        let doy = (secs / 86400.0) % 365.25;
        let cam_pos = plane.pos - plane.forward() * spec.cam_dist;
        // What the anti-aliasing takes to be flying along with the camera:
        // everything within the aircraft's bounds, and a little around them
        // for the gear, flaps and control surfaces as they move.
        let (cy, cp, cd) = args.cam.unwrap_or((0.0, 0.2, spec.cam_dist));
        let mut g = Self {
            gkey: src.google.key(), gkey_show: false, gm_logo: None, stills: vec![None; aircraft::ALL.len()], map: minimap::Map::new(src.clone(), prefs.minimap), window, r, world, src, spec, model, piece_mesh, group_mesh, piece_cut, group_pivots, hangar: Vec::new(), building: Vec::new(), crashed: false, crash: None, crash_real: 0.0, egui_ctx, egui_state, keys: HashSet::new(), drag: false, last_cursor: None,
            cam_yaw: cy.to_radians(), cam_pitch: cp, cam_dist: cd, cam_mode: if args.cockpit { 1 } else { prefs.cam_mode }, cam_pos, cam_dir: -plane.forward(), cam_look: plane.pos, orbit: 0.0, cam_rise: 0.0, cam_clear_t: 0.0, cam_ang: None, cam_hold: 0.0,
            utc: (args.hour - lon / 15.0).rem_euclid(24.0), doy: args.day.unwrap_or(doy), wx, time: 0.0, real: 0.0, last: Instant::now(), acc: 0.0, wave_off: DVec2::ZERO, shore: shore::Shore::default(), ocean_off: DVec2::ZERO, prev_cam: cam_pos, snow_s: 0.0, snow_axis: DVec3::Z, prev_vp: [[0.0; 4]; 4], taa: args.taa && prefs.taa, prev_plane: Mat4::IDENTITY, plane_box,
            frame: 0, fps: 60.0, fps_log: (args.fps_log, Instant::now(), 0, 0.0), paused: false, show_help: false, show_hud: !args.no_hud && prefs.show_hud, goto_open: false, goto_text: String::new(), goto_rx: None, route: args.route.as_deref().map(route::Route::from_list).unwrap_or_else(route::Route::empty), route_rx: Vec::new(), sim_speed: args.sim_speed,
            status, status_until: 8.0, shots: args.shots.clone(), crash_at: args.crash_at, boost_at: args.boost_at, hold: args.hold, gilrs: open_gamepads(), pad: Controls::default(), pad_id: None, pad_fixed: Vec::new(), pad_events: Vec::new(), pad_move: None, pad_skip: None, pad_rep: None, pad_home: false, parking_brake: false, cam_home: (cy.to_radians(), cp), rng: crash::Rng(secs.to_bits() | 1),
            audio: if args.record.is_some() { Some(audio::Audio::offline()) } else if args.mute { None } else { audio::Audio::new() },
            record: args.record.clone(), rec_audio: Vec::new(), rec_n: 0, rec_t0: Instant::now(), sound_on: prefs.sound_on, volume: prefs.volume, info_open: false, info_page: menu::Page::Flight, pause_open: false, pause_page: menu::Page::Game, pause_at: args.pause_menu, map_at: args.map_open, map_pick: args.map_pick.clone(), look_sens: prefs.look_sens, head: cockpit::Head::new(cy.to_radians(), cp), cockpit_fov: args.fov.unwrap_or(prefs.cockpit_fov).clamp(30.0, 90.0), head_live: false, warn: audio::Warn::default(), warn_t: [-10.0; 4],
            g_slow: 1.0, last_bump: 0.0, was_ground: false, prev_vs: 0.0, prev_gear: plane.gear, prev_flaps: plane.flaps, quit: false,
            mode: if args.menu.unwrap_or(true) { Mode::Menu } else { Mode::Fly }, menu, menu_go: args.menu_go, heading: args.heading.to_radians(), destr: destruct::Destruction::new(secs.to_bits()),
            sheds: damage::Sheds::default(), fx_anchor: plane.pos, fx_shift: DVec3::ZERO, smoke_acc: 0.0, flash: Vec::new(), damage_spec: args.damage.clone(), wreck_dirty: false, wreck_t: 0.0, wreck_seen: 0.0,
            start_rx: None, name_rx: None, last_click: -10.0, n_cars: 0, city_glow: 0.0, gauges: gauges::Gauges::new(&plane, secs.to_bits()), plane,
            airport_rx: None, hold_spot: None,
            gfx, gfx0: gfx, gfx_base, prefs_load: prefs.clone(), prefs_start: None, gfx_applied: None, gfx_keep: prefs.graphics, dyn_ms: 0.0, dyn_t: Instant::now(), dyn_up: false, dyn_ban: (0.0, Instant::now()), frame_t0: Instant::now(), frame_ms: 16.0, casc_keep: [([[0.0; 4]; 4], DVec3::ZERO); 3], casc_ok: false,
            prefs_lang: args.lang.or(prefs.lang),
            prefs_t: 0.0,
            live: live::Live::new(prefs.live_traffic),
            live_view: None,
            livewx: livewx::LiveWx::new(),
            prefs: prefs.clone(),
            // --route alone starts on the runway at its departure.
            pending_airport: args.airport.as_deref().or(args.route.as_deref().filter(|_| args.place.is_none() && !std::env::args().any(|x| x == "--lat")).and_then(|r| r.split(',').next())).and_then(|c| {
                let i = airports::Db::get().find(c);
                if i.is_none() {
                    eprintln!("unknown airport {c}");
                }
                i.map(|i| (i, args.runway.clone(), args.approach))
            }),
        };
        // With --menu, an --airport is chosen on the globe rather than flown from.
        if args.menu == Some(true) {
            if let Some((i, rw, appr)) = g.pending_airport.take() {
                g.choose_airport(i);
                g.menu.runway = rw;
                g.menu.start = if appr { menu::Start::Approach } else { menu::Start::Runway };
            }
        }
        Ok(g)
    }

    /// Fly another aircraft from the next spawn on: its model is built (once
    /// a session) and swapped in.
    fn set_aircraft(&mut self, spec: &'static aircraft::Spec) {
        if std::ptr::eq(spec, self.spec) {
            return;
        }
        self.restore_wreck_meshes();
        self.sheds.items.clear();
        let next = match self.hangar.iter().position(|l| std::ptr::eq(l.spec, spec)) {
            Some(i) => self.hangar.swap_remove(i),
            None => match self.building.iter().position(|(s, _)| std::ptr::eq(*s, spec)) {
                // already building in the background: wait for that rather than start over
                Some(i) => {
                    let (_, rx) = self.building.swap_remove(i);
                    match rx.recv() {
                        Ok(m) => Loaded::upload(spec, m, &mut self.r),
                        Err(_) => Loaded::build(spec, &mut self.r),
                    }
                }
                None => Loaded::build(spec, &mut self.r),
            },
        };
        let old = Loaded {
            spec: self.spec,
            model: std::mem::replace(&mut self.model, next.model),
            piece_mesh: std::mem::replace(&mut self.piece_mesh, next.piece_mesh),
            group_mesh: std::mem::replace(&mut self.group_mesh, next.group_mesh),
            piece_cut: std::mem::replace(&mut self.piece_cut, next.piece_cut),
            group_pivots: std::mem::replace(&mut self.group_pivots, next.group_pivots),
            plane_box: std::mem::replace(&mut self.plane_box, next.plane_box),
        };
        self.hangar.push(old);
        self.spec = spec;
        self.cam_dist = spec.cam_dist;
        self.menu.aircraft = aircraft::ALL.iter().position(|s| std::ptr::eq(*s, spec)).unwrap_or(0);
    }

    /// Menu selection: swap to `spec` without stalling the frame. A model not
    /// built yet is generated on a worker thread and swapped in once it's done
    /// (only the GPU upload stays on the main thread).
    fn want_aircraft(&mut self, spec: &'static aircraft::Spec) {
        if self.hangar.iter().any(|l| std::ptr::eq(l.spec, spec)) {
            return self.set_aircraft(spec);
        }
        match self.building.iter().position(|(s, _)| std::ptr::eq(*s, spec)) {
            Some(i) => match self.building[i].1.try_recv() {
                Ok(m) => {
                    self.building.swap_remove(i);
                    let l = Loaded::upload(spec, m, &mut self.r);
                    self.hangar.push(l);
                    self.set_aircraft(spec);
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => {}
                Err(_) => {
                    self.building.swap_remove(i);
                    self.set_aircraft(spec);
                }
            },
            None => {
                let (tx, rx) = std::sync::mpsc::channel();
                std::thread::spawn(move || {
                    let _ = tx.send(Loaded::make(spec));
                });
                self.building.push((spec, rx));
            }
        }
    }

    /// Readings for the cockpit instruments and the state of the lights (see plane.wgsl).
    fn instruments(&self, agl: f64) -> [[f32; 4]; 6] {
        let p = &self.plane;
        let (_, lon, _) = geo::pos_to_lla(p.pos);
        let local = (self.utc + lon.to_degrees() / 15.0).rem_euclid(24.0) * 3600.0;
        let t = self.time;
        let strobe = { let s = t.rem_euclid(1.3); if s < 0.045 || (s > 0.16 && s < 0.2) { 1.0 } else { 0.0 } };
        let beacon = (1.0 - ((t * 1.1).rem_euclid(1.0) - 0.15).abs() * 7.0).max(0.0);
        let (g, card) = self.gauges.readings(p);
        if self.spec.is_jet() {
            // The glass cockpit's engine and air data (b747/deck747.wgsl).
            let (_, _, h) = geo::pos_to_lla(p.pos);
            let mach = p.mach();
            let oat = 15.0 - 6.5 * h.clamp(0.0, 11000.0) / 1000.0;
            let tat = oat + (oat + 273.15) * 0.2 * mach * mach;
            let x = ((p.n1 - 0.2) / 0.82).clamp(0.0, 1.2);
            let running = (p.n1 / 0.15).clamp(0.0, 1.0);
            // (The Hornet's two F404s: hotter and thirstier for their size, 4.9 t of fuel inside.)
            let hornet = self.spec.id == 2;
            let egt = if hornet { oat + (400.0 - oat + 380.0 * x.powf(1.4)) * running } else { oat + (440.0 - oat + 200.0 * x.powf(1.4)) * running };
            let ff = if hornet { (280.0 + 4300.0 * x * x) * running } else { (550.0 + 8200.0 * x * x) * running };
            let (full, n_eng) = if hornet { (4_900.0, 2.0) } else { (110_000.0, 4.0) };
            let fuel = (full - self.time * ff * n_eng / 3600.0).max(0.0);
            let u = p.pos.normalize();
            let gs = (p.vel - u * p.vel.dot(u)).length() * 1.94384;
            return [
                g[0],
                g[1],
                [p.n1 as f32, egt as f32, ff as f32, fuel as f32],
                [mach as f32, gs as f32, (agl * 3.28084) as f32, tat as f32],
                [p.throttle as f32, local as f32, p.gear as f32, p.flap_deg() as f32],
                [1000.0 + card.rem_euclid(360.0), strobe, beacon as f32, if p.gear_down { 1.0 } else { 0.0 }],
            ];
        }
        // Row 5's first slot carries the nav lights (on above 1000) and the compass card.
        [
            g[0],
            g[1],
            g[2],
            g[3],
            [g[4][0], local as f32, p.gear as f32, p.flap_deg() as f32],
            [1000.0 + card.rem_euclid(360.0), strobe, beacon as f32, if p.gear_down { 1.0 } else { 0.0 }],
        ]
    }

    fn sun_dir(&self) -> DVec3 {
        let decl = (-23.44f64).to_radians() * (std::f64::consts::TAU * (self.doy + 10.0) / 365.25).cos();
        let lon = (-15.0 * (self.utc - 12.0)).to_radians();
        DVec3::new(decl.cos() * lon.cos(), decl.cos() * lon.sin(), decl.sin())
    }

    /// Direction to the moon: the sun's ecliptic longitude advanced by the
    /// moon's elongation, carried into the same Earth-fixed frame as the sun.
    fn moon_dir(&self) -> (DVec3, f64) {
        use std::f64::consts::{FRAC_PI_2, TAU};
        let eps = 23.44f64.to_radians();
        let radec = |lam: f64| ((eps.cos() * lam.sin()).atan2(lam.cos()), (eps.sin() * lam.sin()).asin());
        let lam_s = TAU * (self.doy + 10.0) / 365.25 - FRAC_PI_2;
        let elong = TAU * self.wx.moon_age / 29.530588;
        let (ra_s, _) = radec(lam_s);
        let (ra_m, dec_m) = radec(lam_s + elong);
        let lon = (-15.0 * (self.utc - 12.0)).to_radians() + ra_m - ra_s;
        let lit = 0.5 * (1.0 - elong.cos());
        (DVec3::new(dec_m.cos() * lon.cos(), dec_m.cos() * lon.sin(), dec_m.sin()), lit)
    }

    /// In the air, at `alt` (about 450 m up), raised for an aircraft too big
    /// for that to two spans above it.
    fn respawn(&mut self, lat: f64, lon: f64, alt: f64, heading: f64) {
        let alt = alt + (2.0 * self.spec.b - 450.0).max(0.0);
        self.respawn_at(lat, lon, alt, heading, self.spec.cruise);
    }

    fn respawn_at(&mut self, lat: f64, lon: f64, alt: f64, heading: f64, speed: f64) {
        self.hold_spot = None;
        self.parking_brake = false;
        self.plane = Plane::spawn(self.spec, lat, lon, alt, heading, speed);
        // The spawn speed is through the air, not over the ground.
        self.plane.wind = self.wx.steady(self.plane.pos, 300.0);
        self.plane.vel += self.plane.wind;
        if let Some(spec) = &self.damage_spec {
            self.plane.dmg.apply_spec(spec);
        }
        self.restore_wreck_meshes();
        self.flash.clear();
        self.crashed = false;
        self.cam_rise = 0.0;
        self.cam_ang = None;
        self.acc = 0.0;
        self.cam_pos = self.plane.pos - self.plane.forward() * self.spec.cam_dist;
        self.cam_dir = -self.plane.forward();
        self.prev_cam = self.cam_pos;
        self.warn_t = [-10.0; 4];
        (self.g_slow, self.was_ground, self.prev_vs, self.prev_gear, self.prev_flaps) = (1.0, false, 0.0, self.plane.gear, self.plane.flaps);
        self.gauges = gauges::Gauges::new(&self.plane, self.frame ^ 0x9e37);
    }

    /// On the ground, at rest, waiting for the terrain under it to load.
    fn respawn_ground(&mut self, lat: f64, lon: f64, h: f64, heading: f64, pitch: f64) {
        self.respawn_at(lat, lon, h + 10.0, heading, 0.0);
        self.plane = Plane::spawn_ground(self.spec, lat, lon, h, heading, pitch);
        if let Some(spec) = &self.damage_spec {
            self.plane.dmg.apply_spec(spec);
        }
        self.gauges = gauges::Gauges::new(&self.plane, self.frame ^ 0x51ed);
        self.cam_pos = self.plane.pos - self.plane.forward() * self.spec.cam_dist;
        self.cam_dir = -self.plane.forward();
        self.prev_cam = self.cam_pos;
        (self.prev_gear, self.prev_flaps, self.was_ground) = (1.0, 0.0, true);
        self.hold_spot = Some((lat, lon, self.real + 30.0));
    }

    /// Start at an airport: its layout is put together in the background
    /// (see `airport_rx`), then the aircraft appears lined up on the runway
    /// in use, or on a final approach to it.
    fn start_airport(&mut self, idx: u32, runway: Option<String>, approach: bool, keep_local: bool) {
        let a = &airports::Db::get().apts[idx as usize];
        self.status = t!("Preparing {} {}…", "Préparation de {} {}…", a.code(), a.name);
        self.status_until = self.real + 60.0;
        let (tx, rx) = std::sync::mpsc::channel();
        let src = self.src.clone();
        std::thread::spawn(move || {
            let g = airports::geo_for(&src, idx).map_err(|e| log::warn!("airport: {e}")).ok();
            let _ = tx.send((idx, g, runway, approach, keep_local));
        });
        self.airport_rx = Some(rx);
        self.menu.starting = true;
    }

    fn airport_ready(&mut self, idx: u32, g: Option<Arc<airports::Geo>>, runway: Option<String>, approach: bool, keep_local: bool) {
        let a = &airports::Db::get().apts[idx as usize];
        let local = if keep_local { (self.utc + geo::pos_to_lla(self.plane.pos).1.to_degrees() / 15.0).rem_euclid(24.0) } else { self.menu.hour };
        self.utc = (local - a.lon / 15.0).rem_euclid(24.0);
        self.mode = Mode::Fly;
        self.menu.starting = false;
        self.status_until = self.real + 12.0;
        let Some(g) = g.filter(|g| !g.runways.is_empty()) else {
            // Nothing on the map to land on: overhead instead.
            self.respawn(a.lat.to_radians(), a.lon.to_radians(), a.elev as f64 + 450.0, self.heading);
            self.status = t!("{} {} (no runway layout on the map)", "{} {} (pas de plan des pistes sur la carte)", a.code(), a.name);
            return;
        };
        if self.spec.speed_k > 1.0 {
            // One that flies many times a 747's speeds could never get off (or
            // onto) a runway: overhead instead.
            self.respawn(a.lat.to_radians(), a.lon.to_radians(), a.elev as f64 + 450.0, self.heading);
            self.status = t!("{} {} (overhead: no runway is long enough)", "{} {} (à la verticale : aucune piste n'est assez longue)", a.code(), a.name);
            return;
        }
        let want = runway.as_deref().map(airports::norm_ident);
        let pick = want.as_ref().and_then(|w| g.runways.iter().enumerate().find_map(|(ri, r)| (0..2).find(|e| airports::norm_ident(&r.ident[*e]) == *w).map(|e| (ri, e))));
        let (ri, e) = pick.or_else(|| g.active(self.wx.wind_from as f32, (self.wx.wind * 1.94384) as f32)).unwrap_or((0, 0));
        let r = &g.runways[ri];
        let (thr, dir) = r.end(e);
        let heading = (dir.x as f64).atan2(dir.y as f64);
        if approach {
            let d = if r.len > 1500.0 { 7000.0 } else { 4500.0 };
            let p = thr - dir * d;
            let (lat, lon) = g.lla(p);
            let alt = g.h(thr) as f64 + (d + traffic::aim_dist(r.len)) as f64 * 3f64.to_radians().tan();
            // Approach speed (70 kt in the light single), approach flaps, gear
            // down, trimmed down a 3 degree glide.
            let (v, notch) = self.spec.approach;
            self.respawn_at(lat, lon, alt, heading, v);
            self.plane.configure(notch, true, v * 3f64.to_radians().sin());
            self.status = t!("{} {} · final approach runway {} · {:.1} nm", "{} {} · approche finale piste {} · {:.1} NM", a.code(), a.name, r.ident[e], d / 1852.0);
        } else {
            let start = if e == 0 { r.a } else { r.b } + dir * 25.0;
            let (lat, lon) = g.lla(start);
            // On a sloping runway it stands nose up (or down) with the slope.
            let pitch = ((g.h(start + dir * 3.0) - g.h(start - dir * 3.0)) / 6.0).atan() as f64;
            self.respawn_ground(lat, lon, g.h(start) as f64, heading, pitch);
            self.status = t!("{} {} · runway {} · {:.0} m × {:.0} m", "{} {} · piste {} · {:.0} m × {:.0} m", a.code(), a.name, r.ident[e], r.len, r.width);
        }
    }

    /// The nearest airport worth landing at.
    fn nearest_airport(&self) -> Option<u32> {
        let (lat, lon, _) = geo::pos_to_lla(self.plane.pos);
        let db = airports::Db::get();
        db.near(lat.to_degrees(), lon.to_degrees(), 150_000.0).into_iter().find(|(i, _)| { let a = &db.apts[*i as usize]; a.any_paved() && a.longest() > 700.0 }).map(|x| x.0)
    }

    fn controls(&mut self, dt: f64) -> Controls {
        let k = |c: KeyCode| self.keys.contains(&c);
        let ax = |pos: bool, neg: bool| (pos as i32 - neg as i32) as f64;
        let mut c = Controls {
            pitch: ax(k(KeyCode::KeyS) || k(KeyCode::ArrowDown), k(KeyCode::KeyW) || k(KeyCode::ArrowUp)),
            roll: ax(k(KeyCode::KeyD) || k(KeyCode::ArrowRight), k(KeyCode::KeyA) || k(KeyCode::ArrowLeft)),
            yaw: ax(k(KeyCode::KeyE), k(KeyCode::KeyQ)),
            throttle: ax(k(KeyCode::ShiftLeft) || k(KeyCode::ShiftRight), k(KeyCode::ControlLeft) || k(KeyCode::ControlRight)),
            brake: k(KeyCode::KeyB),
            boost: k(KeyCode::Space),
            reverse: k(KeyCode::KeyZ),
        };
        self.gamepad(dt, &mut c);
        c.brake |= self.parking_brake;
        if let Some(t) = self.crash_at {
            if self.real > t && !self.crashed {
                c.pitch = -1.0;
                c.throttle = 1.0;
            }
        }
        if self.boost_at.is_some_and(|t| self.real > t) {
            c.throttle = 1.0;
            c.boost = true;
        }
        c
    }

    /// An Xbox-style gamepad, laid out as Microsoft Flight Simulator (2020)
    /// sets one up by default:
    ///
    /// | input | does |
    /// |---|---|
    /// | left stick | pitch and roll |
    /// | LT / RT | rudder left / right |
    /// | A / B (hold) | throttle up / down |
    /// | X (hold) | wheel brakes |
    /// | Y + B | parking brake on / off |
    /// | Y + d-pad up / down | elevator trim nose down / up |
    /// | LB / RB | flaps up / down a notch |
    /// | left stick click | landing gear |
    /// | right stick | look around |
    /// | right stick click | reset the view |
    /// | d-pad | quick views: ahead, panel, left, right |
    /// | View | cockpit / chase camera |
    /// | Menu | pause menu |
    ///
    /// In the menus A starts the flight and B closes the pause menu; after a
    /// crash A starts again.
    fn gamepad(&mut self, dt: f64, c: &mut Controls) {
        use gilrs::{Axis, Button, EventType};
        let Some(g) = self.gilrs.as_mut() else { return };
        let (mut pressed, mut relayout) = (Vec::new(), false);
        while let Some(ev) = g.next_event() {
            match ev.event {
                EventType::Connected => relayout = true,
                EventType::ButtonPressed(b, _) => {
                    self.pad_id = Some(ev.id);
                    pressed.push(b);
                }
                EventType::AxisChanged(..) | EventType::ButtonChanged(..) => {
                    if self.pad_id.is_none() {
                        self.pad_id = Some(ev.id);
                    }
                }
                EventType::Disconnected if self.pad_id == Some(ev.id) => self.pad_id = None,
                _ => {}
            }
        }
        // A pad plugged in that gilrs has no layout for: open it again with one.
        let unfixed: Vec<[u8; 16]> = g.gamepads().filter(|(_, p)| needs_xbox_layout(p) && !self.pad_fixed.contains(&p.uuid())).map(|(_, p)| p.uuid()).collect();
        if relayout && !unfixed.is_empty() {
            self.pad_fixed.extend(unfixed);
            self.gilrs = open_gamepads();
            self.pad_id = None;
            return;
        }
        let id = self.pad_id.filter(|id| g.connected_gamepad(*id).is_some()).or_else(|| g.gamepads().next().map(|(id, _)| id));
        let Some(pad) = id.and_then(|id| g.connected_gamepad(id)) else {
            self.pad = Controls::default();
            padui::set_active(false);
            self.pad_rep = None;
            return;
        };
        padui::set_active(true);
        // Where the pad's buttons go: the globe, the pause menu, the flight
        // panel (which then has the d-pad, A, B and the bumpers), or the aircraft.
        let nav = if self.mode == Mode::Menu {
            Nav::Globe
        } else if self.pause_open {
            Nav::Pause
        } else if self.info_open {
            Nav::Panel
        } else {
            Nav::Off
        };
        // A small dead zone, the rest of the travel rescaled so there is no step out of it.
        let dz = |v: f32| {
            let v = v as f64;
            let d = 0.12;
            if v.abs() < d { 0.0 } else { v.signum() * (v.abs() - d) / (1.0 - d) }
        };
        // The analogue triggers; on a pad left without a layout they come through as
        // bare Z axes, -1 at rest to +1 squeezed.
        let trig = |b: Button, z: Axis| match (pad.button_data(b), pad.axis_data(z)) {
            (Some(d), _) => d.value() as f64,
            (None, Some(a)) => ((a.value() as f64 + 1.0) * 0.5).clamp(0.0, 1.0),
            _ => 0.0,
        };
        let modifier = pad.is_pressed(Button::North);
        let flying = self.mode == Mode::Fly && !self.pause_open;
        self.pad = Controls {
            pitch: -dz(pad.value(Axis::LeftStickY)),
            roll: dz(pad.value(Axis::LeftStickX)),
            yaw: (trig(Button::RightTrigger2, Axis::RightZ) - trig(Button::LeftTrigger2, Axis::LeftZ)).clamp(-1.0, 1.0),
            // A and B work the flight panel while it is open.
            throttle: if nav == Nav::Panel { 0.0 } else { ax_pad(pad.is_pressed(Button::South), pad.is_pressed(Button::East) && !modifier) },
            brake: pad.is_pressed(Button::West),
            boost: false,
            reverse: false,
        };
        let (look_x, look_y) = (dz(pad.value(Axis::RightStickX)), dz(pad.value(Axis::RightStickY)));
        let trim = if modifier && nav != Nav::Panel { ax_pad(pad.is_pressed(Button::DPadDown), pad.is_pressed(Button::DPadUp)) } else { 0.0 };

        // Moving about the menus: the d-pad (and in the pause menu the left
        // stick too), repeating while held.
        let dirs = [(Button::DPadUp, egui::Key::ArrowUp), (Button::DPadDown, egui::Key::ArrowDown), (Button::DPadLeft, egui::Key::ArrowLeft), (Button::DPadRight, egui::Key::ArrowRight)];
        let mut held = dirs.iter().find(|(b, _)| pad.is_pressed(*b)).map(|d| d.1);
        let (lx, ly) = (pad.value(Axis::LeftStickX), pad.value(Axis::LeftStickY));
        if held.is_none() && nav == Nav::Pause && lx.abs().max(ly.abs()) > 0.6 {
            held = Some(if ly.abs() > lx.abs() { if ly > 0.0 { egui::Key::ArrowUp } else { egui::Key::ArrowDown } } else if lx > 0.0 { egui::Key::ArrowRight } else { egui::Key::ArrowLeft });
        }
        let tap = pressed.iter().find_map(|b| dirs.iter().find(|d| d.0 == *b).map(|d| d.1));
        let zoom = trig(Button::RightTrigger2, Axis::RightZ) - trig(Button::LeftTrigger2, Axis::LeftZ);
        let mut nav_keys = Vec::new();
        if nav == Nav::Off {
            self.pad_rep = None;
        } else {
            match (held, self.pad_rep) {
                (Some(k), Some((pk, t))) if pk == k => {
                    if self.real >= t {
                        nav_keys.push(k);
                        self.pad_rep = Some((k, (t + 0.085).max(self.real)));
                    }
                }
                (Some(k), _) => {
                    nav_keys.push(k);
                    self.pad_rep = Some((k, self.real + 0.4));
                }
                (None, _) => {
                    self.pad_rep = None;
                    nav_keys.extend(tap);
                }
            }
        }
        for k in nav_keys {
            self.pad_key(k);
        }
        if nav == Nav::Globe {
            // Left stick turns the globe (squared, for a fine touch), the triggers zoom.
            let h = self.window.inner_size().height.max(1) as f64;
            let sq = |v: f64| v * v.abs();
            let (sx, sy) = (sq(dz(lx)), sq(dz(ly)));
            if sx != 0.0 || sy != 0.0 {
                let k = 0.8 * h * dt;
                self.menu.drag(-sx * k, sy * k, h, 60f64.to_radians());
            }
            if zoom.abs() > 0.05 {
                self.menu.zoom(zoom * 5.0 * dt);
            }
        }

        c.pitch = (c.pitch + self.pad.pitch).clamp(-1.0, 1.0);
        c.roll = (c.roll + self.pad.roll).clamp(-1.0, 1.0);
        c.yaw = (c.yaw + self.pad.yaw).clamp(-1.0, 1.0);
        c.throttle = (c.throttle + self.pad.throttle).clamp(-1.0, 1.0);
        c.brake |= self.pad.brake;

        if flying {
            // Right stick: look around, as fast as a quick head turn at full deflection.
            // Pushed right the view swings right, in the cockpit and outside.
            let rate = 2.2 * self.look_sens * dt;
            let side = if self.cam_mode == 1 { 1.0 } else { -1.0 };
            self.cam_yaw += look_x * rate * side;
            self.cam_pitch = (self.cam_pitch - look_y * rate * 0.8).clamp(-1.2, 1.4);
            if trim != 0.0 && !self.crashed {
                // Winds the trim (the angle of attack the elevator holds) at about 2 degrees a second.
                self.plane.trim = (self.plane.trim + trim * 0.035 * dt).clamp(-0.05, 0.24);
            }
        }

        for b in pressed {
            self.pad_button(nav, b, modifier);
        }
    }

    /// A gamepad button pressed, for whatever it works at the moment.
    fn pad_button(&mut self, nav: Nav, b: gilrs::Button, modifier: bool) {
        use gilrs::Button;
        match (nav, b) {
            // The globe: Menu takes off, A presses what is picked out, X starts
            // at the crosshair, View shows the controls and B hides them.
            (Nav::Globe, Button::Start) => self.begin(),
            (Nav::Globe, Button::South) => self.pad_key(egui::Key::Enter),
            (Nav::Globe, Button::West) => self.pad_pick_centre(),
            (Nav::Globe, Button::Select) => self.menu.show_controls = !self.menu.show_controls,
            (Nav::Globe, Button::East) => self.menu.show_controls = false,
            (Nav::Globe, _) => {}
            // The pause menu: A presses, the bumpers turn the page, B or Menu resumes.
            (Nav::Pause, Button::Start | Button::East) => self.set_pause_menu(false),
            (Nav::Pause, Button::South) => self.pad_key(egui::Key::Enter),
            (Nav::Pause, Button::LeftTrigger | Button::RightTrigger) => {
                self.pause_page = menu::Page::cycle(&menu::Page::PAUSE, self.pause_page, b == Button::RightTrigger);
                self.pad_home = true;
            }
            (Nav::Pause, _) => {}
            // The flight panel, open beside the flying: the same, and B (or Y + View) closes it.
            (Nav::Panel, Button::South) => self.pad_key(egui::Key::Enter),
            (Nav::Panel, Button::East) => self.info_open = false,
            (Nav::Panel, Button::Select) if modifier => self.info_open = false,
            (Nav::Panel, Button::LeftTrigger | Button::RightTrigger) => {
                self.info_page = menu::Page::cycle(&menu::Page::INFO, self.info_page, b == Button::RightTrigger);
                self.pad_home = true;
            }
            (Nav::Panel, Button::DPadUp | Button::DPadDown | Button::DPadLeft | Button::DPadRight) => {}
            (_, Button::Start) => self.set_pause_menu(true),
            (_, Button::South) if self.crashed => self.respawn_here(),
            (_, Button::Select) if modifier => {
                self.info_open = true;
                self.pad_home = true;
            }
            (_, Button::Select) => self.cam_mode = (self.cam_mode + 1) % 2,
            (_, Button::RightThumb) => {
                (self.cam_yaw, self.cam_pitch) = self.cam_home;
                self.head.zoom = 1.0;
            }
            (_, Button::East) if modifier => {
                self.parking_brake = !self.parking_brake;
                self.status = if self.parking_brake { t!("Parking brake set", "Frein de parc serré") } else { t!("Parking brake released", "Frein de parc desserré") }.into();
                self.status_until = self.real + 2.0;
            }
            (_, Button::DPadUp | Button::DPadDown | Button::DPadLeft | Button::DPadRight) if !modifier => {
                // Quick views. Outside, left and right look at that side of the aircraft.
                let (home_y, home_p) = self.cam_home;
                let side = if self.cam_mode == 1 { 1.0 } else { -1.0 };
                (self.cam_yaw, self.cam_pitch) = match b {
                    Button::DPadUp => (home_y, home_p),
                    Button::DPadDown if self.cam_mode == 1 => (home_y, 0.55),
                    Button::DPadDown => (home_y + std::f64::consts::PI, home_p),
                    Button::DPadLeft => (home_y - side * std::f64::consts::FRAC_PI_2, home_p.min(0.15)),
                    _ => (home_y + side * std::f64::consts::FRAC_PI_2, home_p.min(0.15)),
                };
            }
            (_, Button::LeftTrigger) if !self.crashed => self.plane.set_flaps(-1),
            (_, Button::RightTrigger) if !self.crashed => self.plane.set_flaps(1),
            (_, Button::LeftThumb) if !self.crashed => self.plane.toggle_gear(),
            _ => {}
        }
    }

    /// A key for the menus from the gamepad (d-pad arrows, A as Enter). With
    /// nothing picked out yet it picks out the screen's first choice instead;
    /// in a text box the arrows move on rather than along the text.
    fn pad_key(&mut self, key: egui::Key) {
        if self.egui_ctx.memory(|m| m.focused()).is_none() {
            self.pad_home = true;
            return;
        }
        let dir = match key {
            egui::Key::ArrowUp => Some(egui::FocusDirection::Up),
            egui::Key::ArrowDown => Some(egui::FocusDirection::Down),
            egui::Key::ArrowLeft => Some(egui::FocusDirection::Left),
            egui::Key::ArrowRight => Some(egui::FocusDirection::Right),
            _ => None,
        };
        if let Some(d) = dir {
            self.pad_skip = Some((d, 0, 0));
            if self.egui_ctx.text_edit_focused() {
                self.pad_move = Some(d);
                return;
            }
        }
        for pressed in [true, false] {
            self.pad_events.push(egui::Event::Key { key, physical_key: None, pressed, repeat: false, modifiers: egui::Modifiers::NONE });
        }
    }

    /// The globe: start at the airport or the spot under the crosshair.
    fn pad_pick_centre(&mut self) {
        let size = self.window.inner_size();
        let (w, h) = (size.width.max(1) as f64, size.height.max(1) as f64);
        let near = self.menu.marker_near(w * 0.5, h * 0.5, 28.0 * self.window.scale_factor());
        if let Some(i) = near {
            self.choose_airport(i);
        } else if let Some((lat, lon)) = self.menu.pick_at(w * 0.5, h * 0.5, w, h, 60f64.to_radians()) {
            self.pick(lat, lon);
        }
    }

    fn key(&mut self, code: KeyCode) {
        if self.mode == Mode::Menu {
            match code {
                KeyCode::Escape if self.menu.show_controls => self.menu.show_controls = false,
                KeyCode::Escape => self.quit = true,
                KeyCode::Enter | KeyCode::NumpadEnter => self.begin(),
                KeyCode::F12 => self.shots.insert(0, (0.0, PathBuf::from(format!("screenshot-{}.png", self.frame)), None)),
                _ => {}
            }
            return;
        }
        if self.pause_open {
            // The pause menu has the keyboard (and the game is stopped) until it closes.
            match code {
                KeyCode::Escape => self.set_pause_menu(false),
                KeyCode::F12 => self.shots.insert(0, (0.0, PathBuf::from(format!("screenshot-{}.png", self.frame)), None)),
                _ => {}
            }
            return;
        }
        match code {
            KeyCode::Tab => self.info_open = !self.info_open,
            // Esc closes a pop-up window first; otherwise it pauses.
            KeyCode::Escape if self.show_help => self.show_help = false,
            KeyCode::Escape if self.wx.open => self.wx.open = false,
            KeyCode::Escape if self.map.open => self.map.open = false,
            KeyCode::Escape => self.set_pause_menu(true),
            KeyCode::KeyR => self.respawn_here(),
            KeyCode::KeyC => self.cam_mode = (self.cam_mode + 1) % 2,
            KeyCode::KeyF if !self.crashed => self.plane.set_flaps(1),
            KeyCode::KeyV if !self.crashed => self.plane.set_flaps(-1),
            KeyCode::KeyL if !self.crashed => self.plane.toggle_gear(),
            KeyCode::Slash if !self.crashed => self.plane.toggle_speedbrake(),
            KeyCode::KeyG => {
                self.goto_open = true;
                self.goto_text.clear();
            }
            KeyCode::KeyT | KeyCode::KeyY => {
                // Setting the time by hand leaves real time.
                self.wx.real_time = false;
                self.utc = (self.utc + if code == KeyCode::KeyT { 0.5 } else { -0.5 }).rem_euclid(24.0);
            }
            KeyCode::KeyN => self.wx.clouds = (self.wx.clouds + 0.1).min(1.0),
            KeyCode::KeyX => self.wx.clouds = (self.wx.clouds - 0.1).max(0.0),
            KeyCode::KeyM => self.map.open = !self.map.open,
            KeyCode::F3 => self.wx.open = !self.wx.open,
            KeyCode::KeyP => self.paused = !self.paused,
            KeyCode::BracketRight => self.set_sim_speed(self.sim_speed * 2),
            KeyCode::BracketLeft => self.set_sim_speed(self.sim_speed / 2),
            KeyCode::KeyJ if !self.crashed => {
                if self.route.target().is_some() {
                    self.route.autopilot = !self.route.autopilot;
                    self.status = if self.route.autopilot { t!("Autopilot: heading to {}", "Pilote automatique : cap vers {}", self.route.target().map(|w| w.name.clone()).unwrap_or_default()) } else { t!("Autopilot off", "Pilote automatique coupé").into() };
                } else {
                    self.status = t!("No route to follow: plan one on the pause menu's Route page", "Aucun itinéraire : préparez-en un dans le menu pause, page Itinéraire").into();
                }
                self.status_until = self.real + 3.0;
            }
            KeyCode::KeyH => self.show_hud = !self.show_hud,
            KeyCode::KeyK if !self.crashed => {
                self.gauges.sync_dg(&self.plane);
                self.status = t!("Heading indicator set to the compass", "Conservateur de cap recalé sur le compas").into();
                self.status_until = self.real + 2.0;
            }
            KeyCode::KeyU => {
                self.sound_on = !self.sound_on;
                self.status = if self.sound_on { t!("Sound on", "Son activé") } else { t!("Sound off", "Son coupé") }.into();
                self.status_until = self.real + 2.0;
            }
            KeyCode::F1 => self.show_help = !self.show_help,
            KeyCode::F2 => self.taa = !self.taa,
            KeyCode::F4 => self.gfx.show_fps = !self.gfx.show_fps,
            KeyCode::F11 => self.gfx.fullscreen = !self.gfx.fullscreen,
            KeyCode::F12 => {
                let p = PathBuf::from(format!("screenshot-{}.png", self.frame));
                self.shots.insert(0, (0.0, p, None));
            }
            _ => {}
        }
    }

    /// Open or close the pause menu. The game stops while it is open.
    fn set_pause_menu(&mut self, open: bool) {
        if open == self.pause_open || self.mode != Mode::Fly {
            return;
        }
        self.pause_open = open;
        self.pad_home = open;
        self.drag = false;
        self.show_help = false;
        // A button or slider left focused would keep the keys from the aircraft.
        self.egui_ctx.memory_mut(|m| {
            if let Some(id) = m.focused() {
                m.surrender_focus(id);
            }
        });
    }

    /// Back to the globe, looking at where we were.
    fn to_globe(&mut self) {
        self.set_pause_menu(false);
        self.info_open = false;
        let (lat, lon, _) = geo::pos_to_lla(self.plane.pos);
        self.menu.look_at(lat.to_degrees(), lon.to_degrees(), 60_000.0);
        self.menu.pick = Some((lat.to_degrees(), lon.to_degrees()));
        self.menu.hour = (self.utc + lon.to_degrees() / 15.0).rem_euclid(24.0);
        self.menu.starting = false;
        self.mode = Mode::Menu;
    }

    /// A fresh aircraft in the air over where we are (or where we crashed), on the same heading.
    fn respawn_here(&mut self) {
        let (lat, lon, h) = geo::pos_to_lla(if self.crashed { self.crash.as_ref().map(|c| c.site).unwrap_or(self.plane.pos) } else { self.plane.pos });
        let g = self.world.ground(lat, lon).map(|g| g.h).unwrap_or(h);
        let hd = self.plane.forward().dot(geo::enu(self.plane.pos).0).atan2(self.plane.forward().dot(geo::enu(self.plane.pos).1));
        self.respawn(lat, lon, g + 420.0, hd);
    }

    /// Fly from a known spot: its ground height is looked up in the background,
    /// then the aircraft appears above it (see `goto_rx`).
    fn fly_to(&mut self, lat: f64, lon: f64, name: &str) {
        self.status = t!("Flying to {}…", "En route vers {}…", i18n::tr_owned(name));
        self.status_until = self.real + 30.0;
        let (tx, rx) = std::sync::mpsc::channel();
        let src = self.src.clone();
        let name = name.to_string();
        std::thread::spawn(move || {
            let _ = tx.send(Some((lat, lon, ground_elev(&src, lat, lon), name)));
        });
        self.goto_rx = Some(rx);
    }

    /// Input for the UI, minus Tab: that opens and closes the menu here, not
    /// egui's keyboard focus (which would then keep keys from the aircraft).
    fn egui_input(&mut self) -> egui::RawInput {
        let mut raw = self.egui_state.take_egui_input(&self.window);
        if !self.egui_ctx.text_edit_focused() {
            raw.events.retain(|e| !matches!(e, egui::Event::Key { key: egui::Key::Tab, .. }));
        }
        raw.events.append(&mut self.pad_events);
        // The pad works the game whether or not the window has the keyboard,
        // and egui only lets a focused slider take the arrows in a focused window.
        if padui::active() {
            raw.focused = true;
        }
        raw
    }

    /// At the start of a UI pass: a gamepad move out of a text box (the box
    /// itself would keep the arrow keys).
    /// Text boxes are passed over: a pad cannot type, and a slider's number
    /// box would otherwise keep left and right from the slider. (A widget
    /// only shows as a text box the pass after it takes the focus.)
    fn pad_pass(&mut self, ctx: &egui::Context) {
        if let Some(d) = self.pad_move.take() {
            ctx.memory_mut(|m| m.move_focus(d));
        }
        if let Some((d, stage, hops)) = self.pad_skip {
            self.pad_skip = if stage < 2 {
                Some((d, stage + 1, hops))
            } else if ctx.text_edit_focused() && hops < 6 {
                ctx.memory_mut(|m| m.move_focus(d));
                Some((d, 1, hops + 1))
            } else {
                None
            };
        }
    }

    /// Whether this is the widget to pick out for the gamepad (once asked for).
    fn pad_home_focus(&mut self, resp: &egui::Response) {
        if self.pad_home && padui::active() {
            resp.request_focus();
            self.pad_home = false;
        }
    }

    /// Menu: choose a spot on the globe and look its name up in the background.
    fn pick(&mut self, lat: f64, lon: f64) {
        self.menu.pick = Some((lat, lon));
        self.menu.airport = None;
        self.menu.runway = None;
        self.menu.pick_name = match menu::Menu::near_city(lat, lon) {
            Some((name, d)) if d < 40_000.0 => tr(name).to_string(),
            Some((name, d)) if d < 400_000.0 => t!("{:.0} km from {}", "À {:.0} km de {}", d / 1000.0, tr(name)),
            _ => String::new(),
        };
        let (tx, rx) = std::sync::mpsc::channel();
        let src = self.src.clone();
        std::thread::spawn(move || {
            let url = format!("https://nominatim.openstreetmap.org/reverse?format=json&zoom=12&accept-language={}&lat={lat:.5}&lon={lon:.5}", i18n::lang().code());
            let name = src.fetch.get(&url).ok().and_then(|d| serde_json::from_slice::<serde_json::Value>(&d).ok()).and_then(|j| j["display_name"].as_str().map(String::from));
            if let Some(n) = name {
                let _ = tx.send(n);
            }
        });
        self.name_rx = Some(rx);
    }

    /// Menu: choose an airport to start at.
    fn choose_airport(&mut self, i: u32) {
        let a = &airports::Db::get().apts[i as usize];
        self.menu.look_at(a.lat, a.lon, self.menu.alt.min(25_000.0));
        self.menu.pick = Some((a.lat, a.lon));
        self.menu.pick_name = format!("{} {} ({}, {})", a.code(), a.name, a.city, a.country);
        if self.menu.airport != Some(i) {
            self.menu.runway = None;
        }
        self.menu.airport = Some(i);
        self.name_rx = None;
    }

    /// Menu: take off at the chosen spot (once its ground height is known).
    fn begin(&mut self) {
        let Some((lat, lon)) = self.menu.pick else { return };
        if self.menu.starting {
            return;
        }
        self.set_aircraft(aircraft::ALL[self.menu.aircraft.min(aircraft::ALL.len() - 1)]);
        if let Some(i) = self.menu.airport.filter(|_| self.menu.start != menu::Start::Overhead) {
            self.start_airport(i, self.menu.runway.clone(), self.menu.start == menu::Start::Approach, false);
            return;
        }
        self.menu.starting = true;
        let (tx, rx) = std::sync::mpsc::channel();
        let src = self.src.clone();
        std::thread::spawn(move || {
            let _ = tx.send((lat, lon, ground_elev(&src, lat, lon)));
        });
        self.start_rx = Some(rx);
    }

    /// Particles live in a frame anchored near the action; move it if the action has moved on.
    fn anchor_fx(&mut self, at: DVec3) {
        let stale = self.crash.as_ref().map(|c| c.time > 90.0).unwrap_or(true);
        if stale && (self.fx_anchor - at).length() > 800.0 {
            // Live particles are carried over into the new frame.
            self.fx_shift += self.fx_anchor - at;
            self.crash = None;
            self.fx_anchor = at;
        }
    }

    /// Knock a building down, with any parts standing on it.
    /// `hit`: the aircraft itself went in (the wall shows its outline).
    fn bring_down(&mut self, tile: geo::TileId, idx: usize, at: DVec3, vel: DVec3, hit: Option<destruct::Hitter>, femit: &mut Vec<(DVec3, render::Particle)>) {
        let above = self.world.structure_of(tile, idx);
        let mut parts = Vec::new();
        for (t, i) in std::iter::once((tile, idx)).chain(above) {
            if let Some(w) = self.world.destroy_building(&self.r.queue, t, i) {
                parts.push(w);
            }
        }
        if parts.is_empty() {
            return;
        }
        // Under the 3D tiles the building is the mesh's: it is cut out of the
        // mesh, and its own walls and roof come down on the pieces, which
        // stand as high as it did.
        let rings = self.world.cut_rings(&parts, 2.0);
        let base = parts.iter().map(|w| w.base).fold(f64::MAX, f64::min);
        let photo = self.world.photo_mut().take_building(&rings, base);
        if let Some(src) = &photo {
            for (w, top) in parts.iter_mut().zip(&src.tops) {
                if let Some(t) = *top {
                    w.top = t.max(w.base + 3.0);
                    let top = w.top as f32;
                    w.tiers.retain(|z| z.1 < top - 1.0);
                }
            }
            log::info!("3D tiles: building cut out, {} tiles, {} triangles", src.tiles.len(), src.tiles.iter().map(|t| t.tris.len()).sum::<usize>());
        }
        self.destr.collapse(&parts, at, vel, hit, photo.as_ref(), femit);
    }

    /// The wreck is made of what was left of the aircraft: every group's mesh
    /// is rebuilt without the pieces that had already gone, and cut along its
    /// breaks. Returns the break each group keeps showing.
    fn wreck_meshes(&mut self) -> Vec<[Tear; 2]> {
        let dm = self.plane.dmg;
        let mut out = vec![[Tear::NONE; 2]; airframe::N_GROUPS];
        for g in 0..airframe::N_GROUPS {
            let pieces: Vec<&airframe::Piece> = self.model.pieces.iter().filter(|p| p.group == g && !p.glass).collect();
            let changed = pieces.iter().any(|p| dm.hidden(p) || dm.tears_for(p).iter().any(|t| t.is_some()));
            if !changed {
                continue;
            }
            self.wreck_dirty = true;
            let mut m = airframe::Mesh::default();
            for p in &pieces {
                if dm.hidden(p) {
                    continue;
                }
                let tears = dm.tears_for(p);
                let primary = matches!(p.name, "left wing" | "right wing" | "stabiliser" | "blades");
                if primary {
                    out[g] = tears;
                }
                let base = m.v.len() as u32;
                m.v.extend_from_slice(&p.mesh.v);
                for t in p.mesh.i.chunks(3) {
                    let gone = t.iter().all(|&i| { let q = Vec3::from(p.mesh.v[i as usize].pos); tears.iter().any(|tr| !tr.keeps(q)) });
                    if !gone {
                        m.i.extend(t.iter().map(|i| i + base));
                    }
                }
            }
            if m.i.is_empty() {
                m = self.model.groups[g].clone();
            }
            self.r.replace_mesh(self.group_mesh[g], &m.v, &m.i);
        }
        out
    }
    fn restore_wreck_meshes(&mut self) {
        if self.wreck_dirty {
            for g in 0..airframe::N_GROUPS {
                let m = &self.model.groups[g];
                self.r.replace_mesh(self.group_mesh[g], &m.v, &m.i);
            }
            self.wreck_dirty = false;
        }
    }

    /// The cockpit warnings: stall, terrain ahead, nose dive, too low, sink
    /// rate, overspeed and gear up.
    fn warnings(&mut self) -> audio::Warn {
        let p = &self.plane;
        let mut w = audio::Warn::default();
        if self.crashed || self.mode == Mode::Menu || p.on_ground {
            return w;
        }
        let up = p.pos.normalize();
        let (la, lo, h) = geo::pos_to_lla(p.pos);
        let Some(g) = self.world.ground(la, lo) else { return w };
        let agl = h - g.h;
        if agl < 3.0 {
            return w;
        }
        let vs = p.vel.dot(up);
        let air = p.ias();
        let pitch = p.forward().dot(up).clamp(-1.0, 1.0).asin().to_degrees();
        if air > 8.0 {
            let sa = p.stall_alpha();
            w.stall = smoothstep(sa - 0.07, sa - 0.01, p.alpha.abs()) as f32;
        }
        let over = p.overspeed();
        if over > 0.0 {
            w.overspeed = (over / 13.0).clamp(0.1, 1.0) as f32;
        }
        // Going down nose first, steeply and fast.
        if vs < -10.0 {
            w.dive = smoothstep(-20.0, -55.0, pitch) as f32;
            if pitch < -22.0 || vs < -40.0 {
                w.dive = w.dive.max(0.2);
            }
        }
        // What is along the flight path in the next few seconds: ground, and
        // any building standing on it.
        let (mut tti, mut tower) = (f64::INFINITY, false);
        if p.vel.length() > 15.0 && agl < 1500.0 {
            for t in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0] {
                let (la, lo, h) = geo::pos_to_lla(p.pos + p.vel * t);
                let Some(g) = self.world.ground(la, lo) else { continue };
                let b = if h - g.h < 400.0 { self.world.building(la, lo).filter(|b| !b.bridge && h < b.top + 3.0 && h > b.base - 3.0) } else { None };
                if b.is_some() || h < g.h + 3.0 {
                    (tti, tower) = (t, b.is_some());
                    break;
                }
            }
        }
        // A gentle descent with the gear down is a landing, not a warning
        // (unless it is into a building).
        let landing = p.gear >= 1.0 && vs > -5.0 && !tower;
        let closing = if vs < -1.0 { agl / -vs } else { f64::INFINITY };
        let now = self.real;
        let trip = [
            !landing && (tti <= 5.0 || closing < 5.0),
            !landing && (tti <= 10.0 || (agl < 46.0 && vs < -1.5 && p.gear < 1.0)),
            !landing && agl < 600.0 && vs < -12.0,
            p.gear < 1.0 && agl < 150.0 && p.throttle < 0.35,
        ];
        for (t, on) in self.warn_t.iter_mut().zip(trip) {
            if on {
                *t = now;
            }
        }
        let held = |i: usize| now - self.warn_t[i] < 0.6;
        (w.pull_up, w.too_low, w.sink, w.gear) = (held(0), held(1), held(2), held(3));
        w
    }

    /// Tell the sound what the aircraft and the camera are doing.
    fn update_audio(&mut self, cam: DVec3, fire: f32, in_menu: bool, c: &Controls, dt: f64, scrape: f32) {
        self.warn = if dt > 0.0 { self.warnings() } else { audio::Warn::default() };
        let live = !self.crashed && !in_menu && dt > 0.0;
        let p = &self.plane;
        let up = p.pos.normalize();
        let vs = p.vel.dot(up);
        let air = p.airspeed().length();
        // Jolts: quick changes in load stand out from a slow-following copy.
        let dg = p.g_load - self.g_slow;
        self.g_slow += dg * (dt * 8.0).min(1.0);
        let jolt = if live && !p.on_ground { dg.abs() } else { 0.0 };
        let mut events = Vec::new();
        if live {
            if jolt > 0.35 && self.real - self.last_bump > 0.15 {
                self.last_bump = self.real;
                events.push(audio::Event::Bump(((jolt - 0.2) * 1.5) as f32));
            }
            if p.on_ground && !self.was_ground {
                events.push(audio::Event::Touch((-self.prev_vs).max(0.5) as f32));
            }
            if (p.gear - self.prev_gear).abs() < 0.1 {
                if p.gear >= 1.0 && self.prev_gear < 1.0 {
                    events.push(audio::Event::Clunk(1.0));
                } else if p.gear <= 0.0 && self.prev_gear > 0.0 {
                    events.push(audio::Event::Clunk(0.8));
                } else if (self.prev_gear >= 1.0 || self.prev_gear <= 0.0) && p.gear > 0.0 && p.gear < 1.0 {
                    events.push(audio::Event::Clunk(0.4));
                }
            }
        }
        let gear_moving = live && p.gear > 0.0 && p.gear < 1.0 && p.gear != self.prev_gear;
        let flaps_moving = live && p.flaps != self.prev_flaps;
        (self.was_ground, self.prev_vs, self.prev_gear, self.prev_flaps) = (p.on_ground, vs, p.gear, p.flaps);
        let sa = p.stall_alpha();
        let ground_speed = (p.vel - up * vs).length();
        // The sound was tuned on the light single: a jet's airflow is scaled to match.
        let jet = !matches!(p.spec.sound, aircraft::Sound::Piston);
        // (one flying many times a 747's speeds sounds as the 747 at the same share of them)
        let sk = if jet { 0.45 } else { 1.0 } / p.spec.speed_k;
        let params = audio::Params {
            live: if live { 1.0 } else { 0.0 },
            master: if self.sound_on { self.volume } else { 0.0 },
            rpm: (p.prop_rate * 60.0 / std::f64::consts::TAU) as f32,
            power: p.power as f32,
            throttle: p.throttle as f32,
            health: p.dmg.engine,
            prop: p.dmg.prop,
            airspeed: (if jet { p.ias() } else { air } * sk) as f32,
            vne: (p.spec.vne * sk) as f32,
            jet: if jet { 1.0 } else { 0.0 },
            n1: p.n1 as f32,
            fighter: if matches!(p.spec.sound, aircraft::Sound::Fighter) { 1.0 } else { 0.0 },
            burner: p.burner as f32,
            g_load: p.g_load as f32,
            rough: p.air.sigma_w as f32,
            jolt: jolt as f32,
            buffet: if p.on_ground { 0.0 } else { (smoothstep(sa - 0.07, sa + 0.03, p.alpha.abs()) * smoothstep(10.0, 35.0, air)) as f32 },
            flutter: p.dmg.flutter,
            gear: p.gear as f32,
            gear_moving: if gear_moving { 1.0 } else { 0.0 },
            flaps: p.flaps as f32,
            flaps_moving: if flaps_moving { 1.0 } else { 0.0 },
            rolling: if p.on_ground { ground_speed as f32 } else { 0.0 },
            braking: if c.brake && p.on_ground { 1.0 } else { 0.0 },
            scrape,
            foliage: p.brushes.min(1.0) as f32,
            cracked: if p.dmg.crack[2] > 0.0 { 1.0 } else { 0.0 },
            cabin: if self.cam_mode == 1 && !self.crashed && !in_menu { 1.0 } else { 0.0 },
            dist: (cam - p.pos).length() as f32,
            fire,
            warn: self.warn,
        };
        if let Some(a) = &self.audio {
            for e in events {
                a.event(e);
            }
            a.set(params);
        }
    }

    /// The settings worth keeping, as they are now.
    fn live_settings(&self) -> settings::Settings {
        settings::Settings { lang: self.prefs_lang, sound_on: self.sound_on, volume: self.volume, cam_mode: self.cam_mode, cockpit_fov: self.cockpit_fov, look_sens: self.look_sens, show_hud: self.show_hud, minimap: self.map.mini_on, taa: self.taa, live_traffic: self.live.enabled(), live_weather: self.wx.live, real_time: self.wx.real_time, graphics: self.gfx_keep.map(|g| g.keep_run(&self.gfx0, &self.gfx_base)) }
    }

    /// Saves the settings when they have changed (looked at once a second,
    /// so dragging a slider does not write the file every frame).
    fn keep_settings(&mut self, now: bool) {
        // (the first frame takes the run's starting point: flags applied, nothing changed yet)
        if self.prefs_start.is_none() {
            self.prefs_start = Some(self.live_settings());
        }
        if !now && self.real - self.prefs_t < 1.0 {
            return;
        }
        self.prefs_t = self.real;
        let s = self.live_settings().keep_run(self.prefs_start.as_ref().unwrap_or(&self.prefs_load), &self.prefs_load);
        if s != self.prefs {
            s.save();
            self.prefs = s;
        }
    }

    /// The Graphics settings, as rows of a grid (the pause menu's Graphics
    /// page and the start screen's settings). All apply at once (`apply_graphics`).
    fn graphics_settings(&mut self, ui: &mut egui::Ui, id: &str) {
        use egui::RichText;
        use settings::Preset;
        let integrated = self.r.integrated;
        let (mut g, mut taa) = (self.gfx, self.taa);
        let levels = |ui: &mut egui::Ui, v: &mut u8, names: &[&str]| {
            ui.horizontal(|ui| {
                for (i, n) in names.iter().enumerate() {
                    ui.selectable_value(v, i as u8, *n);
                }
            });
        };
        let (off, low, med, high) = (t!("Off", "Désactivé"), t!("Low", "Faible"), t!("Medium", "Moyen"), t!("High", "Élevé"));
        egui::Grid::new(id).num_columns(2).spacing([16.0, 8.0]).show(ui, |ui| {
            ui.spacing_mut().slider_width = 200.0;
            ui.label(t!("Quality", "Qualité"));
            ui.horizontal(|ui| {
                let cur = g.preset_of(taa, integrated);
                for p in Preset::ALL {
                    if ui.selectable_label(cur == Some(p), preset_name(p)).clicked() {
                        g.apply(p, integrated);
                        taa = true;
                    }
                }
                // (shown, not picked: changing any option below makes it)
                let _ = ui.selectable_label(cur.is_none(), t!("Custom", "Personnalisé"));
            });
            ui.end_row();
            menu::heading(ui, t!("RESOLUTION", "RÉSOLUTION"));
            ui.end_row();
            ui.label(t!("Resolution scale", "Échelle de rendu"));
            ui.add(egui::Slider::new(&mut g.scale, settings::MIN_SCALE..=1.0).step_by(0.05).custom_formatter(|v, _| format!("{:.0}%", v * 100.0)));
            ui.end_row();
            ui.label(t!("Dynamic resolution", "Résolution dynamique"));
            ui.horizontal(|ui| {
                ui.checkbox(&mut g.dyn_res, t!("aim for", "viser"));
                ui.add_enabled_ui(g.dyn_res, |ui| {
                    for f in [30, 45, 60, 90, 120] {
                        ui.selectable_value(&mut g.target_fps, f, format!("{f} fps"));
                    }
                });
                if g.dyn_res {
                    ui.label(RichText::new(t!("now {:.0}%", "{:.0}% en ce moment", self.r.render_scale() * 100.0)).size(12.0).color(menu::DIM));
                }
            });
            ui.end_row();
            ui.label(t!("Upscaler", "Mise à l'échelle"));
            ui.horizontal(|ui| {
                ui.selectable_value(&mut g.upscaler, settings::Upscaler::Taau, "TAAU");
                let fsr = ui.add_enabled(self.r.fsr_supported(), egui::Button::selectable(g.upscaler == settings::Upscaler::Fsr, "AMD FSR 3"));
                if fsr.clicked() {
                    g.upscaler = settings::Upscaler::Fsr;
                }
                fsr.on_disabled_hover_text(t!("This GPU cannot run it", "Ce GPU ne peut pas le faire tourner"));
                if !taa {
                    ui.label(RichText::new(t!("needs temporal AA", "requiert l'AA temporel")).size(12.0).color(menu::DIM));
                }
            });
            ui.end_row();
            menu::heading(ui, t!("SCENE", "SCÈNE"));
            ui.end_row();
            ui.label(t!("Clouds", "Nuages"));
            levels(ui, &mut g.clouds, &[low, med, high]);
            ui.end_row();
            ui.label(t!("Shadows", "Ombres"));
            levels(ui, &mut g.shadows, &[off, low, med, high]);
            ui.end_row();
            ui.label(t!("Ambient occlusion", "Occlusion ambiante"));
            ui.checkbox(&mut g.ao, "");
            ui.end_row();
            ui.label("Bloom");
            ui.checkbox(&mut g.bloom, "");
            ui.end_row();
            ui.label(t!("Terrain detail", "Détail du terrain"));
            levels(ui, &mut g.terrain, &[low, med, high, "Ultra"]);
            ui.end_row();
            ui.label(t!("Water", "Eau"));
            levels(ui, &mut g.water, &[low, high]);
            ui.end_row();
            ui.label(t!("Anti-aliasing", "Anticrénelage"));
            ui.horizontal(|ui| {
                ui.checkbox(&mut taa, t!("temporal (TAA)", "temporel (TAA)"));
                ui.checkbox(&mut g.msaa, t!("MSAA 4x (sharper edges, costly)", "MSAA 4x (bords plus nets, coûteux)"));
            });
            ui.end_row();
            menu::heading(ui, t!("DISPLAY", "AFFICHAGE"));
            ui.end_row();
            ui.label(t!("Fullscreen", "Plein écran"));
            ui.checkbox(&mut g.fullscreen, "F11");
            ui.end_row();
            ui.label(t!("VSync", "Synchro verticale"));
            ui.checkbox(&mut g.vsync, t!("wait for the display", "attendre l'écran"));
            ui.end_row();
            ui.label(t!("Frame rate limit", "Limite d'images"));
            ui.horizontal(|ui| {
                ui.selectable_value(&mut g.fps_limit, 0, t!("Unlimited", "Illimitée"));
                for f in [30, 60, 120] {
                    ui.selectable_value(&mut g.fps_limit, f, f.to_string());
                }
            });
            ui.end_row();
            ui.label(t!("Show FPS", "Afficher les FPS"));
            ui.checkbox(&mut g.show_fps, t!("top right (F4)", "en haut à droite (F4)"));
            ui.end_row();
        });
        ui.add_space(4.0);
        ui.label(RichText::new(t!("Clouds, resolution and shadows cost the most. Shadows Off also hides the aircraft's own.", "Les nuages, la résolution et les ombres coûtent le plus. Sans ombres, l'avion n'a plus la sienne non plus.")).size(12.0).color(menu::DIM));
        // The slider snaps in f64 steps: a value it only re-rounds isn't a change
        // (else merely opening the page would count as the player's choice, and be saved).
        if (g.scale - self.gfx.scale).abs() < 1e-4 {
            g.scale = self.gfx.scale;
        }
        (self.gfx, self.taa) = (g, taa);
    }

    /// The frame rate and frame time in the top right corner (F4), with or without the HUD.
    fn fps_counter(&self, ctx: &egui::Context) {
        use egui::{Color32, FontId, RichText};
        if !self.gfx.show_fps {
            return;
        }
        let ms = self.frame_ms.max(0.01);
        egui::Area::new(egui::Id::unique("fps_counter")).order(egui::Order::Foreground).anchor(egui::Align2::RIGHT_TOP, [-6.0, 4.0]).interactable(false).show(ctx, |ui| {
            egui::Frame::new().fill(Color32::from_black_alpha(150)).corner_radius(4.0).inner_margin(egui::Margin::symmetric(6, 2)).show(ui, |ui| {
                ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Extend);
                ui.label(RichText::new(format!("{:.0} fps · {:.1} ms", 1000.0 / ms, ms)).font(FontId::monospace(12.0)).color(Color32::WHITE));
            });
        });
    }

    /// English / Français, as selectable buttons.
    fn language_picker(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            for l in i18n::Lang::ALL {
                if ui.selectable_label(i18n::lang() == l, l.name()).clicked() {
                    i18n::set_lang(l);
                    self.prefs_lang = Some(l);
                }
            }
        });
    }

    /// Hands the Graphics settings to the renderer and the world when they
    /// have changed (all are live: the menus change them as they are drawn).
    fn apply_graphics(&mut self) {
        let g = self.gfx;
        if self.gfx_applied == Some(g) {
            return;
        }
        // Changes after the start are the player's, to keep.
        if self.gfx_applied.is_some() {
            self.gfx_keep = Some(g);
        }
        self.r.set_clouds(g.clouds);
        self.r.set_shadow_res(if g.shadows == 1 { render::SHADOW_RES / 2 } else { render::SHADOW_RES });
        self.r.ao = g.ao;
        self.r.bloom = g.bloom;
        self.r.shadow_fast = g.shadows <= 2;
        self.r.set_msaa(g.msaa);
        self.r.set_water(g.water);
        self.world.ocean.set_quality(g.water);
        self.r.set_upscaler(g.upscaler == settings::Upscaler::Fsr);
        self.r.set_vsync(g.vsync);
        // (the window's resize event then resizes the surface and the targets)
        if g.fullscreen != self.window.fullscreen().is_some() {
            self.window.set_fullscreen(g.fullscreen.then_some(winit::window::Fullscreen::Borderless(None)));
        }
        self.world.lod_k = settings::TERRAIN_LOD[g.terrain.min(3) as usize];
        // Dynamic resolution starts from the most it may use, and stays under it.
        let keep = g.dyn_res && self.gfx_applied.is_some_and(|a| a.dyn_res);
        self.r.set_render_scale(if keep { self.r.render_scale().min(g.scale) } else { g.scale });
        self.dyn_t = Instant::now();
        // (a new shadow map is empty, and an old one may be stale)
        self.casc_ok = false;
        self.gfx_applied = Some(g);
    }

    /// Dynamic resolution: a step (0.05) down while the smoothed frame time is
    /// over the target's, a step up while it holds the target, at most one every
    /// 2 s (each one remakes the render targets, and needs a moment to show).
    /// The target can't beat the frame rate limit, nor the display with vsync
    /// on (frames can't come faster than it refreshes), else it would only ever
    /// step down. A step up undone within 4 s marks that scale too heavy for
    /// 30 s, so it doesn't see-saw.
    fn dynamic_resolution(&mut self, dt: f64) {
        let g = self.gfx;
        if !g.dyn_res || self.record.is_some() {
            return;
        }
        self.dyn_ms += (dt * 1000.0 - self.dyn_ms) * 0.05;
        if self.dyn_t.elapsed().as_secs_f64() < 2.0 {
            return;
        }
        let mut fps = g.target_fps.max(1) as f64;
        if g.fps_limit > 0 {
            fps = fps.min(g.fps_limit as f64);
        }
        if g.vsync {
            if let Some(hz) = self.window.current_monitor().and_then(|m| m.refresh_rate_millihertz()) {
                fps = fps.min(hz as f64 / 1000.0);
            }
        }
        let target = 1000.0 / fps;
        let s = self.r.render_scale();
        let banned = |n: f32, b: (f32, Instant)| (n - b.0).abs() < 0.01 && b.1.elapsed().as_secs_f64() < 30.0;
        let step = if self.dyn_ms > target * 1.08 {
            -0.05
        } else if self.dyn_ms < target * 1.02 && !banned(((s + 0.05) * 20.0).round() / 20.0, self.dyn_ban) {
            0.05
        } else {
            0.0
        };
        let n = (((s + step) * 20.0).round() / 20.0).clamp(settings::DYN_MIN_SCALE, g.scale);
        if (n - s).abs() > 0.01 {
            if n < s && self.dyn_up && self.dyn_t.elapsed().as_secs_f64() < 4.0 {
                self.dyn_ban = (s, Instant::now());
            }
            self.dyn_up = n > s;
            self.r.set_render_scale(n);
            self.dyn_t = Instant::now();
        }
    }

    /// The frame rate limit: sleep, then spin the last stretch (sleeps
    /// overshoot by a millisecond or so), counting from the last frame's start.
    fn limit_rate(&mut self) {
        let lim = self.gfx.fps_limit;
        if lim > 0 && self.record.is_none() {
            let until = self.frame_t0 + std::time::Duration::from_secs_f64(1.0 / lim as f64);
            let left = until.saturating_duration_since(Instant::now());
            if left > std::time::Duration::from_micros(2000) {
                std::thread::sleep(left - std::time::Duration::from_micros(1500));
            }
            while Instant::now() < until {
                std::hint::spin_loop();
            }
        }
        self.frame_t0 = Instant::now();
    }

    /// The route being flown: move on past each waypoint reached, and with
    /// the autopilot on, bank toward the next one and hold the altitude
    /// (the pilot's own stick input wins).
    fn route_step(&mut self, c: &mut Controls) {
        let (la, lo, _) = geo::pos_to_lla(self.plane.pos);
        let (lat, lon) = (la.to_degrees(), lo.to_degrees());
        if let Some(name) = self.route.advance(lat, lon, self.plane.on_ground, self.plane.vel.length()) {
            self.status = if self.route.done { t!("Arrived: {}", "Arrivée : {}", name) } else { t!("Waypoint {} passed", "Étape {} franchie", name) };
            self.status_until = self.real + 5.0;
        }
        let Some(t) = self.route.target().filter(|_| self.route.autopilot && !self.plane.on_ground) else { return };
        let (e, n, u) = geo::enu(self.plane.pos);
        let f = self.plane.forward();
        let hdg = f.dot(e).atan2(f.dot(n)).to_degrees();
        let err = (route::bearing(lat, lon, t.lat, t.lon) - hdg + 540.0).rem_euclid(360.0) - 180.0;
        let want = (err * 0.8).clamp(-25.0, 25.0).to_radians();
        let bank = (-self.plane.right().dot(u)).clamp(-1.0, 1.0).asin();
        if c.roll == 0.0 {
            c.roll = ((want - bank) * 1.5).clamp(-0.5, 0.5);
        }
        if c.pitch == 0.0 {
            let vs = self.plane.vel.dot(u);
            c.pitch = (-vs * 0.02).clamp(-0.25, 0.25);
        }
    }

    fn frame(&mut self) {
        self.limit_rate();
        self.apply_graphics();
        self.keep_settings(false);
        let now = Instant::now();
        let dt_full = (now - self.last).as_secs_f64();
        self.frame_ms += (dt_full * 1000.0 - self.frame_ms) * 0.1;
        self.dynamic_resolution(dt_full);
        let mut dt_real = dt_full.min(0.05);
        if self.fps_log.0 {
            let l = &mut self.fps_log;
            l.2 += 1;
            l.3 = l.3.max((now - self.last).as_secs_f64());
            let el = (now - l.1).as_secs_f64();
            if el >= 1.0 {
                let n = l.2 as f64;
                let st = std::mem::take(&mut self.r.stats);
                log::info!("{:.1} fps avg ({:.1} ms), slowest frame {:.1} ms | game {:.1}  wait-gpu {:.1}  encode {:.1}  submit {:.1} ms", n / el, el / n * 1000.0, l.3 * 1000.0, st[0] / n * 1000.0, st[1] / n * 1000.0, st[2] / n * 1000.0, st[3] / n * 1000.0);
                (l.1, l.2, l.3) = (now, 0, 0.0);
            }
        }
        if self.record.is_some() {
            // Frozen for a few wall-clock seconds so the city streams in first.
            dt_real = if self.rec_t0.elapsed().as_secs_f64() < 8.0 { 0.0 } else { 1.0 / REC_FPS };
        }
        self.last = now;
        self.real += dt_real;
        self.frame += 1;
        if self.record.is_none() {
            self.fps += (1.0 / dt_real.max(1e-4) - self.fps) * 0.05;
        }

        // Results of background lookups.
        if let Some(rx) = &self.goto_rx {
            if let Ok(res) = rx.try_recv() {
                self.goto_rx = None;
                match res {
                    Some((lat, lon, elev, name)) if self.mode == Mode::Menu => {
                        self.menu.look_at(lat, lon, 45_000.0);
                        self.menu.pick = Some((lat, lon));
                        self.menu.pick_name = name;
                        let _ = elev;
                    }
                    Some((lat, lon, elev, name)) => {
                        let local = (self.utc + geo::pos_to_lla(self.plane.pos).1.to_degrees() / 15.0).rem_euclid(24.0);
                        self.utc = (local - lon / 15.0).rem_euclid(24.0);
                        self.respawn(lat.to_radians(), lon.to_radians(), elev + 450.0, 0.0);
                        self.status = name;
                    }
                    None => {
                        self.status = t!("Place not found", "Lieu introuvable").into();
                        self.menu.pick_name = t!("Place not found", "Lieu introuvable").into();
                    }
                }
                self.status_until = self.real + 8.0;
            }
        }
        // Route stops found online (a stop typed over since is left alone).
        self.route_rx.retain(|rx| match rx.try_recv() {
            Ok((i, q, w)) => {
                if let Some(st) = self.route.stops.get_mut(i).filter(|st| st.text.trim() == q) {
                    st.busy = false;
                    st.missing = w.is_none();
                    st.wp = w;
                }
                false
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => true,
            Err(_) => false,
        });
        if let Some(rx) = &self.name_rx {
            if let Ok(name) = rx.try_recv() {
                self.menu.pick_name = name;
                self.name_rx = None;
            }
        }
        if let Some(rx) = &self.start_rx {
            if let Ok((lat, lon, elev)) = rx.try_recv() {
                self.start_rx = None;
                self.utc = (self.menu.hour - lon / 15.0).rem_euclid(24.0);
                self.respawn(lat.to_radians(), lon.to_radians(), elev + self.menu.start_agl, self.heading);
                self.status = self.menu.pick_name.clone();
                self.status_until = self.real + 10.0;
                self.mode = Mode::Fly;
                self.menu.starting = false;
            }
        }
        if let Some(rx) = &self.airport_rx {
            if let Ok((idx, g, rw, appr, keep)) = rx.try_recv() {
                self.airport_rx = None;
                self.airport_ready(idx, g, rw, appr, keep);
            }
        }
        if let Some((idx, rw, appr)) = self.pending_airport.take() {
            self.start_airport(idx, rw, appr, false);
        }
        // Released once the ground under the wheels is there in full detail.
        if let Some((lat, lon, until)) = self.hold_spot {
            let fine = self.world.ground_zoom(lat, lon).is_some_and(|z| z >= self.world.max_zoom.min(16));
            if (fine && self.real > until - 28.0) || self.real > until {
                if let Some(gr) = self.world.ground(lat, lon) {
                    let up = self.plane.pos.normalize();
                    self.plane.pos = up * (geo::R + gr.h + self.plane.wheel_height() * self.plane.up().dot(up));
                }
                self.plane.hold = false;
                self.plane.t = 2.0;
                self.hold_spot = None;
            }
        }
        if let Some(t) = self.menu_go {
            if self.real > t && self.mode == Mode::Menu && !self.menu.starting {
                self.pick(self.menu.lat, self.menu.lon);
                self.begin();
                self.menu_go = None;
            }
        }
        let in_menu = self.mode == Mode::Menu;

        if let Some((page, t)) = self.pause_at {
            if self.real >= t && self.mode == Mode::Fly {
                if page.is_info() {
                    (self.info_open, self.info_page) = (true, page);
                } else {
                    self.pause_page = page;
                    self.set_pause_menu(true);
                }
                self.pause_at = None;
            }
        }
        if let Some(t) = self.map_at {
            if self.real >= t && self.mode == Mode::Fly {
                self.map.open = true;
                self.map_at = None;
                // the same path as clicks on the map's airports
                for c in std::mem::take(&mut self.map_pick) {
                    if let Some(i) = airports::Db::get().find(&c) {
                        self.status = minimap::plan(&mut self.route, minimap::Plan::Add(i));
                        self.status_until = self.real + 3.0;
                    }
                }
            }
        }
        // Time acceleration drops back to 1× when paused (P), after a crash
        // and when slow on the ground; the pause menu keeps it (it is set there).
        if self.sim_speed > 1 && (self.paused || self.crashed || in_menu || (self.plane.on_ground && self.plane.vel.length() < 15.4)) {
            self.sim_speed = 1;
            self.status = t!("Sim speed 1×", "Vitesse de simulation 1×").into();
            self.status_until = self.real + 2.5;
        }
        let scale = if self.paused || in_menu || self.pause_open || self.real < self.hold { 0.0 } else { self.sim_speed as f64 };
        let dt = dt_real * scale;
        self.time += dt;
        if !in_menu {
            // The clock runs at the panel's rate; past midnight the date turns over.
            let t = self.utc + dt * self.wx.time_rate / 3600.0;
            self.doy = (self.doy + (t / 24.0).floor()).rem_euclid(365.0);
            self.utc = t.rem_euclid(24.0);
            if self.wx.real_time {
                (self.utc, self.doy) = weather::real_clock();
                self.wx.time_rate = 1.0;
            }
            // The air the aircraft flies through, gusts and all.
            let (la, lo, h) = geo::pos_to_lla(self.plane.pos);
            let g = self.world.ground(la, lo);
            let agl = h - g.as_ref().map(|g| g.h).unwrap_or(0.0);
            let water = g.map(|g| g.water).unwrap_or(false);
            self.wx.sun = self.sun_dir().dot(self.plane.pos.normalize());
            // Turbulence, thermals and storm drafts ride on top of the gusting wind.
            let air = self.wx.air(self.plane.pos, agl, self.time, water);
            self.plane.air = air;
            self.plane.wind = self.wx.at(self.plane.pos, agl, self.time) + self.plane.pos.normalize() * air.draft;
            self.wx.advance(dt, self.plane.pos);
        }

        // Physics at a fixed step.
        let mut c = self.controls(dt_real);
        if !in_menu && !self.crashed {
            self.route_step(&mut c);
        }
        let mut emit = Vec::new();
        let mut femit: Vec<(DVec3, render::Particle)> = Vec::new();
        let (mut scrape_n, mut steps) = (0usize, 0usize);
        if !self.crashed && !in_menu {
            self.acc += dt;
            let h = 1.0 / 180.0;
            while self.acc >= h {
                self.acc -= h;
                let imp = self.plane.step(h, &c, &self.world);
                self.gauges.update(h, &self.plane, self.time);
                steps += 1;
                scrape_n += self.plane.sparks.len();
                // Anything the aircraft ploughed through goes over in its direction of travel.
                if !self.plane.tree_hits.is_empty() {
                    let hits: Vec<_> = self.plane.tree_hits.drain(..).collect();
                    self.anchor_fx(hits[0].pos);
                    for t in hits {
                        self.destr.knock(&mut self.world, &self.r.queue, &t, self.plane.vel, 1.3, 0.0, 0.0);
                    }
                    if let Some(a) = &self.audio {
                        a.event(audio::Event::Crunch((self.plane.vel.length() / 50.0).clamp(0.4, 1.6) as f32));
                    }
                }
                // Parts that came off: let go of them where the aircraft is now.
                if !self.plane.shed.is_empty() {
                    self.anchor_fx(self.plane.pos);
                    let pose = self.plane.pose(&self.model.pieces);
                    let reqs: Vec<_> = self.plane.shed.drain(..).collect();
                    for req in &reqs {
                        self.sheds.spawn(req, &self.plane.dmg_prev, &self.model.pieces, &self.piece_mesh, &self.piece_cut, &pose, self.plane.pos, self.plane.rot, self.plane.vel, &mut self.rng, &mut femit);
                    }
                }
                for n in self.plane.notes.drain(..) {
                    // Something broke: back to real time.
                    self.sim_speed = 1;
                    self.flash.push((n, self.real));
                    if let Some(a) = &self.audio {
                        a.event(audio::Event::Snap(1.0));
                    }
                }
                if !self.plane.splashes.is_empty() {
                    self.anchor_fx(self.plane.pos);
                    for (p, v) in self.plane.splashes.drain(..) {
                        if self.rng.f() < 0.25 {
                            let r = &mut self.rng;
                            femit.push((p, render::Particle { vel: (v.as_vec3() + r.dir() * 2.0).into(), life: r.range(1.0, 2.5), size0: r.range(0.15, 0.35), size1: r.range(0.5, 1.2), kind: crash::DROPS, drag: r.range(0.2, 0.8), seed: r.f(), alpha: r.range(0.3, 0.55), ..Default::default() }));
                        }
                    }
                }
                if !self.plane.dirt.is_empty() {
                    self.anchor_fx(self.plane.pos);
                    let up = self.plane.pos.normalize().as_vec3();
                    let field = self.world.airfield_near(self.plane.pos, 3_000.0);
                    for (p, v, amt) in self.plane.dirt.drain(..) {
                        // Paved runways throw up nothing; dirt, grass and fields do.
                        if field.is_some_and(|f| f.geo.on_runway(f.geo.local(p), 2.0).is_some_and(|ri| f.geo.runways[ri].paved)) {
                            continue;
                        }
                        let speed = v.length() as f32;
                        let n = amt * (speed / 8.0).clamp(0.3, 1.5) * 0.2;
                        let r = &mut self.rng;
                        let mut k = n;
                        while k > 0.0 && r.f() < k {
                            k -= 1.0;
                            let big = amt > 1.5;
                            let vel = v.as_vec3() * r.range(0.6, 1.2) + r.dir() * (1.0 + speed * 0.15) + up * r.range(0.5, 2.5);
                            femit.push((p, render::Particle { vel: vel.into(), life: r.range(1.5, 4.0) * if big { 1.5 } else { 1.0 }, size0: r.range(0.3, 0.7), size1: r.range(2.0, 4.0) * if big { 1.6 } else { 1.0 }, kind: crash::DUST, drag: 1.4, buoy: 0.15, seed: r.f(), alpha: r.range(0.45, 0.75), ..Default::default() }));
                        }
                    }
                }
                if !self.plane.sparks.is_empty() {
                    self.anchor_fx(self.plane.pos);
                    let up = self.plane.pos.normalize().as_vec3();
                    for (p, v) in self.plane.sparks.drain(..) {
                        if self.rng.f() < 0.3 {
                            let r = &mut self.rng;
                            femit.push((p, render::Particle { vel: (v.as_vec3() + r.dir() * 3.0 + up * 1.5).into(), life: r.range(0.3, 0.9), size0: 0.05, size1: 0.02, kind: crash::SPARK, drag: 0.6, seed: r.f(), alpha: 0.9, ..Default::default() }));
                        }
                    }
                }
                if let Some(mut imp) = imp {
                    self.r.clear_particles();
                    let tears = self.wreck_meshes();
                    // A house is no match for an airliner: it goes through
                    // it and on, slowed about as much as the house weighs.
                    let size = (self.spec.span / 5.5).max(1.0);
                    let heavy = size > 2.0;
                    let mut through = 0.0f32;
                    if let Some((tile, idx)) = imp.building.filter(|_| heavy) {
                        if let Some((base, top, area)) = self.world.footprint(tile, idx) {
                            let mb = area.max(25.0) as f64 * (top - base).max(3.0) * 120.0 * 0.3;
                            let keep = self.spec.mass / (self.spec.mass + mb);
                            if keep > 0.55 {
                                through = keep as f32;
                                let (la, lo, _) = geo::pos_to_lla(imp.pos);
                                imp.ground_h = self.world.ground(la, lo).map(|g| g.h).unwrap_or(base);
                            }
                        }
                    }
                    let mut crash = Crash::new(&imp, &self.plane, &self.group_mesh, &self.model.groups, &self.group_pivots, &tears, through, self.rng.0 ^ self.frame, self.wx.steady(imp.pos, 40.0));
                    self.r.fluid.latest = None;
                    let (site, power) = (crash.site, crash.power);
                    self.fx_anchor = site;
                    let mut gone = 0.0;
                    // How far the blast reaches beyond a light aircraft's.
                    let reach = if heavy { size.powf(0.6) } else { 1.0 };
                    if let Some((tile, idx)) = imp.building {
                        // The building that was hit comes down, and any shed or small house right beside it.
                        let (la, lo, alt) = geo::pos_to_lla(site);
                        if through == 0.0 {
                            if let Some(g) = self.world.ground(la, lo) {
                                crash.set_lift(alt - g.h);
                            }
                        }
                        let d = &self.plane.dmg;
                        let hit = destruct::Hitter { right: self.plane.right(), up: self.plane.up(), span: d.wing, shape: &self.spec.hole, full: self.spec.span, stab: d.stab, fin: d.fin };
                        self.bring_down(tile, idx, imp.pos, imp.vel, Some(hit), &mut femit);
                        for (t, i) in self.world.buildings_near(site, 9.0 * (power * reach) as f64, if heavy { 700.0 } else { 160.0 }) {
                            self.bring_down(t, i, imp.pos, imp.vel * 0.3, None, &mut femit);
                        }
                    } else if !imp.water {
                        // Open ground: a bite is taken out of it (a big one
                        // where an airliner came down steeply).
                        let steep = (-imp.vel.dot(site.normalize()) / imp.vel.length().max(1.0)).clamp(0.0, 1.0) as f32;
                        let r = (3.0 + 5.0 * power) * (1.0 + (size.sqrt() - 1.0) * steep);
                        self.world.add_crater(site, r);
                        self.destr.eject(site, r, imp.vel, &mut femit);
                        gone = r as f64;
                        for (t, i) in self.world.buildings_near(site, r as f64 + 4.0 * reach as f64, if heavy { 900.0 } else { 400.0 }) {
                            self.bring_down(t, i, imp.pos, imp.vel * 0.3, None, &mut femit);
                        }
                    }
                    if !imp.water {
                        self.destr.blast_trees(&mut self.world, &self.r.queue, site, (26.0 + 30.0 * power as f64) * reach as f64, gone, &mut femit);
                    }
                    self.crash = Some(crash);
                    self.crashed = true;
                    self.crash_real = self.real;
                    self.orbit = 0.0;
                    self.cam_ang = None;
                    self.cam_clear_t = 0.0;
                    self.cam_rise = 0.0;
                    self.acc = 0.0;
                    break;
                }
            }
        }
        if dt > 0.0 {
            self.destr.update(dt as f32, &self.world, &mut femit);
            // Buildings next door that heavy falling pieces went through.
            let hits: Vec<_> = self.destr.crush.drain(..).collect();
            for (t, i, p, v) in hits {
                self.bring_down(t, i, p, v * 0.4, None, &mut femit);
                self.destr.booms.push(0.5);
            }
            self.sheds.update(dt as f32, &self.world);
        }
        let booms: Vec<f32> = self.destr.booms.drain(..).collect();
        if !booms.is_empty() {
            let near = self.crash.as_ref().map(|cr| 1.0 / (1.0 + ((self.cam_pos - cr.site).length() as f32 / 400.0).powi(2))).unwrap_or(0.5);
            for b in booms {
                if let Some(a) = &self.audio {
                    a.boom(b * (0.4 + 0.6 * near));
                }
            }
        }
        // Smoke from a damaged engine: oil-white while it still runs, black once it has seized.
        let sm = self.plane.dmg.smoke;
        if sm > 0.0 && dt > 0.0 && !self.crashed && !in_menu {
            self.anchor_fx(self.plane.pos);
            self.smoke_acc += sm * 22.0 * dt as f32;
            let exhaust = self.plane.pos + self.plane.rot * DVec3::new(0.335, -0.75, -2.10);
            let up = self.plane.pos.normalize().as_vec3();
            let fwd = self.plane.forward().as_vec3();
            let pv = self.plane.vel.as_vec3();
            let black = self.plane.dmg.smoke_black;
            while self.smoke_acc >= 1.0 {
                self.smoke_acc -= 1.0;
                let r = &mut self.rng;
                femit.push((exhaust, render::Particle { vel: (pv * 0.3 - fwd * 2.0 + r.dir() * 1.5 + up * 0.5).into(), life: r.range(2.0, 4.5), size0: r.range(0.25, 0.5), size1: r.range(2.5, 5.0), temp: 0.05, kind: if black { crash::GAS } else { crash::MIST }, drag: 1.2, buoy: 0.8, seed: r.f(), alpha: r.range(0.25, 0.45) * sm.clamp(0.4, 1.0), ..Default::default() }));
            }
        }
        let mut fire = 0.0f32;
        if let Some(cr) = self.crash.as_mut() {
            if dt > 0.0 {
                // Smoke leaving the top of the fire simulation carries on as particles.
                cr.top_out = match (&cr.domain, self.r.fluid.latest) {
                    (Some(dm), Some(st)) if st.out > 1e-3 => {
                        // Just outside the face it leaves through.
                        let f = st.out_c;
                        let edge = |v: f32| if v < 0.08 { -0.01 } else if v > 0.92 { 1.01 } else { v };
                        let fr = Vec3::new(edge(f.x), edge(f.y), if f.z > 0.92 { 1.01 } else { f.z });
                        let at = dm.at_frac(fr) - cr.site;
                        Some((at.as_vec3(), st.out * 1.8 / 110.0, 5.0))
                    }
                    _ => None,
                };
                cr.update(dt, &self.world, &mut emit);
            }
        }
        // What the wreckage ploughed through on its way comes down.
        let crushed: Vec<_> = self.crash.as_mut().map(|cr| cr.crush.drain(..).collect()).unwrap_or_default();
        for (t, i, p, v) in crushed {
            self.bring_down(t, i, p, v * 0.5, None, &mut femit);
            self.destr.booms.push(0.6);
            if let Some(a) = &self.audio {
                a.event(audio::Event::Crunch(1.4));
            }
        }
        if let Some(cr) = self.crash.as_mut() {
            if let Some(a) = &self.audio {
                let near = 1.0 / (1.0 + ((self.cam_pos - cr.site).length() as f32 / 250.0).powi(2));
                for p in &cr.booms {
                    a.boom(p * (0.35 + 0.65 * near));
                }
                for p in &cr.splashes {
                    a.event(audio::Event::Splash(p * (0.35 + 0.65 * near)));
                }
                let f = if cr.water { 0.0 } else { (cr.light().min(12.0) / 12.0) * near };
                fire = (f + self.destr.rumble.min(1.0) * near) * scale as f32;
            }
            cr.booms.clear();
            cr.splashes.clear();
        }
        for (pos, mut p) in femit {
            p.pos = (pos - self.fx_anchor).as_vec3().into();
            emit.push(p);
        }
        // Smoke and dust that would start inside the fire simulation's grid go
        // into the grid instead, where they can mix and flow with the rest.
        if let Some(cr) = self.crash.as_mut().filter(|_| dt > 0.0) {
            if let Some(dm) = &cr.domain {
                let off = (self.fx_anchor - cr.site).as_vec3();
                let mut conv = Vec::new();
                emit.retain(|p| {
                    let k = p.kind as i32;
                    let pos = Vec3::from(p.pos) + off;
                    if matches!(k, 0 | 2 | 7) && dm.contains(pos, 2.0) {
                        conv.push((pos, *p));
                        false
                    } else {
                        true
                    }
                });
                let keep = 48usize;
                let scale = (conv.len() as f32 / keep as f32).max(1.0);
                let step = scale.ceil() as usize;
                for (pos, p) in conv.iter().step_by(step.max(1)) {
                    let r = (p.size1 * 0.55).clamp(1.5, 7.0);
                    let amt = p.alpha * 1.2 / r * scale / dt as f32;
                    let k = p.kind as i32;
                    cr.fluid.push(fluid::FluidSrc {
                        pos: *pos,
                        radius: r,
                        vel: Vec3::from(p.vel),
                        radial: 1.0,
                        soot: if k == 0 { amt * 0.5 } else { 0.0 },
                        dust: if k == 0 { 0.0 } else { amt },
                        temp: if k == 0 { p.temp.min(1.2) } else { 0.0 },
                        vel_k: 2.5,
                        temp_k: 4.0,
                        ..Default::default()
                    });
                }
            }
        }
        if !emit.is_empty() {
            self.r.emit(&emit);
        }

        // ---- camera
        // (in the 3D scene's pixels, which --render-scale may make fewer than the window's)
        let (w, h) = self.r.render_size();
        let (w, h) = (w as f32, h as f32);
        let mut fov = 60f32.to_radians();
        let (cam, dir, upv);
        if !(self.cam_mode == 1 && !self.crashed && !in_menu) {
            self.head_live = false;
        }
        if in_menu {
            self.menu.update(dt_real);
            (cam, dir, upv) = self.menu.camera();
            // Keep the side of the planet you are looking at in daylight.
            self.utc = (12.6 - self.menu.lon / 15.0).rem_euclid(24.0);
            // Real time: the start time is the time now at the chosen spot.
            if self.wx.real_time {
                let (utc, doy) = weather::real_clock();
                let lon = self.menu.pick.map(|p| p.1).unwrap_or(self.menu.lon);
                self.menu.hour = (utc + lon / 15.0).rem_euclid(24.0);
                self.doy = doy;
            }
            self.cam_pos = cam;
        } else if self.crashed {
            let cr = self.crash.as_ref().unwrap();
            // After an impact up the side of a building, the wreck ends up in the street below.
            let street = cr.site - cr.up * cr.lift;
            let mut target = cr.debris.first().filter(|d| !d.sunk).map(|d| d.pos).unwrap_or(street) * 0.5 + street * 0.5 + cr.up * 6.0;
            // (an airliner's fireball is a few hundred metres across: further back)
            let mut dist = (70.0 + 34.0 * cr.power as f64) * (cr.size as f64).sqrt() + self.cam_dist;
            // (into the water there is no fireball to take in: once the
            // spray has come down, closer, to see the wreck afloat)
            if cr.water {
                let settle = ((cr.time - 4.0) / 8.0).clamp(0.0, 1.0);
                let k = settle * settle * (3.0 - 2.0 * settle);
                dist = (38.0 + 14.0 * cr.power as f64) * (cr.size as f64).sqrt() * (1.0 - 0.7 * k) + self.cam_dist * (1.0 - 0.5 * k);
            }
            let (mut centre, mut rise) = (street, 18.0 + dist * 0.24);
            // An airliner's wreck slides a long way: watch the middle of the
            // trail it leaves, and the fireball over it.
            if let Some(w) = cr.wreck_centre() {
                let w = w - cr.up * (w - street).dot(cr.up);
                if cr.water {
                    // Afloat: watch the wreck from low over the water as it
                    // drifts off from the splash.
                    target = w * 0.85 + street * 0.15 + cr.up * (1.5 * (cr.size as f64).sqrt());
                    centre = w * 0.8 + street * 0.2;
                    rise = (6.0 + dist * 0.18) * (cr.size as f64).sqrt().min(2.0);
                } else {
                    target = w * 0.6 + street * 0.4 + cr.up * (6.0 * (cr.size as f64).sqrt());
                    centre = w * 0.5 + street * 0.5;
                }
            }
            // A tower coming down: back off and watch all of it from the side.
            if let Some((mid, hgt)) = self.destr.focus().filter(|f| (f.0 - cr.site).length() < 800.0) {
                target = target * 0.25 + mid * 0.75;
                dist = dist.max(hgt * 1.2 + 60.0 + self.cam_dist);
                centre = mid;
                rise = dist * 0.1;
            }
            self.cam_look += (target - self.cam_look) * (dt_real * 3.0).min(1.0);
            self.orbit += dt_real * 0.1;
            let (e, n, u) = geo::enu(cr.site);
            let back = (self.cam_pos - centre).normalize_or_zero();
            let mut a = self.cam_ang.unwrap_or_else(|| back.dot(n).atan2(back.dot(e))) + dt_real * 0.12;
            let horiz = |a: f64| e * a.cos() + n * a.sin();
            // Keep a clear view: swing round, or rise a little, to see past
            // whatever building stands in the way.
            if self.real - self.cam_clear_t > 0.25 {
                self.cam_clear_t = self.real;
                let look = centre + u * 6.0;
                let clear = |from: DVec3| {
                    !(0..18).any(|k| {
                        let p = from + (look - from) * (k as f64 / 20.0);
                        let (la, lo, h) = geo::pos_to_lla(p);
                        self.world.building(la, lo).is_some_and(|b| h < b.top + 3.0 && h > b.base - 2.0)
                            // (under the 3D tiles the buildings are the mesh, the ground's top)
                            || (self.world.photo().zone_r > 0.0 && self.world.ground(la, lo).is_some_and(|g| h < g.h + 2.0))
                    })
                };
                // Only move when the view is blocked, and then stay a while.
                if !clear(centre + horiz(a) * dist + u * (rise + dist * self.cam_rise)) && self.real > self.cam_hold {
                    let mut best = (0.0, 0.4);
                    'search: for off in [0.0, 0.35, -0.35, 0.7, -0.7, 1.05, -1.05, 1.4, -1.4, 1.75, -1.75, 2.1, -2.1] {
                        for k in [0.0, 0.15, 0.3] {
                            if clear(centre + horiz(a + off) * dist + u * (rise + dist * k)) {
                                best = (off, k);
                                break 'search;
                            }
                        }
                    }
                    a += best.0;
                    self.cam_rise = best.1;
                    self.cam_hold = self.real + 1.5;
                }
            }
            self.cam_ang = Some(a);
            let want = centre + horiz(a) * dist + u * (rise + dist * self.cam_rise);
            self.cam_pos += (want - self.cam_pos) * (dt_real * 1.2).min(1.0);
            cam = self.cam_pos;
            dir = (self.cam_look - cam).normalize();
            upv = u;
        } else if self.cam_mode == 1 {
            if !self.head_live {
                self.head.reset(self.cam_yaw, self.cam_pitch);
                self.head_live = true;
            }
            let up = self.plane.pos.normalize();
            let day = ((up.dot(self.sun_dir()) as f32 + 0.2) / 0.4).clamp(0.0, 1.0);
            let view = self.head.update(self.spec, dt_real, (self.cam_yaw, self.cam_pitch), self.cockpit_fov, self.plane.head, self.plane.rot, up, w / h, day);
            fov = view.fov.to_radians();
            // A rough engine or damage shakes the whole cabin.
            let vibe = self.plane.dmg.vibe.min(2.0) as f64;
            let mut jit = || self.rng.range(-1.0, 1.0) as f64 * vibe;
            let shake = DVec3::new(jit() * 0.006, jit() * 0.006, 0.0);
            let wob = DQuat::from_rotation_y(jit() * 0.004) * DQuat::from_rotation_x(jit() * 0.004);
            let r = self.plane.rot * wob * view.rot;
            cam = self.plane.pos + self.plane.rot * (view.eye + shake);
            dir = r * DVec3::NEG_Z;
            upv = r * DVec3::Y;
            self.cam_pos = cam;
        } else {
            let u = self.plane.pos.normalize();
            let back = -self.plane.forward();
            self.cam_dir = (self.cam_dir + (back - self.cam_dir) * (dt_real * 3.5).min(1.0)).normalize();
            let flat = (self.cam_dir - u * self.cam_dir.dot(u)).normalize_or_zero();
            let tilt = self.cam_dir.dot(u).asin() * 0.6 + self.cam_pitch;
            let d = DQuat::from_axis_angle(u, self.cam_yaw) * (flat * tilt.cos() + u * tilt.sin());
            let vibe = self.plane.dmg.vibe.min(2.0) as f64;
            // (Offsets scale with the aircraft: the camera sits as far back for its size.)
            let sz = self.spec.cam_dist / 16.0;
            cam = self.plane.pos + d * self.cam_dist + u * (1.5 * sz) + (self.plane.right() * self.rng.range(-1.0, 1.0) as f64 + u * self.rng.range(-1.0, 1.0) as f64) * (vibe * 0.03);
            dir = (self.plane.pos + u * (1.2 * sz) + self.plane.forward() * (6.0 * sz) - cam).normalize();
            upv = (u + self.plane.up() * 0.25).normalize();
            self.cam_pos = cam;
            self.cam_look = self.plane.pos;
        }
        // Never let the camera sink into the ground.
        let (clat, clon, calt) = geo::pos_to_lla(cam);
        let ground = self.world.ground(clat, clon).map(|g| g.h);
        let cam = match ground {
            Some(g) if calt < g + 1.5 => cam + cam.normalize() * (g + 1.5 - calt),
            _ => cam,
        };
        let calt = geo::pos_to_lla(cam).2;
        let agl = calt - ground.unwrap_or(0.0);
        let scrape = if steps > 0 { (scrape_n as f32 / steps as f32 * 0.5).min(1.0) } else { 0.0 };
        self.update_audio(cam, fire, in_menu, &c, dt, scrape);

        let near = if in_menu { (calt * 0.02).clamp(0.4, 20_000.0) as f32 } else if self.cam_mode == 1 && !self.crashed { 0.03f32 } else { 0.25f32 };
        let proj = perspective_rev(fov, w / h, near);
        let cam_proj = [fov, near];
        let view = look_to(Vec3::ZERO, dir.as_vec3(), upv.as_vec3());
        let vp = proj * view;
        // Temporal anti-aliasing: each frame is drawn a different fraction of
        // a pixel off centre (a Halton sequence), and the renderer blends the
        // frames back together. Everything else works from the centred `vp`.
        let jitter = if self.taa {
            let halton = |mut i: u64, b: u64| { let (mut f, mut r) = (1.0f32, 0.0f32); while i > 0 { f /= b as f32; r += f * (i % b) as f32; i /= b; } r };
            // (a scene upscaled temporally needs more offsets to cover the
            // window's finer grid: 8 for each of its pixels, render.rs taau;
            // FSR's own count, for any scale, fsr/jitter.rs)
            let s = self.r.render_scale();
            let n = if self.r.fsr_active() {
                fsr::get_jitter_phase_count(w as i32, self.r.config.width as i32).max(1) as u64
            } else if s < render::TAAU_BELOW {
                (8.0 / (s * s)).ceil().min(32.0) as u64
            } else {
                8
            };
            let k = self.frame % n + 1;
            Vec2::new(halton(k, 2) - 0.5, halton(k, 3) - 0.5) * 2.0 / Vec2::new(w, h)
        } else {
            Vec2::ZERO
        };
        let vp_jit = Mat4::from_translation(jitter.extend(0.0)) * vp;
        let frustum = world::Frustum::new(DMat4::from_cols_array(&vp.to_cols_array().map(|v| v as f64)));
        self.world.update(&self.r, cam, &frustum);
        let focus_abs = if in_menu { cam } else if self.crashed { self.crash.as_ref().unwrap().site } else { self.plane.pos };
        {
            // Photorealistic 3D tiles in flight, day and night (after dusk the
            // mesh takes our street lamps and lit windows, photo.wgsl).
            let day = if in_menu { 0.0 } else { 1.0 };
            let px = h as f64 / (2.0 * (fov as f64 * 0.5).tan());
            self.world.update_photo(&self.r, cam, focus_abs, &frustum, px, day, dt_real);
        }
        if dt > 0.0 {
            self.world.update_traffic(dt as f32, focus_abs);
            let local = (self.utc + geo::pos_to_lla(focus_abs).1.to_degrees() / 15.0).rem_euclid(24.0);
            let player = (!self.crashed && !in_menu).then_some(self.plane.pos);
            self.world.update_airports(dt as f32, self.wx.wind_from as f32, (self.wx.wind * 1.94384) as f32, (cars::activity(local as f32) * 1.2).clamp(0.25, 1.0), player, focus_abs);
        }

        // ---- frame constants
        let (east, north, up) = geo::enu(cam);
        let delta = cam - self.prev_cam;
        self.prev_cam = cam;
        // The flakes stream past along the camera's motion through them
        // (they fall at about a metre a second and drift with the wind).
        if delta.length() < 500.0 && dt > 0.0 {
            let m = delta - (self.wx.steady(cam, 20.0) - up * 1.1) * dt;
            let len = m.length();
            if len > 1e-4 {
                self.snow_axis = (self.snow_axis + (m / len - self.snow_axis) * (dt * 3.0).min(1.0)).normalize_or(up);
            }
            self.snow_s = (self.snow_s + len).rem_euclid(SNOW_WRAP);
        }
        if delta.length() < 5000.0 {
            self.wave_off += DVec2::new(delta.dot(east), delta.dot(north));
            self.wave_off = DVec2::new(self.wave_off.x.rem_euclid(65536.0), self.wave_off.y.rem_euclid(65536.0));
            self.ocean_off += DVec2::new(delta.dot(east), delta.dot(north));
            self.ocean_off = DVec2::new(self.ocean_off.x.rem_euclid(ocean::WRAP), self.ocean_off.y.rem_euclid(ocean::WRAP));
        }
        // The sea follows the weather's wind (and the clock).
        let sinlat = cam.normalize().z;
        self.world.ocean.update(dt, self.time, self.wx.wind, self.wx.wind_from, sinlat, cam, east, north, self.ocean_off);
        let sun = self.sun_dir();
        let (moon, moon_lit) = self.moon_dir();
        // Detail building tiles in range stand in for "how much city is around".
        let city = if in_menu { 0.0 } else { (self.world.stats.2 as f32 / 45.0).min(1.0) };
        self.city_glow += (city - self.city_glow) * (dt_real as f32 * 0.5).min(1.0);
        let mu = up.dot(sun) as f32;
        let day = ((mu + 0.2) / 0.4).clamp(0.0, 1.0);
        let day = day * day * (3.0 - 2.0 * day);
        // From the cockpit the eye adapts a little to the cabin.
        // Exposure moves in stops through twilight (a straight blend opens up
        // far too early, washing the city out while the sky still glows).
        // Night stays a night photograph: dark walls, bright windows.
        let exposure = (6.0f32.ln() + (0.5f32.ln() - 6.0f32.ln()) * day).exp() * if self.cam_mode == 1 && !self.crashed && !in_menu { 2f32.powf(self.head.adapt) } else { 1.0 };
        let v4 = |v: DVec3, w: f64| [v.x as f32, v.y as f32, v.z as f32, w as f32];
        let cam_right = dir.cross(upv).normalize();
        let cam_up = cam_right.cross(dir);
        // Shadow cascades: three nested boxes around the aircraft, each snapped
        // to its own texel grid so shadows do not shimmer as it moves.
        // (Graphics settings: shadows Off draw none, not even the aircraft's own)
        let shadow_on = mu > 0.04 && !in_menu && self.gfx.shadows > 0;
        const CASC: [f32; 3] = [260.0, 1500.0, 8500.0];
        let sres = self.r.shadow_res() as f32;
        let mut casc_vp = [[[0f32; 4]; 4]; render::CASCADES + 4];
        let mut casc_draw = 0b111u8;
        {
            let right = { let r = DVec3::Z.cross(sun); if r.length_squared() < 1e-6 { DVec3::X } else { r.normalize() } };
            let upl = sun.cross(right);
            for (i, r) in CASC.iter().enumerate() {
                let texel = (2.0 * r / sres) as f64;
                let snap = |v: f64| (v / texel).floor() * texel - v;
                let centre = focus_abs - cam + right * snap(focus_abs.dot(right)) + upl * snap(focus_abs.dot(upl));
                let d = r + 3500.0;
                let lview = look_to(centre.as_vec3() + sun.as_vec3() * d, -sun.as_vec3(), upl.as_vec3());
                casc_vp[i] = (ortho(-r, *r, -r, *r, 0.0, 2.0 * d) * lview).to_cols_array_2d();
            }
            // The aircraft's own shadow map: a few metres across, deep enough
            // to reach the ground below it.
            let centre = (self.plane.pos - cam).as_vec3();
            let (pr, pb) = (self.spec.shadow_r, PSHADOW_BACK * self.spec.shadow_r / PSHADOW_R);
            let lview = look_to(centre + sun.as_vec3() * pb, -sun.as_vec3(), upl.as_vec3());
            casc_vp[3] = (ortho(-pr, pr, -pr, pr, 0.0, pshadow_depth(pr)) * lview).to_cols_array_2d();
            // Low and Medium shadows amortise: the near cascade is drawn every
            // frame, the middle and far ones every other frame, in turn. One
            // not drawn is read with the matrix it was drawn with, moved by
            // how far the camera has gone since (the matrices are camera-relative),
            // so its shadows stay put on the ground instead of swimming.
            // Low draws one cascade a frame: near, middle, near, far.
            if self.gfx.shadows <= 2 && self.casc_ok && delta.length() < 500.0 {
                casc_draw = match (self.gfx.shadows, self.frame % 4) {
                    (1, 1) => 2,
                    (1, 3) => 4,
                    (1, _) => 1,
                    (_, f) => 1 | if f % 2 == 0 { 2 } else { 4 },
                };
            }
            for c in 0..3 {
                if casc_draw & (1 << c) != 0 {
                    self.casc_keep[c] = (casc_vp[c], cam);
                } else {
                    let (m, at) = self.casc_keep[c];
                    casc_vp[c] = (Mat4::from_cols_array_2d(&m) * Mat4::from_translation((cam - at).as_vec3())).to_cols_array_2d();
                }
            }
            self.casc_ok = shadow_on;
        }
        let mut shadow_vp = [[[0f32; 4]; 4]; 3];
        shadow_vp.copy_from_slice(&casc_vp[..3]);
        let mut g = Globals {
            view_proj: vp_jit.to_cols_array_2d(),
            inv_view_proj: vp_jit.inverse().to_cols_array_2d(),
            shadow_vp,
            planet: v4(-cam, geo::R),
            sun_dir: v4(sun, SUN_E as f64),
            east: v4(east, self.wave_off.x),
            north: v4(north, self.wave_off.y),
            up: v4(up, calt),
            screen: [w, h, 1.0 / w, 1.0 / h],
            time: [self.time as f32, dt as f32, exposure, self.frame as f32],
            cloud: [if in_menu { 0.0 } else { self.wx.clouds }, self.wx.cloud_base, self.wx.cloud_depth, self.wx.cloud_ceiling()],
            light_pos: [0.0; 4],
            light_col: [0.0; 4],
            vol_o: [0.0; 4],
            vol_u: [0.0; 4],
            misc: [near, self.city_glow, 1.0, 2.0 * (fov * 0.5).tan() / h],
            csm: [CASC[0] * 2.0 / sres, CASC[1] * 2.0 / sres, CASC[2] * 2.0 / sres, if shadow_on { 1.0 } else { 0.0 }],
            scorch: [[0.0; 4]; 8],
            crater: [[0.0; 4]; 8],
            prev_vp: self.prev_vp,
            // A jump (teleport, camera switch) invalidates the cloud history.
            cam_delta: v4(delta, if delta.length() < 500.0 { 1.0 } else { 0.0 }),
            moon: v4(moon, (MOON_E as f64) * moon_lit.powf(1.5)),
            vol_e: [0.0; 4],
            vol_n: [0.0; 4],
            pshadow_vp: casc_vp[3],
            pshadow: [self.spec.shadow_r * 2.0 / self.r.plane_shadow_res() as f32, 0.0012 / pshadow_depth(self.spec.shadow_r), self.r.plane_shadow_res() as f32 / sres, if shadow_on && !self.crashed { 1.0 } else { 0.0 }],
            taa: [jitter.x * 0.5, -jitter.y * 0.5, if self.taa { 1.0 } else { 0.0 }, 0.0],
            plane_inv: [[0.0; 4]; 4],
            plane_prev: [[0.0; 4]; 4],
            plane_box: self.plane_box,
            weather: { let o = self.wx.cloud_off; [o.x as f32, o.y as f32, o.z as f32, if in_menu { 1.0 } else { self.wx.haze }] },
            storm: { let w = self.wx.steady(cam, 2000.0); let d = if w.length() > 0.1 { w.normalize() } else { east }; v4(d, 0.0) },
            storm_cells: [[0.0; 4]; 2 * weather::MAX_STORMS],
            det: [[0.0; 4]; 16],
            photo_vp: [[0.0; 4]; 4],
            photo: [0.0; 4],
            photo_c: [0.0; 4],
            photo_sun: [0.0, 0.0, 0.0, -1.0],
            cut: [[0.0; 4]; 3 * tiles3d::MAX_CUTS],
            cut_v: [[0.0; 4]; tiles3d::MAX_CUT_V / 2],
            cs: [[0.0; 4]; 2 * render::CS_MAP],
            alto: [if in_menu { 0.0 } else { self.wx.alto }, 4800f32.max(self.wx.cloud_base + 0.4 * self.wx.cloud_depth + 800.0), if in_menu { 0.0 } else { self.wx.snow }, if in_menu { 0.0 } else { self.wx.snowfall(calt) }],
            ocean: self.world.ocean.globals(),
            shore_vp: [[0.0; 4]; 4],
            shore: [0.0; 4],
            foam: [[0.0; 4]; 16],
            start: [if in_menu { 1.0 } else { 0.0 }, 0.0, 0.0, 0.0],
            snow: v4(self.snow_axis, self.snow_s),
        };
        if !in_menu {
            let cells = self.wx.storm_cells(cam);
            g.storm[3] = cells.len() as f32;
            for (i, c) in cells.iter().enumerate() {
                g.storm_cells[2 * i] = c[0];
                g.storm_cells[2 * i + 1] = c[1];
            }
            // The flight model flies through the same cells.
            self.wx.cells = cells;
        }
        if !self.crashed && !in_menu {
            // The aircraft keeps pace with the camera, so its pixels are found
            // in last frame's image through where it was then, not through
            // where the ground under them was.
            let base = Mat4::from_rotation_translation(self.plane.rot.as_quat(), (self.plane.pos - cam).as_vec3());
            g.taa[3] = 1.0;
            g.plane_inv = base.inverse().to_cols_array_2d();
            g.plane_prev = (Mat4::from_cols_array_2d(&self.prev_vp) * self.prev_plane).to_cols_array_2d();
            self.prev_plane = base;
        }
        self.prev_vp = vp.to_cols_array_2d();
        for (i, (p, r)) in self.world.craters.iter().take(8).enumerate() {
            g.crater[i] = v4(*p - cam, *r as f64);
        }
        let mut psim = PSim { anchor: v4(self.fx_anchor - cam, 0.0), upv: v4(self.fx_anchor.normalize(), dt), cam_r: v4(cam_right, self.time), cam_u: v4(cam_up, 0.0), wind: v4(self.wx.steady(self.fx_anchor, 10.0), 0.0), shift: v4(self.fx_shift, 0.0), ..Default::default() };
        if dt > 0.0 {
            // The simulation runs this frame and applies the shift.
            self.fx_shift = DVec3::ZERO;
        }
        let mut objs = Vec::new();
        if let Some(cr) = &self.crash {
            // The ground under an impact up the side of a building is further down.
            psim.anchor = v4(cr.site - cam, -cr.lift);
            psim.upv = v4(cr.up, dt);
            g.light_pos = v4(cr.site + cr.up * 8.0 * cr.size as f64 - cam, (12.0 + 8.0 * cr.power as f64) * cr.size as f64);
            let l = cr.light();
            g.light_col = [1.0 * l, 0.5 * l, 0.2 * l, 0.0];
            let grow = (cr.time as f32 / 1.2).min(1.0);
            for (i, (p, rad)) in cr.scorch.iter().take(8).enumerate() {
                g.scorch[i] = v4(cr.site + p.as_dvec3() - cam, (*rad * grow) as f64);
            }
            for (i, (p, rad, k, age, seed)) in cr.foam().into_iter().take(8).enumerate() {
                g.foam[2 * i] = v4(cr.site + p.as_dvec3() - cam, rad as f64);
                g.foam[2 * i + 1] = [k, age, seed, 0.0];
            }
            if self.crashed || cr.time < 240.0 {
                for (i, d) in cr.debris.iter().enumerate() {
                    if d.sunk {
                        continue;
                    }
                    let m = Mat4::from_scale_rotation_translation(Vec3::splat(d.scale), d.rot.as_quat(), (d.pos - cam).as_vec3()) * Mat4::from_translation(-d.origin);
                    objs.push(DrawObj { mesh: d.mesh, model: m, tint: [1.0, 1.0, 1.0, d.char], glow: [if d.burn > 0.0 { if d.rigid { 0.35 } else { 1.0 } } else if cr.water { 0.0 } else { 0.25 }, self.spec.id as f32, self.spec.size_k, i as f32 * 0.37], cast: 0b11, cut: true, tear: [d.tear[0].gpu(), d.tear[1].gpu()], dmg: [d.tear[0].mode, d.tear[1].mode, 0.6, d.jag], ..DrawObj::DEFAULT });
                }
            }
        }
        // The fire simulation: this frame's sources, and the light it gives off.
        let mut fluid = None;
        if let Some(cr) = self.crash.as_mut() {
            let wind = self.wx.steady(cr.site, 10.0);
            // Once the fire is out and its smoke has gone, the grid rests.
            let spent = cr.fluid.is_empty() && cr.time > 8.0 && self.r.fluid.latest.is_some_and(|st| st.soot < 2.0 && st.emit < 1.0);
            if let Some(dm) = cr.domain.as_mut().filter(|_| !spent) {
                fluid = Some(dm.frame(dt, cam, wind, sun, &cr.fluid, &self.world));
                if let Some(st) = self.r.fluid.latest {
                    // A ball about as big as the burning gas, as bright as flame.
                    let (t, p, _) = cr.last_blast;
                    let flash = p * 6.0 * (-(t as f32) / 0.25).exp();
                    let rad = (0.75 * st.emit.max(0.0).cbrt()).clamp(3.0, 25.0 * cr.size);
                    let l = flash + 2.5 * (st.emit / 300.0).min(1.0);
                    g.light_pos = v4(dm.at_frac(st.emit_c) - cam, rad as f64);
                    g.light_col = [1.0 * l, 0.48 * l, 0.17 * l, 0.0];
                }
            }
        }
        let params = self.instruments(agl);
        if !self.crashed && !in_menu {
            let base = Mat4::from_rotation_translation(self.plane.rot.as_quat(), (self.plane.pos - cam).as_vec3());
            let pose = self.plane.pose(&self.model.pieces);
            // Blades while they can be followed by eye, a blur once they cannot.
            let blur = ((self.plane.prop_rate - 30.0) / 25.0).clamp(0.0, 1.0) as f32;
            let dm = self.plane.dmg;
            let kind = self.spec.id as f32;
            // What the exhaust shows (plane.wgsl, the plume and the nozzles): how far the
            // afterburners are lit, and how hard the engines are running dry (0 idle .. 1).
            let (burner, dry) = match &self.spec.engine {
                aircraft::Engine::Fan(fe) => (self.plane.burner as f32, (((self.plane.n1 - fe.idle) / (1.0 - fe.idle)).clamp(0.0, 1.0) * if dm.engine > 0.0 { 1.0 } else { 0.0 }) as f32),
                _ => (0.0, 0.0),
            };
            if burner > 0.01 && self.crash.is_none() {
                // The flames light the tail from behind (turned down at night, as they are, for the eye that has opened up).
                let k = burner * (0.12 + 0.88 * day) * 2.2;
                g.light_pos = v4(self.plane.pos + self.plane.rot * DVec3::new(0.0, 0.0, 10.0) - cam, 1.3);
                g.light_col = [k, 0.5 * k, 0.2 * k, 0.0];
            }
            for (i, p) in self.model.pieces.iter().enumerate() {
                let hide = match p.vis {
                    airframe::Vis::Always => false,
                    airframe::Vis::SlowProp => blur >= 1.0,
                    airframe::Vis::FastProp => blur <= 0.0,
                };
                if hide || dm.hidden(p) {
                    continue;
                }
                let tears = dm.tears_for(p);
                let torn = tears[0].is_some() || tears[1].is_some();
                objs.push(DrawObj {
                    mesh: self.piece_mesh[i],
                    model: base * pose[i],
                    tint: [blur, burner, dry, 0.0],
                    glow: [0.0, kind, self.spec.size_k, 0.0],
                    params,
                    cast: 1 << render::PLANE_LAYER,
                    glass: p.glass,
                    cut: self.piece_cut[i] || torn,
                    tear: [tears[0].gpu(), tears[1].gpu()],
                    dmg: [tears[0].mode, tears[1].mode, dm.scuff[p.group], dm.jag_for(p)],
                    dmg2: [dm.oil, dm.crack[0], dm.crack[1], dm.crack[2]],
                    ..DrawObj::DEFAULT
                });
            }
            // A buckled wing: the same pieces again past the break, swung about
            // the torn skin they hang from.
            for (side, f) in dm.fold.iter().enumerate() {
                if !f.on {
                    continue;
                }
                let g = if side == 0 { airframe::G_LWING } else { airframe::G_RWING };
                let swing = base * f.mat();
                for (i, p) in self.model.pieces.iter().enumerate() {
                    let structure = matches!(p.role, airframe::Role::WingStructure(_) | airframe::Role::TailStructure);
                    if p.group != g || p.glass || (dm.hidden(p) && !structure) {
                        continue;
                    }
                    let stick = if structure { 0.12 * self.spec.span / 5.5 } else { 0.0 };
                    let tears = [f.tear.flip().grow(stick), f.outer.grow(stick)];
                    objs.push(DrawObj {
                        mesh: self.piece_mesh[i],
                        model: swing * pose[i],
                        glow: [0.0, kind, self.spec.size_k, 0.0],
                        params,
                        cast: 1 << render::PLANE_LAYER,
                        cut: true,
                        tear: [tears[0].gpu(), tears[1].gpu()],
                        dmg: [tears[0].mode, tears[1].mode, dm.scuff[p.group].max(0.5), f.tear.jag.max(0.03)],
                        ..DrawObj::DEFAULT
                    });
                }
            }
        }
        if !in_menu {
            self.sheds.draw(cam, params, &mut objs);
        }
        let mut insts = Vec::new();
        let mut fall = Vec::new();
        let mut air_lights = Vec::new();
        if !in_menu {
            let local = (self.utc + clon.to_degrees() / 15.0).rem_euclid(24.0);
            let sun_up = geo::enu(cam).2.dot(self.sun_dir());
            self.world.car_instances(cam, cars::activity(local as f32), sun_up, &mut insts, &mut air_lights);
            self.n_cars = insts.len();
            self.destr.instances(cam, &mut insts, &mut fall);
            self.destr.skin_frame(cam);
            self.destr.upload_skins(&self.r);
            self.world.airport_instances(cam, self.wx.wind_from as f32, (self.wx.wind * 1.94384) as f32, self.time, &mut insts, &mut air_lights);
            self.world.photo_beacons(cam, &mut air_lights);
        }
        self.live.step(&self.src, (!in_menu).then_some(self.plane.pos));
        let wx_at = if in_menu {
            self.menu.pick
        } else {
            let (la, lo, _) = geo::pos_to_lla(self.plane.pos);
            Some((la.to_degrees(), lo.to_degrees()))
        };
        self.livewx.step(&self.src, &mut self.wx, wx_at, if in_menu { 0.0 } else { dt_real });
        if !in_menu {
            self.live.instances(cam, &mut insts, &mut air_lights);
        }
        self.live_view = Some((vp, cam));

        // The globe's labels are placed in the window's pixels, not the (maybe smaller) 3D scene's.
        let (ww, wh) = (self.r.config.width as f32, self.r.config.height as f32);
        let ui = if in_menu { self.menu_ui(vp, cam, ww, wh) } else { self.ui(agl) };
        // The smoke volume: a grid around the effects anchor, on its local east/north/up axes.
        {
            let site = cam + DVec3::new(psim.anchor[0] as f64, psim.anchor[1] as f64, psim.anchor[2] as f64);
            let (ve, vn, vu) = geo::enu(site);
            // (coarser round a big fire, as its grid is: see fluid.rs `coarse`)
            let vs = self.crash.as_ref().and_then(|c| c.domain.as_ref()).map(|d| d.sc as f64).unwrap_or(1.0);
            let (cs, zmin) = (render::VOL_CELL as f64 * vs, render::VOL_ZMIN as f64 * vs);
            let lift = self.crash.as_ref().map(|c| c.lift).unwrap_or(0.0);
            let corner = site - ve * (cs * render::VOL_N[0] as f64 * 0.5) - vn * (cs * render::VOL_N[1] as f64 * 0.5) + vu * (zmin - lift);
            g.vol_o = v4(corner - cam, cs);
            g.vol_u = v4(vu, 0.0);
            g.vol_e = v4(ve, 0.0);
            g.vol_n = v4(vn, 0.0);
            psim.ve = v4(ve, cs);
            psim.vn = v4(vn, zmin - lift);
        }
        let pf = self.world.photo_frame(cam);
        let photo = self.world.photo_draws(cam);
        g.photo_vp = pf.mask_vp.to_cols_array_2d();
        g.photo = pf.zone;
        g.photo_c = pf.centre;
        g.photo_sun = pf.sun;
        g.cut = pf.cut;
        g.cut_v = pf.cut_v;
        casc_vp[render::MASK_SLOT] = g.photo_vp;
        let (tiles, details, near) = self.world.draw_lists(cam, focus_abs, CASC);
        // The shore distance field: a step of its rebuild this frame.
        let sf = self.shore.frame(cam);
        casc_vp[render::SHORE_SLOT] = sf.build_vp;
        casc_vp[render::SHORE_WIDE_SLOT] = sf.build_vp_wide;
        g.shore_vp = sf.vp;
        g.shore = [if sf.on { 1.0 } else { 0.0 }, shore::TEXEL as f32, shore::MAX_D, 2.0 * shore::HALF as f32];
        let shore_tiles = if matches!(sf.step, Some(shore::Step::Raster)) { self.world.shore_tiles(cam, sf.build_c) } else { Vec::new() };
        let shore_c = sf.build_c.as_vec3().to_array();
        let fields = if in_menu { Vec::new() } else { self.world.field_draws(cam) };
        let mut shot = match self.shots.first() {
            Some((t, _, _)) if self.real >= *t => {
                // Logged with each scripted shot, for comparing builds.
                log::info!("fps {:.1}", self.fps);
                Some(self.shots.remove(0).1)
            }
            _ => None,
        };
        let mut rec_done = false;
        if let Some((dir, start, secs)) = &self.record {
            // The synth runs every frame so it is settled when recording starts.
            let n = (audio::Audio::RATE as f64 / REC_FPS).round() as usize;
            let samples = self.audio.as_ref().map(|a| a.take(n)).unwrap_or_default();
            if self.real >= *start {
                if shot.is_none() {
                    let _ = std::fs::create_dir_all(dir);
                    shot = Some(dir.join(format!("f{:05}.png", self.rec_n)));
                    self.rec_n += 1;
                }
                self.rec_audio.extend(samples);
                if self.real >= start + secs {
                    write_wav(&dir.join("audio.wav"), &self.rec_audio, audio::Audio::RATE);
                    rec_done = true;
                }
            }
        }
        // Scripted shots can pre-position the chase camera for the next capture.
        if let Some((_, _, Some((y, p, d)))) = self.shots.first() {
            (self.cam_yaw, self.cam_pitch, self.cam_dist) = (y.to_radians(), *p, *d);
        }
        let had_shots = shot.is_some() && self.shots.is_empty();
        let (skin_slots, skins) = if in_menu { (Vec::new(), Vec::new()) } else { self.destr.skin_draws() };
        self.r.stats[0] += now.elapsed().as_secs_f64();
        self.r.render(Frame { g, tiles, details, photo, skins, skin_slots, fields, lights: air_lights, objs, insts, fall, near, casc_vp, shore: sf.step, shore_c, shore_tiles, psim, sim_steps: dt > 0.0, fluid, ocean: self.world.ocean.uniform(), egui: Some(ui), shot, logo: in_menu.then(|| self.window.scale_factor() as f32), casc_draw, cam: cam_proj });
        if (had_shots && std::env::args().any(|a| a == "--shot") && self.record.is_none()) || rec_done {
            self.quit = true;
        }
    }

    /// The start screen drawn over the globe: labels, the chosen spot, and the panel.
    fn menu_ui(&mut self, vp: Mat4, cam: DVec3, w: f32, h: f32) -> (Vec<egui::ClippedPrimitive>, egui::TexturesDelta, f32) {
        let raw = self.egui_input();
        let mut go: Option<String> = None;
        let mut start = false;
        let mut jump: Option<(f64, f64, &'static str)> = None;
        let mut airport: Option<u32> = None;
        let alt = self.menu.alt;
        self.src.google.update_attribution(self.menu.lat, self.menu.lon, alt);
        let out = self.egui_ctx.clone().run_ui(raw, |root| {
            let ctx = &root.ctx().clone();
            self.pad_pass(ctx);
            self.fps_counter(ctx);
            use egui::{Align2, Color32, FontId, RichText};
            let ppp = ctx.pixels_per_point();
            // With a gamepad, X starts at the crosshair in the middle of the screen.
            let aim = padui::active().then(|| egui::pos2(w * 0.5 / ppp, h * 0.5 / ppp));
            // Screen position of a place, if it is on the near side of the planet.
            let project = |lat: f64, lon: f64| -> Option<egui::Pos2> {
                let p = geo::lla_to_pos(lat.to_radians(), lon.to_radians(), 0.0);
                if p.normalize().dot((cam - p).normalize()) < 0.05 {
                    return None;
                }
                let c = vp * (p - cam).as_vec3().extend(1.0);
                if c.w <= 0.0 {
                    return None;
                }
                let (x, y) = ((c.x / c.w * 0.5 + 0.5) * w, (0.5 - c.y / c.w * 0.5) * h);
                (x > -50.0 && y > -50.0 && x < w + 50.0 && y < h + 50.0).then(|| egui::pos2(x / ppp, y / ppp))
            };
            let painter = ctx.layer_painter(egui::LayerId::background());
            let label = |p: egui::Pos2, text: &str, size: f32, col: Color32| {
                painter.text(p + egui::vec2(1.0, 1.0), Align2::LEFT_CENTER, text, FontId::proportional(size), Color32::from_black_alpha(200));
                painter.text(p, Align2::LEFT_CENTER, text, FontId::proportional(size), col);
            };
            if alt > 25_000.0 {
                for (name, lat, lon, tier) in menu::CITIES {
                    if *tier > 1 && alt > 7.0e6 {
                        continue;
                    }
                    if let Some(p) = project(*lat, *lon) {
                        painter.circle_filled(p, 2.5, Color32::from_rgba_unmultiplied(255, 255, 255, 220));
                        label(p + egui::vec2(7.0, 0.0), tr(name), if *tier == 1 { 14.0 } else { 12.5 }, Color32::from_rgba_unmultiplied(255, 255, 255, 225));
                    }
                }
            }
            // Airports, once zoomed in far enough to tell them apart. Each one can be clicked.
            self.menu.markers.clear();
            if alt < 450_000.0 {
                let db = airports::Db::get();
                let mut n = 0;
                for (i, _) in db.near(self.menu.lat, self.menu.lon, (alt * 1.6).min(600_000.0)) {
                    let a = &db.apts[i as usize];
                    if a.kind < 1 && alt > 60_000.0 || a.kind < 2 && alt > 200_000.0 || !a.any_paved() || n >= 80 {
                        continue;
                    }
                    if let Some(p) = project(a.lat, a.lon) {
                        n += 1;
                        self.menu.markers.push((i, p.x * ppp, p.y * ppp));
                        let chosen = self.menu.airport == Some(i);
                        let hover = ctx.pointer_hover_pos().is_some_and(|h| (h - p).length() < 14.0 / ppp) || aim.is_some_and(|c| (c - p).length() < 28.0);
                        let col = if chosen || hover { Color32::from_rgb(255, 196, 64) } else if a.kind == 2 { Color32::from_rgb(120, 200, 255) } else { Color32::from_rgb(170, 200, 220) };
                        if hover {
                            painter.circle_stroke(p, 8.0, egui::Stroke::new(1.5, col));
                            label(p + egui::vec2(6.0, 14.0), &a.name.chars().take(40).collect::<String>(), 11.0, col);
                        }
                        painter.rect_filled(egui::Rect::from_center_size(p, egui::vec2(5.0, 5.0)), 1.0, col);
                        label(p + egui::vec2(6.0, 0.0), &a.ident, if a.kind == 2 { 12.0 } else { 10.5 }, col);
                    }
                }
            }
            if let Some((lat, lon)) = self.menu.pick {
                if let Some(p) = project(lat, lon) {
                    let accent = Color32::from_rgb(255, 196, 64);
                    let pulse = 11.0 + 3.0 * (self.real * 3.0).sin() as f32;
                    painter.circle_stroke(p, pulse, egui::Stroke::new(2.0, accent));
                    painter.circle_filled(p, 3.5, accent);
                    label(p + egui::vec2(18.0, 0.0), t!("Start here", "Départ ici"), 15.0, accent);
                }
            }
            if let Some(c) = aim {
                let st = egui::Stroke::new(1.5, Color32::from_white_alpha(200));
                let sh = egui::Stroke::new(3.0, Color32::from_black_alpha(110));
                for s in [sh, st] {
                    painter.circle_stroke(c, 9.0, s);
                    for d in [egui::vec2(1.0, 0.0), egui::vec2(-1.0, 0.0), egui::vec2(0.0, 1.0), egui::vec2(0.0, -1.0)] {
                        painter.line_segment([c + d * 13.0, c + d * 20.0], s);
                    }
                }
            }
            use menu::{ACCENT, DIM, TEXT};
            const W: f32 = 340.0;
            const M: f32 = 22.0;
            let screen = ctx.content_rect();
            // The panels start below the title (logo.rs: capitals 9..36 points down).
            const TOP: f32 = 46.0;
            // On a short window the aircraft picture shrinks and its blurb goes,
            // so the start-location panel below still fits whole.
            let air_room = screen.height() - TOP - (14.0 + 290.0 + M);
            let show_blurb = air_room >= 58.0 + 196.0 + 52.0;
            let pic_h = (air_room - 58.0 - if show_blurb { 52.0 } else { 0.0 }).clamp(100.0, 196.0);
            let n_air = aircraft::ALL.len();
            let mut choose_air: Option<usize> = None;

            // Top left: the aircraft window. Its name in the title bar with
            // arrows to the others, and a picture of it below.
            let air = egui::Area::new(egui::Id::unique("menu_air")).anchor(Align2::LEFT_TOP, [M, TOP]).show(ctx, |ui| {
                ui.set_width(W);
                ui.spacing_mut().item_spacing.y = 0.0;
                let spec = aircraft::ALL[self.menu.aircraft.min(n_air - 1)];
                let top = egui::CornerRadius { nw: 12, ne: 12, sw: 0, se: 0 };
                let bottom = egui::CornerRadius { nw: 0, ne: 0, sw: 12, se: 12 };
                egui::Frame::new().fill(menu::PANEL).corner_radius(top).inner_margin(egui::Margin { left: 10, right: 10, top: 12, bottom: 10 }).show(ui, |ui| {
                    ui.set_width(W - 20.0);
                    ui.horizontal(|ui| {
                        let arrow = |ui: &mut egui::Ui, turn: f32| {
                            let r = ui.add(egui::Button::new("").min_size(egui::vec2(34.0, 34.0)).fill(Color32::from_white_alpha(8)).corner_radius(8.0));
                            menu::chevron(ui.painter(), r.rect.center(), 6.0, turn, if r.hovered() || r.has_focus() { Color32::WHITE } else { DIM });
                            r
                        };
                        if arrow(ui, 2.0).clicked() {
                            choose_air = Some((self.menu.aircraft + n_air - 1) % n_air);
                        }
                        let name_w = ui.available_width() - 34.0 - ui.spacing().item_spacing.x;
                        ui.allocate_ui_with_layout(egui::vec2(name_w, 34.0), egui::Layout::top_down(egui::Align::Center), |ui| {
                            ui.label(RichText::new(t!("AIRCRAFT", "AVION")).size(10.5).strong().color(DIM).extra_letter_spacing(1.5));
                            ui.label(RichText::new(spec.name).size(18.0).strong().color(Color32::WHITE));
                        });
                        if arrow(ui, 0.0).clicked() {
                            choose_air = Some((self.menu.aircraft + 1) % n_air);
                        }
                    });
                });
                let k = self.menu.aircraft.min(n_air - 1);
                let tex = self.stills[k].get_or_insert_with(|| {
                    let img = image::load_from_memory(spec.still).expect("aircraft still").to_rgba8();
                    let size = [img.width() as usize, img.height() as usize];
                    ctx.load_texture(format!("aircraft {k}"), egui::ColorImage::from_rgba_unmultiplied(size, img.as_raw()), egui::TextureOptions::LINEAR)
                });
                let (pic, _) = ui.allocate_exact_size(egui::vec2(W, pic_h), egui::Sense::hover());
                ui.painter().rect_filled(pic, 0.0, menu::PANEL);
                egui::Image::new(&*tex).paint_at(ui, egui::Rect::from_center_size(pic.center(), egui::vec2(W * pic_h / 196.0, pic_h)));
                egui::Frame::new().fill(menu::PANEL).corner_radius(bottom).inner_margin(egui::Margin { left: 16, right: 16, top: 10, bottom: 12 }).show(ui, |ui| {
                    ui.set_width(W - 32.0);
                    if show_blurb {
                        ui.label(RichText::new(tr(spec.blurb)).size(12.0).color(DIM));
                    }
                });
            });
            let r = air.response.rect;
            ctx.layer_painter(air.response.layer_id).rect_stroke(r, 12.0, egui::Stroke::new(1.0, Color32::from_white_alpha(22)), egui::StrokeKind::Inside);

            // Below it: where to start. Search, places and airports fold away;
            // the chosen spot, take-off, settings and quit stay in view.
            let top = r.bottom() + 14.0;
            let foot_h = 42.0 + 6.0 + 34.0 + 6.0;
            let body_h = (screen.bottom() - M - top - 40.0 - 28.0 - foot_h).max(120.0);
            egui::Area::new(egui::Id::unique("menu")).anchor(Align2::LEFT_TOP, [M, top]).show(ctx, |ui| {
                ui.set_max_height(screen.bottom() - M - top);
                menu::panel().inner_margin(egui::Margin::symmetric(16, 14)).show(ui, |ui| {
                    ui.set_width(W - 32.0);
                    menu::window_title(ui, t!("START LOCATION", "LIEU DE DÉPART"));
                    egui::ScrollArea::vertical().max_height(body_h).auto_shrink([false, false]).show(ui, |ui| {
                        ui.set_width(ui.available_width());
                        let summary = if self.menu.search.trim().is_empty() { String::new() } else { self.menu.search.chars().take(18).collect() };
                        menu::section(ui, "search", t!("Search", "Recherche"), &summary, true, |ui| {
                            ui.horizontal(|ui| {
                                let resp = ui.add(egui::TextEdit::singleline(&mut self.menu.search).desired_width(ui.available_width() - 56.0).hint_text(t!("City, address, or airport (SFO, EGLL)", "Ville, adresse ou aéroport (SFO, EGLL)")).margin(egui::vec2(8.0, 6.0)));
                                let enter = resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
                                if (ui.add(egui::Button::new(RichText::new(t!("Go", "Aller")).size(14.0)).min_size(egui::vec2(48.0, 28.0))).clicked() || enter) && !self.menu.search.trim().is_empty() {
                                    go = Some(self.menu.search.trim().to_string());
                                }
                            });
                            // Airports matching what is typed.
                            let hits = airports::Db::get().search(&self.menu.search, 5);
                            if !hits.is_empty() {
                                ui.add_space(2.0);
                                ui.horizontal_wrapped(|ui| {
                                    ui.spacing_mut().item_spacing = egui::vec2(5.0, 5.0);
                                    for i in hits {
                                        let a = &airports::Db::get().apts[i as usize];
                                        let label = format!("✈ {}  {}", a.ident, a.name.chars().take(28).collect::<String>());
                                        if ui.add(egui::Button::new(RichText::new(label).size(12.0)).selected(self.menu.airport == Some(i))).clicked() {
                                            airport = Some(i);
                                        }
                                    }
                                });
                            }
                            ui.add_space(2.0);
                            let tip = if padui::active() { t!("Or turn the globe (left stick, triggers to zoom) and press X at the crosshair.", "Ou tournez le globe (stick gauche, gâchettes pour zoomer) et appuyez sur X au viseur.") } else { t!("Or drag the globe, scroll to zoom, and click where you want to start.", "Ou faites glisser le globe, zoomez à la molette et cliquez là où vous voulez partir.") };
                            ui.label(RichText::new(tip).size(11.5).color(DIM));
                        });
                        let place = menu::QUICK.iter().find(|q| self.menu.pick == Some((q.1, q.2))).map(|q| tr(q.0)).unwrap_or("");
                        menu::section(ui, "places", t!("Places", "Lieux"), place, false, |ui| {
                            ui.horizontal_wrapped(|ui| {
                                ui.spacing_mut().item_spacing = egui::vec2(5.0, 5.0);
                                for (name, lat, lon) in menu::QUICK {
                                    let sel = self.menu.pick == Some((*lat, *lon));
                                    if ui.add(egui::Button::new(RichText::new(tr(name)).size(12.5)).selected(sel)).clicked() {
                                        jump = Some((*lat, *lon, name));
                                    }
                                }
                            });
                        });
                        let db = airports::Db::get();
                        let apt = self.menu.airport.map(|i| db.apts[i as usize].ident.clone()).unwrap_or_default();
                        menu::section(ui, "airports", t!("Airports", "Aéroports"), &apt, false, |ui| {
                            ui.horizontal_wrapped(|ui| {
                                ui.spacing_mut().item_spacing = egui::vec2(5.0, 5.0);
                                for (code, name) in menu::AIRPORTS {
                                    if let Some(i) = db.find(code) {
                                        if ui.add(egui::Button::new(RichText::new(*code).size(12.5)).selected(self.menu.airport == Some(i))).on_hover_text(tr(name)).clicked() {
                                            airport = Some(i);
                                        }
                                    }
                                }
                            });
                            ui.label(RichText::new(if padui::active() { t!("Or put an airport square on the globe in the crosshair and press X.", "Ou placez un carré d'aéroport du globe dans le viseur et appuyez sur X.") } else { t!("Or click an airport square on the globe; double-click to go right away.", "Ou cliquez sur un carré d'aéroport du globe ; double-clic pour y aller directement.") }).size(11.5).color(DIM));
                        });
                        ui.add_space(6.0);
                        menu::heading(ui, t!("START POINT", "POINT DE DÉPART"));
                        egui::Frame::new().fill(Color32::from_white_alpha(8)).corner_radius(8.0).inner_margin(10.0).show(ui, |ui| {
                            ui.set_width(ui.available_width());
                            match self.menu.pick {
                                Some((lat, lon)) => {
                                    let name = if self.menu.pick_name.is_empty() { t!("Unnamed spot", "Lieu sans nom").to_string() } else { self.menu.pick_name.chars().take(90).collect() };
                                    ui.label(RichText::new(name).size(14.0).color(Color32::WHITE));
                                    ui.label(RichText::new(t!("{}   ·   {} local", "{}   ·   {} heure locale", menu::latlon(lat, lon), menu::hhmm(self.menu.hour))).size(12.0).color(DIM));
                                    ui.add_space(4.0);
                                    match self.menu.airport {
                                        Some(i) => {
                                            let a = &db.apts[i as usize];
                                            ui.horizontal_wrapped(|ui| {
                                                ui.spacing_mut().item_spacing = egui::vec2(4.0, 4.0);
                                                ui.selectable_value(&mut self.menu.start, menu::Start::Runway, t!("On the runway", "Sur la piste"));
                                                ui.selectable_value(&mut self.menu.start, menu::Start::Approach, t!("Final approach", "Approche finale"));
                                                ui.selectable_value(&mut self.menu.start, menu::Start::Overhead, t!("In the air above", "En vol au-dessus"));
                                            });
                                            if self.menu.start != menu::Start::Overhead && !a.rwys.is_empty() {
                                                ui.horizontal_wrapped(|ui| {
                                                    ui.spacing_mut().item_spacing = egui::vec2(4.0, 4.0);
                                                    ui.label(RichText::new(t!("Runway", "Piste")).size(12.5).color(TEXT));
                                                    ui.selectable_value(&mut self.menu.runway, None, RichText::new(t!("into the wind", "face au vent")).size(12.0));
                                                    for r in &a.rwys {
                                                        for id in &r.ident {
                                                            if !id.is_empty() {
                                                                ui.selectable_value(&mut self.menu.runway, Some(id.clone()), RichText::new(id).size(12.0).monospace()).on_hover_text(format!("{:.0} m × {:.0} m{}", r.len, r.width, if r.paved { "" } else { t!(", unpaved", ", non revêtue") }));
                                                            }
                                                        }
                                                    }
                                                });
                                            }
                                        }
                                        None => {
                                            // Offer the nearest airfield, for a start on the ground.
                                            let near = db.near(lat, lon, 60_000.0).into_iter().find(|(i, _)| db.apts[*i as usize].any_paved());
                                            if let Some((i, d)) = near {
                                                let a = &db.apts[i as usize];
                                                ui.horizontal(|ui| {
                                                    ui.label(RichText::new(t!("Nearest airport {} · {:.0} km", "Aéroport le plus proche : {} · {:.0} km", a.ident, d / 1000.0)).size(12.0).color(DIM));
                                                    if ui.button(RichText::new(t!("Start there", "Partir de là")).size(12.0)).clicked() {
                                                        airport = Some(i);
                                                    }
                                                });
                                            }
                                        }
                                    }
                                }
                                None => {
                                    ui.label(RichText::new(t!("Pick a place, or click the globe.", "Choisissez un lieu, ou cliquez sur le globe.")).size(13.5).color(DIM));
                                }
                            }
                        });
                        padui::follow_focus(ui);
                    });
                    ui.add_space(10.0);
                    let text = if self.menu.starting { t!("Preparing…", "Préparation…") } else { t!("TAKE OFF", "DÉCOLLER") };
                    let ready = self.menu.pick.is_some() && !self.menu.starting;
                    let take_off = menu::wide_button(ui, text, if ready { Hint::of("Enter", "Menu") } else { Hint::None }, true, ready);
                    if take_off.clicked() {
                        start = true;
                    }
                    if ready {
                        self.pad_home_focus(&take_off);
                    }
                    ui.add_space(6.0);
                    let mut home = None;
                    ui.columns(2, |c| {
                        let b = menu::wide_button(&mut c[0], t!("Settings", "Réglages"), Hint::of("", "View"), false, true);
                        if b.clicked() {
                            self.menu.show_controls = !self.menu.show_controls;
                        }
                        home = Some(b);
                        if menu::wide_button(&mut c[1], t!("Quit", "Quitter"), Hint::of("Esc", ""), false, true).clicked() {
                            self.quit = true;
                        }
                    });
                    if let Some(b) = home {
                        self.pad_home_focus(&b);
                    }
                    padui::follow_focus(ui);
                });
            });

            // Settings: map imagery and the controls, in a window over the globe.
            if self.menu.show_controls {
                egui::Area::new(egui::Id::unique("menu_keys")).anchor(Align2::CENTER_CENTER, [0.0, 20.0]).show(ctx, |ui| {
                    ui.set_max_height(screen.height() - 120.0);
                    menu::panel().show(ui, |ui| {
                        ui.set_width(720.0);
                        ui.horizontal(|ui| {
                            ui.label(RichText::new(t!("Settings", "Réglages")).size(22.0).strong().color(Color32::WHITE));
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                if ui.add(egui::Button::new(RichText::new(t!("Close", "Fermer")).size(13.0)).min_size(egui::vec2(64.0, 28.0))).clicked() {
                                    self.menu.show_controls = false;
                                }
                            });
                        });
                        ui.add_space(6.0);
                        egui::ScrollArea::vertical().max_height(screen.height() - 220.0).min_scrolled_height(screen.height() - 220.0).show(ui, |ui| {
                            menu::section(ui, "language", t!("Language", "Langue"), i18n::lang().name(), true, |ui| {
                                self.language_picker(ui);
                            });
                            let on = if self.src.google.enabled() { "Google" } else { t!("Default", "Par défaut") };
                            menu::section(ui, "imagery", t!("Map imagery", "Imagerie cartographique"), on, false, |ui| {
                                ui.spacing_mut().slider_width = 130.0;
                                self.google_settings(ui);
                            });
                            let q = self.gfx.preset_of(self.taa, self.r.integrated).map_or(t!("Custom", "Personnalisé"), preset_name);
                            menu::section(ui, "graphics", t!("Graphics", "Graphismes"), q, false, |ui| {
                                self.graphics_settings(ui, "menu_gfx");
                            });
                            menu::section(ui, "controls", t!("Controls", "Commandes"), if padui::active() { t!("Gamepad", "Manette") } else { t!("Keyboard", "Clavier") }, true, |ui| {
                                let t = if padui::active() { t!("In flight, Y + View shows the flight panel (instruments, places, weather) and Menu pauses: settings, the globe, this list.", "En vol, Y + View affiche le panneau de vol (instruments, destinations, météo) et Menu met en pause : réglages, globe, cette liste.") } else { t!("In flight, Tab shows the flight panel (instruments, places, weather) and Esc pauses: settings, the globe, this list.", "En vol, Tab affiche le panneau de vol (instruments, destinations, météo) et Échap met en pause : réglages, globe, cette liste.") };
                                ui.label(RichText::new(t).size(12.5).color(DIM));
                                ui.add_space(6.0);
                                menu::controls_list(ui);
                            });
                            padui::follow_focus(ui);
                        });
                    });
                });
            }

            // On the right: how the flight starts, the weather, the time.
            let right_h = (screen.height() - M - 64.0 - 28.0).max(200.0);
            egui::Area::new(egui::Id::unique("menu_setup")).anchor(Align2::RIGHT_TOP, [-M, TOP]).show(ctx, |ui| {
                // (an area lays out in last frame's size: let it grow as sections open)
                ui.set_max_height(right_h);
                menu::panel().inner_margin(egui::Margin::symmetric(16, 14)).show(ui, |ui| {
                    ui.set_width(320.0);
                    menu::window_title(ui, "CONDITIONS");
                    egui::ScrollArea::vertical().max_height(right_h - 40.0).auto_shrink([false, true]).show(ui, |ui| {
                        ui.set_width(ui.available_width());
                        ui.spacing_mut().slider_width = 116.0;
                        let ground = self.menu.airport.is_some() && self.menu.start != menu::Start::Overhead;
                        let flight = if ground {
                            (if self.menu.start == menu::Start::Runway { t!("On the runway", "Sur la piste") } else { t!("Final approach", "Approche finale") }).to_string()
                        } else {
                            format!("{:.0} ft · {:03.0}°", (self.menu.start_agl * 3.28084 / 50.0).round() * 50.0, self.heading.to_degrees().rem_euclid(360.0))
                        };
                        menu::section(ui, "flight", t!("Flight", "Vol"), &flight, false, |ui| {
                            if ground {
                                ui.label(RichText::new(if self.menu.start == menu::Start::Runway { t!("Starting on the runway, ready for take-off.", "Départ sur la piste, prêt au décollage.") } else { t!("Starting on a 3° final approach, gear and flaps down.", "Départ en approche finale à 3°, train et volets sortis.") }).size(12.0).color(DIM));
                            } else {
                                ui.label(RichText::new(t!("Starting in the air:", "Départ en vol :")).size(12.0).color(DIM));
                            }
                            ui.add_enabled_ui(!ground, |ui| {
                                egui::Grid::new("start_air").num_columns(2).min_col_width(92.0).spacing([10.0, 5.0]).show(ui, |ui| {
                                    ui.label(t!("Height", "Hauteur"));
                                    let mut ft = self.menu.start_agl * 3.28084;
                                    if ui.add(egui::Slider::new(&mut ft, 100.0..=15000.0).logarithmic(true).step_by(50.0).suffix(" ft").custom_formatter(|v, _| format!("{:.0}", (v / 50.0).round() * 50.0))).changed() {
                                        self.menu.start_agl = ft / 3.28084;
                                    }
                                    ui.end_row();
                                    ui.label("");
                                    ui.label(RichText::new(t!("above the ground", "au-dessus du sol")).size(11.5).weak());
                                    ui.end_row();
                                    ui.label(t!("Heading", "Cap"));
                                    let mut hd = self.heading.to_degrees().rem_euclid(360.0);
                                    if ui.add(egui::Slider::new(&mut hd, 0.0..=359.0).step_by(1.0).custom_formatter(|v, _| format!("{v:03.0}°"))).changed() {
                                        self.heading = hd.to_radians();
                                    }
                                    ui.end_row();
                                });
                            });
                        });
                        let wx = self.wx.summary();
                        menu::section(ui, "weather", t!("Weather", "Météo"), &wx, false, |ui| {
                            self.wx.start_weather(ui);
                        });
                        let when = format!("{} · {}", menu::hhmm(self.menu.hour), weather::date_label(self.doy));
                        menu::section(ui, "time", t!("Time", "Heure"), &when, false, |ui| {
                            weather::Weather::start_time(ui, &mut self.menu.hour, &mut self.doy, &mut self.wx.real_time);
                        });
                        padui::follow_focus(ui);
                    });
                });
            });
            if let Some(k) = choose_air {
                self.menu.aircraft = k;
            }
            self.credits(ctx);
            padui::focus_ring(ctx, ACCENT);
        });
        self.egui_state.handle_platform_output(&self.window, out.platform_output);
        if let Some((lat, lon, name)) = jump {
            self.menu.look_at(lat, lon, 30_000.0);
            self.menu.pick = Some((lat, lon));
            self.menu.pick_name = tr(name).to_string();
            self.menu.airport = None;
            self.menu.runway = None;
        }
        if let Some(i) = airport {
            self.choose_airport(i);
        }
        let want = aircraft::ALL[self.menu.aircraft.min(aircraft::ALL.len() - 1)];
        if !std::ptr::eq(want, self.spec) {
            self.want_aircraft(want);
        }
        if let Some(i) = go.as_deref().and_then(|q| airports::Db::get().find(q)) {
            self.choose_airport(i);
            go = None;
        }
        if let Some(q) = go {
            self.menu.pick_name = t!("Searching for {}…", "Recherche de {}…", q);
            let (tx, rx) = std::sync::mpsc::channel();
            let src = self.src.clone();
            std::thread::spawn(move || {
                let _ = tx.send(geocode(&src.fetch, &q).map(|(la, lo, name)| (la, lo, 0.0, name)));
            });
            self.goto_rx = Some(rx);
        }
        if start {
            self.begin();
        }
        let jobs = self.egui_ctx.tessellate(out.shapes, out.pixels_per_point);
        (jobs, out.textures_delta, out.pixels_per_point)
    }

    /// The data credits for the corner of the screen (the Google Maps logo
    /// goes before them while Google's data is shown, see `credits`).
    fn attribution(&self) -> String {
        let g = &self.src.google;
        let p = self.world.photo();
        let photo = if p.zone_r > 0.0 && p.stats.0 > 0 { format!("3D: {}  ·  ", if p.credits.is_empty() { "Google" } else { &p.credits }) } else { String::new() };
        let live = if self.live.enabled() && !self.live.planes.is_empty() { format!("  ·  {}", t!("Traffic: adsb.lol (ODbL)", "Trafic : adsb.lol (ODbL)")) } else { String::new() };
        photo + &if g.enabled() {
            let c = g.attribution.lock();
            let c = if c.is_empty() { t!("Imagery © Google", "Imagerie © Google").to_string() } else { c.clone() };
            format!("{c}  ·  {}", attribution_rest())
        } else {
            format!("{}  ·  {}", attribution_default(), attribution_rest())
        } + &live
    }

    /// The bottom-right credits: Google's logo (its official outlined
    /// artwork, unaltered, 22 px with its outline, clear space round it) while
    /// their imagery or 3D tiles are on screen, then the data credits.
    fn credits(&mut self, ctx: &egui::Context) -> egui::Rect {
        use egui::{Align2, Color32, RichText};
        let logo = self.google_shown().then(|| {
            self.gm_logo
                .get_or_insert_with(|| {
                    let img = image::load_from_memory(include_bytes!("../assets/google_maps_logo.png")).expect("logo").to_rgba8();
                    let size = [img.width() as usize, img.height() as usize];
                    ctx.load_texture("google maps logo", egui::ColorImage::from_rgba_unmultiplied(size, img.as_raw()), egui::TextureOptions::LINEAR)
                })
                .clone()
        });
        let text = self.attribution();
        // The logo on a line of its own at the right, above the credits (which
        // can run long), so nothing else on screen comes near it.
        // (returns where they are drawn, for the minimap to sit above)
        egui::Area::new(egui::Id::unique("attr")).anchor(Align2::RIGHT_BOTTOM, [-10.0, -5.0]).show(ctx, |ui| {
            ui.with_layout(egui::Layout::bottom_up(egui::Align::Max), |ui| {
                let mut r = ui.label(RichText::new(text).size(10.0).color(Color32::from_rgba_unmultiplied(255, 255, 255, 110))).rect;
                if let Some(t) = logo {
                    let s = t.size_vec2();
                    ui.add_space(4.0);
                    r = r.union(ui.add(egui::Image::new(&t).fit_to_exact_size(egui::vec2(s.x * 22.0 / s.y, 22.0)).alt_text("Google Maps")).rect);
                }
                r
            })
            .inner
        })
        .inner
    }

    /// Whether Google's data is on screen (its credit is shown, HUD or not).
    fn google_shown(&self) -> bool {
        self.src.google.enabled() || self.world.photo().stats.0 > 0
    }

    /// Map imagery: the Google Maps switch, its API key and what it has used.
    fn google_settings(&mut self, ui: &mut egui::Ui) {
        use egui::{Color32, RichText};
        use menu::{DIM, TEXT};
        let g = self.src.google.clone();
        menu::heading(ui, t!("MAP IMAGERY", "IMAGERIE CARTOGRAPHIQUE"));
        let mut on = g.enabled();
        let has_key = !self.gkey.trim().is_empty();
        if ui.add_enabled(has_key, egui::Checkbox::new(&mut on, t!("Google Maps satellite imagery", "Imagerie satellite Google Maps"))).on_disabled_hover_text(t!("Enter a Google Maps Platform API key first", "Saisissez d'abord une clé API Google Maps Platform")).changed() {
            g.set_key(&self.gkey);
            g.set_enabled(on);
        }
        ui.label(RichText::new(if on { t!("Terrain and roofs take Google's tiles (Map Tiles API); what has loaded already is redone. Tiles not yet on disk are billed to your key.", "Le terrain et les toits utilisent les tuiles de Google (Map Tiles API) ; ce qui est déjà chargé est refait. Les tuiles absentes du disque sont facturées sur votre clé.") } else { t!("Off: the free default imagery. On: Google's Map Tiles API with your own key, billed per tile request.", "Désactivé : l'imagerie gratuite par défaut. Activé : la Map Tiles API de Google avec votre propre clé, facturée à la tuile.") }).size(11.5).color(DIM));
        let mut photo = g.photoreal();
        if ui.add_enabled(has_key, egui::Checkbox::new(&mut photo, t!("Photorealistic 3D cities (beta)", "Villes 3D photoréalistes (bêta)"))).on_disabled_hover_text(t!("Enter a Google Maps Platform API key first", "Saisissez d'abord une clé API Google Maps Platform")).changed() {
            g.set_key(&self.gkey);
            g.set_photoreal(photo);
        }
        let detail = g.detail();
        let zone_km = tiles3d::Params::of(detail).zone_r / 1000.0;
        ui.label(RichText::new(if photo { t!("Google's photogrammetry meshes replace the ground, buildings and trees within about {zone_km:.0} km of the aircraft; at night they light up with our street lamps and windows. Each flight opens one billed 3D session (a root request, good for about 3 hours); the tiles under it are kept on disk only for the 4 hours Google allows.", "Les maillages photogrammétriques de Google remplacent le sol, les bâtiments et les arbres dans un rayon d'environ {zone_km:.0} km autour de l'avion ; la nuit, nos lampadaires et fenêtres les éclairent. Chaque vol ouvre une session 3D facturée (une requête racine, valable environ 3 heures) ; les tuiles sont gardées sur le disque seulement pendant les 4 heures autorisées par Google.", zone_km = zone_km) } else { t!("Off: our own 3D world everywhere. On: Google's Photorealistic 3D Tiles near the aircraft, one billed session per flight.", "Désactivé : notre propre monde 3D partout. Activé : les Photorealistic 3D Tiles de Google près de l'avion, une session facturée par vol.").to_string() }).size(11.5).color(DIM));
        ui.add_enabled_ui(photo, |ui| {
            ui.horizontal(|ui| {
                ui.label(t!("3D detail", "Détail 3D"));
                let mut d = detail;
                for c in gmaps::Detail::ALL {
                    let name = match c {
                        gmaps::Detail::Low => t!("Low", "Faible"),
                        gmaps::Detail::Medium => t!("Medium", "Moyen"),
                        gmaps::Detail::High => t!("High", "Élevé"),
                        gmaps::Detail::Ultra => "Ultra",
                    };
                    ui.selectable_value(&mut d, c, name);
                }
                if d != detail {
                    g.set_detail(d);
                }
            });
            ui.label(RichText::new(t!("{} One billed session per flight at every level; the first minute over a city takes up to a third more.", "{} Une session facturée par vol à chaque niveau ; la première minute au-dessus d'une ville consomme jusqu'à un tiers de plus.", detail_hint(detail))).size(11.5).color(DIM));
        });
        ui.horizontal(|ui| {
            ui.label(t!("API key", "Clé API"));
            let r = ui.add(egui::TextEdit::singleline(&mut self.gkey).password(!self.gkey_show).hint_text(t!("Google Maps Platform key", "Clé Google Maps Platform")).desired_width(ui.available_width() - 52.0));
            if r.lost_focus() || (r.changed() && self.gkey.trim().is_empty()) {
                g.set_key(&self.gkey);
            }
            if ui.small_button(if self.gkey_show { t!("hide", "masquer") } else { t!("show", "afficher") }).clicked() {
                self.gkey_show = !self.gkey_show;
            }
        });
        ui.horizontal_wrapped(|ui| {
            ui.hyperlink_to(RichText::new(t!("How to get a key", "Obtenir une clé")).size(12.0), "https://developers.google.com/maps/documentation/tile/get-api-key");
            ui.label(RichText::new("·").size(12.0).color(DIM));
            ui.hyperlink_to(RichText::new(t!("Cloud console: credentials", "Console Cloud : identifiants")).size(12.0), "https://console.cloud.google.com/google/maps-apis/credentials");
        });
        let status = g.status.lock().clone();
        if !status.is_empty() {
            ui.label(RichText::new(status.chars().take(160).collect::<String>()).size(11.5).color(Color32::from_rgb(255, 150, 70)));
        }
        let u = g.usage.lock().clone();
        egui::Grid::new(ui.scope_id().with("gusage")).num_columns(2).spacing([12.0, 3.0]).show(ui, |ui| {
            let row = |ui: &mut egui::Ui, k: &str, v: String| {
                ui.label(RichText::new(k).size(12.0).color(DIM));
                ui.label(RichText::new(v).size(12.0).color(TEXT));
                ui.end_row();
            };
            let req = |n: u64| i18n::plural(n, "request", "requests", "requête", "requêtes");
            row(ui, t!("This session", "Cette session"), format!("{} · {}", req(u.session_requests as u64), bytes(u.session_bytes)));
            row(ui, t!("   imagery", "   imagerie"), t!("{} tiles downloaded · {} from disk", "{} tuiles téléchargées · {} depuis le disque", u.session_tiles, u.session_cached));
            if u.session_roots > 0 || u.total_roots > 0 {
                let ses = i18n::plural(u.session_roots as u64, "session", "sessions", "session", "sessions");
                row(ui, "   3D", t!("{} (billed) · {} tiles · {} from disk", "{} (facturées) · {} tuiles · {} depuis le disque", ses, u.session_3d, u.session_3d_cached));
            }
            if u.session_errors > 0 {
                row(ui, t!("   failed", "   échecs"), format!("{}", u.session_errors));
            }
            row(ui, &t!("Month {}", "Mois {}", u.month), format!("{} · {}", req(u.month_requests as u64), bytes(u.month_bytes)));
            row(ui, t!("All time", "Depuis le début"), format!("{} · {}", req(u.total_requests as u64), bytes(u.total_bytes)));
            if u.total_roots > 0 {
                row(ui, t!("3D sessions", "Sessions 3D"), t!("{} this month · {} all time", "{} ce mois-ci · {} au total", u.month_roots, u.total_roots));
            }
        });
        ui.label(RichText::new(t!("Counts only calls that reach Google; tiles already on disk are free. Check the Cloud console for billing.", "Ne compte que les appels qui atteignent Google ; les tuiles déjà sur le disque sont gratuites. Consultez la console Cloud pour la facturation.")).size(11.0).color(DIM));
    }

    /// The pause menu (Esc), drawn over the dimmed, stopped game.
    fn pause_menu(&mut self, ctx: &egui::Context, act: &mut Option<PauseAct>, lat: f64, lon: f64, local: f64) {
        use egui::{Align2, Color32, RichText};
        use menu::{Page, ACCENT, DIM, TEXT};
        // The backdrop dims the scene and keeps the mouse from the camera behind it.
        let screen = ctx.content_rect();
        egui::Area::new(egui::Id::unique("pause_dim")).order(egui::Order::Middle).fixed_pos(screen.min).show(ctx, |ui| {
            ui.allocate_response(screen.size(), egui::Sense::click());
            ui.painter().rect_filled(screen, 0.0, Color32::from_black_alpha(130));
        });
        if self.pause_page.is_info() {
            self.pause_page = Page::Game;
        }
        let w = match self.pause_page {
            Page::Controls => 640.0,
            Page::Graphics => 560.0,
            Page::Route => 520.0,
            _ => 420.0,
        };
        egui::Area::new(egui::Id::unique("pause")).order(egui::Order::Foreground).anchor(Align2::CENTER_CENTER, [0.0, 0.0]).show(ctx, |ui| {
            menu::panel().show(ui, |ui| {
                ui.set_width(w);
                ui.horizontal(|ui| {
                    ui.vertical(|ui| {
                        ui.label(RichText::new(t!("PAUSED", "PAUSE")).size(28.0).strong().color(Color32::WHITE));
                        ui.label(RichText::new("SAMIAM'S FLIGHT SIM").size(11.5).strong().color(ACCENT));
                    });
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.vertical(|ui| {
                            ui.with_layout(egui::Layout::top_down(egui::Align::Max), |ui| {
                                ui.label(RichText::new(menu::latlon(lat, lon)).size(13.0).color(TEXT));
                                ui.label(RichText::new(t!("{} local", "{} heure locale", menu::hhmm(local))).size(12.0).color(DIM));
                            });
                        });
                    });
                });
                ui.add_space(10.0);
                let pad = padui::active();
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = 4.0;
                    if pad {
                        padui::glyphs(ui, "LB", 20.0);
                    }
                    for p in Page::PAUSE {
                        let b = ui.add(egui::Button::selectable(self.pause_page == p, RichText::new(p.label()).size(15.0)).min_size(egui::vec2(0.0, 30.0)));
                        if b.clicked() {
                            self.pause_page = p;
                        }
                        if p == self.pause_page && p != Page::Game {
                            self.pad_home_focus(&b);
                        }
                    }
                    if pad {
                        padui::glyphs(ui, "RB", 20.0);
                    }
                });
                ui.separator();
                ui.add_space(4.0);
                match self.pause_page {
                    Page::Settings => egui::ScrollArea::vertical().max_height((screen.height() - 230.0).max(200.0)).show(ui, |ui| {
                        egui::Grid::new("pause_set").num_columns(2).spacing([16.0, 8.0]).show(ui, |ui| {
                            ui.spacing_mut().slider_width = 200.0;
                            menu::heading(ui, t!("LANGUAGE", "LANGUE"));
                            ui.end_row();
                            ui.label(t!("Language", "Langue"));
                            self.language_picker(ui);
                            ui.end_row();
                            menu::heading(ui, t!("SOUND", "SON"));
                            ui.end_row();
                            ui.label(t!("Sound", "Son"));
                            let t = if self.sound_on { t!("on", "activé") } else { t!("off", "coupé") };
                            ui.checkbox(&mut self.sound_on, t);
                            ui.end_row();
                            ui.label("Volume");
                            ui.add_enabled(self.sound_on, egui::Slider::new(&mut self.volume, 0.0..=1.0).custom_formatter(|v, _| format!("{:.0}%", v * 100.0)));
                            ui.end_row();
                            menu::heading(ui, t!("VIEW", "VUE"));
                            ui.end_row();
                            ui.label(t!("Camera", "Caméra"));
                            ui.horizontal(|ui| {
                                ui.selectable_value(&mut self.cam_mode, 0, t!("Chase", "Extérieure"));
                                ui.selectable_value(&mut self.cam_mode, 1, "Cockpit");
                            });
                            ui.end_row();
                            ui.label(t!("Cockpit field of view", "Champ de vision cockpit"));
                            ui.add(egui::Slider::new(&mut self.cockpit_fov, 35.0..=85.0).custom_formatter(|v, _| format!("{v:.0}° vertical")));
                            ui.end_row();
                            ui.label(t!("Mouse look speed", "Vitesse du regard (souris)"));
                            ui.add(egui::Slider::new(&mut self.look_sens, 0.25..=3.0).logarithmic(true).custom_formatter(|v, _| format!("{v:.2}×")));
                            ui.end_row();
                            ui.label("HUD");
                            let t = if self.show_hud { t!("shown", "affiché") } else { t!("hidden", "masqué") };
                            ui.checkbox(&mut self.show_hud, t);
                            ui.end_row();
                            ui.label(t!("Minimap", "Mini-carte"));
                            let t = if self.map.mini_on { t!("shown (M: large map)", "affichée (M : grande carte)") } else { t!("hidden (M: large map)", "masquée (M : grande carte)") };
                            ui.checkbox(&mut self.map.mini_on, t);
                            ui.end_row();
                            menu::heading(ui, t!("TRAFFIC", "TRAFIC"));
                            ui.end_row();
                            ui.label(t!("Live air traffic", "Trafic aérien réel"));
                            let mut on = self.live.enabled();
                            if ui.checkbox(&mut on, t!("real aircraft, from adsb.lol", "vrais avions, via adsb.lol")).changed() {
                                self.live.set_enabled(on);
                            }
                            ui.end_row();
                        });
                        ui.add_space(6.0);
                        self.google_settings(ui);
                        ui.add_space(10.0);
                        if pad {
                            ui.label(RichText::new(t!("In flight, View switches the camera.", "En vol, View change de caméra.")).size(12.0).color(DIM));
                        } else {
                            ui.label(RichText::new(t!("Shortcuts in flight: U sound, C camera, H HUD.", "Raccourcis en vol : U son, C caméra, H HUD.")).size(12.0).color(DIM));
                        }
                        padui::follow_focus(ui);
                    }).inner,
                    Page::Graphics => egui::ScrollArea::vertical().max_height((screen.height() - 230.0).max(200.0)).show(ui, |ui| {
                        self.graphics_settings(ui, "pause_gfx");
                        ui.add_space(10.0);
                        if !pad {
                            ui.label(RichText::new(t!("Shortcuts in flight: F2 anti-aliasing, F4 frame rate counter, F11 fullscreen.", "Raccourcis en vol : F2 anticrénelage, F4 compteur d'images, F11 plein écran.")).size(12.0).color(DIM));
                        }
                        padui::follow_focus(ui);
                    }).inner,
                    Page::Controls => menu::controls_list(ui),
                    Page::Route => egui::ScrollArea::vertical().max_height((screen.height() - 230.0).max(200.0)).show(ui, |ui| {
                        self.route_page(ui, act, lat, lon);
                        padui::follow_focus(ui);
                    }).inner,
                    _ => {
                        ui.add_space(4.0);
                        let resume = menu::wide_button(ui, t!("RESUME", "REPRENDRE"), Hint::of("Esc", "B"), true, true);
                        if resume.clicked() {
                            *act = Some(PauseAct::Resume);
                        }
                        self.pad_home_focus(&resume);
                        ui.add_space(4.0);
                        if menu::wide_button(ui, t!("Respawn in the air", "Réapparaître en vol"), Hint::of("R", ""), false, true).clicked() {
                            *act = Some(PauseAct::Respawn);
                        }
                        ui.add_space(4.0);
                        ui.horizontal(|ui| {
                            ui.label(RichText::new(t!("Sim speed", "Vitesse de simulation")).size(14.0).color(TEXT));
                            ui.add_space(8.0);
                            for k in [1u32, 2, 4, 8] {
                                if ui.add(egui::Button::selectable(self.sim_speed == k, RichText::new(format!("{k}×")).size(14.0)).min_size(egui::vec2(40.0, 26.0))).clicked() {
                                    self.set_sim_speed(k);
                                }
                            }
                            ui.label(RichText::new("[ / ]").size(12.0).color(DIM));
                        });
                        ui.add_space(4.0);
                        if menu::wide_button(ui, t!("Back to the globe", "Retour au globe"), Hint::Text(t!("choose a new start", "choisir un nouveau départ")), false, true).clicked() {
                            *act = Some(PauseAct::Globe);
                        }
                        if menu::wide_button(ui, t!("Quit to desktop", "Quitter vers le bureau"), Hint::None, false, true).clicked() {
                            *act = Some(PauseAct::Quit);
                        }
                        ui.add_space(8.0);
                        let t = if pad { t!("Places, airports and the weather are on the flight panel: Y + View, while you fly.", "Lieux, aéroports et météo sont sur le panneau de vol : Y + View, pendant le vol.") } else { t!("Places, airports and the weather are on the flight panel: Tab, while you fly.", "Lieux, aéroports et météo sont sur le panneau de vol : Tab, pendant le vol.") };
                        ui.label(RichText::new(t).size(12.0).color(DIM));
                    }
                }
                ui.add_space(6.0);
                ui.separator();
                if pad {
                    padui::prompts(ui, &[("Dpad", tr("move")), ("A", tr("press")), ("LB / RB", tr("page")), ("B", tr("resume"))], 20.0, 12.0, DIM);
                } else {
                    ui.label(RichText::new(t!("Esc to resume  ·  Menu or B on a gamepad", "Échap pour reprendre  ·  Menu ou B à la manette")).size(12.0).color(DIM));
                }
            });
        });
    }

    /// The pause menu's Route page: from, any waypoints, to; the distance
    /// and time it takes; start at the departure or fly it from here.
    fn route_page(&mut self, ui: &mut egui::Ui, act: &mut Option<PauseAct>, lat: f64, lon: f64) {
        use egui::RichText;
        use menu::{ACCENT, DIM, TEXT};
        let n = self.route.stops.len();
        let mut remove = None;
        let mut find = false;
        menu::heading(ui, t!("ROUTE", "ITINÉRAIRE"));
        ui.label(RichText::new(t!("Airport codes (KSFO, LHR) or place names.", "Codes d'aéroport (KSFO, LHR) ou noms de lieux.")).size(12.0).color(DIM));
        ui.add_space(4.0);
        egui::Grid::new("route_stops").num_columns(3).spacing([10.0, 6.0]).show(ui, |ui| {
            for i in 0..n {
                let label = if i == 0 { t!("From", "Départ").to_string() } else if i + 1 == n { t!("To", "Arrivée").to_string() } else { t!("Via {}", "Via {}", i) };
                ui.label(RichText::new(label).size(14.0).color(TEXT));
                let st = &mut self.route.stops[i];
                let r = ui.add_sized([150.0, 26.0], egui::TextEdit::singleline(&mut st.text).hint_text(if i == 0 { "KSFO" } else if i + 1 == n { "KOAK" } else { "" }));
                if r.changed() {
                    (st.wp, st.missing) = (None, false);
                }
                if r.lost_focus() && ui.input(|k| k.key_pressed(egui::Key::Enter)) {
                    find = true;
                }
                ui.horizontal(|ui| {
                    let (txt, col) = match (&st.wp, st.busy, st.missing) {
                        (Some(w), _, _) => (w.name.chars().take(34).collect::<String>(), ACCENT),
                        (None, true, _) => (t!("searching…", "recherche…").to_string(), DIM),
                        (None, _, true) => (t!("not found", "introuvable").to_string(), egui::Color32::from_rgb(255, 120, 80)),
                        _ => (String::new(), DIM),
                    };
                    ui.label(RichText::new(txt).size(12.0).color(col));
                    if i > 0 && i + 1 < n && ui.small_button("✕").clicked() {
                        remove = Some(i);
                    }
                });
                ui.end_row();
            }
        });
        if let Some(i) = remove {
            self.route.stops.remove(i);
            self.route.restart();
        }
        ui.add_space(6.0);
        ui.horizontal(|ui| {
            if ui.button(t!("+ Add waypoint", "+ Ajouter une étape")).clicked() {
                let at = self.route.stops.len() - 1;
                self.route.stops.insert(at, route::Stop::default());
            }
            if ui.button(t!("Find", "Chercher")).clicked() {
                find = true;
            }
            if ui.button(t!("Clear", "Effacer")).clicked() {
                self.route = route::Route::empty();
                self.route_rx.clear();
            }
        });
        if find {
            self.find_route_stops();
        }
        ui.add_space(8.0);
        if self.route.ready() {
            let total = self.route.total();
            let gs = self.plane.vel.length();
            let cruise = self.spec.cruise;
            ui.label(RichText::new(t!("Total {:.1} nm ({:.0} km)", "Total {:.1} NM ({:.0} km)", total / 1852.0, total / 1000.0)).size(15.0).strong().color(TEXT));
            let mut eta = t!("ETA {} at cruise ({:.0} kt)", "Durée {} en croisière ({:.0} kt)", route::eta(total / cruise), cruise * 1.94384);
            if gs > 20.0 && !self.plane.on_ground {
                eta += &t!("  ·  {} at {:.0} kt now", "  ·  {} à {:.0} kt actuels", route::eta(self.route.remaining(lat, lon) / gs), gs * 1.94384);
            }
            ui.label(RichText::new(eta).size(13.0).color(DIM));
            let w = self.route.wps();
            for (k, p) in w.windows(2).enumerate() {
                let d = route::dist(p[0].lat, p[0].lon, p[1].lat, p[1].lon);
                let b = route::bearing(p[0].lat, p[0].lon, p[1].lat, p[1].lon);
                let code = |w: &route::Wp| w.apt.map(|i| airports::Db::get().apts[i as usize].code()).unwrap_or_else(|| w.name.split(',').next().unwrap_or("").chars().take(16).collect());
                let now = !self.route.done && k + 1 == self.route.next;
                ui.label(RichText::new(format!("{} {} → {}   {:03.0}°   {:.1} nm", if now { ">" } else { " " }, code(p[0]), code(p[1]), b, d / 1852.0)).monospace().size(12.5).color(if now { ACCENT } else { DIM }));
            }
            ui.add_space(8.0);
            let dep_apt = w.first().and_then(|w| w.apt).is_some();
            let start = if dep_apt { t!("Start at departure (on the runway)", "Partir du départ (sur la piste)") } else { t!("Start at departure", "Partir du départ") };
            if menu::wide_button(ui, start, Hint::None, true, true).clicked() {
                *act = Some(PauseAct::RouteStart);
            }
            if menu::wide_button(ui, t!("Fly it from here", "Le suivre d'ici"), Hint::None, false, true).clicked() {
                *act = Some(PauseAct::RouteFly);
            }
        } else {
            let any_busy = self.route.stops.iter().any(|s| s.busy);
            let msg = if any_busy { t!("Looking places up…", "Recherche des lieux…") } else { t!("Type the stops, then Find (or Enter).", "Saisissez les étapes, puis Chercher (ou Entrée).") };
            ui.label(RichText::new(msg).size(13.0).color(DIM));
        }
        ui.add_space(6.0);
        ui.label(RichText::new(t!("In flight the HUD shows the next waypoint and a magenta course; J flies there (heading hold). Waypoints are passed within 1.5 km.", "En vol, le HUD montre la prochaine étape et une route magenta ; J s'y dirige (tenue de cap). Une étape est franchie à 1,5 km.")).size(12.0).color(DIM));
    }

    /// Look up the route's stops: airport codes at once, place names online.
    fn find_route_stops(&mut self) {
        for (i, q) in self.route.resolve_codes() {
            self.route.stops[i].busy = true;
            let (tx, rx) = std::sync::mpsc::channel();
            let src = self.src.clone();
            std::thread::spawn(move || {
                let w = geocode(&src.fetch, &q).map(|(la, lo, name)| route::Wp { name, lat: la, lon: lo, elev: ground_elev(&src, la, lo), apt: None });
                let _ = tx.send((i, q, w));
            });
            self.route_rx.push(rx);
        }
        self.route.restart();
    }

    /// Time acceleration: 1, 2, 4 or 8 times, only while flying.
    fn set_sim_speed(&mut self, k: u32) {
        let k = k.clamp(1, 8).next_power_of_two().min(8);
        if k > 1 && (self.crashed || (self.plane.on_ground && self.plane.vel.length() < 15.4)) {
            self.status = t!("Sim speed: in flight only", "Vitesse de simulation : en vol seulement").into();
        } else {
            self.sim_speed = k;
            self.status = t!("Sim speed {}×", "Vitesse de simulation {}×", k);
        }
        self.status_until = self.real + 2.5;
    }

    /// The flight panel (Tab): a see-through column on the right, over the
    /// game that keeps running: instruments, places to go and the weather.
    #[allow(clippy::too_many_arguments)]
    fn info_panel(&mut self, ctx: &egui::Context, act: &mut Option<PauseAct>, lat: f64, lon: f64, local: f64, speed: f64, alt: f64, agl: f64, hdg: f64) {
        use egui::{Align2, Color32, RichText};
        use menu::{Page, ACCENT, DIM, TEXT};
        if !self.info_page.is_info() {
            self.info_page = Page::Flight;
        }
        const W: f32 = 340.0;
        let max_h = ctx.content_rect().height() - 110.0;
        let frame = egui::Frame::new().fill(menu::GLASS).corner_radius(10.0).inner_margin(14.0).stroke(egui::Stroke::new(1.0, Color32::from_white_alpha(14)));
        egui::Area::new(egui::Id::unique("info_panel")).anchor(Align2::RIGHT_TOP, [-14.0, 14.0]).show(ctx, |ui| {
            frame.show(ui, |ui| {
                ui.set_width(W);
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = 3.0;
                    for p in Page::INFO {
                        let b = ui.add(egui::Button::selectable(self.info_page == p, RichText::new(p.label()).size(14.0)).min_size(egui::vec2(0.0, 26.0)));
                        if b.clicked() {
                            self.info_page = p;
                        }
                        if p == self.info_page {
                            self.pad_home_focus(&b);
                        }
                    }
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if padui::active() {
                            ui.label(RichText::new(t!("close", "fermer")).size(11.0).color(DIM));
                            padui::glyphs(ui, "B", 17.0);
                            ui.add_space(4.0);
                            padui::glyphs(ui, "LB RB", 17.0);
                        } else {
                            ui.label(RichText::new("Tab").size(11.0).color(DIM));
                        }
                    });
                });
                ui.add_space(2.0);
                egui::ScrollArea::vertical().max_height(max_h).min_scrolled_height(max_h).show(ui, |ui| {
                    let row = |ui: &mut egui::Ui, k: &str, v: String, col: Color32| {
                        ui.label(RichText::new(k).size(12.5).color(DIM));
                        ui.label(RichText::new(v).monospace().size(13.0).color(col));
                        ui.end_row();
                    };
                    match self.info_page {
                        Page::Places => {
                            ui.horizontal(|ui| {
                                let pad = padui::active();
                                if ui.button(if pad { t!("Respawn here", "Réapparaître ici") } else { t!("Respawn here  R", "Réapparaître ici  R") }).clicked() {
                                    *act = Some(PauseAct::Respawn);
                                }
                                if ui.button(if pad { t!("Search a place…", "Chercher un lieu…") } else { t!("Search a place…  G", "Chercher un lieu…  G") }).clicked() {
                                    *act = Some(PauseAct::GoTo);
                                }
                            });
                            ui.add_space(4.0);
                            let db = airports::Db::get();
                            let (la, lo, _) = geo::pos_to_lla(self.plane.pos);
                            menu::heading(ui, t!("NEAREST AIRPORTS", "AÉROPORTS LES PLUS PROCHES"));
                            let near: Vec<(u32, f64)> = db.near(la.to_degrees(), lo.to_degrees(), 120_000.0).into_iter().filter(|(i, _)| { let a = &db.apts[*i as usize]; a.any_paved() || a.kind > 0 }).take(6).collect();
                            if near.is_empty() {
                                ui.label(RichText::new(t!("No airfield within 120 km.", "Aucun aérodrome à moins de 120 km.")).size(12.5).color(DIM));
                            }
                            let here = geo::enu(self.plane.pos);
                            for (i, d) in near {
                                let a = &db.apts[i as usize];
                                let to = geo::lla_to_pos(a.lat.to_radians(), a.lon.to_radians(), 0.0) - self.plane.pos;
                                let brg = to.dot(here.0).atan2(to.dot(here.1)).to_degrees().rem_euclid(360.0);
                                ui.horizontal(|ui| {
                                    ui.label(RichText::new(&a.ident).monospace().size(13.0).color(TEXT));
                                    ui.label(RichText::new(format!("{:.0} km {:03.0}°", d / 1000.0, brg)).size(12.0).color(DIM));
                                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                        ui.spacing_mut().item_spacing.x = 3.0;
                                        if ui.small_button(t!("approach", "approche")).clicked() {
                                            *act = Some(PauseAct::Airport(i, true));
                                        }
                                        if ui.small_button(t!("runway", "piste")).clicked() {
                                            *act = Some(PauseAct::Airport(i, false));
                                        }
                                    });
                                });
                                ui.label(RichText::new(t!("{} · {:.0} m runway", "{} · piste de {:.0} m", a.name.chars().take(38).collect::<String>(), a.longest())).size(11.5).color(DIM));
                            }
                            ui.add_space(8.0);
                            menu::heading(ui, t!("AIRPORTS AROUND THE WORLD", "AÉROPORTS DU MONDE"));
                            let t = if padui::active() { t!("Lines up on the runway (\"approach\" above for a final approach).", "Aligne sur la piste (« approche » ci-dessus pour une approche finale).") } else { t!("Click to line up on the runway; shift-click for a final approach.", "Clic pour s'aligner sur la piste ; Maj+clic pour une approche finale.") };
                            ui.label(RichText::new(t).size(11.5).color(DIM));
                            ui.horizontal_wrapped(|ui| {
                                ui.spacing_mut().item_spacing = egui::vec2(4.0, 4.0);
                                for (code, name) in menu::AIRPORTS {
                                    if let Some(i) = db.find(code) {
                                        if ui.button(RichText::new(*code).size(12.0)).on_hover_text(tr(name)).clicked() {
                                            *act = Some(PauseAct::Airport(i, ui.input(|inp| inp.modifiers.shift)));
                                        }
                                    }
                                }
                            });
                            ui.add_space(8.0);
                            menu::heading(ui, t!("FLY TO", "ALLER À"));
                            ui.horizontal_wrapped(|ui| {
                                ui.spacing_mut().item_spacing = egui::vec2(4.0, 4.0);
                                for (name, la, lo) in menu::QUICK {
                                    if ui.button(RichText::new(tr(name)).size(12.0)).clicked() {
                                        *act = Some(PauseAct::FlyTo(*la, *lo, name));
                                    }
                                }
                            });
                        }
                        Page::Weather => {
                            ui.spacing_mut().slider_width = 140.0;
                            self.wx.controls(ui, &mut self.utc, &mut self.doy, lon, false);
                        }
                        _ => {
                            let p = &self.plane;
                            let (_, _, u) = geo::enu(p.pos);
                            let vs = p.vel.dot(u);
                            let wind_kt = self.wx.wind * 1.94384;
                            let rpm = p.prop_rate * 60.0 / std::f64::consts::TAU;
                            let jet = p.spec.is_jet();
                            let trk = { let (e, n, _) = geo::enu(p.pos); p.vel.dot(e).atan2(p.vel.dot(n)).to_degrees().rem_euclid(360.0) };
                            menu::heading(ui, t!("FLIGHT", "VOL"));
                            egui::Grid::new("info_fl").num_columns(2).spacing([14.0, 3.0]).show(ui, |ui| {
                                row(ui, t!("Airspeed", "Vitesse air"), format!("{:.0} kt IAS  ·  {:.0} TAS", speed * 1.94384, p.airspeed().length() * 1.94384), TEXT);
                                row(ui, t!("Ground speed", "Vitesse sol"), t!("{:.0} kt  ·  track {trk:03.0}°", "{:.0} kt  ·  route {trk:03.0}°", p.vel.length() * 1.94384), TEXT);
                                row(ui, "Altitude", format!("{:.0} ft", alt * 3.28084), TEXT);
                                row(ui, t!("Above ground", "Hauteur sol"), format!("{:.0} ft", agl.max(0.0) * 3.28084), TEXT);
                                row(ui, t!("Vertical speed", "Vitesse verticale"), t!("{:+.0} fpm", "{:+.0} ft/min", vs * 196.85), TEXT);
                                row(ui, t!("Heading", "Cap"), format!("{hdg:03.0}°"), TEXT);
                                row(ui, t!("Angle of attack", "Incidence"), format!("{:.1}°", p.alpha.to_degrees()), if p.stalled { Color32::from_rgb(255, 90, 60) } else { TEXT });
                                row(ui, t!("Load", "Facteur de charge"), format!("{:.1} g", p.g_load), if p.g_load.abs() > 3.8 { Color32::from_rgb(255, 180, 50) } else { TEXT });
                            });
                            ui.add_space(6.0);
                            menu::heading(ui, t!("AIRCRAFT", "AVION"));
                            egui::Grid::new("info_ac").num_columns(2).spacing([14.0, 3.0]).show(ui, |ui| {
                                row(ui, t!("Aircraft", "Avion"), p.spec.name.into(), TEXT);
                                if jet {
                                    row(ui, t!("Thrust", "Poussée"), format!("{:.0}%  ·  N1 {:.0}%{}", p.throttle * 100.0, p.n1 * 100.0, if p.reverser > 0.0 { "  · REV" } else { "" }), TEXT);
                                } else {
                                    row(ui, t!("Throttle", "Gaz"), format!("{:.0}%  ·  {:.0} rpm", p.throttle * 100.0, rpm), if rpm > 2710.0 { Color32::from_rgb(255, 180, 50) } else { TEXT });
                                }
                                row(ui, t!("Flaps", "Volets"), format!("{:.0}°", p.flap_notch_deg()), TEXT);
                                if p.spec.spoiler_cd > 0.0 {
                                    row(ui, t!("Speedbrake", "Aérofreins"), if p.spoilers > 0.6 { t!("ground spoilers", "spoilers sol").into() } else if p.speedbrake > 0.0 { t!("flight detent", "cran vol").into() } else { t!("down", "rentrés").to_string() }, TEXT);
                                }
                                row(ui, t!("Gear", "Train"), if p.gear >= 1.0 { t!("down", "sorti") } else if p.gear <= 0.0 { t!("up", "rentré") } else { t!("moving", "en mouvement") }.into(), TEXT);
                            });
                            let report = p.dmg.report();
                            if self.crashed {
                                ui.label(RichText::new(if padui::active() { t!("Crashed. Close the panel (B), then A to respawn.", "Crash. Fermez le panneau (B), puis A pour réapparaître.") } else { t!("Crashed. R (or A on a gamepad) to respawn.", "Crash. R (ou A à la manette) pour réapparaître.") }).color(Color32::from_rgb(255, 90, 60)));
                            } else if report.is_empty() {
                                ui.label(RichText::new(t!("No damage", "Aucun dégât")).size(12.5).color(Color32::from_rgb(140, 220, 140)));
                            } else {
                                for l in &report {
                                    ui.label(RichText::new(i18n::caps(l)).size(12.5).color(Color32::from_rgb(255, 180, 50)));
                                }
                            }
                            ui.add_space(6.0);
                            menu::heading(ui, "CONDITIONS");
                            egui::Grid::new("info_wx").num_columns(2).spacing([14.0, 3.0]).show(ui, |ui| {
                                row(ui, "Position", menu::latlon(lat, lon), TEXT);
                                row(ui, t!("Local time", "Heure locale"), menu::hhmm(local), TEXT);
                                row(ui, t!("Wind", "Vent"), if wind_kt < 1.0 { t!("calm", "calme").into() } else { format!("{:03.0}° / {:.0} kt", self.wx.wind_from, wind_kt) }, TEXT);
                                row(ui, t!("Clouds", "Nuages"), if self.wx.clouds < 0.05 { t!("clear", "dégagé").into() } else { format!("{:.0}%, base {:.0} ft", self.wx.clouds * 100.0, self.wx.cloud_base * 3.28084) }, TEXT);
                                row(ui, t!("Visibility", "Visibilité"), format!("{:.0} km", self.wx.visibility_km()), TEXT);
                            });
                            ui.add_space(6.0);
                            menu::heading(ui, t!("LIVE TRAFFIC", "TRAFIC RÉEL"));
                            if !self.live.enabled() {
                                ui.label(RichText::new(t!("Off (pause menu, Settings).", "Désactivé (menu pause, Réglages).")).size(12.0).color(DIM));
                            } else if self.live.planes.is_empty() {
                                let e = self.live.error();
                                let msg = if e.is_empty() { t!("Waiting for aircraft…", "En attente d'avions…").to_string() } else { t!("No data: {}", "Pas de données : {}", e.chars().take(60).collect::<String>()) };
                                ui.label(RichText::new(msg).size(12.0).color(DIM));
                            } else {
                                let me = self.plane.pos;
                                let mut near: Vec<&live::Plane> = self.live.planes.iter().collect();
                                near.sort_by(|a, b| (a.pos - me).length().total_cmp(&(b.pos - me).length()));
                                ui.label(RichText::new(t!("{} aircraft within 110 km", "{} avions à moins de 110 km", near.len())).size(12.0).color(DIM));
                                egui::Grid::new("info_live").num_columns(3).spacing([10.0, 2.0]).show(ui, |ui| {
                                    for p in near.iter().take(6) {
                                        ui.label(RichText::new(&p.call).monospace().size(12.5).color(TEXT));
                                        ui.label(RichText::new(if p.ty.is_empty() { "?" } else { p.ty.as_str() }).monospace().size(12.0).color(DIM));
                                        ui.label(RichText::new(format!("{:.1} km · {:.0} ft", (p.pos - me).length() / 1000.0, p.alt_ft)).monospace().size(12.0).color(DIM));
                                        ui.end_row();
                                    }
                                });
                            }
                            ui.add_space(4.0);
                            ui.label(RichText::new(if padui::active() { t!("Menu pauses: settings, the globe, quit.", "Menu met en pause : réglages, globe, quitter.") } else { t!("Esc pauses: settings, the globe, quit.", "Échap met en pause : réglages, globe, quitter.") }).size(11.5).color(ACCENT.gamma_multiply(0.8)));
                        }
                    }
                    padui::follow_focus(ui);
                });
            });
        });
    }

    /// What the airport panel says: the airfield, the runway in use and the
    /// wind, and on the way in a guide down the approach (distance, which side
    /// of the centreline, above or below the 3° glide path); on the ground,
    /// where on the airfield the aircraft is. (text, colour: 0 plain, 1 good, 2 caution, 3 warning)
    fn airport_hud(&self) -> Option<Vec<(String, u8)>> {
        if self.crashed {
            return None;
        }
        let f = self.world.airfield_near(self.plane.pos, 18_000.0)?;
        let g = &f.geo;
        let a = g.apt();
        let mut out = vec![(format!("{}  {}", a.code(), a.name), 0u8)];
        let wind_kt = self.wx.wind * 1.94384;
        let wind = if wind_kt < 1.0 { t!("calm", "calme").to_string() } else { format!("{:03.0}°/{:.0} kt", self.wx.wind_from, wind_kt) };
        let moving = f.traffic.movers.len();
        let active = f.traffic.active.map(|(ri, e)| t!("runway {} in use", "piste {} en service", g.runways[ri].ident[e])).unwrap_or_else(|| t!("no runway on the map", "aucune piste sur la carte").into());
        out.push((t!("{active}  ·  wind {wind}  ·  elevation {:.0} ft  ·  {} aircraft moving", "{active}  ·  vent {wind}  ·  altitude {:.0} ft  ·  {} avions en mouvement", g.elev * 3.28084, moving), 0));
        let p = g.local(self.plane.pos);
        let alt = geo::pos_to_lla(self.plane.pos).2 as f32;
        let fw = self.plane.forward();
        let hd = glam::Vec2::new(fw.dot(g.e) as f32, fw.dot(g.n) as f32).normalize_or_zero();
        if self.plane.on_ground || alt - g.h(p) < 3.0 {
            if let Some(ri) = g.on_runway(p, 0.0) {
                let r = &g.runways[ri];
                let e = if hd.dot(r.dir) >= 0.0 { 0 } else { 1 };
                let (s, _) = r.local(p);
                let left = if e == 0 { r.len - s } else { s };
                out.push((t!("On runway {}  ·  {:.0} m of runway ahead", "Sur la piste {}  ·  {:.0} m de piste devant", r.ident[e], left), 1));
            } else if let Some(h) = g.holds.iter().find(|h| (h.p - p).length() < 25.0) {
                let r = &g.runways[h.rwy as usize];
                out.push((t!("Holding point, runway {}", "Point d'attente, piste {}", r.name()), 2));
            }
            return Some(out);
        }
        // The runway end ahead that we are lined up with best.
        let mut best: Option<(f32, usize, usize)> = None;
        for (ri, r) in g.runways.iter().enumerate() {
            for e in 0..2 {
                let (thr, dir) = r.end(e);
                let along = (thr - p).dot(dir);
                let x = (p - thr).dot(airports::right_of(dir));
                if along > -200.0 && along < 20_000.0 && hd.dot(dir) > 0.75 && x.abs() < along.max(0.0) * 0.25 + 300.0 && best.map(|b| along < b.0).unwrap_or(true) {
                    best = Some((along, ri, e));
                }
            }
        }
        if let Some((along, ri, e)) = best {
            let r = &g.runways[ri];
            let (thr, dir) = r.end(e);
            let x = (p - thr).dot(airports::right_of(dir));
            let h = alt - g.h(thr);
            let aim = (along + traffic::aim_dist(r.len)).max(1.0);
            let gs = h.atan2(aim).to_degrees();
            let dev = gs - 3.0;
            let side = if x.abs() < 0.01 * along.max(300.0) + 5.0 { t!("on the centreline", "sur l'axe").to_string() } else { t!("{:.0} m {} of the centreline", "{:.0} m à {} de l'axe", x.abs(), if x > 0.0 { t!("right", "droite") } else { t!("left", "gauche") }) };
            let glide = if dev.abs() < 0.3 { t!("on glide path", "sur le plan de descente").to_string() } else if dev > 0.0 { t!("{:.1}° high", "{:.1}° trop haut", dev) } else { t!("{:.1}° low", "{:.1}° trop bas", -dev) };
            let col = if dev < -0.7 || x.abs() > 0.06 * along + 30.0 { 3 } else if dev.abs() > 0.3 { 2 } else { 1 };
            out.push((t!("Runway {}  {:.1} nm  ·  {}  ·  {}  ·  {:.0} ft above it", "Piste {}  {:.1} NM  ·  {}  ·  {}  ·  {:.0} ft au-dessus", r.ident[e], along.max(0.0) / 1852.0, side, glide, h * 3.28084), col));
        }
        Some(out)
    }

    fn ui(&mut self, agl: f64) -> (Vec<egui::ClippedPrimitive>, egui::TexturesDelta, f32) {
        let raw = self.egui_input();
        let (lat, lon, alt) = geo::pos_to_lla(self.plane.pos);
        let (e, n, u) = geo::enu(self.plane.pos);
        let f = self.plane.forward();
        let hdg = f.dot(e).atan2(f.dot(n)).to_degrees().rem_euclid(360.0);
        // Indicated airspeed: what the wing feels, and what the overspeed warnings are about.
        let speed = self.plane.ias();
        let tas = self.plane.airspeed().length();
        let gs = self.plane.vel.length();
        let vs = self.plane.vel.dot(u);
        let wind_kt = self.wx.wind * 1.94384;
        let local = (self.utc + lon.to_degrees() / 15.0).rem_euclid(24.0);
        let mut go: Option<String> = None;
        let mut close_goto = false;
        let mut act = None;
        let stats = self.world.stats;
        let airport_hud = self.airport_hud();
        self.src.google.update_attribution(lat.to_degrees(), lon.to_degrees(), agl);
        self.flash.retain(|(_, t)| self.real - t < 4.0);
        let out = self.egui_ctx.clone().run_ui(raw, |root| {
            let ctx = &root.ctx().clone();
            self.pad_pass(ctx);
            self.fps_counter(ctx);
            use egui::{Align2, Color32, FontId, RichText};
            let pad = padui::active();
            let white = Color32::from_rgba_unmultiplied(255, 255, 255, 235);
            let dim = Color32::from_rgba_unmultiplied(255, 255, 255, 150);
            let mono = |s: f32| FontId::monospace(s);
            let shadow = |ui: &mut egui::Ui, text: String, size: f32, col: Color32| {
                ui.label(RichText::new(text).font(mono(size)).color(col));
            };
            let panel = egui::Frame::new().fill(Color32::from_rgba_unmultiplied(8, 12, 18, 120)).corner_radius(6.0).inner_margin(10.0);
            if self.pause_open {
                self.pause_menu(ctx, &mut act, lat.to_degrees(), lon.to_degrees(), local);
                padui::focus_ring(ctx, menu::ACCENT);
                return;
            }
            if self.info_open {
                self.info_panel(ctx, &mut act, lat.to_degrees(), lon.to_degrees(), local, speed, alt, agl, hdg);
            }
            let mut banner_bottom = 14.0;
            let mut info_r = egui::Rect::NOTHING;
            if self.show_hud {
                let hud_r = egui::Area::new(egui::Id::unique("hud")).anchor(Align2::LEFT_TOP, [14.0, 14.0]).show(ctx, |ui| {
                    panel.show(ui, |ui| {
                        shadow(ui, format!("IAS {:>4.0} kt", speed * 1.94384), 20.0, white);
                        shadow(ui, format!("ALT {:>5.0} ft", alt * 3.28084), 20.0, white);
                        shadow(ui, format!("AGL {:>5.0} ft", agl.max(0.0) * 3.28084), 14.0, dim);
                        shadow(ui, format!("HDG {:>4.0}°   VS {:>+6.0} fpm", hdg, vs * 196.85), 14.0, dim);
                        if wind_kt >= 0.5 {
                            shadow(ui, format!("TAS {:>3.0}  GS {:>3.0} kt   WIND {:03.0}°/{:.0}", tas * 1.94384, gs * 1.94384, self.wx.wind_from, wind_kt), 14.0, dim);
                        } else {
                            shadow(ui, format!("TAS {:>4.0} kt", tas * 1.94384), 14.0, dim);
                        }
                        let p = &self.plane;
                        if p.spec.is_jet() {
                            let rev = if p.reverser > 0.95 { "   REV" } else if p.reverser > 0.0 { "   rev" } else { "" };
                            shadow(ui, format!("THR {:>3.0}%   N1 {:>3.0}%   G {:>4.1}{rev}", p.throttle * 100.0, p.n1 * 100.0, p.g_load), 14.0, dim);
                        } else {
                            shadow(ui, format!("THR {:>3.0}%   {:>4.0} rpm   G {:>4.1}", p.throttle * 100.0, p.prop_rate * 60.0 / std::f64::consts::TAU, p.g_load), 14.0, if p.prop_rate * 60.0 / std::f64::consts::TAU > 2710.0 { Color32::from_rgb(255, 190, 60) } else { dim });
                        }
                        let gear = if p.gear >= 1.0 { "DOWN".to_string() } else if p.gear <= 0.0 { "UP".to_string() } else { "transit".to_string() };
                        let sb = if p.spoilers > 0.05 { "   SPDBRK" } else { "" };
                        shadow(ui, format!("FLAPS {:>2.0}°   GEAR {}{sb}", p.flap_notch_deg(), gear), 14.0, if p.gear > 0.0 && p.gear < 1.0 { Color32::from_rgb(255, 190, 60) } else { dim });
                        if p.gear < 1.0 && agl < 150.0 && p.throttle < 0.35 && !self.crashed {
                            if pad {
                                ui.horizontal(|ui| {
                                    shadow(ui, "GEAR UP".into(), 20.0, Color32::from_rgb(255, 90, 60));
                                    padui::glyphs(ui, "L3", 22.0);
                                });
                            } else {
                                shadow(ui, "GEAR UP  (L)".into(), 20.0, Color32::from_rgb(255, 90, 60));
                            }
                        }
                        if self.plane.stalled && !self.crashed {
                            shadow(ui, "STALL".into(), 20.0, Color32::from_rgb(255, 90, 60));
                        }
                        if let Some(t) = self.route.target() {
                            // The next stop of the route, in the course line's magenta.
                            let (la, lo) = (lat.to_degrees(), lon.to_degrees());
                            let d = route::dist(la, lo, t.lat, t.lon);
                            let brg = route::bearing(la, lo, t.lat, t.lon);
                            let v = if gs > 20.0 { gs } else { self.spec.cruise };
                            let code = t.apt.map(|i| airports::Db::get().apts[i as usize].code()).unwrap_or_else(|| t.name.split(',').next().unwrap_or("").chars().take(18).collect());
                            ui.add_space(4.0);
                            shadow(ui, format!("→ {code}{}", if self.route.autopilot { "   AP" } else { "" }), 16.0, MAGENTA);
                            shadow(ui, format!("BRG {:03.0}°  {:.1} nm  ETE {}", brg, d / 1852.0, route::eta(d / v)), 14.0, MAGENTA);
                            if !self.route.is_last() {
                                shadow(ui, t!("DEST {:.0} nm  ETE {}", "DEST {:.0} NM  ETE {}", self.route.remaining(la, lo) / 1852.0, route::eta(self.route.remaining(la, lo) / v)), 12.0, dim);
                            }
                        }
                    });
                }).response.rect;
                if let Some(lines) = &airport_hud {
                    // Right of the HUD panel (never over it), wrapping before the
                    // fps counter and the info panel on the right.
                    let x = hud_r.right() + 12.0;
                    let right = ctx.content_rect().right() - if self.info_open { 14.0 + 340.0 + 28.0 } else { 140.0 };
                    let max_w = (right - x - 20.0).max(220.0);
                    let r = egui::Area::new(egui::Id::unique("airport")).anchor(Align2::LEFT_TOP, [x, 14.0]).show(ctx, |ui| {
                        ui.set_max_width(max_w + 20.0);
                        panel.show(ui, |ui| {
                            ui.set_max_width(max_w);
                            ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Wrap);
                            for (k, (l, col)) in lines.iter().enumerate() {
                                let (size, col) = (if k == 0 { 15.0 } else { 13.0 }, match col { 1 => Color32::from_rgb(120, 230, 140), 2 => Color32::from_rgb(255, 190, 60), 3 => Color32::from_rgb(255, 90, 60), _ => if k == 0 { white } else { dim } });
                                // Wrap only between the " · " items, never inside one.
                                ui.horizontal_wrapped(|ui| {
                                    ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Extend);
                                    for (j, part) in l.split('·').enumerate() {
                                        shadow(ui, if j == 0 { part.trim().to_string() } else { format!("· {}", part.trim()) }, size, col);
                                    }
                                });
                            }
                        });
                    });
                    banner_bottom = r.response.rect.bottom() + 8.0;
                }
                info_r = egui::Area::new(egui::Id::unique("info")).anchor(Align2::LEFT_BOTTOM, [14.0, -14.0]).show(ctx, |ui| {
                    panel.show(ui, |ui| {
                        ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Extend);
                        if self.real < self.status_until && !self.status.is_empty() {
                            shadow(ui, self.status.chars().take(70).collect(), 13.0, white);
                        }
                        shadow(ui, t!("{:.4}°{}  {:.4}°{}   {:02}:{:02} local", "{:.4}°{}  {:.4}°{}   {:02}:{:02} heure locale", lat.to_degrees().abs(), if lat >= 0.0 { 'N' } else { 'S' }, lon.to_degrees().abs(), if lon >= 0.0 { 'E' } else { 'W' }, local as u32, (local.fract() * 60.0) as u32), 13.0, dim);
                        if pad {
                            ui.horizontal(|ui| {
                                shadow(ui, t!("{:.0} fps   {} tiles   {} loading   {} cars", "{:.0} fps   {} tuiles   {} en chargement   {} voitures", self.fps, stats.0, stats.1, self.n_cars), 12.0, dim);
                                ui.add_space(6.0);
                                padui::prompts(ui, &[("Y + View", tr("panel")), ("Menu", tr("menu"))], 16.0, 12.0, dim);
                            });
                        } else {
                            shadow(ui, t!("{:.0} fps   {} tiles   {} loading   {} cars   Tab panel   Esc menu", "{:.0} fps   {} tuiles   {} en chargement   {} voitures   Tab panneau   Échap menu", self.fps, stats.0, stats.1, self.n_cars), 12.0, dim);
                        }
                        let p = self.world.photo();
                        if p.active {
                            let datum = p.datum.map(|d| format!("{d:+.1} m")).unwrap_or_else(|| t!("measuring", "mesure en cours").into());
                            shadow(ui, t!("3D tiles: {} drawn   {} in memory ({})   {} loading   datum {datum}", "Tuiles 3D : {} affichées   {} en mémoire ({})   {} en chargement   référence {datum}", p.stats.0, p.stats.1, bytes(p.gpu_bytes_shown as u64), p.stats.2), 12.0, dim);
                        }
                    });
                })
                .response
                .rect;
            }
            if self.sim_speed > 1 {
                egui::Area::new(egui::Id::unique("simspeed")).anchor(Align2::CENTER_TOP, [0.0, banner_bottom]).show(ctx, |ui| {
                    panel.show(ui, |ui| shadow(ui, format!("SIM ×{}", self.sim_speed), 20.0, Color32::from_rgb(255, 190, 60)));
                });
            }
            // The route: a magenta course line a little below the aircraft
            // and a marker over the next stop (an arrow at the edge when off screen).
            if let (true, Some(t), Some((vp, cam))) = (self.show_hud, self.route.target(), self.live_view) {
                let screen = ctx.content_rect();
                let painter = ctx.layer_painter(egui::LayerId::background());
                let proj = |p: DVec3| -> Option<egui::Pos2> {
                    let c = vp * (p - cam).as_vec3().extend(1.0);
                    (c.w > 1.0).then(|| egui::pos2(screen.left() + (c.x / c.w * 0.5 + 0.5) * screen.width(), screen.top() + (0.5 - c.y / c.w * 0.5) * screen.height()))
                };
                let h_line = (alt - 120.0).max(t.elev + 150.0);
                let ws = self.route.wps();
                let from = if self.route.next > 0 { ws.get(self.route.next - 1).map(|w| (w.lat, w.lon)) } else { None }.unwrap_or((lat.to_degrees(), lon.to_degrees()));
                // From here to the next stop, bright; the planned leg there, faint.
                let mut legs = vec![((lat.to_degrees(), lon.to_degrees()), (t.lat, t.lon), 255u8), (from, (t.lat, t.lon), 90)];
                for p in ws[self.route.next.min(ws.len())..].windows(2) {
                    legs.push(((p[0].lat, p[0].lon), (p[1].lat, p[1].lon), 120));
                }
                for (a, b, al) in legs {
                    let d = route::dist(a.0, a.1, b.0, b.1);
                    let pts = route::arc(a, b, ((d / 1500.0) as usize).clamp(8, 160));
                    let col = Color32::from_rgba_unmultiplied(MAGENTA.r(), MAGENTA.g(), MAGENTA.b(), al);
                    let mut prev: Option<egui::Pos2> = None;
                    for u in pts {
                        let q = proj(u * (geo::R + h_line));
                        if let (Some(p0), Some(p1)) = (prev, q) {
                            if screen.expand(400.0).contains(p0) || screen.expand(400.0).contains(p1) {
                                painter.line_segment([p0 + egui::vec2(1.0, 1.0), p1 + egui::vec2(1.0, 1.0)], egui::Stroke::new(3.0, Color32::from_black_alpha(al / 3)));
                                painter.line_segment([p0, p1], egui::Stroke::new(2.0, col));
                            }
                        }
                        prev = q;
                    }
                }
                let at = geo::lla_to_pos(t.lat.to_radians(), t.lon.to_radians(), t.elev + 250.0);
                let d = route::dist(lat.to_degrees(), lon.to_degrees(), t.lat, t.lon);
                let label = format!("{}  {:.1} nm", t.apt.map(|i| airports::Db::get().apts[i as usize].code()).unwrap_or_else(|| t.name.split(',').next().unwrap_or("").chars().take(18).collect()), d / 1852.0);
                match proj(at).filter(|p| screen.shrink(20.0).contains(*p)) {
                    Some(p) => {
                        // A diamond with a pole down to the ground.
                        if let Some(g) = proj(geo::lla_to_pos(t.lat.to_radians(), t.lon.to_radians(), t.elev)) {
                            painter.line_segment([p, g], egui::Stroke::new(1.5, MAGENTA));
                        }
                        let r = 9.0;
                        let pts = vec![p + egui::vec2(0.0, -r), p + egui::vec2(r, 0.0), p + egui::vec2(0.0, r), p + egui::vec2(-r, 0.0)];
                        painter.add(egui::Shape::convex_polygon(pts, Color32::from_rgba_unmultiplied(MAGENTA.r(), MAGENTA.g(), MAGENTA.b(), 90), egui::Stroke::new(2.0, MAGENTA)));
                        painter.text(p + egui::vec2(13.0, 1.0), Align2::LEFT_CENTER, &label, FontId::proportional(13.0), Color32::from_black_alpha(160));
                        painter.text(p + egui::vec2(12.0, 0.0), Align2::LEFT_CENTER, &label, FontId::proportional(13.0), MAGENTA);
                    }
                    None => {
                        // Off screen: a chevron at the edge, pointing the way.
                        let c = vp * (at - cam).as_vec3().extend(1.0);
                        let mut dir = egui::vec2(c.x, -c.y) * c.w.signum();
                        if dir.length() < 1e-6 {
                            dir = egui::vec2(0.0, 1.0);
                        }
                        let dir = dir.normalized();
                        let mid = screen.center();
                        let k = ((screen.width() * 0.5 - 50.0) / dir.x.abs().max(1e-4)).min((screen.height() * 0.5 - 50.0) / dir.y.abs().max(1e-4));
                        let p = mid + dir * k;
                        let side = egui::vec2(-dir.y, dir.x);
                        for o in [0.0, 10.0] {
                            let tip = p + dir * (12.0 - o);
                            painter.line_segment([tip - dir * 12.0 + side * 11.0, tip], egui::Stroke::new(3.0, MAGENTA));
                            painter.line_segment([tip - dir * 12.0 - side * 11.0, tip], egui::Stroke::new(3.0, MAGENTA));
                        }
                        let align = if dir.x > 0.5 { Align2::RIGHT_CENTER } else if dir.x < -0.5 { Align2::LEFT_CENTER } else if dir.y > 0.0 { Align2::CENTER_BOTTOM } else { Align2::CENTER_TOP };
                        painter.text(p - dir * 26.0, align, &label, FontId::proportional(13.0), MAGENTA);
                    }
                }
            }
            // Real traffic: a tag beside each aircraft within range.
            if let (true, Some((vp, cam))) = (self.show_hud && self.live.enabled(), self.live_view) {
                let screen = ctx.content_rect();
                let painter = ctx.layer_painter(egui::LayerId::background());
                for p in &self.live.planes {
                    let rel = p.pos - cam;
                    let d = rel.length();
                    if d > 40_000.0 || d < 30.0 {
                        continue;
                    }
                    let c = vp * rel.as_vec3().extend(1.0);
                    if c.w <= 0.0 {
                        continue;
                    }
                    let at = egui::pos2(screen.left() + (c.x / c.w * 0.5 + 0.5) * screen.width(), screen.top() + (0.5 - c.y / c.w * 0.5) * screen.height());
                    if !screen.expand(40.0).contains(at) {
                        continue;
                    }
                    let a = (255.0 * (1.0 - d / 40_000.0).clamp(0.35, 1.0)) as u8;
                    let col = Color32::from_rgba_unmultiplied(150, 225, 255, a);
                    painter.circle_stroke(at, 5.0, egui::Stroke::new(1.3, col));
                    let line2 = format!("{}  {:.0} ft  {:.0} kt  {:.1} km", if p.ty.is_empty() { "?" } else { p.ty.as_str() }, p.alt_ft, p.gs_kt, d / 1000.0);
                    for (k, (s, size)) in [(p.call.as_str(), 12.5), (line2.as_str(), 11.0)].into_iter().enumerate() {
                        let pos = at + egui::vec2(9.0, -10.0 + 14.0 * k as f32);
                        painter.text(pos + egui::vec2(1.0, 1.0), Align2::LEFT_TOP, s, FontId::proportional(size), Color32::from_black_alpha(a / 2));
                        painter.text(pos, Align2::LEFT_TOP, s, FontId::proportional(size), col);
                    }
                }
            }
            // Google's imagery has to carry its credit, HUD or not.
            let credits_r = if self.show_hud || self.google_shown() { self.credits(ctx) } else { egui::Rect::NOTHING };
            // The maps: the minimap in the lower-right corner, above the
            // credits and clear of the info panel; the large one on M.
            let mini = self.show_hud && self.map.mini_on && !self.info_open && !self.map.open && !self.crashed;
            if mini || self.map.open {
                self.map.begin(ctx);
                let (hdg_d, gs_d) = (hdg, gs);
                let traffic: Vec<(f64, f64, f64)> = if self.live.enabled() {
                    self.live
                        .planes
                        .iter()
                        .map(|p| {
                            let (la, lo, _) = geo::pos_to_lla(p.pos);
                            let (e, n, _) = geo::enu(p.pos);
                            (la.to_degrees(), lo.to_degrees(), p.fwd.dot(e).atan2(p.fwd.dot(n)).to_degrees())
                        })
                        .collect()
                } else {
                    Vec::new()
                };
                let scene = minimap::Scene { lat: lat.to_degrees(), lon: lon.to_degrees(), hdg: hdg_d, gs: gs_d, cruise: self.spec.cruise, agl, route: &self.route, traffic };
                if mini {
                    let screen = ctx.content_rect();
                    let bottom = if credits_r.is_positive() { credits_r.top() - 6.0 } else { screen.bottom() - 14.0 };
                    let right = screen.right() - 14.0;
                    // As big as fits beside the info panel; above it when that is too small.
                    let base = 240f32.min(screen.width() * 0.3).min(screen.height() * 0.38).max(120.0);
                    let beside = if info_r.is_positive() { right - info_r.right() - 14.0 } else { f32::MAX };
                    let (size, bottom) = if beside >= base.min(170.0) || !info_r.is_positive() { (base.min(beside), bottom) } else { (base, bottom.min(info_r.top() - 8.0)) };
                    let rect = egui::Rect::from_min_max(egui::pos2(right - size, bottom - size), egui::pos2(right, bottom));
                    self.map.mini(ctx, rect, &scene);
                }
                let picked = if self.map.open { self.map.full(ctx, &scene) } else { None };
                self.map.end();
                if let Some(p) = picked {
                    self.status = minimap::plan(&mut self.route, p);
                    self.status_until = self.real + 3.0;
                }
            }
            // Damage: warnings while the airframe is being overstressed, what has
            // already broken, and a flash for each new loss.
            let p = &self.plane;
            let mut warn: Vec<(&str, Color32)> = Vec::new();
            let red = Color32::from_rgb(255, 80, 50);
            let amber = Color32::from_rgb(255, 180, 50);
            if !self.crashed && self.show_hud {
                let w = &self.warn;
                if w.pull_up {
                    warn.push(("PULL UP", red));
                } else if w.dive > 0.0 {
                    warn.push(("NOSE DIVE", red));
                } else if w.too_low {
                    warn.push(("TOO LOW  TERRAIN", amber));
                } else if w.sink {
                    warn.push(("SINK RATE", amber));
                }
                let over = p.overspeed();
                if over > 13.0 {
                    warn.push(("OVERSPEED  AIRFRAME BREAKING UP", red));
                } else if over > 0.0 {
                    warn.push(("OVERSPEED", amber));
                }
                if p.flaps > 0.25 && speed > p.vfe() {
                    warn.push(("FLAPS OVERSPEED", if p.dmg.flap_stress > 0.5 { red } else { amber }));
                }
                if p.g_load.abs() > 5.0 {
                    warn.push(("OVER G", if p.dmg.fatigue > 1.2 { red } else { amber }));
                }
            }
            let report = if self.crashed { Vec::new() } else { p.dmg.report() };
            if self.show_hud && (!warn.is_empty() || !report.is_empty()) {
                egui::Area::new(egui::Id::unique("damage")).anchor(Align2::RIGHT_TOP, [if self.info_open { -14.0 - 340.0 - 28.0 - 14.0 } else { -14.0 }, 14.0]).show(ctx, |ui| {
                    panel.show(ui, |ui| {
                        ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Extend);
                        for (w, col) in &warn {
                            shadow(ui, (*w).into(), 18.0, *col);
                        }
                        if !report.is_empty() {
                            shadow(ui, t!("DAMAGE", "DÉGÂTS").into(), 12.0, dim);
                            for l in &report {
                                shadow(ui, i18n::caps(l), 14.0, amber);
                            }
                        }
                    });
                });
            }
            if !self.flash.is_empty() && !self.crashed {
                egui::Area::new(egui::Id::unique("flash")).anchor(Align2::CENTER_TOP, [0.0, 70.0]).show(ctx, |ui| {
                    ui.vertical_centered(|ui| {
                        for (s, t0) in &self.flash {
                            let a = (1.0 - ((self.real - t0 - 2.5) / 1.5).max(0.0)).clamp(0.0, 1.0);
                            shadow(ui, i18n::caps(s), 24.0, Color32::from_rgba_unmultiplied(255, 70, 40, (a * 255.0) as u8));
                        }
                    });
                });
            }
            // Come to rest in pieces without burning: say so, and how to go on.
            if !self.crashed && self.plane.wrecked() && self.plane.vel.length() < 2.0 && (self.plane.on_hull || self.plane.on_ground) {
                self.wreck_t += self.real - self.wreck_seen;
            } else {
                self.wreck_t = 0.0;
            }
            self.wreck_seen = self.real;
            if !self.crashed && self.wreck_t > 2.0 && self.mode == Mode::Fly {
                egui::Area::new(egui::Id::unique("wreck")).anchor(Align2::CENTER_BOTTOM, [0.0, -70.0]).show(ctx, |ui| {
                    panel.show(ui, |ui| {
                        ui.vertical_centered(|ui| {
                            shadow(ui, t!("AIRCRAFT WRECKED", "AVION DÉTRUIT").into(), 22.0, Color32::from_rgb(255, 150, 60));
                            if pad {
                                padui::prompts(ui, &[("Y + View", tr("go somewhere else")), ("Menu", tr("menu: respawn, globe"))], 24.0, 18.0, white);
                            } else {
                                shadow(ui, t!("R  respawn      G  go somewhere else      Esc  menu", "R  réapparaître      G  aller ailleurs      Échap  menu").into(), 18.0, white);
                            }
                        });
                    });
                });
            }
            if self.crashed && self.real - self.crash_real > 3.5 {
                egui::Area::new(egui::Id::unique("crash")).anchor(Align2::CENTER_BOTTOM, [0.0, -70.0]).show(ctx, |ui| {
                    panel.show(ui, |ui| {
                        if pad {
                            padui::prompts(ui, &[("A", tr("respawn")), ("Y + View", tr("go somewhere else")), ("Menu", tr("menu"))], 24.0, 18.0, white);
                        } else {
                            shadow(ui, t!("R  respawn      G  go somewhere else      Esc  menu", "R  réapparaître      G  aller ailleurs      Échap  menu").into(), 18.0, white);
                        }
                    });
                });
            }
            if self.show_help {
                egui::Area::new(egui::Id::unique("help")).anchor(Align2::CENTER_CENTER, [0.0, 0.0]).show(ctx, |ui| {
                    menu::panel().show(ui, |ui| {
                        ui.set_width(600.0);
                        ui.horizontal(|ui| {
                            ui.label(RichText::new(t!("Controls", "Commandes")).size(22.0).strong().color(Color32::WHITE));
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                ui.label(RichText::new(t!("F1 / Esc  close", "F1 / Échap  fermer")).size(12.0).color(menu::DIM));
                            });
                        });
                        ui.add_space(4.0);
                        menu::controls_list(ui);
                    });
                });
            }
            if self.wx.open {
                self.wx.panel(ctx, &mut self.utc, &mut self.doy, lon.to_degrees());
            }
            if self.goto_open {
                egui::Window::new(t!("Go to", "Aller à")).id(egui::Id::unique("goto_window")).collapsible(false).resizable(false).anchor(Align2::CENTER_CENTER, [0.0, -80.0]).show(ctx, |ui| {
                    ui.label(t!("City, landmark, street address, or an airport code (KSFO, LHR):", "Ville, site, adresse ou code d'aéroport (KSFO, LHR) :"));
                    let resp = ui.add(egui::TextEdit::singleline(&mut self.goto_text).desired_width(360.0));
                    resp.request_focus();
                    if ui.input(|i| i.key_pressed(egui::Key::Enter)) && !self.goto_text.trim().is_empty() {
                        go = Some(self.goto_text.trim().to_string());
                    }
                    if ui.input(|i| i.key_pressed(egui::Key::Escape)) {
                        close_goto = true;
                    }
                });
            }
            padui::focus_ring(ctx, menu::ACCENT);
        });
        self.egui_state.handle_platform_output(&self.window, out.platform_output);
        // A button or slider left focused on the flight panel would keep the
        // keys from the aircraft, which goes on flying while the panel is open.
        // (Except where the gamepad works the panel.)
        if !self.pause_open && !(self.info_open && padui::active()) && !self.egui_ctx.text_edit_focused() {
            self.egui_ctx.memory_mut(|m| {
                if let Some(id) = m.focused() {
                    m.surrender_focus(id);
                }
            });
        }
        if let Some(i) = go.as_deref().and_then(|q| airports::Db::get().find(q)) {
            self.goto_open = false;
            go = None;
            self.start_airport(i, None, false, true);
        }
        if let Some(q) = go {
            self.goto_open = false;
            self.status = t!("Searching for {}…", "Recherche de {}…", q);
            self.status_until = self.real + 30.0;
            let (tx, rx) = std::sync::mpsc::channel();
            let src = self.src.clone();
            std::thread::spawn(move || {
                let _ = tx.send(geocode(&src.fetch, &q).map(|(la, lo, name)| (la, lo, ground_elev(&src, la, lo), name)));
            });
            self.goto_rx = Some(rx);
        }
        if close_goto {
            self.goto_open = false;
        }
        match act {
            Some(PauseAct::Resume) => self.set_pause_menu(false),
            Some(PauseAct::Respawn) => {
                self.set_pause_menu(false);
                self.info_open = false;
                self.respawn_here();
            }
            Some(PauseAct::GoTo) => {
                self.set_pause_menu(false);
                self.goto_open = true;
                self.goto_text.clear();
            }
            Some(PauseAct::FlyTo(lat, lon, name)) => {
                self.set_pause_menu(false);
                self.info_open = false;
                self.fly_to(lat, lon, name);
            }
            Some(PauseAct::Airport(i, appr)) => {
                self.set_pause_menu(false);
                self.info_open = false;
                self.start_airport(i, None, appr, true);
            }
            Some(PauseAct::RouteStart) => {
                self.set_pause_menu(false);
                self.info_open = false;
                self.route.restart();
                if let Some(w) = self.route.wps().first().map(|w| (*w).clone()) {
                    match w.apt {
                        Some(i) => self.start_airport(i, None, false, true),
                        None => self.fly_to(w.lat, w.lon, &w.name),
                    }
                }
            }
            Some(PauseAct::RouteFly) => {
                self.set_pause_menu(false);
                self.route.restart();
            }
            Some(PauseAct::Globe) => self.to_globe(),
            Some(PauseAct::Quit) => self.quit = true,
            None => {}
        }
        let jobs = self.egui_ctx.tessellate(out.shapes, out.pixels_per_point);
        (jobs, out.textures_delta, out.pixels_per_point)
    }
}

struct App {
    args: Args,
    game: Option<Game>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        if self.game.is_some() {
            return;
        }
        let attrs = Window::default_attributes().with_title("Samiam's Flight Sim").with_inner_size(winit::dpi::PhysicalSize::new(self.args.size.0, self.args.size.1));
        let window = Arc::new(el.create_window(attrs).expect("window"));
        match Game::new(window, &self.args) {
            Ok(g) => self.game = Some(g),
            Err(e) => {
                eprintln!("failed to start: {e:#}");
                el.exit();
            }
        }
    }

    fn window_event(&mut self, el: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        let Some(g) = self.game.as_mut() else { return };
        let consumed = g.egui_state.on_window_event(&g.window, &event).consumed;
        match event {
            WindowEvent::CloseRequested => {
                g.keep_settings(true);
                g.src.google.save();
                el.exit()
            }
            WindowEvent::Resized(s) => g.r.resize(s.width, s.height),
            WindowEvent::RedrawRequested => {
                g.frame();
                if g.quit {
                    g.keep_settings(true);
                    g.src.google.save();
                    // Worker threads may be mid-download, so the process just
                    // exits; but the GPU device is torn down first: exiting
                    // under the driver's shader compiler threads can crash it.
                    self.game = None;
                    std::process::exit(0);
                }
            }
            WindowEvent::Focused(false) => g.keys.clear(),
            WindowEvent::KeyboardInput { event: k, .. } => {
                if let PhysicalKey::Code(code) = k.physical_key {
                    if k.state == ElementState::Pressed {
                        // egui always claims Tab (and, while one of its buttons has
                        // focus, every key), so the menu keys are let through here.
                        let typing = g.goto_open || g.egui_ctx.text_edit_focused();
                        let menu_key = code == KeyCode::Tab || code == KeyCode::Escape;
                        // A key while the gamepad has a button of the flight panel picked
                        // out: the keyboard flies again (egui would keep the keys).
                        let mut consumed = consumed;
                        if consumed && g.mode == Mode::Fly && !g.pause_open && !typing {
                            g.egui_ctx.memory_mut(|m| {
                                if let Some(id) = m.focused() {
                                    m.surrender_focus(id);
                                }
                            });
                            consumed = false;
                        }
                        if menu_key && !typing {
                            if !k.repeat {
                                g.key(code);
                            }
                        } else if !(consumed || g.goto_open) {
                            if !k.repeat {
                                g.key(code);
                            }
                            g.keys.insert(code);
                        }
                    } else {
                        g.keys.remove(&code);
                    }
                }
            }
            WindowEvent::MouseInput { state, button: MouseButton::Left, .. } if g.mode == Mode::Menu => {
                let size = g.window.inner_size();
                let (w, h) = (size.width.max(1) as f64, size.height.max(1) as f64);
                if state == ElementState::Pressed {
                    if !consumed {
                        g.menu.press = g.last_cursor;
                        g.menu.dragged = 0.0;
                    }
                } else if let Some((x, y)) = g.menu.press.take() {
                    // A click (not a drag) chooses the spot; a second click on it starts.
                    if g.menu.dragged < 6.0 {
                        if let Some(i) = g.menu.marker_at(x, y) {
                            let double = g.real - g.last_click < 0.45;
                            g.last_click = g.real;
                            if double && g.menu.airport == Some(i) {
                                g.begin();
                            } else {
                                g.choose_airport(i);
                            }
                        } else if let Some((lat, lon)) = g.menu.pick_at(x, y, w, h, 60f64.to_radians()) {
                            let double = g.real - g.last_click < 0.45;
                            g.last_click = g.real;
                            if double && g.menu.pick.is_some() {
                                g.begin();
                            } else {
                                g.pick(lat, lon);
                            }
                        }
                    }
                }
            }
            WindowEvent::MouseInput { state, button: MouseButton::Left | MouseButton::Right, .. } if !consumed => {
                g.drag = state == ElementState::Pressed;
                g.last_cursor = None;
            }
            WindowEvent::CursorMoved { position, .. } => {
                if g.mode == Mode::Menu {
                    if let (Some(_), Some((x, y))) = (g.menu.press, g.last_cursor) {
                        let h = g.window.inner_size().height.max(1) as f64;
                        g.menu.drag(position.x - x, position.y - y, h, 60f64.to_radians());
                    }
                } else if g.drag && !g.pause_open {
                    if let Some((x, y)) = g.last_cursor {
                        // Zoomed in, the view turns slower under the mouse.
                        let k = g.look_sens * if g.cam_mode == 1 { g.head.zoom } else { 1.0 };
                        g.cam_yaw -= (position.x - x) * 0.005 * k;
                        g.cam_pitch = (g.cam_pitch + (position.y - y) * 0.004 * k).clamp(-1.2, 1.4);
                    }
                }
                g.last_cursor = Some((position.x, position.y));
            }
            WindowEvent::MouseWheel { delta, .. } if !consumed => {
                let d = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y as f64,
                    MouseScrollDelta::PixelDelta(p) => p.y / 40.0,
                };
                if g.mode == Mode::Menu {
                    g.menu.zoom(d);
                } else if g.cam_mode == 1 && !g.crashed {
                    g.head.zoom_by(d);
                } else {
                    g.cam_dist = (g.cam_dist * (1.0 - d * 0.1)).clamp(6.0, (2.5 * g.spec.cam_dist).max(400.0));
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _: &ActiveEventLoop) {
        if let Some(g) = &self.game {
            g.window.request_redraw();
        }
    }
}

/// The UI library ships no fonts of its own here, so borrow the system's.
fn load_fonts(ctx: &egui::Context) {
    let find = |kind: &str, fallbacks: &[&str]| -> Option<Vec<u8>> {
        let fc = std::process::Command::new("fc-match").args(["-f", "%{file}", kind]).output().ok().and_then(|o| String::from_utf8(o.stdout).ok());
        fc.iter().map(|s| s.trim().to_string()).chain(fallbacks.iter().map(|s| s.to_string())).filter(|p| p.ends_with(".ttf") || p.ends_with(".otf")).find_map(|p| std::fs::read(p).ok())
    };
    let mut fonts = egui::FontDefinitions::empty();
    // fontconfig's answer where there is one (Linux), then each system's own fonts.
    let mono: &[&str] = &[
        "/usr/share/fonts/google-noto/NotoSansMono-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/dejavu-sans-mono-fonts/DejaVuSansMono.ttf",
        "/System/Library/Fonts/SFNSMono.ttf",
        "/System/Library/Fonts/Monaco.ttf",
        "/System/Library/Fonts/Supplemental/Andale Mono.ttf",
        "/System/Library/Fonts/Supplemental/Courier New.ttf",
        "C:/Windows/Fonts/consola.ttf",
        "C:/Windows/Fonts/cour.ttf",
    ];
    let sans: &[&str] = &[
        "/usr/share/fonts/google-noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
        "/System/Library/Fonts/SFNS.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Geneva.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ];
    for (name, family, kind, fb) in [("mono", egui::FontFamily::Monospace, "monospace", mono), ("sans", egui::FontFamily::Proportional, "sans", sans)] {
        if let Some(data) = find(kind, fb) {
            fonts.font_data.insert(name.into(), Arc::new(egui::FontData::from_owned(data)));
            fonts.families.entry(family).or_default().push(name.into());
        } else {
            log::warn!("no {kind} font found; HUD text will be missing");
        }
    }
    ctx.set_fonts(fonts);
}

/// Right-handed view matrix looking along `dir`.
fn look_to(eye: Vec3, dir: Vec3, up: Vec3) -> Mat4 {
    let f = dir.normalize();
    let s = f.cross(up).normalize();
    let u = s.cross(f);
    Mat4::from_cols(s.x_axis4(u, f, 0), s.x_axis4(u, f, 1), s.x_axis4(u, f, 2), glam::Vec4::new(-s.dot(eye), -u.dot(eye), f.dot(eye), 1.0))
}
trait Axis4 {
    fn x_axis4(self, u: Vec3, f: Vec3, i: usize) -> glam::Vec4;
}
impl Axis4 for Vec3 {
    fn x_axis4(self, u: Vec3, f: Vec3, i: usize) -> glam::Vec4 {
        glam::Vec4::new(self[i], u[i], -f[i], 0.0)
    }
}
/// Infinite far plane, reversed depth (1 at the near plane, 0 at infinity).
fn perspective_rev(fov_y: f32, aspect: f32, near: f32) -> Mat4 {
    let f = 1.0 / (fov_y * 0.5).tan();
    Mat4::from_cols_array(&[f / aspect, 0.0, 0.0, 0.0, 0.0, f, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, near, 0.0])
}
fn ortho(l: f32, r: f32, b: f32, t: f32, n: f32, f: f32) -> Mat4 {
    Mat4::from_cols_array(&[2.0 / (r - l), 0.0, 0.0, 0.0, 0.0, 2.0 / (t - b), 0.0, 0.0, 0.0, 0.0, -1.0 / (f - n), 0.0, -(r + l) / (r - l), -(t + b) / (t - b), -n / (f - n), 1.0])
}

fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn,fablegame=info")).init();
    let args = parse_args();
    i18n::set_lang(args.lang.or(settings::Settings::load().lang).unwrap_or_else(i18n::Lang::detect));
    let el = EventLoop::new()?;
    el.set_control_flow(ControlFlow::Poll);
    el.run_app(&mut App { args, game: None })?;
    Ok(())
}


const REC_FPS: f64 = 30.0;
/// Airborne starts: 107 kt, about a 172's cruise down low.

/// 16-bit stereo WAV.
fn write_wav(path: &std::path::Path, x: &[[f32; 2]], rate: u32) {
    let data = (x.len() * 4) as u32;
    let mut w = Vec::with_capacity(44 + data as usize);
    w.extend_from_slice(b"RIFF");
    w.extend_from_slice(&(36 + data).to_le_bytes());
    w.extend_from_slice(b"WAVEfmt ");
    for v in [16u32.to_le_bytes(), 0x0002_0001u32.to_le_bytes(), rate.to_le_bytes(), (rate * 4).to_le_bytes(), 0x0010_0004u32.to_le_bytes()] {
        w.extend_from_slice(&v);
    }
    w.extend_from_slice(b"data");
    w.extend_from_slice(&data.to_le_bytes());
    for s in x {
        for c in s {
            w.extend_from_slice(&((c.clamp(-1.0, 1.0) * 32767.0) as i16).to_le_bytes());
        }
    }
    if let Err(e) = std::fs::write(path, w) {
        log::error!("writing {}: {e}", path.display());
    }
}

fn smoothstep(a: f64, b: f64, x: f64) -> f64 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// +1, -1 or 0 from a pair of buttons.
fn ax_pad(pos: bool, neg: bool) -> f64 {
    (pos as i32 - neg as i32) as f64
}

/// The xpad driver's Xbox layout, in SDL's game controller mapping format.
const XBOX_LAYOUT: &str = "a:b0,b:b1,back:b6,dpdown:h0.4,dpleft:h0.8,dpright:h0.2,dpup:h0.1,guide:b8,leftshoulder:b4,leftstick:b9,lefttrigger:a2,leftx:a0,lefty:a1,rightshoulder:b5,rightstick:b10,righttrigger:a5,rightx:a3,righty:a4,start:b7,x:b2,y:b3,platform:Linux,";

/// An Xbox pad gilrs found no mapping for: a firmware newer than its copy of
/// the SDL controller database, say. Left so, its triggers show up as bare
/// axes and X and Y trade places (xpad reports X as BTN_NORTH).
fn needs_xbox_layout(p: &gilrs::Gamepad) -> bool {
    let name = p.os_name().to_lowercase();
    cfg!(target_os = "linux")
        && p.mapping_source() != gilrs::MappingSource::SdlMappings
        && (p.vendor_id() == Some(0x045e) || name.contains("xbox") || name.contains("x-box"))
}

/// The gamepads, with any Xbox pad that has no mapping given the standard Xbox one.
fn open_gamepads() -> Option<gilrs::Gilrs> {
    // (scripted test runs: a pad someone is holding would steer the aircraft)
    if std::env::var_os("FABLE_NO_PAD").is_some() {
        return None;
    }
    let g = gilrs::Gilrs::new().map_err(|e| log::warn!("gamepads: {e}")).ok()?;
    let extra: String = g
        .gamepads()
        .filter(|(_, p)| needs_xbox_layout(p))
        .map(|(_, p)| {
            let guid: String = p.uuid().iter().map(|b| format!("{b:02x}")).collect();
            log::info!("gamepad: {} ({guid}) has no mapping; using the Xbox layout", p.os_name());
            format!("{guid},{},{XBOX_LAYOUT}\n", p.os_name().replace(',', " "))
        })
        .collect();
    if extra.is_empty() {
        return Some(g);
    }
    drop(g);
    let g = gilrs::GilrsBuilder::new().add_mappings(&extra).build().map_err(|e| log::warn!("gamepads: {e}")).ok()?;
    for (_, p) in g.gamepads() {
        log::info!("gamepad: {} mapping {:?}", p.os_name(), p.mapping_source());
    }
    Some(g)
}
