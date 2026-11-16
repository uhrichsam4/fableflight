//! Weather and time of day: the live state behind the F3 panel, the wind the
//! aircraft flies through (gusts, turbulence, thermals and storm drafts), and
//! how far that wind has carried the clouds.
use crate::geo;
use glam::{DVec2, DVec3};

pub struct Weather {
    /// cloud cover, 0..1
    pub clouds: f32,
    /// cloud layer base and thickness, metres
    pub cloud_base: f32,
    pub cloud_depth: f32,
    /// how many cumulonimbus storm cells tower out of the layer, 0..1
    pub storms: f32,
    /// how much mid-level cloud (altocumulus, thickening into altostratus), 0..1
    pub alto: f32,
    /// snow, 0..1: lying on what faces up (more of it higher up), and
    /// falling while the sky is overcast (see `snowfall`)
    pub snow: f32,
    /// aerosol haze, as a multiple of a clear day's
    pub haze: f32,
    /// wind 10 m above the ground, m/s, and the compass direction it blows from
    pub wind: f64,
    pub wind_from: f64,
    /// gustiness, 0..1
    pub gusts: f64,
    /// game seconds per real second for the clock (0 stops it)
    pub time_rate: f64,
    /// days since new moon
    pub moon_age: f64,
    /// how far the wind has carried the clouds, planet frame
    pub cloud_off: DVec3,
    /// the F3 panel is showing
    pub open: bool,
    /// sine of the sun's elevation where the aircraft is (drives the thermals)
    pub sun: f64,
    /// how far the boundary-layer wind has carried the thermals, planet frame
    pub thermal_off: DVec3,
    /// the storm cells handed to the shaders last frame (see `storm_cells`)
    pub cells: Vec<[[f32; 4]; 2]>,
    /// real weather (see livewx.rs) sets the sliders: where its report came
    /// from ("EBBR", "Open-Meteo") and the report itself
    pub live: bool,
    pub live_station: String,
    pub live_raw: String,
    /// the clock follows the real time and date (see `real_clock`)
    pub real_time: bool,
}

/// What the air around the aircraft is doing beyond the mean wind.
#[derive(Clone, Copy, Debug)]
pub struct Air {
    /// RMS turbulence, m/s: vertical, and horizontal (larger near the ground)
    pub sigma_w: f64,
    pub sigma_h: f64,
    /// turbulence length scales, m: vertical, horizontal
    pub len_w: f64,
    pub len_h: f64,
    /// steady vertical motion of the air (thermals, storm drafts), m/s, + up
    pub draft: f64,
}

impl Default for Air {
    fn default() -> Self {
        Self { sigma_w: 0.0, sigma_h: 0.0, len_w: 300.0, len_h: 300.0, draft: 0.0 }
    }
}

/// (name, cover, base, thickness, storms, mid-level cloud, haze, wind, gusts)
const PRESETS: &[(&str, f32, f32, f32, f32, f32, f32, f64, f64)] = &[
    ("Clear", 0.05, 1500.0, 4500.0, 0.0, 0.0, 0.6, 2.0, 0.0),
    ("Fair", 0.45, 1500.0, 4500.0, 0.2, 0.25, 1.0, 4.0, 0.15),
    ("Hazy", 0.2, 1800.0, 3000.0, 0.0, 0.2, 8.0, 2.0, 0.05),
    ("Overcast", 0.95, 800.0, 2500.0, 0.0, 0.7, 3.0, 7.0, 0.3),
    ("Windy", 0.55, 1200.0, 4000.0, 0.1, 0.35, 1.5, 14.0, 0.7),
    ("Storms", 0.6, 1300.0, 5000.0, 0.85, 0.5, 1.4, 9.0, 0.5),
];
/// Where storm towers flatten into anvils (tropopause), metres.
pub const STORM_TOP: f32 = 11000.0;
/// Most storm cells handed to the shaders at once (nearest first).
pub const MAX_STORMS: usize = 16;
/// Spacing of candidate storm sites, km.
const STORM_GRID: f64 = 40.0;
/// Storm cells are found out to this distance from the camera, km.
const STORM_RANGE: f64 = 190.0;

fn shash(x: u32) -> u32 {
    let h = x.wrapping_mul(747796405).wrapping_add(2891336453);
    let h = ((h >> ((h >> 28) + 4)) ^ h).wrapping_mul(277803737);
    (h >> 22) ^ h
}
fn unit(h: u32) -> f64 {
    (h & 0xffff) as f64 / 65535.0
}
/// Smooth 0..1 value noise over a 3D lattice, for where storms cluster.
fn value_noise(p: DVec3) -> f64 {
    let c = p.floor();
    let f = p - c;
    let f = f * f * (DVec3::splat(3.0) - 2.0 * f);
    let at = |i: i32, j: i32, k: i32| {
        let x = c.x as i32 + i;
        let y = c.y as i32 + j;
        let z = c.z as i32 + k;
        unit(shash((x as u32).wrapping_mul(73856093) ^ (y as u32).wrapping_mul(19349663) ^ (z as u32).wrapping_mul(83492791)))
    };
    let lerp = |a: f64, b: f64, t: f64| a + (b - a) * t;
    let x00 = lerp(at(0, 0, 0), at(1, 0, 0), f.x);
    let x10 = lerp(at(0, 1, 0), at(1, 1, 0), f.x);
    let x01 = lerp(at(0, 0, 1), at(1, 0, 1), f.x);
    let x11 = lerp(at(0, 1, 1), at(1, 1, 1), f.x);
    lerp(lerp(x00, x10, f.y), lerp(x01, x11, f.y), f.z)
}
fn smoothstep(a: f64, b: f64, x: f64) -> f64 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}
/// The cube face (0..6) whose direction dominates `q`, with its normal and in-plane axes.
fn cube_face(q: DVec3) -> (u32, DVec3, DVec3, DVec3) {
    let a = q.abs();
    if a.x >= a.y && a.x >= a.z {
        ((q.x < 0.0) as u32, DVec3::X * q.x.signum(), DVec3::Y, DVec3::Z)
    } else if a.y >= a.z {
        (2 + (q.y < 0.0) as u32, DVec3::Y * q.y.signum(), DVec3::Z, DVec3::X)
    } else {
        (4 + (q.z < 0.0) as u32, DVec3::Z * q.z.signum(), DVec3::X, DVec3::Y)
    }
}

impl Weather {
    pub fn new(clouds: f32) -> Self {
        // A light west-south-westerly, as the smoke always drifted before there was a slider for it.
        Self { clouds, cloud_base: 1500.0, cloud_depth: 4500.0, storms: 0.2, alto: 0.25, snow: 0.0, haze: 1.0, wind: 4.0, wind_from: 248.0, gusts: 0.15, time_rate: 1.0, moon_age: 11.5, cloud_off: DVec3::ZERO, open: false, sun: 0.5, thermal_off: DVec3::ZERO, cells: Vec::new(), live: false, live_station: String::new(), live_raw: String::new(), real_time: false }
    }

    /// Top of the cloud layer including any storm towers, metres.
    pub fn cloud_ceiling(&self) -> f32 {
        let top = self.cloud_base + self.cloud_depth;
        if self.storms > 0.0 { top.max(STORM_TOP).max(top + 1500.0) } else { top }
    }

    /// Storm cells (cumulonimbus) near the camera, nearest first, two vec4s each for
    /// the shaders: centre in cloud space (km on the sphere, planet frame, moving with
    /// the wind like the rest of the cloud field) and tower radius in km; maturity 0..1.
    /// Candidate sites sit on a jittered grid on each face of a cube projected onto
    /// the planet, so they stay put; a hash decides which are active, their size and
    /// age, and a coarse noise makes them cluster.
    pub fn storm_cells(&self, cam: DVec3) -> Vec<[[f32; 4]; 2]> {
        let mut out: Vec<(f64, [[f32; 4]; 2])> = Vec::new();
        let prob_amount = self.storms as f64 * 0.5 * smoothstep(0.12, 0.45, self.clouds as f64);
        if prob_amount <= 0.0 {
            return Vec::new();
        }
        let q = cam.normalize() * 6371.0 + self.cloud_off / 1000.0;
        let reach = (STORM_RANGE / STORM_GRID).ceil() as i32 + 1;
        for face in 0..6u32 {
            let n = [DVec3::X, -DVec3::X, DVec3::Y, -DVec3::Y, DVec3::Z, -DVec3::Z][face as usize];
            if q.normalize().dot(n) < 0.45 {
                continue;
            }
            let (_, _, u, v) = cube_face(n);
            let g = DVec2::new(q.dot(u), q.dot(v)) * (6371.0 / q.dot(n)) / STORM_GRID;
            for j in -reach..=reach {
                for i in -reach..=reach {
                    let (cx, cy) = (g.x.floor() as i32 + i, g.y.floor() as i32 + j);
                    let h0 = shash((cx as u32).wrapping_mul(73856093) ^ (cy as u32).wrapping_mul(19349663) ^ face.wrapping_mul(83492791));
                    let (h1, h2) = (shash(h0), shash(shash(h0)));
                    let jit = DVec2::new(unit(h1), unit(h1 >> 16));
                    let cg = (DVec2::new(cx as f64, cy as f64) + 0.2 + 0.6 * jit) * STORM_GRID;
                    let cen = (n * 6371.0 + u * cg.x + v * cg.y).normalize() * 6371.0;
                    // Each site belongs to exactly one face: no gaps or doubles at the seams.
                    if cube_face(cen).0 != face {
                        continue;
                    }
                    let dist = cen.distance(q);
                    if dist > STORM_RANGE {
                        continue;
                    }
                    let region = value_noise(cen / 320.0 + DVec3::new(0.41, 0.07, 0.66));
                    let prob = prob_amount * smoothstep(0.2, 0.6, region + self.storms as f64 * 0.25);
                    // Fades in rather than pops as the slider or the region changes.
                    let r0 = unit(h0);
                    let on = smoothstep(r0 - 0.04, r0 + 0.04, prob);
                    if on <= 0.0 {
                        continue;
                    }
                    // A cell fading out shrinks into a smaller, younger (lower) tower.
                    let rad = (2.2 + 3.0 * unit(h2)) * (0.6 + 0.4 * on);
                    // Stormier days have more mature cells (with anvils).
                    let u = unit(h2 >> 16);
                    let age = (u + (1.0 - (1.0 - u) * (1.0 - u) - u) * self.storms as f64) * on;
                    out.push((dist, [[cen.x as f32, cen.y as f32, cen.z as f32, rad as f32], [age as f32, 0.0, 0.0, 0.0]]));
                }
            }
        }
        out.sort_by(|a, b| a.0.total_cmp(&b.0));
        out.into_iter().take(MAX_STORMS).map(|c| c.1).collect()
    }

    /// Visibility at sea level through the haze (Koschmieder, green light), km.
    pub fn visibility_km(&self) -> f64 {
        3.912 / (13.558e-6 + (3.996e-6 + 4.4e-6) * self.haze as f64) / 1000.0
    }

    /// The steady wind (where the air moves to) at a point, in the planet frame,
    /// stronger with height above the ground (a 1/7 power law from 10 m).
    pub fn steady(&self, pos: DVec3, agl: f64) -> DVec3 {
        let (e, n, _) = geo::enu(pos);
        let to = (self.wind_from + 180.0).to_radians();
        (e * to.sin() + n * to.cos()) * self.wind * (agl.max(2.0) / 10.0).powf(1.0 / 7.0).min(2.0)
    }

    /// Wind with gusts at a point and time: a few incommensurate swells per
    /// axis, so it never quite repeats; the vertical part is weaker.
    pub fn at(&self, pos: DVec3, agl: f64, t: f64) -> DVec3 {
        let base = self.steady(pos, agl);
        if self.gusts <= 0.0 {
            return base;
        }
        let swell = |k: f64| ((0.53 * t + k).sin() * 0.6 + (1.37 * t + 2.1 * k).sin() * 0.45 + (3.1 * t + 3.7 * k).sin() * 0.3 + (7.9 * t + 5.3 * k).sin() * 0.15) / 0.85;
        let sigma = self.gusts * (1.0 + 0.35 * self.wind);
        let (e, n, u) = geo::enu(pos);
        // Gusts die out in the still air close to the ground.
        let near = (agl / 30.0).clamp(0.2, 1.0);
        base + (e * swell(0.0) + n * swell(1.9) + u * swell(4.3) * 0.5) * sigma * near
    }

    /// Carry the clouds along with the wind aloft (about twice the surface wind),
    /// and the thermals with the wind in the mixed layer below them.
    pub fn advance(&mut self, dt: f64, cam: DVec3) {
        // The cloud field is sampled at (position + offset), so it moves against the offset.
        self.cloud_off -= self.steady(cam, 2000.0) * dt * 1.6;
        self.thermal_off -= self.steady(cam, 300.0) * dt;
        // Keep the offset small so it never costs precision (the thermal lattice repeats nowhere, so a jump is harmless).
        if self.thermal_off.length() > 2.0e5 {
            self.thermal_off = DVec3::ZERO;
        }
    }

    /// How strongly the ground is heating the air: high sun, not overcast. 0..1
    pub fn convection(&self) -> f64 {
        smoothstep(0.1, 0.65, self.sun) * (1.0 - 0.85 * smoothstep(0.6, 0.97, self.clouds as f64))
    }

    /// Top of the convective mixed layer, metres above the ground: cumulus sit on
    /// it; on a clear day it still reaches up a kilometre or two.
    fn mixed_top(&self) -> f64 {
        let clear = 700.0 + 1400.0 * self.convection();
        if self.clouds > 0.1 { (self.cloud_base as f64).min(clear.max(600.0) + 800.0) } else { clear }
    }

    /// Thermals: columns of rising air a few hundred metres across, a kilometre
    /// or two apart, each ringed by sinking air, coming and going over a quarter
    /// of an hour. They sit on a jittered lattice laid over the planet's surface
    /// (altitude dropped, so the columns are vertical) and drift with the wind.
    pub fn thermal(&self, pos: DVec3, agl: f64, t: f64) -> f64 {
        let conv = self.convection();
        let top = self.mixed_top();
        if conv <= 0.01 || agl > top * 1.1 {
            return 0.0;
        }
        const CELL: f64 = 1400.0;
        let p = (pos + self.thermal_off).normalize() * 6.371e6;
        let c = (p / CELL).floor();
        let mut w = 0.0;
        for k in -1..=1 {
            for j in -1..=1 {
                for i in -1..=1 {
                    let (x, y, z) = (c.x as i64 + i, c.y as i64 + j, c.z as i64 + k);
                    let h0 = shash((x as u32).wrapping_mul(73856093) ^ (y as u32).wrapping_mul(19349663) ^ (z as u32).wrapping_mul(83492791) ^ 0x5bd1e995);
                    let (h1, h2) = (shash(h0), shash(h0 ^ 0x68e31da4));
                    // About half the sites are working at any one time, each on its own cycle.
                    let life = (t / (600.0 + 900.0 * unit(h2)) + unit(h1 >> 16)) * std::f64::consts::TAU;
                    let on = life.sin() * 1.4 - 0.35;
                    if on <= 0.0 {
                        continue;
                    }
                    let jit = DVec3::new(unit(h0), unit(h0 >> 16), unit(h1));
                    let cen = ((DVec3::new(x as f64, y as f64, z as f64) + jit) * CELL).normalize() * 6.371e6;
                    let r2 = cen.distance_squared(p);
                    let rad = 110.0 + 200.0 * unit(h2 >> 16);
                    if r2 > (rad * 3.0).powi(2) {
                        continue;
                    }
                    let s = r2 / (rad * rad);
                    // Lift in the core, a ring of sink around it (Allen's profile, simplified).
                    w += (1.6 + 2.6 * unit(h1)) * on.min(1.0) * (-s).exp() * (1.0 - 0.75 * s);
                }
            }
        }
        // Weak to start with near the ground, strongest through the middle of the
        // layer, petering out at its top; the air between the thermals sinks a little.
        let z = (agl / top).max(0.0);
        let shape = (z * 8.0).min(1.0).powf(0.5) * (1.0 - smoothstep(0.75, 1.1, z));
        (w * shape - 0.35 * (1.0 - smoothstep(0.9, 1.1, z))) * conv
    }

    /// The turbulence and vertical air motion at a point: wind tumbling over the
    /// ground, convection on a sunny day, the inside of the cloud layer, and the
    /// up- and downdrafts of storm cells. `water` damps the thermals.
    pub fn air(&self, pos: DVec3, agl: f64, t: f64, water: bool) -> Air {
        let agl = agl.max(1.0);
        let ft = agl * 3.28084;
        // Mechanical: about a tenth of the wind near the ground (MIL-F-8785C),
        // more on a gusty day, dying out above the surface layer.
        let w20 = self.wind * 0.93;
        let mech = (0.1 + 0.2 * self.gusts) * w20 * (1.0 - 0.65 * smoothstep(250.0, 900.0, agl));
        // Convective eddies through the mixed layer.
        let conv = self.convection() * if water { 0.25 } else { 1.0 };
        let top = self.mixed_top();
        let convt = 1.1 * conv * (1.0 - smoothstep(0.85 * top, 1.15 * top, agl)) * smoothstep(0.0, 60.0, agl).max(0.3);
        // In the cloud layer itself (strongest where it is cumulus-like rather than a flat deck).
        let (_, _, alt) = geo::pos_to_lla(pos);
        let (base, ctop) = (self.cloud_base as f64, (self.cloud_base + self.cloud_depth) as f64);
        let inside = smoothstep(base - 150.0, base + 100.0, alt) * (1.0 - smoothstep(ctop - 100.0, ctop + 200.0, alt));
        let cloud = inside * self.clouds as f64 * (0.7 + 0.8 * self.convection() + 0.6 * self.storms as f64);
        // A light background chop everywhere, as there always is a little.
        let background = 0.12 + 0.03 * self.wind;

        // Storm cells: severe turbulence in and around a tower, an updraft in its
        // heart that pulls in air from below the base, and a downdraft (the rain
        // shaft) on its flank that spreads out over the ground.
        let mut storm = 0.0f64;
        let mut draft = 0.0;
        if !self.cells.is_empty() {
            let q = pos.normalize() * 6371.0 + self.cloud_off / 1000.0;
            for c in &self.cells {
                let cen = DVec3::new(c[0][0] as f64, c[0][1] as f64, c[0][2] as f64);
                let (rad, age) = (c[0][3] as f64, c[1][0] as f64);
                let d = cen.distance(q.normalize() * 6371.0);
                if d > rad * 3.0 {
                    continue;
                }
                let tower_top = base + (STORM_TOP as f64 - base) * age.min(1.0).sqrt();
                let up = if alt < base { smoothstep(0.0, base, alt) * 0.45 } else { 1.0 - smoothstep(tower_top - 1500.0, tower_top + 500.0, alt) };
                let core = (-(d / (rad * 0.55)).powi(2)).exp();
                let shaft = (-((d - rad * 1.05) / (rad * 0.35)).powi(2)).exp();
                draft += age * (14.0 * core * up - 9.0 * shaft * (1.0 - smoothstep(base, tower_top, alt) * 0.6) * smoothstep(0.0, 200.0, agl));
                storm = storm.max(age * 7.0 * (1.0 - smoothstep(rad * 0.8, rad * 2.6, d)) * (0.35 + 0.65 * up.max(inside)));
            }
        }

        let sigma_w = (mech * mech + convt * convt + cloud * cloud + storm * storm + background * background).sqrt().min(9.0);
        // Near the ground the eddies are as small as the height and the horizontal
        // gusts larger than the vertical; aloft they are a few hundred metres across.
        let k = 0.177 + 0.000823 * ft.min(1000.0);
        let len_w = agl.clamp(12.0, 530.0);
        let len_h = (agl / k.powf(1.2)).clamp(40.0, 530.0);
        let sigma_h = sigma_w / k.powf(0.4).max(0.5).min(1.0);
        Air { sigma_w, sigma_h, len_w, len_h, draft: self.thermal(pos, agl, t) * if water { 0.2 } else { 1.0 } + draft }
    }

    /// The floating weather & time window (F3). `utc` and `doy` belong to the
    /// game clock; `lon` (degrees) sets local time.
    pub fn panel(&mut self, ctx: &egui::Context, utc: &mut f64, doy: &mut f64, lon: f64) {
        let mut open = self.open;
        egui::Window::new(t!("Weather & time  (F3)", "Météo et heure  (F3)"))
            .id(egui::Id::unique("weather_window"))
            .open(&mut open)
            .collapsible(true)
            .resizable(false)
            .default_pos([14.0, 250.0])
            .show(ctx, |ui| {
                ui.spacing_mut().slider_width = 200.0;
                self.controls(ui, utc, doy, lon, false)
            });
        self.open = open;
    }

    /// The sliders and presets, for the F3 window (one column) and the
    /// in-flight menu (two).
    pub fn controls(&mut self, ui: &mut egui::Ui, utc: &mut f64, doy: &mut f64, lon: f64, two_columns: bool) {
        self.live_switch(ui);
        let free = !self.live;
        if two_columns {
            ui.columns(2, |c| {
                c[0].spacing_mut().slider_width = 115.0;
                c[1].spacing_mut().slider_width = 115.0;
                self.day_controls(&mut c[0], utc, doy, lon);
                c[0].add_space(8.0);
                c[0].add_enabled_ui(free, |ui| self.wind_controls(ui));
                c[1].add_enabled_ui(free, |ui| {
                    self.sky_controls(ui);
                    ui.add_space(8.0);
                    self.presets(ui);
                });
            });
        } else {
            self.day_controls(ui, utc, doy, lon);
            ui.add_enabled_ui(free, |ui| {
                self.sky_controls(ui);
                self.wind_controls(ui);
                ui.add_space(6.0);
                self.presets(ui);
            });
        }
    }

    /// The real-weather switch, and where the report comes from.
    fn live_switch(&mut self, ui: &mut egui::Ui) {
        ui.checkbox(&mut self.live, t!("Real weather (METAR, else Open-Meteo)", "Météo réelle (METAR, sinon Open-Meteo)"));
        if self.live {
            let text = if self.live_station.is_empty() {
                if self.live_raw.is_empty() { t!("Waiting for a report…", "En attente d'un bulletin…").to_string() } else { self.live_raw.clone() }
            } else if self.live_raw.is_empty() {
                t!("From {} (weather model)", "Selon {} (modèle météo)", self.live_station)
            } else {
                self.live_raw.clone()
            };
            ui.label(RichText::new(text).monospace().size(11.0).weak());
        }
    }

    /// The start screen's version: local time and date at the chosen spot
    /// (the clock is set from them on take-off), then the sky and the wind.
    /// The start screen's weather: presets, sky and wind.
    pub fn start_weather(&mut self, ui: &mut egui::Ui) {
        self.live_switch(ui);
        ui.add_enabled_ui(!self.live, |ui| {
            self.presets(ui);
            ui.add_space(4.0);
            self.sky_controls(ui);
            self.wind_controls(ui);
        });
    }

    /// The start screen's time: local time of day and the date.
    pub fn start_time(ui: &mut egui::Ui, hour: &mut f64, doy: &mut f64, real_time: &mut bool) {
        real_time_switch(ui, real_time);
        let free = !*real_time;
        egui::Grid::new("start_time").num_columns(2).min_col_width(92.0).spacing([10.0, 5.0]).show(ui, |ui| {
            ui.label(t!("Local time", "Heure locale"));
            ui.add_enabled_ui(free, |ui| add(ui, Slider::new(hour, 0.0..=23.99).custom_formatter(|v, _| crate::menu::hhmm(v))));
            ui.end_row();
            ui.label("Date");
            ui.add_enabled_ui(free, |ui| add(ui, Slider::new(doy, 0.0..=364.0).step_by(1.0).custom_formatter(|v, _| date_label(v))));
            ui.end_row();
        });
    }

    /// How hard it is snowing, 0..1: snow with an overcast sky, below the
    /// clouds (and a little way into them).
    pub fn snowfall(&self, alt: f64) -> f32 {
        let below = ((self.cloud_base as f64 + 300.0 - alt) / 300.0).clamp(0.0, 1.0) as f32;
        self.snow * ((self.clouds - 0.4) / 0.4).clamp(0.0, 1.0) * below
    }

    /// A few words for the weather: the preset it matches, else the cloud
    /// cover and wind.
    pub fn summary(&self) -> String {
        if self.live {
            return if self.live_station.is_empty() { t!("Real", "Réelle").to_string() } else { t!("Real · {}", "Réelle · {}", self.live_station) };
        }
        let near = |a: f64, b: f64| (a - b).abs() < 1e-3 * (1.0 + b.abs());
        for &(name, c, b, d, s, a, h, w, g) in PRESETS {
            if near(self.clouds as f64, c as f64) && near(self.cloud_base as f64, b as f64) && near(self.cloud_depth as f64, d as f64) && near(self.storms as f64, s as f64) && near(self.alto as f64, a as f64) && near(self.haze as f64, h as f64) && near(self.wind as f64, w as f64) && near(self.gusts as f64, g as f64) {
                return tr(name).to_string();
            }
        }
        t!("{:.0}% cloud · {:.0} kt", "{:.0}% de nuages · {:.0} kt", self.clouds * 100.0, self.wind * 1.94384)
    }

    fn day_controls(&mut self, ui: &mut egui::Ui, utc: &mut f64, doy: &mut f64, lon: f64) {
        real_time_switch(ui, &mut self.real_time);
        let free = !self.real_time;
        section(ui, tr("DAY"), |ui| {
            ui.label(t!("Local time", "Heure locale"));
            let mut local = (*utc + lon / 15.0).rem_euclid(24.0);
            let hhmm = |v: f64, _: std::ops::RangeInclusive<usize>| format!("{:02}:{:02}", v as u32 % 24, (v.fract() * 60.0) as u32);
            if ui.add_enabled_ui(free, |ui| add(ui, Slider::new(&mut local, 0.0..=23.99).custom_formatter(hhmm))).inner.changed() {
                *utc = (local - lon / 15.0).rem_euclid(24.0);
            }
            ui.end_row();
            ui.label("Date");
            let date = |v: f64, _: std::ops::RangeInclusive<usize>| date_label(v);
            ui.add_enabled_ui(free, |ui| add(ui, Slider::new(doy, 0.0..=364.0).step_by(1.0).custom_formatter(date)));
            ui.end_row();
            ui.label(t!("Clock speed", "Vitesse de l'horloge"));
            let rate = &mut self.time_rate;
            ui.add_enabled_ui(free, |ui| add(ui, Slider::new(rate, 0.0..=3600.0).logarithmic(true).smallest_positive(1.0).custom_formatter(|v, _| if v == 0.0 { t!("stopped", "arrêtée").into() } else { format!("{v:.0}×") })));
            ui.end_row();
            ui.label(t!("Moon", "Lune"));
            add(ui, Slider::new(&mut self.moon_age, 0.0..=29.5).custom_formatter(|v, _| tr(moon_phase(v)).into()));
            ui.end_row();
        });
    }

    fn sky_controls(&mut self, ui: &mut egui::Ui) {
        section(ui, tr("SKY"), |ui| {
            ui.label(t!("Cloud cover", "Couverture nuageuse"));
            add(ui, Slider::new(&mut self.clouds, 0.0..=1.0).custom_formatter(|v, _| format!("{:.0}%", v * 100.0)));
            ui.end_row();
            ui.label(t!("Cloud base", "Base des nuages"));
            add(ui, Slider::new(&mut self.cloud_base, 300.0..=4000.0).step_by(50.0).suffix(" m"));
            ui.end_row();
            ui.label(t!("Cloud depth", "Épaisseur des nuages"));
            add(ui, Slider::new(&mut self.cloud_depth, 500.0..=8000.0).step_by(50.0).suffix(" m"));
            ui.end_row();
            ui.label(t!("Storm cells", "Cellules orageuses"));
            add(ui, Slider::new(&mut self.storms, 0.0..=1.0).custom_formatter(|v, _| format!("{:.0}%", v * 100.0)));
            ui.end_row();
            ui.label(t!("Mid-level cloud", "Nuages moyens"));
            add(ui, Slider::new(&mut self.alto, 0.0..=1.0).custom_formatter(|v, _| format!("{:.0}%", v * 100.0)));
            ui.end_row();
            ui.label(t!("Snow", "Neige"));
            add(ui, Slider::new(&mut self.snow, 0.0..=1.0).custom_formatter(|v, _| if v <= 0.0 { t!("none", "aucune").to_string() } else { format!("{:.0}%", v * 100.0) }));
            ui.end_row();
            ui.label(t!("Haze", "Brume"));
            add(ui, Slider::new(&mut self.haze, 0.2..=40.0).logarithmic(true).custom_formatter(|v, _| format!("{v:.1}×")));
            ui.end_row();
            ui.label("");
            ui.label(RichText::new(t!("visibility ~{:.0} km", "visibilité ~{:.0} km", self.visibility_km())).size(11.5).weak());
            ui.end_row();
        });
    }

    fn wind_controls(&mut self, ui: &mut egui::Ui) {
        section(ui, tr("WIND"), |ui| {
            ui.label(t!("Speed", "Vitesse"));
            let mut kt = self.wind * 1.94384;
            if add(ui, Slider::new(&mut kt, 0.0..=50.0).fixed_decimals(0).suffix(" kt")).changed() {
                self.wind = kt / 1.94384;
            }
            ui.end_row();
            ui.label(t!("From", "Direction"));
            add(ui, Slider::new(&mut self.wind_from, 0.0..=359.0).step_by(1.0).custom_formatter(|v, _| format!("{:03.0}° {}", v, tr(compass(v)))));
            ui.end_row();
            ui.label(t!("Gusts", "Rafales"));
            add(ui, Slider::new(&mut self.gusts, 0.0..=1.0).custom_formatter(|v, _| format!("{:.0}%", v * 100.0)));
            ui.end_row();
        });
    }

    fn presets(&mut self, ui: &mut egui::Ui) {
        crate::menu::heading(ui, tr("PRESETS"));
        ui.horizontal_wrapped(|ui| {
            ui.spacing_mut().item_spacing = egui::vec2(5.0, 5.0);
            for &(name, c, b, d, s, a, h, w, g) in PRESETS {
                if ui.button(tr(name)).clicked() {
                    (self.clouds, self.cloud_base, self.cloud_depth, self.storms, self.alto, self.haze, self.wind, self.gusts) = (c, b, d, s, a, h, w, g);
                }
            }
        });
    }
}

use crate::i18n::{t, tr};
use egui::{RichText, Slider};

/// The "real time" switch above the clock's sliders.
fn real_time_switch(ui: &mut egui::Ui, on: &mut bool) {
    ui.checkbox(on, t!("Real time (the sun where you are, now)", "Heure réelle (le soleil là où vous êtes, maintenant)"));
}

/// The real time now: hours UTC and the day of the year (0 = 1 January,
/// the 29th of February counted as the 28th), from the system clock.
pub fn real_clock() -> (f64, f64) {
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0);
    clock_at(secs)
}

fn clock_at(secs: f64) -> (f64, f64) {
    let days = (secs / 86_400.0).floor();
    let hours = (secs - days * 86_400.0) / 3600.0;
    // Days since 1970-01-01 to a civil date (Howard Hinnant's algorithm).
    let z = days as i64 + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let dy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * dy + 2) / 153;
    let d = dy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    const BEFORE: [i64; 12] = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
    let doy = (BEFORE[(m - 1) as usize] + d.min(if m == 2 { 28 } else { 31 }) - 1).min(364);
    (hours, doy as f64)
}

/// "Jun 21", or "21 juin" in French.
pub fn date_label(doy: f64) -> String {
    if !crate::i18n::is_fr() {
        return month_day(doy);
    }
    const M: [&str; 12] = ["janv.", "févr.", "mars", "avr.", "mai", "juin", "juil.", "août", "sept.", "oct.", "nov.", "déc."];
    const LEN: [u32; 12] = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    let mut d = doy.max(0.0) as u32 % 365;
    for (name, len) in M.iter().zip(LEN) {
        if d < len {
            return format!("{} {}", d + 1, name);
        }
        d -= len;
    }
    "31 déc.".into()
}

/// A headed two-column grid of labelled sliders.
fn section(ui: &mut egui::Ui, title: &str, rows: impl FnOnce(&mut egui::Ui)) {
    crate::menu::heading(ui, title);
    egui::Grid::new(("wx", title)).num_columns(2).min_col_width(92.0).spacing([10.0, 5.0]).show(ui, rows);
}

/// A focused slider would also take the arrow keys that fly the aircraft, so it lets go.
fn add(ui: &mut egui::Ui, s: Slider) -> egui::Response {
    let r = ui.add(s);
    if r.drag_stopped() || r.clicked() || r.lost_focus() {
        r.surrender_focus();
    }
    r
}

pub fn month_day(doy: f64) -> String {
    const M: [(&str, u32); 12] = [("Jan", 31), ("Feb", 28), ("Mar", 31), ("Apr", 30), ("May", 31), ("Jun", 30), ("Jul", 31), ("Aug", 31), ("Sep", 30), ("Oct", 31), ("Nov", 30), ("Dec", 31)];
    let mut d = doy.max(0.0) as u32 % 365;
    for (name, len) in M {
        if d < len {
            return format!("{} {}", name, d + 1);
        }
        d -= len;
    }
    "Dec 31".into()
}

fn moon_phase(age: f64) -> &'static str {
    match (age / 29.53 * 8.0 + 0.5) as u32 % 8 {
        0 => "new",
        1 => "waxing crescent",
        2 => "first quarter",
        3 => "waxing gibbous",
        4 => "full",
        5 => "waning gibbous",
        6 => "last quarter",
        _ => "waning crescent",
    }
}

fn compass(deg: f64) -> &'static str {
    const P: [&str; 8] = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"];
    P[((deg / 45.0 + 0.5) as usize) % 8]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn labels() {
        assert_eq!(month_day(0.0), "Jan 1");
        assert_eq!(month_day(171.0), "Jun 21");
        assert_eq!(month_day(364.0), "Dec 31");
        assert_eq!(moon_phase(0.0), "new");
        assert_eq!(moon_phase(14.8), "full");
        assert_eq!(compass(248.0), "W");
        assert_eq!(compass(350.0), "N");
        // 2026-11-15 17:20 UTC, 2024-02-29 (leap day) and 2024-12-31 noon
        let (h, d) = clock_at(1_794_763_200.0);
        assert_eq!((month_day(d).as_str(), (h * 60.0).round()), ("Nov 15", 17.0 * 60.0 + 20.0));
        assert_eq!(month_day(clock_at(1_709_208_000.0).1), "Feb 28");
        assert_eq!(month_day(clock_at(1_735_646_400.0).1), "Dec 31");
        assert_eq!(clock_at(0.0), (0.0, 0.0));
    }

    #[test]
    fn wind_blows_from_its_direction() {
        let mut w = Weather::new(0.5);
        w.gusts = 0.0;
        (w.wind, w.wind_from) = (10.0, 270.0);
        let p = geo::lla_to_pos(0.6, -2.1, 100.0);
        let (e, _, _) = geo::enu(p);
        let v = w.at(p, 10.0, 3.0);
        // A westerly blows toward the east, at the stated speed 10 m up.
        assert!((v.dot(e) - 10.0).abs() < 1e-6, "{v:?}");
        assert!(w.steady(p, 300.0).length() > 10.0);
        assert!((w.visibility_km() - 3.912 / (13.558e-6 + 8.396e-6) / 1000.0).abs() < 1e-9);
    }

    #[test]
    fn turbulence_follows_the_weather() {
        let p = geo::lla_to_pos(0.6, -2.1, 300.0);
        let mut w = Weather::new(0.3);
        (w.wind, w.gusts, w.sun) = (2.0, 0.0, -0.2);
        let calm = w.air(p, 300.0, 0.0, false);
        assert!(calm.sigma_w < 0.6, "{calm:?}");
        assert_eq!(w.thermal(p, 300.0, 0.0), 0.0);
        (w.wind, w.gusts) = (14.0, 0.7);
        let windy = w.air(geo::lla_to_pos(0.6, -2.1, 80.0), 80.0, 0.0, false);
        assert!(windy.sigma_w > 1.5 && windy.sigma_h > windy.sigma_w, "{windy:?}");
        // A sunny day: somewhere along a line there is lift and somewhere sink.
        w.sun = 0.8;
        let (e, _, _) = geo::enu(p);
        let ws: Vec<f64> = (0..400).map(|i| w.thermal(p + e * (i as f64 * 25.0), 500.0, 0.0)).collect();
        let (lo, hi) = ws.iter().fold((0.0f64, 0.0f64), |(a, b), &x| (a.min(x), b.max(x)));
        assert!(hi > 1.0 && lo < -0.2, "{lo} {hi}");
    }
}

