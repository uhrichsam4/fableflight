//! Main menu: the planet itself is the map. Spin it, zoom in, click where
//! you want to start.
use crate::geo;
use glam::DVec3;

/// (name, latitude, longitude, tier). Tier 1 is always labelled, tier 2 once zoomed in a little.
pub const CITIES: &[(&str, f64, f64, u8)] = &[
    ("San Francisco", 37.7749, -122.4194, 1),
    ("Los Angeles", 34.0522, -118.2437, 1),
    ("New York", 40.7128, -74.0060, 1),
    ("Chicago", 41.8781, -87.6298, 2),
    ("Seattle", 47.6062, -122.3321, 2),
    ("Vancouver", 49.2827, -123.1207, 2),
    ("Denver", 39.7392, -104.9903, 2),
    ("Miami", 25.7617, -80.1918, 2),
    ("Toronto", 43.6532, -79.3832, 2),
    ("Mexico City", 19.4326, -99.1332, 1),
    ("Havana", 23.1136, -82.3666, 2),
    ("Bogota", 4.7110, -74.0721, 2),
    ("Lima", -12.0464, -77.0428, 2),
    ("Rio de Janeiro", -22.9068, -43.1729, 1),
    ("Buenos Aires", -34.6037, -58.3816, 1),
    ("Santiago", -33.4489, -70.6693, 2),
    ("Honolulu", 21.3069, -157.8583, 1),
    ("Anchorage", 61.2181, -149.9003, 2),
    ("Reykjavik", 64.1466, -21.9426, 2),
    ("London", 51.5074, -0.1278, 1),
    ("Paris", 48.8566, 2.3522, 1),
    ("Madrid", 40.4168, -3.7038, 2),
    ("Lisbon", 38.7223, -9.1393, 2),
    ("Rome", 41.9028, 12.4964, 1),
    ("Berlin", 52.5200, 13.4050, 2),
    ("Amsterdam", 52.3676, 4.9041, 2),
    ("Zermatt", 46.0207, 7.7491, 2),
    ("Venice", 45.4408, 12.3155, 2),
    ("Athens", 37.9838, 23.7275, 2),
    ("Istanbul", 41.0082, 28.9784, 1),
    ("Stockholm", 59.3293, 18.0686, 2),
    ("Oslo", 59.9139, 10.7522, 2),
    ("Moscow", 55.7558, 37.6173, 1),
    ("Cairo", 30.0444, 31.2357, 1),
    ("Marrakesh", 31.6295, -7.9811, 2),
    ("Lagos", 6.5244, 3.3792, 2),
    ("Nairobi", -1.2921, 36.8219, 2),
    ("Cape Town", -33.9249, 18.4241, 1),
    ("Dubai", 25.2048, 55.2708, 1),
    ("Tehran", 35.6892, 51.3890, 2),
    ("Mumbai", 19.0760, 72.8777, 1),
    ("Delhi", 28.6139, 77.2090, 2),
    ("Kathmandu", 27.7172, 85.3240, 2),
    ("Bangkok", 13.7563, 100.5018, 2),
    ("Singapore", 1.3521, 103.8198, 1),
    ("Hong Kong", 22.3193, 114.1694, 1),
    ("Shanghai", 31.2304, 121.4737, 2),
    ("Beijing", 39.9042, 116.4074, 1),
    ("Seoul", 37.5665, 126.9780, 2),
    ("Tokyo", 35.6762, 139.6503, 1),
    ("Manila", 14.5995, 120.9842, 2),
    ("Jakarta", -6.2088, 106.8456, 2),
    ("Perth", -31.9523, 115.8613, 2),
    ("Sydney", -33.8688, 151.2093, 1),
    ("Melbourne", -37.8136, 144.9631, 2),
    ("Auckland", -36.8509, 174.7645, 2),
    ("Queenstown", -45.0312, 168.6626, 2),
];

/// Places offered as one-click starts.
pub const QUICK: &[(&str, f64, f64)] = &[
    ("Golden Gate", 37.8085, -122.4650),
    ("Manhattan", 40.7580, -73.9855),
    ("Yosemite Valley", 37.7340, -119.6000),
    ("Matterhorn", 45.9950, 7.7000),
    ("Paris", 48.8584, 2.2945),
    ("Rio de Janeiro", -22.9519, -43.2105),
    ("Hong Kong", 22.2930, 114.1694),
    ("Sydney Harbour", -33.8523, 151.2108),
    ("Dubai", 25.1972, 55.2744),
    ("Black Forest", 48.0500, 8.1500),
];

/// Airports offered as one-click starts: big and small, flat and mountain, beach and city.
pub const AIRPORTS: &[(&str, &str)] = &[
    ("KSFO", "San Francisco"),
    ("KJFK", "New York JFK"),
    ("KLAX", "Los Angeles"),
    ("EGLL", "London Heathrow"),
    ("LFPG", "Paris CDG"),
    ("LSZH", "Zürich"),
    ("OMDB", "Dubai"),
    ("VHHH", "Hong Kong"),
    ("RJTT", "Tokyo Haneda"),
    ("YSSY", "Sydney"),
    ("NZQN", "Queenstown"),
    ("TNCM", "St Maarten"),
    ("LXGB", "Gibraltar"),
    ("LFLJ", "Courchevel"),
    ("KHAF", "Half Moon Bay"),
    ("KOSH", "Oshkosh"),
];

pub struct Menu {
    /// what the camera is looking straight down at, degrees
    pub lat: f64,
    pub lon: f64,
    pub alt: f64,
    target_alt: f64,
    pub pick: Option<(f64, f64)>,
    pub pick_name: String,
    pub search: String,
    pub hour: f64,
    pub press: Option<(f64, f64)>,
    pub dragged: f64,
    pub starting: bool,
    /// the controls list beside the start panel
    pub show_controls: bool,
    spin: f64,
    /// the chosen spot is this airport, and how to start there
    pub airport: Option<u32>,
    pub start: Start,
    /// runway end to start on (its ident); None takes the one into the wind
    pub runway: Option<String>,
    /// height above the ground for a start in the air (m)
    pub start_agl: f64,
    /// airport markers drawn on the globe last frame: (index, x, y in physical pixels)
    pub markers: Vec<(u32, f32, f32)>,
    /// which aircraft to fly (index into `aircraft::ALL`)
    pub aircraft: usize,
}

/// How to start at an airport chosen on the globe.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Start {
    /// at rest on the runway, lined up for take-off
    Runway,
    /// on a 3° final approach to it
    Approach,
    /// in the air above it, at the chosen height
    Overhead,
}

impl Menu {
    pub fn new(lat: f64, lon: f64, hour: f64) -> Self {
        Self { lat, lon, alt: 1.5e7, target_alt: 1.1e7, pick: None, pick_name: String::new(), search: String::new(), hour, press: None, dragged: 0.0, starting: false, show_controls: false, spin: 1.0, airport: None, start: Start::Runway, runway: None, start_agl: 450.0, markers: Vec::new(), aircraft: 0 }
    }

    pub fn look_at(&mut self, lat: f64, lon: f64, alt: f64) {
        (self.lat, self.lon, self.target_alt) = (lat, lon, alt);
        self.alt = self.alt.max(alt);
        self.spin = 0.0;
    }

    /// Eye position and the view basis (direction, up).
    pub fn camera(&self) -> (DVec3, DVec3, DVec3) {
        let cam = geo::lla_to_pos(self.lat.to_radians(), self.lon.to_radians(), self.alt);
        let (_, n, u) = geo::enu(cam);
        (cam, -u, n)
    }

    pub fn update(&mut self, dt: f64) {
        self.alt += (self.target_alt - self.alt) * (dt * 6.0).min(1.0);
        // The globe drifts slowly until you touch it.
        if self.spin > 0.0 && self.press.is_none() {
            self.lon = (self.lon + dt * 2.0 * self.spin + 180.0).rem_euclid(360.0) - 180.0;
        }
    }

    pub fn zoom(&mut self, d: f64) {
        self.target_alt = (self.target_alt * 0.82f64.powf(d)).clamp(900.0, 2.6e7);
        self.spin = 0.0;
    }

    /// Metres on the ground per screen pixel, at the centre of the view.
    fn metres_per_pixel(&self, h: f64, fov: f64) -> f64 {
        self.alt.min(geo::R * 1.1) * 2.0 * (fov * 0.5).tan() / h
    }

    pub fn drag(&mut self, dx: f64, dy: f64, h: f64, fov: f64) {
        let mpp = self.metres_per_pixel(h, fov);
        self.lat = (self.lat + (dy * mpp / geo::R).to_degrees()).clamp(-84.0, 84.0);
        self.lon -= (dx * mpp / (geo::R * self.lat.to_radians().cos().max(0.15))).to_degrees();
        self.lon = (self.lon + 180.0).rem_euclid(360.0) - 180.0;
        self.dragged += dx.abs() + dy.abs();
        self.spin = 0.0;
    }

    /// The point on the globe under a screen position, if any.
    pub fn pick_at(&self, x: f64, y: f64, w: f64, h: f64, fov: f64) -> Option<(f64, f64)> {
        let (cam, dir, up) = self.camera();
        let right = dir.cross(up);
        let t = (fov * 0.5).tan();
        let d = (dir + right * ((x / w * 2.0 - 1.0) * t * w / h) + up * ((1.0 - y / h * 2.0) * t)).normalize();
        let b = cam.dot(d);
        let c = cam.length_squared() - geo::R * geo::R;
        let disc = b * b - c;
        if disc < 0.0 {
            return None;
        }
        let hit = cam + d * (-b - disc.sqrt());
        let (lat, lon, _) = geo::pos_to_lla(hit);
        Some((lat.to_degrees(), lon.to_degrees()))
    }

    /// The airport marker under a screen position (physical pixels), if any.
    pub fn marker_at(&self, x: f64, y: f64) -> Option<u32> {
        self.marker_near(x, y, 14.0)
    }

    /// The airport marker nearest a screen position, within `r` pixels.
    pub fn marker_near(&self, x: f64, y: f64, r: f64) -> Option<u32> {
        self.markers.iter().map(|&(i, mx, my)| (i, (mx as f64 - x).hypot(my as f64 - y))).filter(|m| m.1 < r).min_by(|a, b| a.1.total_cmp(&b.1)).map(|m| m.0)
    }

    /// Nearest listed city, for naming a picked spot before the lookup returns.
    pub fn near_city(lat: f64, lon: f64) -> Option<(&'static str, f64)> {
        let p = geo::lla_to_pos(lat.to_radians(), lon.to_radians(), 0.0);
        CITIES.iter().map(|c| (c.0, (geo::lla_to_pos(c.1.to_radians(), c.2.to_radians(), 0.0) - p).length())).min_by(|a, b| a.1.total_cmp(&b.1))
    }
}

// ---- shared look and widgets for the start screen and the in-flight menu

use crate::i18n::tr;
use crate::padui::{self, Hint};
use egui::{Color32, RichText, Stroke};

pub const ACCENT: Color32 = Color32::from_rgb(255, 196, 64);
/// dark text on the accent colour
pub const ON_ACCENT: Color32 = Color32::from_rgb(24, 18, 4);
pub const TEXT: Color32 = Color32::from_rgb(232, 235, 240);
pub const DIM: Color32 = Color32::from_rgb(150, 158, 170);
pub const PANEL: Color32 = Color32::from_rgba_premultiplied(9, 12, 17, 228);
/// the see-through glass of the flight panel, over the running game
pub const GLASS: Color32 = Color32::from_rgba_premultiplied(5, 8, 12, 140);

/// Pages of the in-flight panels: the flight panel (Tab) runs beside the
/// game; the pause menu (Esc) stops it.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Page {
    Flight,
    Places,
    Weather,
    Game,
    Route,
    Settings,
    Graphics,
    Controls,
}

impl Page {
    pub const INFO: [Page; 3] = [Page::Flight, Page::Places, Page::Weather];
    pub const PAUSE: [Page; 5] = [Page::Game, Page::Route, Page::Settings, Page::Graphics, Page::Controls];

    pub fn name(self) -> &'static str {
        match self {
            Page::Flight => "Flight",
            Page::Places => "Go to",
            Page::Weather => "Weather",
            Page::Game => "Game",
            Page::Route => "Route",
            Page::Settings => "Settings",
            Page::Graphics => "Graphics",
            Page::Controls => "Controls",
        }
    }

    /// The name shown on its tab, in the interface language (`name` stays
    /// English: the command line's `--pause-menu` goes by it).
    pub fn label(self) -> &'static str {
        crate::i18n::tr(self.name())
    }

    /// on the flight panel (Tab), not the pause menu
    pub fn is_info(self) -> bool {
        Page::INFO.contains(&self)
    }

    /// The page before or after this one in a set, round and round.
    pub fn cycle(set: &[Page], p: Page, next: bool) -> Page {
        let i = set.iter().position(|q| *q == p).unwrap_or(0);
        set[(i + if next { 1 } else { set.len() - 1 }) % set.len()]
    }

    pub fn parse(s: &str) -> Option<Page> {
        let s = s.to_lowercase();
        match s.as_str() {
            "pause" | "menu" => return Some(Page::Game),
            "airports" | "places" => return Some(Page::Places),
            _ => {}
        }
        Page::INFO.into_iter().chain(Page::PAUSE).find(|p| p.name().to_lowercase().starts_with(&s))
    }
}

/// Keyboard and mouse: (keys, what they do). Shown on F1 and the Controls page.
pub const CONTROLS: &[(&str, &[(&str, &str)])] = &[
    ("FLYING", &[
        ("W / S", "pitch down / up"),
        ("A / D", "roll left / right"),
        ("Q / E", "rudder left / right"),
        ("Shift / Ctrl", "throttle up / down"),
        ("Space", "boost"),
        ("B", "wheel brakes"),
        ("F / V", "flaps down / up a notch"),
        ("L", "landing gear down / up"),
        ("/", "speedbrake (airliners)"),
        ("Z", "thrust reversers, on the ground (hold)"),
        ("K", "set the heading indicator to the compass"),
        ("J", "autopilot: fly to the next route waypoint"),
    ]),
    ("VIEW", &[
        ("C", "chase / cockpit camera"),
        ("Mouse drag", "look around"),
        ("Mouse wheel", "camera distance / cockpit zoom"),
        ("H", "hide / show the HUD"),
        ("M", "large map (wheel zoom, drag pan)"),
        ("F12", "screenshot"),
    ]),
    ("WORLD", &[
        ("T / Y", "time of day forward / back"),
        ("N / X", "more / fewer clouds"),
        ("F3", "weather & time window"),
        ("[ / ]", "sim speed slower / faster (1× to 8×)"),
    ]),
    ("GAME", &[
        ("Tab", "flight panel: instruments, places, weather"),
        ("Esc", "pause menu: settings, globe, quit"),
        ("P", "pause without the menu"),
        ("G", "go to a place or address"),
        ("R", "respawn in the air"),
        ("U", "sound on / off"),
        ("F2", "anti-aliasing (TAA) on / off"),
        ("F4", "frame rate counter"),
        ("F11", "fullscreen on / off"),
        ("F1", "quick key reference"),
    ]),
    ("GAMEPAD (MSFS 2020 layout)", &[
        ("Left stick", "pitch and roll"),
        ("LT / RT", "rudder left / right"),
        ("A / B", "throttle up / down (hold)"),
        ("X", "wheel brakes (hold)"),
        ("Y + B", "parking brake on / off"),
        ("Y + D-pad up / dn", "trim nose down / up"),
        ("LB / RB", "flaps up / down a notch"),
        ("Left stick click", "landing gear"),
        ("Right stick", "look around"),
        ("Right stick click", "reset the view"),
        ("D-pad", "quick views"),
        ("View", "chase / cockpit camera"),
        ("Menu", "pause menu (B closes it)"),
        ("Y + View", "flight panel"),
        ("A after a crash", "fly again"),
]),
];

/// The one look shared by every panel: dark glass, amber accent, roomy widgets.
pub fn apply_theme(ctx: &egui::Context) {
    ctx.set_theme(egui::Theme::Dark);
    ctx.all_styles_mut(|s| {
        use egui::{FontFamily, FontId, TextStyle};
        s.text_styles = [
            (TextStyle::Small, FontId::new(11.0, FontFamily::Proportional)),
            (TextStyle::Body, FontId::new(14.0, FontFamily::Proportional)),
            (TextStyle::Button, FontId::new(14.0, FontFamily::Proportional)),
            (TextStyle::Heading, FontId::new(20.0, FontFamily::Proportional)),
            (TextStyle::Monospace, FontId::new(13.0, FontFamily::Monospace)),
        ]
        .into();
        s.spacing.item_spacing = egui::vec2(8.0, 6.0);
        s.spacing.button_padding = egui::vec2(10.0, 5.0);
        s.spacing.slider_width = 200.0;
        s.spacing.interact_size.y = 24.0;
        let v = &mut s.visuals;
        v.dark_mode = true;
        v.window_fill = PANEL;
        v.panel_fill = PANEL;
        v.window_corner_radius = 10.into();
        v.window_stroke = Stroke::new(1.0, Color32::from_white_alpha(22));
        v.extreme_bg_color = Color32::from_rgb(16, 20, 27);
        v.faint_bg_color = Color32::from_white_alpha(6);
        v.slider_trailing_fill = true;
        v.selection.bg_fill = ACCENT;
        v.selection.stroke = Stroke::new(1.5, ON_ACCENT);
        v.hyperlink_color = ACCENT;
        v.interact_cursor = Some(egui::CursorIcon::PointingHand);
        let w = &mut v.widgets;
        w.noninteractive.fg_stroke = Stroke::new(1.0, TEXT);
        w.noninteractive.bg_stroke = Stroke::new(1.0, Color32::from_white_alpha(20));
        for (st, bg, fg) in [
            (&mut w.inactive, Color32::from_rgb(34, 40, 50), Color32::from_rgb(222, 226, 232)),
            (&mut w.hovered, Color32::from_rgb(50, 59, 73), Color32::WHITE),
            (&mut w.active, Color32::from_rgb(66, 78, 96), Color32::WHITE),
            (&mut w.open, Color32::from_rgb(50, 59, 73), Color32::WHITE),
        ] {
            st.bg_fill = bg;
            st.weak_bg_fill = bg;
            st.fg_stroke = Stroke::new(1.5, fg);
            st.corner_radius = 6.into();
        }
        w.inactive.bg_stroke = Stroke::NONE;
        w.hovered.bg_stroke = Stroke::new(1.0, ACCENT.gamma_multiply(0.6));
        w.active.bg_stroke = Stroke::new(1.0, ACCENT);
    });
}

/// A panel frame in the shared style.
pub fn panel() -> egui::Frame {
    egui::Frame::new().fill(PANEL).corner_radius(12.0).inner_margin(20.0).stroke(Stroke::new(1.0, Color32::from_white_alpha(18))).shadow(egui::Shadow { offset: [0, 6], blur: 24, spread: 0, color: Color32::from_black_alpha(110) })
}

/// A full-width menu button; the primary one is filled with the accent
/// colour. Its hint (a key, or the gamepad's buttons) sits after the text.
pub fn wide_button(ui: &mut egui::Ui, text: &str, hint: Hint, primary: bool, enabled: bool) -> egui::Response {
    let w = ui.available_width();
    let (fill, col) = if !enabled {
        (Color32::from_rgb(40, 44, 50), Color32::from_rgb(110, 114, 120))
    } else if primary {
        (ACCENT, ON_ACCENT)
    } else {
        (Color32::from_rgb(34, 40, 50), TEXT)
    };
    let mut job = egui::text::LayoutJob::default();
    let font = egui::FontId::proportional(if primary { 17.0 } else { 15.0 });
    job.append(text, 0.0, egui::TextFormat { font_id: font, color: col, ..Default::default() });
    let hc = if primary { ON_ACCENT.gamma_multiply(0.7) } else { DIM };
    if let Hint::Text(hint) = hint {
        job.append(&format!("   {hint}"), 0.0, egui::TextFormat { font_id: egui::FontId::proportional(12.0), color: hc, ..Default::default() });
    }
    let btn = egui::Button::new(job).fill(fill).corner_radius(8.0).min_size(egui::vec2(w, if primary { 42.0 } else { 34.0 }));
    let resp = ui.add_enabled(enabled, btn);
    if let Hint::Pad(spec) = hint {
        // The buttons, right-aligned in the button.
        let h = if primary { 22.0 } else { 19.0 };
        let p = ui.painter().with_clip_rect(resp.rect);
        let x = resp.rect.right() - 10.0 - padui::spec_width(&p, spec, h);
        padui::paint_spec(&p, x, resp.rect.center().y, spec, h, hc);
    }
    resp
}

/// A chevron pointing right, turned `turn` quarter turns clockwise (1 = down).
pub fn chevron(p: &egui::Painter, c: egui::Pos2, size: f32, turn: f32, col: Color32) {
    let rot = egui::emath::Rot2::from_angle(turn * std::f32::consts::FRAC_PI_2);
    let pts = [egui::vec2(-0.3, -0.6), egui::vec2(0.3, 0.0), egui::vec2(-0.3, 0.6)].map(|v| c + rot * v * size);
    p.line(pts.to_vec(), Stroke::new(1.8, col));
}

/// A collapsible section: a full-width header row (its title, a few words
/// on what is inside while it is shut, a chevron) that opens and closes the
/// body under it. Shut ones keep a first look at the start screen calm.
pub fn section(ui: &mut egui::Ui, id: &str, title: &str, summary: &str, open: bool, body: impl FnOnce(&mut egui::Ui)) {
    let id = ui.make_persistent_id(("section", id));
    let mut state = egui::collapsing_header::CollapsingState::load_with_default_open(ui.ctx(), id, open);
    let (rect, resp) = ui.allocate_exact_size(egui::vec2(ui.available_width(), 36.0), egui::Sense::click());
    let k = state.openness(ui.ctx());
    let lit = resp.hovered() || resp.has_focus();
    let p = ui.painter();
    p.rect_filled(rect, 8.0, Color32::from_white_alpha(if lit { 16 } else { 7 }));
    let title_col = if k > 0.5 { ACCENT } else if lit { Color32::WHITE } else { TEXT };
    p.text(rect.left_center() + egui::vec2(12.0, 0.0), egui::Align2::LEFT_CENTER, title, egui::FontId::proportional(15.0), title_col);
    if k < 1.0 && !summary.is_empty() {
        let col = DIM.gamma_multiply(1.0 - k);
        p.text(rect.right_center() - egui::vec2(30.0, 0.0), egui::Align2::RIGHT_CENTER, summary, egui::FontId::proportional(12.0), col);
    }
    chevron(p, rect.right_center() - egui::vec2(15.0, 0.0), 6.0, k, if lit { Color32::WHITE } else { DIM });
    if resp.clicked() {
        state.toggle(ui);
    }
    state.show_body_unindented(ui, |ui| {
        ui.add_space(2.0);
        egui::Frame::new().inner_margin(egui::Margin { left: 4, right: 4, top: 0, bottom: 4 }).show(ui, |ui| {
            ui.set_width(ui.available_width());
            body(ui);
        });
    });
    ui.add_space(2.0);
}

/// The title bar of a start-screen window: a small capitalised name, then a hairline.
pub fn window_title(ui: &mut egui::Ui, s: &str) {
    ui.label(RichText::new(s).size(11.5).strong().color(DIM).extra_letter_spacing(1.5));
    let r = ui.available_rect_before_wrap();
    ui.painter().hline(r.x_range(), r.top() + 2.0, Stroke::new(1.0, Color32::from_white_alpha(18)));
    ui.add_space(8.0);
}

/// A small section heading.
pub fn heading(ui: &mut egui::Ui, s: &str) {
    if !ui.layout().is_horizontal() {
        ui.add_space(4.0);
    }
    ui.label(RichText::new(s).size(11.5).strong().color(ACCENT));
}

/// One `keys … action` row of the controls list.
pub fn key_row(ui: &mut egui::Ui, keys: &'static str, what: &'static str) {
    egui::Frame::new().fill(Color32::from_rgb(38, 45, 56)).corner_radius(4.0).inner_margin(egui::Margin::symmetric(6, 0)).show(ui, |ui| {
        ui.label(RichText::new(tr(keys)).monospace().size(12.0).color(Color32::WHITE));
    });
    ui.label(RichText::new(tr(what)).size(13.0).color(TEXT));
    ui.end_row();
}

/// One `buttons … action` row of the gamepad controls list.
pub fn pad_row(ui: &mut egui::Ui, spec: &str, what: &'static str) {
    padui::glyphs(ui, spec, 20.0);
    ui.label(RichText::new(tr(what)).size(13.0).color(TEXT));
    ui.end_row();
}

/// The gamepad, flying: (buttons, what they do).
pub const PAD_FLYING: &[(&str, &str)] = &[
    ("LS", "pitch and roll"),
    ("LT / RT", "rudder left / right"),
    ("A / B", "throttle up / down (hold)"),
    ("X", "wheel brakes (hold)"),
    ("Y + B", "parking brake on / off"),
    ("Y + Up / Down", "trim nose down / up"),
    ("LB / RB", "flaps up / down a notch"),
    ("L3", "landing gear"),
    ("RS", "look around"),
    ("R3", "reset the view"),
    ("Dpad", "quick views"),
    ("View", "chase / cockpit camera"),
    ("A", "fly again after a crash"),
];

/// The gamepad, in the menus and on the globe.
pub const PAD_MENUS: &[(&str, &[(&str, &str)])] = &[
    ("GAMEPAD · MENUS", &[
        ("Menu", "pause menu"),
        ("Y + View", "flight panel (beside the flying)"),
        ("Dpad", "move between choices"),
        ("A", "press / choose"),
        ("Left / Right", "slide a slider"),
        ("LB / RB", "turn the page"),
        ("B", "back, close"),
    ]),
    ("GAMEPAD · GLOBE", &[
        ("LS", "turn the globe"),
        ("LT / RT", "zoom out / in"),
        ("X", "start at the crosshair"),
        ("Menu", "take off"),
        ("View", "settings and this controls list"),
    ]),
];

/// The whole controls list, in two columns of sections: the gamepad's while
/// one is plugged in, else the keyboard's.
pub fn controls_list(ui: &mut egui::Ui) {
    if padui::active() {
        ui.columns(2, |cols| {
            heading(&mut cols[0], tr("GAMEPAD · FLYING"));
            egui::Grid::new("pad_fly").num_columns(2).min_row_height(23.0).spacing([10.0, 3.0]).show(&mut cols[0], |ui| {
                for (k, w) in PAD_FLYING {
                    pad_row(ui, k, w);
                }
            });
            for (title, rows) in PAD_MENUS {
                heading(&mut cols[1], tr(title));
                egui::Grid::new(("pad", *title)).num_columns(2).min_row_height(23.0).spacing([10.0, 3.0]).show(&mut cols[1], |ui| {
                    for (k, w) in rows.iter() {
                        pad_row(ui, k, w);
                    }
                });
                cols[1].add_space(4.0);
            }
        });
        return;
    }
    let half = (CONTROLS.len() + 1) / 2;
    ui.columns(2, |cols| {
        for (c, ui) in cols.iter_mut().enumerate() {
            let range = if c == 0 { 0..half } else { half..CONTROLS.len() };
            for (title, rows) in &CONTROLS[range] {
                heading(ui, tr(title));
                egui::Grid::new(("keys", *title)).num_columns(2).min_row_height(19.0).spacing([10.0, 3.0]).show(ui, |ui| {
                    for (k, w) in rows.iter() {
                        key_row(ui, k, w);
                    }
                });
                ui.add_space(4.0);
            }
        }
    });
}

/// "hh:mm" for an hour of the day.
pub fn hhmm(h: f64) -> String {
    let h = h.rem_euclid(24.0);
    format!("{:02}:{:02}", h as u32, (h.fract() * 60.0) as u32)
}

/// "37.8085°N  122.4650°W"
pub fn latlon(lat: f64, lon: f64) -> String {
    format!("{:.4}°{}  {:.4}°{}", lat.abs(), if lat >= 0.0 { 'N' } else { 'S' }, lon.abs(), if lon >= 0.0 { 'E' } else { 'W' })
}
