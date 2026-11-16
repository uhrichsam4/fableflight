//! The route planner: from one airport (or place) to another through any
//! waypoints in between. Each stop is typed as an airport code or a place
//! name; codes are looked up in the airports database at once, places go to
//! the same geocoder as Go to (in main.rs). In flight the next stop is shown
//! on the HUD and drawn in the world, and moves on as each is reached.
use crate::airports;
use crate::geo;
use glam::DVec3;

/// How close counts as having reached a waypoint on the way (m).
pub const REACH_M: f64 = 1_500.0;
/// The destination counts as reached once stopped on the ground this near (m).
pub const ARRIVE_M: f64 = 5_000.0;

#[derive(Clone, Debug)]
pub struct Wp {
    pub name: String,
    /// degrees
    pub lat: f64,
    pub lon: f64,
    /// metres above sea level, where known (airports)
    pub elev: f64,
    pub apt: Option<u32>,
}

/// One line of the plan: what was typed and, once found, where it is.
#[derive(Clone, Debug, Default)]
pub struct Stop {
    pub text: String,
    pub wp: Option<Wp>,
    /// being looked up online
    pub busy: bool,
    /// looked up and not found
    pub missing: bool,
}

impl Stop {
    pub fn new(text: &str) -> Self {
        Stop { text: text.trim().to_string(), ..Default::default() }
    }
}

#[derive(Default)]
pub struct Route {
    /// From, any waypoints, To (at least two lines while planning)
    pub stops: Vec<Stop>,
    /// the stop being flown to (index into `stops`)
    pub next: usize,
    /// arrived at the destination
    pub done: bool,
    /// heading hold to the next stop
    pub autopilot: bool,
}

/// An airport by its code (ICAO or IATA), as a waypoint.
pub fn airport_wp(code: &str) -> Option<Wp> {
    let db = airports::Db::get();
    let i = db.find(code)?;
    let a = &db.apts[i as usize];
    Some(Wp { name: format!("{} {}", a.code(), a.name), lat: a.lat, lon: a.lon, elev: a.elev as f64, apt: Some(i) })
}

/// Great-circle distance (m) between two points in degrees.
pub fn dist(la1: f64, lo1: f64, la2: f64, lo2: f64) -> f64 {
    let (p1, p2) = (la1.to_radians(), la2.to_radians());
    let dp = p2 - p1;
    let dl = (lo2 - lo1).to_radians();
    let h = (dp * 0.5).sin().powi(2) + p1.cos() * p2.cos() * (dl * 0.5).sin().powi(2);
    2.0 * geo::R * h.sqrt().min(1.0).asin()
}

/// Initial true bearing (degrees, 0 = north) from the first point to the second.
pub fn bearing(la1: f64, lo1: f64, la2: f64, lo2: f64) -> f64 {
    let (p1, p2) = (la1.to_radians(), la2.to_radians());
    let dl = (lo2 - lo1).to_radians();
    let y = dl.sin() * p2.cos();
    let x = p1.cos() * p2.sin() - p1.sin() * p2.cos() * dl.cos();
    y.atan2(x).to_degrees().rem_euclid(360.0)
}

/// Points along the great circle between two points (degrees), `n` + 1 of them, as unit vectors.
pub fn arc(a: (f64, f64), b: (f64, f64), n: usize) -> Vec<DVec3> {
    let pa = geo::lla_to_pos(a.0.to_radians(), a.1.to_radians(), 0.0).normalize();
    let pb = geo::lla_to_pos(b.0.to_radians(), b.1.to_radians(), 0.0).normalize();
    let ang = pa.dot(pb).clamp(-1.0, 1.0).acos();
    (0..=n)
        .map(|k| {
            let t = k as f64 / n.max(1) as f64;
            if ang < 1e-9 {
                return pa;
            }
            ((pa * ((1.0 - t) * ang).sin() + pb * (t * ang).sin()) / ang.sin()).normalize()
        })
        .collect()
}

impl Route {
    /// A plan from a list of codes or names (the command line's `--route KSFO,KOAK`).
    pub fn from_list(list: &str) -> Self {
        let mut r = Route { stops: list.split(',').filter(|s| !s.trim().is_empty()).map(Stop::new).collect(), next: 1, ..Default::default() };
        while r.stops.len() < 2 {
            r.stops.push(Stop::default());
        }
        r.resolve_codes();
        r
    }

    pub fn empty() -> Self {
        Route { stops: vec![Stop::default(), Stop::default()], next: 1, ..Default::default() }
    }

    /// Fill in the stops typed as airport codes; returns the others still to look up online.
    pub fn resolve_codes(&mut self) -> Vec<(usize, String)> {
        let mut rest = Vec::new();
        for (i, s) in self.stops.iter_mut().enumerate() {
            if s.wp.is_some() || s.busy || s.text.trim().is_empty() {
                continue;
            }
            match airport_wp(&s.text) {
                Some(w) => {
                    s.wp = Some(w);
                    s.missing = false;
                }
                None => rest.push((i, s.text.trim().to_string())),
            }
        }
        rest
    }

    /// The found stops, in order.
    pub fn wps(&self) -> Vec<&Wp> {
        self.stops.iter().filter_map(|s| s.wp.as_ref()).collect()
    }

    /// Every stop found, and at least two of them: there is a route to fly.
    pub fn ready(&self) -> bool {
        self.stops.len() >= 2 && self.stops.iter().all(|s| s.wp.is_some())
    }

    /// The stop flown to now, if the route is being flown.
    pub fn target(&self) -> Option<&Wp> {
        if !self.ready() || self.done {
            return None;
        }
        self.stops.get(self.next).and_then(|s| s.wp.as_ref())
    }

    pub fn is_last(&self) -> bool {
        self.next + 1 >= self.stops.len()
    }

    /// Total length of the route (m).
    pub fn total(&self) -> f64 {
        self.wps().windows(2).map(|w| dist(w[0].lat, w[0].lon, w[1].lat, w[1].lon)).sum()
    }

    /// Distance still to fly from here (degrees) through the remaining stops (m).
    pub fn remaining(&self, lat: f64, lon: f64) -> f64 {
        let Some(t) = self.target() else { return 0.0 };
        let mut d = dist(lat, lon, t.lat, t.lon);
        let w: Vec<&Wp> = self.stops[self.next..].iter().filter_map(|s| s.wp.as_ref()).collect();
        for p in w.windows(2) {
            d += dist(p[0].lat, p[0].lon, p[1].lat, p[1].lon);
        }
        d
    }

    /// Move on past the waypoint once within reach; the destination once
    /// stopped on the ground near it. Returns the name reached, if any.
    pub fn advance(&mut self, lat: f64, lon: f64, on_ground: bool, speed: f64) -> Option<String> {
        let t = self.target()?;
        let d = dist(lat, lon, t.lat, t.lon);
        let name = t.name.clone();
        if self.is_last() {
            if on_ground && speed < 15.0 && d < ARRIVE_M {
                self.done = true;
                self.autopilot = false;
                return Some(name);
            }
        } else if d < REACH_M {
            self.next += 1;
            return Some(name);
        }
        None
    }

    /// Start flying it from the first leg again.
    pub fn restart(&mut self) {
        self.next = 1;
        self.done = false;
    }
}

/// "1 h 05" or "12 min".
pub fn eta(seconds: f64) -> String {
    if !seconds.is_finite() || seconds <= 0.0 {
        return "--".into();
    }
    let m = (seconds / 60.0).round() as u64;
    if m >= 60 {
        format!("{} h {:02}", m / 60, m % 60)
    } else {
        format!("{m} min")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_ksfo_koak() {
        let mut r = Route::from_list("KSFO,KOAK");
        assert!(r.ready());
        let d = r.total();
        assert!(d > 15_000.0 && d < 25_000.0, "{d}");
        let b = bearing(37.62, -122.38, 37.72, -122.22);
        assert!(b > 30.0 && b < 70.0, "{b}");
        let o = r.stops[1].wp.clone().unwrap();
        assert!(r.advance(o.lat, o.lon, false, 50.0).is_none());
        assert!(r.advance(o.lat, o.lon, true, 5.0).is_some());
        assert!(r.done && r.target().is_none());
        assert_eq!(eta(3900.0), "1 h 05");
        assert_eq!(arc((0.0, 0.0), (0.0, 90.0), 2).len(), 3);
    }
}
