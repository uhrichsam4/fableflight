//! The view from the left seat. The eye sits on the pilot's head, and the
//! head turns on its neck: the eyes are ahead of and above the joint it turns
//! about, so looking round moves them as well as turning them. Looking far to
//! one side, back over a shoulder or down into the footwell, the pilot leans
//! that way. In a bank the head tilts part of the way back toward the
//! horizon (the reflex every pilot has). The view zooms, and the eye opens up
//! when it looks at the dim cabin and stops down again out of the window.
use crate::aircraft::Spec;
use glam::{DQuat, DVec3, Vec3};
use std::f64::consts::PI;

/// Default vertical field of view in the cockpit (degrees).
pub const FOV: f32 = 58.0;
/// The joint the head turns on, from the eye (body frame, m).
const NECK: DVec3 = DVec3::new(0.0, -0.075, 0.085);

pub struct Head {
    /// where the head is looking now (it follows the look asked for quickly
    /// but not instantly), rad: yaw positive to the right, pitch positive down
    pub yaw: f64,
    pub pitch: f64,
    lean: DVec3,
    roll: f64,
    /// zoom asked for (a factor on the field of view), and where it is now
    pub zoom: f64,
    zoom_s: f64,
    /// the eye's adaptation, in stops over the outside exposure
    pub adapt: f32,
    pub inside: f32,
}

/// The eye: its position and orientation in the body frame, the vertical field of view.
pub struct View {
    pub eye: DVec3,
    pub rot: DQuat,
    pub fov: f32,
}

fn smooth(a: f64, b: f64, x: f64) -> f64 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}
fn follow(v: f64, to: f64, rate: f64, dt: f64) -> f64 {
    v + (to - v) * (1.0 - (-rate * dt).exp())
}
/// Body-frame look rotation for yaw (right +) and pitch (down +).
pub fn look(yaw: f64, pitch: f64) -> DQuat {
    DQuat::from_rotation_y(-yaw) * DQuat::from_rotation_x(-pitch)
}

impl Head {
    pub fn new(yaw: f64, pitch: f64) -> Head {
        Head { yaw, pitch, lean: DVec3::ZERO, roll: 0.0, zoom: 1.0, zoom_s: 1.0, adapt: 0.55, inside: 0.5 }
    }

    /// Snap to the asked-for look at once (entering the cockpit, respawning).
    pub fn reset(&mut self, yaw: f64, pitch: f64) {
        self.yaw = yaw;
        self.pitch = pitch;
        self.lean = DVec3::ZERO;
        self.roll = 0.0;
    }

    pub fn zoom_by(&mut self, steps: f64) {
        self.zoom = (self.zoom * (1.0 - steps * 0.08)).clamp(0.35, 1.3);
    }

    /// One frame. `want` is the look asked for (yaw, pitch), `fov` the
    /// vertical field of view setting (deg), `shake` the head's sway on its
    /// neck from the physics (body frame), `rot` the aircraft's attitude and
    /// `up` the local vertical (world). Returns the eye for this frame.
    pub fn update(&mut self, spec: &'static Spec, dt: f64, want: (f64, f64), fov: f32, shake: DVec3, rot: DQuat, up: DVec3, aspect: f32, day: f32) -> View {
        // Keep the head the short way round from where it is.
        let wy = self.yaw + (want.0 - self.yaw + PI).rem_euclid(2.0 * PI) - PI;
        self.yaw = follow(self.yaw, wy, 22.0, dt);
        self.pitch = follow(self.pitch, want.1, 22.0, dt);
        self.zoom_s = follow(self.zoom_s, self.zoom, 10.0, dt);
        let (y, p) = (self.yaw.rem_euclid(2.0 * PI) - if self.yaw.rem_euclid(2.0 * PI) > PI { 2.0 * PI } else { 0.0 }, self.pitch);

        // Lean: toward the side window (not far: the door is close), across
        // to the right one, turned round in the seat to look back, forward
        // and down at the panel and the floor, back a little to look up.
        let side = smooth(0.9, 1.8, y.abs());
        let back = smooth(1.9, 2.7, y.abs());
        let down = smooth(0.3, 0.9, p) * (1.0 - side * 0.5);
        let upl = smooth(-0.25, -0.7, p);
        let mut lean = DVec3::ZERO;
        if y < 0.0 {
            lean += DVec3::new(-0.06, 0.0, -0.04) * side + DVec3::new(0.04, 0.01, 0.06) * back;
        } else {
            lean += DVec3::new(0.10, -0.01, -0.05) * side + DVec3::new(0.03, 0.01, 0.07) * back;
        }
        lean += DVec3::new(0.0, -0.04, -0.07) * down + DVec3::new(0.0, -0.01, 0.04) * upl;
        let r = 1.0 - (-7.0 * dt).exp();
        self.lean += (lean - self.lean) * r;

        let q = look(self.yaw, self.pitch);
        // The eyes swing round the neck joint behind them.
        let eye = spec.eye.as_dvec3() + shake + self.lean - NECK + q * NECK;

        // Tilt part of the way back toward the horizon (only near level).
        let d = rot * (q * DVec3::NEG_Z);
        let u = rot * (q * DVec3::Y);
        let flat = (up - d * up.dot(d)).normalize_or_zero();
        let mut tilt = 0.0;
        if flat != DVec3::ZERO && d.dot(up).abs() < 0.85 {
            let a = u.cross(flat).dot(d).atan2(u.dot(flat));
            tilt = (a * 0.18).clamp(-0.14, 0.14) * smooth(0.95, 0.6, d.dot(up).abs());
        }
        self.roll = follow(self.roll, tilt, 3.0, dt);
        let rot_b = q * DQuat::from_rotation_z(-self.roll);
        let fov_now = fov * self.zoom_s as f32;

        // How much of the view is the cabin: the eye opens up for it (more by
        // day; at night the cabin and the world are both dark).
        let inside = interior(spec, rot_b, fov_now.to_radians(), aspect);
        self.inside = inside;
        let want = (inside * 1.3 - 0.12) * (0.45 + 0.55 * day);
        // Opening up for the dark takes longer than stopping down for the light.
        let rate = if want > self.adapt { 1.4 } else { 3.5 };
        self.adapt += (want - self.adapt) * (1.0 - (-rate * dt as f32).exp());
        View { eye, rot: rot_b, fov: fov_now }
    }
}

/// The share of the view (centre weighted) that the cabin fills, from a
/// table of which way the eye sees out.
fn interior(spec: &'static Spec, rot: DQuat, fov: f32, aspect: f32) -> f32 {
    let t = table(spec);
    let (ty, tx) = ((fov * 0.5).tan(), (fov * 0.5).tan() * aspect);
    let (mut sum, mut wsum) = (0.0f32, 0.0f32);
    for j in 0..7 {
        for i in 0..9 {
            let (sx, sy) = (i as f32 / 4.0 - 1.0, j as f32 / 3.0 - 1.0);
            let d = (rot * DVec3::new((sx * tx * 0.9) as f64, (sy * ty * 0.9) as f64, -1.0)).normalize();
            let w = 1.0 - 0.6 * (sx * sx + sy * sy) * 0.5;
            sum += w * lookup(t, d);
            wsum += w;
        }
    }
    sum / wsum
}

const LW: usize = 96;
const LH: usize = 48;
fn dir_of(yaw: f32, pitch: f32) -> Vec3 {
    Vec3::new(pitch.cos() * yaw.sin(), -pitch.sin(), -pitch.cos() * yaw.cos())
}
fn lookup(t: &[f32], d: DVec3) -> f32 {
    let d = d.as_vec3();
    let yaw = d.x.atan2(-d.z);
    let pitch = (-d.y).clamp(-1.0, 1.0).asin();
    let u = ((yaw / std::f32::consts::TAU + 0.5) * LW as f32 - 0.5).rem_euclid(LW as f32);
    let v = ((pitch / std::f32::consts::PI + 0.5) * LH as f32 - 0.5).clamp(0.0, LH as f32 - 1.001);
    let (i0, j0) = (u.floor() as usize, v.floor() as usize);
    let (fu, fv) = (u.fract(), v.fract());
    let at = |i: usize, j: usize| t[j.min(LH - 1) * LW + i % LW];
    let a = at(i0, j0) + (at(i0 + 1, j0) - at(i0, j0)) * fu;
    let b = at(i0, j0 + 1) + (at(i0 + 1, j0 + 1) - at(i0, j0 + 1)) * fu;
    a + (b - a) * fv
}
/// For each look direction from the seat, how much of it lands inside the cabin.
fn table(spec: &'static Spec) -> &'static [f32] {
    static T: [std::sync::OnceLock<Vec<f32>>; 5] = [const { std::sync::OnceLock::new() }; 5];
    T[spec.id as usize].get_or_init(|| {
        let mut t = vec![0.0f32; LW * LH];
        std::thread::scope(|s| {
            for (j, row) in t.chunks_mut(LW).enumerate() {
                s.spawn(move || {
                    for (i, c) in row.iter_mut().enumerate() {
                        let mut n = 0;
                        for (a, b) in [(0.25f32, 0.25f32), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)] {
                            let yaw = ((i as f32 + a) / LW as f32 - 0.5) * std::f32::consts::TAU;
                            let pitch = ((j as f32 + b) / LH as f32 - 0.5) * std::f32::consts::PI;
                            n += (spec.sees_inside)(spec.eye, dir_of(yaw, pitch)) as u32;
                        }
                        *c = n as f32 / 4.0;
                    }
                });
            }
        });
        t
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn panel_is_inside_and_windshield_is_out() {
        let t = table(&crate::aircraft::c172::SPEC);
        // Straight ahead and a little up: out through the windshield.
        assert!(lookup(t, dir_of(0.0, -0.1).as_dvec3()) < 0.3);
        // Down at the panel: inside.
        assert!(lookup(t, dir_of(0.0, 0.6).as_dvec3()) > 0.7);
        // Straight up at the headliner: inside.
        assert!(lookup(t, dir_of(0.0, -1.5).as_dvec3()) > 0.7);
        // Out of the left front window (straight left is the door post).
        assert!(lookup(t, dir_of(-1.15, 0.25).as_dvec3()) < 0.4);
    }
}
