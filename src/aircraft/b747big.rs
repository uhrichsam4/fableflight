//! The 747-400 blown up: five times its size (353 m long, 322 m across the
//! winglets) at the 747's own speeds, and twenty times its size (1.4 km long,
//! 1.3 km across) at twenty times its speeds, so it covers its own length in
//! the same time a 747 does. Both are the same model and data, scaled by
//! `aircraft::scaled`. Only the function fields need wrapping here, since a
//! plain `fn` cannot carry the factor: they are generic over it, `K`.
//! plane.wgsl draws them as the 747 (`Spec::id` 3 and 4: its windows, wells
//! and paint looked up at 1/`Spec::size_k` of the size, see `kscale` there).
use super::{b747, scale_model, scaled, Spec};
use crate::airframe::{Model, Sect};
use glam::Vec3;
use std::sync::LazyLock;

fn build<const K: u32>() -> Model {
    scale_model((b747::SPEC.build)(), K as f32)
}

fn wing_sect<const K: u32>(x: f32) -> Sect {
    let k = K as f32;
    let s = (b747::SPEC.wing_sect)(x / k);
    Sect { le: s.le * k, c: s.c * k, ..s }
}

fn fuse<const K: u32>(z: f32) -> (f32, f32, f32) {
    let k = K as f32;
    let (t, m, b) = (b747::SPEC.fuse)(z / k);
    (t * k, m * k, b * k)
}

fn sees_inside<const K: u32>(eye: Vec3, d: Vec3) -> bool {
    (b747::SPEC.sees_inside)(eye / K as f32, d)
}

/// The 747 `K` times its size, `kv` times as fast.
fn big<const K: u32>(id: u32, name: &'static str, keys: &'static [&'static str], blurb: &'static str, kv: f64) -> Spec {
    Spec { id, name, keys, blurb, build: build::<K>, wing_sect: wing_sect::<K>, fuse: fuse::<K>, sees_inside: sees_inside::<K>, ..scaled(&b747::SPEC, K as f64, kv) }
}

pub static X5: LazyLock<Spec> = LazyLock::new(|| {
    big::<5>(3, "Boeing 747 ×5", &["b747x5", "747x5", "mega"], "The 747-400 at five times its size: 353 m long, 322 m across, 7,250 t. Same speeds and handling as the 747. / speedbrake, Z reversers.", 1.0)
});

pub static X20: LazyLock<Spec> = LazyLock::new(|| {
    big::<20>(
        4,
        "Boeing 747 ×20",
        &["b747x20", "747x20", "giga"],
        "The 747-400 at twenty times its size, 1.4 km long and 1.3 km across, flying twenty times as fast: 5,000 kt. Starts in the air; no runway is long enough. / speedbrake.",
        20.0,
    )
});

#[cfg(test)]
mod tests {
    use super::*;

    fn bigger(s: &'static Spec, k: f32, key: &str) {
        let (c, h) = (s.build)().bounds();
        let (c1, h1) = (b747::SPEC.build)().bounds();
        assert!((h - h1 * k).abs().max_element() < 0.01 * h.max_element(), "{h:?} vs {h1:?}");
        assert!((c - c1 * k).abs().max_element() < 0.01 * k, "{c:?} vs {c1:?}");
        assert!(super::super::by_key(key).is_some_and(|x| std::ptr::eq(x, s)));
        assert_eq!(s.gear.len(), b747::SPEC.gear.len());
        assert_eq!(s.size_k, k);
    }

    #[test]
    fn five_times_the_747() {
        let s: &'static Spec = &X5;
        bigger(s, 5.0, "mega");
        // Wing loading and speeds as the 747's.
        assert!((s.mass / s.wing_area - b747::SPEC.mass / b747::SPEC.wing_area).abs() < 1e-6);
        assert_eq!(s.cruise, b747::SPEC.cruise);
        assert_eq!(s.handling.q_ref, b747::SPEC.handling.q_ref);
    }

    #[test]
    fn twenty_times_the_747_twenty_times_as_fast() {
        let s: &'static Spec = &X20;
        let b = &b747::SPEC;
        bigger(s, 20.0, "giga");
        assert!((s.cruise / b.cruise - 20.0).abs() < 1e-9);
        assert!((s.vne / b.vne - 20.0).abs() < 1e-9 && (s.approach.0 / b.approach.0 - 20.0).abs() < 1e-9);
        assert_eq!(s.speed_k, 20.0);
        // Wing loading grows with the dynamic pressure, kv² = 400 times: at
        // twenty times the speed it flies at the 747's lift coefficient.
        let cl = |s: &Spec, v: f64| s.mass * 9.81 / (0.5 * 1.225 * v * v * s.wing_area);
        assert!(((s.mass / s.wing_area) / (b.mass / b.wing_area) - 400.0).abs() < 1e-6);
        assert!((cl(s, s.cruise) - cl(b, b.cruise)).abs() < 1e-9);
        // Thrust to weight as the 747's; the handling's reference pressure follows q.
        let t = |s: &Spec| match &s.engine {
            super::super::Engine::Fan(f) => f.n as f64 * f.thrust / (s.mass * 9.81),
            _ => 0.0,
        };
        assert!((t(s) - t(b)).abs() < 1e-9);
        assert!((s.handling.q_ref / b.handling.q_ref - 400.0).abs() < 1e-9);
        // Moments of inertia: mass times length².
        assert!((s.inertia.x / b.inertia.x - 400.0 * 400.0 * 400.0).abs() < 1.0);
    }
}
