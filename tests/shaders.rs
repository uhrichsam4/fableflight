//! Parses and validates every shader without needing a GPU.
fn check(name: &str, src: &str) -> bool {
    match naga::front::wgsl::parse_str(src) {
        Err(e) => {
            eprintln!("--- {name}\n{}", e.emit_to_string(src));
            false
        }
        Ok(m) => match naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all()).validate(&m) {
            Err(e) => {
                eprintln!("--- {name}\n{}", e.emit_to_string(src));
                false
            }
            Ok(_) => true,
        },
    }
}

#[test]
fn shaders_validate() {
    let common = include_str!("../src/shaders/common.wgsl");
    // Modules that draw the sky get sky.wgsl too, as in render.rs.
    let sky = include_str!("../src/shaders/sky.wgsl");
    let mut ok = check("noise3d", include_str!("../src/shaders/noise3d.wgsl"));
    ok &= check("resolve", include_str!("../src/shaders/resolve.wgsl"));
    ok &= check("detail", include_str!("../src/shaders/detail.wgsl"));
    ok &= check("ocean", include_str!("../src/shaders/ocean.wgsl"));
    ok &= check("shore", include_str!("../src/shaders/shore.wgsl"));
    for (name, src) in [
        ("lut", include_str!("../src/shaders/lut.wgsl")),
        ("terrain", include_str!("../src/shaders/terrain.wgsl")),
        ("photo", include_str!("../src/shaders/photo.wgsl")),
        ("post", include_str!("../src/shaders/post.wgsl")),
        ("particles_sim", include_str!("../src/shaders/particles_sim.wgsl")),
        ("particles_draw", include_str!("../src/shaders/particles_draw.wgsl")),
        ("roads", include_str!("../src/shaders/roads.wgsl")),
        ("trees", include_str!("../src/shaders/trees.wgsl")),
        ("vol", include_str!("../src/shaders/vol.wgsl")),
        ("ao", include_str!("../src/shaders/ao.wgsl")),
        ("taa", include_str!("../src/shaders/taa.wgsl")),
        ("airfield", include_str!("../src/shaders/airfield.wgsl")),
        ("lights", include_str!("../src/shaders/lights.wgsl")),
    ] {
        ok &= check(name, &format!("{common}\n{src}"));
    }
    // As in render.rs, buildings and instanced solids share the facade shading.
    let facade = include_str!("../src/shaders/facade.wgsl");
    for (name, src) in [("buildings", include_str!("../src/shaders/buildings.wgsl")), ("inst", include_str!("../src/shaders/inst.wgsl"))] {
        ok &= check(name, &format!("{common}\n{facade}\n{src}"));
    }
    for (name, src) in [("composite", include_str!("../src/shaders/composite.wgsl")), ("clouds", include_str!("../src/shaders/clouds.wgsl"))] {
        ok &= check(name, &format!("{common}\n{sky}\n{src}"));
    }
    // plane.wgsl takes the 747 deck's string table: checked in b747::deck's tests.
    assert!(ok, "shader errors above");
}

#[path = "../src/fsr/spirv.rs"]
mod fsr_spirv;

/// The vendored FSR 3 SPIR-V (src/fsr), patched as the renderer loads it,
/// goes through naga and on to Metal Shading Language.
#[test]
fn fsr_spirv_validates() {
    let mut ok = true;
    for (name, spv, spd) in [
        ("accumulate", &include_bytes!("../src/fsr/spv/accumulate_pass.spv")[..], false),
        ("accumulate_sharpen", include_bytes!("../src/fsr/spv/accumulate_sharpen_pass.spv"), false),
        ("luma_instability", include_bytes!("../src/fsr/spv/luma_instability_pass.spv"), false),
        ("luma_pyramid", include_bytes!("../src/fsr/spv/luma_pyramid_pass.spv"), true),
        ("prepare_inputs", include_bytes!("../src/fsr/spv/prepare_inputs_pass.spv"), false),
        ("prepare_reactivity", include_bytes!("../src/fsr/spv/prepare_reactivity_pass.spv"), false),
        ("rcas", include_bytes!("../src/fsr/spv/rcas_pass.spv"), false),
        ("shading_change", include_bytes!("../src/fsr/spv/shading_change_pass.spv"), false),
        ("shading_change_pyramid", include_bytes!("../src/fsr/spv/shading_change_pyramid_pass.spv"), true),
    ] {
        let words = fsr_spirv::patch(spv, spd);
        let m = match naga::front::spv::parse_u8_slice(bytemuck::cast_slice(&words), &Default::default()) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("--- fsr {name}: {e}");
                ok = false;
                continue;
            }
        };
        let info = match naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all()).validate(&m) {
            Ok(i) => i,
            Err(e) => {
                eprintln!("--- fsr {name}: {e:?}");
                ok = false;
                continue;
            }
        };
        let opts = naga::back::msl::Options { lang_version: (2, 1), fake_missing_bindings: true, ..Default::default() };
        if let Err(e) = naga::back::msl::write_string(&m, &info, &opts, &naga::back::msl::PipelineOptions::default()) {
            eprintln!("--- fsr {name} (to MSL): {e}");
            ok = false;
        }
    }
    assert!(ok, "FSR shader errors above");
}
