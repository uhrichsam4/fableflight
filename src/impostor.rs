//! Tree impostors: pictures of every tree model at its full detail (solid
//! crown and leaf cards), taken at start-up from IMP_N x IMP_N directions over
//! the upper hemisphere (a hemi-octahedral layout). Beyond `IMP_DIST` a tree
//! is drawn as a single camera-facing quad that blends the three nearest
//! pictures and lights them from the stored albedo, normal, depth and
//! openness to the sky (see the impostor section of trees.wgsl).
//!
//! Each picture is stored premultiplied by its coverage (the multisampled
//! bake is resolved over a transparent clear), so averaging, both in the
//! mip chain and between pictures, keeps colour and coverage consistent.
use crate::render::{tex_entry, ub_entry, TreeVertex, AMB, DEPTH, HDR, MSAA};
use crate::trees::{CARDS, LODS, TREE_SPECIES, VARIANTS};

/// Pictures per side of the grid. Must match IMP_N in trees.wgsl.
pub const N: u32 = 8;
/// Pixels per side of one picture. Must match IMP_CELL in trees.wgsl.
pub const CELL: u32 = 128;
const SIZE: u32 = N * CELL;
pub const LAYERS: u32 = (TREE_SPECIES * VARIANTS) as u32;
/// Down to two pixels a picture; a box filter of whole texels never mixes
/// neighbouring pictures, since every level's picture size is a whole number.
const MIPS: u32 = 7;
const FRAMES: u32 = N * N;
const FMT_A: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
const FMT_N: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

pub struct Impostors {
    tex_a: wgpu::Texture,
    tex_n: wgpu::Texture,
    bounds: wgpu::Buffer,
    bg_bake: wgpu::BindGroup,
    pub bg: wgpu::BindGroup,
    p_bake: wgpu::RenderPipeline,
    p_bake_card: wgpu::RenderPipeline,
    p_mip: wgpu::RenderPipeline,
    l_mip: wgpu::BindGroupLayout,
    /// as the G-buffer: multisampled or not (render.rs `Renderer::msaa`)
    pub p_draw: [wgpu::RenderPipeline; 2],
    /// index buffer of one quad (the vertex index picks the corner and, via the
    /// base vertex, the model)
    pub quad: wgpu::Buffer,
    pub ready: bool,
}

const MIP_WGSL: &str = r#"
@group(0) @binding(0) var src_a: texture_2d<f32>;
@group(0) @binding(1) var src_n: texture_2d<f32>;
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
  return vec4f(p * 2.0 - 1.0, 0.0, 1.0);
}
struct O { @location(0) a: vec4f, @location(1) n: vec4f };
@fragment fn fs(@builtin(position) pos: vec4f) -> O {
  let q = vec2i(pos.xy) * 2;
  var o: O;
  for (var k = 0; k < 4; k++) {
    let t = q + vec2i(k & 1, k >> 1);
    o.a += textureLoad(src_a, t, 0);
    o.n += textureLoad(src_n, t, 0);
  }
  o.a *= 0.25;
  o.n *= 0.25;
  return o;
}
"#;

impl Impostors {
    pub fn new(d: &wgpu::Device, l0: &wgpu::BindGroupLayout, l_tileu: &wgpu::BindGroupLayout, m_tree: &wgpu::ShaderModule) -> Self {
        let tex = |label: &str, format| {
            d.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d { width: SIZE, height: SIZE, depth_or_array_layers: LAYERS },
                mip_level_count: MIPS,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            })
        };
        let (tex_a, tex_n) = (tex("impostor albedo", FMT_A), tex("impostor normal", FMT_N));
        let bsize = LAYERS as u64 * 16;
        let bounds = d.create_buffer(&wgpu::BufferDescriptor { label: Some("impostor bounds"), size: bsize, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let arr = wgpu::TextureViewDimension::D2Array;
        let l_bake = d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("impostor bake"), entries: &[ub_entry(1, false, bsize)] });
        let l_imp = d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("impostors"), entries: &[ub_entry(1, false, bsize), tex_entry(2, arr, false), tex_entry(3, arr, false)] });
        let all = |t: &wgpu::Texture| t.create_view(&wgpu::TextureViewDescriptor { dimension: Some(arr), ..Default::default() });
        let ub = wgpu::BindGroupEntry { binding: 1, resource: bounds.as_entire_binding() };
        let bg_bake = d.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &l_bake, entries: &[ub.clone()] });
        let bg = d.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &l_imp,
            entries: &[ub, wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&all(&tex_a)) }, wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&all(&tex_n)) }],
        });

        const A_TREEV: [wgpu::VertexAttribute; 5] = wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x4, 6 => Float32x4, 7 => Float32x4];
        const A_TREE: [wgpu::VertexAttribute; 3] = [
            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 0, shader_location: 3 },
            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 16, shader_location: 4 },
            wgpu::VertexAttribute { format: wgpu::VertexFormat::Unorm8x4, offset: 32, shader_location: 5 },
        ];
        let target = |f| Some(wgpu::ColorTargetState { format: f, blend: None, write_mask: wgpu::ColorWrites::ALL });
        let bake_targets = [target(FMT_A), target(FMT_N)];
        let draw_targets = [target(HDR), target(HDR), target(AMB)];
        let pipe = |label: &str, groups: &[Option<&wgpu::BindGroupLayout>], m: &wgpu::ShaderModule, vs: &str, fs: &str, buffers: &[Option<wgpu::VertexBufferLayout>], targets: &[Option<wgpu::ColorTargetState>], depth: bool, samples: u32| {
            let layout = d.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: groups, immediate_size: 0 });
            d.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                vertex: wgpu::VertexState { module: m, entry_point: Some(vs), compilation_options: Default::default(), buffers },
                primitive: wgpu::PrimitiveState { cull_mode: None, ..Default::default() },
                depth_stencil: depth.then(|| wgpu::DepthStencilState { format: DEPTH, depth_write_enabled: Some(true), depth_compare: Some(wgpu::CompareFunction::Greater), stencil: Default::default(), bias: Default::default() }),
                multisample: wgpu::MultisampleState { count: samples, mask: !0, alpha_to_coverage_enabled: false },
                fragment: Some(wgpu::FragmentState { module: m, entry_point: Some(fs), compilation_options: Default::default(), targets }),
                multiview_mask: None,
                cache: None,
            })
        };
        let vb_tree = [Some(wgpu::VertexBufferLayout { array_stride: std::mem::size_of::<TreeVertex>() as u64, step_mode: wgpu::VertexStepMode::Vertex, attributes: &A_TREEV })];
        let vb_inst = [Some(wgpu::VertexBufferLayout { array_stride: 36, step_mode: wgpu::VertexStepMode::Instance, attributes: &A_TREE })];
        let g_bake = [Some(l0), Some(l_tileu), Some(&l_bake)];
        let p_bake = pipe("impostor bake", &g_bake, m_tree, "vs_bake", "fs_bake", &vb_tree, &bake_targets, true, MSAA);
        let p_bake_card = pipe("impostor bake cards", &g_bake, m_tree, "vs_bake_card", "fs_bake_card", &vb_tree, &bake_targets, true, MSAA);
        let p_draw = [MSAA, 1].map(|n| pipe("impostors", &[Some(l0), Some(l_tileu), Some(&l_imp)], m_tree, "vs_imp", "fs_imp", &vb_inst, &draw_targets, true, n));
        let m_mip = d.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("impostor mips"), source: wgpu::ShaderSource::Wgsl(MIP_WGSL.into()) });
        let d2 = wgpu::TextureViewDimension::D2;
        let l_mip = d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("impostor mips"), entries: &[tex_entry(0, d2, false), tex_entry(1, d2, false)] });
        let p_mip = pipe("impostor mips", &[Some(&l_mip)], &m_mip, "vs", "fs", &[], &bake_targets, false, 1);
        let quad = wgpu::util::DeviceExt::create_buffer_init(d, &wgpu::util::BufferInitDescriptor { label: Some("impostor quad"), contents: bytemuck::cast_slice(&[0u32, 1, 2, 2, 1, 3]), usage: wgpu::BufferUsages::INDEX });
        Self { tex_a, tex_n, bounds, bg_bake, bg, p_bake, p_bake_card, p_mip, l_mip, p_draw, quad, ready: false }
    }

    /// Bounding sphere (centre, radius) of each tree model at full detail:
    /// what every one of its pictures is framed on.
    pub fn bounds(meshes: &[(Vec<TreeVertex>, Vec<u32>)]) -> Vec<[f32; 4]> {
        (0..LAYERS as usize)
            .map(|k| {
                let mut pts: Vec<glam::Vec3> = Vec::new();
                for v in &meshes[k * LODS].0 {
                    let p = glam::Vec3::from(v.pos);
                    pts.push(p);
                    // Leaf masses draw in under the cards (see vs_bake).
                    if v.col[3] > 0.5 {
                        pts.push(p + glam::Vec3::new(v.aux[0], v.aux[1], v.aux[2]));
                    }
                }
                pts.extend(meshes[k * LODS + CARDS as usize].0.iter().map(|v| glam::Vec3::from(v.pos)));
                let (lo, hi) = pts.iter().fold((glam::Vec3::splat(1e9), glam::Vec3::splat(-1e9)), |(lo, hi), p| (lo.min(*p), hi.max(*p)));
                let c = (lo + hi) * 0.5;
                let r = pts.iter().map(|p| (*p - c).length()).fold(0.1, f32::max) * 1.02;
                [c.x, c.y, c.z, r]
            })
            .collect()
    }

    /// Take every picture of every tree model, then build the mip chains.
    #[allow(clippy::too_many_arguments)]
    pub fn bake(&mut self, d: &wgpu::Device, q: &wgpu::Queue, bg0: &wgpu::BindGroup, bg_tileu: &wgpu::BindGroup, vbuf: &wgpu::Buffer, ibuf: &wgpu::Buffer, ranges: &[(u32, u32, i32)], bounds: &[[f32; 4]]) {
        q.write_buffer(&self.bounds, 0, bytemuck::cast_slice(bounds));
        let ms = |f| {
            d.create_texture(&wgpu::TextureDescriptor {
                label: Some("impostor bake target"),
                size: wgpu::Extent3d { width: SIZE, height: SIZE, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: MSAA,
                dimension: wgpu::TextureDimension::D2,
                format: f,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            })
            .create_view(&Default::default())
        };
        let (ms_a, ms_n, ms_d) = (ms(FMT_A), ms(FMT_N), ms(DEPTH));
        let one = |t: &wgpu::Texture, layer: u32, mip: u32| {
            t.create_view(&wgpu::TextureViewDescriptor { dimension: Some(wgpu::TextureViewDimension::D2), base_mip_level: mip, mip_level_count: Some(1), base_array_layer: layer, array_layer_count: Some(1), ..Default::default() })
        };
        let mut enc = d.create_command_encoder(&Default::default());
        for layer in 0..LAYERS {
            let (ra, rn) = (one(&self.tex_a, layer, 0), one(&self.tex_n, layer, 0));
            let att = |v, r| {
                Some(wgpu::RenderPassColorAttachment { view: v, depth_slice: None, resolve_target: Some(r), ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Discard } })
            };
            let colors = [att(&ms_a, &ra), att(&ms_n, &rn)];
            let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("impostor bake"),
                color_attachments: &colors,
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment { view: &ms_d, depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(0.0), store: wgpu::StoreOp::Discard }), stencil_ops: None }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            rp.set_bind_group(0, bg0, &[]);
            rp.set_bind_group(1, bg_tileu, &[0]);
            rp.set_bind_group(2, &self.bg_bake, &[]);
            rp.set_vertex_buffer(0, vbuf.slice(..));
            rp.set_index_buffer(ibuf.slice(..), wgpu::IndexFormat::Uint32);
            let solid = ranges[layer as usize * LODS];
            let cards = ranges[layer as usize * LODS + CARDS as usize];
            for f in 0..FRAMES {
                rp.set_viewport(((f % N) * CELL) as f32, ((f / N) * CELL) as f32, CELL as f32, CELL as f32, 0.0, 1.0);
                // The instance index tells the shader which model and which picture.
                let inst = layer * FRAMES + f;
                rp.set_pipeline(&self.p_bake);
                rp.draw_indexed(solid.0..solid.0 + solid.1, solid.2, inst..inst + 1);
                if cards.1 > 0 {
                    rp.set_pipeline(&self.p_bake_card);
                    rp.draw_indexed(cards.0..cards.0 + cards.1, cards.2, inst..inst + 1);
                }
            }
        }
        for layer in 0..LAYERS {
            for mip in 1..MIPS {
                let (sa, sn) = (one(&self.tex_a, layer, mip - 1), one(&self.tex_n, layer, mip - 1));
                let bg = d.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &self.l_mip,
                    entries: &[wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&sa) }, wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&sn) }],
                });
                let (da, dn) = (one(&self.tex_a, layer, mip), one(&self.tex_n, layer, mip));
                let att = |v| Some(wgpu::RenderPassColorAttachment { view: v, depth_slice: None, resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store } });
                let colors = [att(&da), att(&dn)];
                let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor { label: Some("impostor mips"), color_attachments: &colors, depth_stencil_attachment: None, timestamp_writes: None, occlusion_query_set: None, multiview_mask: None });
                rp.set_pipeline(&self.p_mip);
                rp.set_bind_group(0, &bg, &[]);
                rp.draw(0..3, 0..1);
            }
        }
        q.submit([enc.finish()]);
        self.ready = true;
    }
}
