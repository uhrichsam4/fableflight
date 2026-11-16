//! --gpu-profile: GPU time per render pass, from timestamps written at each
//! pass's start and end (Apple's GPUs only take them at pass boundaries).
//! Logs the mean milliseconds per label every 2 s. Passes overlap on a
//! tile-based GPU (and a pass's start is stamped while it still waits on
//! the one before), so each is charged only the time it adds past the end
//! of all before it: they add up to "~frame", first start to last end. Also `opt_on`: FABLE_OPT_OFF=name,... switches optimisations
//! off by name, for before/after numbers and pictures from the same build.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};

/// Whether optimisation `name` is on: FABLE_OPT_OFF lists those that are
/// not ("all" for every one).
pub fn opt_on(name: &str) -> bool {
    static OFF: OnceLock<Vec<String>> = OnceLock::new();
    let off = OFF.get_or_init(|| {
        let v: Vec<String> = std::env::var("FABLE_OPT_OFF").unwrap_or_default().split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect();
        if !v.is_empty() {
            log::info!("FABLE_OPT_OFF: {}", v.join(","));
        }
        v
    });
    !off.iter().any(|n| n == name || n == "all")
}

/// FABLE_SKIP=name,...: leave out a group of the main view's draws (photo,
/// terrain, tess, field, road, build, trees, inst, objs), to see what each
/// costs. A profiling aid only: the frame is wrong.
pub fn skip(name: &str) -> bool {
    static SKIP: OnceLock<Vec<String>> = OnceLock::new();
    SKIP.get_or_init(|| std::env::var("FABLE_SKIP").unwrap_or_default().split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect()).iter().any(|n| n == name)
}

/// Passes timed at most per frame.
const MAX_PASSES: u32 = 160;
const RING: usize = 3;

pub struct GpuProf {
    qs: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    read: Vec<wgpu::Buffer>,
    /// this frame's passes, in the order they were begun
    labels: RefCell<Vec<&'static str>>,
    /// per ring slot: the labels it holds, and whether it is mapped yet
    pending: Vec<Option<(Vec<&'static str>, Arc<AtomicBool>)>>,
    slot: usize,
    period_ns: f64,
    acc: BTreeMap<&'static str, f64>,
    frames: u32,
    t: std::time::Instant,
}

impl GpuProf {
    pub fn new(d: &wgpu::Device, q: &wgpu::Queue) -> Self {
        let n = MAX_PASSES as u64 * 2;
        let qs = d.create_query_set(&wgpu::QuerySetDescriptor { label: Some("gpu profile"), ty: wgpu::QueryType::Timestamp, count: n as u32 });
        let resolve = d.create_buffer(&wgpu::BufferDescriptor { label: Some("gpu profile"), size: n * 8, usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
        let read = (0..RING).map(|_| d.create_buffer(&wgpu::BufferDescriptor { label: Some("gpu profile read"), size: n * 8, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false })).collect();
        Self { qs, resolve, read, labels: RefCell::new(Vec::new()), pending: (0..RING).map(|_| None).collect(), slot: 0, period_ns: q.get_timestamp_period() as f64, acc: BTreeMap::new(), frames: 0, t: std::time::Instant::now() }
    }

    /// The next two query slots for a pass called `label`, if any are left.
    fn next(&self, label: &'static str) -> Option<u32> {
        let mut l = self.labels.borrow_mut();
        if l.len() as u32 >= MAX_PASSES || self.pending[self.slot].is_some() {
            return None;
        }
        l.push(label);
        Some((l.len() as u32 - 1) * 2)
    }
    pub fn render(&self, label: &'static str) -> Option<wgpu::RenderPassTimestampWrites<'_>> {
        self.next(label).map(|i| wgpu::RenderPassTimestampWrites { query_set: &self.qs, beginning_of_pass_write_index: Some(i), end_of_pass_write_index: Some(i + 1) })
    }
    pub fn compute(&self, label: &'static str) -> Option<wgpu::ComputePassTimestampWrites<'_>> {
        self.next(label).map(|i| wgpu::ComputePassTimestampWrites { query_set: &self.qs, beginning_of_pass_write_index: Some(i), end_of_pass_write_index: Some(i + 1) })
    }

    /// Before the frame's commands are finished: copy its timestamps out.
    pub fn resolve(&self, enc: &mut wgpu::CommandEncoder) {
        let n = self.labels.borrow().len() as u32 * 2;
        // (a slot still waiting on its map from RING frames ago is skipped: this frame goes unmeasured)
        if n > 0 && self.pending[self.slot].is_none() {
            enc.resolve_query_set(&self.qs, 0..n, &self.resolve, 0);
            enc.copy_buffer_to_buffer(&self.resolve, 0, &self.read[self.slot], 0, n as u64 * 8);
        }
    }

    /// After the submit: map this frame's copy, take in the frames whose
    /// copies are ready, and log the means every 2 s.
    pub fn after_submit(&mut self) {
        let labels = std::mem::take(&mut *self.labels.borrow_mut());
        if !labels.is_empty() && self.pending[self.slot].is_none() {
            let done = Arc::new(AtomicBool::new(false));
            let d2 = done.clone();
            self.read[self.slot].slice(..).map_async(wgpu::MapMode::Read, move |r| d2.store(r.is_ok(), Ordering::Release));
            self.pending[self.slot] = Some((labels, done));
        }
        self.slot = (self.slot + 1) % RING;
        for k in 0..RING {
            let ready = matches!(&self.pending[k], Some((_, d)) if d.load(Ordering::Acquire));
            if !ready {
                continue;
            }
            let (labels, _) = self.pending[k].take().unwrap();
            {
                let data = self.read[k].slice(..).get_mapped_range().expect("mapped");
                let ts: &[u64] = bytemuck::cast_slice(&data[..labels.len() * 16]);
                let ms = |a: u64, b: u64| b.saturating_sub(a) as f64 * self.period_ns * 1e-6;
                // Each pass is charged what it adds to the timeline: from the
                // later of its start and the end of all before it, to its end.
                // (Its raw start-to-end includes waiting on those before it.)
                let (mut lo, mut hi) = (u64::MAX, 0u64);
                for (i, l) in labels.iter().enumerate() {
                    let (a, b) = (ts[i * 2], ts[i * 2 + 1]);
                    if a == 0 || b < a {
                        continue;
                    }
                    lo = lo.min(a);
                    *self.acc.entry(l).or_default() += ms(a.max(hi), b);
                    hi = hi.max(b);
                }
                *self.acc.entry("~frame").or_default() += if hi > lo { ms(lo, hi) } else { 0.0 };
                self.frames += 1;
            }
            self.read[k].unmap();
        }
        if self.t.elapsed().as_secs_f64() >= 2.0 && self.frames > 0 {
            let n = self.frames as f64;
            let mut v: Vec<_> = self.acc.iter().map(|(l, s)| (*l, s / n)).collect();
            v.sort_by(|a, b| b.1.total_cmp(&a.1));
            let line: Vec<String> = v.iter().map(|(l, s)| format!("{l} {s:.2}")).collect();
            log::info!("gpu ms ({} frames): {}", self.frames, line.join(" | "));
            self.acc.clear();
            self.frames = 0;
            self.t = std::time::Instant::now();
        }
    }
}
