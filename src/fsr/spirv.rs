//! The vendored FSR SPIR-V, made fit for naga and Metal as it is loaded
//! (also checked by tests/shaders.rs, which includes this file).

/// The words of `spv`, with:
/// - every `Coherent` decoration (the SPD's globallycoherent mip) turned
///   into `RelaxedPrecision`, which naga ignores: naga only takes memory
///   decorations on storage buffers;
/// - with `spd`, every Rg16f storage image declared Rgba16f instead: Metal
///   has no read-write Rg16Float storage (in the two pyramid passes the only
///   Rg16f images are the SPD mips, made Rgba16Float to match).
pub fn patch(spv: &[u8], spd: bool) -> Vec<u32> {
    let mut w: Vec<u32> = spv.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    // (5 header words, then instructions: word count << 16 | opcode)
    let mut i = 5;
    while i < w.len() {
        let (n, op) = ((w[i] >> 16) as usize, w[i] & 0xffff);
        const DECORATE: u32 = 71;
        const MEMBER_DECORATE: u32 = 72;
        const TYPE_IMAGE: u32 = 25;
        const COHERENT: u32 = 23;
        const RELAXED_PRECISION: u32 = 0;
        const RG16F: u32 = 7;
        const RGBA16F: u32 = 2;
        match op {
            DECORATE if n >= 3 && w[i + 2] == COHERENT => w[i + 2] = RELAXED_PRECISION,
            MEMBER_DECORATE if n >= 4 && w[i + 3] == COHERENT => w[i + 3] = RELAXED_PRECISION,
            TYPE_IMAGE if spd && n >= 9 && w[i + 8] == RG16F => w[i + 8] = RGBA16F,
            _ => {}
        }
        i += n.max(1);
    }
    w
}
