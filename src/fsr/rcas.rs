use crate::fsr::constants::RcasConstants;

/// Setup required constant values for RCAS.
///
/// The scale is {0.0 := maximum sharpness, to N>0, where N is the number of
/// stops (halving) of the reduction of sharpness}.
///
/// # Arguments
/// * `sharpness` - Sharpness value where 0.0 is maximum and higher values reduce sharpness
pub fn populate_rcas_constants(sharpness: f32) -> RcasConstants {
    // Transform from stops to linear value
    let sharpness_linear = (-sharpness).exp2();

    let h_sharp = [sharpness_linear, sharpness_linear];

    RcasConstants {
        rcas_config: [sharpness_linear.to_bits(), pack_half_2x16(h_sharp), 0, 0],
    }
}

/// Pack two f32 values into a single u32 as f16 values
fn pack_half_2x16(values: [f32; 2]) -> u32 {
    f16_bits(values[0]) as u32 | (f16_bits(values[1]) as u32) << 16
}

/// An f32 as the bits of the nearest f16 (round to nearest even; the
/// values here are normal numbers in (0, 1]).
fn f16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
    let man = b & 0x7f_ffff;
    if exp <= 0 {
        return sign;
    }
    if exp >= 31 {
        return sign | 0x7c00;
    }
    let mut h = (exp as u32) << 10 | man >> 13;
    let rest = man & 0x1fff;
    if rest > 0x1000 || (rest == 0x1000 && h & 1 == 1) {
        h += 1;
    }
    sign | h as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rcas_constants_zero_sharpness() {
        let constants = populate_rcas_constants(0.0);
        // At sharpness 0.0, exp2(0) = 1.0
        assert_eq!(constants.rcas_config[0], 1.0f32.to_bits());
    }

    #[test]
    fn test_f16_bits() {
        assert_eq!(f16_bits(1.0), 0x3c00);
        assert_eq!(f16_bits(0.25), 0x3400);
        assert_eq!(f16_bits(0.5), 0x3800);
        assert_eq!(pack_half_2x16([1.0, 0.5]), 0x3800_3c00);
    }

    #[test]
    fn test_rcas_constants_max_sharpness() {
        let constants = populate_rcas_constants(2.0);
        // At sharpness 2.0, exp2(-2.0) = 0.25
        assert_eq!(constants.rcas_config[0], 0.25f32.to_bits());
    }
}
