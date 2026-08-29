//! IEEE-754 binary16 / bfloat16 <-> f32. GGUF Q4_0/Q8_0 scales are `ggml_half`.

/// Convert IEEE binary16 bits to `f32`.
pub(crate) fn f16_to_f32(h: u16) -> f32 {
    let sign = (u32::from(h) & 0x8000) << 16;
    let exp = i32::from((h >> 10) & 0x1f);
    let man = u32::from(h & 0x3ff);
    let bits = if exp == 0 {
        if man == 0 {
            sign
        } else {
            // Subnormal binary16: the true value is `man * 2^-24`. Shift the
            // mantissa left until the implicit bit reaches position 10, counting
            // `k` shifts; the value is then `(1 + frac) * 2^(-14 - k)`, so the f32
            // exponent field is `113 - k`. With `e = -1 - k` that is `e + 114`.
            let mut m = man;
            let mut e = -1i32;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3ff;
            let exp_bits = u32::try_from(e.saturating_add(114)).unwrap_or(0);
            sign | (exp_bits << 23) | (m << 13)
        }
    } else if exp == 31 {
        sign | 0x7f80_0000 | (man << 13)
    } else {
        let exp_bits = u32::try_from(exp.saturating_add(112)).unwrap_or(0);
        sign | (exp_bits << 23) | (man << 13)
    };
    f32::from_bits(bits)
}

/// Convert `f32` to IEEE binary16 bits (round-toward-zero, subnormals preserved).
pub(crate) fn f32_to_f16(f: f32) -> u16 {
    let b = f.to_bits();
    let sign = u16::try_from((b >> 16) & 0x8000).unwrap_or(0);
    let biased = i32::try_from((b >> 23) & 0xff).unwrap_or(0);
    let exp = biased - 127 + 15;
    let man = (b >> 13) & 0x3ff;
    if exp >= 31 {
        return sign | 0x7c00;
    }
    if exp > 0 {
        let exp_u = u16::try_from(exp).unwrap_or(0);
        let man_u = u16::try_from(man).unwrap_or(0);
        return sign | (exp_u << 10) | man_u;
    }
    // Subnormal binary16 (`|f| < 2^-14`): encode `man * 2^-24`. An f32 zero or
    // f32 subnormal has no representable binary16 magnitude and flushes to zero.
    if biased == 0 {
        return sign;
    }
    let shift = 1i32.saturating_sub(exp);
    if shift > 11 {
        return sign;
    }
    let full = 0x0400u32 | man;
    let m = full >> u32::try_from(shift).unwrap_or(11);
    sign | u16::try_from(m & 0x3ff).unwrap_or(0)
}

/// Load a little-endian binary16 at the start of `bytes`.
pub(crate) fn load_f16_le(bytes: &[u8]) -> Option<f32> {
    let a = *bytes.first()?;
    let b = *bytes.get(1)?;
    Some(f16_to_f32(u16::from_le_bytes([a, b])))
}

/// Independent IEEE binary16 → `f32` for oracles.
///
/// Derived from the arithmetic definition in the standard, not from the
/// bit-surgery in [`f16_to_f32`]:
///
/// - `exp == 0`     → `sign * man * 2^-24` (zero / subnormal)
/// - `0 < exp < 31` → `sign * (1 + man/1024) * 2^(exp-15)`
/// - `exp == 31`    → inf / NaN
///
/// Decode and quant oracles must call this. A production bug in
/// [`f16_to_f32`] (the subnormal off-by-one that halved real Q4_K / Q6_K
/// scales) is invisible if the oracle shares that primitive.
#[cfg(test)]
pub(crate) fn oracle_f16_to_f32(h: u16) -> f32 {
    let sign = if h & 0x8000 == 0 { 1.0f32 } else { -1.0f32 };
    let exp = i32::from((h >> 10) & 0x1f);
    let man = f32::from(h & 0x3ff);
    if exp == 0 {
        sign * man * 2.0f32.powi(-24)
    } else if exp == 31 {
        if h & 0x3ff == 0 {
            sign * f32::INFINITY
        } else {
            f32::NAN
        }
    } else {
        sign * (1.0 + man / 1024.0) * 2.0f32.powi(exp - 15)
    }
}

/// Load a little-endian binary16 through [`oracle_f16_to_f32`].
#[cfg(test)]
pub(crate) fn oracle_load_f16_le(bytes: &[u8]) -> Option<f32> {
    let a = *bytes.first()?;
    let b = *bytes.get(1)?;
    Some(oracle_f16_to_f32(u16::from_le_bytes([a, b])))
}

/// Store `scale` as little-endian binary16.
pub(crate) fn store_f16_le(scale: f32) -> [u8; 2] {
    f32_to_f16(scale).to_le_bytes()
}

/// ggml `GGML_BF16_TO_FP32`: place the 16 bits in the high half of an `f32`.
pub(crate) fn bf16_to_f32(h: u16) -> f32 {
    f32::from_bits(u32::from(h) << 16)
}

/// ggml `ggml_compute_fp32_to_bf16` (round to nearest even; quiet NaN).
pub(crate) fn f32_to_bf16(f: f32) -> u16 {
    let u = f.to_bits();
    if (u & 0x7fff_ffff) > 0x7f80_0000 {
        let hi = u16::try_from(u >> 16).unwrap_or(0);
        return hi | 64;
    }
    let lsb = (u >> 16) & 1;
    let rounded = u.wrapping_add(0x7fff + lsb);
    u16::try_from(rounded >> 16).unwrap_or(0)
}

/// Load a little-endian bfloat16 at the start of `bytes`.
pub(crate) fn load_bf16_le(bytes: &[u8]) -> Option<f32> {
    let a = *bytes.first()?;
    let b = *bytes.get(1)?;
    Some(bf16_to_f32(u16::from_le_bytes([a, b])))
}

/// Store `scale` as little-endian bfloat16.
pub(crate) fn store_bf16_le(scale: f32) -> [u8; 2] {
    f32_to_bf16(scale).to_le_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_values() {
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xbc00), -1.0);
        assert_eq!(f16_to_f32(0x3800), 0.5);
        assert_eq!(f32_to_f16(1.0), 0x3c00);
        assert_eq!(f32_to_f16(-1.0), 0xbc00);
        assert_eq!(f32_to_f16(0.5), 0x3800);
    }

    /// Independent arithmetic definition of IEEE binary16, written from the
    /// standard rather than from bit surgery. See [`oracle_f16_to_f32`].
    fn f16_to_f32_reference(h: u16) -> f32 {
        oracle_f16_to_f32(h)
    }

    /// Every finite binary16 bit pattern must match the arithmetic definition.
    ///
    /// This is the regression guard for a subnormal bug that made every
    /// `|value| < 2^-14` exactly half its true magnitude. Real GGUF super-block
    /// scales land in that range, so the error silently halved Q4_K / Q5_K /
    /// Q6_K / Q8_0 weights whose scale was subnormal. Decode/quant oracles
    /// call [`oracle_f16_to_f32`], not this primitive, so a repeat of that
    /// class of bug is visible to the rest of the suite.
    #[test]
    fn f16_to_f32_matches_arithmetic_definition_exhaustively() {
        let mut checked = 0usize;
        let mut subnormals = 0usize;
        for bits in 0u32..=0xffff {
            let h = u16::try_from(bits).expect("16-bit");
            if (h >> 10) & 0x1f == 0x1f {
                continue; // inf / NaN are not value-comparable
            }
            let got = f16_to_f32(h);
            let want = f16_to_f32_reference(h);
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "binary16 {h:#06x}: got {got:e}, want {want:e}"
            );
            if (h >> 10) & 0x1f == 0 && h & 0x3ff != 0 {
                subnormals += 1;
            }
            checked += 1;
        }
        assert_eq!(checked, 63488);
        assert_eq!(subnormals, 2046);
    }

    /// The observed real-model failure: a Q6_K super-block scale of
    /// `-299 * 2^-24` from `Qwen2.5-0.5B-Instruct-Q4_K_M`. ggml reports
    /// `-1.7821788787841797e-5`; the subnormal bug returned `-8.910894e-6`.
    #[test]
    fn f16_subnormal_scale_from_real_gguf() {
        let bits = 0x8000u16 | 299;
        let got = f16_to_f32(bits);
        let want = -299.0f32 * 2.0f32.powi(-24);
        assert_eq!(got.to_bits(), want.to_bits());
        assert!((got - -1.782_178_9e-5).abs() < 1e-12, "{got:e}");
        // The old off-by-one produced exactly half.
        assert!((got / -8.910_894e-6 - 2.0).abs() < 1e-4, "{got:e}");
    }

    /// Subnormals must survive a write/read round trip, otherwise writer-built
    /// fixtures can never exercise the subnormal decode path.
    #[test]
    fn f16_subnormal_roundtrip() {
        for man in [1u16, 2, 17, 299, 512, 1022, 1023] {
            for sign in [0u16, 0x8000] {
                let bits = sign | man;
                let v = f16_to_f32(bits);
                assert_eq!(f32_to_f16(v), bits, "man {man} sign {sign:#06x}");
            }
        }
        // Below the smallest subnormal magnitude, flush to zero.
        assert_eq!(f32_to_f16(2.0f32.powi(-26)), 0);
        assert_eq!(f32_to_f16(-2.0f32.powi(-26)), 0x8000);
    }

    #[test]
    fn roundtrip_normals() {
        for &v in &[1.0 / 127.0, 0.125, 1.0, 3.5, -2.0] {
            let back = f16_to_f32(f32_to_f16(v));
            let rel = (back - v).abs() / (1.0 + v.abs());
            assert!(rel * 1000.0 < 1.0, "{v} -> {back}");
        }
    }

    #[test]
    fn bf16_known_values() {
        assert_eq!(bf16_to_f32(0x0000), 0.0);
        assert_eq!(bf16_to_f32(0x3f80), 1.0);
        assert_eq!(bf16_to_f32(0xbf80), -1.0);
        assert_eq!(bf16_to_f32(0x3f00), 0.5);
        assert_eq!(f32_to_bf16(1.0), 0x3f80);
        assert_eq!(f32_to_bf16(-1.0), 0xbf80);
        assert_eq!(f32_to_bf16(0.5), 0x3f00);
        assert_eq!(load_bf16_le(&[0x80, 0x3f]), Some(1.0));
        assert_eq!(store_bf16_le(1.0), [0x80, 0x3f]);
    }

    #[test]
    fn bf16_roundtrip_normals() {
        for &v in &[1.0 / 127.0, 0.125, 1.0, 3.5, -2.0] {
            let back = bf16_to_f32(f32_to_bf16(v));
            let rel = (back - v).abs() / (1.0 + v.abs());
            assert!(rel * 1000.0 < 1.0, "{v} -> {back}");
        }
    }
}
