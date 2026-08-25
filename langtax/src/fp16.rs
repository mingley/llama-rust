//! IEEE-754 binary16 <-> f32. GGUF Q4_0/Q8_0 scales are `ggml_half`.

pub fn f16_to_f32(h: u16) -> f32 {
    let sign = (h as u32 & 0x8000) << 16;
    let exp = i32::from((h >> 10) & 0x1f);
    let man = u32::from(h & 0x3ff);
    let bits = if exp == 0 {
        if man == 0 {
            sign
        } else {
            let mut m = man;
            let mut e = -1i32;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3ff;
            sign | (((e + 127 - 15 + 1) as u32) << 23) | (m << 13)
        }
    } else if exp == 31 {
        sign | 0x7f80_0000 | (man << 13)
    } else {
        sign | (((exp + 127 - 15) as u32) << 23) | (man << 13)
    };
    f32::from_bits(bits)
}

pub fn f32_to_f16(f: f32) -> u16 {
    let b = f.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
    let man = (b >> 13) & 0x3ff;
    if exp <= 0 {
        sign
    } else if exp >= 31 {
        sign | 0x7c00
    } else {
        sign | ((exp as u16) << 10) | (man as u16)
    }
}

pub fn load_f16_le(bytes: &[u8]) -> f32 {
    f16_to_f32(u16::from_le_bytes([bytes[0], bytes[1]]))
}

pub fn store_f16_le(scale: f32) -> [u8; 2] {
    f32_to_f16(scale).to_le_bytes()
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

    #[test]
    fn roundtrip_normals() {
        for &v in &[0.007_874_016_f32, 0.125, 1.0, 3.5, -2.0] {
            let back = f16_to_f32(f32_to_f16(v));
            let rel = (back - v).abs() / (1.0 + v.abs());
            assert!(rel < 0.001, "{v} -> {back}");
        }
    }
}
