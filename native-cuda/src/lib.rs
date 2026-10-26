//! Shape and result validation shared by the native CUDA GEMV smoke runner.

/// Checked row-major GEMV dimensions. Dimensions must fit the kernel's u32 ABI.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shape {
    /// Number of output rows.
    pub rows: u32,
    /// Number of input columns.
    pub cols: u32,
}

impl Shape {
    /// Validate dimensions and exact backing-buffer lengths before any GPU work.
    pub fn checked(rows: usize, cols: usize, weights: usize, input: usize) -> Result<Self, String> {
        if rows == 0 || cols == 0 {
            return Err("GEMV dimensions must be positive".into());
        }
        let elements = rows.checked_mul(cols).ok_or("GEMV shape overflow")?;
        // Device indexing uses u32 so the flattened index must also fit.
        let _ = u32::try_from(elements).map_err(|_| "GEMV matrix exceeds u32 indexing")?;
        if weights != elements || input != cols {
            return Err("GEMV buffer lengths do not match shape".into());
        }
        Ok(Self {
            rows: u32::try_from(rows).map_err(|_| "GEMV rows exceed u32")?,
            cols: u32::try_from(cols).map_err(|_| "GEMV columns exceed u32")?,
        })
    }
}

/// Compare GPU output against a scalar f64 oracle, rejecting NaN and infinity.
pub fn validate(actual: &[f32], expected: &[f64]) -> Result<(), String> {
    if actual.len() != expected.len() {
        return Err("GEMV output length mismatch".into());
    }
    for (row, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let tolerance = 1e-4 + 1e-4 * expected.abs();
        if !actual.is_finite()
            || !expected.is_finite()
            || (f64::from(actual) - expected).abs() > tolerance
        {
            return Err(format!(
                "GEMV mismatch at row {row}: expected {expected}, got {actual}"
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_shapes_that_would_make_device_access_unsafe() {
        assert!(Shape::checked(0, 2, 0, 2).is_err());
        assert!(Shape::checked(2, 3, 5, 3).is_err());
        assert!(Shape::checked(2, 3, 6, 2).is_err());
        assert!(Shape::checked(usize::MAX, 2, 0, 2).is_err());
        assert!(Shape::checked(17, 33, 561, 33).is_ok());
    }

    #[test]
    fn oracle_comparison_catches_nonfinite_and_wrong_results() {
        assert!(validate(&[f32::NAN], &[0.0]).is_err());
        assert!(validate(&[f32::INFINITY], &[f64::INFINITY]).is_err());
        assert!(validate(&[1.0], &[2.0]).is_err());
        assert!(validate(&[], &[0.0]).is_err());
        assert!(validate(&[1.00001, -2.0], &[1.0, -2.0]).is_ok());
    }
}
