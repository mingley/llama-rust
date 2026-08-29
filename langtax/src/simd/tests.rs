//! Differential tests: every SIMD kernel against the scalar kernel it replaces.
//!
//! # What is asserted, and why
//!
//! **Q8_0 is asserted bit-identical.** Its inner product is accumulated in
//! `i32`, where addition is exact and associative and (as the kernels argue in
//! place) nothing can overflow, so lane order cannot change the result. Only
//! the per-block `f32` scaling is floating point, and both kernels perform it
//! in the same order with the same values. Anything less than `==` there would
//! be a bug.
//!
//! **The eight float kernels are asserted within a proven bound.** They are not
//! bit-identical and cannot be: the SIMD kernels keep one accumulator per
//! vector lane, which reassociates the summation, and they use fused
//! multiply-add, which drops one intermediate rounding. Both kernels evaluate
//! the same exact-arithmetic quantity, so the difference is pure rounding.
//!
//! Write `u = 2^-24` for the binary32 unit roundoff and
//! `gamma(k) = k*u / (1 - k*u)`. Classical forward error analysis bounds the
//! deviation of sequentially summed, individually rounded products from the
//! exact value by `gamma(n+1) * sum|w_i * x_i|`. The lane-parallel order has a
//! strictly shorter dependency chain, so its own bound is no larger, and fusing
//! a multiply into an add only removes a rounding. Allowing two further
//! roundings for the dequantization of `w_i` gives
//!
//! ```text
//! |simd - scalar| <= 2 * gamma(n + 3) * sum|w_i * x_i|
//! ```
//!
//! which is what [`agreement_bound`] computes and every float test enforces.
//! Each kernel is *separately* held to `gamma(n + 3) * sum|w_i * x_i|` against
//! an `f64` reference, so a wrong kernel cannot hide behind an equally wrong
//! partner.
//!
//! The bound is stated against `sum|w_i * x_i|`, not against `|result|`,
//! because that is the quantity forward error analysis actually bounds; using
//! `|result|` would be vacuous under cancellation, and the sweeps below
//! deliberately include cancelling inputs.
//!
//! Both bounds are enforced; on top of that each test tracks the largest
//! observed `|simd - scalar| / sum|w_i * x_i|` in units of `u` and fails if it
//! exceeds [`observed_envelope_u`], which sits far under the proven bound and
//! would catch a kernel that regressed without becoming outright wrong.

use super::{
    f16_row_dot, f32_row_dot, q4_0_f32_row_dot, q4_k_f32_row_dot, q5_0_f32_row_dot,
    q5_1_f32_row_dot, q6_k_f32_row_dot, q8_0_f32_row_dot, q8_0_row_dot,
};
use crate::fp16::{f16_to_f32, f32_to_f16};
use crate::quant::{
    dequant_f16_row, dequant_f32_row, dequant_q4_0_row, dequant_q4_k_row, dequant_q5_0_row,
    dequant_q5_1_row, dequant_q6_k_row, dequant_q8_0_row, pack_f16, pack_f32, pack_q4_0_from_i4,
    pack_q4_k_block, pack_q5_0_block, pack_q5_1_block, pack_q6_k_block, pack_q8_0_block,
    scalar as sc, Q4_0_BLOCK, Q4_K_BLOCK, Q5_0_BLOCK, Q5_1_BLOCK, Q6_K_BLOCK, Q8_0_BLOCK, QK4_0,
    QK5_0, QK5_1, QK8_0, QK_K,
};
use crate::sample::splitmix64;

/// IEEE binary32 unit roundoff, `2^-24`.
const U: f64 = 5.960_464_477_539_063e-8;

/// Empirical ceiling on `|simd - scalar| / sum|w_i * x_i|`, in units of `u`.
///
/// Reassociating a sum of `n` terms moves the result by about `sqrt(n) * u`
/// relative to the term magnitudes, so the envelope scales the same way, with
/// eight times the headroom and a floor for short rows. That is far under the
/// proven `2 * gamma(n + 3)` bound (linear in `n`), tight enough to catch a
/// kernel that starts drifting, and loose enough not to chase host-specific
/// rounding.
fn observed_envelope_u(n: usize) -> f64 {
    (8.0 * (n as f64).sqrt()).max(64.0)
}

/// Bound on `|simd - scalar|` for a dot product of `n` terms whose products
/// have total magnitude `abs_sum`. See the module docs for the derivation.
fn agreement_bound(n: usize, abs_sum: f64) -> f64 {
    2.0 * single_kernel_bound(n, abs_sum)
}

/// Bound on one kernel's deviation from the exact value: `gamma(n+3) *
/// abs_sum`, where the `+3` covers the product rounding, the `n` summation
/// roundings and two roundings in dequantizing `w_i`.
fn single_kernel_bound(n: usize, abs_sum: f64) -> f64 {
    let k = (n as f64) + 3.0;
    let gamma = if k * U < 0.5 {
        k * U / (1.0 - k * U)
    } else {
        k * U
    };
    gamma * abs_sum
}

/// Exact-ish reference: the `f64` dot product and the total term magnitude.
/// `f64` rounding here is smaller than the `f32` bound by nine orders of
/// magnitude, so it does not participate in the comparison.
fn reference(weights: &[f32], x: &[f32]) -> (f64, f64) {
    let mut exact = 0.0f64;
    let mut abs_sum = 0.0f64;
    for (w, xv) in weights.iter().zip(x.iter()) {
        let term = f64::from(*w) * f64::from(*xv);
        exact += term;
        abs_sum += term.abs();
    }
    (exact, abs_sum)
}

/// Worst observed `|simd - scalar| / abs_sum`, in units of `u`, across a test.
#[derive(Default)]
struct Observed(f64);

impl Observed {
    /// Compare one kernel pair and fold the result into the running worst case.
    fn check(&mut self, label: &str, weights: &[f32], x: &[f32], scalar: f32, simd: f32) {
        let n = weights.len().min(x.len());
        let (exact, abs_sum) = reference(weights, x);
        assert!(
            scalar.is_finite() && simd.is_finite(),
            "{label}: non-finite result scalar={scalar} simd={simd}"
        );
        if abs_sum == 0.0 {
            assert_eq!(scalar, 0.0, "{label}: scalar nonzero for an all-zero row");
            assert_eq!(simd, 0.0, "{label}: simd nonzero for an all-zero row");
            return;
        }
        let single = single_kernel_bound(n, abs_sum);
        let scalar_err = (f64::from(scalar) - exact).abs();
        let simd_err = (f64::from(simd) - exact).abs();
        assert!(
            scalar_err <= single,
            "{label}: scalar off the f64 reference by {scalar_err:e}, bound {single:e}"
        );
        assert!(
            simd_err <= single,
            "{label}: simd off the f64 reference by {simd_err:e}, bound {single:e}"
        );
        let diff = (f64::from(simd) - f64::from(scalar)).abs();
        let bound = agreement_bound(n, abs_sum);
        assert!(
            diff <= bound,
            "{label}: |simd - scalar| = {diff:e} exceeds proven bound {bound:e} \
             (n={n}, abs_sum={abs_sum:e}, scalar={scalar}, simd={simd})"
        );
        let ratio = diff / abs_sum / U;
        let envelope = observed_envelope_u(n);
        assert!(
            ratio <= envelope,
            "{label}: |simd - scalar| is {ratio:.1}u of the term magnitude, \
             over the {envelope:.1}u envelope (n={n})"
        );
        if ratio > self.0 {
            self.0 = ratio;
        }
    }

    /// Report the worst case so a regression shows up in `--nocapture` runs.
    fn report(&self, label: &str) {
        eprintln!(
            "{label}: worst |simd - scalar| = {:.2}u of sum|w*x|",
            self.0
        );
    }
}

/// 24 pseudorandom bits scaled into `[0, 1)`, no lossy casts.
fn unit(state: &mut u64) -> f32 {
    let bits = u32::try_from(splitmix64(state) >> 40).unwrap_or(0);
    f32::from_bits(0x3f80_0000 | (bits & 0x007f_ffff)) - 1.0
}

/// Pseudorandom `f32` in `[-scale, scale)`.
fn signed(state: &mut u64, scale: f32) -> f32 {
    (unit(state) * 2.0 - 1.0) * scale
}

fn rand_u8(state: &mut u64) -> u8 {
    u8::try_from(splitmix64(state) & 0xff).unwrap_or(0)
}

/// Small `usize` as `f32`, exact for every count here and free of a lossy cast.
fn as_f32(n: usize) -> f32 {
    f32::from(u8::try_from(n).unwrap_or(u8::MAX))
}

fn rand_i8(state: &mut u64) -> i8 {
    i8::from_le_bytes([rand_u8(state)])
}

/// Element counts that straddle every vector width in play (4 and 8 lanes) and
/// the tail boundaries around them, plus a few wide rows.
const ELEM_LENGTHS: &[usize] = &[
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 23, 24, 25, 31, 32, 33, 63, 64,
    65, 127, 128, 129, 255, 256, 257, 511, 1000, 1024, 4096,
];

/// Super-block counts for the `QK_K`-structured dtypes.
const K_BLOCKS: &[usize] = &[0, 1, 2, 3, 4, 5, 8, 16];

/// Q8_0 block counts, including the 128-block (4096 element) reference width.
const Q8_BLOCKS: &[usize] = &[0, 1, 2, 3, 4, 5, 7, 8, 9, 16, 128];

/// The differential tests are worthless if dispatch never engages. On a host
/// this module is compiled for and whose CPU has the required features, every
/// selector must hand back a kernel.
#[test]
fn dispatch_engages_on_this_host() {
    let f32k = f32_row_dot();
    let f16k = f16_row_dot();
    let q4k = q4_k_f32_row_dot();
    let q6k = q6_k_f32_row_dot();
    let q40 = q4_0_f32_row_dot();
    let q50 = q5_0_f32_row_dot();
    let q51 = q5_1_f32_row_dot();
    let q80f = q8_0_f32_row_dot();
    let q8k = q8_0_row_dot();
    eprintln!(
        "simd dispatch: f32={} f16={} q4_0={} q4_k={} q6_k={} q5_0={} q5_1={} q8_0-f32={} q8_0={}",
        f32k.is_some(),
        f16k.is_some(),
        q40.is_some(),
        q4k.is_some(),
        q6k.is_some(),
        q50.is_some(),
        q51.is_some(),
        q80f.is_some(),
        q8k.is_some()
    );
    #[cfg(all(target_arch = "x86_64", target_endian = "little"))]
    {
        let have = std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma");
        let have_f16c = have && std::arch::is_x86_feature_detected!("f16c");
        assert_eq!(f32k.is_some(), have, "F32 dispatch disagrees with AVX2+FMA");
        assert_eq!(q4k.is_some(), have, "Q4_K dispatch disagrees with AVX2+FMA");
        assert_eq!(q6k.is_some(), have, "Q6_K dispatch disagrees with AVX2+FMA");
        // Every kernel that decodes a block scale with `VCVTPH2PS` needs F16C.
        for (got, what) in [
            (f16k.is_some(), "F16"),
            (q40.is_some(), "Q4_0"),
            (q50.is_some(), "Q5_0"),
            (q51.is_some(), "Q5_1"),
            (q80f.is_some(), "Q8_0-f32"),
            (q8k.is_some(), "Q8_0"),
        ] {
            assert_eq!(
                got, have_f16c,
                "{what} dispatch disagrees with AVX2+FMA+F16C"
            );
        }
    }
    #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
    {
        for (got, what) in [
            (f32k.is_some(), "F32"),
            (f16k.is_some(), "F16"),
            (q40.is_some(), "Q4_0"),
            (q4k.is_some(), "Q4_K"),
            (q6k.is_some(), "Q6_K"),
            (q50.is_some(), "Q5_0"),
            (q51.is_some(), "Q5_1"),
            (q80f.is_some(), "Q8_0-f32"),
            (q8k.is_some(), "Q8_0"),
        ] {
            assert!(
                got,
                "NEON is mandatory on aarch64, but {what} did not engage"
            );
        }
    }
}

/// Detection must be stable: the cache cannot hand back a different answer on a
/// later call, or a stored function pointer could outlive its precondition.
#[test]
fn detection_is_cached_and_stable() {
    let first = super::caps();
    assert_ne!(first, super::CAP_NONE, "caps() must mark itself detected");
    for _ in 0..64 {
        assert_eq!(super::caps(), first, "caps() changed between calls");
    }
}

// ---------------------------------------------------------------- F32

fn f32_row_case(n: usize, state: &mut u64) -> (Vec<u8>, Vec<f32>, Vec<f32>) {
    let w: Vec<f32> = (0..n).map(|_| signed(state, 4.0)).collect();
    let x: Vec<f32> = (0..n).map(|_| signed(state, 1.0)).collect();
    (pack_f32(&w), x, w)
}

#[test]
fn f32_matches_scalar_over_length_sweep() {
    let Some(simd) = f32_row_dot() else { return };
    let mut state = 0x5eed_0f32_u64;
    let mut obs = Observed::default();
    for &n in ELEM_LENGTHS {
        for trial in 0..4 {
            let (row, x, w) = f32_row_case(n, &mut state);
            obs.check(
                &format!("f32 n={n} trial={trial}"),
                &w,
                &x,
                sc::f32_row(&row, &x),
                simd(&row, &x),
            );
        }
    }
    obs.report("f32");
}

/// Ragged inputs: a weight row and an activation row of different lengths, and
/// weight bytes that are not a whole number of elements. Both kernels must
/// stop at `min(row.len() / 4, x.len())`.
#[test]
fn f32_matches_scalar_on_ragged_inputs() {
    let Some(simd) = f32_row_dot() else { return };
    let mut state = 0x5eed_0f33_u64;
    let mut obs = Observed::default();
    for &n in &[1usize, 7, 8, 9, 33, 64, 129] {
        for &trim in &[0usize, 1, 2, 3, 5] {
            let (mut row, x, w) = f32_row_case(n, &mut state);
            row.truncate(row.len().saturating_sub(trim));
            let live = (row.len() / 4).min(x.len());
            obs.check(
                &format!("f32 ragged n={n} trim={trim}"),
                w.get(..live).unwrap_or(&w),
                x.get(..live).unwrap_or(&x),
                sc::f32_row(&row, &x),
                simd(&row, &x),
            );
            let (row, x, w) = f32_row_case(n, &mut state);
            for short in 0..x.len().min(4) {
                let xs = x.get(..x.len() - short).unwrap_or(&x);
                let live = (row.len() / 4).min(xs.len());
                obs.check(
                    &format!("f32 short-x n={n} short={short}"),
                    w.get(..live).unwrap_or(&w),
                    xs,
                    sc::f32_row(&row, xs),
                    simd(&row, xs),
                );
            }
        }
    }
    obs.report("f32 ragged");
}

// ---------------------------------------------------------------- F16

fn f16_row_case(n: usize, state: &mut u64) -> (Vec<u8>, Vec<f32>, Vec<f32>) {
    let w: Vec<f32> = (0..n).map(|_| signed(state, 4.0)).collect();
    let x: Vec<f32> = (0..n).map(|_| signed(state, 1.0)).collect();
    let bytes = pack_f16(&w);
    // The packed row is what both kernels read, so the reference weights are
    // the round-tripped values, not the originals.
    let rounded: Vec<f32> = w.iter().map(|v| f16_to_f32(f32_to_f16(*v))).collect();
    let mut viadequant = vec![0.0f32; n];
    assert!(
        dequant_f16_row(n, &bytes, &mut viadequant).is_ok(),
        "f16 dequant rejected a {n}-element row"
    );
    assert_eq!(rounded, viadequant, "f16 round-trip disagrees with dequant");
    (bytes, x, rounded)
}

#[test]
fn f16_matches_scalar_over_length_sweep() {
    let Some(simd) = f16_row_dot() else { return };
    let mut state = 0x5eed_0f16_u64;
    let mut obs = Observed::default();
    for &n in ELEM_LENGTHS {
        for trial in 0..4 {
            let (row, x, w) = f16_row_case(n, &mut state);
            obs.check(
                &format!("f16 n={n} trial={trial}"),
                &w,
                &x,
                sc::f16_row(&row, &x),
                simd(&row, &x),
            );
        }
    }
    obs.report("f16");
}

/// Every representable binary16 bit pattern that is not a NaN or an infinity,
/// dotted one at a time against 1.0. A one-element dot is `0.0 + w * 1.0`, so
/// this pins the hardware conversion against `crate::fp16::f16_to_f32` exactly,
/// subnormals included. The one thing it cannot distinguish is the sign of
/// zero, which `0.0 + (-0.0)` collapses to `+0.0` on both paths.
#[test]
fn f16_conversion_is_bit_identical_for_all_finite_patterns() {
    let Some(simd) = f16_row_dot() else { return };
    let x = [1.0f32];
    for bits in 0u16..=u16::MAX {
        if bits & 0x7c00 == 0x7c00 {
            continue;
        }
        let row = bits.to_le_bytes();
        let want = sc::f16_row(&row, &x);
        let got = simd(&row, &x);
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "f16 pattern {bits:#06x}: simd {got} != scalar {want}"
        );
        if bits & 0x7fff != 0 {
            assert_eq!(
                want.to_bits(),
                f16_to_f32(bits).to_bits(),
                "f16 pattern {bits:#06x}: scalar dot is not the converted value"
            );
        }
    }
}

/// The same sweep run through the eight-wide vector body rather than the scalar
/// tail, so a lane mixup in the conversion cannot hide.
#[test]
fn f16_conversion_is_bit_identical_eight_wide() {
    let Some(simd) = f16_row_dot() else { return };
    let mut row = Vec::with_capacity(2 * 65536);
    let mut want = Vec::with_capacity(65536);
    for bits in 0u16..=u16::MAX {
        if bits & 0x7c00 == 0x7c00 {
            continue;
        }
        row.extend_from_slice(&bits.to_le_bytes());
        want.push(f16_to_f32(bits));
    }
    let x = vec![1.0f32; want.len()];
    let mut obs = Observed::default();
    obs.check(
        "f16 all-finite",
        &want,
        &x,
        sc::f16_row(&row, &x),
        simd(&row, &x),
    );
    // Every eight-element window: the vector body handles all eight lanes, so a
    // per-lane conversion or ordering fault shows up bit for bit.
    for start in 0..want.len().saturating_sub(8) {
        let bytes = row.get(start * 2..start * 2 + 16).unwrap();
        let xs = x.get(..8).unwrap();
        assert_eq!(
            simd(bytes, xs).to_bits(),
            sc::f16_row(bytes, xs).to_bits(),
            "f16 window at {start} differs"
        );
    }
    obs.report("f16 all-finite");
}

// ---------------------------------------------------------------- Q4_K

fn q4_k_row_case(n_blocks: usize, state: &mut u64) -> (Vec<u8>, Vec<f32>, Vec<f32>) {
    let mut row = Vec::with_capacity(n_blocks * Q4_K_BLOCK);
    for _ in 0..n_blocks {
        let d = signed(state, 0.05);
        let dmin = signed(state, 0.02);
        let mut scales = [0u8; 8];
        let mut mins = [0u8; 8];
        for s in &mut scales {
            *s = rand_u8(state) & 63;
        }
        for m in &mut mins {
            *m = rand_u8(state) & 63;
        }
        let mut qs = [0u8; QK_K];
        for q in &mut qs {
            *q = rand_u8(state) & 0x0f;
        }
        row.extend_from_slice(&pack_q4_k_block(d, dmin, &scales, &mins, &qs));
    }
    let n = n_blocks * QK_K;
    let x: Vec<f32> = (0..n).map(|_| signed(state, 1.0)).collect();
    let mut w = vec![0.0f32; n];
    assert!(
        dequant_q4_k_row(n, &row, &mut w).is_ok(),
        "q4_k dequant rejected {n_blocks} whole blocks"
    );
    (row, x, w)
}

#[test]
fn q4_k_matches_scalar_over_block_sweep() {
    let Some(simd) = q4_k_f32_row_dot() else {
        return;
    };
    let mut state = 0x5eed_4b00_u64;
    let mut obs = Observed::default();
    for &nb in K_BLOCKS {
        for trial in 0..4 {
            let (row, x, w) = q4_k_row_case(nb, &mut state);
            obs.check(
                &format!("q4_k blocks={nb} trial={trial}"),
                &w,
                &x,
                sc::q4_k_f32_row(&row, &x),
                simd(&row, &x),
            );
        }
    }
    obs.report("q4_k");
}

/// Partial trailing block bytes and a short activation row. `as_chunks` drops
/// an incomplete block, and a block whose activation super-block is missing
/// contributes nothing; both kernels must agree on where the row stops.
#[test]
fn q4_k_matches_scalar_on_ragged_inputs() {
    let Some(simd) = q4_k_f32_row_dot() else {
        return;
    };
    let mut state = 0x5eed_4b01_u64;
    let mut obs = Observed::default();
    for &nb in &[1usize, 2, 3, 4] {
        for &trim in &[1usize, 7, 143, Q4_K_BLOCK] {
            let (mut row, x, _) = q4_k_row_case(nb, &mut state);
            row.truncate(row.len().saturating_sub(trim));
            let (w, xs) = q4_k_live_terms(&row, &x);
            obs.check(
                &format!("q4_k trim={trim} blocks={nb}"),
                &w,
                xs,
                sc::q4_k_f32_row(&row, &x),
                simd(&row, &x),
            );
        }
        for &short in &[1usize, QK_K, QK_K + 1] {
            let (row, x, _) = q4_k_row_case(nb, &mut state);
            let xs = x.get(..x.len().saturating_sub(short)).unwrap_or(&[]);
            let (w, live) = q4_k_live_terms(&row, xs);
            obs.check(
                &format!("q4_k short-x={short} blocks={nb}"),
                &w,
                live,
                sc::q4_k_f32_row(&row, xs),
                simd(&row, xs),
            );
        }
    }
    obs.report("q4_k ragged");
}

/// The weights and activations a Q4_K row dot actually touches: whole blocks
/// with a whole matching activation super-block.
fn q4_k_live_terms<'a>(row: &[u8], x: &'a [f32]) -> (Vec<f32>, &'a [f32]) {
    let blocks = (row.len() / Q4_K_BLOCK).min(x.len() / QK_K);
    let n = blocks * QK_K;
    let mut w = vec![0.0f32; n];
    let whole = row.get(..blocks * Q4_K_BLOCK).unwrap_or(&[]);
    assert!(
        dequant_q4_k_row(n, whole, &mut w).is_ok(),
        "q4_k dequant rejected {blocks} whole blocks"
    );
    (w, x.get(..n).unwrap_or(&[]))
}

// ---------------------------------------------------------------- Q6_K

fn q6_k_row_case(n_blocks: usize, state: &mut u64) -> (Vec<u8>, Vec<f32>, Vec<f32>) {
    let mut row = Vec::with_capacity(n_blocks * Q6_K_BLOCK);
    for _ in 0..n_blocks {
        let d = signed(state, 0.05);
        let mut scales = [0i8; 16];
        for s in &mut scales {
            *s = rand_i8(state);
        }
        let mut qs = [0i8; QK_K];
        for q in &mut qs {
            // ggml Q6_K codes live in -32..=31.
            *q = i8::try_from(i32::from(rand_u8(state) & 63) - 32).unwrap_or(0);
        }
        row.extend_from_slice(&pack_q6_k_block(d, &scales, &qs));
    }
    let n = n_blocks * QK_K;
    let x: Vec<f32> = (0..n).map(|_| signed(state, 1.0)).collect();
    let mut w = vec![0.0f32; n];
    assert!(
        dequant_q6_k_row(n, &row, &mut w).is_ok(),
        "q6_k dequant rejected {n_blocks} whole blocks"
    );
    (row, x, w)
}

#[test]
fn q6_k_matches_scalar_over_block_sweep() {
    let Some(simd) = q6_k_f32_row_dot() else {
        return;
    };
    let mut state = 0x5eed_6b00_u64;
    let mut obs = Observed::default();
    for &nb in K_BLOCKS {
        for trial in 0..4 {
            let (row, x, w) = q6_k_row_case(nb, &mut state);
            obs.check(
                &format!("q6_k blocks={nb} trial={trial}"),
                &w,
                &x,
                sc::q6_k_f32_row(&row, &x),
                simd(&row, &x),
            );
        }
    }
    obs.report("q6_k");
}

#[test]
fn q6_k_matches_scalar_on_ragged_inputs() {
    let Some(simd) = q6_k_f32_row_dot() else {
        return;
    };
    let mut state = 0x5eed_6b01_u64;
    let mut obs = Observed::default();
    for &nb in &[1usize, 2, 3, 4] {
        for &trim in &[1usize, 7, 209, Q6_K_BLOCK] {
            let (mut row, x, _) = q6_k_row_case(nb, &mut state);
            row.truncate(row.len().saturating_sub(trim));
            let (w, xs) = q6_k_live_terms(&row, &x);
            obs.check(
                &format!("q6_k trim={trim} blocks={nb}"),
                &w,
                xs,
                sc::q6_k_f32_row(&row, &x),
                simd(&row, &x),
            );
        }
        for &short in &[1usize, QK_K, QK_K + 1] {
            let (row, x, _) = q6_k_row_case(nb, &mut state);
            let xs = x.get(..x.len().saturating_sub(short)).unwrap_or(&[]);
            let (w, live) = q6_k_live_terms(&row, xs);
            obs.check(
                &format!("q6_k short-x={short} blocks={nb}"),
                &w,
                live,
                sc::q6_k_f32_row(&row, xs),
                simd(&row, xs),
            );
        }
    }
    obs.report("q6_k ragged");
}

/// The weights and activations a Q6_K row dot actually touches.
fn q6_k_live_terms<'a>(row: &[u8], x: &'a [f32]) -> (Vec<f32>, &'a [f32]) {
    let blocks = (row.len() / Q6_K_BLOCK).min(x.len() / QK_K);
    let n = blocks * QK_K;
    let mut w = vec![0.0f32; n];
    let whole = row.get(..blocks * Q6_K_BLOCK).unwrap_or(&[]);
    assert!(
        dequant_q6_k_row(n, whole, &mut w).is_ok(),
        "q6_k dequant rejected {blocks} whole blocks"
    );
    (w, x.get(..n).unwrap_or(&[]))
}

// ---------------------------------------------------------------- Q4_0

fn q4_0_row_case(n_blocks: usize, state: &mut u64) -> (Vec<u8>, Vec<f32>, Vec<f32>) {
    let mut row = Vec::with_capacity(n_blocks * Q4_0_BLOCK);
    for _ in 0..n_blocks {
        let mut v = [0i8; QK4_0];
        for q in &mut v {
            let n = i32::from(rand_u8(state) & 0x0f);
            *q = i8::try_from(n - 8).unwrap_or(0);
        }
        row.extend_from_slice(&pack_q4_0_from_i4(signed(state, 0.05), &v));
    }
    let n = n_blocks * QK4_0;
    let x: Vec<f32> = (0..n).map(|_| signed(state, 1.0)).collect();
    let mut w = vec![0.0f32; n];
    assert!(
        dequant_q4_0_row(n, &row, &mut w).is_ok(),
        "q4_0 dequant rejected {n_blocks} whole blocks"
    );
    (row, x, w)
}

fn q4_0_live_terms<'a>(row: &[u8], x: &'a [f32]) -> (Vec<f32>, &'a [f32]) {
    let blocks = (row.len() / Q4_0_BLOCK).min(x.len() / QK4_0);
    let n = blocks * QK4_0;
    let mut w = vec![0.0f32; n];
    let whole = row.get(..blocks * Q4_0_BLOCK).unwrap_or(&[]);
    assert!(
        dequant_q4_0_row(n, whole, &mut w).is_ok(),
        "q4_0 dequant rejected {blocks} whole blocks"
    );
    (w, x.get(..n).unwrap_or(&[]))
}

#[test]
fn q4_0_matches_scalar_over_block_sweep() {
    let Some(simd) = q4_0_f32_row_dot() else {
        return;
    };
    let mut state = 0x4e4d_4000_u64;
    let mut obs = Observed::default();
    for &nb in Q8_BLOCKS {
        for trial in 0..4 {
            let (row, x, w) = q4_0_row_case(nb, &mut state);
            obs.check(
                &format!("q4_0 blocks={nb} trial={trial}"),
                &w,
                &x,
                sc::q4_0_f32_row(&row, &x),
                simd(&row, &x),
            );
        }
    }
    obs.report("q4_0");
}

#[test]
fn q4_0_matches_scalar_on_ragged_inputs() {
    let Some(simd) = q4_0_f32_row_dot() else {
        return;
    };
    let mut state = 0x4e4d_4001_u64;
    let mut obs = Observed::default();
    for &nb in &[1usize, 2, 3, 5] {
        for &trim in &[1usize, 5, 15, Q4_0_BLOCK] {
            let (mut row, x, _) = q4_0_row_case(nb, &mut state);
            row.truncate(row.len().saturating_sub(trim));
            let (w, xs) = q4_0_live_terms(&row, &x);
            obs.check(
                &format!("q4_0 trim={trim} blocks={nb}"),
                &w,
                xs,
                sc::q4_0_f32_row(&row, &x),
                simd(&row, &x),
            );
        }
        for &short in &[1usize, QK4_0, QK4_0 + 1] {
            let (row, x, _) = q4_0_row_case(nb, &mut state);
            let xs = x.get(..x.len().saturating_sub(short)).unwrap_or(&[]);
            let (w, live) = q4_0_live_terms(&row, xs);
            obs.check(
                &format!("q4_0 short-x={short} blocks={nb}"),
                &w,
                live,
                sc::q4_0_f32_row(&row, xs),
                simd(&row, xs),
            );
        }
    }
    obs.report("q4_0 ragged");
}

/// Every nibble `0..=15` in every of the 32 positions of a Q4_0 block.
#[test]
fn q4_0_covers_every_code_in_every_position() {
    let Some(simd) = q4_0_f32_row_dot() else {
        return;
    };
    let mut obs = Observed::default();
    for lane in 0..QK4_0 {
        for nibble in 0u8..16 {
            let mut v = [0i8; QK4_0];
            if let Some(slot) = v.get_mut(lane) {
                *slot = i8::try_from(i32::from(nibble) - 8).unwrap_or(0);
            }
            let row = pack_q4_0_from_i4(0.25, &v);
            let x: Vec<f32> = (0..QK4_0).map(|i| 1.0 + as_f32(i)).collect();
            let mut w = vec![0.0f32; QK4_0];
            assert!(dequant_q4_0_row(QK4_0, &row, &mut w).is_ok(), "dequant");
            let want = sc::q4_0_f32_row(&row, &x);
            let got = simd(&row, &x);
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "q4_0 nibble={nibble} lane={lane}: {got} != {want}"
            );
            obs.check(
                &format!("q4_0 nibble={nibble} lane={lane}"),
                &w,
                &x,
                want,
                got,
            );
        }
    }
    obs.report("q4_0 code coverage");
}

/// Every finite binary16 as a Q4_0 block scale.
///
/// Same reason as the Q5_0 sweep: the kernels decode `d` with `VCVTPH2PS` /
/// `FCVTL` rather than [`f16_to_f32`], so the shortcut has to be exact on
/// every pattern, not just the ones a writer can emit.
#[test]
fn q4_0_block_scales_are_bit_identical_for_all_finite_patterns() {
    let Some(simd) = q4_0_f32_row_dot() else {
        return;
    };
    let mut state = 0x4e4d_5f16_u64;
    let mut v = [0i8; QK4_0];
    for q in &mut v {
        let n = i32::from(rand_u8(&mut state) & 0x0f);
        *q = i8::try_from(n - 8).unwrap_or(0);
    }
    let x: Vec<f32> = (0..QK4_0).map(|_| signed(&mut state, 1.0)).collect();
    for bits in 0u16..=u16::MAX {
        if bits & 0x7c00 == 0x7c00 {
            continue;
        }
        let mut row = pack_q4_0_from_i4(1.0, &v);
        if let Some(slot) = row.get_mut(..2) {
            slot.copy_from_slice(&bits.to_le_bytes());
        }
        let want = sc::q4_0_f32_row(&row, &x);
        let got = simd(&row, &x);
        let d = f64::from(f16_to_f32(bits));
        let abs_sum: f64 = (0..QK4_0)
            .map(|i| {
                let q = f64::from(v.get(i).copied().unwrap_or(0));
                (q * d * f64::from(x.get(i).copied().unwrap_or(0.0))).abs()
            })
            .sum();
        assert!(
            (f64::from(got) - f64::from(want)).abs() <= agreement_bound(QK4_0, abs_sum),
            "q4_0 scale {bits:#06x}: {got} vs {want}"
        );
    }
}

// ---------------------------------------------------------------- Q5_0

fn q5_0_row_case(n_blocks: usize, state: &mut u64) -> (Vec<u8>, Vec<f32>, Vec<f32>) {
    let mut row = Vec::with_capacity(n_blocks * Q5_0_BLOCK);
    for _ in 0..n_blocks {
        let mut qs = [0u8; QK5_0];
        for q in &mut qs {
            *q = rand_u8(state) & 0x1f;
        }
        row.extend_from_slice(&pack_q5_0_block(signed(state, 0.05), &qs));
    }
    let n = n_blocks * QK5_0;
    let x: Vec<f32> = (0..n).map(|_| signed(state, 1.0)).collect();
    let mut w = vec![0.0f32; n];
    assert!(
        dequant_q5_0_row(n, &row, &mut w).is_ok(),
        "q5_0 dequant rejected {n_blocks} whole blocks"
    );
    (row, x, w)
}

/// The weights and activations a Q5_0 row dot actually touches.
fn q5_0_live_terms<'a>(row: &[u8], x: &'a [f32]) -> (Vec<f32>, &'a [f32]) {
    let blocks = (row.len() / Q5_0_BLOCK).min(x.len() / QK5_0);
    let n = blocks * QK5_0;
    let mut w = vec![0.0f32; n];
    let whole = row.get(..blocks * Q5_0_BLOCK).unwrap_or(&[]);
    assert!(
        dequant_q5_0_row(n, whole, &mut w).is_ok(),
        "q5_0 dequant rejected {blocks} whole blocks"
    );
    (w, x.get(..n).unwrap_or(&[]))
}

#[test]
fn q5_0_matches_scalar_over_block_sweep() {
    let Some(simd) = q5_0_f32_row_dot() else {
        return;
    };
    let mut state = 0x5eed_5000_u64;
    let mut obs = Observed::default();
    for &nb in Q8_BLOCKS {
        for trial in 0..4 {
            let (row, x, w) = q5_0_row_case(nb, &mut state);
            obs.check(
                &format!("q5_0 blocks={nb} trial={trial}"),
                &w,
                &x,
                sc::q5_0_f32_row(&row, &x),
                simd(&row, &x),
            );
        }
    }
    obs.report("q5_0");
}

#[test]
fn q5_0_matches_scalar_on_ragged_inputs() {
    let Some(simd) = q5_0_f32_row_dot() else {
        return;
    };
    let mut state = 0x5eed_5001_u64;
    let mut obs = Observed::default();
    for &nb in &[1usize, 2, 3, 5] {
        for &trim in &[1usize, 5, 15, Q5_0_BLOCK] {
            let (mut row, x, _) = q5_0_row_case(nb, &mut state);
            row.truncate(row.len().saturating_sub(trim));
            let (w, xs) = q5_0_live_terms(&row, &x);
            obs.check(
                &format!("q5_0 trim={trim} blocks={nb}"),
                &w,
                xs,
                sc::q5_0_f32_row(&row, &x),
                simd(&row, &x),
            );
        }
        for &short in &[1usize, QK5_0, QK5_0 + 1] {
            let (row, x, _) = q5_0_row_case(nb, &mut state);
            let xs = x.get(..x.len().saturating_sub(short)).unwrap_or(&[]);
            let (w, live) = q5_0_live_terms(&row, xs);
            obs.check(
                &format!("q5_0 short-x={short} blocks={nb}"),
                &w,
                live,
                sc::q5_0_f32_row(&row, xs),
                simd(&row, xs),
            );
        }
    }
    obs.report("q5_0 ragged");
}

/// Every five-bit code in every position of a block.
///
/// The fifth bit of element `j` lives in bit `j` of the block's `qh` word for
/// the low half and bit `j + 16` for the high half, and the two halves are
/// interleaved in memory: `qs[j]`'s low nibble feeds element `j` and its high
/// nibble feeds element `j + 16`. That is the whole novelty of Q5 against
/// Q4_0, it is where an off-by-one silently costs 16 units of magnitude, and a
/// random sweep would only ever exercise it on average. So every code
/// `0..=31` is placed in every one of the 32 positions, alone, with the rest
/// of the block zero.
#[test]
fn q5_0_covers_every_code_in_every_position() {
    let Some(simd) = q5_0_f32_row_dot() else {
        return;
    };
    let mut obs = Observed::default();
    for lane in 0..QK5_0 {
        for code in 0u8..32 {
            let mut qs = [0u8; QK5_0];
            if let Some(slot) = qs.get_mut(lane) {
                *slot = code;
            }
            let row = pack_q5_0_block(0.25, &qs);
            let x: Vec<f32> = (0..QK5_0).map(|i| 1.0 + as_f32(i)).collect();
            let mut w = vec![0.0f32; QK5_0];
            assert!(dequant_q5_0_row(QK5_0, &row, &mut w).is_ok(), "dequant");
            // `(code - 16) * 0.25 * x[lane]` is exact in binary32, so the two
            // kernels must agree to the bit here, not merely within the bound.
            let want = sc::q5_0_f32_row(&row, &x);
            let got = simd(&row, &x);
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "q5_0 code={code} lane={lane}: {got} != {want}"
            );
            obs.check(&format!("q5_0 code={code} lane={lane}"), &w, &x, want, got);
        }
    }
    obs.report("q5_0 code coverage");
}

/// Every finite binary16 as a Q5_0 block scale.
///
/// The kernels decode the scale with `VCVTPH2PS` / `FCVTL` rather than
/// [`f16_to_f32`], so the shortcut has to be exact on every pattern, not just
/// the ones a writer can emit. `pack_q5_0_block` rounds an `f32`, so the bits
/// are written directly.
#[test]
fn q5_0_block_scales_are_bit_identical_for_all_finite_patterns() {
    let Some(simd) = q5_0_f32_row_dot() else {
        return;
    };
    let mut state = 0x5eed_5f16_u64;
    let mut qs = [0u8; QK5_0];
    for q in &mut qs {
        *q = rand_u8(&mut state) & 0x1f;
    }
    let x: Vec<f32> = (0..QK5_0).map(|_| signed(&mut state, 1.0)).collect();
    for bits in 0u16..=u16::MAX {
        if bits & 0x7c00 == 0x7c00 {
            continue;
        }
        let mut row = pack_q5_0_block(1.0, &qs);
        if let Some(slot) = row.get_mut(..2) {
            slot.copy_from_slice(&bits.to_le_bytes());
        }
        // One block of 32 terms: the scalar kernel and the SIMD kernel differ
        // only by summation order, and with a single shared scale a wrong
        // conversion moves every term by the same factor. Comparing against
        // the scalar kernel scaled by the same `f16_to_f32` would be circular,
        // so the assertion is the bound, and `f16_scale` is pinned exactly by
        // `f16_conversion_is_bit_identical_for_all_finite_patterns`.
        let want = sc::q5_0_f32_row(&row, &x);
        let got = simd(&row, &x);
        let d = f64::from(f16_to_f32(bits));
        let abs_sum: f64 = (0..QK5_0)
            .map(|i| {
                let q = f64::from(qs.get(i).copied().unwrap_or(0)) - 16.0;
                (q * d * f64::from(x.get(i).copied().unwrap_or(0.0))).abs()
            })
            .sum();
        assert!(
            (f64::from(got) - f64::from(want)).abs() <= agreement_bound(QK5_0, abs_sum),
            "q5_0 scale {bits:#06x}: {got} vs {want}"
        );
    }
}

// ---------------------------------------------------------------- Q5_1

fn q5_1_row_case(n_blocks: usize, state: &mut u64) -> (Vec<u8>, Vec<f32>, Vec<f32>) {
    let mut row = Vec::with_capacity(n_blocks * Q5_1_BLOCK);
    for _ in 0..n_blocks {
        let mut qs = [0u8; QK5_1];
        for q in &mut qs {
            *q = rand_u8(state) & 0x1f;
        }
        row.extend_from_slice(&pack_q5_1_block(
            signed(state, 0.05),
            signed(state, 0.02),
            &qs,
        ));
    }
    let n = n_blocks * QK5_1;
    let x: Vec<f32> = (0..n).map(|_| signed(state, 1.0)).collect();
    let mut w = vec![0.0f32; n];
    assert!(
        dequant_q5_1_row(n, &row, &mut w).is_ok(),
        "q5_1 dequant rejected {n_blocks} whole blocks"
    );
    (row, x, w)
}

/// The weights and activations a Q5_1 row dot actually touches.
fn q5_1_live_terms<'a>(row: &[u8], x: &'a [f32]) -> (Vec<f32>, &'a [f32]) {
    let blocks = (row.len() / Q5_1_BLOCK).min(x.len() / QK5_1);
    let n = blocks * QK5_1;
    let mut w = vec![0.0f32; n];
    let whole = row.get(..blocks * Q5_1_BLOCK).unwrap_or(&[]);
    assert!(
        dequant_q5_1_row(n, whole, &mut w).is_ok(),
        "q5_1 dequant rejected {blocks} whole blocks"
    );
    (w, x.get(..n).unwrap_or(&[]))
}

#[test]
fn q5_1_matches_scalar_over_block_sweep() {
    let Some(simd) = q5_1_f32_row_dot() else {
        return;
    };
    let mut state = 0x5eed_5100_u64;
    let mut obs = Observed::default();
    for &nb in Q8_BLOCKS {
        for trial in 0..4 {
            let (row, x, w) = q5_1_row_case(nb, &mut state);
            obs.check(
                &format!("q5_1 blocks={nb} trial={trial}"),
                &w,
                &x,
                sc::q5_1_f32_row(&row, &x),
                simd(&row, &x),
            );
        }
    }
    obs.report("q5_1");
}

#[test]
fn q5_1_matches_scalar_on_ragged_inputs() {
    let Some(simd) = q5_1_f32_row_dot() else {
        return;
    };
    let mut state = 0x5eed_5101_u64;
    let mut obs = Observed::default();
    for &nb in &[1usize, 2, 3, 5] {
        for &trim in &[1usize, 5, 15, Q5_1_BLOCK] {
            let (mut row, x, _) = q5_1_row_case(nb, &mut state);
            row.truncate(row.len().saturating_sub(trim));
            let (w, xs) = q5_1_live_terms(&row, &x);
            obs.check(
                &format!("q5_1 trim={trim} blocks={nb}"),
                &w,
                xs,
                sc::q5_1_f32_row(&row, &x),
                simd(&row, &x),
            );
        }
        for &short in &[1usize, QK5_1, QK5_1 + 1] {
            let (row, x, _) = q5_1_row_case(nb, &mut state);
            let xs = x.get(..x.len().saturating_sub(short)).unwrap_or(&[]);
            let (w, live) = q5_1_live_terms(&row, xs);
            obs.check(
                &format!("q5_1 short-x={short} blocks={nb}"),
                &w,
                live,
                sc::q5_1_f32_row(&row, xs),
                simd(&row, xs),
            );
        }
    }
    obs.report("q5_1 ragged");
}

/// As [`q5_0_covers_every_code_in_every_position`]. Q5_1 shares the bit layout
/// but shifts every field along by the two bytes of `m` and drops the -16
/// bias, so the same exhaustive sweep is run against its own scalar kernel.
#[test]
fn q5_1_covers_every_code_in_every_position() {
    let Some(simd) = q5_1_f32_row_dot() else {
        return;
    };
    let mut obs = Observed::default();
    for lane in 0..QK5_1 {
        for code in 0u8..32 {
            let mut qs = [0u8; QK5_1];
            if let Some(slot) = qs.get_mut(lane) {
                *slot = code;
            }
            // A zero `m` keeps every term but one exactly zero, so a stray
            // lane cannot be masked by the offset.
            let row = pack_q5_1_block(0.25, 0.0, &qs);
            let x: Vec<f32> = (0..QK5_1).map(|i| 1.0 + as_f32(i)).collect();
            let mut w = vec![0.0f32; QK5_1];
            assert!(dequant_q5_1_row(QK5_1, &row, &mut w).is_ok(), "dequant");
            let want = sc::q5_1_f32_row(&row, &x);
            let got = simd(&row, &x);
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "q5_1 code={code} lane={lane}: {got} != {want}"
            );
            obs.check(&format!("q5_1 code={code} lane={lane}"), &w, &x, want, got);
        }
    }
    obs.report("q5_1 code coverage");
}

/// The `m` offset must be added, not fused: the scalar kernel writes
/// `q * d + m` as a multiply and then an add, so a contracted FMA would change
/// the dequantized weight. With `d` and `m` chosen so the exact product needs
/// more than 24 significant bits, a fused version differs in the last bit and
/// this catches it.
#[test]
fn q5_1_offset_is_added_not_fused() {
    let Some(simd) = q5_1_f32_row_dot() else {
        return;
    };
    let mut state = 0x5eed_5102_u64;
    for trial in 0..64 {
        let mut qs = [0u8; QK5_1];
        for q in &mut qs {
            *q = rand_u8(&mut state) & 0x1f;
        }
        // `d` with a full mantissa makes `q * d` inexact for most codes, so
        // rounding it before adding `m` is observable.
        let d = f32::from_bits(0x3f7f_ffff) * (1.0 + unit(&mut state));
        let m = f32::from_bits(0x3400_0001);
        let row = pack_q5_1_block(d, m, &qs);
        let x = vec![1.0f32; QK5_1];
        let want = sc::q5_1_f32_row(&row, &x);
        let got = simd(&row, &x);
        let mut w = vec![0.0f32; QK5_1];
        assert!(dequant_q5_1_row(QK5_1, &row, &mut w).is_ok(), "dequant");
        let (_, abs_sum) = reference(&w, &x);
        assert!(
            (f64::from(got) - f64::from(want)).abs() <= agreement_bound(QK5_1, abs_sum),
            "q5_1 fma probe trial={trial}: {got} vs {want}"
        );
    }
}

// -------------------------------------------------------- Q8_0 (f32 x)

fn q8_0_f32_row_case(n_blocks: usize, state: &mut u64) -> (Vec<u8>, Vec<f32>, Vec<f32>) {
    let mut row = Vec::with_capacity(n_blocks * Q8_0_BLOCK);
    for _ in 0..n_blocks {
        let mut qs = [0i8; QK8_0];
        for q in &mut qs {
            *q = rand_i8(state);
        }
        row.extend_from_slice(&pack_q8_0_block(signed(state, 0.1), &qs));
    }
    let n = n_blocks * QK8_0;
    let x: Vec<f32> = (0..n).map(|_| signed(state, 1.0)).collect();
    let mut w = vec![0.0f32; n];
    assert!(
        dequant_q8_0_row(n, &row, &mut w).is_ok(),
        "q8_0 dequant rejected {n_blocks} whole blocks"
    );
    (row, x, w)
}

/// The weights and activations a Q8_0 `f32` row dot actually touches.
fn q8_0_f32_live_terms<'a>(row: &[u8], x: &'a [f32]) -> (Vec<f32>, &'a [f32]) {
    let blocks = (row.len() / Q8_0_BLOCK).min(x.len() / QK8_0);
    let n = blocks * QK8_0;
    let mut w = vec![0.0f32; n];
    let whole = row.get(..blocks * Q8_0_BLOCK).unwrap_or(&[]);
    assert!(
        dequant_q8_0_row(n, whole, &mut w).is_ok(),
        "q8_0 dequant rejected {blocks} whole blocks"
    );
    (w, x.get(..n).unwrap_or(&[]))
}

/// The model-path Q8_0 kernel: `f32` activations, so unlike the
/// Q8_0-against-Q8_0 kernel it accumulates in floating point and is held to
/// the bound rather than to equality.
#[test]
fn q8_0_f32_matches_scalar_over_block_sweep() {
    let Some(simd) = q8_0_f32_row_dot() else {
        return;
    };
    let mut state = 0x5eed_80f3_u64;
    let mut obs = Observed::default();
    for &nb in Q8_BLOCKS {
        for trial in 0..4 {
            let (row, x, w) = q8_0_f32_row_case(nb, &mut state);
            obs.check(
                &format!("q8_0-f32 blocks={nb} trial={trial}"),
                &w,
                &x,
                sc::q8_0_f32_row(&row, &x),
                simd(&row, &x),
            );
        }
    }
    obs.report("q8_0-f32");
}

#[test]
fn q8_0_f32_matches_scalar_on_ragged_inputs() {
    let Some(simd) = q8_0_f32_row_dot() else {
        return;
    };
    let mut state = 0x5eed_80f4_u64;
    let mut obs = Observed::default();
    for &nb in &[1usize, 2, 3, 5] {
        for &trim in &[1usize, 7, 33, Q8_0_BLOCK] {
            let (mut row, x, _) = q8_0_f32_row_case(nb, &mut state);
            row.truncate(row.len().saturating_sub(trim));
            let (w, xs) = q8_0_f32_live_terms(&row, &x);
            obs.check(
                &format!("q8_0-f32 trim={trim} blocks={nb}"),
                &w,
                xs,
                sc::q8_0_f32_row(&row, &x),
                simd(&row, &x),
            );
        }
        for &short in &[1usize, QK8_0, QK8_0 + 1] {
            let (row, x, _) = q8_0_f32_row_case(nb, &mut state);
            let xs = x.get(..x.len().saturating_sub(short)).unwrap_or(&[]);
            let (w, live) = q8_0_f32_live_terms(&row, xs);
            obs.check(
                &format!("q8_0-f32 short-x={short} blocks={nb}"),
                &w,
                live,
                sc::q8_0_f32_row(&row, xs),
                simd(&row, xs),
            );
        }
    }
    obs.report("q8_0-f32 ragged");
}

/// Every `i8` code in every position, sign extension included. `i8::MIN` is
/// the value a zero-extending widen would turn into +128.
#[test]
fn q8_0_f32_covers_every_code_in_every_position() {
    let Some(simd) = q8_0_f32_row_dot() else {
        return;
    };
    for lane in 0..QK8_0 {
        for code in [i8::MIN, -127, -1, 0, 1, 126, i8::MAX] {
            let mut qs = [0i8; QK8_0];
            if let Some(slot) = qs.get_mut(lane) {
                *slot = code;
            }
            let row = pack_q8_0_block(0.25, &qs);
            let x: Vec<f32> = (0..QK8_0).map(|i| 1.0 + as_f32(i)).collect();
            // A single nonzero term, exact in binary32, so equality holds.
            let want = sc::q8_0_f32_row(&row, &x);
            let got = simd(&row, &x);
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "q8_0-f32 code={code} lane={lane}: {got} != {want}"
            );
        }
    }
}

// ---------------------------------------------------------------- Q8_0

fn q8_0_row_case(n_blocks: usize, state: &mut u64) -> (Vec<u8>, Vec<u8>) {
    let mut row = Vec::with_capacity(n_blocks * Q8_0_BLOCK);
    let mut x = Vec::with_capacity(n_blocks * Q8_0_BLOCK);
    for _ in 0..n_blocks {
        let mut qs = [0i8; QK8_0];
        for q in &mut qs {
            *q = rand_i8(state);
        }
        row.extend_from_slice(&pack_q8_0_block(signed(state, 0.1), &qs));
        for q in &mut qs {
            *q = rand_i8(state);
        }
        x.extend_from_slice(&pack_q8_0_block(signed(state, 0.02), &qs));
    }
    (row, x)
}

/// Q8_0 must be bit-identical, not merely close. See the module docs.
#[test]
fn q8_0_is_bit_identical_to_scalar() {
    let Some(simd) = q8_0_row_dot() else { return };
    let mut state = 0x5eed_8000_u64;
    for &nb in Q8_BLOCKS {
        for trial in 0..4 {
            let (row, x) = q8_0_row_case(nb, &mut state);
            let want = sc::q8_0_row(&row, &x);
            let got = simd(&row, &x);
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "q8_0 blocks={nb} trial={trial}: {got} != {want}"
            );
        }
    }
}

/// Q8_0 decodes its two block scales in hardware rather than through
/// [`f16_to_f32`], which is the difference between a 1.2x and a 2.6x kernel.
/// That shortcut is only sound if the hardware agrees on every pattern, so
/// every finite binary16 is used as a scale here, on both sides of the product.
#[test]
fn q8_0_block_scales_are_bit_identical_for_all_finite_patterns() {
    let Some(simd) = q8_0_row_dot() else { return };
    let mut state = 0x5eed_8f16_u64;
    let mut qs = [0i8; QK8_0];
    for q in &mut qs {
        *q = rand_i8(&mut state);
    }
    // A fixed second scale of 1.0 keeps the product equal to the swept one, so
    // the sweep is not masked by a zero or infinite partner.
    let one = pack_q8_0_block(1.0, &qs);
    for bits in 0u16..=u16::MAX {
        if bits & 0x7c00 == 0x7c00 {
            continue;
        }
        let scaled = pack_q8_0_scale_bits(bits, &qs);
        for (row, x) in [(&scaled, &one), (&one, &scaled)] {
            assert_eq!(
                simd(row, x).to_bits(),
                sc::q8_0_row(row, x).to_bits(),
                "q8_0 scale {bits:#06x}"
            );
        }
    }
}

/// A Q8_0 block with a literal binary16 scale, which `pack_q8_0_block` cannot
/// produce: it takes an `f32` and rounds, so subnormal scales flush to zero.
fn pack_q8_0_scale_bits(scale: u16, qs: &[i8; QK8_0]) -> [u8; Q8_0_BLOCK] {
    let mut block = pack_q8_0_block(1.0, qs);
    if let Some(slot) = block.get_mut(..2) {
        slot.copy_from_slice(&scale.to_le_bytes());
    }
    block
}

/// `i8::MIN` is the one input where `-x` overflows and where a saturating
/// multiply-accumulate (`_mm256_maddubs_epi16` and friends) would go wrong.
/// Both extremes are pinned here, in every position of a block.
#[test]
fn q8_0_is_bit_identical_at_the_i8_extremes() {
    let Some(simd) = q8_0_row_dot() else { return };
    for &(wq, xq) in &[
        (i8::MIN, i8::MIN),
        (i8::MIN, i8::MAX),
        (i8::MAX, i8::MIN),
        (i8::MAX, i8::MAX),
        (i8::MIN, -1i8),
        (-1i8, i8::MIN),
    ] {
        for blocks in 1..=4usize {
            let mut row = Vec::new();
            let mut x = Vec::new();
            for _ in 0..blocks {
                row.extend_from_slice(&pack_q8_0_block(0.125, &[wq; QK8_0]));
                x.extend_from_slice(&pack_q8_0_block(0.25, &[xq; QK8_0]));
            }
            let want = sc::q8_0_row(&row, &x);
            let got = simd(&row, &x);
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "q8_0 saturation probe w={wq} x={xq} blocks={blocks}"
            );
        }
    }
    // One extreme lane per position, the rest zero: catches a lane mixup that a
    // uniform block would hide.
    for lane in 0..QK8_0 {
        let mut row_qs = [0i8; QK8_0];
        let mut x_qs = [0i8; QK8_0];
        if let (Some(r), Some(xv)) = (row_qs.get_mut(lane), x_qs.get_mut(lane)) {
            *r = i8::MIN;
            *xv = i8::MAX;
        }
        let row = pack_q8_0_block(0.5, &row_qs);
        let x = pack_q8_0_block(0.5, &x_qs);
        assert_eq!(
            simd(&row, &x).to_bits(),
            sc::q8_0_row(&row, &x).to_bits(),
            "q8_0 lane {lane} differs"
        );
    }
}

/// Mismatched block counts: both kernels zip and stop at the shorter side.
#[test]
fn q8_0_is_bit_identical_on_ragged_inputs() {
    let Some(simd) = q8_0_row_dot() else { return };
    let mut state = 0x5eed_8001_u64;
    for &nb in &[1usize, 2, 5] {
        let (row, x) = q8_0_row_case(nb, &mut state);
        for &trim in &[1usize, 2, 17, Q8_0_BLOCK - 1, Q8_0_BLOCK] {
            let short_row = row.get(..row.len().saturating_sub(trim)).unwrap_or(&[]);
            assert_eq!(
                simd(short_row, &x).to_bits(),
                sc::q8_0_row(short_row, &x).to_bits(),
                "q8_0 short-row blocks={nb} trim={trim}"
            );
            let short_x = x.get(..x.len().saturating_sub(trim)).unwrap_or(&[]);
            assert_eq!(
                simd(&row, short_x).to_bits(),
                sc::q8_0_row(&row, short_x).to_bits(),
                "q8_0 short-x blocks={nb} trim={trim}"
            );
        }
    }
}

// ------------------------------------------------- degenerate inputs

/// All-zero and single-nonzero rows, where any stray lane shows up immediately.
#[test]
fn degenerate_rows_agree() {
    let mut obs = Observed::default();
    if let Some(simd) = f32_row_dot() {
        for n in [0usize, 1, 7, 8, 9, 64] {
            let w = vec![0.0f32; n];
            let x = vec![0.0f32; n];
            let row = pack_f32(&w);
            obs.check(
                &format!("f32 zeros n={n}"),
                &w,
                &x,
                sc::f32_row(&row, &x),
                simd(&row, &x),
            );
            for lane in 0..n {
                let mut w = vec![0.0f32; n];
                let mut x = vec![0.0f32; n];
                if let (Some(wv), Some(xv)) = (w.get_mut(lane), x.get_mut(lane)) {
                    *wv = 3.5;
                    *xv = -0.25;
                }
                let row = pack_f32(&w);
                assert_eq!(
                    simd(&row, &x).to_bits(),
                    sc::f32_row(&row, &x).to_bits(),
                    "f32 single-lane n={n} lane={lane}"
                );
            }
        }
    }
    if let Some(simd) = q4_0_f32_row_dot() {
        // All-zero bytes are a scale of +0.0 and nibbles of 0, which the -8
        // bias turns into -8: the dequantized weights are -0.0, not 0.0.
        let row = vec![0u8; Q4_0_BLOCK * 2];
        let x = vec![0.0f32; QK4_0 * 2];
        assert_eq!(simd(&row, &x), sc::q4_0_f32_row(&row, &x));
        assert_eq!(simd(&[], &[]), sc::q4_0_f32_row(&[], &[]));
    }
    if let Some(simd) = q4_k_f32_row_dot() {
        let row = vec![0u8; Q4_K_BLOCK * 2];
        let x = vec![0.0f32; QK_K * 2];
        assert_eq!(simd(&row, &x), sc::q4_k_f32_row(&row, &x));
        assert_eq!(simd(&[], &[]), sc::q4_k_f32_row(&[], &[]));
    }
    if let Some(simd) = q6_k_f32_row_dot() {
        let row = vec![0u8; Q6_K_BLOCK * 2];
        let x = vec![0.0f32; QK_K * 2];
        assert_eq!(simd(&row, &x), sc::q6_k_f32_row(&row, &x));
        assert_eq!(simd(&[], &[]), sc::q6_k_f32_row(&[], &[]));
    }
    if let Some(simd) = q5_0_f32_row_dot() {
        // All-zero bytes are a scale of +0.0 and codes of 0, which the -16
        // bias turns into -16: the dequantized weights are -0.0, not 0.0.
        let row = vec![0u8; Q5_0_BLOCK * 2];
        let x = vec![0.0f32; QK5_0 * 2];
        assert_eq!(simd(&row, &x), sc::q5_0_f32_row(&row, &x));
        assert_eq!(simd(&[], &[]), sc::q5_0_f32_row(&[], &[]));
    }
    if let Some(simd) = q5_1_f32_row_dot() {
        let row = vec![0u8; Q5_1_BLOCK * 2];
        let x = vec![0.0f32; QK5_1 * 2];
        assert_eq!(simd(&row, &x), sc::q5_1_f32_row(&row, &x));
        assert_eq!(simd(&[], &[]), sc::q5_1_f32_row(&[], &[]));
    }
    if let Some(simd) = q8_0_f32_row_dot() {
        let row = vec![0u8; Q8_0_BLOCK * 2];
        let x = vec![0.0f32; QK8_0 * 2];
        assert_eq!(simd(&row, &x), sc::q8_0_f32_row(&row, &x));
        assert_eq!(simd(&[], &[]), sc::q8_0_f32_row(&[], &[]));
    }
    if let Some(simd) = q8_0_row_dot() {
        let row = vec![0u8; Q8_0_BLOCK * 2];
        assert_eq!(simd(&row, &row), sc::q8_0_row(&row, &row));
        assert_eq!(simd(&[], &[]), sc::q8_0_row(&[], &[]));
    }
    obs.report("degenerate");
}

/// Cancelling inputs, where the result is near zero but the term magnitudes are
/// not. The bound is stated against `sum|w*x|` precisely so this case is still
/// meaningful.
#[test]
fn cancelling_rows_stay_within_the_bound() {
    let Some(simd) = f32_row_dot() else { return };
    let mut state = 0x5eed_ca11_u64;
    let mut obs = Observed::default();
    for &n in &[8usize, 64, 256, 4096] {
        let mut w = Vec::with_capacity(n);
        let mut x = Vec::with_capacity(n);
        for i in 0..n {
            let v = signed(&mut state, 1000.0);
            w.push(v);
            x.push(if i % 2 == 0 { 1.0 } else { -1.0 });
        }
        // Force near-total cancellation by mirroring the first half.
        for i in 0..n / 2 {
            let mirrored = w.get(i).copied().unwrap_or(0.0);
            if let Some(slot) = w.get_mut(n - 1 - i) {
                *slot = if (n - 1 - i) % 2 == i % 2 {
                    -mirrored
                } else {
                    mirrored
                };
            }
        }
        let row = pack_f32(&w);
        obs.check(
            &format!("f32 cancelling n={n}"),
            &w,
            &x,
            sc::f32_row(&row, &x),
            simd(&row, &x),
        );
    }
    obs.report("f32 cancelling");
}

/// Wide dynamic range: one huge term among many tiny ones, and the reverse.
#[test]
fn wide_dynamic_range_stays_within_the_bound() {
    let Some(simd) = f32_row_dot() else { return };
    let mut state = 0x5eed_d17a_u64;
    let mut obs = Observed::default();
    for &n in &[9usize, 33, 257, 1024] {
        for &spike in &[0usize, 1, 7, 8] {
            if spike >= n {
                continue;
            }
            let mut w: Vec<f32> = (0..n).map(|_| signed(&mut state, 1e-6)).collect();
            let x: Vec<f32> = (0..n).map(|_| signed(&mut state, 1e-6)).collect();
            if let Some(slot) = w.get_mut(spike) {
                *slot = 1e12;
            }
            let row = pack_f32(&w);
            obs.check(
                &format!("f32 spike n={n} at={spike}"),
                &w,
                &x,
                sc::f32_row(&row, &x),
                simd(&row, &x),
            );
        }
    }
    obs.report("f32 dynamic range");
}

// ------------------------------------------------- GEMV / GEMM wiring

/// The dispatch really is wired into the public entry points: a multi-row GEMV
/// must reproduce the per-row scalar kernel for every row.
#[test]
fn gemv_entry_points_match_row_scalars() {
    use crate::quant::{gemv_f16, gemv_f32, gemv_q4_k_f32, gemv_q6_k_f32, gemv_q8_0};

    let mut state = 0x5eed_9e11_u64;
    let n_rows = 5usize;

    let n = 320usize;
    let mut rows = Vec::new();
    let mut w_bytes = Vec::new();
    let x: Vec<f32> = (0..n).map(|_| signed(&mut state, 1.0)).collect();
    for _ in 0..n_rows {
        let vals: Vec<f32> = (0..n).map(|_| signed(&mut state, 2.0)).collect();
        let packed = pack_f32(&vals);
        w_bytes.extend_from_slice(&packed);
        rows.push(packed);
    }
    let mut y = vec![0.0f32; n_rows];
    gemv_f32(n, &w_bytes, &x, &mut y).unwrap();
    for (r, row) in rows.iter().enumerate() {
        let scalar = sc::f32_row(row, &x);
        let got = y.get(r).copied().unwrap_or(f32::NAN);
        let mut w = vec![0.0f32; n];
        dequant_f32_row(n, row, &mut w).unwrap();
        let (_, abs_sum) = reference(&w, &x);
        assert!(
            (f64::from(got) - f64::from(scalar)).abs() <= agreement_bound(n, abs_sum),
            "gemv_f32 row {r}: {got} vs {scalar}"
        );
    }

    let mut w_bytes = Vec::new();
    let mut rows = Vec::new();
    for _ in 0..n_rows {
        let vals: Vec<f32> = (0..n).map(|_| signed(&mut state, 2.0)).collect();
        let packed = pack_f16(&vals);
        w_bytes.extend_from_slice(&packed);
        rows.push(packed);
    }
    let mut y = vec![0.0f32; n_rows];
    gemv_f16(n, &w_bytes, &x, &mut y).unwrap();
    for (r, row) in rows.iter().enumerate() {
        let scalar = sc::f16_row(row, &x);
        let got = y.get(r).copied().unwrap_or(f32::NAN);
        let mut w = vec![0.0f32; n];
        dequant_f16_row(n, row, &mut w).unwrap();
        let (_, abs_sum) = reference(&w, &x);
        assert!(
            (f64::from(got) - f64::from(scalar)).abs() <= agreement_bound(n, abs_sum),
            "gemv_f16 row {r}: {got} vs {scalar}"
        );
    }

    let nk = 2 * QK_K;
    let mut w_bytes = Vec::new();
    let mut rows = Vec::new();
    let mut xk = Vec::new();
    for r in 0..n_rows {
        let (row, x_case, _) = q4_k_row_case(2, &mut state);
        if r == 0 {
            xk = x_case;
        }
        w_bytes.extend_from_slice(&row);
        rows.push(row);
    }
    let mut y = vec![0.0f32; n_rows];
    gemv_q4_k_f32(nk, &w_bytes, &xk, &mut y).unwrap();
    for (r, row) in rows.iter().enumerate() {
        let scalar = sc::q4_k_f32_row(row, &xk);
        let got = y.get(r).copied().unwrap_or(f32::NAN);
        let mut w = vec![0.0f32; nk];
        dequant_q4_k_row(nk, row, &mut w).unwrap();
        let (_, abs_sum) = reference(&w, &xk);
        assert!(
            (f64::from(got) - f64::from(scalar)).abs() <= agreement_bound(nk, abs_sum),
            "gemv_q4_k_f32 row {r}: {got} vs {scalar}"
        );
    }

    let mut w_bytes = Vec::new();
    let mut rows = Vec::new();
    let mut xk = Vec::new();
    for r in 0..n_rows {
        let (row, x_case, _) = q6_k_row_case(2, &mut state);
        if r == 0 {
            xk = x_case;
        }
        w_bytes.extend_from_slice(&row);
        rows.push(row);
    }
    let mut y = vec![0.0f32; n_rows];
    gemv_q6_k_f32(nk, &w_bytes, &xk, &mut y).unwrap();
    for (r, row) in rows.iter().enumerate() {
        let scalar = sc::q6_k_f32_row(row, &xk);
        let got = y.get(r).copied().unwrap_or(f32::NAN);
        let mut w = vec![0.0f32; nk];
        dequant_q6_k_row(nk, row, &mut w).unwrap();
        let (_, abs_sum) = reference(&w, &xk);
        assert!(
            (f64::from(got) - f64::from(scalar)).abs() <= agreement_bound(nk, abs_sum),
            "gemv_q6_k_f32 row {r}: {got} vs {scalar}"
        );
    }

    let n8 = 4 * QK8_0;
    let mut w_bytes = Vec::new();
    let mut rows = Vec::new();
    let mut x8 = Vec::new();
    for r in 0..n_rows {
        let (row, x_case) = q8_0_row_case(4, &mut state);
        if r == 0 {
            x8 = x_case;
        }
        w_bytes.extend_from_slice(&row);
        rows.push(row);
    }
    let mut y = vec![0.0f32; n_rows];
    gemv_q8_0(n8, &w_bytes, &x8, &mut y).unwrap();
    for (r, row) in rows.iter().enumerate() {
        assert_eq!(
            y.get(r).copied().unwrap_or(f32::NAN).to_bits(),
            sc::q8_0_row(row, &x8).to_bits(),
            "gemv_q8_0 row {r} is not bit-identical"
        );
    }
}

/// The model-path entry points for the four 32-element-block dtypes that
/// carry the bulk of a real checkpoint plus Q4_0. `gemv_q8_0_f32` in
/// particular is a *different* function from `gemv_q8_0`: the former takes
/// `f32` activations and is what `output.weight` and `attn_v` go through, the
/// latter takes a quantized activation row and belongs to the GEMV benchmark.
#[test]
fn model_path_gemv_entry_points_match_row_scalars() {
    use crate::quant::{gemv_q4_0_f32, gemv_q5_0_f32, gemv_q5_1_f32, gemv_q8_0_f32};

    let mut state = 0x5eed_9e12_u64;
    let n_rows = 5usize;
    let n = 4 * QK5_0;

    for dtype in 0..4u8 {
        let mut w_bytes = Vec::new();
        let mut rows = Vec::new();
        let mut x = Vec::new();
        for r in 0..n_rows {
            let (row, x_case, _) = match dtype {
                0 => q5_0_row_case(4, &mut state),
                1 => q5_1_row_case(4, &mut state),
                2 => q8_0_f32_row_case(4, &mut state),
                _ => q4_0_row_case(4, &mut state),
            };
            if r == 0 {
                x = x_case;
            }
            w_bytes.extend_from_slice(&row);
            rows.push(row);
        }
        let mut y = vec![0.0f32; n_rows];
        let call = match dtype {
            0 => gemv_q5_0_f32(n, &w_bytes, &x, &mut y),
            1 => gemv_q5_1_f32(n, &w_bytes, &x, &mut y),
            2 => gemv_q8_0_f32(n, &w_bytes, &x, &mut y),
            _ => gemv_q4_0_f32(n, &w_bytes, &x, &mut y),
        };
        assert!(call.is_ok(), "dtype {dtype} gemv rejected its own fixture");
        for (r, row) in rows.iter().enumerate() {
            let (scalar, w) = match dtype {
                0 => (sc::q5_0_f32_row(row, &x), q5_0_live_terms(row, &x).0),
                1 => (sc::q5_1_f32_row(row, &x), q5_1_live_terms(row, &x).0),
                2 => (sc::q8_0_f32_row(row, &x), q8_0_f32_live_terms(row, &x).0),
                _ => (sc::q4_0_f32_row(row, &x), q4_0_live_terms(row, &x).0),
            };
            let got = y.get(r).copied().unwrap_or(f32::NAN);
            let (_, abs_sum) = reference(&w, &x);
            assert!(
                (f64::from(got) - f64::from(scalar)).abs() <= agreement_bound(n, abs_sum),
                "dtype {dtype} gemv row {r}: {got} vs {scalar}"
            );
        }
    }
}

/// The GEMM path resolves its kernel through a different table than GEMV, so
/// wiring one and not the other is an easy miss. Two tokens through
/// `gemm_*_f32` must reproduce the per-row scalar kernel for both.
#[test]
fn model_path_gemm_entry_points_match_row_scalars() {
    use crate::quant::{gemm_q4_0_f32, gemm_q5_0_f32, gemm_q5_1_f32, gemm_q8_0_f32};

    let mut state = 0x5eed_9e13_u64;
    let n_rows = 3usize;
    let n_tokens = 2usize;
    let n = 4 * QK5_0;

    for dtype in 0..4u8 {
        let mut w_bytes = Vec::new();
        let mut rows = Vec::new();
        let mut x = Vec::new();
        for _ in 0..n_rows {
            let (row, x_case, _) = match dtype {
                0 => q5_0_row_case(4, &mut state),
                1 => q5_1_row_case(4, &mut state),
                2 => q8_0_f32_row_case(4, &mut state),
                _ => q4_0_row_case(4, &mut state),
            };
            if x.len() < n * n_tokens {
                x.extend_from_slice(&x_case);
            }
            w_bytes.extend_from_slice(&row);
            rows.push(row);
        }
        let mut y = vec![0.0f32; n_rows * n_tokens];
        let call = match dtype {
            0 => gemm_q5_0_f32(n, n_tokens, &w_bytes, &x, &mut y),
            1 => gemm_q5_1_f32(n, n_tokens, &w_bytes, &x, &mut y),
            2 => gemm_q8_0_f32(n, n_tokens, &w_bytes, &x, &mut y),
            _ => gemm_q4_0_f32(n, n_tokens, &w_bytes, &x, &mut y),
        };
        assert!(call.is_ok(), "dtype {dtype} gemm rejected its own fixture");
        for (r, row) in rows.iter().enumerate() {
            for t in 0..n_tokens {
                let xt = x.get(t * n..(t + 1) * n).unwrap_or(&[]);
                let (scalar, w) = match dtype {
                    0 => (sc::q5_0_f32_row(row, xt), q5_0_live_terms(row, xt).0),
                    1 => (sc::q5_1_f32_row(row, xt), q5_1_live_terms(row, xt).0),
                    2 => (sc::q8_0_f32_row(row, xt), q8_0_f32_live_terms(row, xt).0),
                    _ => (sc::q4_0_f32_row(row, xt), q4_0_live_terms(row, xt).0),
                };
                // `gemm_f32_x` writes token-major: `y[t * n_rows + r]`.
                let got = y.get(t * n_rows + r).copied().unwrap_or(f32::NAN);
                let (_, abs_sum) = reference(&w, xt);
                assert!(
                    (f64::from(got) - f64::from(scalar)).abs() <= agreement_bound(n, abs_sum),
                    "dtype {dtype} gemm row {r} token {t}: {got} vs {scalar}"
                );
            }
        }
    }
}

// ------------------------------------------------- throughput

/// Rows and columns per benchmarked GEMV. Matches the M=K=4096 point the
/// repository already has scalar and Metal numbers for.
const BENCH_N: usize = 4096;

/// Timed sweeps per kernel, after one untimed warm-up sweep.
const BENCH_SWEEPS: usize = 4;

/// `usize` as `f64`, exact for every count here and free of a lossy cast.
fn as_f64(n: usize) -> f64 {
    let hi = u32::try_from(n >> 32).unwrap_or(u32::MAX);
    let lo = u32::try_from(n & 0xffff_ffff).unwrap_or(0);
    f64::from(hi) * 4_294_967_296.0 + f64::from(lo)
}

/// One weight matrix laid out exactly as `gemv_*` sees it: `n_rows` rows of
/// `row_bytes` contiguous bytes in a single blob.
struct BenchMat {
    w: Vec<u8>,
    row_bytes: usize,
    n_rows: usize,
}

impl BenchMat {
    fn row(&self, r: usize) -> &[u8] {
        let start = r * self.row_bytes;
        self.w.get(start..start + self.row_bytes).unwrap_or(&[])
    }

    /// Bytes of weight touched by `BENCH_SWEEPS` sweeps.
    fn swept_bytes(&self) -> f64 {
        as_f64(self.row_bytes) * as_f64(self.n_rows) * as_f64(BENCH_SWEEPS)
    }
}

/// Time `BENCH_SWEEPS` GEMV sweeps. The accumulator is returned so the whole
/// loop cannot be optimized away.
fn time_f32(dot: fn(&[u8], &[f32]) -> f32, m: &BenchMat, x: &[f32]) -> (f64, f32) {
    let mut acc = 0.0f32;
    for r in 0..m.n_rows {
        acc += dot(m.row(r), x);
    }
    let t0 = std::time::Instant::now();
    for _ in 0..BENCH_SWEEPS {
        for r in 0..m.n_rows {
            acc += dot(m.row(r), x);
        }
    }
    (t0.elapsed().as_secs_f64(), acc)
}

/// As [`time_f32`], for the Q8_0-activation kernel.
fn time_q8(dot: fn(&[u8], &[u8]) -> f32, m: &BenchMat, x: &[u8]) -> (f64, f32) {
    let mut acc = 0.0f32;
    for r in 0..m.n_rows {
        acc += dot(m.row(r), x);
    }
    let t0 = std::time::Instant::now();
    for _ in 0..BENCH_SWEEPS {
        for r in 0..m.n_rows {
            acc += dot(m.row(r), x);
        }
    }
    (t0.elapsed().as_secs_f64(), acc)
}

/// One line per kernel: GEMV rate, weight bandwidth and the speedup.
fn report_bench(kind: &str, m: &BenchMat, scalar: (f64, f32), simd: (f64, f32)) {
    let sweeps = as_f64(BENCH_SWEEPS);
    let rate = |secs: f64| if secs > 0.0 { sweeps / secs } else { 0.0 };
    let gbs = |secs: f64| {
        if secs > 0.0 {
            m.swept_bytes() / secs / 1e9
        } else {
            0.0
        }
    };
    let speedup = if simd.0 > 0.0 { scalar.0 / simd.0 } else { 0.0 };
    eprintln!(
        "kernel={kind} M={} K={BENCH_N} scalar={:.1} gemv/s ({:.1} GB/s) \
         simd={:.1} gemv/s ({:.1} GB/s) speedup={speedup:.2}x [acc {} {}]",
        m.n_rows,
        rate(scalar.0),
        gbs(scalar.0),
        rate(simd.0),
        gbs(simd.0),
        scalar.1,
        simd.1
    );
}

/// Per-kernel throughput, scalar against SIMD, measured in one process on one
/// data set so the only variable is the kernel. Ignored by default because it
/// is a measurement, not an assertion:
///
/// ```text
/// cargo test --release --lib -- --ignored --nocapture simd::tests::kernel_throughput
/// ```
#[test]
#[ignore = "measurement, not an assertion; run with --ignored --nocapture"]
fn kernel_throughput() {
    let mut state = 0xbec4_0000_u64;
    let x: Vec<f32> = (0..BENCH_N).map(|_| signed(&mut state, 1.0)).collect();

    if let Some(simd) = f32_row_dot() {
        let vals: Vec<f32> = (0..BENCH_N).map(|_| signed(&mut state, 1.0)).collect();
        let row = pack_f32(&vals);
        let m = BenchMat {
            row_bytes: row.len(),
            n_rows: BENCH_N,
            w: row.repeat(BENCH_N),
        };
        report_bench(
            "f32",
            &m,
            time_f32(sc::f32_row, &m, &x),
            time_f32(simd, &m, &x),
        );
    }

    if let Some(simd) = f16_row_dot() {
        let vals: Vec<f32> = (0..BENCH_N).map(|_| signed(&mut state, 1.0)).collect();
        let row = pack_f16(&vals);
        let m = BenchMat {
            row_bytes: row.len(),
            n_rows: BENCH_N,
            w: row.repeat(BENCH_N),
        };
        report_bench(
            "f16",
            &m,
            time_f32(sc::f16_row, &m, &x),
            time_f32(simd, &m, &x),
        );
    }

    if let Some(simd) = q4_0_f32_row_dot() {
        let (row, _, _) = q4_0_row_case(BENCH_N / QK4_0, &mut state);
        let m = BenchMat {
            row_bytes: row.len(),
            n_rows: BENCH_N,
            w: row.repeat(BENCH_N),
        };
        report_bench(
            "q4_0",
            &m,
            time_f32(sc::q4_0_f32_row, &m, &x),
            time_f32(simd, &m, &x),
        );
    }

    if let Some(simd) = q4_k_f32_row_dot() {
        let (row, _, _) = q4_k_row_case(BENCH_N / QK_K, &mut state);
        let m = BenchMat {
            row_bytes: row.len(),
            n_rows: BENCH_N,
            w: row.repeat(BENCH_N),
        };
        report_bench(
            "q4_k",
            &m,
            time_f32(sc::q4_k_f32_row, &m, &x),
            time_f32(simd, &m, &x),
        );
    }

    if let Some(simd) = q6_k_f32_row_dot() {
        let (row, _, _) = q6_k_row_case(BENCH_N / QK_K, &mut state);
        let m = BenchMat {
            row_bytes: row.len(),
            n_rows: BENCH_N,
            w: row.repeat(BENCH_N),
        };
        report_bench(
            "q6_k",
            &m,
            time_f32(sc::q6_k_f32_row, &m, &x),
            time_f32(simd, &m, &x),
        );
    }

    if let Some(simd) = q5_0_f32_row_dot() {
        let (row, _, _) = q5_0_row_case(BENCH_N / QK5_0, &mut state);
        let m = BenchMat {
            row_bytes: row.len(),
            n_rows: BENCH_N,
            w: row.repeat(BENCH_N),
        };
        report_bench(
            "q5_0",
            &m,
            time_f32(sc::q5_0_f32_row, &m, &x),
            time_f32(simd, &m, &x),
        );
    }

    if let Some(simd) = q5_1_f32_row_dot() {
        let (row, _, _) = q5_1_row_case(BENCH_N / QK5_1, &mut state);
        let m = BenchMat {
            row_bytes: row.len(),
            n_rows: BENCH_N,
            w: row.repeat(BENCH_N),
        };
        report_bench(
            "q5_1",
            &m,
            time_f32(sc::q5_1_f32_row, &m, &x),
            time_f32(simd, &m, &x),
        );
    }

    if let Some(simd) = q8_0_f32_row_dot() {
        let (row, _, _) = q8_0_f32_row_case(BENCH_N / QK8_0, &mut state);
        let m = BenchMat {
            row_bytes: row.len(),
            n_rows: BENCH_N,
            w: row.repeat(BENCH_N),
        };
        report_bench(
            "q8_0-f32",
            &m,
            time_f32(sc::q8_0_f32_row, &m, &x),
            time_f32(simd, &m, &x),
        );
    }

    if let Some(simd) = q8_0_row_dot() {
        let (row, x8) = q8_0_row_case(BENCH_N / QK8_0, &mut state);
        let m = BenchMat {
            row_bytes: row.len(),
            n_rows: BENCH_N,
            w: row.repeat(BENCH_N),
        };
        report_bench(
            "q8_0",
            &m,
            time_q8(sc::q8_0_row, &m, &x8),
            time_q8(simd, &m, &x8),
        );
    }
}
