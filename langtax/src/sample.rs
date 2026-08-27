//! Temperature, top-k, top-p, and repeat-penalty sampling.
//!
//! Seedless greedy (`temperature <= 0`) is argmax of the penalized logits and
//! does not use an RNG. Stochastic sampling (`temperature > 0`) requires a
//! seed and draws from a filtered categorical with SplitMix64.
//!
//! Order: unique-id repeat penalty → (greedy return) → temperature → softmax →
//! top-k → top-p → renormalize → categorical. Repeat penalty is the
//! HuggingFace / llama.cpp rule: `logit > 0` then `/= penalty`, else `*=`.

use std::cmp::Ordering;

/// Failure while reading sample knobs or drawing a token.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SampleError {
    /// Logit vector was empty.
    Empty,
    /// `temperature > 0` but no seed was given.
    NeedSeed,
    /// A knob is NaN, infinite, or otherwise unusable. Carries the knob name.
    Invalid(String),
}

impl std::fmt::Display for SampleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Empty => write!(f, "empty logits"),
            Self::NeedSeed => write!(f, "temperature > 0 requires a seed"),
            Self::Invalid(what) => write!(f, "invalid sample params: {what}"),
        }
    }
}

impl std::error::Error for SampleError {}

/// Sampling knobs. [`SampleParams::greedy`] is seedless argmax.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SampleParams {
    /// Divide logits by this. `<= 0` is seedless greedy (RNG unused).
    pub temperature: f32,
    /// Keep this many highest probabilities. `0` disables.
    pub top_k: usize,
    /// Nucleus cutoff on remaining mass. `>= 1` disables. `0` keeps one token.
    pub top_p: f32,
    /// Repeat penalty on unique ids in `prev`. `1` disables. Must be `> 0`.
    pub repeat_penalty: f32,
    /// SplitMix64 seed. Required when `temperature > 0`.
    pub seed: Option<u64>,
}

impl SampleParams {
    /// Seedless greedy: `temperature = 0`, filters off, no seed.
    pub const fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            repeat_penalty: 1.0,
            seed: None,
        }
    }

    /// `true` when the draw is argmax (no RNG).
    pub fn is_greedy(self) -> bool {
        self.temperature <= 0.0
    }

    fn validate(self) -> Result<(), SampleError> {
        if !self.temperature.is_finite() {
            return Err(SampleError::Invalid("temperature".into()));
        }
        if !self.top_p.is_finite() || self.top_p < 0.0 {
            return Err(SampleError::Invalid("top_p".into()));
        }
        if !self.repeat_penalty.is_finite() || self.repeat_penalty <= 0.0 {
            return Err(SampleError::Invalid("repeat_penalty".into()));
        }
        if self.temperature > 0.0 && self.seed.is_none() {
            return Err(SampleError::NeedSeed);
        }
        Ok(())
    }
}

impl Default for SampleParams {
    fn default() -> Self {
        Self::greedy()
    }
}

/// Stateful wrapper that advances SplitMix64 across tokens.
pub struct Sampler {
    params: SampleParams,
    rng: Option<u64>,
}

impl Sampler {
    /// Validate knobs and, for a stochastic draw, load the seed into the RNG.
    pub fn new(params: SampleParams) -> Result<Self, SampleError> {
        params.validate()?;
        let rng = if params.is_greedy() {
            None
        } else {
            Some(params.seed.ok_or(SampleError::NeedSeed)?)
        };
        Ok(Self { params, rng })
    }

    /// Next token id from `logits`, applying the repeat penalty to `prev`.
    pub fn sample(&mut self, logits: &[f32], prev: &[u32]) -> Result<u32, SampleError> {
        sample_next(&self.params, self.rng.as_mut(), logits, prev)
    }
}

/// Pick the next token id from `logits`.
///
/// `rng` is advanced once per stochastic draw. Greedy ignores `rng`.
pub fn sample_next(
    params: &SampleParams,
    rng: Option<&mut u64>,
    logits: &[f32],
    prev: &[u32],
) -> Result<u32, SampleError> {
    params.validate()?;
    if logits.is_empty() {
        return Err(SampleError::Empty);
    }
    let mut work = logits.to_vec();
    apply_repeat_penalty(&mut work, prev, params.repeat_penalty);
    if params.is_greedy() {
        return Ok(argmax(&work));
    }
    if work.iter().any(|v| !v.is_finite()) {
        return Err(SampleError::Invalid("logit".into()));
    }
    apply_temperature(&mut work, params.temperature);
    let mut probs = softmax(&work);
    keep_top_k(&mut probs, params.top_k);
    keep_top_p(&mut probs, params.top_p);
    renormalize(&mut probs);
    if !probs.iter().any(|p| *p > 0.0) {
        return Ok(argmax(&work));
    }
    let state = rng.ok_or(SampleError::NeedSeed)?;
    let u = splitmix_unit(state);
    Ok(categorical(&probs, u))
}

fn apply_repeat_penalty(logits: &mut [f32], prev: &[u32], penalty: f32) {
    if penalty == 1.0 {
        return;
    }
    let mut seen = vec![false; logits.len()];
    for id in prev {
        let Ok(i) = usize::try_from(*id) else {
            continue;
        };
        let Some(flag) = seen.get_mut(i) else {
            continue;
        };
        if *flag {
            continue;
        }
        *flag = true;
        let Some(v) = logits.get_mut(i) else { continue };
        if *v > 0.0 {
            *v /= penalty;
        } else {
            *v *= penalty;
        }
    }
}

fn apply_temperature(logits: &mut [f32], temperature: f32) {
    if temperature == 1.0 {
        return;
    }
    for v in logits.iter_mut() {
        *v /= temperature;
    }
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let m = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut out: Vec<f32> = logits.iter().map(|v| (v - m).exp()).collect();
    let s: f32 = out.iter().copied().sum();
    if s > 0.0 {
        for v in &mut out {
            *v /= s;
        }
    }
    out
}

fn keep_top_k(probs: &mut [f32], k: usize) {
    if k == 0 || k >= probs.len() {
        return;
    }
    let order = sort_ids_desc(probs);
    let mut keep = vec![false; probs.len()];
    for idx in order.iter().take(k) {
        if let Some(slot) = keep.get_mut(*idx) {
            *slot = true;
        }
    }
    zero_unkept(probs, &keep);
}

fn keep_top_p(probs: &mut [f32], p: f32) {
    if p >= 1.0 {
        return;
    }
    let order = sort_ids_desc(probs);
    let mut keep = vec![false; probs.len()];
    let mut cum = 0.0f32;
    for idx in &order {
        let Some(&pr) = probs.get(*idx) else { continue };
        if pr <= 0.0 && cum > 0.0 {
            break;
        }
        if let Some(slot) = keep.get_mut(*idx) {
            *slot = true;
        }
        cum += pr;
        if cum >= p {
            break;
        }
    }
    if let Some(&first) = order.first() {
        if let Some(slot) = keep.get_mut(first) {
            *slot = true;
        }
    }
    zero_unkept(probs, &keep);
}

fn zero_unkept(probs: &mut [f32], keep: &[bool]) {
    for (pr, kept) in probs.iter_mut().zip(keep.iter()) {
        if !kept {
            *pr = 0.0;
        }
    }
}

fn renormalize(probs: &mut [f32]) {
    let s: f32 = probs.iter().copied().sum();
    if s > 0.0 {
        for p in probs.iter_mut() {
            *p /= s;
        }
    }
}

fn sort_ids_desc(probs: &[f32]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..probs.len()).collect();
    order.sort_by(|&a, &b| cmp_prob_desc(probs, a, b));
    order
}

fn cmp_prob_desc(probs: &[f32], a: usize, b: usize) -> Ordering {
    let pa = probs.get(a).copied().unwrap_or(0.0);
    let pb = probs.get(b).copied().unwrap_or(0.0);
    match pb.partial_cmp(&pa) {
        Some(ord) if ord != Ordering::Equal => ord,
        _ => a.cmp(&b),
    }
}

fn categorical(probs: &[f32], u: f32) -> u32 {
    let mut acc = 0.0f32;
    let mut last = 0usize;
    for (i, p) in probs.iter().enumerate() {
        if *p <= 0.0 {
            continue;
        }
        last = i;
        acc += *p;
        if u < acc {
            return u32::try_from(i).unwrap_or(0);
        }
    }
    u32::try_from(last).unwrap_or(0)
}

/// First index of the strict maximum. Ties keep the earlier index. Empty → 0.
pub fn argmax(x: &[f32]) -> u32 {
    let mut best_i = 0usize;
    let mut best = f32::NEG_INFINITY;
    for (i, v) in x.iter().enumerate() {
        if *v > best {
            best = *v;
            best_i = i;
        }
    }
    u32::try_from(best_i).unwrap_or(0)
}

/// SplitMix64 step (Steele, Lea, Flood). Returns the 64-bit output.
pub fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn splitmix_unit(state: &mut u64) -> f32 {
    u24_to_unit(u32::try_from(splitmix64(state) >> 40).unwrap_or(0))
}

fn u24_to_unit(bits24: u32) -> f32 {
    let capped = bits24.min(16_777_215);
    let hi = u16::try_from(capped >> 8).unwrap_or(0);
    let lo = u8::try_from(capped & 0xFF).unwrap_or(0);
    (f32::from(hi) * 256.0 + f32::from(lo)) / 16_777_216.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode::{generate, greedy_generate, tiny_llama_gguf};
    use crate::{load_gguf, Llama, Tokenizer};

    /// Independent sampler: candidate structs, unique-id list, same published math.
    struct Cand {
        id: usize,
        logit: f32,
        prob: f32,
    }

    fn oracle_argmax(cs: &[Cand]) -> u32 {
        let mut best_i = 0usize;
        let mut best = f32::NEG_INFINITY;
        for c in cs {
            if c.logit > best {
                best = c.logit;
                best_i = c.id;
            }
        }
        u32::try_from(best_i).unwrap_or(0)
    }

    fn oracle_unique(prev: &[u32], n: usize) -> Vec<usize> {
        let mut out = Vec::new();
        for id in prev {
            let Ok(i) = usize::try_from(*id) else {
                continue;
            };
            if i >= n || out.contains(&i) {
                continue;
            }
            out.push(i);
        }
        out
    }

    fn oracle_next(
        params: &SampleParams,
        rng: Option<&mut u64>,
        logits: &[f32],
        prev: &[u32],
    ) -> Result<u32, SampleError> {
        params.validate()?;
        if logits.is_empty() {
            return Err(SampleError::Empty);
        }
        let mut cs: Vec<Cand> = logits
            .iter()
            .enumerate()
            .map(|(id, logit)| Cand {
                id,
                logit: *logit,
                prob: 0.0,
            })
            .collect();
        if params.repeat_penalty != 1.0 {
            for i in oracle_unique(prev, cs.len()) {
                let Some(c) = cs.iter_mut().find(|c| c.id == i) else {
                    continue;
                };
                if c.logit > 0.0 {
                    c.logit /= params.repeat_penalty;
                } else {
                    c.logit *= params.repeat_penalty;
                }
            }
        }
        if params.is_greedy() {
            return Ok(oracle_argmax(&cs));
        }
        if cs.iter().any(|c| !c.logit.is_finite()) {
            return Err(SampleError::Invalid("logit".into()));
        }
        if params.temperature != 1.0 {
            for c in &mut cs {
                c.logit /= params.temperature;
            }
        }
        let m = cs.iter().map(|c| c.logit).fold(f32::NEG_INFINITY, f32::max);
        let mut zsum = 0.0f32;
        for c in &mut cs {
            c.prob = (c.logit - m).exp();
            zsum += c.prob;
        }
        if zsum > 0.0 {
            for c in &mut cs {
                c.prob /= zsum;
            }
        }
        cs.sort_by(|a, b| match b.prob.partial_cmp(&a.prob) {
            Some(ord) if ord != Ordering::Equal => ord,
            _ => a.id.cmp(&b.id),
        });
        if params.top_k > 0 && params.top_k < cs.len() {
            cs.truncate(params.top_k);
        }
        if params.top_p < 1.0 {
            let mut kept = Vec::new();
            let mut cum = 0.0f32;
            for c in cs.drain(..) {
                let dead = c.prob <= 0.0 && cum > 0.0;
                if dead {
                    break;
                }
                cum += c.prob;
                let done = cum >= params.top_p;
                kept.push(c);
                if done {
                    break;
                }
            }
            cs = kept;
        }
        let mass: f32 = cs.iter().map(|c| c.prob).sum();
        if mass > 0.0 {
            for c in &mut cs {
                c.prob /= mass;
            }
        }
        let state = rng.ok_or(SampleError::NeedSeed)?;
        let u = {
            *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = *state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            let bits = u32::try_from((z ^ (z >> 31)) >> 40).unwrap_or(0);
            let hi = u16::try_from(bits.min(16_777_215) >> 8).unwrap_or(0);
            let lo = u8::try_from(bits.min(16_777_215) & 0xFF).unwrap_or(0);
            (f32::from(hi) * 256.0 + f32::from(lo)) / 16_777_216.0
        };
        let mut acc = 0.0f32;
        let mut last = cs.first().map(|c| c.id).unwrap_or(0);
        for c in &cs {
            if c.prob <= 0.0 {
                continue;
            }
            last = c.id;
            acc += c.prob;
            if u < acc {
                return Ok(u32::try_from(c.id).unwrap_or(0));
            }
        }
        Ok(u32::try_from(last).unwrap_or(0))
    }

    fn sto(temp: f32, top_k: usize, top_p: f32, penalty: f32, seed: u64) -> SampleParams {
        SampleParams {
            temperature: temp,
            top_k,
            top_p,
            repeat_penalty: penalty,
            seed: Some(seed),
        }
    }

    #[test]
    fn greedy_argmax_is_first_strict_max() {
        assert_eq!(argmax(&[1.0, 3.0, 3.0, 2.0]), 1);
        assert_eq!(argmax(&[-1.0, -3.0, -2.0]), 0);
        assert_eq!(argmax(&[]), 0);
        let p = SampleParams::greedy();
        assert_eq!(sample_next(&p, None, &[1.0, 3.0, 3.0], &[]).unwrap(), 1);
        assert_eq!(p, SampleParams::default());
        assert!(p.is_greedy());
    }

    #[test]
    fn repeat_penalty_once_per_id_and_sign_rule() {
        let p = SampleParams {
            repeat_penalty: 2.0,
            ..SampleParams::greedy()
        };
        // 4 / 2 once, not 4/2/2/2. 2.0 vs 1.0 → id 1 still wins, vs 3.0 would lose.
        assert_eq!(sample_next(&p, None, &[3.0, 4.0], &[1, 1, 1]).unwrap(), 0);
        assert_eq!(sample_next(&p, None, &[0.0, 4.0], &[1, 1, 1]).unwrap(), 1);
        // negative logit is multiplied: -0.5 * 2 = -1.0, flips to id 1.
        assert_eq!(sample_next(&p, None, &[-0.5, -0.6], &[0]).unwrap(), 1);
        // penalty 1 is a no-op even with prev.
        let noop = SampleParams::greedy();
        assert_eq!(sample_next(&noop, None, &[1.0, 4.0], &[1, 1]).unwrap(), 1);
    }

    #[test]
    fn temperature_zero_is_seedless_and_stochastic_needs_seed() {
        let z = SampleParams {
            temperature: 0.0,
            top_k: 3,
            top_p: 0.5,
            repeat_penalty: 1.25,
            seed: None,
        };
        assert!(z.is_greedy());
        assert_eq!(sample_next(&z, None, &[0.0, 2.0, 1.0], &[0]).unwrap(), 1);
        let sto = SampleParams {
            temperature: 1.0,
            seed: None,
            ..SampleParams::greedy()
        };
        assert_eq!(
            sample_next(&sto, None, &[0.0, 1.0], &[]).unwrap_err(),
            SampleError::NeedSeed
        );
        assert!(Sampler::new(sto).is_err());
    }

    #[test]
    fn invalid_knobs_and_empty_logits_error() {
        assert_eq!(
            sample_next(&SampleParams::greedy(), None, &[], &[]).unwrap_err(),
            SampleError::Empty
        );
        let nan = SampleParams {
            temperature: f32::NAN,
            ..SampleParams::greedy()
        };
        assert!(matches!(
            sample_next(&nan, None, &[1.0], &[]),
            Err(SampleError::Invalid(_))
        ));
        let bad_p = SampleParams {
            top_p: -1.0,
            ..SampleParams::greedy()
        };
        assert!(matches!(
            sample_next(&bad_p, None, &[1.0], &[]),
            Err(SampleError::Invalid(_))
        ));
        let bad_r = SampleParams {
            repeat_penalty: 0.0,
            ..SampleParams::greedy()
        };
        assert!(matches!(
            sample_next(&bad_r, None, &[1.0], &[]),
            Err(SampleError::Invalid(_))
        ));
    }

    #[test]
    fn splitmix64_matches_published_first_outputs() {
        let mut s = 0u64;
        assert_eq!(splitmix64(&mut s), 0xe220_a839_7b1d_cdaf);
        assert_eq!(splitmix64(&mut s), 0x6e78_9e6a_a1b9_65f4);
        assert_eq!(splitmix64(&mut s), 0x06c4_5d18_8009_454f);
        assert_eq!(splitmix64(&mut s) >> 40, 16_288_696);
        let mut one = 1u64;
        assert_eq!(splitmix64(&mut one), 0x910a_2dec_8902_5cc1);
    }

    #[test]
    fn categorical_thresholds_and_top_filters() {
        // Equal logits → uniform. top_k=1 is the first id after desc+index sort.
        let k1 = sto(1.0, 1, 1.0, 1.0, 0);
        assert_eq!(
            sample_next(&k1, Some(&mut 0), &[0.0, 0.0, 0.0], &[]).unwrap(),
            0
        );
        // top_p = 0 keeps the single largest (id 2).
        let p0 = sto(1.0, 0, 0.0, 1.0, 0);
        assert_eq!(
            sample_next(&p0, Some(&mut 0), &[1.0, 2.0, 5.0], &[]).unwrap(),
            2
        );
        // One-hot after softmax: any u in [0, 1) picks the peak.
        let peak = sto(1.0, 0, 1.0, 1.0, 0);
        let mut rng = 0u64;
        assert_eq!(
            sample_next(&peak, Some(&mut rng), &[0.0, 0.0, 80.0], &[]).unwrap(),
            2
        );
    }

    #[test]
    fn sample_next_matches_independent_oracle() {
        let cases: &[(&[f32], &[u32], SampleParams)] = &[
            (&[1.0, 3.0, 2.0], &[], SampleParams::greedy()),
            (
                &[4.0, 4.0, 1.0],
                &[0, 0, 1],
                SampleParams {
                    repeat_penalty: 2.0,
                    ..SampleParams::greedy()
                },
            ),
            (
                &[-0.5, -0.6, 0.1],
                &[0],
                SampleParams {
                    repeat_penalty: 1.5,
                    ..SampleParams::greedy()
                },
            ),
            (&[0.0, 0.0, 0.0, 0.0], &[], sto(1.0, 0, 1.0, 1.0, 0)),
            (&[0.0, 0.0, 0.0, 0.0], &[], sto(1.0, 0, 1.0, 1.0, 1)),
            (&[1.0, 2.0, 3.0, 0.0], &[2], sto(0.5, 3, 0.75, 1.25, 7)),
            (&[2.0, 2.0, 2.0], &[], sto(1.0, 2, 0.5, 1.0, 11)),
            (&[5.0, 1.0, 1.0, 1.0], &[0, 3], sto(1.0, 0, 0.5, 1.5, 99)),
        ];
        for (logits, prev, params) in cases {
            let mut a = params.seed;
            let mut b = params.seed;
            let got = sample_next(params, a.as_mut(), logits, prev).expect("prod");
            let exp = oracle_next(params, b.as_mut(), logits, prev).expect("oracle");
            assert_eq!(
                got, exp,
                "logits={logits:?} prev={prev:?} params={params:?}"
            );
        }
    }

    #[test]
    fn same_seed_two_streams_match_and_can_differ() {
        let logits = [0.0f32, 0.0, 0.0, 0.0];
        let pa = sto(1.0, 0, 1.0, 1.0, 0);
        let pb = sto(1.0, 0, 1.0, 1.0, 1);
        let mut a1 = 0u64;
        let mut a2 = 0u64;
        let mut b1 = 1u64;
        assert_eq!(
            sample_next(&pa, Some(&mut a1), &logits, &[]).unwrap(),
            sample_next(&pa, Some(&mut a2), &logits, &[]).unwrap()
        );
        let ta = sample_next(&pa, Some(&mut 0u64), &logits, &[]).unwrap();
        let tb = sample_next(&pb, Some(&mut b1), &logits, &[]).unwrap();
        assert_ne!(ta, tb);
    }

    #[test]
    fn tiny_greedy_generate_matches_sample_params_and_two_runs() {
        let bytes = tiny_llama_gguf();
        let g = load_gguf(&bytes).expect("load");
        let tok = Tokenizer::from_gguf(&g).expect("tok");
        let model = Llama::from_gguf(g).expect("model");
        let a = greedy_generate(&model, &tok, "ab", 2).expect("greedy");
        let b = generate(&model, &tok, "ab", 2, &SampleParams::greedy()).expect("params");
        let c = generate(&model, &tok, "ab", 2, &SampleParams::default()).expect("default");
        assert_eq!(a, b);
        assert_eq!(a, c);
        let again = greedy_generate(&model, &tok, "ab", 2).expect("again");
        assert_eq!(a, again);
        assert!(a.contains("ab"), "{a}");
        let seeded = sto(1.0, 0, 1.0, 1.0, 1);
        let s1 = generate(&model, &tok, "ab", 2, &seeded).expect("s1");
        let s2 = generate(&model, &tok, "ab", 2, &seeded).expect("s2");
        assert_eq!(s1, s2);
    }
}
