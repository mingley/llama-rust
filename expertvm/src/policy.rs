//! Replacement policies for a bounded expert cache.

/// Policy used by [`crate::replay`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Policy {
    /// Uniform random victim among unleased residents.
    Random,
    /// Least-recently used.
    Lru,
    /// Least-frequently used (frequency from this replay only).
    Lfu,
    /// LRU, but prefer evicting experts from layers other than the next one.
    LayerAhead,
    /// Predict next-layer experts as last layer's set (copy-forward).
    Predictor,
    /// Belady: evict the resident whose next use is furthest (or never).
    Oracle,
}

impl Policy {
    /// CLI name.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Random => "random",
            Self::Lru => "lru",
            Self::Lfu => "lfu",
            Self::LayerAhead => "layer-ahead",
            Self::Predictor => "predictor",
            Self::Oracle => "oracle",
        }
    }

    /// All policies in the order the plan's table uses.
    #[must_use]
    pub fn all() -> [Self; 6] {
        [
            Self::Random,
            Self::Lru,
            Self::Lfu,
            Self::LayerAhead,
            Self::Predictor,
            Self::Oracle,
        ]
    }

    /// Parse a CLI name (`lru`, `layer-ahead`, …).
    pub fn parse(name: &str) -> Result<Self, crate::Error> {
        match name {
            "random" => Ok(Self::Random),
            "lru" => Ok(Self::Lru),
            "lfu" => Ok(Self::Lfu),
            "layer-ahead" => Ok(Self::LayerAhead),
            "predictor" => Ok(Self::Predictor),
            "oracle" => Ok(Self::Oracle),
            _ => Err(crate::Error::Trace("unknown policy")),
        }
    }
}
