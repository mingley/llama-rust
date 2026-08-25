//! Persistent rayon pool so GEMV rows do not spawn a new pool per call.

use std::sync::OnceLock;

use rayon::{ThreadPool, ThreadPoolBuilder};

/// P-core-sized pool for the measured GEMV cells on this machine.
const GEMV_THREADS: usize = 10;

fn pool() -> Option<&'static ThreadPool> {
    static POOL: OnceLock<ThreadPool> = OnceLock::new();
    if let Some(existing) = POOL.get() {
        return Some(existing);
    }
    match ThreadPoolBuilder::new().num_threads(GEMV_THREADS).build() {
        Ok(built) => Some(POOL.get_or_init(move || built)),
        Err(_) => None,
    }
}

/// Run `f` on the GEMV pool, or on the current thread if the pool cannot be built.
pub(crate) fn install<R>(f: impl FnOnce() -> R + Send) -> R
where
    R: Send,
{
    match pool() {
        Some(p) => p.install(f),
        None => f(),
    }
}
