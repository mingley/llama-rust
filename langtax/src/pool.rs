use std::sync::LazyLock;

use rayon::{ThreadPool, ThreadPoolBuilder};

const GEMV_THREADS: usize = 10;

static POOL: LazyLock<ThreadPool> = LazyLock::new(|| {
    ThreadPoolBuilder::new()
        .num_threads(GEMV_THREADS)
        .build()
        .expect("rayon GEMV pool")
});

pub fn install<R>(f: impl FnOnce() -> R + Send) -> R
where
    R: Send,
{
    POOL.install(f)
}
