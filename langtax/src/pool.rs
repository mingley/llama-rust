//! Std-only row dispatch. Persistent worker pools need shared job state the
//! lints forbid (`Mutex`) or `unsafe` lifetime erasure; `thread::scope` borrows
//! the GEMV buffers instead. Sequential fallback is for tests.

use std::cell::Cell;
use std::num::NonZeroUsize;
use std::thread;

thread_local! {
    static SEQUENTIAL: Cell<bool> = const { Cell::new(false) };
}

/// Cap worker count. Extra E-cores past this did not move the 1-thread C ratio.
const MAX_WORKERS: usize = 10;

/// Run `f` with row dispatch forced to a single thread.
#[cfg(test)]
pub(crate) fn with_sequential<R>(f: impl FnOnce() -> R) -> R {
    SEQUENTIAL.with(|flag| {
        let prev = flag.replace(true);
        let out = f();
        flag.set(prev);
        out
    })
}

fn sequential() -> bool {
    SEQUENTIAL.with(Cell::get)
}

fn worker_count(n_rows: usize) -> usize {
    if sequential() || n_rows <= 1 {
        return 1;
    }
    thread::available_parallelism()
        .map(NonZeroUsize::get)
        .unwrap_or(1)
        .min(n_rows)
        .min(MAX_WORKERS)
}

/// Apply `row` to each `y[i]`. One thread when forced or when `y` is a single
/// row; otherwise `thread::scope` over coarse chunks.
pub(crate) fn for_each_row(y: &mut [f32], row: impl Fn(usize, &mut f32) + Sync) {
    if y.is_empty() {
        return;
    }
    let workers = worker_count(y.len());
    if workers <= 1 {
        for (i, out) in y.iter_mut().enumerate() {
            row(i, out);
        }
        return;
    }
    let row = &row;
    let chunk = y.len().div_ceil(workers);
    thread::scope(|scope| {
        let mut base = 0usize;
        let mut joins = Vec::new();
        for piece in y.chunks_mut(chunk) {
            let start = base;
            base = base.saturating_add(piece.len());
            joins.push(scope.spawn(move || {
                for (i, out) in piece.iter_mut().enumerate() {
                    row(start.saturating_add(i), out);
                }
            }));
        }
        for join in joins {
            if let Ok(()) = join.join() {}
        }
    });
}
