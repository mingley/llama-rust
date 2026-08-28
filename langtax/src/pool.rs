//! Std-only row dispatch, in two shapes.
//!
//! [`for_each_row`] / [`for_each_group`] fork and join a `thread::scope` on
//! every call. That is the only shape safe Rust allows for a *borrowed* closure
//! writing into *borrowed* buffers: handing `&mut [f32]` to a thread that
//! outlives the call needs either `unsafe` lifetime erasure or shared job state
//! behind a `Mutex`, and this crate has neither. The cost is ~100 us of
//! spawn/join per call on 4 vCPUs, near enough flat in row count
//! (`bench_dispatch_overhead`), which a GEMM over a whole prompt amortises and
//! a single-token GEMV does not.
//!
//! [`Pool`] keeps its workers between calls, for the GEMV in the decode loop,
//! and costs 4-13 us for the same dispatch. It sidesteps the lifetime problem
//! by never sending a borrow at all:
//!
//! * the job is `Copy` plain data (which rows of which matrix),
//! * the weights live in the worker's own `Arc<K>`, not in the caller's frame,
//! * the input vector is a shared `Arc<Vec<f32>>` that the caller refills in
//!   place between jobs, which `Arc::get_mut` permits exactly because every
//!   worker has dropped its clone by then,
//! * each worker writes into an owned `Vec<f32>` it received and hands back.
//!
//! Everything crosses bounded `mpsc` channels, so there is no lock, no
//! `unsafe`, and nothing borrowed.
//!
//! Two things that look like details are most of the win, and the pool was
//! *slower* than the fork/join it replaced without them: the caller computes a
//! chunk instead of blocking, and workers spin before parking. Both are about
//! wake-up latency rather than throughput; see [`Pool`] and [`SPINS`].

use std::cell::Cell;
use std::num::NonZeroUsize;
use std::sync::mpsc::{sync_channel, Receiver, SyncSender, TryRecvError};
use std::sync::Arc;
use std::thread::{self, JoinHandle};

thread_local! {
    static SEQUENTIAL: Cell<bool> = const { Cell::new(false) };
}

/// Cap worker count. Extra E-cores past this did not move the 1-thread C ratio.
const MAX_WORKERS: usize = 10;

/// Restores [`SEQUENTIAL`] on the way out, so an early return inside `f`
/// cannot leave this thread wrongly marked.
struct SequentialGuard(bool);

impl Drop for SequentialGuard {
    fn drop(&mut self) {
        SEQUENTIAL.with(|flag| flag.set(self.0));
    }
}

fn enter_sequential() -> SequentialGuard {
    SequentialGuard(SEQUENTIAL.with(|flag| flag.replace(true)))
}

/// Run `f` with row dispatch forced to a single thread.
#[cfg(test)]
pub(crate) fn with_sequential<R>(f: impl FnOnce() -> R) -> R {
    let _guard = enter_sequential();
    f()
}

pub(crate) fn sequential() -> bool {
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

/// Apply `row` to each group of `group` outputs. `y.len()` must be
/// `n_rows * group`. Used by GEMM so one weight row writes every token.
pub(crate) fn for_each_group(y: &mut [f32], group: usize, row: impl Fn(usize, &mut [f32]) + Sync) {
    if y.is_empty() || group == 0 || !y.len().is_multiple_of(group) {
        return;
    }
    let n_rows = y.len() / group;
    let workers = worker_count(n_rows);
    if workers <= 1 {
        for (i, out) in y.chunks_mut(group).enumerate() {
            row(i, out);
        }
        return;
    }
    let row = &row;
    let chunk_rows = n_rows.div_ceil(workers);
    let chunk_elems = chunk_rows.saturating_mul(group);
    thread::scope(|scope| {
        let mut base = 0usize;
        let mut joins = Vec::new();
        for piece in y.chunks_mut(chunk_elems) {
            let start = base;
            let n = piece.len() / group;
            base = base.saturating_add(n);
            joins.push(scope.spawn(move || {
                for (i, out) in piece.chunks_mut(group).enumerate() {
                    row(start.saturating_add(i), out);
                }
            }));
        }
        for join in joins {
            if let Ok(()) = join.join() {}
        }
    });
}

/// Computes a contiguous slice of one matrix's rows on a [`Pool`] worker.
///
/// Implemented by the caller, which owns its weights (so the worker does not
/// borrow them) and interprets [`RowKernel::Job`].
pub(crate) trait RowKernel: Send + Sync + 'static {
    /// Which rows of which matrix. Crosses a channel, so it cannot borrow.
    type Job: Copy + Send + 'static;

    /// Write `y[i] = row(first + i) · x` for every `i` in `y`. `false` on any
    /// shape or type the kernel will not handle, which makes [`Pool::run`]
    /// report failure so the caller can compute the whole matrix itself.
    fn rows(&self, job: Self::Job, first: usize, x: &[f32], y: &mut [f32]) -> bool;
}

enum Task<J> {
    /// Rows `first .. first + out.len()`.
    Rows {
        job: J,
        first: usize,
        x: Arc<Vec<f32>>,
        out: Vec<f32>,
    },
    Stop,
}

struct Reply {
    first: usize,
    out: Vec<f32>,
    ok: bool,
}

struct Worker<J> {
    tx: SyncSender<Task<J>>,
    handle: Option<JoinHandle<()>>,
}

/// Worker threads that outlive the call, for repeated GEMV against one kernel.
///
/// Steady state allocates nothing: the input staging keeps its capacity and the
/// output buffers cycle through `spare`.
///
/// The calling thread takes a chunk itself rather than blocking, so a pool that
/// wants `n`-way parallelism spawns `n - 1` threads. Two reasons, both measured
/// (see `bench_decode_step`): `n` workers plus a blocked caller oversubscribes
/// an `n`-CPU host, and a caller that computes instead of waiting keeps its own
/// chunk off the wake-up path entirely.
pub(crate) struct Pool<K: RowKernel> {
    workers: Vec<Worker<K::Job>>,
    kernel: Arc<K>,
    /// Input staging. Between jobs its refcount is 1, which is what lets
    /// [`Pool::run`] refill it through `Arc::get_mut` without allocating.
    x: Arc<Vec<f32>>,
    /// Output buffers not currently out with a worker.
    spare: Vec<Vec<f32>>,
    done: Receiver<Reply>,
}

impl<K: RowKernel> Pool<K> {
    /// Spawn workers for `kernel`, capped the same way [`for_each_row`] caps
    /// its scope. `None` when one thread would be used anyway, or when a
    /// thread could not be spawned.
    ///
    /// `rows_hint` is the row count of the widest matrix the pool will see; it
    /// only bounds the worker count.
    pub(crate) fn new(kernel: Arc<K>, rows_hint: usize) -> Option<Self> {
        let n = worker_count(rows_hint);
        if n <= 1 {
            return None;
        }
        // `n` includes the calling thread, which takes a chunk in `run`.
        let spawn = n.saturating_sub(1);
        let (done_tx, done) = sync_channel(spawn);
        let mut workers = Vec::with_capacity(spawn);
        for _ in 0..spawn {
            // Bound 2, not 1, so `Drop` can always queue `Stop` even if a job
            // is still in flight.
            let (tx, rx) = sync_channel(2);
            let kernel = Arc::clone(&kernel);
            let done_tx = done_tx.clone();
            // Dropping `workers` here closes every job channel, so any worker
            // already spawned observes the disconnect and exits.
            let handle = thread::Builder::new()
                .name("llama-gemv".into())
                .spawn(move || worker_loop(kernel, rx, done_tx))
                .ok()?;
            workers.push(Worker {
                tx,
                handle: Some(handle),
            });
        }
        Some(Self {
            workers,
            kernel,
            x: Arc::new(Vec::new()),
            spare: Vec::new(),
            done,
        })
    }

    /// Compute rows `0 .. y.len()` of `job`, the tail on the workers and the
    /// first chunk on this thread.
    ///
    /// `false` if the pool did not produce every row (kernel refused the job, a
    /// worker is gone, or the input staging is still shared); the caller must
    /// then compute the matrix itself. `y` may have been partly written.
    pub(crate) fn run(&mut self, job: K::Job, x: &[f32], y: &mut [f32]) -> bool {
        let n_rows = y.len();
        let n = self.workers.len().saturating_add(1);
        if n_rows == 0 || self.workers.is_empty() {
            return false;
        }
        {
            let Some(buf) = Arc::get_mut(&mut self.x) else {
                return false;
            };
            buf.clear();
            buf.extend_from_slice(x);
        }
        let chunk = n_rows.div_ceil(n);
        // This thread takes `0 .. chunk`; the workers take what follows. Send
        // first so they are already running while this thread computes.
        let mine = chunk.min(n_rows);
        let mut first = mine;
        let mut sent = 0usize;
        let mut ok = true;
        for w in &self.workers {
            if first >= n_rows {
                break;
            }
            let last = first.saturating_add(chunk).min(n_rows);
            let mut out = self.spare.pop().unwrap_or_default();
            out.clear();
            out.resize(last.saturating_sub(first), 0.0);
            let task = Task::Rows {
                job,
                first,
                x: Arc::clone(&self.x),
                out,
            };
            if w.tx.send(task).is_err() {
                ok = false;
                break;
            }
            sent = sent.saturating_add(1);
            first = last;
        }
        let mut done_rows = 0usize;
        match y.get_mut(..mine) {
            Some(head) => {
                // The kernels dispatch rows themselves; keep that on this
                // thread rather than opening a second level of parallelism.
                let _guard = enter_sequential();
                if self.kernel.rows(job, 0, x, head) {
                    done_rows = mine;
                } else {
                    ok = false;
                }
            }
            None => ok = false,
        }
        for _ in 0..sent {
            let Ok(reply) = self.done.recv() else {
                // A worker vanished mid-job. The buffer it held is gone too.
                return false;
            };
            let end = reply.first.saturating_add(reply.out.len());
            match (reply.ok, y.get_mut(reply.first..end)) {
                (true, Some(dst)) => {
                    dst.copy_from_slice(&reply.out);
                    done_rows = done_rows.saturating_add(reply.out.len());
                }
                _ => ok = false,
            }
            self.spare.push(reply.out);
        }
        ok && done_rows == n_rows
    }

    /// Heap address and capacity of each buffer the pool reuses, for the
    /// steady-state allocation test in `decode`.
    #[cfg(test)]
    pub(crate) fn buffer_ids(&self) -> Vec<(usize, usize)> {
        let mut out = vec![(self.x.as_ptr() as usize, self.x.capacity())];
        // `spare` cycles, so its buffers are not in a stable order; their count
        // and total capacity are what must stop changing.
        out.push((
            self.spare.len(),
            self.spare.iter().map(Vec::capacity).sum::<usize>(),
        ));
        out
    }

    /// Number of worker threads.
    #[cfg(test)]
    pub(crate) fn workers(&self) -> usize {
        self.workers.len()
    }
}

impl<K: RowKernel> Drop for Pool<K> {
    fn drop(&mut self) {
        for w in &self.workers {
            if let Ok(()) = w.tx.send(Task::Stop) {}
        }
        for w in &mut self.workers {
            if let Some(handle) = w.handle.take() {
                if let Ok(()) = handle.join() {}
            }
        }
    }
}

/// Spins before parking. A decode step dispatches on the order of 30 GEMVs, so
/// a worker that parks between them pays a futex wake on every one; on a 4-vCPU
/// guest that latency cost more than the `thread::scope` spawn the pool exists
/// to remove (the pool was 40% *slower* on median until this was added).
/// Spinning is affordable because the pool leaves one CPU for the caller.
const SPINS: u32 = 20_000;

fn next_task<J>(rx: &Receiver<Task<J>>) -> Option<Task<J>> {
    for _ in 0..SPINS {
        match rx.try_recv() {
            Ok(task) => return Some(task),
            Err(TryRecvError::Empty) => std::hint::spin_loop(),
            Err(TryRecvError::Disconnected) => return None,
        }
    }
    rx.recv().ok()
}

fn worker_loop<K: RowKernel>(kernel: Arc<K>, rx: Receiver<Task<K::Job>>, done: SyncSender<Reply>) {
    // The kernels call `for_each_row` themselves. Keep that on this thread
    // instead of opening a second level of workers per job.
    SEQUENTIAL.with(|flag| flag.set(true));
    while let Some(task) = next_task(&rx) {
        let Task::Rows {
            job,
            first,
            x,
            mut out,
        } = task
        else {
            break;
        };
        let ok = kernel.rows(job, first, &x, &mut out);
        // Release the shared input before replying, so the caller's next
        // `Arc::get_mut` sees a refcount of 1.
        drop(x);
        if done.send(Reply { first, out, ok }).is_err() {
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// `y[r] = base[r] + first offset`, so every reply can be checked for
    /// landing at the right row.
    struct Counter {
        calls: AtomicUsize,
    }

    impl RowKernel for Counter {
        type Job = (usize, bool);

        fn rows(&self, job: Self::Job, first: usize, x: &[f32], y: &mut [f32]) -> bool {
            let (scale, ok) = job;
            let _prev = self.calls.fetch_add(1, Ordering::Relaxed);
            if !ok {
                return false;
            }
            for (i, out) in y.iter_mut().enumerate() {
                let r = first + i;
                *out = (r * scale) as f32 + x.iter().sum::<f32>();
            }
            true
        }
    }

    #[test]
    fn pool_covers_every_row_and_reuses_buffers() {
        let kernel = Arc::new(Counter {
            calls: AtomicUsize::new(0),
        });
        let Some(mut pool) = Pool::new(Arc::clone(&kernel), 4096) else {
            // Single-core host: nothing to test.
            return;
        };
        assert!(pool.workers() >= 2);
        let x = vec![1.0f32, 2.0, 3.0];
        for n_rows in [1usize, 2, 3, 7, 64, 4096] {
            let mut y = vec![f32::NAN; n_rows];
            assert!(pool.run((3, true), &x, &mut y), "n_rows {n_rows}");
            for (r, got) in y.iter().enumerate() {
                assert_eq!(*got, (r * 3) as f32 + 6.0, "row {r} of {n_rows}");
            }
        }
        let ids = pool.buffer_ids();
        for _ in 0..20 {
            let mut y = vec![0.0f32; 4096];
            assert!(pool.run((3, true), &x, &mut y));
        }
        assert_eq!(pool.buffer_ids(), ids, "pool regrew a buffer");
    }

    #[test]
    fn pool_reports_a_refusing_kernel_and_stays_usable() {
        let kernel = Arc::new(Counter {
            calls: AtomicUsize::new(0),
        });
        let Some(mut pool) = Pool::new(kernel, 256) else {
            return;
        };
        let x = vec![1.0f32];
        let mut y = vec![0.0f32; 256];
        assert!(!pool.run((1, false), &x, &mut y));
        assert!(pool.run((1, true), &x, &mut y));
        assert_eq!(y.first().copied(), Some(1.0));
    }

    #[test]
    fn pool_is_not_built_under_with_sequential() {
        let kernel = Arc::new(Counter {
            calls: AtomicUsize::new(0),
        });
        with_sequential(|| {
            assert!(Pool::new(kernel, 4096).is_none());
        });
    }

    #[test]
    fn dropping_a_pool_joins_its_workers() {
        // A leaked or blocked worker would hang the test binary rather than
        // fail, so this is really a liveness check on Drop.
        for _ in 0..8 {
            let kernel = Arc::new(Counter {
                calls: AtomicUsize::new(0),
            });
            let Some(mut pool) = Pool::new(Arc::clone(&kernel), 64) else {
                return;
            };
            let x = vec![1.0f32];
            let mut y = vec![0.0f32; 64];
            assert!(pool.run((1, true), &x, &mut y));
            drop(pool);
            assert_eq!(Arc::strong_count(&kernel), 1, "a worker outlived the pool");
        }
    }
}
