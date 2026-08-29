//! Paged KV: vLLM-style block tables on the reference engine.
//!
//! Dense layout stays the default (`KvCache::new` via [`crate::decode::Llama::new_cache`]).
//! [`KvPages`] stores K/V in fixed-size blocks addressed by a sequence's block
//! table. [`PagedKvPool`] is the interned arena: clone the handle so two
//! sequences hit the same completed prefixes. Writing a block with `refs > 1`
//! copy-on-writes. Logits must bit-match dense decode.

use expertvm::prefix_hash;
use std::cell::{RefCell, RefMut};
use std::collections::BTreeMap;
use std::rc::Rc;

/// Physical KV blocks plus a prefix-hash intern table.
pub(crate) struct KvPool {
    k: Vec<f32>,
    v: Vec<f32>,
    block_size: usize,
    n_layers: usize,
    n_head_kv: usize,
    hd: usize,
    cap: usize,
    refs: Vec<u32>,
    by_hash: BTreeMap<u64, u32>,
    hits: u64,
}

/// Shared interned-block arena. Clone the handle so two sequences hit the same prefixes.
///
/// Distinct from `expertvm kv` (simulated VMM pages). This is decode K/V.
#[derive(Clone)]
pub struct PagedKvPool {
    inner: Rc<RefCell<KvPool>>,
    block_size: usize,
    n_layers: usize,
}

/// One sequence's block table on a [`PagedKvPool`].
pub(crate) struct KvPages {
    pool: PagedKvPool,
    table: Vec<u32>,
}

/// Addressing for one K or V buffer: dense `max_seq` stride, or paged blocks.
pub(crate) struct KvGeom<'a> {
    pub(crate) n_head_kv: usize,
    pub(crate) hd: usize,
    pub(crate) n_layers: usize,
    pub(crate) time_stride: usize,
    pub(crate) table: Option<&'a [u32]>,
}

impl KvPool {
    fn stride(&self) -> usize {
        self.n_layers
            .saturating_mul(self.n_head_kv)
            .saturating_mul(self.block_size)
            .saturating_mul(self.hd)
    }

    fn retain(&mut self, id: u32) {
        let i = usize::try_from(id).unwrap_or(usize::MAX);
        if let Some(r) = self.refs.get_mut(i) {
            *r = r.saturating_add(1);
        }
    }

    fn release(&mut self, id: u32) {
        let i = usize::try_from(id).unwrap_or(usize::MAX);
        if let Some(r) = self.refs.get_mut(i) {
            *r = r.saturating_sub(1);
        }
    }

    fn intern(&mut self, hash: u64, id: u32) {
        if self.by_hash.get(&hash).copied() == Some(id) {
            return;
        }
        if self.by_hash.contains_key(&hash) {
            return;
        }
        let _prev = self.by_hash.insert(hash, id);
        self.retain(id);
    }

    fn lookup(&mut self, hash: u64) -> Option<u32> {
        let id = self.by_hash.get(&hash).copied()?;
        self.retain(id);
        self.hits = self.hits.saturating_add(1);
        Some(id)
    }

    fn alloc(&mut self) -> Result<u32, &'static str> {
        for (i, r) in self.refs.iter_mut().enumerate() {
            if *r == 0 {
                *r = 1;
                return u32::try_from(i).map_err(|_| "kv page id");
            }
        }
        if self.refs.len() >= self.cap {
            let victim = self
                .by_hash
                .iter()
                .find(|(_, id)| {
                    let i = usize::try_from(**id).unwrap_or(usize::MAX);
                    self.refs.get(i).copied() == Some(1)
                })
                .map(|(h, id)| (*h, *id));
            if let Some((h, id)) = victim {
                let _removed = self.by_hash.remove(&h);
                self.release(id);
                return self.alloc();
            }
            return Err("kv page cap");
        }
        let id = u32::try_from(self.refs.len()).map_err(|_| "kv page id")?;
        self.refs.push(1);
        let n = self.refs.len().saturating_mul(self.stride());
        self.k.resize(n, 0.0);
        self.v.resize(n, 0.0);
        Ok(id)
    }

    fn copy_block(&mut self, src: u32, dst: u32) -> Result<(), &'static str> {
        let stride = self.stride();
        if stride == 0 {
            return Err("kv page stride");
        }
        let s = usize::try_from(src)
            .ok()
            .and_then(|i| i.checked_mul(stride))
            .ok_or("kv page copy")?;
        let d = usize::try_from(dst)
            .ok()
            .and_then(|i| i.checked_mul(stride))
            .ok_or("kv page copy")?;
        copy_nonoverlapping_range(&mut self.k, s, d, stride)?;
        copy_nonoverlapping_range(&mut self.v, s, d, stride)?;
        Ok(())
    }

    pub(crate) fn kv_mut(&mut self) -> (&mut [f32], &mut [f32]) {
        (&mut self.k, &mut self.v)
    }
}

impl PagedKvPool {
    pub(crate) fn create(
        n_layers: usize,
        n_head_kv: usize,
        hd: usize,
        block_size: usize,
        cap: usize,
    ) -> Result<Self, &'static str> {
        if block_size == 0 || cap == 0 || n_layers == 0 || n_head_kv == 0 || hd == 0 {
            return Err("kv page geom");
        }
        Ok(Self {
            inner: Rc::new(RefCell::new(KvPool {
                k: Vec::new(),
                v: Vec::new(),
                block_size,
                n_layers,
                n_head_kv,
                hd,
                cap,
                refs: Vec::new(),
                by_hash: BTreeMap::new(),
                hits: 0,
            })),
            block_size,
            n_layers,
        })
    }

    /// Tokens per physical block.
    #[must_use]
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Intern-lookup hits across every sequence on this pool.
    #[must_use]
    pub fn hits(&self) -> u64 {
        self.inner.try_borrow().map(|p| p.hits).unwrap_or(0)
    }

    /// Physical blocks with a positive refcount.
    #[must_use]
    pub fn occupied(&self) -> usize {
        self.inner
            .try_borrow()
            .map(|p| p.refs.iter().filter(|r| **r > 0).count())
            .unwrap_or(0)
    }

    /// True when `other` is the same intern arena (shared `Rc`).
    #[must_use]
    pub fn same_as(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.inner, &other.inner)
    }

    fn try_mut(&self) -> Result<RefMut<'_, KvPool>, &'static str> {
        self.inner.try_borrow_mut().map_err(|_| "kv page borrow")
    }
}

impl KvPages {
    /// Pool with `cap` blocks of `block_size` tokens.
    pub(crate) fn new(
        n_layers: usize,
        n_head_kv: usize,
        hd: usize,
        block_size: usize,
        cap: usize,
    ) -> Result<Self, &'static str> {
        Ok(Self::on(PagedKvPool::create(
            n_layers, n_head_kv, hd, block_size, cap,
        )?))
    }

    pub(crate) fn on(pool: PagedKvPool) -> Self {
        Self {
            pool,
            table: Vec::new(),
        }
    }

    pub(crate) fn pool(&self) -> &PagedKvPool {
        &self.pool
    }

    pub(crate) fn block_size(&self) -> usize {
        self.pool.block_size()
    }

    pub(crate) fn n_layers(&self) -> usize {
        self.pool.n_layers
    }

    pub(crate) fn table_ids(&self) -> &[u32] {
        &self.table
    }

    pub(crate) fn hits(&self) -> u64 {
        self.pool.hits()
    }

    pub(crate) fn occupied(&self) -> usize {
        self.pool.occupied()
    }

    pub(crate) fn try_pool_mut(&self) -> Result<RefMut<'_, KvPool>, &'static str> {
        self.pool.try_mut()
    }

    pub(crate) fn rewind_tokens(&mut self, n: usize) {
        let keep = n.div_ceil(self.pool.block_size);
        let Ok(mut pool) = self.pool.try_mut() else {
            self.table.truncate(keep);
            return;
        };
        while self.table.len() > keep {
            if let Some(id) = self.table.pop() {
                pool.release(id);
            }
        }
    }

    /// Attach interned full prefix blocks starting at `n_past`.
    ///
    /// The table must already cover `n_past` (a block multiple). After a
    /// zero-LCP rewind the table is empty and this walks interned hashes of
    /// `tokens`. After a partial LCP it only extends beyond the kept prefix.
    /// Returns how many tokens are covered (a multiple of `block_size` when
    /// intern hits, otherwise `n_past`).
    pub(crate) fn bind_full_prefix(&mut self, tokens: &[u32], n_past: usize) -> usize {
        let Ok(mut pool) = self.pool.try_mut() else {
            return n_past;
        };
        let bs = pool.block_size;
        if !n_past.is_multiple_of(bs) {
            return n_past;
        }
        if self.table.len() != n_past / bs {
            return n_past;
        }
        let mut pos = n_past;
        loop {
            let end = pos.saturating_add(bs);
            if end > tokens.len() {
                break;
            }
            let Some(chunk) = tokens.get(..end) else {
                break;
            };
            let h = prefix_hash(chunk);
            match pool.lookup(h) {
                Some(id) => {
                    self.table.push(id);
                    pos = end;
                }
                None => break,
            }
        }
        pos
    }

    pub(crate) fn intern_full(&mut self, ids: &[u32]) {
        let Ok(mut pool) = self.pool.try_mut() else {
            return;
        };
        let bs = pool.block_size;
        let n_full = ids.len() / bs;
        for i in 0..n_full {
            let end = i.saturating_add(1).saturating_mul(bs);
            let Some(chunk) = ids.get(..end) else {
                break;
            };
            let Some(&id) = self.table.get(i) else {
                break;
            };
            pool.intern(prefix_hash(chunk), id);
        }
    }

    pub(crate) fn ensure_write(&mut self, pos: usize) -> Result<(), &'static str> {
        let mut pool = self.pool.try_mut()?;
        let bs = pool.block_size;
        let bi = pos / bs;
        if self.table.len() == bi {
            let id = pool.alloc()?;
            self.table.push(id);
            return Ok(());
        }
        if self.table.len() != bi.saturating_add(1) {
            return Err("kv page table");
        }
        let Some(&id) = self.table.get(bi) else {
            return Err("kv page table");
        };
        let i = usize::try_from(id).unwrap_or(usize::MAX);
        let refs = pool.refs.get(i).copied().unwrap_or(0);
        if refs <= 1 {
            return Ok(());
        }
        let fresh = pool.alloc()?;
        pool.copy_block(id, fresh)?;
        pool.release(id);
        if let Some(slot) = self.table.get_mut(bi) {
            *slot = fresh;
        }
        Ok(())
    }
}

impl Drop for KvPages {
    fn drop(&mut self) {
        self.rewind_tokens(0);
    }
}

impl<'a> KvGeom<'a> {
    /// Dense head-major layout: `((layer * n_head_kv + head) * max_seq + t) * hd`.
    #[must_use]
    pub(crate) fn dense(n_head_kv: usize, hd: usize, max_seq: usize) -> Self {
        Self {
            n_head_kv,
            hd,
            n_layers: 0,
            time_stride: max_seq,
            table: None,
        }
    }

    /// Byte offset of one head vector at logical time `t`.
    pub(crate) fn offset(&self, layer: usize, head: usize, t: usize) -> Option<usize> {
        match self.table {
            None => layer
                .checked_mul(self.n_head_kv)
                .and_then(|v| v.checked_add(head))
                .and_then(|v| v.checked_mul(self.time_stride))
                .and_then(|v| v.checked_add(t))
                .and_then(|v| v.checked_mul(self.hd)),
            Some(table) => {
                let bs = self.time_stride;
                if bs == 0 {
                    return None;
                }
                let bi = t / bs;
                let t_in = t % bs;
                let b = table
                    .get(bi)
                    .copied()
                    .and_then(|id| usize::try_from(id).ok())?;
                b.checked_mul(self.n_layers)
                    .and_then(|v| v.checked_add(layer))
                    .and_then(|v| v.checked_mul(self.n_head_kv))
                    .and_then(|v| v.checked_add(head))
                    .and_then(|v| v.checked_mul(bs))
                    .and_then(|v| v.checked_add(t_in))
                    .and_then(|v| v.checked_mul(self.hd))
            }
        }
    }
}

fn copy_nonoverlapping_range(
    buf: &mut [f32],
    src: usize,
    dst: usize,
    n: usize,
) -> Result<(), &'static str> {
    if src == dst {
        return Ok(());
    }
    let src_end = src.checked_add(n).ok_or("kv page copy")?;
    let dst_end = dst.checked_add(n).ok_or("kv page copy")?;
    if src_end > buf.len() || dst_end > buf.len() {
        return Err("kv page copy");
    }
    if src < dst {
        let (left, right) = buf.split_at_mut(dst);
        let s = left.get(src..src_end).ok_or("kv page copy")?;
        let d = right.get_mut(..n).ok_or("kv page copy")?;
        d.copy_from_slice(s);
    } else {
        let (left, right) = buf.split_at_mut(src);
        let d = left.get_mut(dst..dst_end).ok_or("kv page copy")?;
        let s = right.get(..n).ok_or("kv page copy")?;
        d.copy_from_slice(s);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::KvPages;

    #[test]
    fn interned_blocks_bind_after_rewind_and_cow_on_shared_write() {
        let mut p = KvPages::new(1, 1, 1, 2, 8).expect("geom");
        for pos in 0..4 {
            p.ensure_write(pos).expect("alloc");
        }
        p.intern_full(&[1, 2, 3, 4]);
        p.rewind_tokens(0);
        assert_eq!(p.table_ids().len(), 0);
        assert_eq!(p.bind_full_prefix(&[1, 2, 3, 4], 0), 4);
        assert!(p.hits() > 0, "lookup must count intern hits");
        let old = *p.table_ids().get(1).expect("block");
        p.rewind_tokens(3);
        p.ensure_write(3).expect("cow");
        let fresh = *p.table_ids().get(1).expect("cow block");
        assert_ne!(fresh, old, "write into a shared interned block must COW");
    }

    #[test]
    fn dense_offset_is_head_major() {
        let g = super::KvGeom::dense(2, 4, 8);
        assert_eq!(g.offset(1, 1, 3), Some(108));
        assert_eq!(g.offset(0, 0, 0), Some(0));
    }

    #[test]
    fn two_tables_on_one_pool_intern_hit() {
        let pool = super::PagedKvPool::create(1, 1, 1, 2, 8).expect("pool");
        let mut a = super::KvPages::on(pool.clone());
        let mut b = super::KvPages::on(pool.clone());
        for pos in 0..4 {
            a.ensure_write(pos).expect("a");
        }
        a.intern_full(&[1, 2, 3, 4]);
        assert_eq!(b.bind_full_prefix(&[1, 2, 3, 4], 0), 4);
        assert!(pool.hits() > 0);
        assert_eq!(b.table_ids().len(), 2);
    }

    #[test]
    fn drop_releases_unique_blocks_so_the_next_table_can_alloc() {
        let pool = super::PagedKvPool::create(1, 1, 1, 2, 2).expect("pool");
        {
            let mut a = super::KvPages::on(pool.clone());
            for pos in 0..4 {
                a.ensure_write(pos).expect("a");
            }
            assert_eq!(pool.occupied(), 2);
        }
        let mut b = super::KvPages::on(pool.clone());
        for pos in 0..4 {
            b.ensure_write(pos).expect("reuse after drop");
        }
        assert_eq!(b.table_ids().len(), 2);
    }
}
