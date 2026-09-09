//! Paged KV cache (VRAM) + block tables — `core-01`.
//!
//! The [`KvPool`] auto-sizes from a byte budget (production: free VRAM after
//! weights + graphs; tests pass a small number) and doles out fixed-size
//! physical blocks. A [`BlockTable`] maps a request's logical block index to
//! a physical [`BlockId`]. Blocks are reference-counted so sibling requests
//! can share a read-only prefix (core-07) and the KV-RAM host tier can pin a
//! snapshot without re-prefilling (core-06).

use std::collections::VecDeque;

/// A physical block id inside the KV pool.
pub type BlockId = u32;

/// The paged KV pool: a fixed number of physical blocks derived from a byte
/// budget. Allocation fails cleanly (`None`) when the pool is exhausted —
/// the pool never over-allocates (core-01: no OOM under an N=8 load).
#[derive(Debug)]
pub struct KvPool {
    budget: usize,
    bytes_per_block: usize,
    block_count: usize,
    refcounts: Vec<u32>,
    free: VecDeque<BlockId>,
    used: usize,
}

impl KvPool {
    /// Create a pool sized from the runtime's reported page geometry
    /// (GitHub #55, P1-19) instead of a constant: `page_count` physical
    /// pages of `page_bytes` each — the leaf's `ignis_paged_kv_page_budget`
    /// query (`kernel/include/ignis_paged_kv_budget.h`) already picked
    /// `page_count` to fit the VRAM left after weights, so this pool's
    /// block count exactly matches the device-resident pool's page count.
    ///
    /// Scope note: this sizes `ignis-core`'s own [`KvPool`] from real
    /// numbers, as P1-19 requires. Wiring a real budget query result into
    /// [`crate::concrete::SchedulerConfig::kv_capacity_pages`] (a
    /// differently-grained admission-accounting page, `kv_page_tokens`
    /// wide, not the device's fixed 64-token physical page) needs the real
    /// `Compute` adapter and is P1-24's job (GitHub #60), not this one's.
    pub fn from_page_geometry(page_count: u32, page_bytes: u64) -> Self {
        Self::new(page_count as usize * page_bytes as usize, page_bytes as usize)
    }

    /// Create a pool holding `budget` bytes of KV, in blocks of
    /// `bytes_per_block` bytes each. The block count auto-sizes from the
    /// budget: `budget / bytes_per_block`.
    pub fn new(budget: usize, bytes_per_block: usize) -> Self {
        assert!(bytes_per_block > 0, "kv blocks must be non-empty");
        let block_count = budget / bytes_per_block;
        let mut free = VecDeque::with_capacity(block_count);
        for b in 0..block_count as BlockId {
            free.push_back(b);
        }
        Self {
            budget,
            bytes_per_block,
            block_count,
            refcounts: vec![0; block_count],
            free,
            used: 0,
        }
    }

    /// The byte budget this pool was sized from.
    pub fn budget(&self) -> usize {
        self.budget
    }

    /// The size of one physical block in bytes.
    pub fn bytes_per_block(&self) -> usize {
        self.bytes_per_block
    }

    /// The number of physical blocks (auto-sized from the budget).
    pub fn block_count(&self) -> usize {
        self.block_count
    }

    /// Blocks holding at least one live reference.
    pub fn used_blocks(&self) -> usize {
        self.used
    }

    /// Blocks currently unowned (on the free list).
    pub fn free_blocks(&self) -> usize {
        self.free.len()
    }

    /// Fraction of the pool in use (0.0..=1.0).
    pub fn usage_ratio(&self) -> f64 {
        if self.block_count == 0 {
            0.0
        } else {
            self.used as f64 / self.block_count as f64
        }
    }

    /// The current reference count of a block (0 = free).
    pub fn refcount(&self, block: BlockId) -> u32 {
        self.refcounts
            .get(block as usize)
            .copied()
            .unwrap_or(0)
    }

    /// Allocate a fresh block with a single reference (the holder's).
    /// Returns `None` when the pool is exhausted — a clean failure, never an
    /// over-allocation.
    pub fn alloc(&mut self) -> Option<BlockId> {
        let block = self.free.pop_front()?;
        self.refcounts[block as usize] = 1;
        self.used += 1;
        Some(block)
    }

    /// Add a reference to an existing block — used when a block is *shared*:
    /// a sibling prefix (core-07) or a host-tier snapshot (core-06) pins it
    /// alongside the owner. Fails (returns `false`) when the block is free,
    /// so a dropped snapshot can never be silently re-claimed.
    pub fn claim(&mut self, block: BlockId) -> bool {
        let rc = match self.refcounts.get_mut(block as usize) {
            Some(rc) => rc,
            None => return false,
        };
        if *rc == 0 {
            return false;
        }
        *rc += 1;
        true
    }

    /// Release one reference. When the last reference drops, the block
    /// returns to the free list. Returns `false` when the block is free or
    /// unknown (refcount 0 or out of range) — a caller dropping a block it
    /// never held is a bug, surfaced rather than silently swallowed.
    pub fn free(&mut self, block: BlockId) -> bool {
        let rc = match self.refcounts.get_mut(block as usize) {
            Some(rc) => rc,
            None => return false,
        };
        if *rc == 0 {
            return false;
        }
        *rc -= 1;
        if *rc == 0 {
            self.free.push_back(block);
            self.used -= 1;
        }
        true
    }
}

/// The scheduler's expected page-count capacity disagreed with the pool the
/// leaf actually built (GitHub #98, P3-02) — a stale formula or a config
/// that drifted from the leaf, surfaced at load rather than trusted, so it
/// fails loudly instead of showing up later as an opaque `ignis_seq_alloc`
/// exhaustion the admission machine never saw coming.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CapacityMismatch {
    /// The page count the scheduler's own configuration expects.
    pub expected_pages: u32,
    /// The page count the leaf's pool actually reports.
    pub leaf_pages: u32,
}

impl std::fmt::Display for CapacityMismatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "KV capacity mismatch: the scheduler expects {} pages, the leaf's paged KV pool actually holds {}",
            self.expected_pages, self.leaf_pages
        )
    }
}

impl std::error::Error for CapacityMismatch {}

/// The leaf's reported paged-KV pool geometry (`ignis_seq_pool_stats`,
/// GitHub #55): the page count and per-page bytes it actually built, always
/// read back together — bundled here so the two can't be transposed at a
/// call site the way two bare same-typed integers could be.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LeafPoolGeometry {
    /// Physical KV pages the pool holds (`kv_page_group_count`).
    pub page_count: u32,
    /// Bytes of one physical KV page across every plane (`kv_page_bytes`).
    pub page_bytes: u64,
}

/// The [`KvPool`] built from the leaf's reported page geometry
/// (`ignis_seq_pool_stats`, GitHub #55), once verified to match the
/// scheduler's independently-computed page-count capacity. The two sides
/// derive their page count from the same configured engine shape by
/// construction, but only asking the leaf what it actually built (rather
/// than trusting the formula that requested it) turns a possible future
/// drift into a load-time failure instead of a mid-serving surprise —
/// `Err` on a disagreement, never a silently trusted guess.
pub fn verified_kv_pool(
    expected_pages: u32,
    leaf: LeafPoolGeometry,
) -> Result<KvPool, CapacityMismatch> {
    if expected_pages != leaf.page_count {
        return Err(CapacityMismatch {
            expected_pages,
            leaf_pages: leaf.page_count,
        });
    }
    Ok(KvPool::from_page_geometry(leaf.page_count, leaf.page_bytes))
}

/// A per-request block table: logical block index → physical block.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BlockTable {
    blocks: Vec<BlockId>,
}

impl BlockTable {
    pub fn new() -> Self {
        Self {
            blocks: Vec::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.blocks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }

    /// The physical block backing logical block `i`.
    pub fn get(&self, i: usize) -> Option<BlockId> {
        self.blocks.get(i).copied()
    }

    pub fn as_slice(&self) -> &[BlockId] {
        &self.blocks
    }

    /// Append a physical block; its logical index is the current length.
    pub fn push(&mut self, block: BlockId) {
        self.blocks.push(block);
    }

    /// Share read-only blocks from another table (a cached sibling prefix,
    /// core-07). The caller is responsible for claiming the blocks in the
    /// pool (one reference each).
    pub fn extend_shared(&mut self, other: &[BlockId]) {
        self.blocks.extend_from_slice(other);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pool_sizes_from_runtime_reported_page_geometry() {
        // A stub of what `ignis_paged_kv_page_budget` reports (no GPU
        // needed here — this is the sizing math, not the query itself):
        // 12 physical pages of 2048 bytes each fit the budget.
        let pool = KvPool::from_page_geometry(12, 2048);
        assert_eq!(pool.block_count(), 12, "block count matches the reported page count");
        assert_eq!(pool.bytes_per_block(), 2048, "block size matches the reported page bytes");
        assert_eq!(pool.budget(), 12 * 2048);
        assert_eq!(pool.free_blocks(), 12);
    }

    #[test]
    fn pool_sizes_from_budget() {
        let pool = KvPool::new(8 * 512, 512);
        assert_eq!(pool.block_count(), 8);
        assert_eq!(pool.used_blocks(), 0);
        assert_eq!(pool.free_blocks(), 8);
        assert_eq!(pool.usage_ratio(), 0.0);
        assert_eq!(pool.budget(), 8 * 512);
    }

    #[test]
    fn block_table_maps_logical_to_physical() {
        let mut pool = KvPool::new(4 * 512, 512);
        let mut table = BlockTable::new();
        for _ in 0..3 {
            table.push(pool.alloc().unwrap());
        }
        let ids = table.as_slice().to_vec();
        assert_eq!(table.len(), 3);
        assert_eq!(table.get(0), Some(ids[0]));
        assert_eq!(table.get(2), Some(ids[2]));
        assert_eq!(table.get(3), None);
        assert_eq!(table.get(0).unwrap(), table.as_slice()[0]);
    }

    #[test]
    fn no_oom_under_n8_load() {
        // 8 requests × 8 blocks against an 8-block pool: the pool never
        // over-allocates; allocations fail cleanly once it is full.
        let mut pool = KvPool::new(8 * 256, 256);
        let mut granted = Vec::new();
        for _req in 0..8 {
            for _ in 0..8 {
                if let Some(b) = pool.alloc() {
                    granted.push(b);
                }
            }
        }
        assert_eq!(granted.len(), 8);
        assert_eq!(pool.used_blocks(), 8);
        assert_eq!(pool.usage_ratio(), 1.0);
        assert!(pool.alloc().is_none());
        // everything can be released again
        for b in &granted {
            pool.free(*b);
        }
        assert_eq!(pool.used_blocks(), 0);
        assert_eq!(pool.free_blocks(), 8);
    }

    #[test]
    fn sharing_refcounts() {
        let mut pool = KvPool::new(4 * 512, 512);
        let b = pool.alloc().unwrap();
        assert!(pool.claim(b)); // a sibling pins the block
        pool.free(b); // owner drops its reference; the sibling keeps it
        assert_eq!(pool.refcount(b), 1);
        assert_eq!(pool.used_blocks(), 1);
        // claiming a *free* block fails (no silent re-claim), and freeing a
        // free block fails the same way (no double-free):
        let free_block = pool.alloc().unwrap();
        pool.free(free_block);
        assert_eq!(pool.refcount(free_block), 0);
        assert!(!pool.claim(free_block));
        assert!(!pool.free(free_block));
        pool.free(b); // last reference drops → back to the pool
        assert_eq!(pool.refcount(b), 0);
        // Every block is now free (the two allocated + the two never used).
        assert_eq!(pool.used_blocks(), 0);
        assert_eq!(pool.free_blocks(), 4);
    }

    #[test]
    fn verified_kv_pool_builds_the_real_pool_when_the_two_sides_agree() {
        // The scheduler's own formula and the leaf's `ignis_seq_pool_stats`
        // report the same page count: the real `KvPool` is handed back. Its
        // bytes-per-page geometry is already pinned by
        // `pool_sizes_from_runtime_reported_page_geometry` above — this test
        // only needs to show the *verified* pool is the one built from the
        // leaf's numbers (GitHub #98).
        let leaf = LeafPoolGeometry {
            page_count: 1024,
            page_bytes: 65536,
        };
        let pool = verified_kv_pool(1024, leaf).expect("the two sides agree");
        assert_eq!(pool.block_count(), 1024);
        assert_eq!(pool.budget(), 1024 * 65536);
    }

    #[test]
    fn verified_kv_pool_rejects_a_disagreement_instead_of_trusting_the_formula() {
        // The scheduler expects 1024 pages (its own formula); the leaf's
        // pool actually holds 900 — a drift that must fail loudly at load,
        // not surface later as a mid-serving `ignis_seq_alloc` exhaustion.
        let leaf = LeafPoolGeometry {
            page_count: 900,
            page_bytes: 65536,
        };
        let err = verified_kv_pool(1024, leaf).expect_err("a disagreement is an error");
        assert_eq!(
            err,
            CapacityMismatch {
                expected_pages: 1024,
                leaf_pages: 900,
            }
        );
        assert!(err.to_string().contains("1024"));
        assert!(err.to_string().contains("900"));
    }
}