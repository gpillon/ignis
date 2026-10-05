//! Flash-Next's n-gram rows on the host (spec flash-next/04, stories 22-23;
//! slice S4 of GitHub #302): the RAM hot-row cache loaded from the artifact's
//! hot list, the NVMe reader, and the gathers the scheduler calls.
//!
//! [`NgramTable`] owns everything a load needs to turn tokens into the bytes
//! the device's `fn_ngram_add` reads (`[tokens][heads][row_bytes]`, layout.md
//! 7.1): the checkpoint's hasher ([`crate::ngram::NgramHasher`]), the hot rows
//! in RAM ([`crate::ngram::HotRows`], a byte budget, 1 GiB by default), and a
//! pool of worker threads, each with its own unbuffered handle to the
//! artifact file, executing [`crate::ngram::plan_gather`]'s aligned reads.
//!
//! A gather is two calls so the NVMe latency overlaps other work:
//! [`NgramTable::begin`] hashes the tokens, plans and submits the reads and
//! returns at once; [`PendingRows::finish`] waits for them and copies every
//! row into the caller's staging buffer — the model instance's pinned buffer,
//! which the step uploads with its inputs. A prompt's rows begin at
//! admission, a decode step's right after the previous token is sampled;
//! both finish before the graph launch. [`NgramTable::stage`] is the two
//! chained.

use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use ignis_artifact::flash_next::FlashNextPlan;
use ignis_artifact::{AlignedBuffer, DirectReader, Object, Reader, DIRECT_IO_ALIGNMENT};

use crate::compute::NgramGeometry;
use crate::ngram::{
    plan_gather, AlignedRead, GatherPlan, HotRows, NgramContext, NgramHashBuffers, NgramHasher, ReadPolicy,
    TableLayout,
};

/// The RAM the hot-row cache may take, its index included (spec 04's
/// default).
pub const DEFAULT_HOT_BYTES: u64 = 1 << 30;
/// Worker threads issuing the table's reads.
pub const DEFAULT_READ_THREADS: usize = 4;
/// The longest read a step issues: rows whose sectors touch share one up to
/// this.
pub const DEFAULT_MAX_READ_BYTES: u64 = 64 << 10;
/// The longest read the hot-row load issues: the hot rows are dense enough in
/// the table that the load is close to a scan.
const HOT_LOAD_MAX_READ_BYTES: u64 = 4 << 20;
/// Hot rows loaded per plan (bounds the load's buffers).
const HOT_LOAD_BATCH_ROWS: usize = 1 << 20;

/// What an [`NgramTable`] is opened with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NgramTableOptions {
    /// The hot-row cache's budget in bytes (rows and index).
    pub hot_bytes: u64,
    pub read_threads: usize,
    pub max_read_bytes: u64,
}

impl Default for NgramTableOptions {
    fn default() -> Self {
        Self {
            hot_bytes: DEFAULT_HOT_BYTES,
            read_threads: DEFAULT_READ_THREADS,
            max_read_bytes: DEFAULT_MAX_READ_BYTES,
        }
    }
}

/// Rows gathered so far, by source (the table's metrics).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct NgramCounters {
    /// Rows staged.
    pub rows: u64,
    /// Of them, rows the hot-row cache held.
    pub hot_rows: u64,
    /// Reads issued to the file, and their bytes.
    pub reads: u64,
    pub read_bytes: u64,
}

/// One read's result: its index in the plan, its buffer and the bytes that
/// arrived (fewer than asked only at the end of the file).
type ReadResult = (usize, Result<(AlignedBuffer, usize), String>);

struct Job {
    index: usize,
    read: AlignedRead,
    reply: mpsc::Sender<ReadResult>,
}

/// Worker threads, each with its own unbuffered handle, running reads.
struct ReadPool {
    jobs: Option<mpsc::Sender<Job>>,
    workers: Vec<JoinHandle<()>>,
    unbuffered: bool,
}

impl ReadPool {
    fn new(path: &Path, threads: usize) -> Result<Self, String> {
        if threads == 0 {
            return Err("an n-gram reader needs at least one thread".to_string());
        }
        let (jobs, queue) = mpsc::channel::<Job>();
        let queue = Arc::new(Mutex::new(queue));
        let mut workers = Vec::with_capacity(threads);
        let mut unbuffered = true;
        for worker in 0..threads {
            let reader = DirectReader::open(path).map_err(|e| e.to_string())?;
            unbuffered &= reader.is_unbuffered();
            let queue = Arc::clone(&queue);
            let handle = std::thread::Builder::new()
                .name(format!("ngram-read-{worker}"))
                .spawn(move || loop {
                    let job = match queue.lock() {
                        Ok(queue) => queue.recv(),
                        Err(_) => return,
                    };
                    let Ok(job) = job else { return };
                    let result = AlignedBuffer::new(job.read.len as usize)
                        .and_then(|mut buffer| {
                            let got = reader.read_at(job.read.offset, buffer.as_mut_slice())?;
                            Ok((buffer, got))
                        })
                        .map_err(|e| e.to_string());
                    // The receiver may have given up (an error elsewhere in
                    // its gather): nothing to deliver to.
                    let _ = job.reply.send((job.index, result));
                })
                .map_err(|e| format!("spawn an n-gram reader thread: {e}"))?;
            workers.push(handle);
        }
        Ok(Self { jobs: Some(jobs), workers, unbuffered })
    }

    /// Queue `reads`; their results arrive on the returned receiver, in any
    /// order, tagged with their index.
    fn submit(&self, reads: &[AlignedRead]) -> Result<mpsc::Receiver<ReadResult>, String> {
        let (reply, results) = mpsc::channel();
        let jobs = self.jobs.as_ref().expect("the pool's queue lives as long as the pool");
        for (index, &read) in reads.iter().enumerate() {
            jobs.send(Job { index, read, reply: reply.clone() })
                .map_err(|_| "the n-gram reader threads have stopped".to_string())?;
        }
        Ok(results)
    }
}

impl Drop for ReadPool {
    fn drop(&mut self) {
        // Closing the queue ends every worker's loop.
        self.jobs.take();
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

/// The n-gram table of a load (see the module doc). Shared by reference
/// across threads: every method takes `&self`.
pub struct NgramTable {
    hasher: NgramHasher,
    hot: HotRows,
    hot_bytes: Vec<u8>,
    layout: TableLayout,
    policy: ReadPolicy,
    pool: ReadPool,
    rows: AtomicU64,
    hot_hits: AtomicU64,
    reads: AtomicU64,
    read_bytes: AtomicU64,
}

impl NgramTable {
    /// Open the table at `layout` in the file at `path`, hashing with
    /// `hasher`, and load into RAM the head of `ranked_hot` (row ids, most
    /// frequent first) that fits `options.hot_bytes`.
    pub fn open(
        path: &Path,
        layout: TableLayout,
        hasher: NgramHasher,
        ranked_hot: &[u64],
        options: NgramTableOptions,
    ) -> Result<Self, String> {
        if layout.rows < hasher.padded_table_rows() {
            return Err(format!(
                "the table holds {} rows, the hash ranges reach {}",
                layout.rows,
                hasher.padded_table_rows()
            ));
        }
        let policy = ReadPolicy { alignment: DIRECT_IO_ALIGNMENT, max_read_bytes: options.max_read_bytes };
        let hot = HotRows::from_ranked(ranked_hot, options.hot_bytes, layout.row_bytes)?;
        let mut table = Self {
            hasher,
            hot,
            hot_bytes: Vec::new(),
            layout,
            policy,
            pool: ReadPool::new(path, options.read_threads)?,
            rows: AtomicU64::new(0),
            hot_hits: AtomicU64::new(0),
            reads: AtomicU64::new(0),
            read_bytes: AtomicU64::new(0),
        };
        table.hot_bytes = table.load_hot_rows()?;
        Ok(table)
    }

    /// Open the table of a bound Flash-Next artifact (spec flash-next/01):
    /// its host-streamed range, its hash buffers and its hot list.
    pub fn from_artifact(
        path: &Path,
        reader: &Reader,
        bound: &FlashNextPlan,
        geometry: NgramGeometry,
        options: NgramTableOptions,
    ) -> Result<Self, String> {
        let prefix = format!("layers.{}.ple.ple_embedding", geometry.layer);
        let table_name = format!("{prefix}.ngram_embedding.weight");
        let handle = *bound
            .handles
            .get(&table_name)
            .ok_or_else(|| format!("the artifact has no {table_name}"))?;
        let placement = bound
            .plan
            .streamed_objects
            .iter()
            .find(|p| p.handle == handle)
            .ok_or_else(|| format!("{table_name} is not host-streamed in the plan"))?;
        let Some(Object::Tensor(descriptor)) = reader.find(&table_name) else {
            return Err(format!("{table_name} is not a tensor"));
        };
        let (rows, columns) = (descriptor.shape[0], descriptor.shape[1]);
        let row_bytes = placement.bytes / rows;
        if row_bytes * rows != placement.bytes || columns == 0 {
            return Err(format!("{table_name}: {} bytes for {rows} rows", placement.bytes));
        }
        let layout = TableLayout { base_offset: placement.file_offset, row_stride: row_bytes, row_bytes, rows };

        let words = |name: &str, width: usize| -> Result<Vec<i64>, String> {
            let data = reader.payload(name).map_err(|e| e.to_string())?.data;
            Ok(data
                .chunks_exact(width)
                .map(|w| if width == 8 { i64::from_le_bytes(w.try_into().unwrap()) } else { i64::from(i32::from_le_bytes(w.try_into().unwrap())) })
                .collect())
        };
        let buffers = NgramHashBuffers {
            layer_multipliers: words(&format!("{prefix}.layer_multipliers"), 8)?,
            head_vocab_sizes: words(&format!("{prefix}.ngram_heads_vocab_sizes"), 8)?,
            head_offsets: words(&format!("{prefix}.ngram_heads_offsets"), 8)?,
        };
        let ranked: Vec<u64> = words(&format!("{prefix}.ngram_embedding.hot_rows"), 4)?
            .into_iter()
            .map(|row| row as u64)
            .collect();
        let hasher = NgramHasher::new(geometry, buffers)?;
        Self::open(path, layout, hasher, &ranked, options)
    }

    pub fn hasher(&self) -> &NgramHasher {
        &self.hasher
    }

    /// A sequence that has hashed no token yet.
    pub fn new_context(&self) -> NgramContext {
        NgramContext::new(&self.hasher)
    }

    /// Rows the hot-row cache holds, and the RAM it takes.
    pub fn hot_rows(&self) -> usize {
        self.hot.len()
    }

    pub fn hot_bytes(&self) -> u64 {
        self.hot.bytes()
    }

    /// Whether the reads bypass the page cache (false only where the file
    /// system refuses unbuffered reads).
    pub fn is_unbuffered(&self) -> bool {
        self.pool.unbuffered
    }

    pub fn row_bytes(&self) -> usize {
        self.layout.row_bytes as usize
    }

    /// The staging bytes one token's rows take: `heads * row_bytes`.
    pub fn token_bytes(&self) -> usize {
        self.hasher.heads() * self.row_bytes()
    }

    /// Hash `tokens` (moving `context` past them) and start fetching their
    /// rows.
    pub fn begin(&self, context: &mut NgramContext, tokens: &[u32]) -> Result<PendingRows<'_>, String> {
        let mut rows = Vec::with_capacity(tokens.len() * self.hasher.heads());
        self.hasher.hash(context, tokens, &mut rows);
        self.begin_rows(&rows)
    }

    /// Start fetching table rows `rows` (any order, repeats allowed).
    pub fn begin_rows(&self, rows: &[u64]) -> Result<PendingRows<'_>, String> {
        let plan = plan_gather(rows, &self.hot, &self.layout, self.policy)?;
        let results = self.pool.submit(&plan.reads)?;
        Ok(PendingRows { table: self, plan, results })
    }

    /// [`NgramTable::begin`] then [`PendingRows::finish`].
    pub fn stage(&self, context: &mut NgramContext, tokens: &[u32], out: &mut [u8]) -> Result<(), String> {
        self.begin(context, tokens)?.finish(out)
    }

    /// The rows gathered so far, by source.
    pub fn counters(&self) -> NgramCounters {
        NgramCounters {
            rows: self.rows.load(Ordering::Relaxed),
            hot_rows: self.hot_hits.load(Ordering::Relaxed),
            reads: self.reads.load(Ordering::Relaxed),
            read_bytes: self.read_bytes.load(Ordering::Relaxed),
        }
    }

    /// Read the hot rows from the file into their slots, a batch of rows at
    /// a time, with the coalescing reads the step path uses but longer.
    fn load_hot_rows(&self) -> Result<Vec<u8>, String> {
        let row_bytes = self.row_bytes();
        let mut bytes = vec![0u8; self.hot.len() * row_bytes];
        let empty = HotRows::from_ranked(&[], 0, self.layout.row_bytes)?;
        let policy = ReadPolicy { alignment: DIRECT_IO_ALIGNMENT, max_read_bytes: HOT_LOAD_MAX_READ_BYTES.max(self.policy.max_read_bytes) };
        let rows: Vec<u64> = self.hot.rows().iter().map(|&row| u64::from(row)).collect();
        for (batch, chunk) in rows.chunks(HOT_LOAD_BATCH_ROWS).enumerate() {
            let plan = plan_gather(chunk, &empty, &self.layout, policy)?;
            let results = self.pool.submit(&plan.reads)?;
            let start = batch * HOT_LOAD_BATCH_ROWS * row_bytes;
            collect_and_gather(&plan, results, &[], &mut bytes[start..start + chunk.len() * row_bytes])?;
        }
        Ok(bytes)
    }
}

/// A gather in flight: its reads are queued or running on the table's
/// threads.
pub struct PendingRows<'a> {
    table: &'a NgramTable,
    plan: GatherPlan,
    results: mpsc::Receiver<ReadResult>,
}

impl PendingRows<'_> {
    /// Rows the gather fetches.
    pub fn rows(&self) -> usize {
        self.plan.sources.len()
    }

    /// Wait for the reads and copy every row, in request order, into `out`
    /// (`rows() * row_bytes` bytes: the model's pinned staging).
    pub fn finish(self, out: &mut [u8]) -> Result<(), String> {
        let table = self.table;
        collect_and_gather(&self.plan, self.results, &table.hot_bytes, out)?;
        let hot = self.plan.sources.iter().filter(|s| matches!(s, crate::ngram::RowSource::Hot { .. })).count();
        table.rows.fetch_add(self.plan.sources.len() as u64, Ordering::Relaxed);
        table.hot_hits.fetch_add(hot as u64, Ordering::Relaxed);
        table.reads.fetch_add(self.plan.reads.len() as u64, Ordering::Relaxed);
        table.read_bytes.fetch_add(self.plan.read_bytes(), Ordering::Relaxed);
        Ok(())
    }
}

/// Wait for every read of `plan` and gather its rows into `out`.
fn collect_and_gather(
    plan: &GatherPlan,
    results: mpsc::Receiver<ReadResult>,
    hot: &[u8],
    out: &mut [u8],
) -> Result<(), String> {
    let mut buffers: Vec<Option<(AlignedBuffer, usize)>> = (0..plan.reads.len()).map(|_| None).collect();
    for _ in 0..plan.reads.len() {
        let (index, result) = results
            .recv()
            .map_err(|_| "an n-gram reader thread stopped mid-gather".to_string())?;
        buffers[index] = Some(result.map_err(|e| format!("n-gram read {index}: {e}"))?);
    }
    let reads: Vec<&[u8]> = buffers
        .iter()
        .map(|b| {
            let (buffer, got) = b.as_ref().expect("every read reported");
            &buffer.as_slice()[..*got]
        })
        .collect();
    plan.gather(hot, &reads, out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    const ROW_BYTES: u64 = 90;

    fn row_content(row: u64) -> Vec<u8> {
        (0..ROW_BYTES).map(|i| ((row * 31 + i * 7) % 251) as u8).collect()
    }

    /// A table file of `rows` rows after a `base`-byte header (the artifact's
    /// table starts 4096-aligned; an odd base exercises the planner too).
    struct TableFile {
        path: std::path::PathBuf,
        layout: TableLayout,
    }

    impl TableFile {
        fn write(tag: &str, rows: u64, base: u64) -> Self {
            let path = std::env::temp_dir().join(format!("ignis-ngram-table-{tag}-{}.bin", std::process::id()));
            let mut file = std::fs::File::create(&path).unwrap();
            file.write_all(&vec![0xEE; base as usize]).unwrap();
            for row in 0..rows {
                file.write_all(&row_content(row)).unwrap();
            }
            let layout = TableLayout { base_offset: base, row_stride: ROW_BYTES, row_bytes: ROW_BYTES, rows };
            Self { path, layout }
        }
    }

    impl Drop for TableFile {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.path);
        }
    }

    /// Two heads (n-grams of 2 and 3 tokens, one head each) whose ranges
    /// fill `rows` rows.
    fn hasher(rows: u64) -> NgramHasher {
        let geometry = NgramGeometry {
            ngram_size: 3,
            heads_per_ngram: 1,
            embed_dim: 320,
            conv_kernel: 4,
            layer: 1,
            vocab_size_base: 490,
            vocab_divisor: 1,
            split_parts: 1,
            seed: 0,
            eos_token_id: 7,
        };
        let first = rows / 2;
        NgramHasher::new(
            geometry,
            NgramHashBuffers {
                layer_multipliers: vec![3, 5, 7],
                head_vocab_sizes: vec![first as i64, (rows - first) as i64],
                head_offsets: vec![0, first as i64],
            },
        )
        .unwrap()
    }

    fn expected(rows: &[u64]) -> Vec<u8> {
        rows.iter().flat_map(|&r| row_content(r)).collect()
    }

    fn options(hot_bytes: u64) -> NgramTableOptions {
        NgramTableOptions { hot_bytes, ..NgramTableOptions::default() }
    }

    #[test]
    fn rows_come_back_in_request_order_from_cache_and_file() {
        let table_file = TableFile::write("mixed", 5_000, 4096 + 123);
        // Room for the first 100 ranked rows.
        let ranked: Vec<u64> = (0..400).map(|i| (i * 37) % 5_000).collect();
        let table = NgramTable::open(&table_file.path, table_file.layout, hasher(5_000), &ranked, options(100 * 94)).unwrap();
        assert_eq!(table.hot_rows(), 100);
        let rows = [4_999u64, 0, 37, 37, 2_500, 74, 1, 4_998, 3_700];
        let mut out = vec![0u8; rows.len() * ROW_BYTES as usize];
        table.begin_rows(&rows).unwrap().finish(&mut out).unwrap();
        assert_eq!(out, expected(&rows));
        let counters = table.counters();
        assert_eq!(counters.rows, rows.len() as u64);
        assert_eq!(counters.hot_rows, 4, "rows 0, 37 twice and 74 are hot");
        assert!(counters.reads >= 1 && counters.read_bytes % 4096 == 0);
    }

    #[test]
    fn a_cache_that_holds_every_row_issues_no_read_and_an_empty_one_reads_them_all() {
        let table_file = TableFile::write("budgets", 1_000, 4096);
        let ranked: Vec<u64> = (0..1_000).collect();
        let all = NgramTable::open(&table_file.path, table_file.layout, hasher(1_000), &ranked, options(1 << 20)).unwrap();
        let none = NgramTable::open(&table_file.path, table_file.layout, hasher(1_000), &ranked, options(0)).unwrap();
        assert_eq!((all.hot_rows(), none.hot_rows()), (1_000, 0));
        let rows: Vec<u64> = (0..1_000).rev().step_by(7).collect();
        for table in [&all, &none] {
            let mut out = vec![0u8; rows.len() * ROW_BYTES as usize];
            table.begin_rows(&rows).unwrap().finish(&mut out).unwrap();
            assert_eq!(out, expected(&rows));
        }
        assert_eq!(all.counters().reads, 0);
        assert_eq!(none.counters().hot_rows, 0);
    }

    #[test]
    fn the_last_row_reads_through_the_files_short_end() {
        // 1,000 rows end mid-sector: the last read comes back short.
        let table_file = TableFile::write("tail", 1_000, 4096);
        let table = NgramTable::open(&table_file.path, table_file.layout, hasher(1_000), &[], options(0)).unwrap();
        let rows = [999u64, 998];
        let mut out = vec![0u8; 2 * ROW_BYTES as usize];
        table.begin_rows(&rows).unwrap().finish(&mut out).unwrap();
        assert_eq!(out, expected(&rows));
    }

    #[test]
    fn gathers_run_from_many_threads_at_once() {
        let table_file = TableFile::write("threads", 20_000, 4096);
        let ranked: Vec<u64> = (0..20_000).step_by(3).collect();
        let table = NgramTable::open(&table_file.path, table_file.layout, hasher(20_000), &ranked, options(500 * 94)).unwrap();
        std::thread::scope(|scope| {
            for t in 0..6u64 {
                let table = &table;
                scope.spawn(move || {
                    for round in 0..50u64 {
                        let rows: Vec<u64> = (0..16).map(|i| (t * 7_919 + round * 104_729 + i * 1_237) % 20_000).collect();
                        let mut out = vec![0u8; rows.len() * ROW_BYTES as usize];
                        table.begin_rows(&rows).unwrap().finish(&mut out).unwrap();
                        assert_eq!(out, expected(&rows));
                    }
                });
            }
        });
        assert_eq!(table.counters().rows, 6 * 50 * 16);
    }

    #[test]
    fn staging_tokens_hashes_them_and_continues_the_sequence() {
        let table_file = TableFile::write("tokens", 1_000, 4096);
        let table = NgramTable::open(&table_file.path, table_file.layout, hasher(1_000), &[3, 400, 800], options(1 << 16)).unwrap();
        let tokens = [11u32, 12, 7, 13, 14];
        // One call for the whole prompt ...
        let mut whole = vec![0u8; tokens.len() * table.token_bytes()];
        table.stage(&mut table.new_context(), &tokens, &mut whole).unwrap();
        // ... is the prompt's first three tokens, then two decode steps.
        let mut context = table.new_context();
        let mut pieces = Vec::new();
        for part in [&tokens[..3], &tokens[3..4], &tokens[4..]] {
            let mut out = vec![0u8; part.len() * table.token_bytes()];
            table.stage(&mut context, part, &mut out).unwrap();
            pieces.extend(out);
        }
        assert_eq!(pieces, whole);
        // And the bytes are the hashed rows'.
        let mut ids = Vec::new();
        table.hasher().hash(&mut table.new_context(), &tokens, &mut ids);
        assert_eq!(whole, expected(&ids));
    }

    #[test]
    fn a_flash_next_artifact_opens_with_its_buffers_and_hot_list() {
        use ignis_artifact::flash_next::{self, fixture, FlashNextGeometry};
        let artifact = fixture::build("ngram-table").unwrap();
        let reader = Reader::open(&artifact.path).unwrap();
        let bound = flash_next::bind(&reader, &FlashNextGeometry::fixture()).unwrap();
        // The fixture's n-gram block: 2 heads of 160, ranges 491 + 499 padded
        // to its 1,000 rows.
        let geometry = NgramGeometry {
            ngram_size: 3,
            heads_per_ngram: 1,
            embed_dim: 320,
            conv_kernel: 4,
            layer: 1,
            vocab_size_base: 490,
            vocab_divisor: 1_000,
            split_parts: 2,
            seed: 0,
            eos_token_id: 248_044,
        };
        // Room for three hot rows: the hot list's first three, 17, 3, 999.
        let table =
            NgramTable::from_artifact(&artifact.path, &reader, &bound, geometry, options(3 * 94)).unwrap();
        assert_eq!(table.hot_rows(), 3);
        assert_eq!(table.token_bytes(), 2 * 90);

        let tokens = [100u32, 2_000, 248_044, 31, 31, 77_777];
        let mut staged = vec![0u8; tokens.len() * table.token_bytes()];
        table.stage(&mut table.new_context(), &tokens, &mut staged).unwrap();
        let mut ids = Vec::new();
        table.hasher().hash(&mut table.new_context(), &tokens, &mut ids);
        let rows = reader.payload(&flash_next::ngram_table_name(&FlashNextGeometry::fixture())).unwrap().data;
        let want: Vec<u8> = ids.iter().flat_map(|&r| rows[r as usize * 90..r as usize * 90 + 90].to_vec()).collect();
        assert_eq!(staged, want, "each token's rows, in head order, as the artifact stores them");
    }

    #[test]
    fn a_table_smaller_than_the_hash_ranges_is_refused() {
        let table_file = TableFile::write("small", 500, 4096);
        let err = NgramTable::open(&table_file.path, table_file.layout, hasher(1_000), &[], options(0))
            .err()
            .expect("refused");
        assert!(err.contains("the table holds 500 rows, the hash ranges reach 1000"), "{err}");
    }
}
