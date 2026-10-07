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
use ignis_artifact::{AlignedBuffer, DirectReader, NumericFormat, Object, Reader, DIRECT_IO_ALIGNMENT};

use crate::compute::NgramGeometry;
use crate::ngram::{
    plan_gather, AlignedRead, GatherPlan, HotRows, NgramContext, NgramHashBuffers, NgramHasher, ReadPolicy,
    TableLayout,
};

/// The RAM the hot-row cache may take, its index included (spec 04's
/// default).
pub const DEFAULT_HOT_BYTES: u64 = 1 << 30;
/// Worker threads issuing the table's reads: the queue depth a prefill
/// chunk's tens of thousands of row reads get from the NVMe (GitHub #306).
pub const DEFAULT_READ_THREADS: usize = 16;
/// The longest read a step issues: rows whose sectors touch share one up to
/// this.
pub const DEFAULT_MAX_READ_BYTES: u64 = 64 << 10;
/// The longest read the hot-row load issues: the hot rows are dense enough in
/// the table that the load is close to a scan.
const HOT_LOAD_MAX_READ_BYTES: u64 = 4 << 20;
/// The file span one hot-row load plan covers: its reads, all held until
/// the plan is gathered, take at most this much RAM beside the cache.
const HOT_LOAD_SPAN_BYTES: u64 = 64 << 20;

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

/// Rows gathered so far, by source: plain counters the server reads and exposes.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct NgramCounters {
    /// Rows staged: the hot rows and the file's.
    pub rows: u64,
    /// Of them, rows the hot-row cache held.
    pub hot_rows: u64,
    /// Of them, rows read from the file.
    pub file_rows: u64,
    /// Reads issued to the file, and their bytes.
    pub reads: u64,
    pub read_bytes: u64,
}

/// The table's counts, which a reader may hold past the table: every one
/// counted at its source and only growing, so no reader derives one from two.
#[derive(Debug, Default)]
pub struct NgramCounts {
    hot_rows: AtomicU64,
    file_rows: AtomicU64,
    reads: AtomicU64,
    read_bytes: AtomicU64,
}

impl NgramCounts {
    /// One gather's rows by source, and the reads behind them.
    pub fn record(&self, hot_rows: u64, file_rows: u64, reads: u64, read_bytes: u64) {
        self.hot_rows.fetch_add(hot_rows, Ordering::Relaxed);
        self.file_rows.fetch_add(file_rows, Ordering::Relaxed);
        self.reads.fetch_add(reads, Ordering::Relaxed);
        self.read_bytes.fetch_add(read_bytes, Ordering::Relaxed);
    }

    pub fn read(&self) -> NgramCounters {
        let hot_rows = self.hot_rows.load(Ordering::Relaxed);
        let file_rows = self.file_rows.load(Ordering::Relaxed);
        NgramCounters {
            rows: hot_rows + file_rows,
            hot_rows,
            file_rows,
            reads: self.reads.load(Ordering::Relaxed),
            read_bytes: self.read_bytes.load(Ordering::Relaxed),
        }
    }
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
    /// The cached rows' bytes, in slot order.
    hot_data: Vec<u8>,
    layout: TableLayout,
    policy: ReadPolicy,
    pool: ReadPool,
    counts: Arc<NgramCounts>,
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
        let mut table = Self::open_unloaded(path, layout, hasher, ranked_hot, options)?;
        table.hot_data = table.load_hot_rows()?;
        Ok(table)
    }
    fn open_unloaded(
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
        let table = Self {
            hasher,
            hot,
            hot_data: Vec::new(),
            layout,
            policy,
            pool: ReadPool::new(path, options.read_threads)?,
            counts: Arc::default(),
        };
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
        let mut table = Self::prepare_from_artifact(path, reader, bound, geometry, options)?;
        table.hot_data = table.load_hot_rows()?;
        Ok(table)
    }
    /// Close the artifact mapping before loading or restoring the hot rows.
    pub fn from_cached_artifact(
        path: &Path,
        reader: Reader,
        bound: &FlashNextPlan,
        geometry: NgramGeometry,
        options: NgramTableOptions,
        persistence: &crate::ngram_cache::PersistenceOptions,
    ) -> Result<Self, String> {
        let mut table = Self::prepare_from_artifact(path, &reader, bound, geometry, options)?;
        let key = if persistence.enabled {
            let layout = &table.layout;
            let words = [layout.base_offset, layout.row_stride, layout.row_bytes, layout.rows];
            crate::ngram_cache::identity(path, reader.content_hash(), &words, table.hot.rows())
        } else {
            Err("disabled".into())
        };
        drop(reader);
        let len = table.hot.len().checked_mul(table.row_bytes()).ok_or("hot cache size overflow")?;
        table.hot_data = crate::ngram_cache::load_or_build(persistence, path, key, len, || table.load_hot_rows())?;
        Ok(table)
    }
    fn prepare_from_artifact(
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

        // The container mapping (spec flash-next/01): the hash buffers are I64,
        // the hot list I32 row ids.
        let words = |name: &str, format: NumericFormat| integer_vector(reader, name, format);
        let buffers = NgramHashBuffers {
            layer_multipliers: words(&format!("{prefix}.layer_multipliers"), NumericFormat::I64)?,
            head_vocab_sizes: words(&format!("{prefix}.ngram_heads_vocab_sizes"), NumericFormat::I64)?,
            head_offsets: words(&format!("{prefix}.ngram_heads_offsets"), NumericFormat::I64)?,
        };
        let ranked = words(&format!("{prefix}.ngram_embedding.hot_rows"), NumericFormat::I32)?
            .into_iter()
            .map(|row| u64::try_from(row).map_err(|_| format!("hot row {row} is negative")))
            .collect::<Result<Vec<u64>, String>>()?;
        let hasher = NgramHasher::new(geometry, buffers)?;
        Self::open_unloaded(path, layout, hasher, &ranked, options)
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

    /// Hash a step's tokens for several sequences at once, each lane's
    /// context continuing with its tokens, and start fetching their rows in
    /// one plan, lane-major: the order of a call's `Batch` rows (a decode
    /// step: one token per lane).
    pub fn begin_batch(&self, lanes: &mut [(&mut NgramContext, &[u32])]) -> Result<PendingRows<'_>, String> {
        let mut rows = Vec::with_capacity(lanes.iter().map(|(_, t)| t.len()).sum::<usize>() * self.hasher.heads());
        for (context, tokens) in lanes.iter_mut() {
            self.hasher.hash(context, tokens, &mut rows);
        }
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
        self.counts.read()
    }

    /// The counts themselves, for a reader on another thread.
    pub fn counts(&self) -> Arc<NgramCounts> {
        Arc::clone(&self.counts)
    }

    /// Read the hot rows from the file into their slots, with the coalescing
    /// reads the step path uses but longer, one plan per HOT_LOAD_SPAN_BYTES
    /// of the file: the rows are in ascending order, so a plan's reads lie in
    /// its rows' span and the RAM they hold is bounded by bytes, whatever the
    /// rows' density.
    fn load_hot_rows(&self) -> Result<Vec<u8>, String> {
        let row_bytes = self.row_bytes();
        let mut bytes = vec![0u8; self.hot.len() * row_bytes];
        let empty = HotRows::from_ranked(&[], 0, self.layout.row_bytes)?;
        let policy = ReadPolicy { alignment: DIRECT_IO_ALIGNMENT, max_read_bytes: HOT_LOAD_MAX_READ_BYTES.max(self.policy.max_read_bytes) };
        let rows: Vec<u64> = self.hot.rows().iter().map(|&row| u64::from(row)).collect();
        let mut first = 0;
        while first < rows.len() {
            let span_start = rows[first] * self.layout.row_stride;
            let end = first + rows[first..].partition_point(|&row| row * self.layout.row_stride + self.layout.row_bytes - span_start <= HOT_LOAD_SPAN_BYTES);
            let end = end.max(first + 1);
            let plan = plan_gather(&rows[first..end], &empty, &self.layout, policy)?;
            let results = self.pool.submit(&plan.reads)?;
            collect_and_gather(&plan, results, &[], &mut bytes[first * row_bytes..end * row_bytes])?;
            first = end;
        }
        Ok(bytes)
    }
}

/// An I64 or I32 vector of the artifact, as i64, refused unless the
/// container stores it in exactly that format.
fn integer_vector(reader: &Reader, name: &str, format: NumericFormat) -> Result<Vec<i64>, String> {
    let width = match format {
        NumericFormat::I64 => 8,
        NumericFormat::I32 => 4,
        other => return Err(format!("{} is not an integer format", other.name())),
    };
    match reader.find(name) {
        Some(Object::Tensor(t)) if t.format == format && t.shape.len() == 1 => {}
        _ => return Err(format!("{name} is not a {} vector", format.name())),
    }
    let data = reader.payload(name).map_err(|e| e.to_string())?.data;
    Ok(data
        .chunks_exact(width)
        .map(|w| match width {
            8 => i64::from_le_bytes(w.try_into().unwrap()),
            _ => i64::from(i32::from_le_bytes(w.try_into().unwrap())),
        })
        .collect())
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
        collect_and_gather(&self.plan, self.results, &table.hot_data, out)?;
        let hot = self.plan.sources.iter().filter(|s| matches!(s, crate::ngram::RowSource::Hot { .. })).count();
        table.counts.record(
            hot as u64,
            (self.plan.sources.len() - hot) as u64,
            self.plan.reads.len() as u64,
            self.plan.read_bytes(),
        );
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
    fn a_default_table_keeps_sixteen_reads_in_flight() {
        // GitHub #306: an 8192-token Flash-Next chunk gathers ~46K file reads
        // before the card can start; four readers took ~1.2 s of it with the
        // GPU idle, sixteen take the chunk's TTFT from 3.7 to 2.9 s.
        assert_eq!(NgramTableOptions::default().read_threads, 16);
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
        assert_eq!(counters.file_rows, 5, "the other five came from the file");
        assert!(counters.reads >= 1 && counters.read_bytes % 4096 == 0);
        let counts = table.counts();
        drop(table);
        assert_eq!(counts.read(), counters, "a reader's handle outlives the table and reads what it counted");
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
    fn a_hot_list_spread_over_more_than_one_load_span_loads_every_row() {
        // 800,000 rows of 90 bytes span ~69 MiB: more than one 64 MiB load
        // plan, with hot rows on both sides of the boundary.
        let table_file = TableFile::write("span", 800_000, 4096);
        let ranked: Vec<u64> = (0..800_000).step_by(997).collect();
        let table = NgramTable::open(&table_file.path, table_file.layout, hasher(800_000), &ranked, options(1 << 20)).unwrap();
        assert_eq!(table.hot_rows(), ranked.len());
        let mut out = vec![0u8; ranked.len() * ROW_BYTES as usize];
        table.begin_rows(&ranked).unwrap().finish(&mut out).unwrap();
        assert_eq!(out, expected(&ranked));
        assert_eq!(table.counters().reads, 0, "every row came from the cache");
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
    fn a_decode_step_stages_every_lanes_token_lane_major() {
        let table_file = TableFile::write("lanes", 1_000, 4096);
        let table = NgramTable::open(&table_file.path, table_file.layout, hasher(1_000), &[5, 50], options(1 << 16)).unwrap();
        let prompts: [&[u32]; 3] = [&[1, 2, 3], &[9, 7], &[4]];
        let next = [21u32, 22, 23];
        // Each sequence alone: its prompt, then its next token.
        let mut alone = Vec::new();
        for (prompt, &token) in prompts.iter().zip(&next) {
            let mut context = table.new_context();
            let mut out = vec![0u8; prompt.len() * table.token_bytes()];
            table.stage(&mut context, prompt, &mut out).unwrap();
            let mut step = vec![0u8; table.token_bytes()];
            table.stage(&mut context, &[token], &mut step).unwrap();
            alone.push(step);
        }
        // The three prompts admitted, then one decode step for all lanes.
        let mut contexts: Vec<NgramContext> = (0..3).map(|_| table.new_context()).collect();
        for (context, prompt) in contexts.iter_mut().zip(prompts) {
            let mut out = vec![0u8; prompt.len() * table.token_bytes()];
            table.stage(context, prompt, &mut out).unwrap();
        }
        let mut lanes: Vec<(&mut NgramContext, &[u32])> =
            contexts.iter_mut().zip(&next).map(|(c, t)| (c, std::slice::from_ref(t))).collect();
        let pending = table.begin_batch(&mut lanes).unwrap();
        assert_eq!(pending.rows(), 3 * 2);
        let mut batch = vec![0u8; 3 * table.token_bytes()];
        pending.finish(&mut batch).unwrap();
        assert_eq!(batch, alone.concat());
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
        let unmapped = NgramTable::from_cached_artifact(
            &artifact.path,
            Reader::open(&artifact.path).unwrap(),
            &bound,
            geometry,
            options(3 * 94),
            &crate::ngram_cache::PersistenceOptions { enabled: false, location: crate::ngram_cache::CacheLocation::Model },
        )
        .unwrap();
        assert_eq!(unmapped.hot_data, table.hot_data, "early unmapping preserves all cached bytes");
        let cache_path = artifact.path.with_extension("ngram-test-cache");
        let persistence = crate::ngram_cache::PersistenceOptions {
            enabled: true,
            location: crate::ngram_cache::CacheLocation::Directory(cache_path.clone()),
        };
        for budget in [3 * 94, 3 * 94, 2 * 94] {
            let cached = NgramTable::from_cached_artifact(
                &artifact.path,
                Reader::open(&artifact.path).unwrap(),
                &bound,
                geometry,
                options(budget),
                &persistence,
            )
            .unwrap();
            assert_eq!(cached.hot_data, table.hot_data[..cached.hot_rows() * 90]);
            let selected = [0, 3, 17, 491, 499, 999, 17];
            let mut restored = vec![0u8; selected.len() * 90];
            let mut expected = restored.clone();
            cached.begin_rows(&selected).unwrap().finish(&mut restored).unwrap();
            table.begin_rows(&selected).unwrap().finish(&mut expected).unwrap();
            assert_eq!(restored, expected, "restored hot rows and cold gathers agree");
        }
        let saved = std::fs::read_dir(&cache_path)
            .unwrap()
            .filter(|entry| entry.as_ref().unwrap().path().extension().is_some_and(|ext| ext == "bin"))
            .count();
        assert_eq!(saved, 1, "the 2-row budget's cache replaced the 3-row one");
        std::fs::remove_dir_all(cache_path).unwrap();
        let selected = [0, 3, 17, 491, 499, 999, 17];
        let mut cold = vec![0u8; selected.len() * 90];
        let mut early = cold.clone();
        table.begin_rows(&selected).unwrap().finish(&mut cold).unwrap();
        unmapped.begin_rows(&selected).unwrap().finish(&mut early).unwrap();
        assert_eq!(cold, early, "cache hits, direct reads, boundary and duplicate rows agree");
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
    fn the_artifacts_n_gram_vectors_are_read_only_in_their_stored_format() {
        let artifact = ignis_artifact::flash_next::fixture::build("ngram-types").unwrap();
        let reader = Reader::open(&artifact.path).unwrap();
        let prefix = "layers.1.ple.ple_embedding";
        let hot = format!("{prefix}.ngram_embedding.hot_rows");
        let multipliers = format!("{prefix}.layer_multipliers");
        let rows = integer_vector(&reader, &hot, NumericFormat::I32).unwrap();
        assert_eq!(rows, [17, 3, 999, 0, 512, 64]);
        assert_eq!(integer_vector(&reader, &multipliers, NumericFormat::I64).unwrap().len(), 3);
        // The other width would decode garbage: refused instead.
        let err = integer_vector(&reader, &hot, NumericFormat::I64).unwrap_err();
        assert!(err.contains("is not a I64 vector"), "{err}");
        let err = integer_vector(&reader, &multipliers, NumericFormat::I32).unwrap_err();
        assert!(err.contains("is not a I32 vector"), "{err}");
        let err = integer_vector(&reader, &format!("{prefix}.ngram_embedding.weight"), NumericFormat::I32).unwrap_err();
        assert!(err.contains("is not a I32 vector"), "{err}");
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
