//! The n-gram table's gather plan against a small table file (spec
//! flash-next/04, GitHub #302): a step's rows come from the RAM hot-row cache
//! or from sector-aligned reads of the table's file range, and every row
//! arrives intact whatever the reads look like.
//!
//! The table here is a temporary file laid out the way the artifact lays out
//! its host-streamed table (rows directly addressable at `base + row *
//! stride`, the base not sector-aligned); the plan is executed with plain
//! buffered reads, which is what an unbuffered read returns too.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::PathBuf;

use ignis_core::ngram::{plan_gather, AlignedRead, GatherPlan, HotRows, ReadPolicy, RowSource, TableLayout};

const ROW_BYTES: u64 = 90;
const SECTOR: u64 = 4096;

fn row_content(row: u64) -> Vec<u8> {
    (0..ROW_BYTES).map(|i| ((row * 31 + i * 7) % 251) as u8).collect()
}

/// A table of `rows` rows after `base` header bytes, `stride` bytes apart.
struct Table {
    path: PathBuf,
    layout: TableLayout,
}

impl Table {
    fn write(name: &str, rows: u64, base: u64, stride: u64) -> Self {
        let path = std::env::temp_dir().join(format!("ignis-ngram-{name}-{}.bin", std::process::id()));
        let mut file = File::create(&path).expect("create the table file");
        file.write_all(&vec![0xEE; base as usize]).expect("header");
        for row in 0..rows {
            file.write_all(&row_content(row)).expect("row");
            file.write_all(&vec![0xAA; (stride - ROW_BYTES) as usize]).expect("row padding");
        }
        let layout = TableLayout { base_offset: base, row_stride: stride, row_bytes: ROW_BYTES, rows };
        Self { path, layout }
    }

    fn read(&self, read: &AlignedRead) -> Vec<u8> {
        let mut file = File::open(&self.path).expect("open the table file");
        file.seek(SeekFrom::Start(read.offset)).expect("seek");
        let mut data = Vec::new();
        // A read past the end of the file comes back short, as an unbuffered
        // one does.
        file.take(read.len).read_to_end(&mut data).expect("read");
        data
    }

    /// The bytes the hot-row cache holds: its rows in slot order.
    fn hot_bytes(&self, hot: &HotRows) -> Vec<u8> {
        hot.rows().iter().flat_map(|&row| row_content(u64::from(row))).collect()
    }

    fn execute(&self, plan: &GatherPlan, hot: &HotRows) -> Vec<u8> {
        let data: Vec<Vec<u8>> = plan.reads.iter().map(|read| self.read(read)).collect();
        let reads: Vec<&[u8]> = data.iter().map(Vec::as_slice).collect();
        let mut out = vec![0u8; plan.sources.len() * ROW_BYTES as usize];
        plan.gather(&self.hot_bytes(hot), &reads, &mut out).expect("gather");
        out
    }
}

impl Drop for Table {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

const POLICY: ReadPolicy = ReadPolicy { alignment: SECTOR, max_read_bytes: 64 * 1024 };

fn assert_rows(out: &[u8], rows: &[u64]) {
    for (i, &row) in rows.iter().enumerate() {
        let got = &out[i * ROW_BYTES as usize..(i + 1) * ROW_BYTES as usize];
        assert_eq!(got, row_content(row).as_slice(), "request {i} (row {row})");
    }
}

/// A row whose bytes cross a sector boundary of the file.
fn straddling_row(layout: &TableLayout) -> u64 {
    (0..layout.rows)
        .find(|&row| {
            let start = layout.base_offset + row * layout.row_stride;
            start / SECTOR != (start + ROW_BYTES - 1) / SECTOR
        })
        .expect("a row that crosses a sector")
}

#[test]
fn every_requested_row_arrives_from_ram_or_an_aligned_read() {
    let table = Table::write("mixed", 2000, 1000, ROW_BYTES);
    let hot = HotRows::from_ranked(&[5, 17, 1999, 3, 600], 4 * HOT_ROW_COST, ROW_BYTES).expect("hot rows");
    let straddling = straddling_row(&table.layout);
    let rows = [5, 6, 7, 5, straddling, 1999, 45, 46, 1000, 1998, 0, 6, straddling];
    let plan = plan_gather(&rows, &hot, &table.layout, POLICY).expect("plan");

    assert_rows(&table.execute(&plan, &hot), &rows);
    for read in &plan.reads {
        assert_eq!(read.offset % SECTOR, 0, "{read:?} starts on a sector");
        assert_eq!(read.len % SECTOR, 0, "{read:?} is whole sectors");
        assert!(read.len <= POLICY.max_read_bytes, "{read:?}");
    }
    // The hot rows (5 and 1999; 600 fell outside the 4-row budget) come from
    // RAM, everything else from a read.
    for (i, source) in plan.sources.iter().enumerate() {
        let hot_row = matches!(rows[i], 5 | 1999);
        assert_eq!(matches!(source, RowSource::Hot { .. }), hot_row, "request {i} (row {})", rows[i]);
    }
}

#[test]
fn a_row_asked_twice_is_read_once_and_rows_sharing_a_sector_share_one_read() {
    let table = Table::write("shared", 2000, 1000, ROW_BYTES);
    let hot = HotRows::from_ranked(&[], 0, ROW_BYTES).expect("no hot rows");
    // Rows 1..=3 sit in the first sector after the 1000-byte header.
    let rows = [2, 1, 3, 2, 2];
    let plan = plan_gather(&rows, &hot, &table.layout, POLICY).expect("plan");
    assert_eq!(plan.reads, vec![AlignedRead { offset: 0, len: SECTOR }]);
    assert_eq!(plan.read_bytes(), SECTOR);
    assert_rows(&table.execute(&plan, &hot), &rows);
}

#[test]
fn far_apart_rows_are_separate_reads_and_no_read_passes_the_cap() {
    let table = Table::write("cap", 4000, 1000, ROW_BYTES);
    let hot = HotRows::from_ranked(&[], 0, ROW_BYTES).expect("no hot rows");
    let policy = ReadPolicy { alignment: SECTOR, max_read_bytes: 2 * SECTOR };
    // 300 consecutive rows span ~27 KB: several capped reads. Rows 0 and
    // 3999 are ~350 KB apart: never one read.
    let mut rows: Vec<u64> = (500..800).collect();
    rows.push(0);
    rows.push(3999);
    let plan = plan_gather(&rows, &hot, &table.layout, policy).expect("plan");
    assert!(plan.reads.len() >= 4, "{} reads", plan.reads.len());
    for read in &plan.reads {
        assert!(read.len <= policy.max_read_bytes, "{read:?}");
        assert_eq!((read.offset % SECTOR, read.len % SECTOR), (0, 0), "{read:?}");
    }
    assert_rows(&table.execute(&plan, &hot), &rows);
}

#[test]
fn a_padded_stride_reads_only_the_rows_bytes() {
    // A 128-byte stride (rows padded to a power of two) with an aligned base.
    let table = Table::write("stride", 1000, SECTOR, 128);
    let hot = HotRows::from_ranked(&[31], HOT_ROW_COST, ROW_BYTES).expect("hot rows");
    let rows = [31, 32, 63, 64, 999];
    let plan = plan_gather(&rows, &hot, &table.layout, POLICY).expect("plan");
    assert_rows(&table.execute(&plan, &hot), &rows);
}

/// A hot row costs its bytes and its 4-byte index entry: the budget is the
/// cache's whole RAM.
const HOT_ROW_COST: u64 = ROW_BYTES + 4;

#[test]
fn the_hot_set_is_the_ranked_lists_head_within_the_budget() {
    // Ranked most frequent first, a repeat ignored, the budget fitting three
    // rows (and most of a fourth, which does not count).
    let hot = HotRows::from_ranked(&[9, 4, 9, 7, 2], 4 * HOT_ROW_COST - 1, ROW_BYTES).expect("hot rows");
    assert_eq!(hot.len(), 3);
    let mut held: Vec<u32> = hot.rows().to_vec();
    held.sort_unstable();
    assert_eq!(held, vec![4, 7, 9]);
    for row in [4, 7, 9] {
        let slot = hot.slot(row).expect("a hot row has a slot");
        assert_eq!(u64::from(hot.rows()[slot]), row, "slot {slot} holds row {row}");
    }
    assert_eq!(hot.slot(2), None);
    assert_eq!(hot.bytes(), 3 * HOT_ROW_COST);
    // Three rows' bytes alone do not hold three rows and their index.
    let tight = HotRows::from_ranked(&[9, 4, 7], 3 * ROW_BYTES, ROW_BYTES).expect("hot rows");
    assert_eq!(tight.len(), 2);
    assert!(tight.bytes() <= 3 * ROW_BYTES);
}

#[test]
fn a_whole_table_cache_serves_every_row_from_its_own_slot_and_plans_no_read() {
    // A budget that holds the whole table (GitHub #306): every row is in RAM
    // at slot = row id, whatever the ranking held, and costs no index.
    let table = Table::write("whole", 2000, 1000, ROW_BYTES);
    let hot = HotRows::whole(table.layout.rows, ROW_BYTES).expect("whole table");
    assert!(hot.is_whole());
    assert_eq!(hot.len(), 2000);
    assert_eq!(hot.bytes(), 2000 * ROW_BYTES, "a whole table needs no index");
    let straddling = straddling_row(&table.layout);
    let rows = [1999, 0, 7, 7, 1000, straddling, 1998];
    let plan = plan_gather(&rows, &hot, &table.layout, POLICY).expect("plan");
    assert!(plan.reads.is_empty(), "{:?}", plan.reads);
    for (source, &row) in plan.sources.iter().zip(&rows) {
        assert_eq!(*source, RowSource::Hot { slot: row as usize });
    }
    let every_row: Vec<u8> = (0..2000).flat_map(row_content).collect();
    let mut out = vec![0u8; rows.len() * ROW_BYTES as usize];
    plan.gather(&every_row, &[], &mut out).expect("gather");
    assert_rows(&out, &rows);
    // The table's bounds still hold.
    assert!(plan_gather(&[2000], &hot, &table.layout, POLICY).is_err());
}

#[test]
fn loading_the_hot_rows_is_a_gather_of_its_own() {
    // The cache's rows are read from the table at load with the same plan,
    // in slot order.
    let table = Table::write("load", 2000, 1000, ROW_BYTES);
    let hot = HotRows::from_ranked(&[1500, 3, 700, 4], 4 * HOT_ROW_COST, ROW_BYTES).expect("hot rows");
    let none = HotRows::from_ranked(&[], 0, ROW_BYTES).expect("no hot rows");
    let wanted: Vec<u64> = hot.rows().iter().map(|&row| u64::from(row)).collect();
    let plan = plan_gather(&wanted, &none, &table.layout, POLICY).expect("plan");
    assert_eq!(table.execute(&plan, &none), table.hot_bytes(&hot));
}

#[test]
fn a_plan_refuses_what_it_cannot_read() {
    let layout = TableLayout { base_offset: 1000, row_stride: ROW_BYTES, row_bytes: ROW_BYTES, rows: 2000 };
    let hot = HotRows::from_ranked(&[], 0, ROW_BYTES).expect("no hot rows");
    let past = plan_gather(&[2000], &hot, &layout, POLICY).unwrap_err();
    assert!(past.contains("2000"), "{past}");
    let odd = plan_gather(&[1], &hot, &layout, ReadPolicy { alignment: 3000, max_read_bytes: 6000 }).unwrap_err();
    assert!(odd.contains("power of two"), "{odd}");
    let small = plan_gather(&[1], &hot, &layout, ReadPolicy { alignment: SECTOR, max_read_bytes: SECTOR }).unwrap_err();
    assert!(small.contains("max_read_bytes"), "{small}");
    // A cache of rows of another width than the table's would hand out the
    // wrong bytes for every hot row.
    let other = HotRows::from_ranked(&[1], 2 * HOT_ROW_COST, ROW_BYTES + 6).expect("hot rows");
    let width = plan_gather(&[1], &other, &layout, POLICY).unwrap_err();
    assert!(width.contains("96") && width.contains("90"), "{width}");
}
