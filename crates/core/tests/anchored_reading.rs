//! The **anchored reading** (spec 14, GitHub #263, acceptance 2): a pure host
//! function of the pointing head's map, the head set's argmax cells, the
//! grid and the image size.
//!
//! Two kinds of case. The **golden** ones are real: seven scenes the vision
//! study dumped through `crates/server/tests/attention_head_point_gpu.rs`
//! (buttons and large rectangles at 1024 px, the owner's Doom screenshot on
//! its non-square 15x27 grid, a 4096 px rectangle on the 128x128 grid), read
//! by `tools/pointing-scenes/ensemble_score.py golden` with the served head
//! set — the reference every number in spec 14 was measured with. The
//! **table** ones pin the rule's edges by hand, where the real maps rarely
//! go.

use ignis_core::pointing::{AnchoredReading, read_anchored};
use serde_json::Value;

struct Case {
    name: String,
    rows: usize,
    cols: usize,
    width: u32,
    height: u32,
    anchor_scores: Vec<f32>,
    set_argmax: Vec<u32>,
    anchor: Vec<f64>,
    point: Vec<f64>,
    extent: Vec<f64>,
}

impl Case {
    fn from_json(case: &Value) -> Self {
        let floats = |key: &str| -> Vec<f64> {
            case[key].as_array().unwrap_or_else(|| panic!("{key}")).iter().map(|v| v.as_f64().expect(key)).collect()
        };
        let whole = |key: &str| case[key].as_u64().unwrap_or_else(|| panic!("{key}"));
        Self {
            name: case["name"].as_str().expect("name").to_owned(),
            rows: whole("rows") as usize,
            cols: whole("cols") as usize,
            width: whole("width") as u32,
            height: whole("height") as u32,
            // The dump's scores are f16, so every one is exact in f32.
            anchor_scores: floats("anchor_scores").into_iter().map(|s| s as f32).collect(),
            set_argmax: floats("set_argmax").into_iter().map(|c| c as u32).collect(),
            anchor: floats("anchor"),
            point: floats("point"),
            extent: floats("extent"),
        }
    }
}

/// Pixels. The reference computes in NumPy's float64 with the same
/// operations in the same order; what is left is `exp`'s last bit.
const TOLERANCE: f64 = 1e-6;

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() < TOLERANCE
}

#[test]
fn the_rule_reproduces_the_reference_on_real_dumps() {
    let text = include_str!("fixtures/anchored_reading.json");
    let golden: Value = serde_json::from_str(text).expect("the golden file parses");
    let cases: Vec<Case> = golden["cases"].as_array().expect("cases").iter().map(Case::from_json).collect();
    assert_eq!(cases.len(), 7);
    for case in &cases {
        let reading = read_anchored(
            &case.anchor_scores,
            &case.set_argmax,
            case.rows,
            case.cols,
            case.width,
            case.height,
        )
        .unwrap_or_else(|| panic!("{}: the rule reads", case.name));
        let (cw, ch) = (
            f64::from(case.width) / case.cols as f64,
            f64::from(case.height) / case.rows as f64,
        );
        let anchor = (reading.anchor.x * cw, reading.anchor.y * ch);
        let extent = reading.extent;
        let point = reading.point();
        assert!(
            close(anchor.0, case.anchor[0]) && close(anchor.1, case.anchor[1]),
            "{}: anchor {anchor:?}, reference {:?}",
            case.name,
            case.anchor
        );
        assert!(
            close(extent.x0, case.extent[0])
                && close(extent.y0, case.extent[1])
                && close(extent.x1, case.extent[2])
                && close(extent.y1, case.extent[3]),
            "{}: extent {extent:?}, reference {:?}",
            case.name,
            case.extent
        );
        assert!(
            close(point.0, case.point[0]) && close(point.1, case.point[1]),
            "{}: point {point:?}, reference {:?}",
            case.name,
            case.point
        );
    }
}

// ── the table ───────────────────────────────────────────────────────────

/// A 32x32 grid over a 1024 px image: 32 px cells, exact in binary.
const GRID: usize = 32;
const SIDE: u32 = 1024;
const CELL: f64 = 32.0;

/// A pointing-head map peaked on one cell, so the anchor is that cell's
/// centre exactly.
fn peaked(rows: usize, cols: usize, row: usize, col: usize) -> Vec<f32> {
    let mut scores = vec![0.0; rows * cols];
    scores[row * cols + col] = 10.0;
    scores
}

fn cell(row: usize, col: usize) -> u32 {
    (row * GRID + col) as u32
}

fn read(anchor: &[f32], set: &[u32]) -> AnchoredReading {
    read_anchored(anchor, set, GRID, GRID, SIDE, SIDE).expect("the rule reads")
}

#[test]
fn every_head_on_one_cell_boxes_that_cell() {
    let reading = read(&peaked(GRID, GRID, 10, 12), &[cell(10, 12); 96]);
    let extent = reading.extent;
    assert_eq!((extent.x0, extent.x1), (12.0 * CELL, 13.0 * CELL), "{extent:?}");
    assert_eq!((extent.y0, extent.y1), (10.0 * CELL, 11.0 * CELL), "{extent:?}");
    assert_eq!(reading.point(), (12.5 * CELL, 10.5 * CELL));
    assert_eq!(reading.kept, 96);
    assert_eq!(reading.anchor.cells, 1, "the anchor's confidence is the pointing head's own");
}

/// The anchor decides which object: a third of the set on a distractor far
/// away is dropped whole, and the extent is exactly the one the near cells
/// alone give.
#[test]
fn heads_on_a_far_distractor_are_dropped() {
    let anchor = peaked(GRID, GRID, 5, 5);
    let mut near = Vec::new();
    for i in 0..60 {
        near.push(cell(4 + i % 3, 4 + (i / 3) % 3));
    }
    let mut split = near.clone();
    split.extend(std::iter::repeat_n(cell(25, 25), 36));
    let with_distractor = read(&anchor, &split);
    let alone = read(&anchor, &near);
    assert_eq!(with_distractor.extent, alone.extent);
    assert_eq!(with_distractor.kept, 60);
    let extent = with_distractor.extent;
    assert!(extent.x0 >= 4.0 * CELL && extent.x1 <= 7.0 * CELL, "{extent:?}");
    assert!(extent.y0 >= 4.0 * CELL && extent.y1 <= 7.0 * CELL, "{extent:?}");
}

/// The median of an even count is the mean of its two middle distances, not
/// either one, and a cell exactly at the radius is kept.
///
/// Distances 0, 0, 4 and 7 cells: the median is 2 cells, the radius 4, so
/// the cell 4 away is kept (on the boundary) and the one 7 away dropped. A
/// lower median (0, floored to one cell: radius 2) would drop the first, an
/// upper one (4: radius 8) keep the second.
#[test]
fn an_even_median_averages_and_the_radius_is_inclusive() {
    let reading = read(&peaked(GRID, GRID, 10, 10), &[cell(10, 10), cell(10, 10), cell(10, 14), cell(10, 17)]);
    assert_eq!(reading.kept, 3);
    let centre = |col: f64| (col + 0.5) * CELL;
    // xs kept: 336, 336, 464. The 10% quantile sits between the two 336s,
    // the 90% at 0.8 of the way from 336 to 464, taken from above.
    let x1 = centre(14.0) - (centre(14.0) - centre(10.0)) * (1.0 - 0.8) + CELL / 2.0;
    assert_eq!(reading.extent.x0, centre(10.0) - CELL / 2.0);
    assert!((reading.extent.x1 - x1).abs() < 1e-9, "{:?} vs {x1}", reading.extent);
    assert_eq!((reading.extent.y0, reading.extent.y1), (10.0 * CELL, 11.0 * CELL));
}

/// An object in the image's corner is boxed to the image's edge and never
/// past it, on a cell size that is not exact in binary.
#[test]
fn an_extent_at_the_edge_is_clamped_to_the_image() {
    let (rows, cols, width, height) = (15, 27, 850u32, 478u32);
    let last = |row: usize, col: usize| (row * cols + col) as u32;
    let set = [last(14, 26), last(14, 25), last(13, 26), last(14, 26)];
    let reading = read_anchored(&peaked(rows, cols, 14, 26), &set, rows, cols, width, height).expect("reads");
    let extent = reading.extent;
    assert!(extent.x1 <= f64::from(width) && extent.y1 <= f64::from(height), "{extent:?}");
    assert!((extent.x1 - f64::from(width)).abs() < 1e-9, "{extent:?}");
    assert!((extent.y1 - f64::from(height)).abs() < 1e-9, "{extent:?}");
    let origin = read_anchored(&peaked(rows, cols, 0, 0), &[0, 1, 27, 0], rows, cols, width, height).expect("reads");
    assert_eq!((origin.extent.x0, origin.extent.y0), (0.0, 0.0), "{:?}", origin.extent);
}

/// A non-square grid over a non-square image: each axis by its own cell,
/// and the keep radius by the longer side of one cell.
#[test]
fn a_non_square_grid_scales_each_axis_by_its_own_cell() {
    let (rows, cols, width, height) = (15, 27, 850u32, 478u32);
    let (cw, ch) = (850.0 / 27.0, 478.0 / 15.0);
    let at = |row: usize, col: usize| (row * cols + col) as u32;
    // Every head one column right of the anchor: all at one cell's distance,
    // so the median (cw) is floored to the longer side (ch) and all are kept.
    let reading =
        read_anchored(&peaked(rows, cols, 7, 3), &[at(7, 4); 10], rows, cols, width, height).expect("reads");
    assert_eq!(reading.kept, 10);
    let extent = reading.extent;
    assert!((extent.x0 - 4.0 * cw).abs() < 1e-9 && (extent.x1 - 5.0 * cw).abs() < 1e-9, "{extent:?}");
    assert!((extent.y0 - 7.0 * ch).abs() < 1e-9 && (extent.y1 - 8.0 * ch).abs() < 1e-9, "{extent:?}");
    assert!((reading.anchor.x - 3.5).abs() < 1e-12 && (reading.anchor.y - 7.5).abs() < 1e-12);
}

/// A set the leaf did not read whole is not a set.
#[test]
fn an_empty_set_or_a_cell_off_the_grid_does_not_read() {
    let anchor = peaked(GRID, GRID, 3, 3);
    assert!(read_anchored(&anchor, &[], GRID, GRID, SIDE, SIDE).is_none());
    assert!(read_anchored(&anchor, &[5, (GRID * GRID) as u32], GRID, GRID, SIDE, SIDE).is_none());
    assert!(read_anchored(&anchor[1..], &[5], GRID, GRID, SIDE, SIDE).is_none(), "the anchor's map is short");
}
