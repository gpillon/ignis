//! The **anchored reading** (spec 14 acceptance 2, spec 15 acceptance 2;
//! GitHub #263 and #264): a pure host function of the pointing head's map,
//! the head set's argmax cells with the scores around them, the grid and the
//! image size.
//!
//! Three kinds of case. The **golden** ones are real: seven scenes the vision
//! study dumped through `crates/server/tests/attention_head_point_gpu.rs`
//! (buttons and large rectangles at 1024 px, the owner's Doom screenshot on
//! its non-square 15x27 grid, a 4096 px rectangle on the 128x128 grid), read
//! by `tools/pointing-scenes/ensemble_score.py golden` with the served head
//! set — the reference every number in the findings was measured with. The
//! **sub-cell** ones pin the parabola that reads a peak inside its cell. The
//! **table** ones pin the rule's edges by hand, where the real maps rarely
//! go.

use ignis_core::pointing::{
    ALLOWANCE_FLAT, ALLOWANCE_MIN_CELLS, ALLOWANCE_PER_N, AnchoredReading, read_anchored,
    sub_cell_offset,
};
use serde_json::Value;

struct Case {
    name: String,
    rows: usize,
    cols: usize,
    width: u32,
    height: u32,
    anchor_scores: Vec<f32>,
    set_argmax: Vec<u32>,
    set_peak: Vec<f32>,
    set_neighbours: Vec<Option<f32>>,
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
            set_peak: floats("set_peak").into_iter().map(|s| s as f32).collect(),
            // Four per head, row-major: a `null` is a neighbour off the grid.
            set_neighbours: case["set_neighbours"]
                .as_array()
                .expect("set_neighbours")
                .iter()
                .flat_map(|head| head.as_array().expect("four neighbours"))
                .map(|v| v.as_f64().map(|s| s as f32))
                .collect(),
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
            &case.set_peak,
            &case.set_neighbours,
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

/// A set whose heads sit exactly on their cells' centres: a flat
/// neighbourhood has no sub-cell position to read, so every offset is zero
/// and the table states cell geometry as spec 14's cases did.
fn flat(heads: usize) -> (Vec<f32>, Vec<Option<f32>>) {
    (vec![1.0; heads], vec![Some(0.0); 4 * heads])
}

/// The allowance the rule grows the span by, per side, in cells.
fn allowance(distinct: usize) -> f64 {
    ALLOWANCE_FLAT + ALLOWANCE_PER_N / distinct.max(ALLOWANCE_MIN_CELLS) as f64
}

fn read(anchor: &[f32], set: &[u32]) -> AnchoredReading {
    let (peak, around) = flat(set.len());
    read_anchored(anchor, set, &peak, &around, GRID, GRID, SIDE, SIDE).expect("the rule reads")
}

/// The same, with every head's peak leaning by a known amount inside its
/// cell: `a` before and `c` after on each axis, around a peak of `b`.
fn read_leaning(anchor: &[f32], set: &[u32], x: (f32, f32), y: (f32, f32)) -> AnchoredReading {
    let peak = vec![0.0; set.len()];
    let around: Vec<Option<f32>> = set
        .iter()
        .flat_map(|_| [Some(x.0), Some(x.1), Some(y.0), Some(y.1)])
        .collect();
    read_anchored(anchor, set, &peak, &around, GRID, GRID, SIDE, SIDE).expect("the rule reads")
}

// ── the sub-cell peak ───────────────────────────────────────────────────

/// The parabola over a peak and its two in-line neighbours, in cells.
#[test]
fn a_symmetric_neighbourhood_reads_the_cell_s_centre() {
    assert_eq!(sub_cell_offset(Some(0.0), 1.0, Some(0.0)), 0.0);
    assert_eq!(sub_cell_offset(Some(-7.5), 3.25, Some(-7.5)), 0.0);
}

/// A higher score before the peak pulls the position back, a higher one
/// after pushes it on — and by the amount the estimator states: with the
/// neighbours one and three nats below the peak, a quarter of a cell.
#[test]
fn a_leaning_neighbourhood_reads_towards_the_higher_neighbour() {
    assert_eq!(sub_cell_offset(Some(-3.0), 0.0, Some(-1.0)), 0.25);
    assert_eq!(sub_cell_offset(Some(-1.0), 0.0, Some(-3.0)), -0.25);
    assert!(sub_cell_offset(Some(-1.0), 0.0, Some(-1.5)) < 0.0);
}

/// A triple that is not a strict peak has no sub-cell position to read, and
/// the rule says zero rather than extrapolating one.
#[test]
fn a_flat_or_non_concave_neighbourhood_reads_zero() {
    assert_eq!(sub_cell_offset(Some(1.0), 1.0, Some(1.0)), 0.0, "flat");
    assert_eq!(sub_cell_offset(Some(2.0), 1.0, Some(2.0)), 0.0, "a valley, not a peak");
    assert_eq!(sub_cell_offset(Some(2.0), 1.0, Some(0.0)), 0.0, "a slope, not a peak");
    assert_eq!(sub_cell_offset(Some(f32::NAN), 1.0, Some(0.0)), 0.0, "not a number");
}

/// A peak on the image's border has no score beyond it, and the axis it is
/// missing on reads zero — the other axis still reads.
#[test]
fn an_absent_neighbour_reads_zero_on_its_axis_alone() {
    assert_eq!(sub_cell_offset(None, 0.0, Some(-1.0)), 0.0);
    assert_eq!(sub_cell_offset(Some(-1.0), 0.0, None), 0.0);
    assert_eq!(sub_cell_offset(None, 0.0, None), 0.0);
    let anchor = peaked(GRID, GRID, 10, 10);
    let set = [cell(10, 10); 8];
    let peak = vec![0.0; set.len()];
    // x leans, y has no neighbour above.
    let around: Vec<Option<f32>> =
        set.iter().flat_map(|_| [Some(-3.0), Some(-1.0), None, Some(-1.0)]).collect();
    let reading =
        read_anchored(&anchor, &set, &peak, &around, GRID, GRID, SIDE, SIDE).expect("reads");
    let g = allowance(1) * CELL;
    assert!((reading.point().0 - (10.75 * CELL)).abs() < 1e-9, "{:?}", reading.point());
    assert!((reading.point().1 - (10.5 * CELL)).abs() < 1e-9, "{:?}", reading.point());
    assert!((reading.extent.y1 - reading.extent.y0 - 2.0 * g).abs() < 1e-9, "{:?}", reading.extent);
}

/// The estimator never reaches past the cell it read.
///
/// For a genuine peak it cannot: with `b` the largest of the three,
/// `|a - c| <= |a - 2b + c|`, so the vertex always lands inside the cell and
/// the clamp never bites — asserted below across the whole range. The clamp
/// is there for the triple the argmax makes impossible, a middle score below
/// a neighbour that is still concave, where the vertex runs cells away.
#[test]
fn the_sub_cell_offset_is_clamped_to_the_cell() {
    assert_eq!(sub_cell_offset(Some(5.0), 0.0, Some(-6.0)), -0.5, "not a peak, concave");
    assert_eq!(sub_cell_offset(Some(-6.0), 0.0, Some(5.0)), 0.5, "the mirror");
    for a in -40..=0 {
        for c in -40..=0 {
            let offset = sub_cell_offset(Some(a as f32 / 4.0), 0.0, Some(c as f32 / 4.0));
            assert!((-0.5..=0.5).contains(&offset), "a {a} c {c} read {offset}");
        }
    }
}

// ── the allowance ───────────────────────────────────────────────────────

/// The span is grown by what the heads did not observe: the fewer distinct
/// cells they peak on, the more of the object lies outside their span.
#[test]
fn the_allowance_falls_with_the_distinct_cells_the_heads_occupy() {
    let anchor = peaked(GRID, GRID, 10, 10);
    let one = read(&anchor, &[cell(10, 10); 96]);
    let spread: Vec<u32> = (0..96).map(|i| cell(10, 6 + i % 9)).collect();
    let many = read(&anchor, &spread);
    assert_eq!(one.distinct_cells, 1);
    assert_eq!(many.distinct_cells, 9);
    let grown = |r: &AnchoredReading, span: f64| (r.extent.x1 - r.extent.x0 - span) / 2.0;
    assert!((grown(&one, 0.0) - allowance(1) * CELL).abs() < 1e-9, "{:?}", one.extent);
    assert!(
        grown(&many, 8.0 * CELL * 0.7) < grown(&one, 0.0),
        "nine cells of evidence buy a smaller allowance than one"
    );
}

/// The allowance is never evaluated below the fewest distinct cells the rule
/// was measured at: one head-cell and three read the same growth.
#[test]
fn the_allowance_is_bounded_to_the_range_it_was_measured_on() {
    let anchor = peaked(GRID, GRID, 10, 10);
    let one = read(&anchor, &[cell(10, 10); 96]);
    assert_eq!(one.distinct_cells, 1);
    // One distinct cell: the span is a point, so the whole box is the
    // allowance -- and it is the allowance at three, not at one.
    let half = (one.extent.x1 - one.extent.x0) / 2.0;
    assert!((half - allowance(ALLOWANCE_MIN_CELLS) * CELL).abs() < 1e-9, "{:?}", one.extent);
    assert!(half < ALLOWANCE_PER_N * CELL, "the unbounded rule would grow {ALLOWANCE_PER_N} cells");
    assert!((allowance(1) - allowance(ALLOWANCE_MIN_CELLS)).abs() < 1e-12);
}

/// The sub-cell offset moves the whole reading, not just the point: heads
/// leaning a quarter cell right shift the extent by exactly that.
#[test]
fn a_leaning_set_shifts_the_extent_by_the_sub_cell_offset() {
    let anchor = peaked(GRID, GRID, 10, 10);
    let set = [cell(10, 10); 96];
    let flat = read(&anchor, &set);
    let leaning = read_leaning(&anchor, &set, (-3.0, -1.0), (-1.0, -3.0));
    assert!((leaning.extent.x0 - flat.extent.x0 - 0.25 * CELL).abs() < 1e-9, "{:?}", leaning.extent);
    assert!((leaning.extent.y0 - flat.extent.y0 + 0.25 * CELL).abs() < 1e-9, "{:?}", leaning.extent);
    assert!((leaning.extent.x1 - leaning.extent.x0 - (flat.extent.x1 - flat.extent.x0)).abs() < 1e-9);
}

#[test]
fn every_head_on_one_cell_boxes_the_allowance_around_it() {
    let reading = read(&peaked(GRID, GRID, 10, 12), &[cell(10, 12); 96]);
    let extent = reading.extent;
    let g = allowance(1) * CELL;
    // One cell of evidence: the span is a point at its centre, and the box
    // is the allowance around it.
    assert!((extent.x0 - (12.5 * CELL - g)).abs() < 1e-9, "{extent:?}");
    assert!((extent.x1 - (12.5 * CELL + g)).abs() < 1e-9, "{extent:?}");
    assert!((extent.y0 - (10.5 * CELL - g)).abs() < 1e-9, "{extent:?}");
    assert!((extent.y1 - (10.5 * CELL + g)).abs() < 1e-9, "{extent:?}");
    assert_eq!(reading.point(), (12.5 * CELL, 10.5 * CELL));
    assert_eq!(reading.kept, 96);
    assert_eq!(reading.distinct_cells, 1);
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
    let g = allowance(with_distractor.distinct_cells) * CELL;
    assert!(extent.x0 >= 4.0 * CELL - g && extent.x1 <= 7.0 * CELL + g, "{extent:?}");
    assert!(extent.y0 >= 4.0 * CELL - g && extent.y1 <= 7.0 * CELL + g, "{extent:?}");
}

/// The median of an even count is the mean of its two middle distances, not
/// either one, and a cell exactly at the radius is kept.
///
/// Distances 0, 0, 6 and 13 cells: the median is 3 cells, the radius 9, so
/// the cell 6 away is kept and the one 13 away dropped. A lower median (0,
/// floored to one cell: radius 3) would drop the first, an upper one (6:
/// radius 18) keep the second.
#[test]
fn an_even_median_averages_and_the_radius_is_inclusive() {
    let reading = read(
        &peaked(GRID, GRID, 10, 10),
        &[cell(10, 10), cell(10, 10), cell(10, 16), cell(10, 23)],
    );
    assert_eq!(reading.kept, 3);
    assert_eq!(reading.distinct_cells, 2);
    let centre = |col: f64| (col + 0.5) * CELL;
    let g = allowance(2) * CELL;
    // xs kept: two at centre(10), one at centre(16). The 15% quantile sits
    // between the two equal ones, the 85% at 0.7 of the way to centre(16),
    // taken from above.
    let x1 = centre(16.0) - (centre(16.0) - centre(10.0)) * (1.0 - 0.7) + g;
    assert!((reading.extent.x0 - (centre(10.0) - g)).abs() < 1e-9, "{:?}", reading.extent);
    assert!((reading.extent.x1 - x1).abs() < 1e-9, "{:?} vs {x1}", reading.extent);
    assert!((reading.extent.y0 - (centre(10.0) - g)).abs() < 1e-9, "{:?}", reading.extent);
    assert!((reading.extent.y1 - (centre(10.0) + g)).abs() < 1e-9, "{:?}", reading.extent);
}

/// An object in the image's corner is boxed to the image's edge and never
/// past it, on a cell size that is not exact in binary.
#[test]
fn an_extent_at_the_edge_is_clamped_to_the_image() {
    let (rows, cols, width, height) = (15, 27, 850u32, 478u32);
    let last = |row: usize, col: usize| (row * cols + col) as u32;
    let set = [last(14, 26), last(14, 25), last(13, 26), last(14, 26)];
    let (peak, around) = flat(set.len());
    let reading = read_anchored(
        &peaked(rows, cols, 14, 26), &set, &peak, &around, rows, cols, width, height,
    )
    .expect("reads");
    let extent = reading.extent;
    assert!(extent.x1 <= f64::from(width) && extent.y1 <= f64::from(height), "{extent:?}");
    assert!((extent.x1 - f64::from(width)).abs() < 1e-9, "{extent:?}");
    assert!((extent.y1 - f64::from(height)).abs() < 1e-9, "{extent:?}");
    let corner = [0, 1, 27, 0];
    let (peak, around) = flat(corner.len());
    let origin = read_anchored(
        &peaked(rows, cols, 0, 0), &corner, &peak, &around, rows, cols, width, height,
    )
    .expect("reads");
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
    let set = [at(7, 4); 10];
    let (peak, around) = flat(set.len());
    let reading = read_anchored(
        &peaked(rows, cols, 7, 3), &set, &peak, &around, rows, cols, width, height,
    )
    .expect("reads");
    assert_eq!(reading.kept, 10);
    let extent = reading.extent;
    let (gx, gy) = (allowance(1) * cw, allowance(1) * ch);
    assert!((extent.x0 - (4.5 * cw - gx)).abs() < 1e-9, "{extent:?}");
    assert!((extent.x1 - (4.5 * cw + gx)).abs() < 1e-9, "{extent:?}");
    assert!((extent.y0 - (7.5 * ch - gy)).abs() < 1e-9, "{extent:?}");
    assert!((extent.y1 - (7.5 * ch + gy)).abs() < 1e-9, "{extent:?}");
    assert!((reading.anchor.x - 3.5).abs() < 1e-12 && (reading.anchor.y - 7.5).abs() < 1e-12);
}

/// A set the leaf did not read whole is not a set.
#[test]
fn an_empty_set_or_a_cell_off_the_grid_does_not_read() {
    let anchor = peaked(GRID, GRID, 3, 3);
    let read_with = |set: &[u32], peak: &[f32], around: &[Option<f32>]| {
        read_anchored(&anchor, set, peak, around, GRID, GRID, SIDE, SIDE)
    };
    let (one_peak, one_around) = flat(1);
    let (two_peak, two_around) = flat(2);
    assert!(read_with(&[], &[], &[]).is_none(), "an empty set");
    assert!(
        read_with(&[5, (GRID * GRID) as u32], &two_peak, &two_around).is_none(),
        "a cell off the grid"
    );
    let (short_peak, short_around) = flat(4);
    assert!(read_anchored(&anchor[1..], &[5], &one_peak, &one_around, GRID, GRID, SIDE, SIDE)
        .is_none(), "the anchor's map is short");
    // The peaks and the neighbours are one and four per head, or the leaf
    // did not read the set whole.
    assert!(read_with(&[5], &one_peak, &short_around).is_none(), "too many neighbours");
    assert!(read_with(&[5], &one_peak, &one_around[..3]).is_none(), "too few neighbours");
    assert!(read_with(&[5], &short_peak, &one_around).is_none(), "too many peaks");
}
