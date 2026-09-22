//! The **pointing head**'s region rule (spec 13, GitHub #260): a pure function
//! of one head's scores over an image span and the image's merged token grid.
//!
//! Every number the findings report was measured with this rule (TAG's):
//! `m = exp(s - max s)`, min-max normalize, keep cells at or above 0.5, take
//! the 4-connected region with the highest mean, return its weighted centre.
//! These tests pin its edges on CPU, since the GPU acceptance only ever
//! meets the shapes real maps have — one sharp peak, most of the time.

use ignis_core::pointing::{HeadReading, read_head_map};

/// A map that is `background` everywhere except the `(row, col, score)`
/// cells named.
fn map(rows: usize, cols: usize, background: f32, cells: &[(usize, usize, f32)]) -> Vec<f32> {
    let mut scores = vec![background; rows * cols];
    for &(row, col, score) in cells {
        scores[row * cols + col] = score;
    }
    scores
}

fn read(scores: &[f32], rows: usize, cols: usize) -> HeadReading {
    read_head_map(scores, rows, cols).expect("a finite map of the grid's size is read")
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-9
}

#[test]
fn a_single_peak_is_read_at_its_cells_centre() {
    let scores = map(4, 6, -20.0, &[(2, 3, 5.0)]);
    let reading = read(&scores, 4, 6);
    assert!(close(reading.x, 3.5) && close(reading.y, 2.5), "{reading:?}");
    assert_eq!(reading.cells, 1);
    assert_eq!((reading.rows, reading.cols), (4, 6));
}

/// The share is the region's part of the softmax over the span — the
/// confidence spec 13 exposes — so a peak that stands alone holds nearly all
/// of it and one among many equal cells holds its fraction.
#[test]
fn the_share_is_the_regions_part_of_the_span_softmax() {
    let sharp = read(&map(4, 6, -20.0, &[(2, 3, 5.0)]), 4, 6);
    assert!(sharp.share > 0.999_999, "{sharp:?}");

    // One peak one nat above 23 others: 1 / (1 + 23 e^-1).
    let soft = read(&map(4, 6, 0.0, &[(0, 0, 1.0)]), 4, 6);
    let expected = 1.0 / (1.0 + 23.0 * (-1.0f64).exp());
    assert!(close(soft.share, expected), "{soft:?}, expected {expected}");
}

/// Spec 13: two separated regions, where the one with the higher mean must
/// win over the larger one.
#[test]
fn the_region_with_the_highest_mean_wins_over_the_largest() {
    let low = 0.6f32.ln();
    let scores = map(
        4,
        8,
        -30.0,
        &[(1, 0, low), (1, 1, low), (1, 2, low), (2, 1, low), (3, 6, 0.0)],
    );
    let reading = read(&scores, 4, 8);
    assert!(close(reading.x, 6.5) && close(reading.y, 3.5), "{reading:?}");
    assert_eq!(reading.cells, 1);
}

/// The weighted centre, over a region of more than one cell: each cell
/// weighs its min-max value, so a cell at three quarters of the peak pulls
/// the point three sevenths of the way toward it.
#[test]
fn a_region_is_read_at_its_weighted_centre() {
    let scores = map(3, 5, -30.0, &[(1, 1, 0.0), (1, 2, 0.75f32.ln())]);
    let reading = read(&scores, 3, 5);
    assert_eq!(reading.cells, 2);
    let expected_x = (1.5 * 1.0 + 2.5 * 0.75) / 1.75;
    assert!(close(reading.y, 1.5), "{reading:?}");
    assert!((reading.x - expected_x).abs() < 1e-6, "{reading:?}, expected x {expected_x}");
}

/// Two regions with the same mean: the first in raster order wins, as the
/// scorer every finding used (`np.argmax` over labels numbered in raster
/// order, and the engine harness's strict `>`).
#[test]
fn a_tie_goes_to_the_first_region_in_raster_order() {
    let scores = map(4, 4, -30.0, &[(3, 0, 2.0), (0, 3, 2.0)]);
    let reading = read(&scores, 4, 4);
    assert!(close(reading.x, 3.5) && close(reading.y, 0.5), "{reading:?}");
}

/// 4-connected, not 8: two cells touching only at a corner are two regions,
/// and the reading stays on one of them instead of landing between them.
#[test]
fn cells_touching_at_a_corner_are_two_regions() {
    let scores = map(3, 3, -30.0, &[(0, 0, 1.0), (1, 1, 1.0)]);
    let reading = read(&scores, 3, 3);
    assert_eq!(reading.cells, 1);
    assert!(close(reading.x, 0.5) && close(reading.y, 0.5), "{reading:?}");
}

#[test]
fn a_region_on_the_grids_edge_is_read_inside_it() {
    let scores = map(3, 4, -30.0, &[(2, 3, 1.0), (2, 2, 1.0), (1, 3, 1.0)]);
    let reading = read(&scores, 3, 4);
    assert_eq!(reading.cells, 3);
    assert!(reading.x > 2.0 && reading.x < 4.0 && reading.y > 1.0 && reading.y < 3.0, "{reading:?}");
    assert!(close(reading.x, (2.5 + 3.5 + 3.5) / 3.0), "{reading:?}");
    assert!(close(reading.y, (2.5 + 2.5 + 1.5) / 3.0), "{reading:?}");
}

/// A flat map carries no position at all. The rule the findings measured
/// (and the engine harness) reads it as its first cell; the share, one cell's
/// part of a uniform softmax, is what tells a caller it is worth nothing.
#[test]
fn a_flat_map_reads_its_first_cell_with_the_share_of_one_cell() {
    let scores = vec![3.25f32; 12];
    let reading = read(&scores, 3, 4);
    assert!(close(reading.x, 0.5) && close(reading.y, 0.5), "{reading:?}");
    assert_eq!(reading.cells, 1);
    assert!(close(reading.share, 1.0 / 12.0), "{reading:?}");
}

/// A non-square grid indexes row-major, `cols` wide: a peak in the last cell
/// of the first row is at the right edge, not one row down.
#[test]
fn a_non_square_grid_is_read_row_major() {
    let scores = map(2, 5, -30.0, &[(0, 4, 1.0)]);
    let reading = read(&scores, 2, 5);
    assert!(close(reading.x, 4.5) && close(reading.y, 0.5), "{reading:?}");
}

/// The point in the submitted image's pixels: each axis by its own side,
/// through the grid's own scale, so a wide image whose grid is not square
/// lands where the cell is.
#[test]
fn pixels_scale_each_axis_by_its_own_side() {
    let scores = map(2, 5, -30.0, &[(1, 4, 1.0)]);
    let reading = read(&scores, 2, 5);
    let (x, y) = reading.pixels(1000, 300);
    assert!(close(x, 900.0) && close(y, 225.0), "({x}, {y})");
    let (cell_w, cell_h) = reading.cell_pixels(1000, 300);
    assert!(close(cell_w, 200.0) && close(cell_h, 150.0), "({cell_w}, {cell_h})");
}

/// The model-scale reading beside the pixels: the question's `digits` scale
/// (0-999 at 3), each axis a fraction of its own grid side.
#[test]
fn normalized_reads_on_the_questions_scale() {
    let scores = map(2, 5, -30.0, &[(1, 4, 1.0)]);
    let reading = read(&scores, 2, 5);
    assert_eq!(reading.normalized(999), (899, 749));
    assert_eq!(reading.normalized(99), (89, 74));
    assert_eq!(reading.normalized(9999), (8999, 7499));
    // A cell's centre never reads the top of the range, nor below zero.
    let corner = read(&map(1, 1, 0.0, &[]), 1, 1);
    assert_eq!(corner.normalized(9), (5, 5));
}

#[test]
fn a_map_that_is_not_the_grids_size_or_not_finite_is_not_read() {
    assert!(read_head_map(&[0.0; 5], 2, 3).is_none());
    assert!(read_head_map(&[], 0, 0).is_none());
    assert!(read_head_map(&[0.0, f32::NAN, 1.0, 2.0], 2, 2).is_none());
    assert!(read_head_map(&[0.0, f32::INFINITY, 1.0, 2.0], 2, 2).is_none());
}
