//! **E0 of the study** (`docs/specs/decide/11-point-in-one-pass.md`): which
//! 2-D grids can this tokenizer *name*?
//!
//! A chain-free point needs the whole answer in one vocabulary entry, and the
//! only compositional naming the answer alphabet offers is the uppercase
//! bigram: `AX` reads as row `A`, column `X`. That works only if **every**
//! cell of the grid is an admitted label, and 114 of the 676 bigrams are not
//! (ADR 0034) — a grid with a hole in it is a grid whose missing cell reads
//! as some other cell's logit, which is the one thing the alphabet rule
//! exists to prevent.
//!
//! The rectangle also has to be **contiguous**, because the whole argument
//! for a compositional naming is that its legend is one sentence ("rows A-X
//! top to bottom, columns A-X left to right"). A rectangle over a scattered
//! subset of letters needs the 576-line legend back, and spec 08 measured
//! what happens to those.
//!
//! No GPU, no kernel, no materialization: a directory walk and the
//! tokenizer's own round-trip, the same convention as `real_frontend.rs`.
//! Skips when the artifact is not at its machine-local path.

use std::collections::BTreeSet;
use std::path::Path;

use ignis_artifact::{FrontendSet, Reader};
use ignis_core::decision::AnswerAlphabet;

/// The fork-local model cache, as every other real-artifact test names it.
const ARTIFACT: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";

const LETTERS: usize = 26;

fn letter(i: usize) -> char {
    (b'A' + i as u8) as char
}

#[test]
fn the_largest_grid_a_bigram_can_name() {
    let path = Path::new(ARTIFACT);
    if !path.exists() {
        eprintln!("skip: {ARTIFACT} does not exist");
        return;
    }
    let reader = Reader::open(path).unwrap_or_else(|e| panic!("open {ARTIFACT}: {e}"));
    let set = FrontendSet::from_reader(&reader).expect("frontend set");
    let alphabet = AnswerAlphabet::from_tokenizer(set.tokenizer());

    let labels: BTreeSet<&str> =
        alphabet.tokens().iter().map(|t| t.label.as_str()).collect();

    // `admitted[r][c]` — is the bigram `letter(r) letter(c)` a single token
    // that decodes back to itself?
    let mut admitted = [[false; LETTERS]; LETTERS];
    let mut admitted_bigrams = 0usize;
    for r in 0..LETTERS {
        for c in 0..LETTERS {
            let label = format!("{}{}", letter(r), letter(c));
            if labels.contains(label.as_str()) {
                admitted[r][c] = true;
                admitted_bigrams += 1;
            }
        }
    }
    let singles = alphabet.len() - admitted_bigrams;

    eprintln!("alphabet: {} labels, {singles} singles, {admitted_bigrams} bigrams", alphabet.len());

    // The failure map, printed so a reader can see whether the holes cluster.
    eprintln!("     {}", (0..LETTERS).map(letter).collect::<String>());
    for r in 0..LETTERS {
        let row: String = (0..LETTERS)
            .map(|c| if admitted[r][c] { '.' } else { 'x' })
            .collect();
        eprintln!("  {} {row}", letter(r));
    }

    // Every **contiguous** rectangle, exactly: 26^4 is nothing.
    let mut best_area = (0usize, 0usize, 0usize, 0usize, 0usize);
    let mut best_square = (0usize, 0usize, 0usize);
    for r0 in 0..LETTERS {
        for c0 in 0..LETTERS {
            for rows in 1..=(LETTERS - r0) {
                'cols: for cols in 1..=(LETTERS - c0) {
                    for r in r0..r0 + rows {
                        for c in c0..c0 + cols {
                            if !admitted[r][c] {
                                continue 'cols;
                            }
                        }
                    }
                    let area = rows * cols;
                    if area > best_area.0 {
                        best_area = (area, rows, cols, r0, c0);
                    }
                    if rows == cols && rows > best_square.0 {
                        best_square = (rows, r0, c0);
                    }
                }
            }
        }
    }

    let (area, rows, cols, r0, c0) = best_area;
    eprintln!(
        "largest contiguous rectangle: {rows} x {cols} = {area} cells, \
         rows {}-{}, cols {}-{}",
        letter(r0),
        letter(r0 + rows - 1),
        letter(c0),
        letter(c0 + cols - 1)
    );
    let (side, sr, sc) = best_square;
    eprintln!(
        "largest contiguous square: {side} x {side} = {} cells, rows {}-{}, cols {}-{} \
         ({:.2}% of the side per cell)",
        side * side,
        letter(sr),
        letter(sr + side - 1),
        letter(sc),
        letter(sc + side - 1),
        100.0 / side as f64
    );

    // The **non-contiguous** maximum, for completeness: drop whole rows and
    // whole columns rather than trimming edges. Its legend is two short
    // ordered lists rather than one sentence ("rows A,B,C,...,S top to
    // bottom"), which is a weaker claim than a contiguous range but a far
    // weaker one than spec 08's 576-line table. Maximum edge biclique is
    // NP-hard in general; 26 x 26 with 17% holes is small and dense enough
    // that greedy peeling from every seed row finds the answer or misses it
    // by one, which is all this number is used for.
    let mut best_free = (0usize, 0usize, 0usize, Vec::new(), Vec::new());
    for seed in 0..LETTERS {
        let mut rows_in: Vec<usize> = vec![seed];
        let mut cols_in: Vec<usize> = (0..LETTERS).filter(|&c| admitted[seed][c]).collect();
        loop {
            // The row that costs the fewest columns, among those not in yet.
            let next = (0..LETTERS)
                .filter(|r| !rows_in.contains(r))
                .map(|r| {
                    let kept = cols_in.iter().filter(|&&c| admitted[r][c]).count();
                    (r, kept)
                })
                .max_by_key(|&(_, kept)| kept);
            let Some((r, kept)) = next else { break };
            if kept == 0 {
                break;
            }
            let area_now = rows_in.len() * cols_in.len();
            let area_next = (rows_in.len() + 1) * kept;
            rows_in.push(r);
            cols_in.retain(|&c| admitted[r][c]);
            let _ = area_now;
            let _ = area_next;
            let area = rows_in.len() * cols_in.len();
            if area > best_free.0 {
                best_free = (area, rows_in.len(), cols_in.len(), rows_in.clone(), cols_in.clone());
            }
        }
    }
    let (free_area, free_rows, free_cols, ref free_r, ref free_c) = best_free;
    eprintln!(
        "largest free (non-contiguous) rectangle: {free_rows} x {free_cols} = {free_area} cells"
    );
    eprintln!("  rows: {}", free_r.iter().map(|&r| letter(r)).collect::<String>());
    eprintln!("  cols: {}", free_c.iter().map(|&c| letter(c)).collect::<String>());
    let free_side = (free_rows.min(free_cols)) as f64;
    eprintln!(
        "  as a square that is {} x {} ({:.2}% of the side per cell)",
        free_rows.min(free_cols),
        free_rows.min(free_cols),
        100.0 / free_side
    );

    // The two-readout fallback, for the same reason: single letters are a
    // different alphabet and may have no holes at all, in which case a row
    // read and a column read span the full 26 x 26 — at the price of a
    // second position, which is what E2 is about.
    let full_rows = (0..LETTERS).filter(|&r| labels.contains(letter(r).to_string().as_str())).count();
    eprintln!("single uppercase letters admitted: {full_rows} of {LETTERS}");

    // While the tokenizer is open: **count** the rounds a point and a box
    // actually cost today, rather than estimating them. The schedule is the
    // literals after the first (one step per token, spec 06) plus `digits`
    // per axis, and the run costs one round more than its schedule because
    // a decode round returns the token the previous call drew
    // (`constrained.rs`, the one-round lag).
    let encode = |text: &str| set.tokenizer().encode(text).map(|ids| ids.len());
    for (name, literals) in [
        ("point", &[r#"{"x":"#, r#","y":"#][..]),
        (
            "box",
            &[r#"{"x0":"#, r#","y0":"#, r#","x1":"#, r#","y1":"#][..],
        ),
    ] {
        let digits = 3usize;
        let prefix = encode(literals[0]).expect("prefix encodes");
        let forced: usize = literals[1..]
            .iter()
            .map(|l| encode(l).expect("literal encodes"))
            .sum();
        let steps = forced + digits * literals.len();
        eprintln!(
            "{name} at {digits} digits: prefix {prefix} prompt tokens, \
             schedule {steps} steps ({forced} forced literal + {} digit), \
             {} decode rounds",
            digits * literals.len(),
            steps + 1
        );
    }

    // Not an assertion about the model: an assertion that the alphabet was
    // actually read. A zero here means the tokenizer was not consulted.
    assert!(alphabet.len() > 100, "alphabet came back empty: {}", alphabet.len());
    assert!(best_square.0 >= 2, "no 2 x 2 contiguous grid is nameable at all");
}
