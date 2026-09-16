//! Prompt layout (`processor.cpp` `expand_placeholders`, `assign_positions`):
//! each rendered `<|image_pad|>` becomes its merged grid's run of pad
//! tokens, byte boundaries the caller tracks are shifted across the
//! expansion, and every token gets its three-axis position.

use super::{Grid, ProcessorError, TokenSpan, IMAGE_PAD, IMAGE_PAD_ID, MERGE, VIDEO_PAD, VIDEO_PAD_ID};

/// Token type of a text token.
pub const TEXT: u8 = 0;
/// Token type of an image placeholder token.
pub const IMAGE: u8 = 1;
/// Token type of a video placeholder token.
pub const VIDEO: u8 = 2;

/// The rendered text with every image placeholder expanded, and the
/// caller's byte boundaries moved to the same place in it.
#[derive(Debug, PartialEq, Eq)]
pub struct Expanded {
    pub text: String,
    pub boundaries: Vec<usize>,
}

/// Expand the i-th `<|image_pad|>` of `rendered` into `grids[i]`'s merged
/// token count. A boundary at or after a placeholder's end moves with it; a
/// boundary strictly inside a placeholder is a logic error, as is a
/// placeholder left over or a video placeholder where an image is due.
pub fn expand_placeholders(rendered: &str, grids: &[Grid], boundaries: &[usize]) -> Result<Expanded, ProcessorError> {
    let mut text = String::with_capacity(
        rendered.len() + grids.iter().map(|g| g.vision_tokens() as usize * IMAGE_PAD.len()).sum::<usize>(),
    );
    let mut shifted = boundaries.to_vec();
    let mut search = 0;
    for grid in grids {
        let rest = &rendered[search..];
        let position = match (rest.find(IMAGE_PAD), rest.find(VIDEO_PAD)) {
            (Some(image), Some(video)) if video < image => None,
            (image, _) => image.map(|at| search + at),
        }
        .ok_or(ProcessorError::PlaceholderMismatch("chat media order does not match rendered placeholders"))?;
        let end = position + IMAGE_PAD.len();
        let copies = grid.vision_tokens() as usize;
        let growth = (copies - 1) * IMAGE_PAD.len();
        for (&boundary, out) in boundaries.iter().zip(&mut shifted) {
            if position < boundary && boundary < end {
                return Err(ProcessorError::BoundaryInsidePlaceholder);
            }
            if end <= boundary {
                *out += growth;
            }
        }
        text.push_str(&rendered[search..position]);
        for _ in 0..copies {
            text.push_str(IMAGE_PAD);
        }
        search = end;
    }
    let rest = &rendered[search..];
    if rest.contains(IMAGE_PAD) || rest.contains(VIDEO_PAD) {
        return Err(ProcessorError::PlaceholderMismatch("rendered chat has unbound vision placeholders"));
    }
    text.push_str(rest);
    Ok(Expanded { text, boundaries: shifted })
}

/// Per-token modality: [`IMAGE`] / [`VIDEO`] for the pad ids, [`TEXT`]
/// otherwise.
pub fn token_types(ids: &[u32]) -> Vec<u8> {
    ids.iter()
        .map(|&id| match id {
            IMAGE_PAD_ID => IMAGE,
            VIDEO_PAD_ID => VIDEO,
            _ => TEXT,
        })
        .collect()
}

/// Axis-major `[3, T]` positions (temporal, height, width), each image's
/// token span, and `rope_delta`. Text runs advance all axes together; an
/// image run takes `(current, current + y, current + x)` over its merged
/// grid and advances `current` by the grid's longer merged side.
pub fn assign_positions(types: &[u8], grids: &[Grid]) -> Result<(Vec<i32>, Vec<TokenSpan>, i32), ProcessorError> {
    let length = types.len();
    let mut positions = vec![0i32; length * 3];
    let mut spans = Vec::with_capacity(grids.len());
    let (mut current, mut maximum) = (0i32, 0i32);
    let mut begin = 0;
    while begin < length {
        let modality = types[begin];
        let end = begin + types[begin..].iter().take_while(|&&t| t == modality).count();
        match modality {
            TEXT => {
                for i in begin..end {
                    let position = current + (i - begin) as i32;
                    for axis in 0..3 {
                        positions[axis * length + i] = position;
                    }
                    maximum = maximum.max(position);
                }
                current += (end - begin) as i32;
            }
            IMAGE => {
                let grid = grids
                    .get(spans.len())
                    .ok_or(ProcessorError::PlaceholderMismatch("more image token runs than image inputs"))?;
                let (gh, gw) = ((grid.h as usize / MERGE) as i32, (grid.w as usize / MERGE) as i32);
                if end - begin != (gh * gw) as usize {
                    return Err(ProcessorError::PlaceholderMismatch("vision placeholder run does not match media grid"));
                }
                spans.push(TokenSpan { begin, count: end - begin });
                let mut index = begin;
                for y in 0..gh {
                    for x in 0..gw {
                        positions[index] = current;
                        positions[length + index] = current + y;
                        positions[2 * length + index] = current + x;
                        maximum = maximum.max(current + y.max(x));
                        index += 1;
                    }
                }
                current += gh.max(gw);
            }
            _ => return Err(ProcessorError::PlaceholderMismatch("more video token runs than video temporal grids")),
        }
        begin = end;
    }
    if spans.len() != grids.len() {
        return Err(ProcessorError::PlaceholderMismatch("media grid count does not match placeholder runs"));
    }
    Ok((positions, spans, maximum + 1 - length as i32))
}

#[cfg(test)]
mod tests {
    use super::*;

    const PAD: &str = IMAGE_PAD;

    fn grid(h: u32, w: u32) -> Grid {
        Grid { t: 1, h, w }
    }

    #[test]
    fn each_placeholder_becomes_its_merged_grid_count() {
        let rendered = format!("a<|vision_start|>{PAD}<|vision_end|>b<|vision_start|>{PAD}<|vision_end|>c");
        let out = expand_placeholders(&rendered, &[grid(4, 6), grid(2, 2)], &[]).unwrap();
        assert_eq!(
            out.text,
            format!("a<|vision_start|>{}<|vision_end|>b<|vision_start|>{PAD}<|vision_end|>c", PAD.repeat(6))
        );
    }

    #[test]
    fn boundaries_after_a_placeholder_shift_and_before_it_stay() {
        let rendered = format!("head{PAD}mid{PAD}tail");
        let before = 2;
        let at_first_end = 4 + PAD.len();
        let after_both = rendered.len() - 2;
        let out = expand_placeholders(&rendered, &[grid(2, 4), grid(4, 4)], &[before, at_first_end, after_both]).unwrap();
        let growth_first = PAD.len();
        let growth_both = PAD.len() + 3 * PAD.len();
        assert_eq!(out.boundaries, vec![before, at_first_end + growth_first, after_both + growth_both]);
        assert_eq!(&out.text[..out.boundaries[1]], format!("head{}", PAD.repeat(2)));
        assert_eq!(&out.text[out.boundaries[2]..], "il");
    }

    #[test]
    fn a_boundary_inside_a_placeholder_is_a_logic_error() {
        let rendered = format!("x{PAD}y");
        assert_eq!(
            expand_placeholders(&rendered, &[grid(2, 2)], &[3]),
            Err(ProcessorError::BoundaryInsidePlaceholder)
        );
    }

    #[test]
    fn a_leftover_or_missing_placeholder_is_a_mismatch() {
        let two = format!("{PAD}{PAD}");
        assert!(matches!(
            expand_placeholders(&two, &[grid(2, 2)], &[]),
            Err(ProcessorError::PlaceholderMismatch(_))
        ));
        assert!(matches!(
            expand_placeholders(PAD, &[grid(2, 2), grid(2, 2)], &[]),
            Err(ProcessorError::PlaceholderMismatch(_))
        ));
    }

    #[test]
    fn a_video_placeholder_before_the_image_is_an_order_mismatch() {
        let rendered = format!("{VIDEO_PAD}{PAD}");
        assert!(matches!(
            expand_placeholders(&rendered, &[grid(2, 2)], &[]),
            Err(ProcessorError::PlaceholderMismatch(_))
        ));
    }

    #[test]
    fn text_only_positions_are_equal_on_all_axes_with_zero_delta() {
        let (positions, spans, delta) = assign_positions(&[TEXT; 5], &[]).unwrap();
        assert_eq!(positions, [0, 1, 2, 3, 4].repeat(3));
        assert!(spans.is_empty());
        assert_eq!(delta, 0);
    }

    #[test]
    fn an_image_run_spreads_over_height_and_width_and_advances_by_the_longer_side() {
        // text, text, image 2x3 merged (grid 4x6), text
        let types = [TEXT, TEXT, IMAGE, IMAGE, IMAGE, IMAGE, IMAGE, IMAGE, TEXT];
        let (p, spans, delta) = assign_positions(&types, &[grid(4, 6)]).unwrap();
        let axis = |a: usize| &p[a * 9..(a + 1) * 9];
        assert_eq!(axis(0), [0, 1, 2, 2, 2, 2, 2, 2, 5]);
        assert_eq!(axis(1), [0, 1, 2, 2, 2, 3, 3, 3, 5]);
        assert_eq!(axis(2), [0, 1, 2, 3, 4, 2, 3, 4, 5]);
        assert_eq!(spans, vec![TokenSpan { begin: 2, count: 6 }]);
        // max position 5, 9 tokens.
        assert_eq!(delta, 5 + 1 - 9);
    }

    #[test]
    fn a_run_that_does_not_match_its_grid_is_a_mismatch() {
        assert!(matches!(
            assign_positions(&[IMAGE; 3], &[grid(2, 2)]),
            Err(ProcessorError::PlaceholderMismatch(_))
        ));
        assert!(matches!(assign_positions(&[TEXT, IMAGE], &[]), Err(ProcessorError::PlaceholderMismatch(_))));
        assert!(matches!(assign_positions(&[TEXT], &[grid(2, 2)]), Err(ProcessorError::PlaceholderMismatch(_))));
    }

    #[test]
    fn pad_ids_map_to_their_token_types() {
        assert_eq!(token_types(&[1, IMAGE_PAD_ID, VIDEO_PAD_ID]), vec![TEXT, IMAGE, VIDEO]);
    }
}
