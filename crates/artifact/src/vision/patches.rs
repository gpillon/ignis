//! Normalization and patch packing (`processor.cpp` `to_bf16`,
//! `normalization_lut`, `append_patch`, `prepare_image`): each resized
//! pixel becomes `v / 127.5 - 1` stored as BF16, and the image becomes
//! row-major `[raw_patches, 1536]` rows, channel-major `3 x 2 x 16 x 16`
//! with the single frame repeated as the temporal pair, in 2x2 merge-block
//! order — the order the merger's `[4608, V]` view assumes.

use super::{Rgb8Image, MERGE, PATCH, PATCH_FEATURES, TEMPORAL};

/// IEEE half-precision bits of `value`, rounded to nearest even (the
/// reference's bit trick: exact for the 256 normalized values it is used
/// on, which never leave BF16's finite range).
fn to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    (bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

/// The 256 normalized pixel values, `v / 127.5 - 1` in BF16.
fn normalization_lut() -> [u16; 256] {
    let mut lut = [0u16; 256];
    for (value, slot) in lut.iter_mut().enumerate() {
        *slot = to_bf16(value as f32 / 127.5 - 1.0);
    }
    lut
}

/// Pack a resized image (both sides multiples of 32) into its patch rows.
pub fn pack_patches(image: &Rgb8Image) -> Vec<u16> {
    let (gh, gw) = (image.height as usize / PATCH, image.width as usize / PATCH);
    assert!(gh % MERGE == 0 && gw % MERGE == 0, "image is not on the merge grid");
    let lut = normalization_lut();
    let width = image.width as usize;
    let mut out = Vec::with_capacity(gh * gw * PATCH_FEATURES);
    for block_y in 0..gh / MERGE {
        for block_x in 0..gw / MERGE {
            for merge_y in 0..MERGE {
                for merge_x in 0..MERGE {
                    let (grid_y, grid_x) = (block_y * MERGE + merge_y, block_x * MERGE + merge_x);
                    for channel in 0..3 {
                        for _temporal in 0..TEMPORAL {
                            for y in 0..PATCH {
                                let row = ((grid_y * PATCH + y) * width + grid_x * PATCH) * 3 + channel;
                                for x in 0..PATCH {
                                    out.push(lut[image.rgb[row + x * 3] as usize]);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    debug_assert_eq!(out.len(), gh * gw * PATCH_FEATURES);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// bf16 decode (sign, 8-bit exponent, 7-bit mantissa).
    fn from_bf16(bits: u16) -> f32 {
        f32::from_bits((bits as u32) << 16)
    }

    #[test]
    fn the_lut_is_v_over_127_5_minus_one_in_bf16() {
        let lut = normalization_lut();
        assert_eq!(from_bf16(lut[0]), -1.0);
        assert_eq!(from_bf16(lut[255]), 1.0);
        for v in 0..256 {
            let exact = v as f32 / 127.5 - 1.0;
            let got = from_bf16(lut[v]);
            // bf16 keeps f32's exponent and 7 mantissa bits.
            assert!((got - exact).abs() <= exact.abs() / 128.0 + 1e-7, "{v}: {got} vs {exact}");
        }
    }

    #[test]
    fn rounding_is_to_nearest_even() {
        // Exactly halfway between two bf16 values: 1 + 2^-8 sits between
        // mantissa 0 (even) and 1.
        assert_eq!(to_bf16(1.0 + 2f32.powi(-8)), to_bf16(1.0));
        // 1 + 3*2^-8 sits between mantissa 1 and 2 (even).
        assert_eq!(to_bf16(1.0 + 3.0 * 2f32.powi(-8)), to_bf16(1.0 + 2.0 * 2f32.powi(-7)));
    }

    #[test]
    fn rows_are_channel_major_with_the_frame_repeated_in_merge_block_order() {
        // 64x32 image: 4x2 patch grid, 2x1 merge blocks. Pixel (x, y) has
        // R = patch column, G = patch row, B = 0.
        let (w, h) = (64usize, 32usize);
        let mut rgb = vec![0u8; w * h * 3];
        for y in 0..h {
            for x in 0..w {
                rgb[(y * w + x) * 3] = (x / PATCH) as u8;
                rgb[(y * w + x) * 3 + 1] = (y / PATCH) as u8;
            }
        }
        let lut = normalization_lut();
        let rows = pack_patches(&Rgb8Image { width: w as u32, height: h as u32, rgb });
        assert_eq!(rows.len(), 4 * 2 * PATCH_FEATURES);
        // Merge-block order: (gy, gx) = (0,0) (0,1) (1,0) (1,1) (0,2) (0,3) ...
        let expected = [(0, 0), (0, 1), (1, 0), (1, 1), (0, 2), (0, 3), (1, 2), (1, 3)];
        for (row, (gy, gx)) in rows.chunks(PATCH_FEATURES).zip(expected) {
            let plane = PATCH * PATCH;
            // Channel-major: R for both temporal copies, then G, then B.
            assert!(row[..2 * plane].iter().all(|&v| v == lut[gx]));
            assert!(row[2 * plane..4 * plane].iter().all(|&v| v == lut[gy]));
            assert!(row[4 * plane..].iter().all(|&v| v == lut[0]));
        }
    }
}
