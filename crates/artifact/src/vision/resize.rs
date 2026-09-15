//! Smart resize and the antialiased bicubic filter, step for step as the
//! reference processor (`processor.cpp` `smart_resize_image`,
//! `coefficients`, `resize_bicubic`). Every floating-point operation keeps
//! the reference's type and order — weights computed in `f64`, stored as
//! `f32`, accumulated per channel in `f32`, rounded half-to-even — so the
//! resized pixels are bit-identical, not merely close.

use super::{Rgb8Image, FACTOR};

/// An image size in pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Size {
    pub height: u32,
    pub width: u32,
}

/// Why [`smart_resize`] refused an image.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResizeError {
    /// The longer side is more than 200 times the shorter one.
    AspectRatio,
    /// A zero side, or pixel bounds with `min > max` or `min == 0`.
    InvalidConfiguration,
}

/// `std::nearbyint` under the default rounding mode, cast to `int`.
fn round_even(value: f64) -> i64 {
    value.round_ties_even() as i64
}

/// The reference's `smart_resize_image`: round each side to the 32-pixel
/// merge factor, then rescale into `[min_pixels, max_pixels]` keeping the
/// aspect ratio.
pub fn smart_resize(height: u32, width: u32, min_pixels: u64, max_pixels: u64) -> Result<Size, ResizeError> {
    if height == 0 || width == 0 || min_pixels == 0 || max_pixels < min_pixels {
        return Err(ResizeError::InvalidConfiguration);
    }
    let (height, width) = (height as i64, width as i64);
    if height.max(width) as f64 / height.min(width) as f64 > 200.0 {
        return Err(ResizeError::AspectRatio);
    }
    let factor = FACTOR as i64;
    let mut h = round_even(height as f64 / factor as f64) * factor;
    let mut w = round_even(width as f64 / factor as f64) * factor;
    let area = (h.max(0) as u64) * (w.max(0) as u64);
    if area > max_pixels {
        let beta = (height as f64 * width as f64 / max_pixels as f64).sqrt();
        h = factor.max((height as f64 / beta / factor as f64).floor() as i64 * factor);
        w = factor.max((width as f64 / beta / factor as f64).floor() as i64 * factor);
    } else if area < min_pixels {
        let beta = (min_pixels as f64 / (height as f64 * width as f64)).sqrt();
        h = (height as f64 * beta / factor as f64).ceil() as i64 * factor;
        w = (width as f64 * beta / factor as f64).ceil() as i64 * factor;
    }
    Ok(Size { height: h as u32, width: w as u32 })
}

/// Keys cubic convolution, a = -0.5 (torchvision's antialiased bicubic).
fn cubic(x: f64) -> f64 {
    const A: f64 = -0.5;
    let x = x.abs();
    if x < 1.0 {
        ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        ((A * x - 5.0 * A) * x + 8.0 * A) * x - 4.0 * A
    } else {
        0.0
    }
}

/// Per output sample: the first contributing input index, and the range of
/// its normalized weights in `weights`.
struct Coefficients {
    starts: Vec<usize>,
    offsets: Vec<usize>,
    weights: Vec<f32>,
}

fn coefficients(input: usize, output: usize) -> Coefficients {
    let mut out = Coefficients {
        starts: vec![0; output],
        offsets: vec![0; output + 1],
        weights: Vec::new(),
    };
    let scale = input as f64 / output as f64;
    let invscale = if scale >= 1.0 { 1.0 / scale } else { 1.0 };
    let support = 2.0 * if scale >= 1.0 { scale } else { 1.0 };
    for dst in 0..output {
        let center = scale * (dst as f64 + 0.5);
        // `static_cast<int>` truncates toward zero, as `as` does.
        let begin = ((center - support + 0.5) as i64).max(0);
        let size = ((center + support + 0.5) as i64).min(input as i64) - begin;
        out.starts[dst] = begin as usize;
        out.offsets[dst] = out.weights.len();
        let mut sum = 0.0f64;
        for j in 0..size {
            let weight = cubic(((j + begin) as f64 - center + 0.5) * invscale);
            out.weights.push(weight as f32);
            sum += weight;
        }
        assert!(sum != 0.0, "bicubic resize produced zero weights");
        let first = out.offsets[dst];
        for weight in &mut out.weights[first..] {
            *weight = (*weight as f64 / sum) as f32;
        }
    }
    out.offsets[output] = out.weights.len();
    out
}

/// `std::clamp(round_even(value), 0, 255)` of one accumulated channel.
fn to_u8(value: f32) -> u8 {
    round_even(value as f64).clamp(0, 255) as u8
}

/// Separable antialiased bicubic resize: horizontal pass into a u8
/// intermediate, then vertical. An image already at `size` is returned
/// untouched.
pub fn resize_bicubic(input: Rgb8Image, size: Size) -> Rgb8Image {
    if input.width == size.width && input.height == size.height {
        return input;
    }
    let (in_w, in_h) = (input.width as usize, input.height as usize);
    let (out_w, out_h) = (size.width as usize, size.height as usize);
    let horizontal = coefficients(in_w, out_w);
    let vertical = coefficients(in_h, out_h);

    let mut temp = vec![0u8; in_h * out_w * 3];
    for y in 0..in_h {
        for x in 0..out_w {
            let (first, last) = (horizontal.offsets[x], horizontal.offsets[x + 1]);
            let mut source = (y * in_w + horizontal.starts[x]) * 3;
            let mut value = [0.0f32; 3];
            for &weight in &horizontal.weights[first..last] {
                value[0] += weight * input.rgb[source] as f32;
                value[1] += weight * input.rgb[source + 1] as f32;
                value[2] += weight * input.rgb[source + 2] as f32;
                source += 3;
            }
            let destination = (y * out_w + x) * 3;
            for c in 0..3 {
                temp[destination + c] = to_u8(value[c]);
            }
        }
    }

    let mut rgb = vec![0u8; out_h * out_w * 3];
    for y in 0..out_h {
        let (first, last) = (vertical.offsets[y], vertical.offsets[y + 1]);
        for x in 0..out_w {
            let mut source = (vertical.starts[y] * out_w + x) * 3;
            let mut value = [0.0f32; 3];
            for &weight in &vertical.weights[first..last] {
                value[0] += weight * temp[source] as f32;
                value[1] += weight * temp[source + 1] as f32;
                value[2] += weight * temp[source + 2] as f32;
                source += out_w * 3;
            }
            let destination = (y * out_w + x) * 3;
            for c in 0..3 {
                rgb[destination + c] = to_u8(value[c]);
            }
        }
    }
    Rgb8Image { width: size.width, height: size.height, rgb }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIN: u64 = 65_536;
    const MAX: u64 = 16_777_216;

    #[test]
    fn an_on_grid_size_inside_the_bounds_is_kept() {
        assert_eq!(smart_resize(480, 640, MIN, MAX), Ok(Size { height: 480, width: 640 }));
    }

    #[test]
    fn sides_round_half_to_even_on_the_factor() {
        // 1000/32 = 31.25 rounds to 31; 1008/32 = 31.5 is a tie and goes to
        // the even 32.
        assert_eq!(smart_resize(1000, 1008, 1, MAX), Ok(Size { height: 992, width: 1024 }));
    }

    #[test]
    fn a_small_image_scales_up_to_min_pixels() {
        let size = smart_resize(60, 100, MIN, MAX).unwrap();
        assert_eq!(size, Size { height: 224, width: 352 });
        assert!(size.height as u64 * size.width as u64 >= MIN);
    }

    #[test]
    fn a_large_image_scales_down_below_max_pixels() {
        let size = smart_resize(3600, 4800, MIN, MAX).unwrap();
        assert_eq!(size, Size { height: 3520, width: 4704 });
        assert!(size.height as u64 * size.width as u64 <= MAX);
    }

    #[test]
    fn aspect_ratio_above_200_is_refused_and_200_is_not() {
        assert_eq!(smart_resize(20, 4020, MIN, MAX), Err(ResizeError::AspectRatio));
        assert!(smart_resize(20, 4000, MIN, MAX).is_ok());
    }

    #[test]
    fn invalid_configuration_is_refused() {
        assert_eq!(smart_resize(0, 10, MIN, MAX), Err(ResizeError::InvalidConfiguration));
        assert_eq!(smart_resize(10, 10, MAX, MIN), Err(ResizeError::InvalidConfiguration));
    }

    #[test]
    fn weights_of_every_output_sample_sum_to_one() {
        for (input, output) in [(100, 352), (4800, 4704), (7, 3), (1, 32)] {
            let c = coefficients(input, output);
            for dst in 0..output {
                let sum: f64 = c.weights[c.offsets[dst]..c.offsets[dst + 1]].iter().map(|&w| w as f64).sum();
                assert!((sum - 1.0).abs() < 1e-5, "{input}->{output} dst {dst}: {sum}");
            }
        }
    }

    #[test]
    fn a_flat_image_stays_flat_through_any_resize() {
        let flat = Rgb8Image { width: 37, height: 11, rgb: [9u8, 128, 250].repeat(37 * 11) };
        let out = resize_bicubic(flat, Size { height: 64, width: 32 });
        assert!(out.rgb.chunks(3).all(|px| px == [9, 128, 250]));
    }

    #[test]
    fn an_image_already_at_size_is_returned_untouched() {
        let rgb: Vec<u8> = (0..32 * 32 * 3).map(|i| (i * 7 % 256) as u8).collect();
        let image = Rgb8Image { width: 32, height: 32, rgb: rgb.clone() };
        assert_eq!(resize_bicubic(image, Size { height: 32, width: 32 }).rgb, rgb);
    }
}
