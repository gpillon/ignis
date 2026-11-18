//! Media bytes to RGB8 (`media/decode/decode.cpp` `decode_image`).
//!
//! The reference decodes every image through FFmpeg and converts the first
//! frame to RGB24 with swscale (`SWS_POINT`, alpha dropped, not composited),
//! then applies the JPEG EXIF orientation. The lossless formats are decoded
//! here with pure-Rust decoders and converted with FFmpeg's semantics, so
//! their pixels are exact:
//!
//! - PNG: palette and tRNS expanded, 16-bit samples keep their high byte
//!   (swscale's `rgb48 -> rgb24`), alpha dropped, gray replicated.
//! - GIF: the first frame composited the way `gifdec.c` does — the canvas
//!   filled with the background colour, or with transparent white when the
//!   frame declares a transparent index, then the frame's opaque pixels.
//! - WebP: lossless (`VP8L`) exact; lossy (`VP8`) goes through
//!   `image-webp`'s own YUV conversion, which is not FFmpeg's (a known
//!   divergence, GitHub #176's report).
//! - JPEG: [`super::jpeg`].
//!
//! Each decoder checks the frame size against the pixel limit as soon as
//! the header gives it, before allocating the frame.
//!
//! [`decode_image_rgba`] is the image model's (spec image-model/05): its
//! condition images keep their alpha, so PNG and WebP are decoded with it
//! and a JPEG is opaque.

use std::io::Cursor;

use super::{jpeg, InvalidMedia, Rgb8Image, Rgba8Image};

/// Why [`decode_image`] produced no image.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecodeFailure {
    /// The bytes are not a usable image.
    Invalid(InvalidMedia),
    /// The frame holds this many pixels, more than the limit.
    Pixels(u64),
}

/// Decode `bytes` to RGB8, picking the decoder from the magic bytes. A
/// frame larger than `max_pixels` is refused before it is decoded.
pub fn decode_image(bytes: &[u8], max_pixels: u64) -> Result<Rgb8Image, DecodeFailure> {
    if bytes.is_empty() {
        return Err(undecodable("media bytes are empty"));
    }
    if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        decode_png(bytes, max_pixels)
    } else if bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a") {
        decode_gif(bytes, max_pixels)
    } else if bytes.len() >= 12 && &bytes[..4] == b"RIFF" && &bytes[8..12] == b"WEBP" {
        decode_webp(bytes, max_pixels)
    } else if bytes.starts_with(&[0xff, 0xd8]) {
        jpeg::decode(bytes, max_pixels)
    } else {
        Err(DecodeFailure::Invalid(InvalidMedia::Unsupported(
            "unrecognised image format (PNG, JPEG, WebP and GIF are accepted)".to_owned(),
        )))
    }
}

/// The `(width, height)` [`decode_image`] gives `bytes`, read from the
/// format's header without decoding a pixel (GitHub #673): what a request
/// weighs an image by before deciding whether to prepare it. The same magic
/// bytes pick the format and the same frame-size limit applies; a failure the
/// decoder finds past the header is not seen.
pub fn image_dimensions(bytes: &[u8], max_pixels: u64) -> Result<(u32, u32), DecodeFailure> {
    let (width, height) = if bytes.is_empty() {
        return Err(undecodable("media bytes are empty"));
    } else if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        let mut decoder = png::Decoder::new(Cursor::new(bytes));
        let header = decoder.read_header_info().map_err(undecodable)?;
        (header.width, header.height)
    } else if bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a") {
        // `decode_gif` draws the first frame on the logical screen.
        let decoder = gif::DecodeOptions::new().read_info(Cursor::new(bytes)).map_err(undecodable)?;
        (u32::from(decoder.width()), u32::from(decoder.height()))
    } else if bytes.len() >= 12 && &bytes[..4] == b"RIFF" && &bytes[8..12] == b"WEBP" {
        image_webp::WebPDecoder::new(Cursor::new(bytes)).map_err(undecodable)?.dimensions()
    } else if bytes.starts_with(&[0xff, 0xd8]) {
        return jpeg::dimensions(bytes, max_pixels);
    } else {
        return Err(DecodeFailure::Invalid(InvalidMedia::Unsupported(
            "unrecognised image format (PNG, JPEG, WebP and GIF are accepted)".to_owned(),
        )));
    };
    check_pixels(width, height, max_pixels)?;
    Ok((width, height))
}

/// Decode `bytes` to RGBA8, picking the decoder from the magic bytes: PNG
/// (palette and tRNS expanded, 16-bit samples keep their high byte, gray
/// replicated) and WebP with their alpha, JPEG opaque with its EXIF
/// orientation applied. GIF is refused: the image model takes PNG, JPEG and
/// WebP. A frame larger than `max_pixels` is refused before it is decoded.
pub fn decode_image_rgba(bytes: &[u8], max_pixels: u64) -> Result<Rgba8Image, DecodeFailure> {
    let opaque = |image: Rgb8Image| Rgba8Image {
        width: image.width,
        height: image.height,
        rgba: image.rgb.chunks_exact(3).flat_map(|px| [px[0], px[1], px[2], 0xff]).collect(),
    };
    if bytes.is_empty() {
        return Err(undecodable("media bytes are empty"));
    }
    if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        decode_png_rgba(bytes, max_pixels)
    } else if bytes.len() >= 12 && &bytes[..4] == b"RIFF" && &bytes[8..12] == b"WEBP" {
        decode_webp_rgba(bytes, max_pixels)
    } else if bytes.starts_with(&[0xff, 0xd8]) {
        jpeg::decode(bytes, max_pixels).map(opaque)
    } else {
        Err(DecodeFailure::Invalid(InvalidMedia::Unsupported(
            "unrecognised image format (PNG, JPEG and WebP are accepted)".to_owned(),
        )))
    }
}

fn undecodable(error: impl std::fmt::Display) -> DecodeFailure {
    DecodeFailure::Invalid(InvalidMedia::Undecodable(error.to_string()))
}

fn check_pixels(width: u32, height: u32, max_pixels: u64) -> Result<(), DecodeFailure> {
    let pixels = width as u64 * height as u64;
    if pixels > max_pixels {
        return Err(DecodeFailure::Pixels(pixels));
    }
    Ok(())
}

fn decode_png(bytes: &[u8], max_pixels: u64) -> Result<Rgb8Image, DecodeFailure> {
    let mut decoder = png::Decoder::new(Cursor::new(bytes));
    decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    let header = decoder.read_header_info().map_err(undecodable)?;
    check_pixels(header.width, header.height, max_pixels)?;
    let mut reader = decoder.read_info().map_err(undecodable)?;
    let size = reader.output_buffer_size().ok_or_else(|| undecodable("PNG frame is too large"))?;
    let mut buffer = vec![0u8; size];
    let info = reader.next_frame(&mut buffer).map_err(undecodable)?;
    let (width, height) = (info.width, info.height);
    let channels = info.color_type.samples();
    let mut rgb = Vec::with_capacity(width as usize * height as usize * 3);
    for px in buffer[..info.buffer_size()].chunks_exact(channels) {
        match channels {
            1 | 2 => rgb.extend_from_slice(&[px[0], px[0], px[0]]),
            _ => rgb.extend_from_slice(&px[..3]),
        }
    }
    Ok(Rgb8Image { width, height, rgb })
}

fn decode_png_rgba(bytes: &[u8], max_pixels: u64) -> Result<Rgba8Image, DecodeFailure> {
    let mut decoder = png::Decoder::new(Cursor::new(bytes));
    decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    let header = decoder.read_header_info().map_err(undecodable)?;
    check_pixels(header.width, header.height, max_pixels)?;
    let mut reader = decoder.read_info().map_err(undecodable)?;
    let size = reader.output_buffer_size().ok_or_else(|| undecodable("PNG frame is too large"))?;
    let mut buffer = vec![0u8; size];
    let info = reader.next_frame(&mut buffer).map_err(undecodable)?;
    let channels = info.color_type.samples();
    let mut rgba = Vec::with_capacity(info.width as usize * info.height as usize * 4);
    for px in buffer[..info.buffer_size()].chunks_exact(channels) {
        match channels {
            1 => rgba.extend_from_slice(&[px[0], px[0], px[0], 0xff]),
            2 => rgba.extend_from_slice(&[px[0], px[0], px[0], px[1]]),
            3 => rgba.extend_from_slice(&[px[0], px[1], px[2], 0xff]),
            _ => rgba.extend_from_slice(&px[..4]),
        }
    }
    Ok(Rgba8Image { width: info.width, height: info.height, rgba })
}

fn decode_gif(bytes: &[u8], max_pixels: u64) -> Result<Rgb8Image, DecodeFailure> {
    let mut options = gif::DecodeOptions::new();
    options.set_color_output(gif::ColorOutput::Indexed);
    let mut decoder = options.read_info(Cursor::new(bytes)).map_err(undecodable)?;
    let (screen_w, screen_h) = (decoder.width() as usize, decoder.height() as usize);
    check_pixels(screen_w as u32, screen_h as u32, max_pixels)?;
    let global = decoder.global_palette().map(<[u8]>::to_vec);
    let background = decoder.bg_color();
    let frame = decoder
        .read_next_frame()
        .map_err(undecodable)?
        .ok_or_else(|| undecodable("image contains no decoded frame"))?;
    let palette = frame
        .palette
        .as_deref()
        .or(global.as_deref())
        .ok_or_else(|| undecodable("picture doesn't have either global or local palette"))?;
    let color = |palette: &[u8], index: usize| -> [u8; 3] {
        palette.get(index * 3..index * 3 + 3).map_or([0, 0, 0], |c| [c[0], c[1], c[2]])
    };
    // gifdec.c: a keyframe starts from the background colour when no
    // transparency was declared and a global palette exists, otherwise from
    // its transparent colour (0x00ffffff: white once alpha is dropped).
    let fill = match (frame.transparent, &global) {
        (None, Some(global)) => color(global, background.unwrap_or(0)),
        _ => [0xff, 0xff, 0xff],
    };
    let mut rgb = fill.repeat(screen_w * screen_h);
    let (left, top) = (frame.left as usize, frame.top as usize);
    let (width, height) = (frame.width as usize, frame.height as usize);
    for y in 0..height.min(screen_h.saturating_sub(top)) {
        for x in 0..width.min(screen_w.saturating_sub(left)) {
            let index = frame.buffer[y * width + x];
            if Some(index) != frame.transparent {
                let at = ((top + y) * screen_w + left + x) * 3;
                rgb[at..at + 3].copy_from_slice(&color(palette, index as usize));
            }
        }
    }
    Ok(Rgb8Image { width: screen_w as u32, height: screen_h as u32, rgb })
}

fn decode_webp(bytes: &[u8], max_pixels: u64) -> Result<Rgb8Image, DecodeFailure> {
    let mut decoder = image_webp::WebPDecoder::new(Cursor::new(bytes)).map_err(undecodable)?;
    let (width, height) = decoder.dimensions();
    check_pixels(width, height, max_pixels)?;
    let size = decoder.output_buffer_size().ok_or_else(|| undecodable("WebP frame is too large"))?;
    let mut buffer = vec![0u8; size];
    decoder.read_image(&mut buffer).map_err(undecodable)?;
    let rgb = if decoder.has_alpha() {
        buffer.chunks_exact(4).flat_map(|px| [px[0], px[1], px[2]]).collect()
    } else {
        buffer
    };
    Ok(Rgb8Image { width, height, rgb })
}

fn decode_webp_rgba(bytes: &[u8], max_pixels: u64) -> Result<Rgba8Image, DecodeFailure> {
    let mut decoder = image_webp::WebPDecoder::new(Cursor::new(bytes)).map_err(undecodable)?;
    let (width, height) = decoder.dimensions();
    check_pixels(width, height, max_pixels)?;
    let size = decoder.output_buffer_size().ok_or_else(|| undecodable("WebP frame is too large"))?;
    let mut buffer = vec![0u8; size];
    decoder.read_image(&mut buffer).map_err(undecodable)?;
    let rgba = if decoder.has_alpha() {
        buffer
    } else {
        buffer.chunks_exact(3).flat_map(|px| [px[0], px[1], px[2], 0xff]).collect()
    };
    Ok(Rgba8Image { width, height, rgba })
}

#[cfg(test)]
mod tests {
    use super::*;

    const LIMIT: u64 = 1 << 26;

    #[test]
    fn empty_and_unknown_bytes_are_typed_errors() {
        assert!(matches!(decode_image(&[], LIMIT), Err(DecodeFailure::Invalid(InvalidMedia::Undecodable(_)))));
        assert!(matches!(
            decode_image(b"BM\0\0 a bitmap", LIMIT),
            Err(DecodeFailure::Invalid(InvalidMedia::Unsupported(_)))
        ));
    }

    #[test]
    fn a_truncated_png_is_undecodable() {
        let mut bytes = b"\x89PNG\r\n\x1a\n".to_vec();
        bytes.extend((0..=255u8).cycle().take(1024));
        assert!(matches!(decode_image(&bytes, LIMIT), Err(DecodeFailure::Invalid(InvalidMedia::Undecodable(_)))));
    }

    /// The image model's decode keeps a PNG's alpha (the values Pillow reads
    /// from the fixture), its colour is the RGB decode's, and an opaque
    /// format is alpha 255; GIF is refused.
    #[test]
    fn the_rgba_decode_keeps_the_alpha_and_makes_the_rest_opaque() {
        let png = include_bytes!("../../tests/fixtures/vision/images/alpha.png");
        let rgba = decode_image_rgba(png, LIMIT).unwrap();
        assert_eq!((rgba.width, rgba.height), (300, 200));
        let alpha: Vec<u8> = rgba.rgba.chunks_exact(4).map(|px| px[3]).collect();
        assert_eq!(alpha.iter().map(|&a| u64::from(a)).sum::<u64>(), 7_620_200);
        assert_eq!(alpha.iter().filter(|&&a| a == 0).count(), 400);
        assert_eq!(&rgba.rgba[..4], &[0, 0, 211, 0]);
        let at = |x: usize, y: usize| &rgba.rgba[(y * 300 + x) * 4..][..4];
        assert_eq!(at(150, 100), &[128, 128, 248, 127]);
        let rgb = decode_image(png, LIMIT).unwrap();
        assert!(rgba.rgba.chunks_exact(4).zip(rgb.rgb.chunks_exact(3)).all(|(a, b)| a[..3] == *b));
        for opaque in [
            &include_bytes!("../../tests/fixtures/vision/images/lossless.webp")[..],
            &include_bytes!("../../tests/fixtures/vision/images/odd_width.jpg")[..],
            &include_bytes!("../../tests/fixtures/vision/images/on_grid.png")[..],
        ] {
            let rgba = decode_image_rgba(opaque, LIMIT).unwrap();
            let rgb = decode_image(opaque, LIMIT).unwrap();
            assert_eq!((rgba.width, rgba.height), (rgb.width, rgb.height));
            assert!(rgba.rgba.chunks_exact(4).zip(rgb.rgb.chunks_exact(3)).all(|(a, b)| a[..3] == *b && a[3] == 0xff));
        }
        let gif = include_bytes!("../../tests/fixtures/vision/images/animated.gif");
        assert!(matches!(decode_image_rgba(gif, LIMIT), Err(DecodeFailure::Invalid(InvalidMedia::Unsupported(_)))));
        assert_eq!(decode_image_rgba(png, 59_999), Err(DecodeFailure::Pixels(60_000)));
    }

    #[test]
    fn a_frame_over_the_pixel_limit_is_refused_before_decoding() {
        // The committed 4800x3600 fixture.
        let png = include_bytes!("../../tests/fixtures/vision/images/downscale.png");
        assert_eq!(decode_image(png, 17_279_999), Err(DecodeFailure::Pixels(17_280_000)));
        assert!(decode_image(png, 17_280_000).is_ok());
    }
}
