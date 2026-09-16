//! Shared media fixtures (GitHub #179): flat test images, base64 data URIs,
//! the real vision processor over a word-level tokenizer, and a gated
//! preparer a test can hold, count and cancel.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use ignis_artifact::vision::{
    InvalidMedia, PreparedMedia, ProcessorError, ProcessorOptions, VisionProcessor, IMAGE_PAD, IMAGE_PAD_ID,
    VIDEO_PAD, VIDEO_PAD_ID,
};
use ignis_artifact::Tokenizer;
use ignis_server::media::Preparer;

/// A flat RGB PNG.
pub fn png(width: u32, height: u32) -> Vec<u8> {
    let mut out = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut out, width, height);
        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header().unwrap();
        writer.write_image_data(&vec![100u8; (width * height * 3) as usize]).unwrap();
    }
    out
}

/// `bytes` as a base64 PNG data URI.
pub fn data_uri(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = "data:image/png;base64,".to_owned();
    for chunk in bytes.chunks(3) {
        let n = chunk.iter().enumerate().fold(0u32, |n, (i, &b)| n | (b as u32) << (16 - 8 * i));
        for i in 0..4 {
            out.push(if i <= chunk.len() { ALPHABET[(n >> (18 - 6 * i) & 63) as usize] as char } else { '=' });
        }
    }
    out
}

/// The real processor under `limits`, over a word-level tokenizer carrying
/// the model contract's placeholder ids.
pub fn processor(limits: ProcessorOptions) -> VisionProcessor {
    let added = |id: u32, content: &str| {
        serde_json::json!({"id": id, "content": content, "single_word": false, "lstrip": false,
                           "rstrip": false, "normalized": false, "special": true})
    };
    let json = serde_json::json!({
        "version": "1.0",
        "added_tokens": [added(IMAGE_PAD_ID, IMAGE_PAD), added(VIDEO_PAD_ID, VIDEO_PAD)],
        "pre_tokenizer": {"type": "Whitespace"},
        "model": {"type": "WordLevel", "vocab": {"x": 0, IMAGE_PAD: IMAGE_PAD_ID, VIDEO_PAD: VIDEO_PAD_ID}, "unk_token": "x"},
    });
    let tokenizer = Tokenizer::from_bytes(json.to_string().as_bytes()).unwrap();
    VisionProcessor::new(&tokenizer, limits).unwrap()
}

/// Poll `condition` until it holds (bounded, so a broken build fails the
/// test instead of hanging it).
pub async fn until(what: &str, condition: impl Fn() -> bool) {
    let start = Instant::now();
    while !condition() {
        assert!(start.elapsed() < Duration::from_secs(10), "timed out waiting for {what}");
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
}

/// A processor behind a gate: counts builds and, while closed, holds each
/// one until nobody wants it — which it records.
pub struct Gated {
    inner: VisionProcessor,
    pub open: AtomicBool,
    pub builds: AtomicUsize,
    pub observed_cancel: AtomicBool,
}

impl Gated {
    pub fn new(open: bool, inner: VisionProcessor) -> Arc<Self> {
        Arc::new(Self {
            inner,
            open: AtomicBool::new(open),
            builds: AtomicUsize::new(0),
            observed_cancel: AtomicBool::new(false),
        })
    }
}

impl Preparer for Gated {
    fn prepare_media(&self, item: usize, bytes: &[u8], cancelled: &dyn Fn() -> bool) -> Result<PreparedMedia, ProcessorError> {
        self.builds.fetch_add(1, Ordering::SeqCst);
        loop {
            if self.open.load(Ordering::SeqCst) {
                return self.inner.prepare_media(item, bytes);
            }
            if cancelled() {
                self.observed_cancel.store(true, Ordering::SeqCst);
                return Err(ProcessorError::InvalidMedia { item, reason: InvalidMedia::Undecodable("cancelled".to_owned()) });
            }
            std::thread::sleep(Duration::from_millis(1));
        }
    }
}
