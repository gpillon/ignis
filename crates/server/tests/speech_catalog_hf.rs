//! The speech catalog entries against the real Hugging Face (spec speech/01
//! AC 22, GitHub #351): the pins in `catalog.yaml` are the repos' files.
//!
//! Needs the network and downloads the smallest entry (about 1 GB), so it is
//! `--ignored`: `cargo test -p ignis-server --test speech_catalog_hf --
//! --ignored --nocapture`. `fetch` refuses a file whose size or SHA-256 is not
//! the pin's, so a pass is the check. CI runs it in the release workflow's
//! `network` job (GitHub #709).

use ignis_server::download::{Catalog, Downloader};

/// The GPU profile's serialized pass runs every `--ignored` test it does not
/// skip; this leg is a download, so `scripts/gpu-profile.ps1` skips it by name.
#[test]
fn the_gpu_profile_skips_this_leg() {
    let script = include_str!("../../../scripts/gpu-profile.ps1");
    let listed = script.split("$NotGpuLegs = @(").nth(1).and_then(|rest| rest.split(')').next()).expect("gpu-profile.ps1 lists $NotGpuLegs");
    assert!(listed.contains("\"the_smallest_speech_entry_downloads_and_verifies_against_its_pins\""), "{listed}");
}

#[tokio::test]
#[ignore = "downloads about 1 GB from huggingface.co"]
async fn the_smallest_speech_entry_downloads_and_verifies_against_its_pins() {
    let entry = Catalog::built_in().entry("qwen3-asr-0.6b").expect("built in");
    let dir = std::env::temp_dir().join(format!("ignis-speech-catalog-hf-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let downloader = Downloader::new("https://huggingface.co", None).expect("downloader");
    let decoder = downloader.fetch(entry, &dir).await.expect("every file matches its pin");
    assert_eq!(decoder, dir.join("Qwen3-ASR-0.6B-Q8_0.gguf"));
    for file in &entry.files {
        let len = std::fs::metadata(dir.join(&file.name)).unwrap_or_else(|e| panic!("{}: {e}", file.name)).len();
        assert_eq!(len, file.bytes, "{}", file.name);
    }
    let _ = std::fs::remove_dir_all(&dir);
}
