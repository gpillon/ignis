//! The n-gram table of the real Flash-Next artifact, host side (spec
//! flash-next/04 stories 21-22, slice S4 of GitHub #302). Machine-local: it
//! skips with a note while `F:/ai/models/Qwen3.8-Flash-Next-ignis/` holds no
//! packed artifact (the conversion runs ~11 h), as the artifact crate's
//! real-artifact tests do. CPU only: it reads the file, never the GPU.
//!
//! It opens the table the way a load does -- the host-streamed range the
//! binder hands over, the stored hash buffers, the hot list within the 1 GiB
//! default -- checks the stored buffers are the ones the checkpoint's config
//! derives, stages a prompt, and compares every row with the file's bytes.

use std::path::Path;
use std::time::Instant;

use ignis_artifact::flash_next::{self, FlashNextGeometry};
use ignis_artifact::packer::ARTIFACT_FILE_NAME;
use ignis_artifact::Reader;
use ignis_core::compute::ModelConfig;
use ignis_core::ngram::NgramHashBuffers;
use ignis_core::ngram_table::{NgramTable, NgramTableOptions};

const MODEL_DIR: &str = "F:/ai/models/Qwen3.8-Flash-Next-ignis";

#[test]
fn the_real_table_stages_a_prompts_rows_from_cache_and_nvme() {
    let path = Path::new(MODEL_DIR).join(ARTIFACT_FILE_NAME);
    if !path.exists() {
        eprintln!("skip: {} does not exist (packed after the conversion)", path.display());
        return;
    }
    let reader = Reader::open(&path).unwrap();
    let bound = flash_next::bind(&reader, &FlashNextGeometry::qwen38_flash_next()).unwrap();
    let config = ModelConfig::qwen38_flash_next();
    let geometry = config.ngram.expect("Flash-Next has an n-gram embedding");

    let opened = Instant::now();
    let table = NgramTable::from_artifact(&path, &reader, &bound, geometry, NgramTableOptions::default()).unwrap();
    eprintln!(
        "opened in {:.1} s: {} hot rows ({} bytes), unbuffered {}",
        opened.elapsed().as_secs_f64(),
        table.hot_rows(),
        table.hot_bytes(),
        table.is_unbuffered()
    );
    assert!(table.hot_bytes() <= 1 << 30, "the hot rows fit the 1 GiB default");
    assert!(table.hot_rows() > 0);

    // The stored hash buffers are what the checkpoint's config derives.
    let prefix = format!("layers.{}.ple.ple_embedding", geometry.layer);
    let words = |name: &str| -> Vec<i64> {
        reader.payload(&format!("{prefix}.{name}")).unwrap().data.chunks_exact(8)
            .map(|w| i64::from_le_bytes(w.try_into().unwrap()))
            .collect()
    };
    let derived = NgramHashBuffers::derive(&geometry, config.vocab);
    assert_eq!(words("layer_multipliers"), derived.layer_multipliers);
    assert_eq!(words("ngram_heads_vocab_sizes"), derived.head_vocab_sizes);
    assert_eq!(words("ngram_heads_offsets"), derived.head_offsets);

    // A prompt's rows, staged, are the file's rows for the hashed ids.
    let tokens: Vec<u32> = (0..512u32).map(|i| (i * 7_919 + 13) % 248_000).collect();
    let staged_at = Instant::now();
    let mut staged = vec![0u8; tokens.len() * table.token_bytes()];
    table.stage(&mut table.new_context(), &tokens, &mut staged).unwrap();
    let elapsed = staged_at.elapsed();
    let mut ids = Vec::new();
    table.hasher().hash(&mut table.new_context(), &tokens, &mut ids);
    let rows = reader.payload(&flash_next::ngram_table_name(&FlashNextGeometry::qwen38_flash_next())).unwrap().data;
    let row_bytes = table.row_bytes();
    for (i, &id) in ids.iter().enumerate() {
        let id = id as usize;
        assert_eq!(&staged[i * row_bytes..(i + 1) * row_bytes], &rows[id * row_bytes..(id + 1) * row_bytes], "row {id}");
    }
    let counters = table.counters();
    eprintln!(
        "{} rows in {:.2} ms: {} from the cache, {} reads of {} bytes",
        counters.rows,
        elapsed.as_secs_f64() * 1e3,
        counters.hot_rows,
        counters.reads,
        counters.read_bytes
    );
}
