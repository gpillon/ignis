//! Integration smoke test against the real Qwen 3.8-27B `nvfp4full` v2
//! artifact (the 1,325-object container). Skips gracefully when the file is
//! not at its machine-local path.

use ignis_artifact::{NumericFormat, Object, Reader};

/// The fork-local model cache (see `docs/agents/issue-tracker.md` conventions;
/// the artifact is the one the running `ninfer-serve` loads).
const ARTIFACT: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";

#[test]
fn real_nvfp4full_inventory() {
    let path = std::path::Path::new(ARTIFACT);
    if !path.exists() {
        eprintln!("skip: {ARTIFACT} not present");
        return;
    }

    let reader = Reader::open(path).unwrap_or_else(|e| panic!("open {ARTIFACT}: {e}"));

    // 1,319 tensors + 6 frontend resources = 1,325 objects
    // (qwen3.8-27b artifact reference).
    let objects = reader.objects();
    let tensors = objects.iter().filter(|o| matches!(o, Object::Tensor(_))).count();
    let resources = objects.iter().filter(|o| matches!(o, Object::Resource(_))).count();
    assert_eq!(objects.len(), 1325, "expected 1325 objects");
    assert_eq!(tensors, 1319);
    assert_eq!(resources, 6);

    assert_eq!(reader.identity().model_id, "qwen3.8-27b");

    // The 6 frontend resources carry the tokenizer + chat template (ADR 0002:
    // the engine reads them straight from the container).
    let resource_names: Vec<String> = objects
        .iter()
        .filter_map(|o| match o {
            Object::Resource(r) => Some(r.name.clone()),
            _ => None,
        })
        .collect();
    for expected in ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja"] {
        assert!(
            resource_names.iter().any(|n| n.contains(expected)),
            "expected a resource containing '{expected}'; got {resource_names:?}"
        );
    }

    // NVFP4 tensors must use the blockscale layout.
    let nvfp4_tensors = objects.iter().filter_map(|o| match o {
        Object::Tensor(t) => Some(t),
        _ => None,
    });
    let nvfp4_count = nvfp4_tensors
        .filter(|t| t.format == NumericFormat::Nvfp4)
        .count();
    assert!(nvfp4_count > 0, "expected NVFP4 tensors");
    for o in objects.iter() {
        if let Object::Tensor(t) = o
            && t.format == NumericFormat::Nvfp4 {
                assert_eq!(
                    t.layout,
                    ignis_artifact::StorageLayout::BlockScaleK16M128x4V1,
                    "NVFP4 tensor {} must use blockscale layout",
                    t.name
                );
            }
    }

    // read_direct sanity: an aligned 4096-byte direct read must equal the
    // mapping view (exercises the NO_BUFFERING / O_DIRECT path).
    let payload_offset = reader.payload_offset();
    if let Some(first) = objects.first()
        && first.offset() % 4096 == 0 && first.bytes() >= 4096 {
            let mut buf = vec![0u8; 8192];
            let ptr = buf.as_mut_ptr() as usize;
            let aligned_base = ptr.div_ceil(4096) * 4096;
            let off = aligned_base - ptr;
            let dst = &mut buf[off..off + 4096];
            let abs = payload_offset + first.offset();
            let n = reader.read_direct(abs, dst).expect("direct read");
            assert_eq!(n, 4096);
            assert_eq!(
                dst,
                &reader.mapped_bytes()[abs as usize..(abs + 4096) as usize]
            );
        }
}