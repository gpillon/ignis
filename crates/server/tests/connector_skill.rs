//! The connector's skill (spec #656, ticket #662): `GET /v1/connector/skill`
//! serves the `ignis` skill packed at build, and the packer that makes it.
//!
//! The archive is read back with a tar reader written here, apart from the
//! packer's writer, and compared with the skill directory read apart from the
//! packer's walk, so a mistake in either shows as a difference.

use std::collections::BTreeMap;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use axum::body::Body;
use axum::http::{header, Request, StatusCode};
use http_body_util::BodyExt;
use ignis_core::{ConcreteScheduler, MockCompute, SchedulerConfig};
use ignis_server::config::schema::ConnectorGroup;
use ignis_server::config::ApiKey;
use ignis_server::connector::skill::ARCHIVE;
use ignis_server::engine::Engine;
use ignis_server::template::SimpleTemplateProvider;
use ignis_server::Server;
use tower::ServiceExt;

#[allow(dead_code)]
#[path = "../skill_pack.rs"]
mod skill_pack;

const KEY: &str = "sk-skill-test";

/// The skill directory in the repo, `connector/ignis`.
fn skill_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../connector").join("ignis")
}

fn server() -> Server {
    let scheduler = ConcreteScheduler::with_config(
        SchedulerConfig { model: "test-model".into(), ..SchedulerConfig::default() },
        Arc::new(MockCompute::new()),
    );
    Server::new(Engine::new(Box::new(scheduler)), Box::new(SimpleTemplateProvider))
}

/// One entry of a tar: its header's fields and its bytes.
#[derive(Debug)]
struct Entry {
    path: String,
    kind: u8,
    mode: u32,
    uid: u32,
    gid: u32,
    mtime: u64,
    bytes: Vec<u8>,
}

fn octal(field: &[u8]) -> u64 {
    let text = std::str::from_utf8(field).unwrap().trim_matches(|c| c == '\0' || c == ' ');
    u64::from_str_radix(text, 8).unwrap_or_else(|_| panic!("not octal: {text:?}"))
}

/// A gzip'd ustar archive read back, its checksums verified.
fn entries(archive: &[u8]) -> Vec<Entry> {
    let mut tar = Vec::new();
    flate2::read::GzDecoder::new(archive).read_to_end(&mut tar).expect("a gzip stream");
    let mut entries = Vec::new();
    let mut at = 0;
    while at + 512 <= tar.len() {
        let header = &tar[at..at + 512];
        if header.iter().all(|byte| *byte == 0) {
            break;
        }
        let mut sum = 0u64;
        for (i, byte) in header.iter().enumerate() {
            sum += if (148..156).contains(&i) { u64::from(b' ') } else { u64::from(*byte) };
        }
        assert_eq!(sum, octal(&header[148..156]), "the checksum of the header at {at}");
        assert_eq!(&header[257..263], b"ustar\0");
        let end = header[..100].iter().position(|byte| *byte == 0).unwrap_or(100);
        let size = octal(&header[124..136]) as usize;
        entries.push(Entry {
            path: String::from_utf8(header[..end].to_vec()).unwrap(),
            kind: header[156],
            mode: octal(&header[100..108]) as u32,
            uid: octal(&header[108..116]) as u32,
            gid: octal(&header[116..124]) as u32,
            mtime: octal(&header[136..148]),
            bytes: tar[at + 512..at + 512 + size].to_vec(),
        });
        at += 512 + size.div_ceil(512) * 512;
    }
    assert!(tar[at..].iter().all(|byte| *byte == 0) && tar.len() - at >= 1024, "the archive ends in two empty blocks");
    entries
}

/// Every file below `dir` as `ignis/<path>` -> bytes, walked here.
fn on_disk(dir: &Path, prefix: &str, files: &mut BTreeMap<String, Vec<u8>>) {
    for entry in std::fs::read_dir(dir).unwrap() {
        let entry = entry.unwrap();
        let path = format!("{prefix}/{}", entry.file_name().to_str().unwrap());
        if entry.file_type().unwrap().is_dir() {
            on_disk(&entry.path(), &path, files);
        } else {
            files.insert(path, std::fs::read(entry.path()).unwrap());
        }
    }
}

// ── the archive ──────────────────────────────────────────────────────────

/// Spec #656: the archive holds exactly the skill directory's files, byte
/// for byte, and every entry is under `ignis/`; extracting it into a skills
/// folder makes the skill's directory.
#[test]
fn the_archive_holds_exactly_the_skill_directorys_files_byte_for_byte_all_under_ignis() {
    let mut expected = BTreeMap::new();
    on_disk(&skill_dir(), "ignis", &mut expected);
    assert!(expected.contains_key("ignis/SKILL.md") && expected.contains_key("ignis/scripts/transcribe.mjs"), "{:?}", expected.keys());

    let entries = entries(ARCHIVE);
    let mut files = BTreeMap::new();
    for entry in &entries {
        assert!(entry.path.starts_with("ignis/"), "{}", entry.path);
        if entry.kind == b'0' {
            assert!(files.insert(entry.path.clone(), entry.bytes.clone()).is_none(), "{} twice", entry.path);
        } else {
            assert_eq!((entry.kind, entry.bytes.len()), (b'5', 0), "{}", entry.path);
            assert!(entry.path.ends_with('/'), "{}", entry.path);
        }
    }
    assert_eq!(files, expected);
    let directories: Vec<&str> = entries.iter().filter(|e| e.kind == b'5').map(|e| e.path.as_str()).collect();
    assert_eq!(directories, ["ignis/", "ignis/scripts/"]);
}

/// Spec #656: entries sorted, mtime and modes fixed, no owner.
#[test]
fn the_entries_are_sorted_with_a_fixed_mtime_and_modes() {
    let entries = entries(ARCHIVE);
    let paths: Vec<&str> = entries.iter().map(|entry| entry.path.as_str()).collect();
    let mut sorted = paths.clone();
    sorted.sort();
    assert_eq!(paths, sorted);
    assert_eq!(paths[0], "ignis/", "the directory comes before what it holds");
    for entry in &entries {
        assert_eq!(entry.mtime, 1_767_225_600, "{}", entry.path);
        assert_eq!((entry.uid, entry.gid), (0, 0), "{}", entry.path);
        assert_eq!(entry.mode, if entry.kind == b'5' { 0o755 } else { 0o644 }, "{}", entry.path);
    }
}

/// Spec #656: the same tree gives the same bytes packed twice, whatever
/// order it was written in; and the binary embeds what the packer makes of
/// the repo's directory.
#[test]
fn the_packer_gives_the_same_bytes_for_the_same_tree_packed_twice() {
    assert_eq!(skill_pack::pack(&skill_dir()), skill_pack::pack(&skill_dir()));
    assert_eq!(skill_pack::pack(&skill_dir()), ARCHIVE, "the binary embeds the packer's archive of the directory");

    let scratch = std::env::temp_dir().join(format!("ignis-skill-pack-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&scratch);
    let files = [("SKILL.md", "a"), ("scripts/z.mjs", "z"), ("scripts/a.mjs", "a"), ("scripts/deep/x.mjs", "x")];
    let mut packed = Vec::new();
    for (round, order) in [files.to_vec(), files.iter().rev().copied().collect()].into_iter().enumerate() {
        let root = scratch.join(format!("round-{round}")).join("ignis");
        for (path, text) in order {
            std::fs::create_dir_all(root.join(path).parent().unwrap()).unwrap();
            std::fs::write(root.join(path), text).unwrap();
        }
        packed.push(skill_pack::pack(&root));
    }
    std::fs::remove_dir_all(&scratch).unwrap();
    assert_eq!(packed[0], packed[1]);
    let paths: Vec<String> = entries(&packed[0]).into_iter().map(|entry| entry.path).collect();
    assert_eq!(
        paths,
        ["ignis/", "ignis/SKILL.md", "ignis/scripts/", "ignis/scripts/a.mjs", "ignis/scripts/deep/", "ignis/scripts/deep/x.mjs", "ignis/scripts/z.mjs"]
    );
}

/// Spec #656: `SKILL.md`'s frontmatter has `name` and `description` and
/// nothing else, and the name is the directory's.
#[test]
fn the_frontmatter_has_name_and_description_only() {
    let skill = String::from_utf8(std::fs::read(skill_dir().join("SKILL.md")).unwrap()).unwrap();
    assert!(!skill.contains('\r'), "LF line ends: the archive is the same bytes on every host");
    let rest = skill.strip_prefix("---\n").expect("frontmatter opens the file");
    let (front, _) = rest.split_once("\n---\n").expect("frontmatter closes");
    let mut keys = Vec::new();
    let mut name = "";
    for line in front.lines() {
        let (key, value) = line.split_once(": ").expect("one `key: value` per line");
        // A plain YAML scalar, as a harness parses it: `: ` would end a key.
        assert!(!value.contains(": ") && !value.starts_with(['"', '[', '{', '|', '>']), "{key}: {value}");
        keys.push(key);
        if key == "name" {
            name = value;
        }
    }
    assert_eq!(keys, ["name", "description"]);
    assert_eq!(name, "ignis");
    assert_eq!(skill_dir().file_name().unwrap(), name);
}

// ── the route ────────────────────────────────────────────────────────────

async fn get(app: &axum::Router, key: Option<&str>) -> (StatusCode, Option<String>, Vec<u8>) {
    let mut request = Request::builder().uri("/v1/connector/skill");
    if let Some(key) = key {
        request = request.header(header::AUTHORIZATION, key);
    }
    let response = app.clone().oneshot(request.body(Body::empty()).unwrap()).await.unwrap();
    let status = response.status();
    let kind = response.headers().get(header::CONTENT_TYPE).map(|v| v.to_str().unwrap().to_owned());
    (status, kind, response.into_body().collect().await.unwrap().to_bytes().to_vec())
}

/// Spec #656: the route serves the archive as `application/gzip`, the same
/// bytes every time.
#[tokio::test]
async fn the_route_serves_the_archive_as_gzip() {
    let app = server().app();
    let (status, kind, body) = get(&app, None).await;
    assert_eq!((status, kind.as_deref()), (StatusCode::OK, Some("application/gzip")));
    assert_eq!(body, ARCHIVE);
    assert_eq!(get(&app, None).await.2, body);
}

/// Spec #656: behind the key. And the install's own header (an empty key
/// when `IGNIS_API_KEY` is unset) is no obstacle on a server without one.
#[tokio::test]
async fn the_route_is_behind_the_key_and_an_install_without_a_key_works_on_a_server_without_one() {
    let keyed = server().with_api_key(ApiKey::new(KEY)).app();
    for presented in [None, Some("Bearer sk-wrong"), Some("Bearer ")] {
        let (status, _, body) = get(&keyed, presented).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED, "{presented:?}");
        let refused: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(refused["error"]["code"], "invalid_api_key");
    }
    let (status, _, body) = get(&keyed, Some(&format!("Bearer {KEY}"))).await;
    assert_eq!((status, body), (StatusCode::OK, ARCHIVE.to_vec()));

    // `-H "Authorization: Bearer $IGNIS_API_KEY"` with the variable unset.
    let open = server().app();
    assert_eq!(get(&open, Some("Bearer ")).await.0, StatusCode::OK);
    assert_eq!(get(&open, Some("Bearer anything")).await.0, StatusCode::OK);
}

/// Spec #656: `connector.enabled` off answers `404`, the same answer as
/// `/v1/mcp`, and the rest of `/v1` serves.
#[tokio::test]
async fn with_the_connector_off_the_route_answers_404() {
    let off = ConnectorGroup { enabled: false, ..ConnectorGroup::default() };
    let app = server().with_connector(off).app();
    let (status, _, body) = get(&app, None).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    let refused: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(refused["error"]["code"], "connector_disabled", "{refused}");
    let request = Request::builder().uri("/v1/models").body(Body::empty()).unwrap();
    assert_eq!(app.clone().oneshot(request).await.unwrap().status(), StatusCode::OK);
}
