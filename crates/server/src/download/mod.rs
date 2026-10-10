//! Fetching a model (GitHub #234 and #313; specs
//! `docs/specs/model-download/01-model-download.md` and
//! `02-catalog-and-model-command.md`; ADR 0033, amended by ADR 0047).
//!
//! A server started with no `.ninfer` artifact used to fall back to the
//! placeholder template and `MockCompute` without a word about where the real
//! model comes from. The models are published, so the server fetches its
//! own: it asks first when somebody is there to answer (stdin is a TTY), and
//! just downloads when nobody is (a container, a daemon, CI). `ignis-server
//! model download` fetches from the same catalog through the same transfer,
//! and never asks — a typed command is the operator's yes.
//!
//! In parts, each its own seam, so nothing here needs the network to be
//! tested:
//!
//! - `catalog.rs` — the [`Catalog`]: the built-in entries and the
//!   operator's, parsed and validated;
//! - [`artifact_source`] — the start-time **decision**: pure, given the
//!   catalog, an `exists` probe and whatever the operator answered; every
//!   cell of spec 01's table is a unit test below;
//! - [`bearer_token`] — which token a request carries: pure;
//! - `transfer.rs` — the [`Downloader`], driven against loopback `axum`
//!   servers (`tests/model_download.rs`);
//! - `command.rs` — `model download` and `model list`.

use std::path::{Path, PathBuf};

use crate::config::ApiKey;

pub mod catalog;
pub mod command;
mod transfer;

pub use catalog::{Catalog, CatalogEntry, CatalogError, CatalogFile, CatalogLayer, BUILT_IN_CATALOG};
pub use command::{ListFormat, ModelCommand, OnDisk};
pub use transfer::{DownloadError, Downloader, Report};

/// The host whose own credential `HF_TOKEN` is.
const HUGGINGFACE_HOST: &str = "huggingface.co";

/// Where the server's artifact comes from, decided once at start.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArtifactSource {
    /// Load this file (it is on disk, or the operator named it and owns the
    /// consequences of it not being).
    Use(PathBuf),
    /// Fetch `entry` — every file of it — into `dir` first, then load what
    /// lands there.
    Download {
        entry: CatalogEntry,
        dir: PathBuf,
    },
    /// No artifact: the placeholder template and `MockCompute`, as before.
    Placeholder(PlaceholderReason),
}

/// Why a start carries no artifact — one warn line each, so the operator is
/// never left guessing which of these happened.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlaceholderReason {
    /// This binary was built without `--features cuda`: it could not run the
    /// weights it downloaded, so it never downloads them.
    NotSupported,
    /// `--model-id` names a model the catalog has no entry for.
    UnknownModel,
    /// `--download-enabled false` / `IGNIS_DOWNLOAD_ENABLED=false`.
    DownloadsDisabled,
    /// The operator was asked and said no.
    Declined,
}

impl PlaceholderReason {
    /// The reason as a log field value.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::NotSupported => "not-supported",
            Self::UnknownModel => "unknown-model",
            Self::DownloadsDisabled => "downloads-disabled",
            Self::Declined => "declined",
        }
    }
}

/// What the config says about fetching a missing model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DownloadSettings<'a> {
    /// `--model-artifact` / `IGNIS_MODEL_ARTIFACT`, when the operator named one.
    pub artifact: Option<&'a Path>,
    /// `--model-id` / `IGNIS_MODEL_ID` — the catalog key.
    pub model: &'a str,
    /// `--download-enabled` / `--download-enabled false`.
    pub enabled: bool,
    /// `--download-path`: where a downloaded artifact lands.
    pub dir: &'a Path,
    /// Whether this binary can run a model at all (built with `cuda`).
    pub supported: bool,
    /// The merged catalog: the built-in entries and the operator's.
    pub catalog: &'a Catalog,
}

/// Decide where the artifact comes from (the spec's table, in one pure
/// function).
///
/// `exists` probes the filesystem; `ask` puts the question to the operator
/// and returns `Some(answer)`, or `None` when there is nobody to ask (stdin
/// is not a TTY). It is called only when the answer can still change the
/// outcome, so a start that already has its model is never interrupted by a
/// question.
pub fn artifact_source(
    settings: &DownloadSettings<'_>,
    exists: impl Fn(&Path) -> bool,
    ask: impl FnOnce(&CatalogEntry, &Path) -> Option<bool>,
) -> ArtifactSource {
    // The operator's own path wins whatever else is true, missing or not:
    // `main` refuses the start on a path it cannot open, and a typo must stay
    // a typo rather than become a 19 GB download somewhere else.
    if let Some(artifact) = settings.artifact {
        return ArtifactSource::Use(artifact.to_path_buf());
    }
    if !settings.supported {
        return ArtifactSource::Placeholder(PlaceholderReason::NotSupported);
    }
    let Some(entry) = settings.catalog.entry(settings.model) else {
        return ArtifactSource::Placeholder(PlaceholderReason::UnknownModel);
    };
    let path = entry.artifact_path(settings.dir);
    // Already downloaded (by this server, by `model download`, by hand, or
    // by the shared model store this directory points at): load it, never
    // fetch it again.
    if exists(&path) {
        return ArtifactSource::Use(path);
    }
    if !settings.enabled {
        return ArtifactSource::Placeholder(PlaceholderReason::DownloadsDisabled);
    }
    let answer = ask(entry, &path);
    let download = ArtifactSource::Download {
        entry: entry.clone(),
        dir: settings.dir.to_path_buf(),
    };
    match answer {
        // Nobody to ask: a container or a daemon cannot answer a question,
        // and coming up on the placeholder would be the wrong surprise.
        None | Some(true) => download,
        Some(false) => ArtifactSource::Placeholder(PlaceholderReason::Declined),
    }
}

/// Which token a request to `endpoint` carries (spec model-download/02
/// §Endpoint and token): `download.token` when it is set; else `HF_TOKEN`,
/// but **only** when the endpoint's host is `huggingface.co` itself — a
/// personal Hugging Face credential is never handed to a mirror; else none.
/// No request carries it across a redirect to another host (the
/// [`Downloader`]'s own rule).
pub fn bearer_token(endpoint: &str, configured: Option<&ApiKey>, hf_token: Option<&str>) -> Option<ApiKey> {
    if let Some(token) = configured.filter(|token| !token.as_str().trim().is_empty()) {
        return Some(token.clone());
    }
    let hf_token = hf_token.map(str::trim).filter(|token| !token.is_empty())?;
    let huggingface = reqwest::Url::parse(endpoint)
        .ok()
        .and_then(|url| url.host_str().map(|host| host.eq_ignore_ascii_case(HUGGINGFACE_HOST)))
        .unwrap_or(false);
    huggingface.then(|| ApiKey::new(hf_token))
}

/// Put the question to the operator on stderr and read the answer from stdin.
///
/// `None` when stdin is not a terminal — there is nobody to ask, and blocking
/// a container's start on a prompt nobody will ever answer is the one outcome
/// worse than either answer. The question goes to **stderr** because stdout
/// carries the plain lines `mk/windows/common.ps1` parses (the generated API
/// key, the public URL).
pub fn ask_on_terminal(entry: &CatalogEntry, path: &Path, endpoint: &str) -> Option<bool> {
    use std::io::IsTerminal;

    if !std::io::stdin().is_terminal() {
        return None;
    }
    Some(ask(
        entry,
        path,
        endpoint,
        &mut std::io::stderr(),
        &mut std::io::stdin().lock(),
    ))
}

/// The question itself, over any pair of streams — the seam the tests drive.
/// It names what the fetch spends (every file of the entry), where it comes
/// from (the configured endpoint, not a hard-coded Hugging Face) and where it
/// lands.
///
/// Everything that is not a plain yes is a no, EOF and a stdin that cannot be
/// read included: past this point the terminal is known to be there, so the
/// only safe reading of "no answer" is the one that spends nothing.
pub fn ask(
    entry: &CatalogEntry,
    path: &Path,
    endpoint: &str,
    out: &mut impl std::io::Write,
    input: &mut impl std::io::BufRead,
) -> bool {
    let _ = writeln!(
        out,
        "\nignis-server: no model at {}\n  {} is published at {}/{} ({:.1} GiB in {} files)\n  it will be saved as {}\n  (--download-enabled false never asks; --download-path puts it elsewhere)",
        path.display(),
        entry.id,
        endpoint.trim_end_matches('/'),
        entry.repo,
        entry.gib(),
        entry.files.len(),
        path.display(),
    );
    let _ = write!(out, "Download it now? [y/N] ");
    let _ = out.flush();

    let mut answer = String::new();
    match input.read_line(&mut answer) {
        Ok(0) | Err(_) => {
            let _ = writeln!(out);
            false
        }
        Ok(_) => matches!(
            answer.trim().to_ascii_lowercase().as_str(),
            "y" | "yes"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DIR: &str = "models";

    fn settings<'a>(model: &'a str, dir: &'a Path) -> DownloadSettings<'a> {
        DownloadSettings {
            artifact: None,
            model,
            enabled: true,
            dir,
            supported: true,
            catalog: Catalog::built_in(),
        }
    }

    /// The built-in entry the server starts on by default.
    fn default_entry() -> &'static CatalogEntry {
        Catalog::built_in().entry(crate::config::DEFAULT_MODEL).expect("the default model is in the catalog")
    }

    /// An `ask` that must never be reached.
    fn never_asked(_: &CatalogEntry, _: &Path) -> Option<bool> {
        panic!("the operator must not be asked in this cell");
    }

    fn nothing_exists(_: &Path) -> bool {
        false
    }

    #[test]
    fn a_named_artifact_wins_whether_or_not_it_is_there() {
        // AC1: `--model-artifact` is the operator's word. Present or missing, the
        // decision is the same — `main` owns the failure, not the downloader.
        let dir = PathBuf::from(DIR);
        let named = PathBuf::from("/elsewhere/custom.ninfer");
        let mut settings = settings(crate::config::DEFAULT_MODEL, &dir);
        settings.artifact = Some(&named);
        assert_eq!(
            artifact_source(&settings, nothing_exists, never_asked),
            ArtifactSource::Use(named.clone())
        );
        assert_eq!(
            artifact_source(&settings, |_| true, never_asked),
            ArtifactSource::Use(named)
        );
    }

    #[test]
    fn a_build_that_cannot_run_the_model_never_downloads_it() {
        // AC1: no `cuda`, no download — `make mock` and every CPU test job
        // stay off the network by construction, not by remembering a flag.
        let dir = PathBuf::from(DIR);
        let mut settings = settings(crate::config::DEFAULT_MODEL, &dir);
        settings.supported = false;
        assert_eq!(
            artifact_source(&settings, nothing_exists, never_asked),
            ArtifactSource::Placeholder(PlaceholderReason::NotSupported)
        );
    }

    #[test]
    fn an_unknown_model_falls_through_to_the_placeholder() {
        let dir = PathBuf::from(DIR);
        let settings = settings("some-other-model", &dir);
        assert_eq!(
            artifact_source(&settings, nothing_exists, never_asked),
            ArtifactSource::Placeholder(PlaceholderReason::UnknownModel)
        );
    }

    #[test]
    fn an_artifact_already_on_disk_is_loaded_and_not_fetched_again() {
        // The everyday case on a machine that has the model: nothing is
        // asked, nothing is downloaded, and the path is the flat one.
        let dir = PathBuf::from(DIR);
        let settings = settings(crate::config::DEFAULT_MODEL, &dir);
        let expected = dir.join(&default_entry().artifact);
        let probed = std::cell::RefCell::new(Vec::new());
        let source = artifact_source(
            &settings,
            |p| {
                probed.borrow_mut().push(p.to_path_buf());
                true
            },
            never_asked,
        );
        assert_eq!(source, ArtifactSource::Use(expected.clone()));
        assert_eq!(probed.into_inner(), vec![expected]);
    }

    #[test]
    fn downloads_off_keeps_the_placeholder_and_asks_nothing() {
        let dir = PathBuf::from(DIR);
        let mut settings = settings(crate::config::DEFAULT_MODEL, &dir);
        settings.enabled = false;
        assert_eq!(
            artifact_source(&settings, nothing_exists, never_asked),
            ArtifactSource::Placeholder(PlaceholderReason::DownloadsDisabled)
        );
    }

    #[test]
    fn downloads_off_still_loads_an_artifact_that_is_there() {
        // `--download-enabled false` forbids the download, not the model: a file
        // already on disk is still what the server loads.
        let dir = PathBuf::from(DIR);
        let mut settings = settings(crate::config::DEFAULT_MODEL, &dir);
        settings.enabled = false;
        assert_eq!(
            artifact_source(&settings, |_| true, never_asked),
            ArtifactSource::Use(dir.join(&default_entry().artifact))
        );
    }

    #[test]
    fn a_tty_is_asked_and_yes_downloads() {
        let dir = PathBuf::from(DIR);
        let settings = settings(crate::config::DEFAULT_MODEL, &dir);
        let source = artifact_source(&settings, nothing_exists, |entry, path| {
            // The question knows what it is about to spend, and where.
            assert_eq!(entry.id, crate::config::DEFAULT_MODEL);
            assert_eq!(path, dir.join(&entry.artifact));
            Some(true)
        });
        assert_eq!(
            source,
            ArtifactSource::Download {
                entry: default_entry().clone(),
                dir: dir.clone()
            }
        );
    }

    #[test]
    fn a_tty_that_says_no_keeps_the_placeholder() {
        let dir = PathBuf::from(DIR);
        let settings = settings(crate::config::DEFAULT_MODEL, &dir);
        assert_eq!(
            artifact_source(&settings, nothing_exists, |_, _| Some(false)),
            ArtifactSource::Placeholder(PlaceholderReason::Declined)
        );
    }

    #[test]
    fn nobody_to_ask_downloads() {
        // A container or a daemon: no TTY, so no question — and coming up on
        // the placeholder would be the wrong surprise.
        let dir = PathBuf::from(DIR);
        let settings = settings(crate::config::DEFAULT_MODEL, &dir);
        assert_eq!(
            artifact_source(&settings, nothing_exists, |_, _| None),
            ArtifactSource::Download {
                entry: default_entry().clone(),
                dir
            }
        );
    }

    /// Spec model-download/02 AC 10: the decision is the same pure function
    /// over the merged catalog, so an operator's entry is fetched, found and
    /// loaded exactly as a built-in one is — and a built-in id other than
    /// the default one too.
    #[test]
    fn an_operator_entry_and_any_built_in_one_take_the_same_decision() {
        let operator = catalog::parse_operator(
            "models:
  - id: acme-ft
    repo: acme/ft
    revision: v1
    artifact: acme_ft.ninfer
    files:
      - { name: acme_ft.ninfer.conversion.json, bytes: 10, sha256: 0000000000000000000000000000000000000000000000000000000000000001 }
      - { name: acme_ft.ninfer, bytes: 20, sha256: 0000000000000000000000000000000000000000000000000000000000000002 }
",
            crate::config::file::Format::Yaml,
            "acme.yaml",
        )
        .expect("valid");
        let dir = PathBuf::from(DIR);
        for id in ["acme-ft", "qwen3.8-flash-next", "qwen3.8-27b-abliterated"] {
            let mut settings = settings(id, &dir);
            settings.catalog = &operator;
            let entry = operator.entry(id).expect("listed");
            assert_eq!(
                artifact_source(&settings, nothing_exists, |asked, _| {
                    assert_eq!(asked, entry);
                    None
                }),
                ArtifactSource::Download { entry: entry.clone(), dir: dir.clone() },
                "{id}"
            );
            assert_eq!(artifact_source(&settings, |_| true, never_asked), ArtifactSource::Use(dir.join(&entry.artifact)), "{id}");
        }
        // Without the operator's catalog, its id is just an unknown model.
        assert_eq!(
            artifact_source(&settings("acme-ft", &dir), nothing_exists, never_asked),
            ArtifactSource::Placeholder(PlaceholderReason::UnknownModel)
        );
    }

    /// Spec model-download/02 AC 5, one cell each: the configured token wins
    /// everywhere; `HF_TOKEN` reaches Hugging Face itself and nothing else;
    /// otherwise no token.
    #[test]
    fn the_token_is_the_configured_one_else_hf_token_for_hugging_face_only() {
        let configured = ApiKey::new("tok-configured");
        let token = |endpoint: &str, configured: Option<&ApiKey>, hf: Option<&str>| {
            bearer_token(endpoint, configured, hf).map(|key| key.as_str().to_owned())
        };
        let hf = "https://huggingface.co";
        let mirror = "https://artifactory.example.com/api/huggingfaceml/hf-remote";
        // download.token set: it wins, whatever the endpoint and HF_TOKEN.
        assert_eq!(token(hf, Some(&configured), Some("hf_personal")).as_deref(), Some("tok-configured"));
        assert_eq!(token(mirror, Some(&configured), Some("hf_personal")).as_deref(), Some("tok-configured"));
        assert_eq!(token(mirror, Some(&configured), None).as_deref(), Some("tok-configured"));
        // Unset: HF_TOKEN, on huggingface.co only (any case, any port, a
        // trailing slash or a path).
        assert_eq!(token(hf, None, Some("hf_personal")).as_deref(), Some("hf_personal"));
        assert_eq!(token("https://HuggingFace.co/", None, Some("hf_personal")).as_deref(), Some("hf_personal"));
        assert_eq!(token(mirror, None, Some("hf_personal")), None, "a mirror never gets a personal Hugging Face credential");
        for lookalike in ["https://huggingface.co.example.com", "https://cdn-lfs.huggingface.co", "http://127.0.0.1:8080"] {
            assert_eq!(token(lookalike, None, Some("hf_personal")), None, "{lookalike}");
        }
        // Neither: none. An empty value is no value.
        assert_eq!(token(hf, None, None), None);
        assert_eq!(token(hf, None, Some("  ")), None);
        assert_eq!(token(hf, Some(&ApiKey::new("")), Some("hf_personal")).as_deref(), Some("hf_personal"));
    }

    /// A stdin that fails instead of answering (a pipe that broke, a byte
    /// sequence `read_line` cannot decode).
    struct FailingInput;

    impl std::io::Read for FailingInput {
        fn read(&mut self, _: &mut [u8]) -> std::io::Result<usize> {
            Err(std::io::Error::other("stdin is gone"))
        }
    }

    impl std::io::BufRead for FailingInput {
        fn fill_buf(&mut self) -> std::io::Result<&[u8]> {
            Err(std::io::Error::other("stdin is gone"))
        }
        fn consume(&mut self, _: usize) {}
    }

    #[test]
    fn the_question_names_the_size_the_source_and_the_destination() {
        // AC6: what the operator is about to spend, where it comes from and
        // where it lands — all three in the question, on the stream the
        // caller gives it (stderr in `ask_on_terminal`, never stdout, which
        // carries the lines `mk/windows/common.ps1` parses).
        let entry = default_entry();
        let path = PathBuf::from(DIR).join(&entry.artifact);
        let mut out = Vec::new();
        let answered = ask(entry, &path, "https://huggingface.co", &mut out, &mut &b"y\n"[..]);
        let asked = String::from_utf8(out).expect("utf-8");
        assert!(answered);
        assert!(asked.contains(&entry.id), "{asked}");
        assert!(asked.contains(&format!("https://huggingface.co/{}", entry.repo)), "{asked}");
        assert!(asked.contains("18.1 GiB"), "{asked}");
        assert!(asked.contains(&path.display().to_string()), "{asked}");
        assert!(asked.contains("[y/N]"), "{asked}");
    }

    /// Spec model-download/02 AC 11: the question names the endpoint the
    /// bytes will come from — a mirror, when one is configured — never a
    /// hard-coded Hugging Face.
    #[test]
    fn the_question_names_the_configured_endpoint() {
        let entry = default_entry();
        let path = PathBuf::from(DIR).join(&entry.artifact);
        let mut out = Vec::new();
        ask(entry, &path, "https://mirror.example.com/hf/", &mut out, &mut &b"n\n"[..]);
        let asked = String::from_utf8(out).expect("utf-8");
        assert!(asked.contains(&format!("https://mirror.example.com/hf/{}", entry.repo)), "{asked}");
        assert!(!asked.contains("huggingface.co"), "{asked}");
    }

    #[test]
    fn only_a_plain_yes_spends_the_bandwidth() {
        // AC6: EOF reads as no, and so does everything that is not a yes —
        // 19.4 GB is not something to spend on a stray Enter.
        let entry = default_entry();
        let path = PathBuf::from(DIR).join(&entry.artifact);
        for (answer, want) in [
            ("y\n", true),
            ("Y\n", true),
            ("yes\n", true),
            ("  yes  \n", true),
            ("n\n", false),
            ("\n", false),
            ("later\n", false),
            ("", false), // EOF: stdin closed without an answer
        ] {
            let mut out = Vec::new();
            assert_eq!(
                ask(entry, &path, "https://huggingface.co", &mut out, &mut answer.as_bytes()),
                want,
                "{answer:?}"
            );
        }
    }

    #[test]
    fn a_stdin_that_cannot_be_read_is_a_no() {
        // The terminal was there (`ask_on_terminal` checked), so a read that
        // fails is not "nobody to ask" — it is an operator who never said
        // yes, and the fail-closed answer is the one that downloads nothing.
        let entry = default_entry();
        let path = PathBuf::from(DIR).join(&entry.artifact);
        let mut out = Vec::new();
        assert!(!ask(entry, &path, "https://huggingface.co", &mut out, &mut FailingInput));
    }
}
