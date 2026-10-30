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
//! - `transfer.rs` — the [`Downloader`], and the [`Download`] it runs in the
//!   background, driven against loopback `axum` servers
//!   (`tests/model_download.rs`);
//! - `lock.rs` — the [`EntryLock`] every fetch holds: one writer per entry
//!   in a folder, the server's and `model download`'s alike;
//! - `command.rs` — `model download` and `model list`.

use std::path::{Path, PathBuf};

use crate::config::ApiKey;

pub mod catalog;
pub mod command;
mod lock;
mod transfer;

pub use catalog::{Catalog, CatalogEntry, CatalogError, CatalogFile, CatalogLayer, ClassRefusal, BUILT_IN_CATALOG};
pub use command::{ListFormat, ModelCommand, OnDisk};
pub use lock::{EntryLock, Holder, NotLocked};
pub use transfer::{Download, DownloadError, Downloader, Ended, Report};

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

/// The served id a start with no artifact named loads (ADR 0064, spec
/// 27b-16gb/04 §6): the catalog's class-aware step ([`Catalog::answering`])
/// on `model` and the card's class `card`, before [`artifact_source`] looks
/// it up. `model` itself when its own entry fits the card, no class is known,
/// or the catalog does not hold it (for [`artifact_source`] to answer as
/// before); on a card below its entry's class, the entry of its family that
/// answers to it there — the default id `qwen3.8-27b` on a 16 GB card is the
/// 16 GB 27B's, which is how the `rtx50-16gb` profile's default model is
/// realized, served under that entry's own id; refused, naming both classes,
/// when no entry that answers to it fits: the start is refused before
/// anything is fetched. Matched as [`artifact_source`] matches `model`,
/// case-insensitively. A start naming its artifact never comes here: that
/// path is the operator's word.
pub fn start_model(catalog: &Catalog, model: &str, card: Option<crate::config::profile::CardClass>) -> Result<String, ClassRefusal> {
    let Some(own) = catalog.entry(model) else {
        return Ok(model.to_owned());
    };
    Ok(catalog.answering(&own.id, card)?.map_or_else(|| model.to_owned(), |serving| serving.id.clone()))
}

/// The start options a start runs with once [`start_model`] has said which
/// id it serves and [`artifact_source`] where its artifact is (ADR 0064 §2):
/// `options` as they are when it serves the id they were resolved with — a
/// named artifact, or an id whose own entry fits the card — or loads no
/// artifact (the placeholder); else `served`, the entry that answers to
/// `options.model` on this card, under its own id: its artifact and id over
/// the options, as a switch to it would serve it
/// ([`crate::config::target_patch`]), so `GET /v1/config`, the switcher and
/// the load all name the model that runs.
///
/// An alias resolves only within one model family (ADR 0064 §2), and the
/// family is the artifact's header's (ADR 0047), which is known only once the
/// artifact is on disk — after the fetch when the start fetched it: `header`
/// reads it (`loader::artifact_family`), and an artifact of another family
/// than the configured id's refuses the start before it loads, naming both
/// ([`Catalog::alias_holds`]).
pub fn served_start_options(
    options: crate::config::Config,
    served: &str,
    artifact: Option<&Path>,
    catalog: &Catalog,
    card: Option<crate::config::profile::CardClass>,
    header: &dyn Fn(&Path) -> Option<ignis_core::compute::ModelFamily>,
) -> Result<crate::config::Config, StartRefusal> {
    let Some(path) = artifact.filter(|_| served != options.model) else {
        return Ok(options);
    };
    if let (Some(own), Some(entry)) = (catalog.entry(&options.model), catalog.entry(served)) {
        catalog.alias_holds(&own.id, entry, header(path), card).map_err(StartRefusal::AboveClass)?;
    }
    crate::config::target_patch(path, served).and_then(|patch| options.general_with(&patch)).map_err(StartRefusal::Config)
}

/// Why a start the card's class fitted is refused once its artifact is on
/// disk ([`served_start_options`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StartRefusal {
    /// The entry that answers to the configured id holds an artifact of
    /// another family (ADR 0064 §2): the alias does not resolve.
    AboveClass(ClassRefusal),
    /// Its artifact and id over the start options do not resolve.
    Config(crate::config::ConfigError),
}

/// The start-time fetch ([`ArtifactSource::Download`]): every file of `entry`
/// into `dir` from `endpoint` with `token`, once the volume `dir` is on has
/// room for what of it is still missing plus `free_margin`
/// (`download.free_margin`, GitHub #378), checked before a byte is written,
/// as the Server view's Download and `model download` check it
/// ([`command::room_for`]). Whatever stops it — no room, an HTTP client that
/// cannot be built, the transfer's own failure — is logged as
/// `ignis.model.download_failed` at ERROR, "refusing to start", and
/// returned: `main` refuses the start on it, with exit code 1. The artifact's
/// path once every file verified.
pub async fn fetch_at_start(entry: &CatalogEntry, dir: &Path, endpoint: &str, token: Option<ApiKey>, free_margin: u64) -> Result<PathBuf, DownloadError> {
    let refused = |err: DownloadError| {
        tracing::error!(
            name: "ignis.model.download_failed",
            model = %entry.id,
            repo = %entry.repo,
            endpoint = %endpoint,
            error = %err,
            "refusing to start"
        );
        err
    };
    command::room_for(entry, dir, free_margin).map_err(|reason| refused(DownloadError(reason)))?;
    let downloader = Downloader::new(endpoint, token).map_err(refused)?;
    downloader.fetch(entry, dir).await.map_err(refused)
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
    is_huggingface(endpoint).then(|| ApiKey::new(hf_token))
}

/// Whether `endpoint`'s host is `huggingface.co` itself, the one host
/// `HF_TOKEN` is sent to ([`bearer_token`]).
pub(crate) fn is_huggingface(endpoint: &str) -> bool {
    reqwest::Url::parse(endpoint)
        .ok()
        .and_then(|url| url.host_str().map(|host| host.eq_ignore_ascii_case(HUGGINGFACE_HOST)))
        .unwrap_or(false)
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
    use ignis_core::compute::ModelFamily;

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

    /// ADR 0064, spec 27b-16gb/04 §6: with no artifact named, the configured
    /// id is fitted to the card's class before the catalog is searched for
    /// its file — on a 16 GB card the 27B's default id starts the 16 GB
    /// entry; on a 32 GB card, or one of no known class, it is its own.
    #[test]
    fn a_start_with_no_artifact_loads_the_entry_that_serves_its_id_on_the_card() {
        use crate::config::profile::CardClass;
        let catalog = catalog::testing::built_in();
        assert_eq!(start_model(&catalog, crate::config::DEFAULT_MODEL, Some(CardClass::Gb16)), Ok("qwen3.8-27b-16g".to_owned()));
        for card in [Some(CardClass::Gb32), Some(CardClass::Gb96), None] {
            assert_eq!(start_model(&catalog, crate::config::DEFAULT_MODEL, card), Ok(crate::config::DEFAULT_MODEL.to_owned()), "{card:?}");
        }
        assert_eq!(start_model(&catalog, "qwen3.8-27b-16g", Some(CardClass::Gb32)), Ok("qwen3.8-27b-16g".to_owned()), "a 16 GB entry fits a larger card");
        assert_eq!(start_model(&catalog, "some-other-model", Some(CardClass::Gb16)), Ok("some-other-model".to_owned()), "artifact_source's to answer");
        // The decision then finds the entry the start serves.
        let dir = PathBuf::from(DIR);
        let mut settings = settings("qwen3.8-27b-16g", &dir);
        settings.catalog = &catalog;
        assert_eq!(artifact_source(&settings, |_| true, never_asked), ArtifactSource::Use(dir.join("qwen3_8_27b_trellis_sc30-v2.ninfer")));
    }

    /// The start options a configuration resolves to on a card of `total`
    /// bytes, as `main` reads them (`config::resolve_on`, the probe injected).
    fn start_config_on(total: u64) -> crate::config::Config {
        start_config_with(&[], total)
    }

    /// [`start_config_on`] with `flags` on the command line.
    fn start_config_with(flags: &[&str], total: u64) -> crate::config::Config {
        let card = move || crate::config::profile::Card::Total(total);
        let flags: Vec<String> = flags.iter().map(|flag| flag.to_string()).collect();
        match crate::config::resolve_on(&flags, |_: &str| None, &crate::config::file::NoFiles, &card).expect("resolves") {
            crate::config::ConfigOutcome::Config(config) => config,
            other => panic!("expected a start, got {other:?}"),
        }
    }

    /// ADR 0064 §2, `main`'s start path end to end: a start with no artifact
    /// named on a 16 GB card — the class `Config::card_class` reads from the
    /// probe's total — finds the entry that answers to the default id on
    /// disk and serves it under that entry's own id, its artifact and id over
    /// the start options, as a switch to it would serve it. On a 32 GB card
    /// the options are the start's own; so are they on the placeholder path,
    /// which loads no artifact.
    #[test]
    fn a_start_the_card_class_fitted_serves_the_answering_entry_under_its_own_id() {
        use crate::config::profile::CardClass;
        let catalog = catalog::testing::built_in();
        let dir = PathBuf::from(DIR);
        let options = start_config_on(16 << 30);
        assert_eq!(options.card_class(), Some(CardClass::Gb16));
        let served = start_model(&catalog, &options.model, options.card_class()).unwrap();
        assert_eq!(served, "qwen3.8-27b-16g");
        let mut settings = settings(&served, &dir);
        settings.catalog = &catalog;
        let ArtifactSource::Use(path) = artifact_source(&settings, |_| true, never_asked) else { panic!("on disk") };
        let card = options.card_class();
        let serving = served_start_options(options.clone(), &served, Some(&path), &catalog, card, &|_| Some(ModelFamily::Qwen38_27b)).unwrap();
        assert_eq!((serving.model.as_str(), serving.artifact.as_deref()), ("qwen3.8-27b-16g", Some(dir.join("qwen3_8_27b_trellis_sc30-v2.ninfer").as_path())));
        let placeholder = served_start_options(options.clone(), &served, None, &catalog, card, &|_| panic!("no artifact, no header"));
        assert_eq!(placeholder.unwrap(), options, "a placeholder start loads nothing");

        let options = start_config_on(crate::config::profile::RTX_5090_TOTAL_BYTES);
        let served = start_model(&catalog, &options.model, options.card_class()).unwrap();
        assert_eq!(served, crate::config::DEFAULT_MODEL);
        let own = dir.join("qwen3_8_27b_nvfp4full-v2.ninfer");
        assert_eq!(served_start_options(options.clone(), &served, Some(&own), &catalog, options.card_class(), &|_| panic!("its own entry")).unwrap(), options);
    }

    /// ADR 0064 §2: an alias resolves only within one model family, by the
    /// artifact's header (ADR 0047). A start that has to fetch the entry
    /// answering to the default id knows the header only once the fetch has
    /// landed: an artifact of another family — or of none — then refuses the
    /// start before it loads, naming both families.
    #[test]
    fn a_start_whose_answering_entry_holds_another_family_is_refused_once_fetched() {
        let catalog = catalog::testing::built_in();
        let dir = PathBuf::from(DIR);
        let options = start_config_on(16 << 30);
        let served = start_model(&catalog, &options.model, options.card_class()).unwrap();
        let mut settings = settings(&served, &dir);
        settings.catalog = &catalog;
        let ArtifactSource::Download { entry, dir } = artifact_source(&settings, nothing_exists, |_, _| None) else { panic!("fetched") };
        let fetched = entry.artifact_path(&dir);
        for header in [Some(ModelFamily::FlashNext), None] {
            let Err(StartRefusal::AboveClass(refusal)) = served_start_options(options.clone(), &served, Some(&fetched), &catalog, options.card_class(), &|_| header) else {
                panic!("{header:?}: refused")
            };
            let said = refusal.to_string();
            assert!(said.contains("qwen3.8-27b-16g lists it as an alias") && said.contains("one model family"), "{said}");
        }
    }

    /// ADR 0074, `main`'s start path end to end with the built-in catalog
    /// (GitHub #423): `--model-id qwen3.8-27b-abliterated` on a 16 GB card
    /// starts the abliterated 16 GB build, which answers to it, under its own
    /// id —
    /// its artifact and id over the start options — when its artifact's
    /// header names the 27B's family, the one both entries state; one of
    /// another family refuses the start before it loads, naming both. On a
    /// 32 GB card, a 96 GB one or one of no known class the id is its own
    /// entry's.
    #[test]
    fn a_start_naming_the_uncensored_variant_on_a_16gb_card_serves_its_16gb_build() {
        use crate::config::profile::CardClass;
        let catalog = catalog::testing::built_in();
        let dir = PathBuf::from(DIR);
        let options = start_config_with(&["--model-id", "qwen3.8-27b-abliterated"], 16 << 30);
        assert_eq!((options.model.as_str(), options.card_class()), ("qwen3.8-27b-abliterated", Some(CardClass::Gb16)));
        let served = start_model(&catalog, &options.model, options.card_class()).unwrap();
        assert_eq!(served, "qwen3.8-27b-abliterated-16g");
        let mut settings = settings(&served, &dir);
        settings.catalog = &catalog;
        let ArtifactSource::Use(path) = artifact_source(&settings, |_| true, never_asked) else { panic!("on disk") };
        assert_eq!(path, dir.join("qwen3_8_27b_trellis_sc30-v2-huihui-abliterated.ninfer"));
        let card = options.card_class();
        let serving = served_start_options(options.clone(), &served, Some(&path), &catalog, card, &|_| Some(ModelFamily::Qwen38_27b)).unwrap();
        assert_eq!((serving.model.as_str(), serving.artifact.as_deref()), ("qwen3.8-27b-abliterated-16g", Some(path.as_path())));
        let Err(StartRefusal::AboveClass(refusal)) = served_start_options(options.clone(), &served, Some(&path), &catalog, card, &|_| Some(ModelFamily::FlashNext)) else {
            panic!("a header of another family refuses the start")
        };
        let said = refusal.to_string();
        for named in ["qwen3.8-27b-abliterated-16g lists it as an alias", "Qwen3.8-Flash-Next", "Qwen3.8-27B", "one model family"] {
            assert!(said.contains(named), "{named}: {said}");
        }

        for card in [Some(CardClass::Gb32), Some(CardClass::Gb96), None] {
            assert_eq!(start_model(&catalog, "qwen3.8-27b-abliterated", card), Ok("qwen3.8-27b-abliterated".to_owned()), "{card:?}");
        }
        let options = start_config_with(&["--model-id", "qwen3.8-27b-abliterated"], crate::config::profile::RTX_5090_TOTAL_BYTES);
        let served = start_model(&catalog, &options.model, options.card_class()).unwrap();
        assert_eq!(served, "qwen3.8-27b-abliterated");
        let own = dir.join("qwen3_8_27b_nvfp4full-v2-huihui-abliterated.ninfer");
        assert_eq!(served_start_options(options.clone(), &served, Some(&own), &catalog, options.card_class(), &|_| panic!("its own entry")).unwrap(), options);
    }

    /// ADR 0064 §3: a start whose model no entry loads on this card is
    /// refused naming both classes — the default model on a 16 GB card in a
    /// catalog without the 16 GB 27B.
    #[test]
    fn a_start_with_no_artifact_the_card_cannot_load_is_refused_naming_both_classes() {
        use crate::config::profile::CardClass;
        let refusal = start_model(&catalog::testing::thirty_two(), crate::config::DEFAULT_MODEL, Some(CardClass::Gb16)).unwrap_err();
        assert_eq!((refusal.model.as_str(), refusal.needs, refusal.card), ("qwen3.8-27b", CardClass::Gb32, CardClass::Gb16));
        let refusal = start_model(&catalog::testing::built_in(), "Qwen3.8-Flash-Next", Some(CardClass::Gb16)).unwrap_err();
        assert!(refusal.to_string().contains("qwen3.8-flash-next needs a 32 GB card"), "{refusal}");
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
    card_class: 32
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
