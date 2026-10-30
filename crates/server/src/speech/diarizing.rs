//! The diarizer in the server's catalog (spec speech/08, ADR 0069): which
//! model `diarizer.model` names, whether it can serve, and the executor that
//! loads it at the first session that asks for speakers.
//!
//! Its state is computed where it is read, as the speech model's is
//! ([`super::Speech::resolve`]): `diarizer.model` is `off`, an id or `auto`
//! (the catalog's one `kind: diarizer` entry), and a live change of it
//! ([`DiarizerExecutor::follow`]) takes the next session's open, never one
//! under way. It is checked in the order a person fixes it: off; an id the
//! catalog lacks; an entry that is not a diarizer; a host without AVX2; the
//! files missing, naming the command that fetches them.

use std::path::PathBuf;
use std::sync::{Arc, Mutex, MutexGuard};

use super::cpu::CpuDiarizer;
use super::cpu::CpuExecutor;
use super::executor::{DiarizerExecutor, DiarizerStream, SpeechError, SpeechExecutor, StreamOpen};
use super::{SpeechModels, Unavailable};
use crate::config::schema::DiarizerGroup;
use crate::download::Catalog;

/// The id `diarizer.model` names: `auto` is the catalog's diarizer entry
/// (none when it has none), `off` is none.
pub fn served_id(catalog: &Catalog, model: &str) -> Option<String> {
    match model {
        "off" => None,
        "auto" => catalog.entries().iter().find(|entry| entry.is_diarizer()).map(|entry| entry.id.clone()),
        id => Some(id.to_owned()),
    }
}

/// The diarizer's weights, once resolved.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiarizerFiles {
    pub id: String,
    pub weights: PathBuf,
}

/// The weights to serve with, or why there is no diarizer: `host` is what the
/// speech executor says of this CPU.
pub fn resolve(models: &SpeechModels, host: Result<(), String>, model: &str) -> Result<DiarizerFiles, Unavailable> {
    if model == "off" {
        return Err(Unavailable("the diarizer is turned off (`diarizer.model` is `off`)".to_owned()));
    }
    let Some(id) = served_id(&models.catalog, model) else {
        return Err(Unavailable("`diarizer.model` is `auto` and the model catalog has no diarizer entry".to_owned()));
    };
    let Some(entry) = models.catalog.entries().iter().find(|entry| entry.id == id) else {
        return Err(Unavailable(format!("`diarizer.model` names `{id}`, which is not in the model catalog")));
    };
    if !entry.is_diarizer() {
        return Err(Unavailable(format!("`diarizer.model` names `{id}`, which is not a diarizer: name a diarizer entry of the catalog")));
    }
    host.map_err(Unavailable)?;
    let weights = entry.artifact_path(&models.dir);
    if !entry.files.iter().all(|file| models.dir.join(&file.name).is_file()) {
        return Err(Unavailable(format!("no diarizer files for `{id}` under download.path; fetch them with `ignis-server model download {id}`")));
    }
    Ok(DiarizerFiles { id, weights })
}

/// The diarizer a server serves from its catalog: `diarizer.model`'s entry,
/// with the speech model's CPU executor's ISA, on a pool of its own.
pub struct CatalogDiarizer {
    models: SpeechModels,
    speech: Arc<CpuExecutor>,
    /// `diarizer.model` as the live configuration last said.
    model: Mutex<String>,
    /// The weights in use, and the adapter that holds them.
    current: Mutex<Option<(PathBuf, CpuDiarizer)>>,
}

impl CatalogDiarizer {
    pub fn new(models: SpeechModels, speech: Arc<CpuExecutor>) -> Self {
        Self { models, speech, model: Mutex::new(crate::config::DEFAULT_DIARIZER_MODEL.to_owned()), current: Mutex::new(None) }
    }

    fn model(&self) -> String {
        self.model.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    fn current(&self) -> MutexGuard<'_, Option<(PathBuf, CpuDiarizer)>> {
        self.current.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn resolve(&self) -> Result<DiarizerFiles, Unavailable> {
        resolve(&self.models, self.speech.availability(), &self.model())
    }
}

impl DiarizerExecutor for CatalogDiarizer {
    fn id(&self) -> Option<String> {
        served_id(&self.models.catalog, &self.model())
    }

    fn follow(&self, group: &DiarizerGroup) {
        let changed = std::mem::replace(&mut *self.model.lock().unwrap_or_else(|e| e.into_inner()), group.model.clone()) != group.model;
        if changed {
            // The weights of the model that was named give their memory back.
            if let Some((_, old)) = self.current().take() {
                old.unload();
            }
        }
    }

    fn availability(&self) -> Result<(), String> {
        self.resolve().map(|_| ()).map_err(|unavailable| unavailable.0)
    }

    fn open(&self, open: StreamOpen) -> Result<Box<dyn DiarizerStream>, SpeechError> {
        let files = self.resolve().map_err(|unavailable| SpeechError(unavailable.0))?;
        let mut current = self.current();
        if current.as_ref().is_none_or(|(weights, _)| *weights != files.weights) {
            *current = Some((files.weights.clone(), CpuDiarizer::new(files.weights, false, Arc::clone(&self.speech))));
        }
        let (_, diarizer) = current.as_ref().expect("the adapter was just set");
        diarizer.open(open)
    }

    fn unload(&self) {
        if let Some((_, diarizer)) = self.current().as_ref() {
            diarizer.unload();
        }
    }

    fn is_loaded(&self) -> bool {
        self.current().as_ref().is_some_and(|(_, diarizer)| diarizer.is_loaded())
    }

    fn loaded_weights(&self) -> Option<PathBuf> {
        self.current().as_ref().filter(|(_, diarizer)| diarizer.is_loaded()).map(|(weights, _)| weights.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::download::catalog::{CatalogEntry, CatalogFile, CatalogLayer};
    use crate::speech::tests::{fixture_models, tempfile_dir};
    use ignis_core::compute::ModelKind;

    /// The speech fixtures' catalog with a diarizer entry of our own beside
    /// them, not the built-in one, its GGUF named `<id>.gguf`.
    fn models(dir: &std::path::Path, ids: &[&str]) -> SpeechModels {
        let mut models = fixture_models(dir);
        let mut entries = models.catalog.entries().to_vec();
        for id in ids {
            entries.push(CatalogEntry {
                id: (*id).to_owned(),
                repo: "test/diarizer".to_owned(),
                revision: "0".repeat(40),
                artifact: format!("{id}.gguf"),
                card_class: None,
                family: None,
                aliases: Vec::new(),
                kind: Some(ModelKind::Diarizer),
                files: vec![CatalogFile { name: format!("{id}.gguf"), bytes: 4, sha256: "0".repeat(64) }],
                layer: CatalogLayer::Operator,
            });
        }
        models.catalog = Arc::new(Catalog::from_entries(entries));
        models
    }

    fn reason(models: &SpeechModels, host: Result<(), &str>, model: &str) -> String {
        resolve(models, host.map_err(str::to_owned), model).unwrap_err().0
    }

    /// AC 14: `auto` is the catalog's diarizer entry, an id is its entry,
    /// `off` none; `auto` with no diarizer in the catalog is none, saying so.
    #[test]
    fn diarizer_model_resolves_auto_an_id_and_off() {
        let dir = tempfile_dir("diarizer-auto");
        let with = models(&dir, &["acme-diarizer"]);
        assert_eq!(served_id(&with.catalog, "auto").as_deref(), Some("acme-diarizer"));
        assert_eq!(served_id(&with.catalog, "other").as_deref(), Some("other"));
        assert_eq!(served_id(&with.catalog, "off"), None);
        let without = fixture_models(&dir);
        assert_eq!(served_id(&without.catalog, "auto"), None);
        assert!(reason(&without, Ok(()), "auto").contains("no diarizer entry"));
        std::fs::write(dir.join("acme-diarizer.gguf"), b"gguf").unwrap();
        assert_eq!(resolve(&with, Ok(()), "auto"), Ok(DiarizerFiles { id: "acme-diarizer".to_owned(), weights: dir.join("acme-diarizer.gguf") }));
    }

    /// AC 14: each unavailable state reports its reason, in the order a
    /// person fixes them: off, an id the catalog lacks, an entry that is not
    /// a diarizer, a host that cannot run it (downloading cannot fix it),
    /// files missing, naming the command that fetches them.
    #[test]
    fn each_unavailable_state_reports_its_reason_in_order() {
        let dir = tempfile_dir("diarizer-reasons");
        let models = models(&dir, &["acme-diarizer"]);
        let no_avx2 = Err("the CPU lacks AVX2, which the speech model needs");
        let off = reason(&models, no_avx2, "off");
        assert!(off.contains("turned off") && off.contains("diarizer.model"), "{off}");
        let unknown = reason(&models, no_avx2, "nemotron-x");
        assert!(unknown.contains("`nemotron-x`") && unknown.contains("not in the model catalog"), "{unknown}");
        let wrong = reason(&models, no_avx2, "qwen3-asr-1.7b");
        assert!(wrong.contains("`qwen3-asr-1.7b`") && wrong.contains("not a diarizer"), "{wrong}");
        assert!(reason(&models, no_avx2, "acme-diarizer").contains("AVX2"));
        let files = reason(&models, Ok(()), "acme-diarizer");
        assert!(files.contains("no diarizer files") && files.contains("model download acme-diarizer"), "{files}");
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 0, "asking downloaded nothing");
    }

    /// AC 15, and a live change: the executor follows `diarizer.model`, its
    /// state is computed where it is read, and a change of the model gives the
    /// old weights' memory back.
    #[test]
    fn the_executor_follows_the_live_model() {
        let dir = tempfile_dir("diarizer-follow");
        let diarizer = CatalogDiarizer::new(models(&dir, &["acme-diarizer", "acme-other"]), Arc::new(CpuExecutor::new()));
        let group = |model: &str| DiarizerGroup { model: model.to_owned(), ..DiarizerGroup::default() };
        assert_eq!(diarizer.id().as_deref(), Some("acme-diarizer"), "auto");
        diarizer.follow(&group("acme-other"));
        assert_eq!(diarizer.id().as_deref(), Some("acme-other"));
        diarizer.follow(&group("off"));
        assert_eq!((diarizer.id(), diarizer.is_loaded()), (None, false));
        assert!(diarizer.availability().unwrap_err().contains("turned off"));
        diarizer.follow(&group("acme-other"));
        if CpuExecutor::new().availability().is_ok() {
            let missing = diarizer.availability().unwrap_err();
            assert!(missing.contains("model download acme-other"), "{missing}");
            std::fs::write(dir.join("acme-other.gguf"), b"gguf").unwrap();
            assert_eq!(diarizer.availability(), Ok(()));
        }
        assert!(!diarizer.is_loaded(), "asking loads nothing");
        diarizer.unload();
    }
}
