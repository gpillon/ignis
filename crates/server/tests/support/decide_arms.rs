//! The arms the two `/v1/decide` regression sets run on (GitHub #675): the
//! entailment set's `measurement.arms` (`fixtures/entailment/cases.json`),
//! which `decide_entailment_gpu.rs` and `decide_scalar_gpu.rs` load by name
//! and whose shapes `decide_entailment_fixture.rs` resolves on CPU. Included
//! via `#[path]`, like `entailment.rs`.
//!
//! An arm names every value that moves these reads: the exception weights,
//! the context and the rope table, the GDN prefill route, the prompt
//! attention band and prompt reuse. None reads a config file. A `harness`
//! arm is the shape a GPU test builds by hand ([`EngineShape::default`] with
//! the arm's values: no drafter, no vision, chunk 1,024); a `served` arm is a
//! start under the `rtx5090` profile with every one of those values named as
//! a flag, resolved with no file and no environment and loaded through the
//! server's own load path. Three of the five arms the set registered on
//! 2026-11-08 ran FP8 exception weights that a user config file set and
//! nothing named.
//!
//! `IGNIS_DECIDE_ARMS` picks the arms a GPU test runs: a comma list of arm
//! names, or `all`; unset, [`DEFAULT_ARMS`].

// Each binary reads a different subset (the CPU check loads nothing).
#![allow(dead_code)]

use std::path::PathBuf;

use ignis_core::{ExceptionWeights, GdnPrefill, PrefillAttention, RopeScaling, RopeSetting};
use ignis_server::runtime::EngineShape;
use serde::de::{Deserializer, Error as _};
use serde::Deserialize;

pub const ARTIFACT: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";
pub const MODEL: &str = "qwen3.8-27b";

/// The env var naming the arms a GPU test runs.
pub const ARMS_ENV: &str = "IGNIS_DECIDE_ARMS";

/// The arms a GPU test runs when `IGNIS_DECIDE_ARMS` names none: the
/// harness's, once under each exception weights.
pub const DEFAULT_ARMS: &[&str] = &["h-bf16", "h-fp8"];

/// The measurement switch that names the GDN prefill route, read by the leaf
/// at each load.
const GDN_PREFILL_ENV: &str = "IGNIS_GDN_PREFILL";

/// Where an arm's shape comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Surface {
    /// The shape a GPU test builds by hand.
    Harness,
    /// A start under the `rtx5090` profile, through the server's load path.
    Served,
}

/// One route the sets run on, every value that moves a read named.
#[derive(Debug, Deserialize)]
pub struct Arm {
    pub name: String,
    pub surface: Surface,
    #[serde(deserialize_with = "exception_weights")]
    pub exception_weights: ExceptionWeights,
    /// The context the load runs, in tokens: named, so it never adapts.
    pub max_context: u32,
    #[serde(deserialize_with = "rope_scaling")]
    pub rope_scaling: RopeScaling,
    #[serde(deserialize_with = "gdn_prefill")]
    pub gdn_prefill: GdnPrefill,
    #[serde(deserialize_with = "prefill_attention")]
    pub prefill_attention: PrefillAttention,
    pub prompt_reuse: bool,
    /// What to run to redo the arm.
    pub redo: String,
}

/// `--model-exception-weights`'s values.
fn exception_weights<'de, D: Deserializer<'de>>(deserializer: D) -> Result<ExceptionWeights, D::Error> {
    ExceptionWeights::parse(&String::deserialize(deserializer)?).map_err(D::Error::custom)
}

/// `--model-rope-scaling`'s spelled-out tables (`none`, `yarn:F`): an arm
/// names its table, so `auto` is not one.
fn rope_scaling<'de, D: Deserializer<'de>>(deserializer: D) -> Result<RopeScaling, D::Error> {
    RopeScaling::parse(&String::deserialize(deserializer)?).map_err(|e| D::Error::custom(e.to_string()))
}

/// `IGNIS_GDN_PREFILL`'s two values.
fn gdn_prefill<'de, D: Deserializer<'de>>(deserializer: D) -> Result<GdnPrefill, D::Error> {
    let name = String::deserialize(deserializer)?;
    [GdnPrefill::Ours, GdnPrefill::Vendored]
        .into_iter()
        .find(|route| route.as_str() == name)
        .ok_or_else(|| D::Error::custom(format!("unknown GDN prefill `{name}` (expected `ours` or `vendored`)")))
}

/// `--model-prefill-attention`'s values.
fn prefill_attention<'de, D: Deserializer<'de>>(deserializer: D) -> Result<PrefillAttention, D::Error> {
    PrefillAttention::parse(&String::deserialize(deserializer)?).map_err(D::Error::custom)
}

impl Arm {
    /// A served arm's start: the `rtx5090` profile, with every value that
    /// moves a read named as a flag, so neither the profile's `auto` nor a
    /// config file chooses one, and the loop guard off.
    pub fn flags(&self) -> Vec<String> {
        [
            "--profile",
            "rtx5090",
            "--model-exception-weights",
            self.exception_weights.as_str(),
            "--model-max-context",
            &self.max_context.to_string(),
            "--model-rope-scaling",
            &self.rope_scaling.to_string(),
            "--model-prefill-attention",
            self.prefill_attention.as_str(),
            "--reuse-prompt",
            if self.prompt_reuse { "on" } else { "off" },
            "--model-loop-guard",
            "off",
        ]
        .iter()
        .map(|flag| flag.to_string())
        .collect()
    }

    /// The 27B's options a served arm's start resolves `flags` to, with no
    /// config file and no environment, served under [`MODEL`].
    pub fn config(&self, flags: &[String]) -> ignis_server::config::Config {
        let ignis_server::config::ConfigOutcome::Config(config) = ignis_server::config::resolve(flags, |_| None)
            .unwrap_or_else(|e| panic!("arm {}: resolve {flags:?}: {e}", self.name))
        else {
            panic!("arm {}: {flags:?} is not a start", self.name);
        };
        let config = config
            .for_family(ignis_core::compute::ModelFamily::Qwen38_27b)
            .unwrap_or_else(|e| panic!("arm {}: the 27B from {flags:?}: {e}", self.name));
        ignis_server::config::Config { model: MODEL.to_owned(), model_named: true, ..config }
    }

    /// The shape the arm loads: what the CPU check holds to the values it
    /// names.
    pub fn shape(&self) -> EngineShape {
        match self.surface {
            Surface::Harness => EngineShape {
                exception_weights: self.exception_weights,
                max_context: self.max_context,
                rope_scaling: self.rope_scaling,
                rope_setting: RopeSetting::Fixed(self.rope_scaling),
                prefill_attention: self.prefill_attention,
                prompt_reuse: self.prompt_reuse,
                ..EngineShape::default()
            },
            Surface::Served => ignis_server::load::load_shape(&self.config(&self.flags()), None),
        }
    }
}

/// The arms registered with the entailment set, which the scalar set runs
/// on too.
pub fn registered() -> Vec<Arm> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests").join("fixtures").join("entailment").join("cases.json");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let fixture: serde_json::Value = serde_json::from_str(&text).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    serde_json::from_value(fixture["measurement"]["arms"].clone()).unwrap_or_else(|e| panic!("{}: arms: {e}", path.display()))
}

/// The arms `IGNIS_DECIDE_ARMS` names among `arms`, in its order (`all`:
/// every one); unset or empty, `default`. A name no arm has fails, so a typo
/// runs nothing silently.
pub fn selected<'a>(arms: &'a [Arm], default: &[&str]) -> Vec<&'a Arm> {
    let named = std::env::var(ARMS_ENV).ok().filter(|value| !value.trim().is_empty());
    let names: Vec<String> = match named.as_deref().map(str::trim) {
        Some("all") => return arms.iter().collect(),
        Some(list) => list.split(',').map(|name| name.trim().to_owned()).collect(),
        None => default.iter().map(|name| name.to_string()).collect(),
    };
    names
        .iter()
        .map(|name| {
            arms.iter().find(|arm| arm.name == *name).unwrap_or_else(|| {
                let known: Vec<&str> = arms.iter().map(|arm| arm.name.as_str()).collect();
                panic!("{ARMS_ENV}: no arm `{name}` (the arms: {})", known.join(", "))
            })
        })
        .collect()
}

#[cfg(feature = "cuda")]
pub mod gpu {
    use std::path::Path;
    use std::sync::Arc;
    use std::time::Duration;

    use ignis_artifact::{FrontendSet, Reader};
    use ignis_core::gpu_profile;
    use ignis_server::Server;
    use ignis_server::engine::Engine;
    use ignis_server::runtime::{EngineShape, cuda_scheduler};
    use ignis_server::telemetry::SystemClock;

    use super::{ARTIFACT, Arm, GDN_PREFILL_ENV, MODEL, Surface};

    /// The 27B loaded on an arm: the router over it, and what its VRAM plan
    /// reserved. Dropping it shuts the model thread down and joins it, which
    /// releases the weights before the next load.
    pub struct Loaded {
        app: Option<axum::Router>,
        model: Option<ignis_server::ActiveModel>,
        driver: Option<std::thread::JoinHandle<()>>,
        pub reservations: Option<ignis_server::metrics::LoadReservations>,
    }

    impl Loaded {
        pub fn app(&self) -> &axum::Router {
            self.app.as_ref().expect("the app is only taken by Drop")
        }
    }

    impl Drop for Loaded {
        fn drop(&mut self) {
            drop(self.app.take());
            if let Some(model) = self.model.take() {
                model.shut_down();
            }
            if let Some(driver) = self.driver.take() {
                let _ = driver.join();
            }
        }
    }

    impl Arm {
        /// The 27B loaded on this arm, its GDN prefill route named for the
        /// load and `speculation` beside it when given; `None` where the
        /// profile allows a skip.
        pub fn load(&self, speculation: Option<ignis_core::Speculation>) -> Option<Loaded> {
            let path = Path::new(ARTIFACT);
            if !path.exists() && gpu_profile::skip_or_fail(&format!("artifact absent: {ARTIFACT}")) {
                return None;
            }
            // SAFETY: the GPU tests run one at a time (`--test-threads=1`) and
            // the previous arm's model thread is joined; the leaf reads the
            // switch from the operating system's block at each load.
            unsafe { std::env::set_var(GDN_PREFILL_ENV, self.gdn_prefill.as_str()) };
            println!("arm {}: {:?} weights={} context={} rope={} gdn={} attention={} reuse={}",
                self.name, self.surface, self.exception_weights, self.max_context, self.rope_scaling,
                self.gdn_prefill.as_str(), self.prefill_attention, self.prompt_reuse);
            match self.surface {
                Surface::Harness => self.harness(path, speculation),
                Surface::Served => self.served(path, speculation),
            }
        }

        fn harness(&self, path: &Path, speculation: Option<ignis_core::Speculation>) -> Option<Loaded> {
            let reader = Reader::open(path).unwrap_or_else(|e| panic!("open artifact: {e}"));
            let frontend = FrontendSet::from_reader(&reader).unwrap_or_else(|e| panic!("frontend: {e}"));
            let eos = frontend
                .eos_token_id()
                .unwrap_or_else(|| panic!("qwen3.8-27b generation config must carry eos_token_id"));
            let shape = EngineShape { speculation, ..self.shape() };
            let (scheduler, reservations) = match cuda_scheduler(path, MODEL.into(), eos, shape) {
                Ok(loaded) => loaded,
                Err(e) => {
                    gpu_profile::skip_or_fail(&format!("arm {}: cuda_scheduler: {e}", self.name));
                    return None;
                }
            };
            let (engine, driver) = Engine::with_clock_and_driver(Box::new(scheduler), Arc::new(SystemClock));
            let provider = ignis_server::artifact_template::ArtifactTemplateProvider::new(frontend);
            let server = Server::new(engine, Box::new(provider)).with_request_timeout(Duration::from_secs(180));
            Some(Loaded { app: Some(server.app()), model: None, driver: Some(driver), reservations: Some(reservations) })
        }

        fn served(&self, path: &Path, speculation: Option<ignis_core::Speculation>) -> Option<Loaded> {
            let mut flags = self.flags();
            if let Some(speculation) = speculation {
                flags.extend(
                    ["--spec-backend", speculation.backend().as_str(), "--spec-draft-tokens", &speculation.draft_tokens().to_string()]
                        .map(String::from),
                );
            }
            let config = self.config(&flags);
            let loaded = match ignis_server::load::prepare_model(&config, path).and_then(ignis_server::load::load_model) {
                Ok(loaded) => loaded,
                Err(e) => {
                    gpu_profile::skip_or_fail(&format!("arm {}: load the 27B from {flags:?}: {e}", self.name));
                    return None;
                }
            };
            // Its context, rope and weights are named, and so is nothing
            // the load adapted: one that did is not this arm.
            assert!(loaded.model.adapted.is_empty(), "arm {}: the load adapted: {}", self.name, loaded.model.adapted);
            let server = Server::from_active(loaded.model.clone()).with_request_timeout(Duration::from_secs(180));
            Some(Loaded { app: Some(server.app()), model: Some(loaded.model), driver: None, reservations: loaded.reservations })
        }
    }
}
