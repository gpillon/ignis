//! The release workflow's `macos` job checks that the workspace compiles on a
//! Mac and never blocks anything (GitHub #496). macOS is not a target: ignis
//! serves on CUDA cards, under Windows and Linux. The job keeps the workspace
//! compiling there for the Metal study (#439), so its failure shows in the
//! run without failing it: a `ci/**` dry run stays the signal a tag is cut
//! on, a tag's run skips the job, so the `success` `docs.yml` waits for
//! before it deploys the guide never hangs on it, and no build or publish
//! waits on it.

use std::path::PathBuf;

use serde_yaml::{Mapping, Value};

fn workflow() -> Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.github/workflows/release.yml");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    serde_yaml::from_str(&text).unwrap_or_else(|e| panic!("release.yml is YAML: {e}"))
}

fn get<'a>(map: &'a Mapping, key: &str) -> Option<&'a Value> {
    map.get(Value::String(key.to_owned()))
}

fn jobs(workflow: &Value) -> &Mapping {
    let workflow = workflow.as_mapping().expect("a workflow is a mapping");
    get(workflow, "jobs").and_then(Value::as_mapping).expect("release.yml has jobs")
}

/// A condition as a run evaluates it: `${{ }}` taken off, whitespace
/// collapsed to single spaces.
fn condition(value: &Value) -> String {
    let text = value.as_str().expect("a condition is a string").trim();
    let text = text.strip_prefix("${{").and_then(|t| t.strip_suffix("}}")).unwrap_or(text);
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// The job names a job's `needs` lists, written as one name or a list.
fn needs(job: &Mapping) -> Vec<String> {
    match get(job, "needs") {
        None => Vec::new(),
        Some(Value::String(name)) => vec![name.clone()],
        Some(Value::Sequence(names)) => names.iter().map(|name| name.as_str().expect("a job name").to_owned()).collect(),
        Some(other) => panic!("needs: {other:?}"),
    }
}

#[test]
fn the_macos_job_checks_the_workspace_without_cuda_and_blocks_nothing() {
    let workflow = workflow();
    let jobs = jobs(&workflow);
    let job = get(jobs, "macos").and_then(Value::as_mapping).expect("release.yml has a `macos` job");

    let runner = get(job, "runs-on").and_then(Value::as_str).expect("the macos job names its runner");
    assert!(runner.starts_with("macos-"), "the macos job runs on a macOS runner, not {runner}");
    assert_eq!(get(job, "continue-on-error"), Some(&Value::Bool(true)), "a failing macos job must not fail the run");
    assert!(needs(job).is_empty(), "the macos job waits on no job: {:?}", needs(job));
    for (name, other) in jobs {
        let other = other.as_mapping().expect("a job is a mapping");
        assert!(!needs(other).iter().any(|need| need == "macos"), "job {name:?} waits on the macos job");
    }
    let run_if = get(job, "if").map(condition);
    assert_eq!(run_if.as_deref(), Some("!startsWith(github.ref, 'refs/tags/')"), "the macos job skips tags");

    let runs: Vec<&str> = get(job, "steps")
        .and_then(Value::as_sequence)
        .expect("the macos job has steps")
        .iter()
        .filter_map(|step| step.as_mapping().and_then(|step| get(step, "run")).and_then(Value::as_str))
        .collect();
    let check = runs.iter().find(|run| run.contains("cargo check")).expect("the macos job runs cargo check");
    for flag in ["--workspace", "--all-targets", "--locked", "--target aarch64-apple-darwin"] {
        assert!(check.contains(flag), "the macos check passes {flag}: {check}");
    }
    assert!(!runs.iter().any(|run| run.contains("cargo test")), "the macos job checks, it runs nothing");
    assert!(!runs.iter().any(|run| run.contains("--features")), "the macos job checks the build without the cuda feature");
}
