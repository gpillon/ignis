//! A fork's push publishes nothing (GitHub #495). A fork that keeps Actions
//! on runs this repository's workflows on its own pushes: the tests and both
//! builds are its contributors' CI, but an image, a release, the `develop`
//! prerelease or the guide's deployment under the fork's name is a build
//! labelled as ignis's that ignis never checked. So every job or step that
//! publishes names the repository, `github.repository == 'gpillon/ignis'`,
//! as the first conjunct of its condition: of the job's `if`, of the step's
//! `if`, or of the `push` input that decides an image push.
//!
//! What publishes is read off the workflow, not listed by job name, so a new
//! publishing step or workflow is held to the guard as it lands:
//! - a job granting `contents: write` or `pages: write` (a release, a tag, a
//!   Pages deployment);
//! - a step using a registry login, a release action or a Pages action;
//! - a step whose `run` calls `gh release` or pushes with git, docker or
//!   podman;
//! - a `push` input that is not plainly `false`.

use std::path::{Path, PathBuf};

use serde_yaml::{Mapping, Value};

/// The guard every publishing job or step starts its condition with.
const GUARD: &str = "github.repository == 'gpillon/ignis'";

/// Actions whose step publishes.
const PUBLISHING_ACTIONS: [&str; 4] =
    ["docker/login-action", "softprops/action-gh-release", "actions/deploy-pages", "actions/upload-pages-artifact"];

/// Commands whose step publishes.
const PUBLISHING_COMMANDS: [&str; 4] = ["gh release ", "git push", "docker push", "podman push"];

/// Permissions whose job publishes.
const PUBLISHING_PERMISSIONS: [&str; 2] = ["contents", "pages"];

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn read(path: &Path) -> String {
    std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// A condition as a run evaluates it: `${{ }}` taken off, whitespace (a
/// folded `>-` block's line breaks among it) collapsed to single spaces.
fn condition(value: &Value) -> String {
    let text = match value {
        Value::String(text) => text.clone(),
        Value::Bool(flag) => flag.to_string(),
        other => format!("{other:?}"),
    };
    let text = text.trim();
    let text = text.strip_prefix("${{").and_then(|t| t.strip_suffix("}}")).unwrap_or(text);
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Whether `value` starts with the guard as a conjunct of its own: the guard
/// alone, or the guard `&&` the rest. A guard behind an `||` guards nothing.
fn guarded(value: Option<&Value>) -> bool {
    let Some(value) = value else { return false };
    let condition = condition(value);
    condition == GUARD || condition.starts_with(&format!("{GUARD} && "))
}

fn get<'a>(map: &'a Mapping, key: &str) -> Option<&'a Value> {
    map.get(Value::String(key.to_owned()))
}

/// What a step publishes, if it does, and the condition that must carry the
/// guard besides the step's own `if` (an image push's `push` input).
fn publishes(step: &Mapping) -> Option<(String, Option<&Value>)> {
    if let Some(uses) = get(step, "uses").and_then(Value::as_str) {
        if let Some(action) = PUBLISHING_ACTIONS.iter().find(|action| uses.starts_with(&format!("{action}@"))) {
            return Some(((*action).to_owned(), None));
        }
    }
    if let Some(run) = get(step, "run").and_then(Value::as_str) {
        if let Some(command) = PUBLISHING_COMMANDS.iter().find(|command| run.contains(*command)) {
            return Some((format!("`{}`", command.trim()), None));
        }
    }
    let push = get(step, "with").and_then(Value::as_mapping).and_then(|with| get(with, "push"))?;
    (condition(push) != "false").then(|| ("the `push` input".to_owned(), Some(push)))
}

/// Every publishing job or step of `workflow` that does not name the
/// repository, each as `<file>: <job> ...`, and how many publishing points
/// the workflow holds in all.
fn unguarded(file: &str, workflow: &str) -> (Vec<String>, usize) {
    let document: Value = serde_yaml::from_str(workflow).unwrap_or_else(|e| panic!("{file} is YAML: {e}"));
    let jobs = get(document.as_mapping().expect("a workflow is a mapping"), "jobs").and_then(Value::as_mapping);
    let (mut refused, mut points) = (Vec::new(), 0);
    for (name, job) in jobs.into_iter().flatten() {
        let (name, job) = (name.as_str().unwrap_or("?"), job.as_mapping().expect("a job is a mapping"));
        let job_guarded = guarded(get(job, "if"));
        let permissions = get(job, "permissions").and_then(Value::as_mapping);
        for permission in PUBLISHING_PERMISSIONS {
            if permissions.and_then(|p| get(p, permission)).and_then(Value::as_str) == Some("write") {
                points += 1;
                if !job_guarded {
                    refused.push(format!("{file}: job `{name}` grants `{permission}: write` and its `if` does not start with {GUARD}"));
                }
            }
        }
        let steps = get(job, "steps").and_then(Value::as_sequence);
        for (index, step) in steps.into_iter().flatten().enumerate() {
            let step = step.as_mapping().expect("a step is a mapping");
            let Some((what, input)) = publishes(step) else { continue };
            points += 1;
            if job_guarded || guarded(get(step, "if")) || guarded(input) {
                continue;
            }
            let label = get(step, "name").and_then(Value::as_str).map(str::to_owned).unwrap_or_else(|| format!("step {}", index + 1));
            refused.push(format!("{file}: job `{name}`, `{label}`: {what} publishes and nothing guarding it starts with {GUARD}"));
        }
    }
    (refused, points)
}

/// Every workflow in the repository, by file name.
fn workflows() -> Vec<(String, String)> {
    let dir = repo().join(".github/workflows");
    let mut found: Vec<(String, String)> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
        .map(|entry| entry.expect("a directory entry").path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "yml" || ext == "yaml"))
        .map(|path| (path.file_name().unwrap().to_string_lossy().into_owned(), read(&path)))
        .collect();
    found.sort();
    found
}

#[test]
fn every_publishing_job_and_step_names_the_repository() {
    let workflows = workflows();
    assert!(workflows.iter().any(|(file, _)| file == "release.yml"), "{workflows:?}");
    let mut refused = Vec::new();
    for (file, text) in &workflows {
        refused.extend(unguarded(file, text).0);
    }
    assert!(refused.is_empty(), "a fork's push would publish:\n{}", refused.join("\n"));
}

/// The check finds what it guards, so it cannot pass by finding nothing:
/// release.yml's registry login, image push, release step, prerelease
/// command and the two jobs that write contents; docs.yml's Pages upload,
/// its deployment and the job that writes Pages.
#[test]
fn the_check_finds_every_publishing_point_the_workflows_hold() {
    let workflows = workflows();
    let points = |name: &str| unguarded(name, &workflows.iter().find(|(file, _)| file == name).expect(name).1).1;
    assert_eq!(points("release.yml"), 6);
    assert_eq!(points("docs.yml"), 3);
}

/// A check that cannot fail is not a pass: each kind of publishing point,
/// unguarded or guarded behind an `||`, is refused, and guarded as the
/// first conjunct of its job's or its own condition, it is not.
#[test]
fn an_unguarded_publishing_point_is_refused() {
    let refused = |yaml: &str| unguarded("fixture.yml", yaml).0;
    let step = |body: &str| format!("jobs:\n  ship:\n    runs-on: ubuntu-latest\n    steps:\n{body}");
    for body in [
        "      - uses: docker/login-action@v3\n",
        "      - uses: softprops/action-gh-release@v2\n",
        "      - uses: actions/deploy-pages@v4\n",
        "      - uses: actions/upload-pages-artifact@v3\n",
        "      - run: gh release create develop dist/*\n",
        "      - run: |\n          git tag x\n          git push origin x\n",
        "      - uses: docker/build-push-action@v6\n        with:\n          push: ${{ github.ref == 'refs/heads/main' }}\n",
        "      - uses: docker/build-push-action@v6\n        with:\n          push: true\n",
        "      - if: github.repository == 'gpillon/ignis' || github.ref == 'refs/heads/main'\n        run: gh release create develop\n",
    ] {
        assert_eq!(refused(&step(body)).len(), 1, "{body}");
    }
    let job = "jobs:\n  ship:\n    runs-on: ubuntu-latest\n    permissions:\n      contents: write\n    steps:\n      - run: echo\n";
    assert_eq!(refused(job).len(), 1);
    assert!(refused(&job.replace("contents: write", "contents: read")).is_empty());

    for body in [
        "      - if: github.repository == 'gpillon/ignis'\n        uses: docker/login-action@v3\n",
        "      - if: github.repository == 'gpillon/ignis' && github.ref == 'refs/heads/main'\n        run: gh release create develop\n",
        "      - uses: docker/build-push-action@v6\n        with:\n          push: ${{ github.repository == 'gpillon/ignis' && github.ref == 'refs/heads/main' }}\n",
        "      - uses: docker/build-push-action@v6\n        with:\n          push: false\n",
        "      - uses: actions/checkout@v4\n",
    ] {
        assert!(refused(&step(body)).is_empty(), "{body}");
    }
    let guarded_job = job.replace("    permissions:", "    if: >-\n      github.repository == 'gpillon/ignis' &&\n      startsWith(github.ref, 'refs/tags/v')\n    permissions:");
    assert!(refused(&guarded_job).is_empty(), "{guarded_job}");
}
