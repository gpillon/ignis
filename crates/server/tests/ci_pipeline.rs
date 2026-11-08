//! The release workflow's shape since GitHub #552 (ADR 0032, amended
//! 2026-11-08): the builds run beside the tests, and nothing is published
//! before the tests pass; the CPU-only suite runs as shards whose count the
//! matrix and the shard script agree on; the Linux build compiles the kernel
//! leaf in a layer of its own.
//!
//! What publishes is read off the workflow, as in fork_publishing.rs, so a
//! publishing job added later is held to `needs: test` as it lands.

use std::path::PathBuf;

use serde_yaml::{Mapping, Value};

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn read(path: &str) -> String {
    let path = repo().join(path);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

fn workflow() -> Value {
    serde_yaml::from_str(&read(".github/workflows/release.yml")).unwrap_or_else(|e| panic!("release.yml is YAML: {e}"))
}

fn get<'a>(map: &'a Mapping, key: &str) -> Option<&'a Value> {
    map.get(Value::String(key.to_owned()))
}

fn jobs(workflow: &Value) -> &Mapping {
    let workflow = workflow.as_mapping().expect("a workflow is a mapping");
    get(workflow, "jobs").and_then(Value::as_mapping).expect("release.yml has jobs")
}

fn job<'a>(jobs: &'a Mapping, name: &str) -> &'a Mapping {
    get(jobs, name).and_then(Value::as_mapping).unwrap_or_else(|| panic!("release.yml has a `{name}` job"))
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

fn steps(job: &Mapping) -> Vec<&Mapping> {
    get(job, "steps")
        .and_then(Value::as_sequence)
        .map(|steps| steps.iter().map(|step| step.as_mapping().expect("a step is a mapping")).collect())
        .unwrap_or_default()
}

fn text<'a>(map: &'a Mapping, key: &str) -> Option<&'a str> {
    get(map, key).and_then(Value::as_str)
}

/// A condition as a run evaluates it: `${{ }}` taken off, whitespace
/// collapsed to single spaces.
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

/// Whether a job publishes: a permission it is granted to write (a release,
/// a tag, a package, Pages), a registry login, a release or Pages action, a
/// command that pushes or releases, or a `push` input that is not `false`.
fn publishes(job: &Mapping) -> bool {
    let writes = get(job, "permissions")
        .and_then(Value::as_mapping)
        .is_some_and(|permissions| permissions.values().any(|value| value.as_str() == Some("write")));
    writes
        || steps(job).iter().any(|step| {
            let uses = text(step, "uses").unwrap_or("");
            let run = text(step, "run").unwrap_or("");
            let push = get(step, "with").and_then(Value::as_mapping).and_then(|with| get(with, "push"));
            ["docker/login-action@", "softprops/action-gh-release@", "actions/deploy-pages@", "actions/upload-pages-artifact@"]
                .iter()
                .any(|action| uses.starts_with(action))
                || ["gh release ", "git push", "docker push", "podman push", "skopeo copy"].iter().any(|command| run.contains(command))
                || push.is_some_and(|push| condition(push) != "false")
        })
}

#[test]
fn the_builds_run_beside_the_tests_and_whatever_publishes_waits_for_them() {
    let workflow = workflow();
    let jobs = jobs(&workflow);
    for build in ["linux", "windows"] {
        let job = job(jobs, build);
        assert!(!needs(job).iter().any(|need| need == "test"), "the {build} build waits for the tests: {:?}", needs(job));
        assert!(!publishes(job), "the {build} build publishes, and runs before the tests have passed");
    }
    let mut publishing: Vec<&str> = jobs
        .iter()
        .filter(|(_, job)| publishes(job.as_mapping().expect("a job is a mapping")))
        .map(|(name, _)| name.as_str().expect("a job name"))
        .collect();
    publishing.sort();
    assert_eq!(publishing, ["image", "prerelease", "release"], "the jobs that publish");
    for name in publishing {
        assert!(needs(job(jobs, name)).iter().any(|need| need == "test"), "job `{name}` publishes without waiting for the tests");
    }
    // The image pushed is the one the `linux` job built, from its archive.
    let image = job(jobs, "image");
    assert!(needs(image).iter().any(|need| need == "linux"), "the image job needs the linux build it pushes");
    assert!(
        steps(image).iter().all(|step| !text(step, "uses").unwrap_or("").starts_with("docker/build-push-action@")),
        "the image job builds an image of its own instead of pushing the linux job's"
    );
    // The archive is written without tags: given several, it holds one image
    // per tag, and `skopeo copy` refuses an archive holding more than one.
    let archive = steps(job(jobs, "linux"))
        .into_iter()
        .filter(|step| text(step, "uses").unwrap_or("").starts_with("docker/build-push-action@"))
        .find_map(|step| {
            let with = get(step, "with").and_then(Value::as_mapping)?;
            text(with, "outputs").filter(|outputs| outputs.contains("type=oci")).map(|_| with)
        })
        .expect("the linux job writes the image to an OCI archive");
    assert!(get(archive, "tags").is_none(), "the OCI archive is given tags, and would hold one image per tag");
}

/// The publishing check reads what it must: a job that pushes from its own
/// steps without asking for the tests is found.
#[test]
fn the_check_finds_a_build_that_publishes() {
    let job = |yaml: &str| -> Mapping { serde_yaml::from_str(yaml).expect("a job") };
    assert!(publishes(&job("permissions:\n  packages: write\n")));
    assert!(publishes(&job("steps:\n  - run: skopeo copy --all oci-archive:x docker://y\n")));
    assert!(publishes(&job("steps:\n  - uses: docker/build-push-action@v6\n    with:\n      push: ${{ github.ref == 'refs/heads/main' }}\n")));
    assert!(publishes(&job("steps:\n  - uses: docker/login-action@v3\n")));
    assert!(!publishes(&job("permissions:\n  contents: read\nsteps:\n  - uses: docker/build-push-action@v6\n    with:\n      push: false\n")));
}

/// The value of a `${{ matrix.<key> }}`-free list in the test job's matrix.
fn matrix_list<'a>(job: &'a Mapping, key: &str) -> Vec<&'a Value> {
    let strategy = get(job, "strategy").and_then(Value::as_mapping).expect("the test job has a strategy");
    let matrix = get(strategy, "matrix").and_then(Value::as_mapping).expect("the test job has a matrix");
    get(matrix, key).and_then(Value::as_sequence).unwrap_or_else(|| panic!("the test matrix lists `{key}`")).iter().collect()
}

#[test]
fn the_test_job_runs_every_shard_of_the_count_it_gives_the_script() {
    let workflow = workflow();
    let jobs = jobs(&workflow);
    let test = job(jobs, "test");
    let hosts: Vec<&str> = matrix_list(test, "host").into_iter().map(|host| host.as_str().expect("a host")).collect();
    assert_eq!(hosts, ["linux", "windows"], "the suite runs on both hosts");
    let shards: Vec<u64> = matrix_list(test, "shard").into_iter().map(|shard| shard.as_u64().expect("a shard number")).collect();
    let count = shards.len() as u64;
    assert!(count >= 1);
    assert_eq!(shards, (1..=count).collect::<Vec<_>>(), "the matrix lists shards 1 to {count}, each once");

    let runs: Vec<&str> = steps(test).iter().filter_map(|step| text(step, "run")).collect();
    let script = format!("bash mk/test-shard.sh ${{{{ matrix.shard }}}} {count} ${{{{ matrix.target }}}}");
    assert_eq!(
        runs.iter().filter(|run| run.trim() == script).count(),
        1,
        "the test job runs `{script}`, the matrix's count: {runs:?}"
    );
    assert!(
        !runs.iter().any(|run| run.contains("cargo test")),
        "the test job runs the suite through mk/test-shard.sh alone, or a part runs twice"
    );
    let name = text(test, "name").expect("the test job is named");
    assert!(name.contains(&format!("/{count})")), "the job's name gives the count: {name}");

    // The selftests and the Playground's checks run once per host, in shard 1,
    // the shard script's own test among them.
    for selftest in ["bash mk/test-shard-selftest.sh", "bash mk/cuda-build-check-selftest.sh", "bash mk/linux/doctor-selftest.sh"] {
        let step = steps(test).into_iter().find(|step| text(step, "run").map(str::trim) == Some(selftest));
        let step = step.unwrap_or_else(|| panic!("the test job runs `{selftest}`"));
        let when = get(step, "if").map(condition);
        assert_eq!(when.as_deref(), Some("matrix.shard == 1"), "`{selftest}` runs in shard 1 of each host");
    }
}

/// The Windows build's kernel/build cache (#552): ninja decides by date and
/// the checkout dates every source later than the cached objects, so the
/// cache only helped once the sources are dated back, and that is safe only
/// where they are the very bytes the objects were built from: an exact hit,
/// on exactly the files its key hashes, with no partial restore.
#[test]
fn the_windows_leaf_cache_dates_back_only_what_an_exact_hit_proves_unchanged() {
    let workflow = workflow();
    let jobs = jobs(&workflow);
    let windows = steps(job(jobs, "windows"));
    let position = |name: &str| {
        windows.iter().position(|step| text(step, "name") == Some(name)).unwrap_or_else(|| panic!("the windows job has a `{name}` step"))
    };
    let cache = windows[position("Cache the kernel leaf build")];
    assert_eq!(text(cache, "id"), Some("kernel-cache"));
    let with = get(cache, "with").and_then(Value::as_mapping).expect("the cache step has inputs");
    assert_eq!(text(with, "path"), Some("kernel/build"));
    assert!(get(with, "restore-keys").is_none(), "a partial restore would date back sources the cached objects were not built from");
    let key = text(with, "key").expect("the cache step has a key");
    let hashed: Vec<&str> = key
        .split("hashFiles(")
        .nth(1)
        .and_then(|rest| rest.split(')').next())
        .expect("the key hashes the kernel sources")
        .split(',')
        .map(|path| path.trim().trim_matches('\'').trim_end_matches("/**"))
        .collect();
    assert!(hashed.contains(&"kernel/src") && hashed.contains(&"kernel/vendor/src"), "{hashed:?}");

    let dating = windows[position("Date the kernel sources back to the cached build's")];
    assert_eq!(get(dating, "if").map(condition).as_deref(), Some("steps.kernel-cache.outputs.cache-hit == 'true'"));
    let run = text(dating, "run").expect("the dating step runs a command");
    for path in &hashed {
        assert!(run.contains(path), "the key hashes {path}, and the step does not date it back: {run}");
    }
    assert!(!run.contains("find kernel ") && !run.contains("find kernel\n"), "the step dates back more than the key hashes: {run}");
    assert!(position("Cache the kernel leaf build") < position("Date the kernel sources back to the cached build's"));
    assert!(position("Date the kernel sources back to the cached build's") < position("Build"), "the sources are dated back before the build");
}

/// The build stage of the Containerfile, up to the next stage.
fn build_stage(containerfile: &str) -> &str {
    containerfile.split(" AS build\n").nth(1).and_then(|rest| rest.split("\nFROM ").next()).expect("a build stage")
}

#[test]
fn the_container_build_compiles_the_leaf_in_a_layer_only_kernel_invalidates() {
    let containerfile = read("Containerfile");
    let stage = build_stage(&containerfile);
    let at = |needle: &str| stage.find(needle).unwrap_or_else(|| panic!("the build stage holds `{needle}`"));
    let copy_kernel = at("\nCOPY kernel/ kernel/\n");
    let build_leaf = at("\nRUN bash kernel/build.sh --archives-only\n");
    let copy_all = at("\nCOPY . .\n");
    let cargo = at("cargo build --release");
    assert!(copy_kernel < build_leaf && build_leaf < copy_all, "the leaf is built from kernel/ alone, before `COPY . .`");
    // What the leaf is built for is set before it is built, or the cargo
    // build's own run of kernel/build.sh would build it again for the list.
    for env in ["ENV IGNIS_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}\n", "ENV IGNIS_KERNEL_BUILD_JOBS=${KERNEL_BUILD_JOBS}\n"] {
        assert!(at(env) < build_leaf, "`{}` is set before the leaf is built", env.trim());
    }
    // The sources `COPY . .` rewrote are dated back before cargo runs ninja.
    let dated = at("find kernel -path kernel/build -prune -o -type f -exec touch -d @1 {} +");
    assert!(copy_all < dated && dated < cargo, "the kernel sources are dated back between `COPY . .` and the cargo build");
    // Only the shipped binaries leave the build step.
    assert!(stage.contains("&& rm -rf target"), "the cargo build step keeps its target directory in the layer");
    for binary in ["ignis-server", "ignis-bench", "ignis-artifact-inspect"] {
        let copies = containerfile.matches(&format!("COPY --from=build /out/{binary} ")).count();
        assert_eq!(copies, 2, "the artifacts and runtime stages copy {binary} from /out");
    }
}
