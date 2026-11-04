//! The cargo build compiles the kernel leaf's two archives and nothing else
//! (GitHub #460). `build.rs` passes a switch to the host script, the script
//! turns it into `--target ignis_kernel ignis_vendor`, and `-Test` / `--test`
//! (and a bare hand run) keep building every CTest executable.
//!
//! No toolchain here: this reads the three files and checks that the three
//! halves still agree, so the switch cannot disappear (or one spelling drift)
//! without `cargo test` going red. Whether nvcc really skips `kernel/tests/`
//! is checked by a `--features cuda` build, not by this test.

use std::path::PathBuf;

fn kernel_file(name: &str) -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../kernel").join(name);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

fn build_rs() -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("build.rs");
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

#[test]
fn build_rs_passes_the_archives_switch_on_both_hosts() {
    let rs = build_rs();
    assert!(rs.contains(r#"c.arg("-ArchivesOnly")"#), "build.rs must pass -ArchivesOnly to build.ps1");
    assert!(rs.contains(r#"c.arg("--archives-only")"#), "build.rs must pass --archives-only to build.sh");
    // The switch is for the cargo build alone: neither test flag may ride along.
    assert!(!rs.contains(r#""-Test""#) && !rs.contains(r#""--test""#));
}

#[test]
fn powershell_script_builds_only_the_two_archives_under_the_switch() {
    let ps = kernel_file("build.ps1");
    assert!(ps.contains("$args -contains '-ArchivesOnly'"));
    assert!(ps.contains("-not $RunTests"), "-Test must win over -ArchivesOnly");
    assert!(ps.contains("'--target', 'ignis_kernel', 'ignis_vendor'"));
    // The default build line carries no --target: a hand run builds everything.
    assert!(ps.contains("@('--build', $BuildPath, '--parallel')"));
}

#[test]
fn shell_script_builds_only_the_two_archives_under_the_switch() {
    let sh = kernel_file("build.sh");
    assert!(sh.contains("--archives-only) archives_only=1 ;;"));
    assert!(sh.contains(r#""$archives_only" = 1 ] && [ "$run_tests" = 0"#), "--test must win");
    assert!(sh.contains("build_targets=(--target ignis_kernel ignis_vendor)"));
}

#[test]
fn the_named_targets_exist_in_cmake() {
    let cmake = kernel_file("CMakeLists.txt");
    assert!(cmake.contains("add_library(ignis_kernel"));
    assert!(cmake.contains("add_library(ignis_vendor"));
    // The tests stay in the project, so the full build and CTest keep them.
    assert!(cmake.contains("add_subdirectory(tests)"));
}
