//! Packs the `ignis` skill directory into the `tar.gz` the server serves
//! (spec #656): shared by `build.rs`, which embeds the result, and by the
//! tests that check it, so both run the one packer.
//!
//! Deterministic: the same tree gives the same bytes, wherever and whenever
//! it is packed. Every entry sits under `ignis/`, entries are sorted by path
//! (a directory before what it holds), and mtime, owner and modes are fixed.
//! The archive is ustar, written by hand: a path over 100 bytes is refused.

use std::fs;
use std::io::Write;
use std::path::Path;

use flate2::write::GzEncoder;
use flate2::Compression;

/// The skill's directory name, and so the one folder an extraction makes.
pub const NAME: &str = "ignis";

/// Every entry's mtime: 2026-01-01T00:00:00Z.
const MTIME: u64 = 1_767_225_600;

/// Every file under `root` as `(path below the archive's root, bytes)`,
/// sorted by path; `root` is the skill directory, and each path starts
/// `ignis/`.
pub fn files(root: &Path) -> Vec<(String, Vec<u8>)> {
    fn walk(dir: &Path, prefix: &str, files: &mut Vec<(String, Vec<u8>)>) {
        for entry in fs::read_dir(dir).unwrap_or_else(|e| panic!("read {}: {e}", dir.display())) {
            let entry = entry.expect("read a skill directory entry");
            let path = format!("{prefix}/{}", entry.file_name().to_string_lossy());
            if entry.path().is_dir() {
                walk(&entry.path(), &path, files);
            } else {
                let bytes = fs::read(entry.path()).unwrap_or_else(|e| panic!("read {}: {e}", entry.path().display()));
                files.push((path, bytes));
            }
        }
    }
    let mut files = Vec::new();
    walk(root, NAME, &mut files);
    files.sort();
    files
}

/// The gzip'd tar of the skill directory `root`.
pub fn pack(root: &Path) -> Vec<u8> {
    let files = files(root);
    // Directories, each before its files: the paths' prefixes, sorted.
    let mut directories: Vec<String> = vec![NAME.to_owned()];
    for (path, _) in &files {
        let mut parent = path.as_str();
        while let Some((above, _)) = parent.rsplit_once('/') {
            if !directories.iter().any(|known| known == above) {
                directories.push(above.to_owned());
            }
            parent = above;
        }
    }
    let mut entries: Vec<(String, Option<&[u8]>)> = directories.into_iter().map(|dir| (format!("{dir}/"), None)).collect();
    entries.extend(files.iter().map(|(path, bytes)| (path.clone(), Some(bytes.as_slice()))));
    entries.sort_by(|a, b| a.0.cmp(&b.0));

    let mut tar = Vec::new();
    for (path, content) in entries {
        let bytes = content.unwrap_or_default();
        tar.extend_from_slice(&header(&path, bytes.len(), content.is_none()));
        tar.extend_from_slice(bytes);
        tar.resize(tar.len().div_ceil(512) * 512, 0);
    }
    tar.extend_from_slice(&[0; 1024]);

    let mut gzip = GzEncoder::new(Vec::new(), Compression::best());
    gzip.write_all(&tar).expect("compress the skill");
    gzip.finish().expect("finish the skill's gzip")
}

/// One ustar header block.
fn header(path: &str, size: usize, directory: bool) -> [u8; 512] {
    assert!(path.len() <= 100, "{path} is over 100 bytes: the packer writes no long names");
    let mut block = [0u8; 512];
    block[..path.len()].copy_from_slice(path.as_bytes());
    let mode = if directory { 0o755 } else { 0o644 };
    octal(&mut block[100..108], mode);
    octal(&mut block[108..116], 0); // uid
    octal(&mut block[116..124], 0); // gid
    octal(&mut block[124..136], size as u64);
    octal(&mut block[136..148], MTIME);
    block[156] = if directory { b'5' } else { b'0' };
    block[257..263].copy_from_slice(b"ustar\0");
    block[263..265].copy_from_slice(b"00");
    // The checksum counts its own field as spaces.
    block[148..156].fill(b' ');
    let sum: u32 = block.iter().map(|byte| u32::from(*byte)).sum();
    octal(&mut block[148..155], u64::from(sum));
    block[155] = b' ';
    block
}

/// `value` as zero-padded octal filling `field` but for a closing NUL.
fn octal(field: &mut [u8], value: u64) {
    let digits = field.len() - 1;
    let text = format!("{value:0digits$o}");
    assert!(text.len() == digits, "{value} does not fit a {digits}-digit field");
    field[..digits].copy_from_slice(text.as_bytes());
    field[digits] = 0;
}
