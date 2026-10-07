//! A disposable, checksummed copy of the selected n-gram hot rows.
//! Cache failures never prevent loading the source artifact.
use sha2::{Digest, Sha256};
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

const MAGIC: &[u8; 8] = b"IGNGRAM1";
const HEADER_BYTES: u64 = 80;
static TEMP_ID: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PersistenceOptions {
    pub enabled: bool,
    /// None means the operating system's per-user cache directory (`auto`).
    pub path: Option<PathBuf>,
}
impl Default for PersistenceOptions {
    fn default() -> Self {
        Self { enabled: true, path: None }
    }
}
impl PersistenceOptions {
    pub fn directory(&self) -> Result<PathBuf, String> {
        if let Some(path) = &self.path {
            return Ok(path.clone());
        }
        auto_directory(cfg!(windows), |key| std::env::var_os(key).map(PathBuf::from))
    }
}
fn auto_directory(windows: bool, env: impl Fn(&str) -> Option<PathBuf>) -> Result<PathBuf, String> {
    let nonempty = |key| env(key).filter(|p| !p.as_os_str().is_empty());
    let base = if windows {
        nonempty("LOCALAPPDATA").ok_or("LOCALAPPDATA is unset")?
    } else {
        nonempty("XDG_CACHE_HOME")
            .filter(|p| p.is_absolute())
            .or_else(|| nonempty("HOME").map(|p| p.join(".cache")))
            .ok_or("XDG_CACHE_HOME and HOME are unset")?
    };
    Ok(if windows { base.join("ignis/cache/ngram") } else { base.join("ignis/ngram") })
}

/// Identity includes the source's filesystem stamp even when the packer has
/// recorded a digest: replacing/editing an artifact invalidates old copies.
/// Legacy artifacts without that digest rely on the stamp and descriptors.
/// Deliberate source edits preserving every stamp require source verification.
pub(crate) fn identity(
    path: &Path,
    directory_hash: [u8; 32],
    layout: &[u64],
    rows: &[u32],
) -> Result<[u8; 32], String> {
    let metadata = path.metadata().map_err(|e| e.to_string())?;
    let mut hash = Sha256::new();
    hash.update(MAGIC);
    hash.update(directory_hash);
    let canonical = path.canonicalize().map_err(|e| e.to_string())?;
    // Encoded OS bytes avoid lossy/non-UTF8 path collisions on either platform.
    hash.update(canonical.as_os_str().as_encoded_bytes());
    hash.update(metadata.len().to_le_bytes());
    for stamp in [metadata.modified(), metadata.created()] {
        if let Ok(stamp) = stamp {
            let duration = stamp.duration_since(std::time::UNIX_EPOCH).map_err(|e| e.to_string())?;
            hash.update(duration.as_secs().to_le_bytes());
            hash.update(duration.subsec_nanos().to_le_bytes());
        }
    }
    let sidecar = ignis_artifact::packer::sidecar_path(path);
    if let Ok(bytes) = std::fs::read(sidecar) {
        if let Ok(value) = serde_json::from_slice::<serde_json::Value>(&bytes) {
            if let Some(source) = value.get("ngram_cache_source") {
                hash.update(source.to_string().as_bytes());
            }
        }
    }
    for word in layout {
        hash.update(word.to_le_bytes());
    }
    for row in rows {
        hash.update(row.to_le_bytes());
    }
    Ok(hash.finalize().into())
}

fn read(path: &Path, key: &[u8; 32], len: usize) -> Result<Vec<u8>, String> {
    let mut file = File::open(path).map_err(|e| e.to_string())?;
    if file.metadata().map_err(|e| e.to_string())?.len() != HEADER_BYTES + len as u64 {
        return Err("wrong cache file length".into());
    }
    let mut header = [0u8; HEADER_BYTES as usize];
    file.read_exact(&mut header).map_err(|e| e.to_string())?;
    if &header[..8] != MAGIC
        || &header[8..40] != key
        || u64::from_le_bytes(header[40..48].try_into().unwrap()) != len as u64
    {
        return Err("cache version, identity or row count changed".into());
    }
    let mut bytes = vec![0; len];
    file.read_exact(&mut bytes).map_err(|e| e.to_string())?;
    if Sha256::digest(&bytes)[..] != header[48..80] {
        return Err("cache checksum mismatch".into());
    }
    Ok(bytes)
}
fn publish(path: &Path, key: &[u8; 32], bytes: &[u8]) -> Result<(), String> {
    let temp = path.with_extension(format!("{}.{}.tmp", std::process::id(), TEMP_ID.fetch_add(1, Ordering::Relaxed)));
    let result = (|| {
        let mut file = OpenOptions::new().write(true).create_new(true).open(&temp).map_err(|e| e.to_string())?;
        file.write_all(MAGIC)
            .and_then(|_| file.write_all(key))
            .and_then(|_| file.write_all(&(bytes.len() as u64).to_le_bytes()))
            .and_then(|_| file.write_all(&Sha256::digest(bytes)))
            .and_then(|_| file.write_all(bytes))
            .and_then(|_| file.sync_all())
            .map_err(|e| e.to_string())?;
        drop(file);
        std::fs::rename(&temp, path).map_err(|e| e.to_string())
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&temp);
    }
    result
}

pub(crate) fn load_or_build(
    options: &PersistenceOptions,
    key: Result<[u8; 32], String>,
    len: usize,
    build: impl FnOnce() -> Result<Vec<u8>, String>,
) -> Result<Vec<u8>, String> {
    if !options.enabled || len == 0 {
        return build();
    }
    let location = key.and_then(|key| options.directory().map(|dir| (dir, key)));
    let (dir, key) = match location {
        Ok(location) => location,
        Err(error) => {
            tracing::warn!(%error, "n-gram cache unavailable; reading the artifact");
            return build();
        }
    };
    let name: String = key.iter().map(|b| format!("{b:02x}")).collect();
    let path = dir.join(format!("{name}.bin"));
    match read(&path, &key, len) {
        Ok(bytes) => {
            tracing::info!(path = %path.display(), "n-gram cache hit");
            return Ok(bytes);
        }
        Err(error) if path.exists() => {
            tracing::warn!(%error, path = %path.display(), "invalid n-gram cache; rebuilding")
        }
        Err(_) => {}
    }
    // Hold an OS lock until publication. No waiting for another loader, and
    // a process crash releases it. The tiny lock file can safely remain.
    let lock = std::fs::create_dir_all(&dir).and_then(|_| {
        OpenOptions::new().read(true).write(true).create(true).truncate(false).open(path.with_extension("lock"))
    });
    let lock = match lock {
        Ok(file) => match file.try_lock() {
            Ok(()) => Some(file),
            Err(error) => {
                tracing::warn!(%error, "n-gram cache writer busy; reading the artifact");
                None
            }
        },
        Err(error) => {
            tracing::warn!(%error, "n-gram cache not writable; reading the artifact");
            None
        }
    };
    if lock.is_some() {
        if let Ok(bytes) = read(&path, &key, len) {
            return Ok(bytes);
        }
    }
    let bytes = build()?;
    if bytes.len() != len {
        return Err("n-gram loader returned the wrong byte count".into());
    }
    if lock.is_some() {
        match publish(&path, &key, &bytes) {
            Ok(()) => tracing::info!(path = %path.display(), bytes = len, "n-gram cache saved"),
            Err(error) => tracing::warn!(%error, "n-gram cache write failed; using loaded rows"),
        }
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    struct Temp(PathBuf);
    impl Temp {
        fn new() -> Self {
            let p = std::env::temp_dir().join(format!(
                "ignis-ngram-cache-{}-{}",
                std::process::id(),
                TEMP_ID.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir(&p).unwrap();
            Self(p)
        }
    }
    impl Drop for Temp {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    #[test]
    fn auto_paths_for_both_platforms() {
        assert_eq!(
            auto_directory(true, |k| (k == "LOCALAPPDATA").then(|| PathBuf::from("local"))).unwrap(),
            PathBuf::from("local/ignis/cache/ngram")
        );
        // Linux's cache base already includes `.cache`; its Ignis path does not add `cache`.
        assert_eq!(
            auto_directory(false, |k| (k == "HOME").then(|| PathBuf::from("home"))).unwrap(),
            PathBuf::from("home/.cache/ignis/ngram")
        );
        let abs = std::env::temp_dir();
        assert_eq!(
            auto_directory(false, |k| (k == "XDG_CACHE_HOME").then(|| abs.clone())).unwrap(),
            abs.join("ignis/ngram")
        );
        assert!(auto_directory(true, |_| None).is_err());
    }
    #[test]
    fn hit_corruption_truncation_and_version_rebuild() {
        let temp = Temp::new();
        let options = PersistenceOptions { enabled: true, path: Some(temp.0.clone()) };
        let key = [3; 32];
        let data = vec![7; 100];
        assert_eq!(load_or_build(&options, Ok(key), 100, || Ok(data.clone())).unwrap(), data);
        assert_eq!(load_or_build(&options, Ok(key), 100, || panic!("cache hit must not read source")).unwrap(), data);
        let path = temp.0.join(format!("{}.bin", "03".repeat(32)));
        for bad in [vec![0; 180], vec![0; 12], {
            let mut b = std::fs::read(&path).unwrap();
            b[100] ^= 1;
            b
        }] {
            std::fs::write(&path, bad).unwrap();
            assert_eq!(load_or_build(&options, Ok(key), 100, || Ok(data.clone())).unwrap(), data);
        }
        assert_eq!(read(&path, &key, 100).unwrap(), data);
    }
    #[test]
    fn disabled_unwritable_and_busy_fall_back() {
        let temp = Temp::new();
        let path = temp.0.join("absent");
        let options = PersistenceOptions { enabled: false, path: Some(path.clone()) };
        load_or_build(&options, Err("no identity".into()), 1, || Ok(vec![1])).unwrap();
        assert!(!path.exists());
        std::fs::write(&path, b"file").unwrap();
        let options = PersistenceOptions { enabled: true, path: Some(path) };
        assert_eq!(load_or_build(&options, Ok([0; 32]), 1, || Ok(vec![1])).unwrap(), vec![1]);
        let options = PersistenceOptions { enabled: true, path: Some(temp.0.clone()) };
        let held = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(temp.0.join(format!("{}.lock", "00".repeat(32))))
            .unwrap();
        held.try_lock().unwrap();
        assert_eq!(load_or_build(&options, Ok([0; 32]), 1, || Ok(vec![2])).unwrap(), vec![2]);
        assert!(!temp.0.join(format!("{}.bin", "00".repeat(32))).exists());
    }
    #[test]
    fn changed_source_rows_layout_and_digest_invalidate() {
        let temp = Temp::new();
        let source = temp.0.join("model.ninfer");
        std::fs::write(&source, b"abc").unwrap();
        let key = identity(&source, [0; 32], &[3, 90], &[1, 2]).unwrap();
        assert_ne!(key, identity(&source, [0; 32], &[3, 90], &[2, 1]).unwrap());
        assert_ne!(key, identity(&source, [0; 32], &[3, 91], &[1, 2]).unwrap());
        assert_ne!(key, identity(&source, [1; 32], &[3, 90], &[1, 2]).unwrap());
        std::fs::write(
            ignis_artifact::packer::sidecar_path(&source),
            br#"{"ngram_cache_source":{"schema":1,"digest":"new"}}"#,
        )
        .unwrap();
        assert_ne!(key, identity(&source, [0; 32], &[3, 90], &[1, 2]).unwrap());
        std::fs::write(&source, b"abcd").unwrap();
        assert_ne!(key, identity(&source, [0; 32], &[3, 90], &[1, 2]).unwrap());
    }
}
