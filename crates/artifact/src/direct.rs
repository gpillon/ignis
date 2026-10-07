//! Positional unbuffered reads of an artifact file, for readers that stream
//! an object from it while serving (Flash-Next's host-streamed n-gram table,
//! spec flash-next/04).
//!
//! [`DirectReader`] opens its own handle that bypasses the page cache
//! (`FILE_FLAG_NO_BUFFERING` on Windows, `O_DIRECT` on Linux, falling back to
//! a buffered handle where the file system refuses `O_DIRECT`, as tmpfs
//! does) and reads at an explicit offset, so any number of threads can share
//! one reader or each open its own: no read moves a shared file pointer.
//! The rules are the platform's: offset, length and buffer address on a
//! [`crate::DIRECT_IO_ALIGNMENT`] boundary ([`AlignedBuffer`] gives such a
//! buffer).

use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::fs::{File, OpenOptions};
use std::path::Path;

use crate::{fail, Result, DIRECT_IO_ALIGNMENT};

/// A zeroed heap buffer whose address is [`DIRECT_IO_ALIGNMENT`]-aligned.
pub struct AlignedBuffer {
    ptr: std::ptr::NonNull<u8>,
    layout: Layout,
}

// SAFETY: the buffer owns its allocation exclusively, like a Vec<u8>.
unsafe impl Send for AlignedBuffer {}
unsafe impl Sync for AlignedBuffer {}

impl AlignedBuffer {
    /// `len` bytes (rounded up to the alignment, at least one block).
    pub fn new(len: usize) -> Result<Self> {
        let align = DIRECT_IO_ALIGNMENT as usize;
        let size = len.max(1).div_ceil(align) * align;
        let layout = Layout::from_size_align(size, align)
            .map_err(|e| fail(format!("aligned buffer of {len} bytes: {e}")))?;
        // SAFETY: the layout has a nonzero size.
        let ptr = unsafe { alloc_zeroed(layout) };
        let ptr = std::ptr::NonNull::new(ptr)
            .ok_or_else(|| fail(format!("aligned buffer of {size} bytes: out of memory")))?;
        Ok(Self { ptr, layout })
    }

    /// The whole buffer (its length is a multiple of the alignment).
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: the allocation is `layout.size()` initialized bytes.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.layout.size()) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: as above, and `&mut self` makes the borrow exclusive.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.layout.size()) }
    }

    pub fn len(&self) -> usize {
        self.layout.size()
    }

    pub fn is_empty(&self) -> bool {
        false
    }
}

impl Drop for AlignedBuffer {
    fn drop(&mut self) {
        // SAFETY: allocated with this layout in `new`.
        unsafe { dealloc(self.ptr.as_ptr(), self.layout) };
    }
}

/// A read-only handle for positional unbuffered reads (see the module doc).
pub struct DirectReader {
    file: File,
    unbuffered: bool,
}

impl DirectReader {
    /// Open `path` for unbuffered reads.
    pub fn open(path: &Path) -> Result<Self> {
        let open = |flags: Option<u32>| {
            let mut options = OpenOptions::new();
            options.read(true);
            if let Some(flags) = flags {
                set_flags(&mut options, flags);
            }
            options.open(path)
        };
        match open(Some(UNBUFFERED_FLAG)) {
            Ok(file) => Ok(Self { file, unbuffered: true }),
            // A file system without unbuffered reads (Linux tmpfs) refuses
            // the flag itself; the alignment rules still hold for callers.
            Err(e) if refuses_unbuffered(&e) => Ok(Self {
                file: open(None).map_err(|e| fail(format!("open {}: {e}", path.display())))?,
                unbuffered: false,
            }),
            Err(e) => Err(fail(format!("open {} unbuffered: {e}", path.display()))),
        }
    }

    /// Whether reads bypass the page cache.
    pub fn is_unbuffered(&self) -> bool {
        self.unbuffered
    }

    /// Read `buf.len()` bytes at `offset`; returns how many arrived (fewer
    /// only at the end of the file). Offset, length and address must be
    /// [`DIRECT_IO_ALIGNMENT`]-aligned.
    pub fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<usize> {
        let align = DIRECT_IO_ALIGNMENT as usize;
        if !offset.is_multiple_of(DIRECT_IO_ALIGNMENT)
            || !buf.len().is_multiple_of(align)
            || !(buf.as_ptr() as usize).is_multiple_of(align)
        {
            return Err(fail(format!(
                "direct read of {} bytes at {offset} is not {align}-byte aligned",
                buf.len()
            )));
        }
        self.read_chunked(offset, buf, MAX_READ_CHUNK)
    }

    /// The read loop, at most `chunk` bytes (a multiple of the alignment) per
    /// system call: one Windows ReadFile moves at most 4 GiB - 1 bytes, not a
    /// sector multiple, which an unbuffered handle refuses.
    fn read_chunked(&self, offset: u64, buf: &mut [u8], chunk: usize) -> Result<usize> {
        let align = DIRECT_IO_ALIGNMENT as usize;
        let mut total = 0usize;
        while total < buf.len() {
            let end = buf.len().min(total + chunk);
            match read_at(&self.file, offset + total as u64, &mut buf[total..end]) {
                Ok(0) => break,
                Ok(n) => total += n,
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                // Windows reports a read that starts at or past the end of
                // the file as an error rather than a 0-byte read.
                Err(e) if is_end_of_file(&e) => break,
                Err(e) => return Err(fail(format!("direct read at {}: {e}", offset + total as u64))),
            }
            // An unbuffered read past the end returns a short count; a
            // partial sector cannot be continued, so stop at the first one.
            if !total.is_multiple_of(align) {
                break;
            }
        }
        Ok(total)
    }
}

/// The most one read system call is asked for (see `read_chunked`).
const MAX_READ_CHUNK: usize = 1 << 30;

/// A write handle for positional unbuffered writes: [`DirectReader`]'s twin,
/// for the disk tier's blob files (KV-disk, spec vram-budget/03).
///
/// It creates its file — never opens one that exists, so a write can only
/// land in a file this handle made — and writes at explicit offsets, so the
/// tier's writer thread can lay a blob's windows down and its header last.
/// The rules are [`DirectReader`]'s: offset, length and buffer address on a
/// [`DIRECT_IO_ALIGNMENT`] boundary.
pub struct DirectWriter {
    file: File,
    unbuffered: bool,
}

impl DirectWriter {
    /// Create `path`, which must not exist, for unbuffered writes.
    pub fn create(path: &Path) -> Result<Self> {
        let create = |flags: Option<u32>| {
            let mut options = OpenOptions::new();
            options.write(true).create_new(true);
            if let Some(flags) = flags {
                set_flags(&mut options, flags);
            }
            options.open(path)
        };
        match create(Some(UNBUFFERED_FLAG)) {
            Ok(file) => Ok(Self { file, unbuffered: true }),
            Err(e) if refuses_unbuffered(&e) => Ok(Self {
                file: create(None).map_err(|e| fail(format!("create {}: {e}", path.display())))?,
                unbuffered: false,
            }),
            Err(e) => Err(fail(format!("create {} unbuffered: {e}", path.display()))),
        }
    }

    /// Whether writes bypass the page cache.
    pub fn is_unbuffered(&self) -> bool {
        self.unbuffered
    }

    /// Write all of `buf` at `offset`. Offset, length and address must be
    /// [`DIRECT_IO_ALIGNMENT`]-aligned.
    pub fn write_at(&self, offset: u64, buf: &[u8]) -> Result<()> {
        let align = DIRECT_IO_ALIGNMENT as usize;
        if !offset.is_multiple_of(DIRECT_IO_ALIGNMENT)
            || !buf.len().is_multiple_of(align)
            || !(buf.as_ptr() as usize).is_multiple_of(align)
        {
            return Err(fail(format!(
                "direct write of {} bytes at {offset} is not {align}-byte aligned",
                buf.len()
            )));
        }
        self.write_chunked(offset, buf, MAX_READ_CHUNK)
    }

    /// The write loop, at most `chunk` bytes per system call (one Windows
    /// WriteFile moves at most 4 GiB - 1 bytes, as a read does).
    fn write_chunked(&self, offset: u64, buf: &[u8], chunk: usize) -> Result<()> {
        let mut total = 0usize;
        while total < buf.len() {
            let end = buf.len().min(total + chunk);
            match write_at(&self.file, offset + total as u64, &buf[total..end]) {
                Ok(0) => return Err(fail(format!("direct write at {}: the volume took nothing", offset + total as u64))),
                Ok(n) => total += n,
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(fail(format!("direct write at {}: {e}", offset + total as u64))),
            }
        }
        Ok(())
    }
}

#[cfg(windows)]
fn write_at(file: &File, offset: u64, buf: &[u8]) -> std::io::Result<usize> {
    use std::os::windows::fs::FileExt;
    file.seek_write(buf, offset)
}

#[cfg(unix)]
fn write_at(file: &File, offset: u64, buf: &[u8]) -> std::io::Result<usize> {
    use std::os::unix::fs::FileExt;
    file.write_at(buf, offset)
}

#[cfg(windows)]
const UNBUFFERED_FLAG: u32 = windows_sys::Win32::Storage::FileSystem::FILE_FLAG_NO_BUFFERING;
#[cfg(unix)]
const UNBUFFERED_FLAG: u32 = libc::O_DIRECT as u32;

#[cfg(windows)]
fn set_flags(options: &mut OpenOptions, flags: u32) {
    use std::os::windows::fs::OpenOptionsExt;
    options.custom_flags(flags);
}

#[cfg(unix)]
fn set_flags(options: &mut OpenOptions, flags: u32) {
    use std::os::unix::fs::OpenOptionsExt;
    options.custom_flags(flags as i32);
}

#[cfg(windows)]
fn read_at(file: &File, offset: u64, buf: &mut [u8]) -> std::io::Result<usize> {
    use std::os::windows::fs::FileExt;
    file.seek_read(buf, offset)
}

#[cfg(unix)]
fn read_at(file: &File, offset: u64, buf: &mut [u8]) -> std::io::Result<usize> {
    use std::os::unix::fs::FileExt;
    file.read_at(buf, offset)
}

#[cfg(windows)]
fn refuses_unbuffered(_: &std::io::Error) -> bool {
    false
}

#[cfg(unix)]
fn refuses_unbuffered(e: &std::io::Error) -> bool {
    e.raw_os_error() == Some(libc::EINVAL)
}

#[cfg(windows)]
fn is_end_of_file(e: &std::io::Error) -> bool {
    // ERROR_HANDLE_EOF
    e.raw_os_error() == Some(38)
}

#[cfg(unix)]
fn is_end_of_file(_: &std::io::Error) -> bool {
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_file(tag: &str, bytes: &[u8]) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!("ignis-direct-{tag}-{}", std::process::id()));
        std::fs::write(&path, bytes).unwrap();
        path
    }

    fn pattern(len: usize) -> Vec<u8> {
        (0..len).map(|i| (i * 7 + i / 4096) as u8).collect()
    }

    #[test]
    fn positional_reads_return_the_files_bytes_and_stop_short_at_the_end() {
        let bytes = pattern(3 * 4096 + 1000);
        let path = temp_file("read", &bytes);
        let reader = DirectReader::open(&path).unwrap();
        let mut buf = AlignedBuffer::new(8192).unwrap();
        assert_eq!(reader.read_at(4096, buf.as_mut_slice()).unwrap(), 8192);
        assert_eq!(buf.as_slice(), &bytes[4096..4096 + 8192]);
        // The last sector is partial: the read comes back short.
        assert_eq!(reader.read_at(3 * 4096, buf.as_mut_slice()).unwrap(), 1000);
        assert_eq!(&buf.as_slice()[..1000], &bytes[3 * 4096..]);
        // Past the end: nothing.
        assert_eq!(reader.read_at(4 * 4096, buf.as_mut_slice()).unwrap(), 0);
        drop(reader);
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn a_read_longer_than_one_chunk_is_issued_in_chunks() {
        let bytes = pattern(5 * 4096 + 700);
        let path = temp_file("chunks", &bytes);
        let reader = DirectReader::open(&path).unwrap();
        let mut buf = AlignedBuffer::new(6 * 4096).unwrap();
        // One block per system call: six calls, the last one short.
        assert_eq!(reader.read_chunked(0, buf.as_mut_slice(), 4096).unwrap(), 5 * 4096 + 700);
        assert_eq!(&buf.as_slice()[..bytes.len()], &bytes[..]);
        drop(reader);
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn an_unaligned_read_is_refused() {
        let path = temp_file("unaligned", &pattern(8192));
        let reader = DirectReader::open(&path).unwrap();
        let mut buf = AlignedBuffer::new(4096).unwrap();
        assert!(reader.read_at(100, buf.as_mut_slice()).is_err());
        assert!(reader.read_at(0, &mut buf.as_mut_slice()[..100]).is_err());
        assert!(reader.read_at(0, &mut buf.as_mut_slice()[1..4096]).is_err());
        drop(reader);
        std::fs::remove_file(&path).unwrap();
    }

    fn temp_path(tag: &str) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!("ignis-direct-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_file(&path);
        path
    }

    #[test]
    fn positional_writes_land_where_they_are_aimed_and_read_back() {
        let path = temp_path("write");
        let writer = DirectWriter::create(&path).unwrap();
        let mut first = AlignedBuffer::new(8192).unwrap();
        first.as_mut_slice().copy_from_slice(&pattern(8192));
        let mut header = AlignedBuffer::new(4096).unwrap();
        header.as_mut_slice().copy_from_slice(&pattern(4096 + 8192)[8192..]);
        // The body first, the header page last: the order the disk tier writes in.
        writer.write_at(4096, first.as_slice()).unwrap();
        writer.write_at(0, header.as_slice()).unwrap();
        drop(writer);
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(bytes.len(), 3 * 4096);
        assert_eq!(&bytes[..4096], header.as_slice());
        assert_eq!(&bytes[4096..], first.as_slice());
        let reader = DirectReader::open(&path).unwrap();
        let mut back = AlignedBuffer::new(8192).unwrap();
        assert_eq!(reader.read_at(4096, back.as_mut_slice()).unwrap(), 8192);
        assert_eq!(back.as_slice(), first.as_slice());
        drop(reader);
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn a_writer_never_opens_a_file_that_exists() {
        let path = temp_file("exists", &pattern(4096));
        assert!(DirectWriter::create(&path).is_err(), "a write can only land in a file the handle made");
        assert_eq!(std::fs::read(&path).unwrap(), pattern(4096), "and the file is left alone");
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn an_unaligned_write_is_refused() {
        let path = temp_path("unaligned-write");
        let writer = DirectWriter::create(&path).unwrap();
        let buf = AlignedBuffer::new(8192).unwrap();
        assert!(writer.write_at(100, &buf.as_slice()[..4096]).is_err());
        assert!(writer.write_at(0, &buf.as_slice()[..100]).is_err());
        assert!(writer.write_at(0, &buf.as_slice()[1..4097]).is_err());
        drop(writer);
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn a_write_longer_than_one_chunk_is_issued_in_chunks() {
        let path = temp_path("write-chunks");
        let writer = DirectWriter::create(&path).unwrap();
        let mut buf = AlignedBuffer::new(5 * 4096).unwrap();
        buf.as_mut_slice().copy_from_slice(&pattern(5 * 4096));
        writer.write_chunked(0, buf.as_slice(), 4096).unwrap();
        drop(writer);
        assert_eq!(std::fs::read(&path).unwrap(), pattern(5 * 4096));
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn threads_share_one_reader() {
        let bytes = pattern(64 * 4096);
        let path = temp_file("threads", &bytes);
        let reader = std::sync::Arc::new(DirectReader::open(&path).unwrap());
        let handles: Vec<_> = (0..8u64)
            .map(|t| {
                let reader = reader.clone();
                let expected = bytes.clone();
                std::thread::spawn(move || {
                    let mut buf = AlignedBuffer::new(4096).unwrap();
                    for round in 0..32u64 {
                        let block = (t * 31 + round * 7) % 64;
                        assert_eq!(reader.read_at(block * 4096, buf.as_mut_slice()).unwrap(), 4096);
                        let at = (block * 4096) as usize;
                        assert_eq!(buf.as_slice(), &expected[at..at + 4096]);
                    }
                })
            })
            .collect();
        for handle in handles {
            handle.join().unwrap();
        }
        drop(reader);
        std::fs::remove_file(&path).unwrap();
    }
}
