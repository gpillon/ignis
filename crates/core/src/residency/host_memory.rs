//! The host plan's one measured input: physical memory available now
//! (spec flash-next/03, acceptance 1). Windows: `GlobalMemoryStatusEx`'s
//! `ullAvailPhys` — free plus standby pages, which Windows hands out before
//! it pages anything. Linux: `/proc/meminfo`'s `MemAvailable`. Elsewhere,
//! or when the query fails, `None`: the caller decides what an unmeasured
//! machine means.

/// Bytes of physical memory available for a new allocation.
pub fn available_physical_bytes() -> Option<u64> {
    imp::available_physical_bytes()
}

#[cfg(windows)]
mod imp {
    /// `MEMORYSTATUSEX` (sysinfoapi.h).
    #[repr(C)]
    struct MemoryStatusEx {
        length: u32,
        memory_load: u32,
        total_phys: u64,
        avail_phys: u64,
        total_page_file: u64,
        avail_page_file: u64,
        total_virtual: u64,
        avail_virtual: u64,
        avail_extended_virtual: u64,
    }

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GlobalMemoryStatusEx(buffer: *mut MemoryStatusEx) -> i32;
    }

    pub fn available_physical_bytes() -> Option<u64> {
        let mut status = MemoryStatusEx {
            length: std::mem::size_of::<MemoryStatusEx>() as u32,
            memory_load: 0,
            total_phys: 0,
            avail_phys: 0,
            total_page_file: 0,
            avail_page_file: 0,
            total_virtual: 0,
            avail_virtual: 0,
            avail_extended_virtual: 0,
        };
        // SAFETY: `status` is a properly sized, writable MEMORYSTATUSEX whose
        // `dwLength` is set, as the API requires.
        let ok = unsafe { GlobalMemoryStatusEx(&mut status) };
        (ok != 0).then_some(status.avail_phys)
    }
}

#[cfg(target_os = "linux")]
mod imp {
    pub fn available_physical_bytes() -> Option<u64> {
        let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
        parse_mem_available(&meminfo)
    }

    /// `MemAvailable:   12345678 kB` → bytes.
    pub(super) fn parse_mem_available(meminfo: &str) -> Option<u64> {
        let line = meminfo.lines().find(|l| l.starts_with("MemAvailable:"))?;
        let kib: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
        kib.checked_mul(1024)
    }

    #[cfg(test)]
    mod tests {
        #[test]
        fn mem_available_is_read_in_kib() {
            let text = "MemTotal:       65536000 kB\nMemAvailable:   52428800 kB\n";
            assert_eq!(super::parse_mem_available(text), Some(52_428_800 * 1024));
            assert_eq!(super::parse_mem_available("MemTotal: 1 kB\n"), None);
        }
    }
}

#[cfg(not(any(windows, target_os = "linux")))]
mod imp {
    pub fn available_physical_bytes() -> Option<u64> {
        None
    }
}
