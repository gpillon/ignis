//! FFI bindings for the kernel leaf (flat C ABI, ADR 0001).
//!
//! Mirror of `kernel/include/ignis_kernel.h` — keep 1:1 when the surface
//! grows (ticket 03+).

unsafe extern "C" {
    /// Ticket 01 smoke test: proves the FFI path end-to-end.
    pub fn ignis_kernel_hello() -> u32;

    /// `c[i] = a[i] + b[i]` for `i in [0, n)`.
    /// Returns 0 on success, -1 on CUDA error.
    pub fn ignis_kernel_vector_sum(a: *const f32, b: *const f32, c: *mut f32, n: usize) -> i32;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hello_smoke() {
        assert_eq!(unsafe { ignis_kernel_hello() }, 42);
    }

    #[test]
    fn vector_sum_smoke() {
        let n = 1024usize;
        let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..n).map(|i| (i as f32) * 2.0).collect();
        let mut c = vec![0.0f32; n];
        let rc = unsafe { ignis_kernel_vector_sum(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), n) };
        assert_eq!(rc, 0, "CUDA vector sum failed (is the GPU available?)");
        for i in 0..n {
            assert_eq!(c[i], (i as f32) * 3.0, "mismatch at {i}");
        }
    }
}