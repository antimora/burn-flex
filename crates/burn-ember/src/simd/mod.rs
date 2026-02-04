//! SIMD-optimized kernels for tensor operations.
//!
//! Provides portable SIMD implementations via `pulp` with automatic
//! dispatch to the best available instruction set:
//! - aarch64: NEON
//! - x86_64: AVX2, AVX512, SSE
//! - wasm32: SIMD128
//! - Other: Scalar fallback
//!
//! Enable with the `simd` feature flag (enabled by default).

#[cfg(target_arch = "aarch64")]
pub mod neon;

// Portable SIMD kernels using pulp
#[cfg(feature = "simd")]
pub mod kernels;

/// SIMD lane count for f32 on current platform.
#[cfg(target_arch = "aarch64")]
pub const F32_LANES: usize = 4; // NEON: 128-bit / 32-bit = 4

#[cfg(target_arch = "x86_64")]
pub const F32_LANES: usize = 8; // AVX2: 256-bit / 32-bit = 8

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
pub const F32_LANES: usize = 1; // Scalar fallback

/// Threshold for using SIMD (elements). Below this, scalar is faster.
pub const SIMD_THRESHOLD: usize = 32;

/// Threshold for using parallel execution (elements).
pub const PARALLEL_THRESHOLD: usize = 8192;

// Re-export platform-specific implementations
#[cfg(target_arch = "aarch64")]
pub use neon::{
    add_f32, add_inplace_f32, add_scalar_f32, div_f32, div_inplace_f32, mul_f32, mul_inplace_f32,
    mul_scalar_f32, sub_f32, sub_inplace_f32,
};

// Scalar fallback for other platforms
#[cfg(not(target_arch = "aarch64"))]
pub use scalar::{
    add_f32, add_inplace_f32, add_scalar_f32, div_f32, div_inplace_f32, mul_f32, mul_inplace_f32,
    mul_scalar_f32, sub_f32, sub_inplace_f32,
};

#[cfg(not(target_arch = "aarch64"))]
mod scalar {
    /// Scalar add for f32 slices.
    #[inline]
    pub fn add_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
        for i in 0..a.len() {
            out[i] = a[i] + b[i];
        }
    }

    /// Scalar sub for f32 slices.
    #[inline]
    pub fn sub_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
        for i in 0..a.len() {
            out[i] = a[i] - b[i];
        }
    }

    /// Scalar mul for f32 slices.
    #[inline]
    pub fn mul_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
        for i in 0..a.len() {
            out[i] = a[i] * b[i];
        }
    }

    /// Scalar div for f32 slices.
    #[inline]
    pub fn div_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
        for i in 0..a.len() {
            out[i] = a[i] / b[i];
        }
    }

    /// Scalar add with scalar value.
    #[inline]
    pub fn add_scalar_f32(a: &[f32], scalar: f32, out: &mut [f32]) {
        for i in 0..a.len() {
            out[i] = a[i] + scalar;
        }
    }

    /// Scalar mul with scalar value.
    #[inline]
    pub fn mul_scalar_f32(a: &[f32], scalar: f32, out: &mut [f32]) {
        for i in 0..a.len() {
            out[i] = a[i] * scalar;
        }
    }

    /// Scalar in-place add.
    #[inline]
    pub fn add_inplace_f32(a: &mut [f32], b: &[f32]) {
        for i in 0..a.len() {
            a[i] += b[i];
        }
    }

    /// Scalar in-place sub.
    #[inline]
    pub fn sub_inplace_f32(a: &mut [f32], b: &[f32]) {
        for i in 0..a.len() {
            a[i] -= b[i];
        }
    }

    /// Scalar in-place mul.
    #[inline]
    pub fn mul_inplace_f32(a: &mut [f32], b: &[f32]) {
        for i in 0..a.len() {
            a[i] *= b[i];
        }
    }

    /// Scalar in-place div.
    #[inline]
    pub fn div_inplace_f32(a: &mut [f32], b: &[f32]) {
        for i in 0..a.len() {
            a[i] /= b[i];
        }
    }
}
