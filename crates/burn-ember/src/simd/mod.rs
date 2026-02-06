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

// SIMD-aligned memory allocation
#[cfg(feature = "simd")]
pub mod aligned;

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
    CmpOp, add_f32, add_inplace_f32, add_scalar_f32, bool_and_inplace_u8, bool_and_u8,
    bool_not_inplace_u8, bool_not_u8, bool_or_inplace_u8, bool_or_u8, bool_xor_inplace_u8,
    bool_xor_u8, cmp_f32, cmp_scalar_f32, div_f32, div_inplace_f32, mul_f32, mul_inplace_f32,
    mul_scalar_f32, sub_f32, sub_inplace_f32,
};

// Scalar fallback for other platforms
#[cfg(not(target_arch = "aarch64"))]
mod scalar;

#[cfg(not(target_arch = "aarch64"))]
pub use scalar::{
    CmpOp, add_f32, add_inplace_f32, add_scalar_f32, bool_and_inplace_u8, bool_and_u8,
    bool_not_inplace_u8, bool_not_u8, bool_or_inplace_u8, bool_or_u8, bool_xor_inplace_u8,
    bool_xor_u8, cmp_f32, cmp_scalar_f32, div_f32, div_inplace_f32, mul_f32, mul_inplace_f32,
    mul_scalar_f32, sub_f32, sub_inplace_f32,
};
