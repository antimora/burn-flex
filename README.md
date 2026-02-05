<p align="center">
  <img src="logo.png" alt="burn-ember logo" width="400">
</p>

## burn-ember - The portable ember that carries Burn anywhere.

A fast, memory-efficient CPU backend for Burn with multi-threading, SIMD, and gemm acceleration. Runs on std, no_std, and WebAssembly. Supports f16/bf16, zero-copy data loading, and is thread-safe by design.

### Features

- **Zero-Copy Operations**: Many operations return strided views without copying data:
  - `transpose`, `permute`, `flip`, `narrow`, `slice`
  - `unfold` (sliding windows as strided view, 1,300-156,000x faster than NdArray)
  - `expand` (broadcast via zero strides)
- **Arc-based Copy-on-Write**: O(1) tensor cloning with automatic COW semantics. In-place mutation when uniquely owned (2.6-4.2x faster binary ops).
- **Convolutions**: Unified 3D implementation with im2col + gemm. Conv1d/2d delegate to conv3d. Supports groups, dilation, padding.
- **Pooling**: Max pool, avg pool, adaptive avg pool. All via unified 3D with backward pass support.
- **Conv Transpose**: Scatter-based transposed convolutions for upsampling.
- **Portable SIMD**: Uses [pulp](https://crates.io/crates/pulp) for automatic dispatch:
  - aarch64: NEON
  - x86_64: AVX2, AVX512, SSE
  - wasm32: SIMD128
  - Embedded/other: Scalar fallback
- **Matrix Multiplication**: Optimized via [gemm](https://crates.io/crates/gemm) with native f16 support (1.3-3.4x faster)
- **Parallel Execution**: Optional rayon for large tensors
- **Dtype Support**: f32, f64, f16 (native), bf16 (via f32 conversion), i8-i64, u8-u64
