<p align="center">
  <img src="logo.png" alt="burn-ember logo" width="400">
</p>

## burn-ember - The portable ember that carries Burn anywhere.

A fast, memory-efficient CPU backend for Burn with multi-threading, SIMD, and gemm acceleration. Runs on std, no_std, and WebAssembly. Supports f16/bf16, zero-copy data loading, and is thread-safe by design.

### Features

- **Portable SIMD**: Uses [pulp](https://crates.io/crates/pulp) for automatic dispatch to the best available instruction set:
  - aarch64: NEON
  - x86_64: AVX2, AVX512, SSE
  - wasm32: SIMD128
  - Embedded/other: Scalar fallback
- **Matrix Multiplication**: Optimized via [gemm](https://crates.io/crates/gemm) with native f16 support
- **Parallel Execution**: Optional rayon integration for large tensor operations
- **Memory Efficient**: Zero-copy views, strided tensor support, minimal allocations
