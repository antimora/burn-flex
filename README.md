<p align="center">
  <img src="logo.png" alt="burn-flex logo" width="400">
</p>

## burn-flex - The portable flex that carries Burn anywhere.

A fast, memory-efficient CPU backend for Burn with multi-threading, SIMD, and optimized matrix
multiplication. Runs on std, no_std, and WebAssembly. Supports f16/bf16, zero-copy data loading, and
is thread-safe by design.

> **[Detailed comparison with burn-ndarray](COMPARISON.md)**: Full architecture, feature coverage,
> operation-by-operation analysis, and migration path.

### Features

- **Zero-Copy Operations**: Many operations return strided views without copying data:
  - `transpose`, `permute`, `flip`, `narrow`, `slice`
  - `unfold` (sliding windows as strided view, 1,300-156,000x faster than NdArray)
  - `expand` (broadcast via zero strides)
- **Arc-based Copy-on-Write**: O(1) tensor cloning with automatic COW semantics. In-place mutation
  when uniquely owned (2.6-4.2x faster binary ops).
- **Convolutions**: Unified 3D implementation with im2col + gemm. Conv1d/2d delegate to conv3d.
  Supports groups, dilation, padding.
- **Pooling**: Max pool, avg pool, adaptive avg pool. All via unified 3D with backward pass support.
- **Conv Transpose**: Scatter-based transposed convolutions for upsampling.
- **Portable SIMD**: Uses [macerator](https://crates.io/crates/macerator) for automatic dispatch:
  - aarch64: NEON
  - x86_64: AVX2, AVX512, SSE
  - wasm32: SIMD128
  - Embedded/other: Scalar fallback
- **Matrix Multiplication**: Optimized via [gemm](https://crates.io/crates/gemm) with native f16
  support (1.3-3.4x faster)
- **Parallel Execution**: Optional rayon for large tensors
- **Quantization**: Full quantize/dequantize support with per-tensor and per-block symmetric
  schemes. All ~40 quantized ops (arithmetic, trig, reductions, sorting, etc.) work out of the box.
  Layout ops on quantized tensors (permute, flip, expand, slice, select) are zero-copy. Stores
  scales separately for direct `scale * x_q` dequantization (135-232x faster than NdArray).
- **Dtype Support**: f32, f64, f16 (native), bf16 (via f32 conversion), i8-i64, u8-u64
- **Built on Burn**: Leverages Burn's native infrastructure (`Bytes`, `Shape`, `TensorData`,
  `Element` trait) from burn-backend and burn-std

### Performance vs burn-ndarray (Apple M3 Max)

burn-flex consistently outperforms burn-ndarray across the board, often using a fraction of the
memory:

| Category          | Speedup            | Highlights                              |
| ----------------- | ------------------ | --------------------------------------- |
| Binary ops (f32)  | **2.4-3.6x**       | 3x less memory allocation               |
| Binary ops (i64)  | **1.5-6.4x**       | Smaller tensors see bigger gains        |
| Matmul (square)   | **1.1-3.4x**       | Up to 2.3x at 1024x1024                 |
| Matmul (batched)  | **1.8-3.2x**       | 3.2x on multi-head attention shapes     |
| Conv2d (3x3)      | **1.4-4.0x**       | Larger kernels and batches benefit most |
| Conv1d            | **4.3-9.6x**       |                                         |
| Pooling           | **1.2-3.1x**       |                                         |
| Interpolation     | **1.2-3.6x**       | All modes: nearest, bilinear, bicubic   |
| Reductions        | **1.6-3.9x**       | Near-zero allocation for scalar results |
| Cumulative ops    | **3.1-93x**        | 1D cumsum: 93x faster                   |
| Gather/scatter    | **1.9-9.7x**       |                                         |
| Unary (tanh, sin) | **1.3-2.7x**       |                                         |
| Comparisons       | **2.1-3.9x**       |                                         |
| Int casting       | **5.0-7.6x**       |                                         |
| Quantized ops     | **1.6-232x**       | Dequant 232x, q_add 117x, quantize 1.6x |
| Slice/narrow      | **2.1-2100x**      | Zero-copy strided views                 |
| Unfold            | **1,200-166,000x** | Zero-copy vs full materialization       |
| Expand            | **550-2,600x**     | Zero-copy broadcast                     |

See [BENCHMARKS.md](BENCHMARKS.md) for the full breakdown.

### Status

- All `burn-backend-tests` pass across all feature flag combinations:
  - `no-default-features` (no_std, no SIMD, no rayon)
  - `no-default-features + simd` (no_std with SIMD)
  - `std`
  - `std + simd`
  - `std + rayon`
  - `std + simd + rayon` (default)
- Burn's `burn-no-std-tests` integration suite passes (MNIST model inference in `#![no_std]`)
- Builds for embedded and WebAssembly targets:
  - `thumbv6m-none-eabi` (ARM Cortex-M0+, no atomic pointers)
  - `thumbv7m-none-eabi` (ARM Cortex-M3)
  - `wasm32-unknown-unknown`
- Tested for edge-case robustness: integer overflow at type boundaries, large-float rounding,
  invalid pooling parameters, zero-sized dimensions. Safe for embedded devices.
- All ONNX model checks in `burn-onnx` pass
- Real model inference verified:
  - [ALBERT](https://huggingface.co/albert/albert-base-v2) (masked language model, all v2 variants)
  - [MiniLM](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) (sentence embeddings, L6
    and L12)

### Documentation

- [COMPARISON.md](COMPARISON.md) - Comprehensive comparison with burn-ndarray
- [ARCHITECTURE.md](ARCHITECTURE.md) - Design decisions, memory strategy, and implementation
  patterns
- [BENCHMARKS.md](BENCHMARKS.md) - Full benchmark results (Flex vs NdArray)
- [ACKNOWLEDGMENTS.md](ACKNOWLEDGMENTS.md) - Projects that influenced burn-flex
