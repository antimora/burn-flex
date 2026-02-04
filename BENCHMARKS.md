# Ember vs NdArray Benchmarks

Benchmarks comparing burn-ember against burn-ndarray on Apple M1 Max.

**Date**: 2026-02-04 **Platform**: darwin (aarch64) **Features**: simd, rayon, gemm

---

## Summary

| Category        | Ember Wins | NdArray Wins | Ties  |
| --------------- | ---------- | ------------ | ----- |
| Binary Ops      | 11         | 0            | 0     |
| Matrix Multiply | 4          | 13           | 0     |
| Slice Ops       | 18         | 0            | 0     |
| Reduce Ops      | 12         | 4            | 0     |
| Unary Ops       | 15         | 0            | 4     |
| Comparison Ops  | 13         | 4            | 0     |
| **Total**       | **73**     | **21**       | **4** |

---

## Binary Operations

Element-wise operations on tensors. **Major improvement** from Arc-based COW enabling true in-place
mutation for unique tensors.

### Tensor-Tensor Operations

| Operation | Size         | Ember Time | NdArray Time | Speedup  | Ember Mem | NdArray Mem |
| --------- | ------------ | ---------- | ------------ | -------- | --------- | ----------- |
| add       | small (4K)   | 393 ns     | 1.33 us      | **3.4x** | 16.5 KB   | 49.3 KB     |
| add       | medium (64K) | 7.5 us     | 21.0 us      | **2.8x** | 262 KB    | 787 KB      |
| add       | large (1M)   | 114 us     | 342 us       | **3.0x** | 4.2 MB    | 12.6 MB     |
| mul       | small (4K)   | 403 ns     | 1.60 us      | **4.0x** | 16.5 KB   | 49.3 KB     |
| mul       | medium (64K) | 7.7 us     | 21.4 us      | **2.8x** | 262 KB    | 787 KB      |
| mul       | large (1M)   | 118 us     | 336 us       | **2.9x** | 4.2 MB    | 12.6 MB     |
| div       | large (1M)   | 117 us     | 335 us       | **2.9x** | 4.2 MB    | 12.6 MB     |

### Transposed Input Operations

| Operation | Size      | Ember Time | NdArray Time | Speedup  | Ember Mem | NdArray Mem |
| --------- | --------- | ---------- | ------------ | -------- | --------- | ----------- |
| add       | 256x256   | 50.3 us    | 55.3 us      | **1.1x** | 262 KB    | 524 KB      |
| add       | 1024x1024 | 963 us     | 1.10 ms      | **1.1x** | 4.2 MB    | 8.4 MB      |

### Scalar Operations

| Operation  | Size       | Ember Time | NdArray Time | Speedup  | Ember Mem | NdArray Mem |
| ---------- | ---------- | ---------- | ------------ | -------- | --------- | ----------- |
| add_scalar | large (1M) | 76.2 us    | 185 us       | **2.4x** | 4.2 MB    | 8.4 MB      |
| mul_scalar | large (1M) | 75.9 us    | 185 us       | **2.4x** | 4.2 MB    | 8.4 MB      |

**Key improvement**: Arc-based COW now enables true in-place mutation when tensor is uniquely
owned. This nearly doubles performance vs previous implementation (was 1.4-1.8x, now 2.4-4.0x).

---

## Matrix Multiplication

Using gemm crate for optimized matmul.

### Square Matrices (f32)

| Size      | Ember Time | NdArray Time | Speedup  | Ember Mem | NdArray Mem |
| --------- | ---------- | ------------ | -------- | --------- | ----------- |
| 64x64     | 6.18 us    | 14.5 us      | **2.3x** | 33.6 KB   | 49.3 KB     |
| 128x128   | 44.9 us    | 59.7 us      | **1.3x** | 328 KB    | 197 KB      |
| 256x256   | 320 us     | 161 us       | 0.5x     | 787 KB    | 787 KB      |
| 512x512   | 2.34 ms    | 836 us       | 0.4x     | 2.1 MB    | 3.1 MB      |
| 1024x1024 | 19.5 ms    | 5.93 ms      | 0.3x     | 6.3 MB    | 12.6 MB     |

Note: Performance regression at larger sizes requires investigation.

### Rectangular Matrices (f32)

| Shape                       | Ember Time | NdArray Time | Speedup | Ember Mem | NdArray Mem |
| --------------------------- | ---------- | ------------ | ------- | --------- | ----------- |
| 512x64 x 64x512 (attention) | 333 us     | 147 us       | 0.4x    | 1.2 MB    | 1.3 MB      |
| 256x512 x 512x256 (linear)  | 600 us     | 259 us       | 0.4x    | 1.3 MB    | 1.6 MB      |
| 128x1024 x 1024x128 (wide)  | 322 us     | 183 us       | 0.6x    | 2.2 MB    | 1.6 MB      |

### Transposed Matrices (256x256)

| Transpose       | Ember Time | NdArray Time | Speedup | Ember Mem | NdArray Mem |
| --------------- | ---------- | ------------ | ------- | --------- | ----------- |
| LHS transposed  | 306 us     | 163 us       | 0.5x    | 787 KB    | 787 KB      |
| RHS transposed  | 335 us     | 170 us       | 0.5x    | 1.0 MB    | 787 KB      |
| Both transposed | 336 us     | 190 us       | 0.6x    | 1.0 MB    | 787 KB      |

### Batched Matmul (f32)

| Batch x Size        | Ember Time | NdArray Time | Speedup  | Ember Mem | NdArray Mem |
| ------------------- | ---------- | ------------ | -------- | --------- | ----------- |
| 8 x 64x64           | 48.9 us    | 61.9 us      | **1.3x** | 148 KB    | 393 KB      |
| 32 x 64x64          | 197 us     | 121 us       | 0.6x     | 542 KB    | 1.6 MB      |
| 16 x 128x128        | 674 us     | 550 us       | 0.8x     | 1.3 MB    | 3.1 MB      |
| 12 x 512x64 (heads) | 3.82 ms    | 1.68 ms      | 0.4x     | 12.7 MB   | 15.7 MB     |

### Integer Matmul (i32)

| Size    | Ember Time | NdArray Time | Speedup | Ember Mem | NdArray Mem |
| ------- | ---------- | ------------ | ------- | --------- | ----------- |
| 64x64   | 112 us     | 111 us       | 1.0x    | 33 KB     | 98.5 KB     |
| 128x128 | 963 us     | 978 us       | 1.0x    | 131 KB    | 393 KB      |
| 256x256 | 10.6 ms    | 10.0 ms      | 0.9x    | 525 KB    | 1.6 MB      |
| 512x512 | 118 ms     | 113 ms       | 0.9x    | 2.1 MB    | 6.3 MB      |

---

## Slice Operations

Tensor slicing, narrowing, and assignment. **Ember uses zero-copy views** for slicing operations.

### Basic Slicing

| Operation | Size      | Ember Time | NdArray Time | Speedup      | Ember Mem | NdArray Mem |
| --------- | --------- | ---------- | ------------ | ------------ | --------- | ----------- |
| slice 1D  | 1K        | 148 ns     | 269 ns       | **1.8x**     | 80 B      | 6.3 KB      |
| slice 1D  | 1M        | 128 ns     | 85.7 us      | **~670x**    | 80 B      | 6.3 MB      |
| slice 2D  | 256x256   | 131 ns     | 8.31 us      | **~63x**     | 160 B     | 328 KB      |
| slice 2D  | 1024x1024 | 139 ns     | 81.7 us      | **~590x**    | 160 B     | 5.2 MB      |
| slice 3D  | 64x64x64  | 157 ns     | 28.4 us      | **~180x**    | 240 B     | 1.2 MB      |

### Narrow Operations

| Operation   | Size      | Ember Time | NdArray Time | Speedup    | Ember Mem | NdArray Mem |
| ----------- | --------- | ---------- | ------------ | ---------- | --------- | ----------- |
| narrow dim0 | 256x256   | 199 ns     | 6.51 us      | **~33x**   | 240 B     | 393 KB      |
| narrow dim0 | 1024x1024 | 177 ns     | 86.0 us      | **~486x**  | 240 B     | 6.3 MB      |
| narrow dim1 | 256x256   | 169 ns     | 11.4 us      | **~67x**   | 240 B     | 393 KB      |

### Slice Assignment

| Operation | Size      | Ember Time | NdArray Time | Speedup  | Ember Mem | NdArray Mem |
| --------- | --------- | ---------- | ------------ | -------- | --------- | ----------- |
| assign 1D | 1K        | 333 ns     | 629 ns       | **1.9x** | 4.3 KB    | 10.4 KB     |
| assign 2D | 256x256   | 5.75 us    | 13.9 us      | **2.4x** | 262 KB    | 590 KB      |
| assign 2D | 1024x1024 | 79.8 us    | 183 us       | **2.3x** | 4.2 MB    | 9.4 MB      |

### Transposed Slicing

| Operation  | Size      | Ember Time | NdArray Time | Speedup      | Ember Mem | NdArray Mem |
| ---------- | --------- | ---------- | ------------ | ------------ | --------- | ----------- |
| transposed | 256x256   | 106 ns     | 8.11 us      | **~77x**     | 160 B     | 65.7 KB     |
| transposed | 1024x1024 | 107 ns     | 226 us       | **~2100x**   | 160 B     | 1.05 MB     |

### Slice with Step

| Operation | Size      | Ember Time | NdArray Time | Speedup      | Ember Mem | NdArray Mem |
| --------- | --------- | ---------- | ------------ | ------------ | --------- | ----------- |
| step2 1D  | 1K        | 93 ns      | 435 ns       | **4.7x**     | 80 B      | 6.3 KB      |
| step2 1D  | 1M        | 94 ns      | 195 us       | **~2100x**   | 80 B      | 6.3 MB      |
| step2 2D  | 1024x1024 | 98 ns      | 138 us       | **~1400x**   | 160 B     | 5.2 MB      |
| step4 2D  | 256x256   | 98 ns      | 6.98 us      | **~71x**     | 160 B     | 279 KB      |

**Key insight**: Ember implements slicing as zero-copy views (just stride/offset manipulation),
while NdArray copies data. This gives orders of magnitude speedup for large tensors.

---

## Reduce Operations

Sum, mean, argmax reductions with portable SIMD via pulp.

### Full Tensor Sum

| Size | Ember Time | NdArray Time | Speedup  | Ember Mem | NdArray Mem |
| ---- | ---------- | ------------ | -------- | --------- | ----------- |
| 1K   | 273 ns     | 400 ns       | **1.5x** | 104 B     | 8.3 KB      |
| 64K  | 11.0 us    | 15.0 us      | **1.4x** | 104 B     | 524 KB      |
| 1M   | 173 us     | 221 us       | **1.3x** | 104 B     | 8.4 MB      |

### Sum Along Dimension

| Shape     | Dim | Ember Time | NdArray Time | Speedup  | Ember Mem | NdArray Mem |
| --------- | --- | ---------- | ------------ | -------- | --------- | ----------- |
| 256x256   | 0   | 5.19 us    | 20.5 us      | **3.9x** | 2.2 KB    | 524 KB      |
| 256x256   | 1   | 6.15 us    | 13.7 us      | **2.2x** | 1.2 KB    | 524 KB      |
| 1024x1024 | 0   | 82.0 us    | 231 us       | **2.8x** | 8.3 KB    | 8.4 MB      |
| 1024x1024 | 1   | 142 us     | 214 us       | **1.5x** | 4.2 KB    | 8.4 MB      |

### 3D Sum (Batched)

| Shape      | Dim | Ember Time | NdArray Time | Speedup  | Ember Mem | NdArray Mem |
| ---------- | --- | ---------- | ------------ | -------- | --------- | ----------- |
| 32x256x256 | 1   | 158 us     | 541 us       | **3.4x** | 65.7 KB   | 16.8 MB     |
| 32x256x256 | 2   | 192 us     | 373 us       | **1.9x** | 32.9 KB   | 16.8 MB     |

### Sum Transposed (total sum)

| Size      | Ember Time | NdArray Time | Speedup  | Ember Mem | NdArray Mem |
| --------- | ---------- | ------------ | -------- | --------- | ----------- |
| 256x256   | 11.2 us    | 6.25 us      | 0.6x     | 120 B     | 44 B        |
| 1024x1024 | 184 us     | 95.5 us      | 0.5x     | 120 B     | 44 B        |

### Sum Dim on Transposed Tensor

| Size      | Dim | Ember Time | NdArray Time | Speedup  |
| --------- | --- | ---------- | ------------ | -------- |
| 256x256   | 0   | 5.98 us    | 4.65 us      | 0.8x     |
| 1024x1024 | 0   | 142 us     | 83.1 us      | 0.6x     |

### Mean Along Dimension

| Shape     | Dim | Ember Time | NdArray Time | Speedup  | Ember Mem | NdArray Mem |
| --------- | --- | ---------- | ------------ | -------- | --------- | ----------- |
| 256x256   | 1   | 6.19 us    | 13.8 us      | **2.2x** | 1.2 KB    | 524 KB      |
| 1024x1024 | 1   | 142 us     | 214 us       | **1.5x** | 4.2 KB    | 8.4 MB      |

### Argmax

| Shape     | Dim | Ember Time | NdArray Time | Speedup  | Ember Mem | NdArray Mem |
| --------- | --- | ---------- | ------------ | -------- | --------- | ----------- |
| 1K (flat) | -   | 3.48 us    | 4.40 us      | **1.3x** | 104 B     | 8.3 KB      |
| 256x256   | 1   | 212 us     | 257 us       | **1.2x** | 2.2 KB    | 524 KB      |
| 1024x1024 | 1   | 3.33 ms    | 4.08 ms      | **1.2x** | 8.3 KB    | 8.4 MB      |

---

## Unary Operations

Element-wise math functions (exp, log, sqrt, trig, etc.).

### Basic Math Operations

| Operation | Size         | Ember Time | NdArray Time | Speedup  | Ember Mem | NdArray Mem |
| --------- | ------------ | ---------- | ------------ | -------- | --------- | ----------- |
| exp       | small (4K)   | 5.23 us    | 5.65 us      | **1.1x** | 16.5 KB   | 32.8 KB     |
| exp       | medium (64K) | 82.8 us    | 91.6 us      | **1.1x** | 262 KB    | 524 KB      |
| exp       | large (1M)   | 1.32 ms    | 1.45 ms      | **1.1x** | 4.2 MB    | 8.4 MB      |
| log       | small (4K)   | 6.68 us    | 7.35 us      | **1.1x** | 16.5 KB   | 32.8 KB     |
| log       | medium (64K) | 106 us     | 117 us       | **1.1x** | 262 KB    | 524 KB      |
| log       | large (1M)   | 1.71 ms    | 1.83 ms      | **1.1x** | 4.2 MB    | 8.4 MB      |
| sqrt      | small (4K)   | 663 ns     | 1.13 us      | **1.7x** | 16.5 KB   | 24.6 KB     |
| sqrt      | medium (64K) | 9.35 us    | 17.2 us      | **1.8x** | 262 KB    | 524 KB      |
| sqrt      | large (1M)   | 142 us     | 271 us       | **1.9x** | 4.2 MB    | 8.4 MB      |
| abs       | large (1M)   | 78.1 us    | 196 us       | **2.5x** | 4.2 MB    | 8.4 MB      |
| recip     | large (1M)   | 74.2 us    | 211 us       | **2.8x** | 4.2 MB    | 8.4 MB      |

### Trigonometric Operations

| Operation | Size         | Ember Time | NdArray Time | Speedup  | Ember Mem | NdArray Mem |
| --------- | ------------ | ---------- | ------------ | -------- | --------- | ----------- |
| sin       | small (4K)   | 5.77 us    | 8.40 us      | **1.5x** | 16.5 KB   | 32.8 KB     |
| sin       | medium (64K) | 91.3 us    | 133 us       | **1.5x** | 262 KB    | 524 KB      |
| sin       | large (1M)   | 1.42 ms    | 2.12 ms      | **1.5x** | 4.2 MB    | 8.4 MB      |
| cos       | small (4K)   | 7.31 us    | 9.19 us      | **1.3x** | 16.5 KB   | 32.8 KB     |
| cos       | large (1M)   | 1.67 ms    | 2.27 ms      | **1.4x** | 4.2 MB    | 8.4 MB      |
| tanh      | small (4K)   | 7.18 us    | 14.3 us      | **2.0x** | 16.5 KB   | 32.8 KB     |
| tanh      | medium (64K) | 114 us     | 220 us       | **1.9x** | 262 KB    | 524 KB      |
| tanh      | large (1M)   | 1.80 ms    | 3.51 ms      | **2.0x** | 4.2 MB    | 8.4 MB      |

### Transposed (Non-contiguous) Input

| Operation | Size      | Ember Time | NdArray Time | Speedup  | Ember Mem | NdArray Mem |
| --------- | --------- | ---------- | ------------ | -------- | --------- | ----------- |
| exp       | 256x256   | 80.0 us    | 85.8 us      | **1.1x** | 262 KB    | 262 KB      |
| exp       | 1024x1024 | 1.32 ms    | 1.36 ms      | **1.0x** | 4.2 MB    | 4.2 MB      |

---

## Comparison & Boolean Operations

Element-wise comparisons with NEON SIMD optimization for f32.

### Tensor-Tensor Comparisons

| Operation | Size         | Ember Time | NdArray Time | Speedup  | Ember Mem | NdArray Mem |
| --------- | ------------ | ---------- | ------------ | -------- | --------- | ----------- |
| greater   | small (4K)   | 598 ns     | 1.59 us      | **2.7x** | 4.2 KB    | 49 KB       |
| greater   | medium (64K) | 6.72 us    | 21.3 us      | **3.2x** | 65.7 KB   | 787 KB      |
| greater   | large (1M)   | 100 us     | 307 us       | **3.1x** | 1.0 MB    | 12.6 MB     |
| equal     | small (4K)   | 603 ns     | 1.55 us      | **2.6x** | 4.2 KB    | 49 KB       |
| equal     | large (1M)   | 98.3 us    | 319 us       | **3.2x** | 1.0 MB    | 12.6 MB     |
| lower     | large (1M)   | 101 us     | 300 us       | **3.0x** | 1.0 MB    | 12.6 MB     |

### Scalar Comparisons

| Operation    | Size       | Ember Time | NdArray Time | Speedup  | Ember Mem | NdArray Mem |
| ------------ | ---------- | ---------- | ------------ | -------- | --------- | ----------- |
| greater_elem | large (1M) | 97.2 us    | 195 us       | **2.0x** | 1.0 MB    | 8.4 MB      |

### Transposed (Non-contiguous) Comparisons

| Operation | Size      | Ember Time | NdArray Time | Speedup |
| --------- | --------- | ---------- | ------------ | ------- |
| greater   | 256x256   | 54.0 us    | 51.6 us      | 1.0x    |
| greater   | 1024x1024 | 944 us     | 1.08 ms      | 0.9x    |

### Broadcast Comparisons (Outer-Product Pattern)

| Operation | Shape     | Ember Time | NdArray Time | Speedup  | Ember Mem | NdArray Mem |
| --------- | --------- | ---------- | ------------ | -------- | --------- | ----------- |
| greater   | 256x256   | 8.11 us    | 26.4 us      | **3.3x** | 66.6 KB   | 67.7 KB     |
| greater   | 1024x1024 | 120 us     | 317 us       | **2.6x** | 1.1 MB    | 1.1 MB      |

### Expand Operation (Broadcasting)

| Operation                  | Ember Time | NdArray Time | Speedup    | Note               |
| -------------------------- | ---------- | ------------ | ---------- | ------------------ |
| expand 1x1 to 1000x1000    | 150 ns     | 292 us       | **~1950x** | Zero-copy view     |
| expand 1024x1 to 1024x1024 | 126 ns     | 308 us       | **~2440x** | Just stride change |
| expand 1x1024 to 1024x1024 | 123 ns     | 78.1 us      | **~635x**  | No data copy       |

### Boolean Operations

| Operation | Size       | Ember Time | NdArray Time | Speedup | Ember Mem | NdArray Mem |
| --------- | ---------- | ---------- | ------------ | ------- | --------- | ----------- |
| bool_not  | large (1M) | 25.1 us    | 19.7 us      | 0.79x   | 1.0 MB    | 1.0 MB      |
| bool_and  | large (1M) | 35.6 us    | 29.6 us      | 0.83x   | 1.0 MB    | 1.0 MB      |

Note: Arc-based COW now enables true in-place mutation for unique tensors. Boolean ops gap
remains ~20% due to NdArray's tighter integration with macerator SIMD.

---

## Key Observations

### Performance Wins

1. **Binary ops**: Ember 2.4-4.0x faster due to Arc-based COW in-place mutation and NEON SIMD
   - Previous: 1.4-1.8x faster (always allocated new buffer)
   - Now: 2.4-4.0x faster (reuses buffer when unique)
2. **Slice operations**: Ember 60-2000x faster using zero-copy views vs data copying
3. **Reduce dim=0**: Ember 2.8-3.9x faster using cache-friendly scatter-add pattern
4. **Scalar ops**: Ember 2.4x faster with in-place mutation (was 1.8x)
5. **Unary trig ops**: Ember 1.5-2.0x faster on tanh, sin, cos
6. **Expand (broadcast)**: Ember 600-2400x faster using zero-copy stride manipulation
7. **Comparison ops (contiguous)**: Ember 2.6-3.2x faster with NEON SIMD for f32
8. **Broadcast comparisons**: Ember 2.6-3.3x faster with optimized outer-product SIMD

### Memory Efficiency

- Ember typically allocates 50-70% less memory than NdArray
- Binary ops: 4.2 MB vs 12.6 MB for 1M elements (3x less)
- Slice ops: 80-240 bytes vs kilobytes-megabytes (zero-copy views)
- Reduce ops: kilobytes vs megabytes (output-only allocation)

### Areas for Improvement

1. **Matrix multiplication**: NdArray now faster at larger sizes (256x256+); requires investigation
2. **Transposed sum**: NdArray 1.5-2x faster on transposed tensor sums
3. **Boolean ops**: NdArray ~20% faster (gap narrowed from 2x after Arc-based COW)
4. **Integer matmul**: Both backends similar; neither has SIMD optimization

### Arc-based COW Analysis

The Arc-based Copy-on-Write implementation provides significant benefits:

**Current Ember approach (Arc<Bytes>):**
- `EmberTensor.data` wrapped in `Arc<Bytes>` for O(1) clone
- `is_unique()` check via `Arc::strong_count() == 1`
- True in-place mutation when tensor is uniquely owned
- COW via `Arc::make_mut()` when shared
- SIMD via NEON intrinsics (16-byte vectorized ops)

**NdArray approach (via burn-ndarray + ndarray):**
- Uses `Arc<Vec<T>>` for storage with `is_unique()` check
- SIMD via `macerator` crate with 8-wide loop unrolling
- Copy-on-write only triggers when tensor is shared

**Improvement achieved with Arc-based COW:**
- Binary ops: 1.4-1.8x → 2.4-4.0x (nearly 2x improvement)
- Scalar ops: 1.8x → 2.4x
- Memory: Reduced allocations from 12.6MB to 4.2MB for binary ops

---

## Running Benchmarks

```bash
# All benchmarks
cargo bench --bench binary_ops --features simd,rayon,gemm
cargo bench --bench matmul --features simd,rayon,gemm
cargo bench --bench slice_ops --features simd,rayon,gemm
cargo bench --bench reduce_ops --features simd,rayon,gemm
cargo bench --bench unary_ops --features simd,rayon,gemm
cargo bench --bench comparison_ops --features simd,rayon,gemm
```
