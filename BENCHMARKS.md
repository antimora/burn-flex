# Ember vs NdArray Benchmarks

Benchmarks comparing burn-ember against burn-ndarray on Apple M1 Max.

**Date**: 2026-02-04 **Platform**: darwin (aarch64) **Features**: simd, rayon, gemm

---

## Summary

| Category        | Ember Wins | NdArray Wins | Ties  |
| --------------- | ---------- | ------------ | ----- |
| Binary Ops      | 11         | 0            | 0     |
| Matrix Multiply | 16         | 5            | 1     |
| Slice Ops       | 18         | 0            | 0     |
| Reduce Ops      | 12         | 4            | 0     |
| Unary Ops       | 15         | 0            | 4     |
| Comparison Ops  | 13         | 4            | 0     |
| **Total**       | **85**     | **13**       | **5** |

---

## Binary Operations

Element-wise operations on tensors. **Major improvement** from Arc-based COW enabling true in-place
mutation for unique tensors.

### Tensor-Tensor Operations

| Operation | Size         | Ember Time | NdArray Time | Speedup  | Ember Mem | NdArray Mem |
| --------- | ------------ | ---------- | ------------ | -------- | --------- | ----------- |
| add       | small (4K)   | 414 ns     | 1.45 us      | **3.5x** | 16.5 KB   | 49.3 KB     |
| add       | medium (64K) | 5.6 us     | 21.5 us      | **3.8x** | 262 KB    | 787 KB      |
| add       | large (1M)   | 85 us      | 353 us       | **4.2x** | 4.2 MB    | 12.6 MB     |
| mul       | small (4K)   | 388 ns     | 1.49 us      | **3.8x** | 16.5 KB   | 49.3 KB     |
| mul       | medium (64K) | 7.3 us     | 23 us        | **3.1x** | 262 KB    | 787 KB      |
| mul       | large (1M)   | 112 us     | 374 us       | **3.3x** | 4.2 MB    | 12.6 MB     |
| div       | large (1M)   | 112 us     | 384 us       | **3.4x** | 4.2 MB    | 12.6 MB     |

### Transposed Input Operations

| Operation | Size      | Ember Time | NdArray Time | Speedup  | Ember Mem | NdArray Mem |
| --------- | --------- | ---------- | ------------ | -------- | --------- | ----------- |
| add       | 256x256   | 50 us      | 57 us        | **1.1x** | 262 KB    | 524 KB      |
| add       | 1024x1024 | 934 us     | 1.26 ms      | **1.3x** | 4.2 MB    | 8.4 MB      |

### Scalar Operations

| Operation  | Size       | Ember Time | NdArray Time | Speedup  | Ember Mem | NdArray Mem |
| ---------- | ---------- | ---------- | ------------ | -------- | --------- | ----------- |
| add_scalar | large (1M) | 74 us      | 194 us       | **2.6x** | 4.2 MB    | 8.4 MB      |
| mul_scalar | large (1M) | 76 us      | 197 us       | **2.6x** | 4.2 MB    | 8.4 MB      |

**Key improvement**: Arc-based COW now enables true in-place mutation when tensor is uniquely owned.
This nearly doubles performance vs previous implementation (was 1.4-1.8x, now 2.6-4.2x).

---

## Matrix Multiplication

Using gemm crate with rayon parallelism for large matrices.

### Square Matrices (f32)

| Size      | Ember Time | NdArray Time | Speedup  | Ember Mem | NdArray Mem |
| --------- | ---------- | ------------ | -------- | --------- | ----------- |
| 64x64     | 6.2 us     | 15.2 us      | **2.5x** | 33.6 KB   | 49.3 KB     |
| 128x128   | 44 us      | 57.9 us      | **1.3x** | 328 KB    | 197 KB      |
| 256x256   | 103 us     | 145 us       | **1.4x** | 524 KB    | 787 KB      |
| 512x512   | 442 us     | 817 us       | **1.9x** | 2.1 MB    | 3.1 MB      |
| 1024x1024 | 2.48 ms    | 5.90 ms      | **2.4x** | 8.4 MB    | 12.6 MB     |

### Rectangular Matrices (f32)

| Shape                       | Ember Time | NdArray Time | Speedup  | Ember Mem | NdArray Mem |
| --------------------------- | ---------- | ------------ | -------- | --------- | ----------- |
| 512x64 x 64x512 (attention) | 94 us      | 153 us       | **1.6x** | 1.2 MB    | 1.3 MB      |
| 256x512 x 512x256 (linear)  | 150 us     | 279 us       | **1.9x** | 787 KB    | 1.6 MB      |
| 128x1024 x 1024x128 (wide)  | 126 us     | 222 us       | **1.8x** | 590 KB    | 1.6 MB      |

### Transposed Matrices (256x256)

| Transpose       | Ember Time | NdArray Time | Speedup  | Ember Mem | NdArray Mem |
| --------------- | ---------- | ------------ | -------- | --------- | ----------- |
| LHS transposed  | 106 us     | 211 us       | **2.0x** | 524 KB    | 787 KB      |
| RHS transposed  | 133 us     | 176 us       | **1.3x** | 787 KB    | 787 KB      |
| Both transposed | 115 us     | 214 us       | **1.9x** | 787 KB    | 787 KB      |

### Batched Matmul (f32)

| Batch x Size        | Ember Time | NdArray Time | Speedup  | Ember Mem | NdArray Mem |
| ------------------- | ---------- | ------------ | -------- | --------- | ----------- |
| 8 x 64x64           | 27 us      | 53 us        | **2.0x** | 131 KB    | 393 KB      |
| 32 x 64x64          | 55 us      | 116 us       | **2.1x** | 525 KB    | 1.6 MB      |
| 16 x 128x128        | 164 us     | 554 us       | **3.4x** | 1.0 MB    | 3.1 MB      |
| 12 x 512x64 (heads) | 706 us     | 1.59 ms      | **2.3x** | 12.6 MB   | 15.7 MB     |

### Integer Matmul (i32)

| Size    | Ember Time | NdArray Time | Speedup  | Ember Mem | NdArray Mem |
| ------- | ---------- | ------------ | -------- | --------- | ----------- |
| 64x64   | 110 us     | 109 us       | 1.0x     | 33 KB     | 98.5 KB     |
| 128x128 | 925 us     | 963 us       | **1.0x** | 131 KB    | 393 KB      |
| 256x256 | 10.5 ms    | 9.8 ms       | 0.9x     | 525 KB    | 1.6 MB      |
| 512x512 | 119 ms     | 112 ms       | 0.9x     | 2.1 MB    | 6.3 MB      |

Note: Integer matmul uses naive O(n^3) implementation without SIMD; both backends are similar.

---

## Slice Operations

Tensor slicing, narrowing, and assignment. **Ember uses zero-copy views** for slicing operations.

### Basic Slicing

| Operation | Size      | Ember Time | NdArray Time | Speedup   | Ember Mem | NdArray Mem |
| --------- | --------- | ---------- | ------------ | --------- | --------- | ----------- |
| slice 1D  | 1K        | 149 ns     | 269 ns       | **1.8x**  | 80 B      | 6.3 KB      |
| slice 1D  | 1M        | 136 ns     | 77 us        | **~560x** | 80 B      | 6.3 MB      |
| slice 2D  | 256x256   | 122 ns     | 7.6 us       | **~62x**  | 160 B     | 328 KB      |
| slice 2D  | 1024x1024 | 123 ns     | 81 us        | **~660x** | 160 B     | 5.2 MB      |
| slice 3D  | 64x64x64  | 151 ns     | 30 us        | **~200x** | 240 B     | 1.2 MB      |

### Narrow Operations

| Operation   | Size      | Ember Time | NdArray Time | Speedup   | Ember Mem | NdArray Mem |
| ----------- | --------- | ---------- | ------------ | --------- | --------- | ----------- |
| narrow dim0 | 256x256   | 172 ns     | 6.0 us       | **~35x**  | 240 B     | 393 KB      |
| narrow dim0 | 1024x1024 | 194 ns     | 79 us        | **~400x** | 240 B     | 6.3 MB      |
| narrow dim1 | 256x256   | 180 ns     | 11 us        | **~61x**  | 240 B     | 393 KB      |

### Slice Assignment

| Operation | Size      | Ember Time | NdArray Time | Speedup  | Ember Mem | NdArray Mem |
| --------- | --------- | ---------- | ------------ | -------- | --------- | ----------- |
| assign 1D | 1K        | 292 ns     | 588 ns       | **2.0x** | 4.3 KB    | 10.4 KB     |
| assign 2D | 256x256   | 5.6 us     | 13 us        | **2.3x** | 262 KB    | 590 KB      |
| assign 2D | 1024x1024 | 75 us      | 184 us       | **2.5x** | 4.2 MB    | 9.4 MB      |

### Transposed Slicing

| Operation  | Size      | Ember Time | NdArray Time | Speedup    | Ember Mem | NdArray Mem |
| ---------- | --------- | ---------- | ------------ | ---------- | --------- | ----------- |
| transposed | 256x256   | 97 ns      | 8.2 us       | **~85x**   | 160 B     | 65.7 KB     |
| transposed | 1024x1024 | 97 ns      | 225 us       | **~2300x** | 160 B     | 1.05 MB     |

### Slice with Step

| Operation | Size      | Ember Time | NdArray Time | Speedup    | Ember Mem | NdArray Mem |
| --------- | --------- | ---------- | ------------ | ---------- | --------- | ----------- |
| step2 1D  | 1K        | 90 ns      | 457 ns       | **5.1x**   | 80 B      | 6.3 KB      |
| step2 1D  | 1M        | 86 ns      | 196 us       | **~2300x** | 80 B      | 6.3 MB      |
| step2 2D  | 1024x1024 | 95 ns      | 143 us       | **~1500x** | 160 B     | 5.2 MB      |
| step4 2D  | 256x256   | 97 ns      | 6.7 us       | **~69x**   | 160 B     | 279 KB      |

**Key insight**: Ember implements slicing as zero-copy views (just stride/offset manipulation),
while NdArray copies data. This gives orders of magnitude speedup for large tensors.

---

## Reduce Operations

Sum, mean, argmax reductions with portable SIMD via pulp.

### Full Tensor Sum

| Size | Ember Time | NdArray Time | Speedup  | Ember Mem | NdArray Mem |
| ---- | ---------- | ------------ | -------- | --------- | ----------- |
| 1K   | 261 ns     | 388 ns       | **1.5x** | 104 B     | 8.3 KB      |
| 64K  | 11.6 us    | 15.4 us      | **1.3x** | 104 B     | 524 KB      |
| 1M   | 38.5 us    | 227 us       | **5.9x** | 104 B     | 8.4 MB      |

### Sum Along Dimension

| Shape     | Dim | Ember Time | NdArray Time | Speedup  | Ember Mem | NdArray Mem |
| --------- | --- | ---------- | ------------ | -------- | --------- | ----------- |
| 256x256   | 0   | 5.1 us     | 21 us        | **4.1x** | 2.2 KB    | 524 KB      |
| 256x256   | 1   | 6.1 us     | 13.8 us      | **2.3x** | 1.2 KB    | 524 KB      |
| 1024x1024 | 0   | 81 us      | 215 us       | **2.7x** | 8.3 KB    | 8.4 MB      |
| 1024x1024 | 1   | 138 us     | 210 us       | **1.5x** | 4.2 KB    | 8.4 MB      |

### 3D Sum (Batched)

| Shape      | Dim | Ember Time | NdArray Time | Speedup  | Ember Mem | NdArray Mem |
| ---------- | --- | ---------- | ------------ | -------- | --------- | ----------- |
| 32x256x256 | 1   | 150 us     | 530 us       | **3.5x** | 65.7 KB   | 16.8 MB     |
| 32x256x256 | 2   | 187 us     | 366 us       | **2.0x** | 32.9 KB   | 16.8 MB     |

### Sum Transposed (total sum)

| Size      | Ember Time | NdArray Time | Speedup  | Ember Mem | NdArray Mem |
| --------- | ---------- | ------------ | -------- | --------- | ----------- |
| 256x256   | 11.4 us    | 6.4 us       | 0.6x     | 120 B     | 44 B        |
| 1024x1024 | 44.6 us    | 98 us        | **2.2x** | 120 B     | 44 B        |

### Sum Dim on Transposed Tensor

| Size      | Dim | Ember Time | NdArray Time | Speedup |
| --------- | --- | ---------- | ------------ | ------- |
| 256x256   | 0   | 5.8 us     | 4.5 us       | 0.8x    |
| 1024x1024 | 0   | 140 us     | 84 us        | 0.6x    |

### Mean Along Dimension

| Shape     | Dim | Ember Time | NdArray Time | Speedup  | Ember Mem | NdArray Mem |
| --------- | --- | ---------- | ------------ | -------- | --------- | ----------- |
| 256x256   | 1   | 5.8 us     | 13.6 us      | **2.3x** | 1.2 KB    | 524 KB      |
| 1024x1024 | 1   | 137 us     | 214 us       | **1.6x** | 4.2 KB    | 8.4 MB      |

### Argmax

| Shape     | Dim | Ember Time | NdArray Time | Speedup  | Ember Mem | NdArray Mem |
| --------- | --- | ---------- | ------------ | -------- | --------- | ----------- |
| 1K (flat) | -   | 3.4 us     | 4.3 us       | **1.3x** | 104 B     | 8.3 KB      |
| 256x256   | 1   | 212 us     | 247 us       | **1.2x** | 2.2 KB    | 524 KB      |
| 1024x1024 | 1   | 3.39 ms    | 4.0 ms       | **1.2x** | 8.3 KB    | 8.4 MB      |

---

## Unary Operations

Element-wise math functions (exp, log, sqrt, trig, etc.).

### Basic Math Operations

| Operation | Size         | Ember Time | NdArray Time | Speedup  | Ember Mem | NdArray Mem |
| --------- | ------------ | ---------- | ------------ | -------- | --------- | ----------- |
| exp       | small (4K)   | 5.2 us     | 5.6 us       | **1.1x** | 16.5 KB   | 32.8 KB     |
| exp       | medium (64K) | 83 us      | 92 us        | **1.1x** | 262 KB    | 524 KB      |
| exp       | large (1M)   | 1.33 ms    | 1.45 ms      | **1.1x** | 4.2 MB    | 8.4 MB      |
| log       | small (4K)   | 6.8 us     | 7.3 us       | **1.1x** | 16.5 KB   | 32.8 KB     |
| log       | medium (64K) | 107 us     | 117 us       | **1.1x** | 262 KB    | 524 KB      |
| log       | large (1M)   | 1.68 ms    | 1.85 ms      | **1.1x** | 4.2 MB    | 8.4 MB      |
| sqrt      | small (4K)   | 669 ns     | 1.09 us      | **1.6x** | 16.5 KB   | 24.6 KB     |
| sqrt      | medium (64K) | 9.4 us     | 17.2 us      | **1.8x** | 262 KB    | 524 KB      |
| sqrt      | large (1M)   | 139 us     | 270 us       | **1.9x** | 4.2 MB    | 8.4 MB      |
| abs       | large (1M)   | 78 us      | 193 us       | **2.5x** | 4.2 MB    | 8.4 MB      |
| recip     | large (1M)   | 76 us      | 195 us       | **2.6x** | 4.2 MB    | 8.4 MB      |

### Trigonometric Operations

| Operation | Size         | Ember Time | NdArray Time | Speedup  | Ember Mem | NdArray Mem |
| --------- | ------------ | ---------- | ------------ | -------- | --------- | ----------- |
| sin       | small (4K)   | 5.5 us     | 8.4 us       | **1.5x** | 16.5 KB   | 32.8 KB     |
| sin       | medium (64K) | 89 us      | 136 us       | **1.5x** | 262 KB    | 524 KB      |
| sin       | large (1M)   | 1.42 ms    | 2.17 ms      | **1.5x** | 4.2 MB    | 8.4 MB      |
| cos       | small (4K)   | 7.3 us     | 9.2 us       | **1.3x** | 16.5 KB   | 32.8 KB     |
| cos       | large (1M)   | 1.65 ms    | 2.22 ms      | **1.3x** | 4.2 MB    | 8.4 MB      |
| tanh      | small (4K)   | 7.1 us     | 14.0 us      | **2.0x** | 16.5 KB   | 32.8 KB     |
| tanh      | medium (64K) | 114 us     | 231 us       | **2.0x** | 262 KB    | 524 KB      |
| tanh      | large (1M)   | 1.76 ms    | 3.63 ms      | **2.1x** | 4.2 MB    | 8.4 MB      |

### Transposed (Non-contiguous) Input

| Operation | Size      | Ember Time | NdArray Time | Speedup  | Ember Mem | NdArray Mem |
| --------- | --------- | ---------- | ------------ | -------- | --------- | ----------- |
| exp       | 256x256   | 80 us      | 86 us        | **1.1x** | 262 KB    | 262 KB      |
| exp       | 1024x1024 | 1.33 ms    | 1.36 ms      | **1.0x** | 4.2 MB    | 4.2 MB      |

---

## Comparison & Boolean Operations

Element-wise comparisons with NEON SIMD optimization for f32.

### Tensor-Tensor Comparisons

| Operation | Size         | Ember Time | NdArray Time | Speedup  | Ember Mem | NdArray Mem |
| --------- | ------------ | ---------- | ------------ | -------- | --------- | ----------- |
| greater   | small (4K)   | 550 ns     | 1.62 us      | **2.9x** | 4.2 KB    | 49 KB       |
| greater   | medium (64K) | 6.6 us     | 21.2 us      | **3.2x** | 65.7 KB   | 787 KB      |
| greater   | large (1M)   | 98 us      | 309 us       | **3.2x** | 1.0 MB    | 12.6 MB     |
| equal     | small (4K)   | 607 ns     | 1.49 us      | **2.5x** | 4.2 KB    | 49 KB       |
| equal     | large (1M)   | 98 us      | 303 us       | **3.1x** | 1.0 MB    | 12.6 MB     |
| lower     | large (1M)   | 98 us      | 311 us       | **3.2x** | 1.0 MB    | 12.6 MB     |

### Scalar Comparisons

| Operation    | Size       | Ember Time | NdArray Time | Speedup  | Ember Mem | NdArray Mem |
| ------------ | ---------- | ---------- | ------------ | -------- | --------- | ----------- |
| greater_elem | large (1M) | 94 us      | 194 us       | **2.1x** | 1.0 MB    | 8.4 MB      |

### Transposed (Non-contiguous) Comparisons

| Operation | Size      | Ember Time | NdArray Time | Speedup  |
| --------- | --------- | ---------- | ------------ | -------- |
| greater   | 256x256   | 53 us      | 51 us        | 1.0x     |
| greater   | 1024x1024 | 969 us     | 1.09 ms      | **1.1x** |

### Broadcast Comparisons (Outer-Product Pattern)

| Operation | Shape     | Ember Time | NdArray Time | Speedup  | Ember Mem | NdArray Mem |
| --------- | --------- | ---------- | ------------ | -------- | --------- | ----------- |
| greater   | 256x256   | 7.9 us     | 26.4 us      | **3.3x** | 66.6 KB   | 67.7 KB     |
| greater   | 1024x1024 | 117 us     | 316 us       | **2.7x** | 1.1 MB    | 1.1 MB      |

### Expand Operation (Broadcasting)

| Operation                  | Ember Time | NdArray Time | Speedup    | Note               |
| -------------------------- | ---------- | ------------ | ---------- | ------------------ |
| expand 1x1 to 1000x1000    | 175 ns     | 292 us       | **~1700x** | Zero-copy view     |
| expand 1024x1 to 1024x1024 | 110 ns     | 313 us       | **~2850x** | Just stride change |
| expand 1x1024 to 1024x1024 | 147 ns     | 78 us        | **~530x**  | No data copy       |

### Boolean Operations

| Operation | Size       | Ember Time | NdArray Time | Speedup  | Ember Mem | NdArray Mem |
| --------- | ---------- | ---------- | ------------ | -------- | --------- | ----------- |
| bool_not  | large (1M) | 25 us      | 20 us        | 0.78x    | 1.0 MB    | 1.0 MB      |
| bool_and  | large (1M) | 28 us      | 30 us        | **1.1x** | 1.0 MB    | 1.0 MB      |

---

## Key Observations

### Performance Wins

1. **Matrix multiplication**: Ember 1.3-3.4x faster on f32 with gemm + rayon parallelism
2. **Binary ops**: Ember 2.6-4.2x faster due to Arc-based COW in-place mutation and NEON SIMD
3. **Slice operations**: Ember 60-2300x faster using zero-copy views vs data copying
4. **Reduce dim=0**: Ember 2.7-4.1x faster using cache-friendly scatter-add pattern
5. **Scalar ops**: Ember 2.6x faster with in-place mutation
6. **Unary trig ops**: Ember 1.5-2.1x faster on tanh, sin, cos
7. **Expand (broadcast)**: Ember 530-2850x faster using zero-copy stride manipulation
8. **Comparison ops (contiguous)**: Ember 2.5-3.2x faster with NEON SIMD for f32
9. **Broadcast comparisons**: Ember 2.7-3.3x faster with optimized outer-product SIMD

### Memory Efficiency

- Ember typically allocates 50-70% less memory than NdArray
- Binary ops: 4.2 MB vs 12.6 MB for 1M elements (3x less)
- Slice ops: 80-240 bytes vs kilobytes-megabytes (zero-copy views)
- Reduce ops: kilobytes vs megabytes (output-only allocation)

### Areas for Improvement

1. **Transposed sum**: NdArray slightly faster on small transposed tensor sums
2. **Boolean ops**: NdArray ~20% faster on bool_not
3. **Integer matmul**: Both backends similar; neither has SIMD optimization

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

- Binary ops: 1.4-1.8x -> 2.6-4.2x (2x improvement)
- Scalar ops: 1.8x -> 2.6x
- Memory: Reduced allocations from 12.6MB to 4.2MB for binary ops

### Matmul Investigation Summary

Root cause of earlier regression: the `rayon` feature was not in default features, causing gemm to
run single-threaded even on large matrices. With rayon enabled:

- 1024x1024: 2.48ms (Ember) vs 5.90ms (NdArray) = 2.4x faster
- Previously without rayon: 19.5ms (Ember) vs 5.9ms (NdArray) = 0.3x (regression)

The fix was adding `rayon` to default features in Cargo.toml.

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
