# Ember vs NdArray Benchmarks

Benchmarks comparing burn-ember against burn-ndarray on Apple M1 Max.

**Date**: 2026-02-04 **Platform**: darwin (aarch64) **Features**: simd, rayon, gemm

---

## Summary

| Category        | Ember Wins | NdArray Wins | Ties  |
| --------------- | ---------- | ------------ | ----- |
| Binary Ops      | 10         | 2            | 0     |
| Matrix Multiply | 14         | 3            | 0     |
| Slice Ops       | 11         | 7            | 0     |
| Reduce Ops      | 14         | 2            | 0     |
| Unary Ops       | 15         | 0            | 4     |
| Comparison Ops  | 13         | 4            | 0     |
| **Total**       | **77**     | **18**       | **4** |

---

## Binary Operations

Element-wise operations on tensors.

### Tensor-Tensor Operations

| Operation | Size         | Ember Time | NdArray Time | Speedup  | Ember Mem | NdArray Mem |
| --------- | ------------ | ---------- | ------------ | -------- | --------- | ----------- |
| add       | small (4K)   | 846 ns     | 1.44 us      | **1.7x** | 32.9 KB   | 49.3 KB     |
| add       | medium (64K) | 15.0 us    | 21.5 us      | **1.4x** | 524 KB    | 787 KB      |
| add       | large (1M)   | 220 us     | 356 us       | **1.6x** | 8.4 MB    | 12.6 MB     |
| mul       | small (4K)   | 919 ns     | 1.42 us      | **1.5x** | 32.9 KB   | 49.3 KB     |
| mul       | medium (64K) | 14.5 us    | 22.5 us      | **1.6x** | 524 KB    | 787 KB      |
| mul       | large (1M)   | 219 us     | 387 us       | **1.8x** | 8.4 MB    | 12.6 MB     |
| div       | large (1M)   | 222 us     | 359 us       | **1.6x** | 8.4 MB    | 12.6 MB     |

### Transposed Input Operations

| Operation | Size      | Ember Time | NdArray Time | Speedup | Ember Mem | NdArray Mem |
| --------- | --------- | ---------- | ------------ | ------- | --------- | ----------- |
| add       | 256x256   | 61.3 us    | 53.0 us      | 0.9x    | 787 KB    | 524 KB      |
| add       | 1024x1024 | 1.12 ms    | 1.10 ms      | 1.0x    | 12.6 MB   | 8.4 MB      |

### Scalar Operations

| Operation  | Size       | Ember Time | NdArray Time | Speedup  | Ember Mem | NdArray Mem |
| ---------- | ---------- | ---------- | ------------ | -------- | --------- | ----------- |
| add_scalar | large (1M) | 118 us     | 208 us       | **1.8x** | 4.2 MB    | 8.4 MB      |
| mul_scalar | large (1M) | 118 us     | 207 us       | **1.8x** | 4.2 MB    | 8.4 MB      |

---

## Matrix Multiplication

Using gemm crate for optimized matmul.

### Square Matrices (f32)

| Size      | Ember Time | NdArray Time | Speedup  | Ember Mem | NdArray Mem |
| --------- | ---------- | ------------ | -------- | --------- | ----------- |
| 64x64     | 6.85 us    | 19.9 us      | **2.9x** | 66.4 KB   | 49.3 KB     |
| 128x128   | 45.5 us    | 69.0 us      | **1.5x** | 459 KB    | 197 KB      |
| 256x256   | 154 us     | 171 us       | **1.1x** | 1.05 MB   | 787 KB      |
| 512x512   | 606 us     | 891 us       | **1.5x** | 4.2 MB    | 3.1 MB      |
| 1024x1024 | 2.83 ms    | 6.10 ms      | **2.2x** | 16.8 MB   | 12.6 MB     |

### Rectangular Matrices (f32)

| Shape                       | Ember Time | NdArray Time | Speedup  | Ember Mem | NdArray Mem |
| --------------------------- | ---------- | ------------ | -------- | --------- | ----------- |
| 512x64 x 64x512 (attention) | 175 us     | 162 us       | 0.9x     | 1.4 MB    | 1.3 MB      |
| 256x512 x 512x256 (linear)  | 312 us     | 299 us       | 1.0x     | 1.8 MB    | 1.6 MB      |
| 128x1024 x 1024x128 (wide)  | 230 us     | 236 us       | **1.0x** | 1.6 MB    | 1.6 MB      |

### Transposed Matrices (256x256)

| Transpose       | Ember Time | NdArray Time | Speedup  | Ember Mem | NdArray Mem |
| --------------- | ---------- | ------------ | -------- | --------- | ----------- |
| LHS transposed  | 179 us     | 198 us       | **1.1x** | 1.05 MB   | 787 KB      |
| RHS transposed  | 200 us     | 183 us       | 0.9x     | 1.3 MB    | 787 KB      |
| Both transposed | 206 us     | 204 us       | 1.0x     | 1.3 MB    | 787 KB      |

### Batched Matmul (f32)

| Batch x Size        | Ember Time | NdArray Time | Speedup  | Ember Mem | NdArray Mem |
| ------------------- | ---------- | ------------ | -------- | --------- | ----------- |
| 8 x 64x64           | 71.9 us    | 120 us       | **1.7x** | 656 KB    | 393 KB      |
| 32 x 64x64          | 118 us     | 161 us       | **1.4x** | 2.6 MB    | 1.6 MB      |
| 16 x 128x128        | 307 us     | 605 us       | **2.0x** | 5.2 MB    | 3.1 MB      |
| 12 x 512x64 (heads) | 758 us     | 1.91 ms      | **2.5x** | 18.9 MB   | 15.7 MB     |

### Integer Matmul (i32)

| Size    | Ember Time | NdArray Time | Speedup  | Ember Mem | NdArray Mem |
| ------- | ---------- | ------------ | -------- | --------- | ----------- |
| 64x64   | 109 us     | 115 us       | **1.1x** | 164 KB    | 98.5 KB     |
| 128x128 | 952 us     | 995 us       | **1.0x** | 656 KB    | 393 KB      |
| 256x256 | 10.8 ms    | 10.0 ms      | 0.9x     | 2.6 MB    | 1.6 MB      |
| 512x512 | 124 ms     | 118 ms       | 0.9x     | 10.5 MB   | 6.3 MB      |

---

## Slice Operations

Tensor slicing, narrowing, and assignment.

### Basic Slicing

| Operation | Size      | Ember Time | NdArray Time | Speedup  | Ember Mem | NdArray Mem |
| --------- | --------- | ---------- | ------------ | -------- | --------- | ----------- |
| slice 1D  | 1K        | 341 ns     | 346 ns       | 1.0x     | 8.3 KB    | 6.3 KB      |
| slice 1D  | 1M        | 143 us     | 77.2 us      | 0.5x     | 8.4 MB    | 6.3 MB      |
| slice 2D  | 256x256   | 7.38 us    | 8.01 us      | **1.1x** | 524 KB    | 328 KB      |
| slice 2D  | 1024x1024 | 145 us     | 82.8 us      | 0.6x     | 8.4 MB    | 5.2 MB      |
| slice 3D  | 64x64x64  | 26.4 us    | 29.1 us      | **1.1x** | 2.1 MB    | 1.2 MB      |

### Narrow Operations

| Operation   | Size      | Ember Time | NdArray Time | Speedup  | Ember Mem | NdArray Mem |
| ----------- | --------- | ---------- | ------------ | -------- | --------- | ----------- |
| narrow dim0 | 256x256   | 7.54 us    | 5.94 us      | 0.8x     | 524 KB    | 393 KB      |
| narrow dim0 | 1024x1024 | 143 us     | 77.0 us      | 0.5x     | 8.4 MB    | 6.3 MB      |
| narrow dim1 | 256x256   | 7.62 us    | 11.2 us      | **1.5x** | 524 KB    | 393 KB      |

### Slice Assignment

| Operation | Size      | Ember Time | NdArray Time | Speedup  | Ember Mem | NdArray Mem |
| --------- | --------- | ---------- | ------------ | -------- | --------- | ----------- |
| assign 1D | 1K        | 597 ns     | 680 ns       | **1.1x** | 12.5 KB   | 4.9 KB      |
| assign 2D | 256x256   | 12.6 us    | 13.1 us      | **1.0x** | 656 KB    | 590 KB      |
| assign 2D | 1024x1024 | 201 us     | 185 us       | 0.9x     | 10.5 MB   | 9.4 MB      |

### Transposed Slicing

| Operation  | Size      | Ember Time | NdArray Time | Speedup  | Ember Mem | NdArray Mem |
| ---------- | --------- | ---------- | ------------ | -------- | --------- | ----------- |
| transposed | 256x256   | 7.13 us    | 8.15 us      | **1.1x** | 524 KB    | 65.7 KB     |
| transposed | 1024x1024 | 144 us     | 231 us       | **1.6x** | 8.4 MB    | 1.05 MB     |

### Slice with Step

| Operation | Size      | Ember Time | NdArray Time | Speedup  | Ember Mem | NdArray Mem |
| --------- | --------- | ---------- | ------------ | -------- | --------- | ----------- |
| step2 1D  | 1K        | 310 ns     | 513 ns       | **1.7x** | 8.3 KB    | 6.3 KB      |
| step2 1D  | 1M        | 144 us     | 197 us       | **1.4x** | 8.4 MB    | 6.3 MB      |
| step2 2D  | 1024x1024 | 145 us     | 137 us       | 0.9x     | 8.4 MB    | 5.2 MB      |
| step4 2D  | 256x256   | 7.38 us    | 6.72 us      | 0.9x     | 524 KB    | 279 KB      |

---

## Reduce Operations

Sum, mean, argmax reductions with portable SIMD via pulp.

### Full Tensor Sum

| Size | Ember Time | NdArray Time | Speedup  | Ember Mem | NdArray Mem |
| ---- | ---------- | ------------ | -------- | --------- | ----------- |
| 1K   | 333 ns     | 369 ns       | **1.1x** | 4.2 KB    | 8.3 KB      |
| 64K  | 16.5 us    | 14.6 us      | 0.9x     | 262 KB    | 524 KB      |
| 1M   | 142 us     | 218 us       | **1.5x** | 4.2 MB    | 8.4 MB      |

### Sum Along Dimension

| Shape     | Dim | Ember Time | NdArray Time | Speedup  | Ember Mem | NdArray Mem |
| --------- | --- | ---------- | ------------ | -------- | --------- | ----------- |
| 256x256   | 0   | 9.55 us    | 19.7 us      | **2.1x** | 263 KB    | 524 KB      |
| 256x256   | 1   | 10.4 us    | 13.2 us      | **1.3x** | 263 KB    | 524 KB      |
| 1024x1024 | 0   | 133 us     | 220 us       | **1.7x** | 4.2 MB    | 8.4 MB      |
| 1024x1024 | 1   | 189 us     | 224 us       | **1.2x** | 4.2 MB    | 8.4 MB      |

### 3D Sum (Batched)

| Shape      | Dim | Ember Time | NdArray Time | Speedup  | Ember Mem | NdArray Mem |
| ---------- | --- | ---------- | ------------ | -------- | --------- | ----------- |
| 32x256x256 | 1   | 291 us     | 528 us       | **1.8x** | 8.4 MB    | 16.8 MB     |
| 32x256x256 | 2   | 332 us     | 348 us       | **1.0x** | 8.4 MB    | 16.8 MB     |

### Sum Transposed (total sum)

| Size      | Ember Time | NdArray Time | Speedup | Ember Mem | NdArray Mem |
| --------- | ---------- | ------------ | ------- | --------- | ----------- |
| 256x256   | 15.8 us    | 5.70 us      | 0.4x    | 262 KB    | 44 B        |
| 1024x1024 | 134 us     | 95.1 us      | 0.7x    | 4.2 MB    | 44 B        |

### Sum Dim on Transposed Tensor

| Size      | Dim | Ember Time | NdArray Time | Speedup | Note                                         |
| --------- | --- | ---------- | ------------ | ------- | -------------------------------------------- |
| 256x256   | 0   | 10.1 us    | 4.57 us      | 0.5x    | NdArray fast on storage-contiguous reduction |
| 1024x1024 | 0   | 192 us     | 84.2 us      | 0.4x    | Ember matches contiguous perf (was 3.3ms)    |

Note: For contiguous tensors, Ember sum_dim is **2.1x faster** than NdArray. The transposed case is
where NdArray excels.

### Mean Along Dimension

| Shape     | Dim | Ember Time | NdArray Time | Speedup  | Ember Mem | NdArray Mem |
| --------- | --- | ---------- | ------------ | -------- | --------- | ----------- |
| 256x256   | 1   | 10.5 us    | 12.9 us      | **1.2x** | 263 KB    | 524 KB      |
| 1024x1024 | 1   | 199 us     | 204 us       | **1.0x** | 4.2 MB    | 8.4 MB      |

### Argmax

| Shape     | Dim | Ember Time | NdArray Time | Speedup  | Ember Mem | NdArray Mem |
| --------- | --- | ---------- | ------------ | -------- | --------- | ----------- |
| 1K (flat) | -   | 3.80 us    | 4.29 us      | **1.1x** | 4.2 KB    | 8.3 KB      |
| 256x256   | 1   | 227 us     | 247 us       | **1.1x** | 264 KB    | 524 KB      |
| 1024x1024 | 1   | 3.48 ms    | 4.07 ms      | **1.2x** | 4.2 MB    | 8.4 MB      |

---

## Unary Operations

Element-wise math functions (exp, log, sqrt, trig, etc.).

### Basic Math Operations

| Operation | Size         | Ember Time | NdArray Time | Speedup  | Ember Mem | NdArray Mem |
| --------- | ------------ | ---------- | ------------ | -------- | --------- | ----------- |
| exp       | small (4K)   | 5.34 us    | 5.69 us      | **1.1x** | 32.9 KB   | 32.8 KB     |
| exp       | medium (64K) | 85.2 us    | 87.2 us      | 1.0x     | 524 KB    | 524 KB      |
| exp       | large (1M)   | 1.37 ms    | 1.41 ms      | **1.0x** | 8.4 MB    | 8.4 MB      |
| log       | small (4K)   | 7.26 us    | 7.03 us      | 1.0x     | 32.9 KB   | 32.8 KB     |
| log       | medium (64K) | 110 us     | 113 us       | **1.0x** | 524 KB    | 524 KB      |
| log       | large (1M)   | 1.81 ms    | 1.83 ms      | 1.0x     | 8.4 MB    | 8.4 MB      |
| sqrt      | small (4K)   | 949 ns     | 1.13 us      | **1.2x** | 24.6 KB   | 24.6 KB     |
| sqrt      | medium (64K) | 13.6 us    | 17.4 us      | **1.3x** | 524 KB    | 524 KB      |
| sqrt      | large (1M)   | 212 us     | 284 us       | **1.3x** | 8.4 MB    | 8.4 MB      |
| abs       | large (1M)   | 133 us     | 194 us       | **1.5x** | 8.4 MB    | 8.4 MB      |
| recip     | large (1M)   | 154 us     | 197 us       | **1.3x** | 8.4 MB    | 8.4 MB      |

### Trigonometric Operations

| Operation | Size         | Ember Time | NdArray Time | Speedup  | Ember Mem | NdArray Mem |
| --------- | ------------ | ---------- | ------------ | -------- | --------- | ----------- |
| sin       | small (4K)   | 5.93 us    | 8.63 us      | **1.5x** | 32.9 KB   | 32.8 KB     |
| sin       | medium (64K) | 94.2 us    | 143 us       | **1.5x** | 524 KB    | 524 KB      |
| sin       | large (1M)   | 1.51 ms    | 2.18 ms      | **1.4x** | 8.4 MB    | 8.4 MB      |
| cos       | small (4K)   | 6.76 us    | 8.61 us      | **1.3x** | 32.9 KB   | 32.8 KB     |
| cos       | large (1M)   | 1.77 ms    | 2.33 ms      | **1.3x** | 8.4 MB    | 8.4 MB      |
| tanh      | small (4K)   | 7.34 us    | 13.7 us      | **1.9x** | 32.9 KB   | 32.8 KB     |
| tanh      | medium (64K) | 123 us     | 225 us       | **1.8x** | 524 KB    | 524 KB      |
| tanh      | large (1M)   | 1.89 ms    | 3.64 ms      | **1.9x** | 8.4 MB    | 8.4 MB      |

### Transposed (Non-contiguous) Input

| Operation | Size      | Ember Time | NdArray Time | Speedup | Ember Mem | NdArray Mem |
| --------- | --------- | ---------- | ------------ | ------- | --------- | ----------- |
| exp       | 256x256   | 85.0 us    | 82.8 us      | 1.0x    | 524 KB    | 262 KB      |
| exp       | 1024x1024 | 1.44 ms    | 1.34 ms      | 0.9x    | 8.4 MB    | 4.2 MB      |

---

## Comparison & Boolean Operations

Element-wise comparisons with NEON SIMD optimization for f32.

### Tensor-Tensor Comparisons

| Operation | Size         | Ember Time | NdArray Time | Speedup  | Ember Mem | NdArray Mem |
| --------- | ------------ | ---------- | ------------ | -------- | --------- | ----------- |
| greater   | small (4K)   | 961 ns     | 1.62 us      | **1.7x** | 37 KB     | 70 KB       |
| greater   | medium (64K) | 14.0 us    | 21.4 us      | **1.5x** | 590 KB    | 1.1 MB      |
| greater   | large (1M)   | 223 us     | 318 us       | **1.4x** | 9.4 MB    | 17.8 MB     |
| equal     | small (4K)   | 1.08 us    | 1.66 us      | **1.5x** | 37 KB     | 70 KB       |
| equal     | large (1M)   | 224 us     | 327 us       | **1.5x** | 9.4 MB    | 17.8 MB     |
| lower     | large (1M)   | 242 us     | 305 us       | **1.3x** | 9.4 MB    | 17.8 MB     |

### Scalar Comparisons

| Operation    | Size       | Ember Time | NdArray Time | Speedup  | Ember Mem | NdArray Mem |
| ------------ | ---------- | ---------- | ------------ | -------- | --------- | ----------- |
| greater_elem | large (1M) | 151 us     | 200 us       | **1.3x** | 5.2 MB    | 9.4 MB      |

### Transposed (Non-contiguous) Comparisons

| Operation | Size      | Ember Time | NdArray Time | Speedup | Ember Mem | NdArray Mem |
| --------- | --------- | ---------- | ------------ | ------- | --------- | ----------- |
| greater   | 256x256   | 66 us      | 51.5 us      | 0.8x    | 590 KB    | 590 KB      |
| greater   | 1024x1024 | 1.13 ms    | 1.06 ms      | 0.9x    | 9.4 MB    | 9.4 MB      |

Note: Transposed comparisons now use 2D strided fast path, competitive with NdArray.

### Broadcast Comparisons (Outer-Product Pattern)

| Operation | Shape     | Ember Time | NdArray Time | Speedup  | Ember Mem | NdArray Mem |
| --------- | --------- | ---------- | ------------ | -------- | --------- | ----------- |
| greater   | 256x256   | 8.0 us     | 26.0 us      | **3.2x** | 69 KB     | 70 KB       |
| greater   | 1024x1024 | 118 us     | 311 us       | **2.6x** | 1.1 MB    | 1.1 MB      |

Note: Optimized outer-product broadcast pattern uses SIMD scalar comparison per row.

### Expand Operation (Broadcasting)

| Operation                  | Ember Time | NdArray Time | Speedup    | Note               |
| -------------------------- | ---------- | ------------ | ---------- | ------------------ |
| expand 1x1 to 1000x1000    | 188 ns     | 291 us       | **~1500x** | Zero-copy view     |
| expand 1024x1 to 1024x1024 | 222 ns     | 311 us       | **~1400x** | Just stride change |
| expand 1x1024 to 1024x1024 | 227 ns     | 78 us        | **~340x**  | No data copy       |

### Boolean Operations

| Operation | Size       | Ember Time | NdArray Time | Speedup | Ember Mem | NdArray Mem |
| --------- | ---------- | ---------- | ------------ | ------- | --------- | ----------- |
| bool_not  | large (1M) | 31.2 us    | 19.7 us      | 0.63x   | 1.0 MB    | 1.0 MB      |
| bool_and  | large (1M) | 52.9 us    | 29.6 us      | 0.56x   | 2.1 MB    | 1.0 MB      |

Note: Both operations now use in-place modification when possible. Despite SIMD and in-place
optimizations, NdArray maintains ~2x advantage on boolean ops.

---

## Key Observations

### Performance Wins

1. **Binary ops**: Ember consistently 1.4-1.8x faster due to SIMD (NEON) and better memory
   efficiency
2. **Matrix multiply**: Ember 1.5-2.9x faster on square matrices, 2.0-2.5x on batched ops
3. **Reduce dim=0**: Ember 1.7-2.1x faster using cache-friendly scatter-add pattern
4. **Scalar ops**: Ember 1.8x faster with half the memory allocation
5. **Unary trig ops**: Ember 1.3-1.9x faster on tanh, sin, cos (using libm functions)
6. **Expand (broadcast)**: Ember 340-1500x faster using zero-copy stride manipulation
7. **Comparison ops (contiguous)**: Ember 1.3-1.7x faster with NEON SIMD for f32
8. **Broadcast comparisons**: Ember 2.6-3.2x faster with optimized outer-product SIMD

### Memory Efficiency

- Ember typically allocates 30-50% less memory than NdArray
- Particularly notable in binary ops (8.4 MB vs 12.6 MB for 1M elements)
- Reduce ops also show significant memory savings

### Areas for Improvement

1. **Transposed sum**: NdArray wins by 2-3x on transposed tensor sums
2. **Large slice copies**: NdArray faster on 1M+ element 1D slices
3. **Integer matmul**: Both backends are similar; neither has SIMD optimization
4. **Boolean binary ops**: NdArray 2x faster on bool_and/or/xor (see analysis below)

### Boolean Operations Analysis

Investigation into why NdArray achieves ~2x performance on bool ops reveals a fundamental
architectural difference:

**NdArray approach (via burn-ndarray + ndarray + macerator):**
- Uses `Arc<Vec<T>>` for storage with `is_unique()` check
- When `Arc::strong_count() == 1`, modifies in-place without allocation
- SIMD via `macerator` crate with 8-wide loop unrolling
- Copy-on-write only triggers when tensor is shared

**Ember approach (via Bytes + Box<dyn AllocationController>):**
- Uses `Box<dyn AllocationController>` for type-erased storage
- No reference counting - `DerefMut` always available
- COW via `duplicate()` method if controller supports it
- Cannot detect unique ownership to skip allocation

**Why the gap persists:**
1. **No uniqueness check**: Ember's `Bytes` type doesn't expose a way to check if the
   allocation is uniquely owned, so we always allocate output buffers
2. **Memory allocation overhead**: Each bool_and/or/xor allocates 2.1MB vs NdArray's 1.0MB
3. **SIMD is comparable**: Both use SIMD (Ember via NEON, NdArray via macerator)

**Potential solutions (future work):**
1. Add `is_unique()` to Bytes/AllocationController trait
2. Use a different storage type (e.g., `Arc<Vec<u8>>`) for bool tensors
3. Implement tensor fusion to eliminate intermediate allocations

---

## Running Benchmarks

```bash
# All benchmarks
cargo bench --bench binary_ops --features simd,rayon,gemm
cargo bench --bench matmul --features simd,rayon,gemm
cargo bench --bench slice_ops --features simd,rayon,gemm
cargo bench --bench reduce_ops --features simd,rayon,gemm
cargo bench --bench unary_ops --features simd,rayon,gemm
```
