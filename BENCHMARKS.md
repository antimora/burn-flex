# burn-ember Benchmarks

Tracking performance compared to burn-ndarray.

## Running Benchmarks

```bash
cargo bench --bench binary_ops
```

## Results (with SIMD)

**Platform:** Apple Silicon M3 (ARM64) **Date:** 2026-02-03

### Contiguous Operations (1M elements, f32)

| Operation  | Ember | NdArray | Speedup |
| ---------- | ----- | ------- | ------- |
| add        | 202µs | 340µs   | 1.7x    |
| mul        | 214µs | 335µs   | 1.6x    |
| div        | 227µs | 334µs   | 1.5x    |
| scalar_add | 117µs | 179µs   | 1.5x    |
| scalar_mul | 117µs | 175µs   | 1.5x    |

### Non-Contiguous Operations (transposed)

| Size      | Ember  | NdArray | Result                |
| --------- | ------ | ------- | --------------------- |
| 256x256   | 58µs   | 53µs    | ~equal                |
| 1024x1024 | 1.06ms | 1.08ms  | ~equal                |

### Optimization History

#### Phase 1: Row-based Iteration (non-contiguous)

**Initial implementation (StridedIter):**

| Size      | Ember | NdArray | Ratio     |
| --------- | ----- | ------- | --------- |
| 256x256   | 388µs | 55µs    | 7x slower |
| 1024x1024 | 6.4ms | 1.2ms   | 5x slower |

**After row-based iteration:**

| Size      | Ember  | NdArray | Ratio       |
| --------- | ------ | ------- | ----------- |
| 256x256   | 58µs   | 54µs    | ~equal      |
| 1024x1024 | 1.09ms | 1.55ms  | 1.4x faster |

**Result: 5.9x faster for large transposed tensors**

#### Phase 2: SIMD Kernels (contiguous)

Added NEON SIMD intrinsics for ARM64 with in-place mutation optimization.

**Before SIMD (baseline):**

| Operation | Ember | NdArray | Speedup |
| --------- | ----- | ------- | ------- |
| add       | 210µs | 397µs   | 1.9x    |

**After SIMD:**

| Operation | Ember | NdArray | Speedup |
| --------- | ----- | ------- | ------- |
| add       | 202µs | 340µs   | 1.7x    |

SIMD provides modest gains for element-wise ops since they are memory-bound, not compute-bound. The main benefit comes from in-place mutation avoiding extra allocations.

### Analysis

#### Row-based iteration for 2D tensors

The original `StridedIter` did per-element index computation with multi-index tracking:

- 5+ operations per element
- Poor branch prediction
- Cache-unfriendly access patterns

The optimized approach uses direct row-column loops:

```rust
for row in 0..rows {
    let row_start = offset + row * row_stride;
    for col in 0..cols {
        result.push(op(src[row_start + col * col_stride], ...));
    }
}
```

Benefits:

- No per-element overhead
- Predictable inner loop (compiler can optimize)
- Better cache prefetching

#### SIMD with in-place mutation

For contiguous tensors, SIMD kernels use NEON intrinsics (ARM64):

```rust
// 4x f32 per 128-bit register
let va = vld1q_f32(a_ptr.add(offset));
let vb = vld1q_f32(b_ptr.add(offset));
let vr = vaddq_f32(va, vb);
vst1q_f32(a_ptr.add(offset), vr);  // in-place write
```

Key insight: Element-wise ops are memory-bound. SIMD helps marginally, but avoiding allocation is more impactful. The in-place kernels write directly back to the input tensor.

### Memory Allocations

Ember consistently uses fewer allocations:

- Ember: 10 allocations, 8.4MB for binary ops
- NdArray: 13 allocations, 16.8MB for binary ops

### Features

- `simd` (default): NEON kernels on ARM64, scalar fallback elsewhere
- `rayon` (optional): Parallel execution for tensors > 4M elements

## Future Optimizations

1. **AVX2 kernels** - SIMD for x86_64
2. **Cache blocking** - Tiled iteration for better locality
3. **3D+ tensor optimization** - Extend row-based approach to higher dimensions
4. **Fused operations** - Combine multiple ops to reduce memory traffic
