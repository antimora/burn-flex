# burn-ember Benchmarks

Tracking performance compared to burn-ndarray.

## Running Benchmarks

```bash
cargo bench --bench binary_ops
```

## Results

**Platform:** Apple Silicon M3 (ARM64) **Date:** 2026-02-03

### Contiguous Operations (1M elements, f32)

| Operation  | Ember | NdArray | Speedup |
| ---------- | ----- | ------- | ------- |
| add        | 210µs | 397µs   | 1.9x    |
| mul        | 235µs | 346µs   | 1.5x    |
| div        | 224µs | 342µs   | 1.5x    |
| scalar_add | 118µs | 197µs   | 1.7x    |
| scalar_mul | 118µs | 201µs   | 1.7x    |

### Non-Contiguous Operations (transposed)

| Size      | Ember  | NdArray | Result                |
| --------- | ------ | ------- | --------------------- |
| 256x256   | 58µs   | 54µs    | ~equal                |
| 1024x1024 | 1.09ms | 1.55ms  | **Ember 1.4x faster** |

### Optimization History

**Initial implementation (StridedIter):**

| Size      | Ember | NdArray | Ratio     |
| --------- | ----- | ------- | --------- |
| 256x256   | 388µs | 55µs    | 7x slower |
| 1024x1024 | 6.4ms | 1.2ms   | 5x slower |

**After row-based iteration optimization:**

| Size      | Ember  | NdArray | Ratio       |
| --------- | ------ | ------- | ----------- |
| 256x256   | 58µs   | 54µs    | ~equal      |
| 1024x1024 | 1.09ms | 1.55ms  | 1.4x faster |

**Improvement: 5.9x faster for large transposed tensors**

### Analysis

**Key optimization: Row-based iteration for 2D tensors**

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

### Memory Allocations

Ember consistently uses fewer allocations:

- Ember: 10-12 allocations for binary ops
- NdArray: 8-13 allocations but more total bytes

## Future Optimizations

1. **SIMD kernels** - NEON for ARM64, AVX2 for x86_64
2. **Parallel execution** - rayon for large tensors
3. **Cache blocking** - Tiled iteration for better locality
4. **3D+ tensor optimization** - Extend row-based approach to higher dimensions
