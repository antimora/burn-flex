# Benchmarks: Flex vs NdArray

All benchmarks run on Apple M3 Max, comparing burn-flex against burn-ndarray. Default features
enabled (`std`, `simd`, `rayon`); `gemm` is a required dependency.

**Date**: 2026-02-07

## How to Read

- **Median** time reported (lower is better)
- **Speedup** = NdArray median / Flex median
- **Mem** = peak allocation (`max alloc` from divan)
- Bold speedup means Flex wins; plain means tie or NdArray wins

---

## Binary Operations (f32)

| Operation   | Size | Flex    | NdArray | Speedup  | Flex Mem | NdArray Mem |
| ----------- | ---- | ------- | ------- | -------- | -------- | ----------- |
| add         | 4K   | 449 ns  | 1.45 us | **3.2x** | 16.5 KB  | 49.3 KB     |
| add         | 64K  | 7.7 us  | 21.0 us | **2.7x** | 262 KB   | 787 KB      |
| add         | 1M   | 118 us  | 344 us  | **2.9x** | 4.2 MB   | 12.6 MB     |
| mul         | 4K   | 449 ns  | 1.75 us | **3.9x** | 16.5 KB  | 32.8 KB     |
| mul         | 64K  | 7.6 us  | 21.3 us | **2.8x** | 262 KB   | 787 KB      |
| mul         | 1M   | 119 us  | 343 us  | **2.9x** | 4.2 MB   | 12.6 MB     |
| div         | 1M   | 115 us  | 347 us  | **3.0x** | 4.2 MB   | 12.6 MB     |
| add_scalar  | 1M   | 83 us   | 209 us  | **2.5x** | 4.2 MB   | 8.4 MB      |
| mul_scalar  | 1M   | 79 us   | 188 us  | **2.4x** | 4.2 MB   | 8.4 MB      |
| powf        | 64K  | 198 us  | 219 us  | **1.1x** | 262 KB   | 787 KB      |
| powf        | 1M   | 3.25 ms | 3.49 ms | **1.1x** | 4.2 MB   | 12.6 MB     |
| powf_scalar | 1M   | 3.10 ms | 3.29 ms | **1.1x** | 4.2 MB   | 8.4 MB      |
| atan2       | 64K  | 147 us  | 155 us  | **1.1x** | 262 KB   | 787 KB      |
| atan2       | 1M   | 2.37 ms | 2.59 ms | **1.1x** | 4.2 MB   | 12.6 MB     |

### Transposed

| Operation | Size      | Flex   | NdArray | Speedup  | Flex Mem | NdArray Mem |
| --------- | --------- | ------ | ------- | -------- | -------- | ----------- |
| add       | 256x256   | 49 us  | 56 us   | **1.1x** | 262 KB   | 524 KB      |
| add       | 1024x1024 | 979 us | 1.30 ms | **1.3x** | 4.2 MB   | 8.4 MB      |

---

## Binary Operations (i64)

| Operation      | Size | Flex    | NdArray | Speedup  | Flex Mem | NdArray Mem |
| -------------- | ---- | ------- | ------- | -------- | -------- | ----------- |
| int_add        | 4K   | 835 ns  | 3.98 us | **4.8x** | 32.9 KB  | 98.4 KB     |
| int_add        | 64K  | 14.6 us | 41.6 us | **2.8x** | 524 KB   | 1.57 MB     |
| int_add        | 1M   | 241 us  | 722 us  | **3.0x** | 8.4 MB   | 25.2 MB     |
| int_mul        | 4K   | 1.38 us | 4.29 us | **3.1x** | 32.9 KB  | 98.4 KB     |
| int_mul        | 64K  | 18.5 us | 53.5 us | **2.9x** | 524 KB   | 1.57 MB     |
| int_mul        | 1M   | 230 us  | 743 us  | **3.2x** | 8.4 MB   | 25.2 MB     |
| int_div        | 1M   | 615 us  | 1.14 ms | **1.9x** | 8.4 MB   | 25.2 MB     |
| int_add_scalar | 1M   | 149 us  | 453 us  | **3.0x** | 8.4 MB   | 16.8 MB     |
| int_mul_scalar | 1M   | 284 us  | 464 us  | **1.6x** | 8.4 MB   | 16.8 MB     |

### Int Power

| Operation       | Size     | Flex    | NdArray | Speedup  | Flex Mem | NdArray Mem |
| --------------- | -------- | ------- | ------- | -------- | -------- | ----------- |
| int_powi        | 256x256  | 95.7 us | 123 us  | **1.3x** | 524 KB   | 1.6 MB      |
| int_powi        | 1024x256 | 375 us  | 493 us  | **1.3x** | 2.1 MB   | 6.3 MB      |
| int_powf_scalar | 256x256  | 56.3 us | 50.6 us | 0.9x     | 524 KB   | 1.6 MB      |
| int_powf_scalar | 1024x256 | 217 us  | 198 us  | 0.9x     | 2.1 MB   | 6.3 MB      |

Both use integer `wrapping_pow`. Flex uses 3x less memory; NdArray is slightly faster due to
ndarray's vectorized internals.

### Transposed (i64)

| Operation | Size      | Flex    | NdArray | Speedup  |
| --------- | --------- | ------- | ------- | -------- |
| int_add   | 256x256   | 57 us   | 65 us   | **1.1x** |
| int_add   | 1024x1024 | 1.50 ms | 1.40 ms | 0.93x    |

---

## Int Cast

| Operation  | Size      | Flex   | NdArray | Speedup  | Flex Mem | NdArray Mem |
| ---------- | --------- | ------ | ------- | -------- | -------- | ----------- |
| i64 to i8  | 256x256   | 5.6 us | 36.7 us | **6.6x** | 65.7 KB  | 1.05 MB     |
| i64 to i32 | 64x64     | 353 ns | 2.68 us | **7.6x** | 16.5 KB  | 65.6 KB     |
| i64 to i32 | 256x256   | 7.4 us | 36.9 us | **5.0x** | 262 KB   | 1.05 MB     |
| i64 to i32 | 1024x1024 | 112 us | 635 us  | **5.7x** | 4.2 MB   | 16.8 MB     |

---

## Int Random

| Operation | Size       | Flex    | NdArray | Speedup |
| --------- | ---------- | ------- | ------- | ------- |
| uniform   | 64x64      | 41 us   | 42 us   | 1.0x    |
| uniform   | 256x256    | 659 us  | 685 us  | 1.0x    |
| uniform   | 1024x1024  | 10.6 ms | 11.0 ms | 1.0x    |
| uniform   | 16x128x128 | 2.63 ms | 2.75 ms | 1.0x    |

---

## Matrix Multiplication

### Square (f32)

| Size      | Flex    | NdArray | Speedup  | Flex Mem | NdArray Mem |
| --------- | ------- | ------- | -------- | -------- | ----------- |
| 64x64     | 5.9 us  | 19.8 us | **3.4x** | 33.6 KB  | 49.3 KB     |
| 128x128   | 43.9 us | 69.1 us | **1.6x** | 328 KB   | 197 KB      |
| 256x256   | 160 us  | 168 us  | **1.1x** | 524 KB   | 787 KB      |
| 512x512   | 569 us  | 868 us  | **1.5x** | 2.1 MB   | 3.1 MB      |
| 1024x1024 | 2.65 ms | 6.04 ms | **2.3x** | 8.4 MB   | 12.6 MB     |

### Rectangular (f32)

| Shape               | Flex   | NdArray | Speedup  |
| ------------------- | ------ | ------- | -------- |
| 512x64 x 64x512     | 169 us | 163 us  | 1.0x     |
| 256x512 x 512x256   | 257 us | 295 us  | **1.1x** |
| 128x1024 x 1024x128 | 192 us | 238 us  | **1.2x** |

### Transposed (256x256)

| Config          | Flex   | NdArray | Speedup  |
| --------------- | ------ | ------- | -------- |
| LHS transposed  | 162 us | 197 us  | **1.2x** |
| RHS transposed  | 187 us | 184 us  | 1.0x     |
| Both transposed | 189 us | 218 us  | **1.2x** |

### Batched (f32)

| Shape              | Flex   | NdArray | Speedup  |
| ------------------ | ------ | ------- | -------- |
| 8x 64x64           | 55 us  | 101 us  | **1.8x** |
| 32x 64x64          | 73 us  | 165 us  | **2.3x** |
| 16x 128x128        | 212 us | 635 us  | **3.0x** |
| 12x 512x64 (heads) | 602 us | 1.90 ms | **3.2x** |

### Broadcast (f32)

| Shape                     | Flex   | NdArray | Speedup  |
| ------------------------- | ------ | ------- | -------- |
| [1,64,64] x [8,64,64]     | 53 us  | 75 us   | **1.4x** |
| [8,64,64] x [1,64,64]     | 50 us  | 78 us   | **1.6x** |
| [2,1,32,32] x [1,4,32,32] | 6.9 us | 42 us   | **6.1x** |
| [4,1,64,64] x [1,4,64,64] | 61 us  | 96 us   | **1.6x** |

### Integer (i32)

| Size    | Flex    | NdArray | Speedup  |
| ------- | ------- | ------- | -------- |
| 64x64   | 107 us  | 115 us  | **1.1x** |
| 128x128 | 959 us  | 984 us  | 1.0x     |
| 256x256 | 10.8 ms | 10.3 ms | 0.95x    |
| 512x512 | 126 ms  | 118 ms  | 0.94x    |

---

## Slice Operations

### Basic Slicing

| Operation | Size      | Flex   | NdArray | Speedup   | Flex Mem | NdArray Mem |
| --------- | --------- | ------ | ------- | --------- | -------- | ----------- |
| slice 1D  | 1K        | 144 ns | 303 ns  | **2.1x**  | 80 B     | 6.3 KB      |
| slice 1D  | 1M        | 116 ns | 80 us   | **~690x** | 18 B     | 6.3 MB      |
| slice 2D  | 256x256   | 126 ns | 8.0 us  | **~64x**  | 36 B     | 328 KB      |
| slice 2D  | 1024x1024 | 112 ns | 85 us   | **~760x** | 36 B     | 5.2 MB      |
| slice 3D  | 64x64x64  | 148 ns | 29.7 us | **~200x** | 60 B     | 1.2 MB      |

### Narrow

| Operation   | Size      | Flex   | NdArray | Speedup   |
| ----------- | --------- | ------ | ------- | --------- |
| narrow dim0 | 256x256   | 179 ns | 6.1 us  | **~34x**  |
| narrow dim0 | 1024x1024 | 167 ns | 81 us   | **~490x** |
| narrow dim1 | 256x256   | 163 ns | 11.6 us | **~71x**  |

### Slice Assignment

| Operation | Size      | Flex    | NdArray | Speedup  |
| --------- | --------- | ------- | ------- | -------- |
| assign 1D | 1K        | 361 ns  | 605 ns  | **1.7x** |
| assign 2D | 256x256   | 5.6 us  | 13.3 us | **2.4x** |
| assign 2D | 1024x1024 | 75.7 us | 187 us  | **2.5x** |

### Transposed Slicing

| Size      | Flex   | NdArray | Speedup    |
| --------- | ------ | ------- | ---------- |
| 256x256   | 98 ns  | 8.2 us  | **~84x**   |
| 1024x1024 | 112 ns | 236 us  | **~2100x** |

### Slice with Step

| Operation | Size      | Flex  | NdArray | Speedup    |
| --------- | --------- | ----- | ------- | ---------- |
| step2 1D  | 1K        | 94 ns | 470 ns  | **5.0x**   |
| step2 1D  | 1M        | 91 ns | 193 us  | **~2100x** |
| step2 2D  | 1024x1024 | 93 ns | 139 us  | **~1500x** |
| step4 2D  | 256x256   | 95 ns | 6.9 us  | **~73x**   |

---

## Concatenation

### Cat (dim 0, contiguous memcpy fast path)

| Tensors | Size     | Flex    | NdArray | Speedup  | Flex Mem | NdArray Mem |
| ------- | -------- | ------- | ------- | -------- | -------- | ----------- |
| 4x      | 256x256  | 13.5 us | 48.9 us | **3.6x** | 1.0 MB   | 3.1 MB      |
| 4x      | 1024x256 | 51.4 us | 201 us  | **3.9x** | 4.2 MB   | 12.6 MB     |
| 16x     | 64x64    | 4.31 us | 15.5 us | **3.6x** | 265 KB   | 791 KB      |
| 4x      | 16K (1D) | 3.20 us | 15.2 us | **4.8x** | 263 KB   | 788 KB      |

### Cat (dim 1, general path)

| Tensors | Size    | Flex    | NdArray | Speedup   | Flex Mem | NdArray Mem |
| ------- | ------- | ------- | ------- | --------- | -------- | ----------- |
| 4x      | 256x64  | 6.42 us | 69.3 us | **10.8x** | 263 KB   | 788 KB      |
| 4x      | 1024x64 | 22.2 us | 361 us  | **16.3x** | 1.0 MB   | 3.1 MB      |

Dim-1 cat is much faster because NdArray's default uses N `slice_assign` calls while Flex copies
contiguous chunks directly.

---

## Reduce Operations

### Full Tensor Sum

| Size | Flex   | NdArray | Speedup  | Flex Mem | NdArray Mem |
| ---- | ------ | ------- | -------- | -------- | ----------- |
| 1K   | 228 ns | 375 ns  | **1.6x** | 100 B    | 558 B       |
| 64K  | 6.2 us | 14.6 us | **2.4x** | 100 B    | 524 KB      |
| 1M   | 59 us  | 222 us  | **3.8x** | 100 B    | 8.4 MB      |

### Full Tensor Max

| Size | Flex    | NdArray | Speedup  | Flex Mem | NdArray Mem |
| ---- | ------- | ------- | -------- | -------- | ----------- |
| 1K   | 629 ns  | 851 ns  | **1.4x** | 100 B    | 8.3 KB      |
| 64K  | 37.5 us | 43.4 us | **1.2x** | 100 B    | 524 KB      |
| 1M   | 566 us  | 679 us  | **1.2x** | 100 B    | 8.4 MB      |

Flex: zero-alloc single-pass reduce. NdArray: allocates a copy via `into_owned`.

### Full Tensor Min

| Size | Flex    | NdArray | Speedup  | Flex Mem | NdArray Mem |
| ---- | ------- | ------- | -------- | -------- | ----------- |
| 1K   | 686 ns  | 838 ns  | **1.2x** | 100 B    | 8.3 KB      |
| 64K  | 35.6 us | 43.4 us | **1.2x** | 100 B    | 524 KB      |
| 1M   | 602 us  | 680 us  | **1.1x** | 100 B    | 8.4 MB      |

### Int Max

| Size      | Flex    | NdArray | Speedup  | Flex Mem | NdArray Mem |
| --------- | ------- | ------- | -------- | -------- | ----------- |
| 256x256   | 9.14 us | 24.1 us | **2.6x** | 120 B    | 1.0 MB      |
| 1024x1024 | 148 us  | 418 us  | **2.8x** | 120 B    | 16.8 MB     |

Flex: zero-alloc single-pass on i64 data. NdArray: reshape + max_dim + alloc.

### Sum Along Dimension

| Shape     | Dim | Flex    | NdArray | Speedup  |
| --------- | --- | ------- | ------- | -------- |
| 256x256   | 0   | 5.0 us  | 19.6 us | **3.9x** |
| 256x256   | 1   | 4.2 us  | 12.9 us | **3.1x** |
| 1024x1024 | 0   | 78.5 us | 226 us  | **2.9x** |
| 1024x1024 | 1   | 79.5 us | 223 us  | **2.8x** |

### 3D Sum (Batched)

| Shape      | Dim | Flex   | NdArray | Speedup  |
| ---------- | --- | ------ | ------- | -------- |
| 32x256x256 | 1   | 150 us | 527 us  | **3.5x** |
| 32x256x256 | 2   | 133 us | 346 us  | **2.6x** |

### Sum Transposed

| Size      | Flex    | NdArray | Speedup  |
| --------- | ------- | ------- | -------- |
| 256x256   | 6.4 us  | 6.5 us  | 1.0x     |
| 1024x1024 | 53.5 us | 102 us  | **1.9x** |

### Sum Dim on Transposed

| Size      | Dim | Flex    | NdArray | Speedup  |
| --------- | --- | ------- | ------- | -------- |
| 256x256   | 0   | 4.0 us  | 4.5 us  | **1.1x** |
| 1024x1024 | 0   | 78.9 us | 83.8 us | **1.1x** |

### Mean Along Dimension

| Shape     | Dim | Flex    | NdArray | Speedup  |
| --------- | --- | ------- | ------- | -------- |
| 256x256   | 1   | 4.2 us  | 12.9 us | **3.1x** |
| 1024x1024 | 1   | 78.1 us | 207 us  | **2.7x** |

### Argmax

| Shape     | Dim | Flex    | NdArray | Speedup  |
| --------- | --- | ------- | ------- | -------- |
| 1K        | -   | 3.4 us  | 4.5 us  | **1.3x** |
| 256x256   | 1   | 227 us  | 243 us  | **1.1x** |
| 1024x1024 | 1   | 3.38 ms | 4.04 ms | **1.2x** |

---

## Cumulative Operations

### Cumsum

| Shape     | Dim | Flex     | NdArray | Speedup  |
| --------- | --- | -------- | ------- | -------- |
| 1K        | 0   | 829 ns   | 65.8 us | **~79x** |
| 64K       | 0   | 44.7 us  | 4.21 ms | **~94x** |
| 1M        | 0   | 704 us   | 68.5 ms | **~97x** |
| 256x256   | 0   | 11.2 us  | 37.7 us | **3.4x** |
| 256x256   | 1   | 42.1 us  | 224 us  | **5.3x** |
| 1024x1024 | 1   | 693 us   | 5.6 ms  | **8.1x** |

### Cumprod

| Shape   | Dim | Flex    | NdArray | Speedup  |
| ------- | --- | ------- | ------- | -------- |
| 1K      | 0   | 1.18 us | 65.7 us | **~56x** |
| 256x256 | 1   | 65.2 us | 222 us  | **3.4x** |

### Cummin

| Shape     | Dim | Flex     | NdArray | Speedup  |
| --------- | --- | -------- | ------- | -------- |
| 1K        | 0   | 1.85 us  | 65.8 us | **~36x** |
| 256x256   | 1   | 103 us   | 199 us  | **1.9x** |
| 1024x1024 | 1   | 1.66 ms  | 5.54 ms | **3.3x** |

### Cummax

| Shape     | Dim | Flex     | NdArray | Speedup  |
| --------- | --- | -------- | ------- | -------- |
| 1K        | 0   | 1.81 us  | 66.6 us | **~37x** |
| 256x256   | 1   | 101 us   | 125 us  | **1.2x** |
| 1024x1024 | 1   | 1.68 ms  | 3.72 ms | **2.2x** |

### 3D Cumsum (Batched)

| Shape    | Dim | Flex     | NdArray | Speedup  |
| -------- | --- | -------- | ------- | -------- |
| 32x64x64 | 1   | 23.3 us  | 89.9 us | **3.9x** |
| 32x64x64 | 2   | 67.1 us  | 239 us  | **3.6x** |

---

## Gather/Scatter Operations

### Gather

| Shape     | Dim | Flex    | NdArray | Speedup  | Flex Mem | NdArray Mem |
| --------- | --- | ------- | ------- | -------- | -------- | ----------- |
| 256x256   | 0   | 34.2 us | 156 us  | **4.6x** | 131 KB   | 787 KB      |
| 256x256   | 1   | 33.2 us | 103 us  | **3.1x** | 131 KB   | 787 KB      |
| 1024x1024 | 1   | 159 us  | 1.56 ms | **9.8x** | 2.1 MB   | 12.6 MB     |

### Scatter Add

| Shape     | Dim | Flex    | NdArray | Speedup  | Flex Mem | NdArray Mem |
| --------- | --- | ------- | ------- | -------- | -------- | ----------- |
| 256x256   | 1   | 32.5 us | 210 us  | **6.5x** | 262 KB   | 918 KB      |
| 1024x1024 | 1   | 490 us  | 3.12 ms | **6.4x** | 4.2 MB   | 14.7 MB     |

### Select

| Shape     | Dim | Flex    | NdArray | Speedup  | Flex Mem | NdArray Mem |
| --------- | --- | ------- | ------- | -------- | -------- | ----------- |
| 256x256   | 0   | 3.3 us  | 24.2 us | **7.3x** | 131 KB   | 525 KB      |
| 256x256   | 1   | 26.0 us | 41.8 us | **1.6x** | 131 KB   | 525 KB      |
| 1024x1024 | 0   | 108 us  | 214 us  | **2.0x** | 2.1 MB   | 8.4 MB      |

### Bool Select

| Shape    | Indices | Flex    | NdArray | Speedup   | Flex Mem | NdArray Mem |
| -------- | ------- | ------- | ------- | --------- | -------- | ----------- |
| 256x256  | 128     | 917 ns  | 16.2 us | **17.7x** | 33 KB    | 132 KB      |
| 1024x256 | 512     | 3.38 us | 57.2 us | **16.9x** | 131 KB   | 529 KB      |

Flex: select directly on u8 data. NdArray default: bool->int->select->compare.

### Select Add

| Shape     | Dim | Flex   | NdArray | Speedup  | Flex Mem | NdArray Mem |
| --------- | --- | ------ | ------- | -------- | -------- | ----------- |
| 256x256   | 0   | 7.4 us | 23.4 us | **3.2x** | 262 KB   | 657 KB      |
| 1024x1024 | 0   | 102 us | 274 us  | **2.7x** | 4.2 MB   | 10.5 MB     |

---

## Unary Operations

### Basic Math

| Operation | Size | Flex    | NdArray | Speedup  |
| --------- | ---- | ------- | ------- | -------- |
| exp       | 4K   | 5.1 us  | 5.6 us  | **1.1x** |
| exp       | 64K  | 80 us   | 89 us   | **1.1x** |
| exp       | 1M   | 1.30 ms | 1.40 ms | **1.1x** |
| log       | 4K   | 6.6 us  | 7.1 us  | **1.1x** |
| log       | 64K  | 105 us  | 113 us  | **1.1x** |
| log       | 1M   | 1.70 ms | 1.82 ms | **1.1x** |
| sqrt      | 4K   | 688 ns  | 1.09 us | **1.6x** |
| sqrt      | 64K  | 9.0 us  | 17.5 us | **1.9x** |
| sqrt      | 1M   | 142 us  | 266 us  | **1.9x** |
| abs       | 1M   | 76 us   | 193 us  | **2.5x** |
| recip     | 1M   | 74 us   | 197 us  | **2.7x** |

### Trigonometric

| Operation | Size | Flex    | NdArray | Speedup  |
| --------- | ---- | ------- | ------- | -------- |
| sin       | 4K   | 5.6 us  | 8.1 us  | **1.4x** |
| sin       | 64K  | 90 us   | 136 us  | **1.5x** |
| sin       | 1M   | 1.46 ms | 2.18 ms | **1.5x** |
| cos       | 4K   | 6.5 us  | 8.6 us  | **1.3x** |
| cos       | 1M   | 1.67 ms | 2.26 ms | **1.4x** |
| tanh      | 4K   | 6.9 us  | 13.8 us | **2.0x** |
| tanh      | 64K  | 110 us  | 223 us  | **2.0x** |
| tanh      | 1M   | 1.80 ms | 3.62 ms | **2.0x** |

### Transposed (Non-contiguous)

| Operation | Size      | Flex    | NdArray | Speedup |
| --------- | --------- | ------- | ------- | ------- |
| exp       | 256x256   | 81 us   | 83 us   | 1.0x    |
| exp       | 1024x1024 | 1.30 ms | 1.34 ms | 1.0x    |

---

## Comparison & Boolean Operations

### Tensor-Tensor Comparisons

| Operation | Size | Flex   | NdArray | Speedup  | Flex Mem | NdArray Mem |
| --------- | ---- | ------ | ------- | -------- | -------- | ----------- |
| greater   | 4K   | 535 ns | 1.45 us | **2.7x** | 4.2 KB   | 49.3 KB     |
| greater   | 64K  | 6.2 us | 21.0 us | **3.4x** | 65.6 KB  | 787 KB      |
| greater   | 1M   | 89 us  | 308 us  | **3.5x** | 1.0 MB   | 12.6 MB     |
| equal     | 4K   | 545 ns | 1.49 us | **2.7x** | 4.2 KB   | 49.3 KB     |
| equal     | 1M   | 89 us  | 307 us  | **3.5x** | 1.0 MB   | 12.6 MB     |
| lower     | 1M   | 88 us  | 304 us  | **3.5x** | 1.0 MB   | 12.6 MB     |

### Scalar Comparisons

| Operation    | Size | Flex  | NdArray | Speedup  |
| ------------ | ---- | ----- | ------- | -------- |
| greater_elem | 1M   | 77 us | 200 us  | **2.6x** |

### Transposed Comparisons

| Operation | Size      | Flex   | NdArray | Speedup  |
| --------- | --------- | ------ | ------- | -------- |
| greater   | 256x256   | 53 us  | 51 us   | 0.96x    |
| greater   | 1024x1024 | 972 us | 1.21 ms | **1.2x** |

### Broadcast Comparisons

| Operation | Shape     | Flex   | NdArray | Speedup  |
| --------- | --------- | ------ | ------- | -------- |
| greater   | 256x256   | 6.2 us | 25.7 us | **4.1x** |
| greater   | 1024x1024 | 86 us  | 319 us  | **3.7x** |

### Expand (Broadcasting)

| Operation           | Flex   | NdArray | Speedup    |
| ------------------- | ------ | ------- | ---------- |
| 1x1 to 1000x1000    | 171 ns | 287 us  | **~1700x** |
| 1024x1 to 1024x1024 | 130 ns | 307 us  | **~2400x** |
| 1x1024 to 1024x1024 | 145 ns | 75 us   | **~520x**  |

### Boolean Operations

| Operation | Size | Flex  | NdArray | Speedup |
| --------- | ---- | ----- | ------- | ------- |
| bool_not  | 1M   | 24 us | 19 us   | 0.79x   |
| bool_and  | 1M   | 32 us | 28 us   | 0.88x   |

---

## Convolutions

### Kernel Size Comparison (4x64x56x56, 64 to 128 channels)

| Kernel | Flex    | NdArray | Speedup  |
| ------ | ------- | ------- | -------- |
| 1x1    | 871 us  | 1.28 ms | **1.5x** |
| 3x3    | 3.82 ms | 10.6 ms | **2.8x** |
| 5x5    | 8.67 ms | 28.0 ms | **3.2x** |
| 7x7    | 16.6 ms | 54.7 ms | **3.3x** |

### ResNet Layers (batch=1, 3x3)

| Layer  | Input       | Channels       | Flex    | NdArray | Speedup  |
| ------ | ----------- | -------------- | ------- | ------- | -------- |
| conv1  | 1x3x224x224 | 3 to 64 (k7s2) | 973 us  | 1.27 ms | **1.3x** |
| layer1 | 1x64x56x56  | 64 to 64       | 1.05 ms | 1.89 ms | **1.8x** |
| layer2 | 1x128x28x28 | 128 to 128     | 1.31 ms | 1.79 ms | **1.4x** |
| layer3 | 1x256x14x14 | 256 to 256     | 2.45 ms | 3.58 ms | **1.5x** |
| layer4 | 1x512x7x7   | 512 to 512     | 6.27 ms | 9.07 ms | **1.4x** |

### Small (batch=1, 3x3)

| Input      | Channels | Flex   | NdArray | Speedup  |
| ---------- | -------- | ------ | ------- | -------- |
| 1x3x32x32  | 3 to 16  | 71 us  | 85 us   | **1.2x** |
| 1x16x32x32 | 16 to 32 | 232 us | 277 us  | **1.2x** |
| 1x32x16x16 | 32 to 64 | 196 us | 352 us  | **1.8x** |

### Large Batched (batch=16, 3x3)

| Input         | Channels   | Flex  | NdArray | Speedup  |
| ------------- | ---------- | ----- | ------- | -------- |
| 16x64x128x128 | 64 to 128  | 81 ms | 205 ms  | **2.5x** |
| 16x128x64x64  | 128 to 256 | 61 ms | 243 ms  | **4.0x** |

### Medium Batched (batch=8, 3x3)

| Input      | Channels  | Flex    | NdArray | Speedup  |
| ---------- | --------- | ------- | ------- | -------- |
| 8x3x64x64  | 3 to 64   | 951 us  | 564 us  | 0.59x    |
| 8x32x64x64 | 32 to 64  | 4.83 ms | 7.29 ms | **1.5x** |
| 8x64x32x32 | 64 to 128 | 3.31 ms | 10.4 ms | **3.1x** |

### Conv1d

| Input      | Kernel | Flex    | NdArray | Speedup  |
| ---------- | ------ | ------- | ------- | -------- |
| 1x16x256   | 3      | 34 us   | 176 us  | **5.2x** |
| 8x32x512   | 5      | 559 us  | 2.40 ms | **4.3x** |
| 16x64x1024 | 7      | 5.32 ms | 50.9 ms | **9.6x** |

---

## Pooling

### Max Pool 2D

| Input        | Kernel | Flex   | NdArray | Speedup  |
| ------------ | ------ | ------ | ------- | -------- |
| 1x64x56x56   | 3x3 s2 | 156 us | 193 us  | **1.2x** |
| 8x64x56x56   | 3x3 s2 | 786 us | 1.17 ms | **1.5x** |
| 16x128x28x28 | 2x2 s2 | 482 us | 859 us  | **1.8x** |
| 1x512x14x14  | 2x2 s2 | 104 us | 130 us  | **1.3x** |

### Max Pool 2D (ResNet)

| Input         | Kernel | Flex    | NdArray | Speedup  |
| ------------- | ------ | ------- | ------- | -------- |
| 1x64x112x112  | 3x3 s2 | 491 us  | 650 us  | **1.3x** |
| 8x64x112x112  | 3x3 s2 | 2.88 ms | 4.02 ms | **1.4x** |
| 16x64x112x112 | 3x3 s2 | 5.43 ms | 7.73 ms | **1.4x** |

### Avg Pool 2D

| Input        | Kernel | Flex   | NdArray | Speedup  |
| ------------ | ------ | ------ | ------- | -------- |
| 1x64x56x56   | 3x3 s2 | 170 us | 212 us  | **1.2x** |
| 8x64x56x56   | 3x3 s2 | 869 us | 1.19 ms | **1.4x** |
| 16x128x28x28 | 2x2 s2 | 554 us | 851 us  | **1.5x** |

### Adaptive Avg Pool 2D

| Input       | Output | Flex   | NdArray | Speedup  |
| ----------- | ------ | ------ | ------- | -------- |
| 1x256x56x56 | 7x7    | 174 us | 296 us  | **1.7x** |
| 1x512x7x7   | 1x1    | 75 us  | 80 us   | **1.1x** |
| 8x512x7x7   | 1x1    | 124 us | 152 us  | **1.2x** |
| 16x2048x7x7 | 1x1    | 294 us | 687 us  | **2.3x** |

### Max Pool 1D

| Input       | Kernel | Flex    | NdArray | Speedup  |
| ----------- | ------ | ------- | ------- | -------- |
| 1x64x256    | 3 s2   | 58 us   | 104 us  | **1.8x** |
| 8x128x512   | 3 s2   | 340 us  | 1.06 ms | **3.1x** |
| 16x256x1024 | 3 s2   | 2.44 ms | 7.19 ms | **2.9x** |

### Kernel Size Comparison (4x64x56x56)

| Kernel | Flex    | NdArray | Speedup  |
| ------ | ------- | ------- | -------- |
| 2x2    | 255 us  | 490 us  | **1.9x** |
| 3x3    | 444 us  | 651 us  | **1.5x** |
| 5x5    | 1.15 ms | 949 us  | 0.83x    |

---

## Transposed Convolutions

### Conv Transpose 2D

| Input          | Output | Flex    | NdArray | Speedup  |
| -------------- | ------ | ------- | ------- | -------- |
| 1x64x7x7       | 14x14  | 1.31 ms | 1.73 ms | **1.3x** |
| 1x128x14x14    | 28x28  | 9.13 ms | 13.6 ms | **1.5x** |
| 1x256x28x28    | 56x56  | 147 ms  | 212 ms  | **1.4x** |
| 1x512x7x7 k3s1 | 7x7    | 39.3 ms | 54.4 ms | **1.4x** |
| 8x64x14x14     | 28x28  | 36.6 ms | 50.4 ms | **1.4x** |

### DCGAN Generator

| Layer          | Flex    | NdArray | Speedup  | Flex Mem | NdArray Mem |
| -------------- | ------- | ------- | -------- | -------- | ----------- |
| 1x1 to 4x4     | 984 us  | 1.79 ms | **1.8x** | 49.6 KB  | 16.8 MB     |
| 4x4 to 8x8     | 2.90 ms | 3.90 ms | **1.3x** | 98.8 KB  | 4.2 MB      |
| 8x8 to 16x16   | 3.23 ms | 4.32 ms | **1.3x** | 197 KB   | 1.1 MB      |
| 16x16 to 32x32 | 1.10 ms | 1.48 ms | **1.3x** | 37.3 KB  | 144 KB      |

### Conv Transpose 1D

| Input     | Flex    | NdArray | Speedup  |
| --------- | ------- | ------- | -------- |
| 1x64x32   | 330 us  | 384 us  | **1.2x** |
| 8x128x64  | 6.33 ms | 9.70 ms | **1.5x** |
| 1x256x128 | 6.68 ms | 10.0 ms | **1.5x** |

### Conv Transpose 3D

| Input      | Output   | Flex    | NdArray | Speedup  |
| ---------- | -------- | ------- | ------- | -------- |
| 1x32x4x4x4 | 8x8x8    | 1.47 ms | 2.58 ms | **1.8x** |
| 1x64x8x8x8 | 16x16x16 | 24.1 ms | 49.3 ms | **2.0x** |

---

## Interpolation

### Nearest

| Input       | Output  | Flex  | NdArray | Speedup  |
| ----------- | ------- | ----- | ------- | -------- |
| 1x3x64x64   | 128x128 | 58 us | 157 us  | **2.7x** |
| 1x3x32x32   | 128x128 | 62 us | 153 us  | **2.5x** |
| 1x3x256x256 | 128x128 | 65 us | 187 us  | **2.9x** |
| 8x3x64x64   | 128x128 | 95 us | 339 us  | **3.6x** |
| 1x64x32x32  | 64x64   | 83 us | 268 us  | **3.2x** |

### Bilinear

| Input       | Output  | Flex   | NdArray | Speedup  |
| ----------- | ------- | ------ | ------- | -------- |
| 1x3x64x64   | 128x128 | 79 us  | 175 us  | **2.2x** |
| 1x3x32x32   | 128x128 | 77 us  | 174 us  | **2.3x** |
| 1x3x256x256 | 128x128 | 78 us  | 199 us  | **2.6x** |
| 8x3x64x64   | 128x128 | 150 us | 392 us  | **2.6x** |
| 1x64x32x32  | 64x64   | 105 us | 314 us  | **3.0x** |

### Bicubic

| Input       | Output  | Flex   | NdArray | Speedup  |
| ----------- | ------- | ------ | ------- | -------- |
| 1x3x64x64   | 128x128 | 176 us | 251 us  | **1.4x** |
| 1x3x32x32   | 128x128 | 167 us | 242 us  | **1.4x** |
| 1x3x256x256 | 128x128 | 178 us | 275 us  | **1.5x** |
| 8x3x64x64   | 128x128 | 851 us | 979 us  | **1.2x** |
| 1x64x32x32  | 64x64   | 587 us | 691 us  | **1.2x** |

---

## Grid Sample 2D

| Input      | Grid  | Flex    | NdArray | Speedup  | Flex Mem | NdArray Mem |
| ---------- | ----- | ------- | ------- | -------- | -------- | ----------- |
| 1x3x32x32  | 32x32 | 11.8 us | 105 us  | **8.9x** | 12.7 KB  | 32.9 KB     |
| 1x3x64x64  | 64x64 | 45.5 us | 159 us  | **3.5x** | 49.5 KB  | 131 KB      |
| 4x3x32x32  | 32x32 | 44.6 us | 172 us  | **3.9x** | 49.5 KB  | 131 KB      |
| 1x16x64x64 | 64x64 | 268 us  | 299 us  | **1.1x** | 263 KB   | 557 KB      |

Flex: direct pixel-coordinate computation with bilinear interpolation. NdArray default: many
intermediate tensor allocations for coordinate transforms and interpolation.

---

## Cross Product & Unfold

### Cross Product

| Shape   | Flex    | NdArray | Speedup  |
| ------- | ------- | ------- | -------- |
| 1Kx3    | 26.7 us | 46.7 us | **1.7x** |
| 64Kx3   | 1.57 ms | 2.95 ms | **1.9x** |
| 256Kx3  | 6.24 ms | 11.7 ms | **1.9x** |
| 64x3x64 | 146 us  | 306 us  | **2.1x** |

### Unfold (1D)

| Input | Window | Step | Flex  | NdArray | Speedup      |
| ----- | ------ | ---- | ----- | ------- | ------------ |
| 1K    | 8      | 1    | 65 ns | 106 us  | **~1600x**   |
| 64K   | 8      | 1    | 50 ns | 6.8 ms  | **~137000x** |
| 64K   | 64     | 1    | 44 ns | 7.3 ms  | **~166000x** |
| 64K   | 64     | 32   | 58 ns | 232 us  | **~4000x**   |

### Unfold (2D/3D)

| Shape    | Dim | Window | Step | Flex  | NdArray | Speedup     |
| -------- | --- | ------ | ---- | ----- | ------- | ----------- |
| 256x256  | 1   | 8      | 1    | 57 ns | 882 us  | **~15000x** |
| 256x256  | 1   | 32     | 16   | 57 ns | 68 us   | **~1200x**  |
| 1024x256 | 1   | 8      | 1    | 51 ns | 3.5 ms  | **~69000x** |
| 32x64x64 | 2   | 8      | 4    | 76 ns | 454 us  | **~6000x**  |

---

## Deformable Convolutions

### Small/Tiny Inputs

| Input     | Config      | Flex    | NdArray | Speedup   |
| --------- | ----------- | ------- | ------- | --------- |
| 1x3x8x8   | 3 to 8, k3  | 8.7 us  | 91.6 us | **10.5x** |
| 1x3x8x8   | no mask     | 7.9 us  | 77.2 us | **9.8x**  |
| 1x3x16x16 | 3 to 16, k3 | 37.6 us | 133 us  | **3.5x**  |
| 1x3x16x16 | stride 2    | 10.4 us | 81.4 us | **7.8x**  |
| 2x8x16x16 | 8 to 16, k3 | 116 us  | 253 us  | **2.2x**  |

### Medium Inputs

| Input      | Config       | Flex   | NdArray | Speedup |
| ---------- | ------------ | ------ | ------- | ------- |
| 1x16x32x32 | 16 to 32, k3 | 833 us | 657 us  | 0.79x   |
| 1x16x32x32 | wg=4         | 835 us | 590 us  | 0.71x   |
| 1x16x32x32 | og=4         | 937 us | 656 us  | 0.70x   |

---

## Quantized Tensor Operations

All quantized ops (except layout ops) go through a dequantize-op-quantize cycle. Flex stores scales
separately and applies `scale * x_q` directly; NdArray reparses `QuantizedBytes` on every dequantize
call, which dominates the cost.

### Quantize (float to i8)

| Size | Flex    | NdArray | Speedup  | Flex Mem | NdArray Mem |
| ---- | ------- | ------- | -------- | -------- | ----------- |
| 4K   | 6.87 us | 11.4 us | **1.7x** | 4.2 KB   | 103 KB      |
| 64K  | 106 us  | 168 us  | **1.6x** | 65.6 KB  | 1.6 MB      |
| 1M   | 1.69 ms | 2.63 ms | **1.6x** | 1.0 MB   | 26.2 MB     |

Fused 2-pass implementation: one pass for min/max, one for quantize. 25x less memory than NdArray.

### Dequantize (i8 to float)

| Size | Flex    | NdArray  | Speedup  | Flex Mem | NdArray Mem |
| ---- | ------- | -------- | -------- | -------- | ----------- |
| 4K   | 378 ns  | 51.2 us  | **135x** | 16.5 KB  | 28.8 KB     |
| 64K  | 3.75 us | 802 us   | **214x** | 262 KB   | 459 KB      |
| 1M   | 55.5 us | 12.87 ms | **232x** | 4.2 MB   | 7.3 MB      |

The hot path for every quantized operation during inference.

### q_add (dequant + add + requant)

| Size | Flex    | NdArray | Speedup  | Flex Mem | NdArray Mem |
| ---- | ------- | ------- | -------- | -------- | ----------- |
| 4K   | 1.2 us  | 102 us  | **85x**  | 33 KB    | 91 KB       |
| 64K  | 15.5 us | 1.64 ms | **106x** | 525 KB   | 1.4 MB      |
| 1M   | 223 us  | 26.1 ms | **117x** | 8.4 MB   | 23.1 MB     |

### q_matmul (dequant + matmul + requant)

| Size    | Flex   | NdArray | Speedup | Flex Mem | NdArray Mem |
| ------- | ------ | ------- | ------- | -------- | ----------- |
| 64x64   | 7.1 us | 141 us  | **20x** | 67 KB    | 140 KB      |
| 256x256 | 173 us | 1.92 ms | **11x** | 1.0 MB   | 2.2 MB      |
| 512x512 | 642 us | 8.0 ms  | **12x** | 4.2 MB   | 8.9 MB      |

### q_sum (dequant + sum)

| Size | Flex    | NdArray | Speedup | Flex Mem | NdArray Mem |
| ---- | ------- | ------- | ------- | -------- | ----------- |
| 4K   | 766 ns  | 50.9 us | **66x** | 16.6 KB  | 45.3 KB     |
| 64K  | 11.0 us | 812 us  | **74x** | 262 KB   | 721 KB      |
| 1M   | 140 us  | 12.9 ms | **92x** | 4.2 MB   | 11.5 MB     |

### q_permute (zero-copy layout op)

| Size      | Flex  | NdArray | Speedup  | Flex Mem | NdArray Mem |
| --------- | ----- | ------- | -------- | -------- | ----------- |
| 256x256   | 57 ns | 3.02 us | **53x**  | 68 B     | 131 KB      |
| 1024x1024 | 68 ns | 28.5 us | **419x** | 68 B     | 2.1 MB      |

Flex: true zero-copy (only metadata allocated). NdArray copies the entire tensor.

### q_argmax (operates on i8 directly)

| Size      | Flex   | NdArray | Speedup  | Flex Mem | NdArray Mem |
| --------- | ------ | ------- | -------- | -------- | ----------- |
| 256x256   | 35 us  | 102 us  | **2.9x** | 2.2 KB   | 131 KB      |
| 1024x1024 | 558 us | 1.68 ms | **3.0x** | 8.4 KB   | 2.1 MB      |

Flex skips dequantization entirely (symmetric quant preserves element ordering). NdArray dequantizes
to f32 first, then finds the argmax.

### q_argmin (operates on i8 directly)

| Size      | Flex   | NdArray | Speedup  | Flex Mem | NdArray Mem |
| --------- | ------ | ------- | -------- | -------- | ----------- |
| 256x256   | 35 us  | 116 us  | **3.3x** | 2.2 KB   | 131 KB      |
| 1024x1024 | 561 us | 1.70 ms | **3.0x** | 8.4 KB   | 2.1 MB      |

Same optimization as q_argmax.

### q_gather (operates on i8 directly for tensor-level quant)

| Size      | Flex   | NdArray | Speedup  | Flex Mem | NdArray Mem |
| --------- | ------ | ------- | -------- | -------- | ----------- |
| 256x256   | 58 us  | 176 us  | **3.0x** | 65.8 KB  | 1.1 MB      |
| 1024x1024 | 181 us | 2.73 ms | **15x**  | 1.0 MB   | 17.8 MB     |

Flex gathers directly on i8 data (single global scale is unchanged). NdArray dequantizes, gathers on
f32, then requantizes. 16x less memory at 1M scale.

---

## Running Benchmarks

```bash
cargo bench --bench binary_ops
cargo bench --bench matmul
cargo bench --bench int_ops
cargo bench --bench slice_ops
cargo bench --bench reduce_ops
cargo bench --bench cumulative_ops
cargo bench --bench gather_scatter_ops
cargo bench --bench unary_ops
cargo bench --bench comparison_ops
cargo bench --bench conv_ops
cargo bench --bench pool_ops
cargo bench --bench conv_transpose_ops
cargo bench --bench interpolate_ops
cargo bench --bench cross_unfold_ops
cargo bench --bench deform_conv_ops
cargo bench --bench quantization_ops
cargo bench --bench cat_max_min_ops
```
