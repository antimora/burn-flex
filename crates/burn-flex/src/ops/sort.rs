//! Sort and argsort operations for FlexTensor.
//!
//! Operates directly on storage without TensorData round-trips.

use alloc::vec;
use alloc::vec::Vec;
use burn_backend::{DType, Element};
use burn_std::{Bytes, Shape, bf16, f16};
use bytemuck::Pod;

#[cfg(feature = "rayon")]
use rayon::prelude::*;

use crate::{FlexTensor, Layout};

use super::INDEX_DTYPE;
#[cfg(feature = "rayon")]
use super::PARALLEL_THRESHOLD;

/// Validate sort dimension and check for empty tensors.
/// Returns `true` if the tensor is empty (caller should return early).
fn validate_sort_args(shape: &Shape, dim: usize) -> bool {
    assert!(
        dim < shape.num_dims(),
        "sort: dim {} out of bounds for tensor with {} dimensions",
        dim,
        shape.num_dims()
    );
    let dim_size = shape[dim];
    assert!(
        dim_size <= isize::MAX as usize,
        "sort: dimension {} has size {} which exceeds isize::MAX",
        dim,
        dim_size
    );
    shape.num_elements() == 0
}

/// Sort elements along a dimension, returning the sorted tensor.
pub fn sort(tensor: FlexTensor, dim: usize, descending: bool) -> FlexTensor {
    match tensor.dtype() {
        DType::F32 => sort_typed::<f32>(tensor, dim, descending, f32::total_cmp),
        DType::F64 => sort_typed::<f64>(tensor, dim, descending, f64::total_cmp),
        DType::F16 => sort_half(tensor, dim, descending, f16::to_f32, f16::from_f32),
        DType::BF16 => sort_half(tensor, dim, descending, bf16::to_f32, bf16::from_f32),
        DType::I64 => sort_typed::<i64>(tensor, dim, descending, Ord::cmp),
        DType::I32 => sort_typed::<i32>(tensor, dim, descending, Ord::cmp),
        DType::I16 => sort_typed::<i16>(tensor, dim, descending, Ord::cmp),
        DType::I8 => sort_typed::<i8>(tensor, dim, descending, Ord::cmp),
        DType::U64 => sort_typed::<u64>(tensor, dim, descending, Ord::cmp),
        DType::U32 => sort_typed::<u32>(tensor, dim, descending, Ord::cmp),
        DType::U16 => sort_typed::<u16>(tensor, dim, descending, Ord::cmp),
        DType::U8 => sort_typed::<u8>(tensor, dim, descending, Ord::cmp),
        dt => panic!("sort: unsupported dtype {:?}", dt),
    }
}

/// Sort elements along a dimension, returning (sorted_values, indices).
pub fn sort_with_indices(
    tensor: FlexTensor,
    dim: usize,
    descending: bool,
) -> (FlexTensor, FlexTensor) {
    match tensor.dtype() {
        DType::F32 => sort_with_indices_typed::<f32>(tensor, dim, descending, f32::total_cmp),
        DType::F64 => sort_with_indices_typed::<f64>(tensor, dim, descending, f64::total_cmp),
        DType::F16 => sort_with_indices_half(tensor, dim, descending, f16::to_f32, f16::from_f32),
        DType::BF16 => {
            sort_with_indices_half(tensor, dim, descending, bf16::to_f32, bf16::from_f32)
        }
        DType::I64 => sort_with_indices_typed::<i64>(tensor, dim, descending, Ord::cmp),
        DType::I32 => sort_with_indices_typed::<i32>(tensor, dim, descending, Ord::cmp),
        DType::I16 => sort_with_indices_typed::<i16>(tensor, dim, descending, Ord::cmp),
        DType::I8 => sort_with_indices_typed::<i8>(tensor, dim, descending, Ord::cmp),
        DType::U64 => sort_with_indices_typed::<u64>(tensor, dim, descending, Ord::cmp),
        DType::U32 => sort_with_indices_typed::<u32>(tensor, dim, descending, Ord::cmp),
        DType::U16 => sort_with_indices_typed::<u16>(tensor, dim, descending, Ord::cmp),
        DType::U8 => sort_with_indices_typed::<u8>(tensor, dim, descending, Ord::cmp),
        dt => panic!("sort_with_indices: unsupported dtype {:?}", dt),
    }
}

/// Argsort along a dimension, returning indices that would sort the tensor.
pub fn argsort(tensor: FlexTensor, dim: usize, descending: bool) -> FlexTensor {
    match tensor.dtype() {
        DType::F32 => argsort_typed::<f32>(tensor, dim, descending, f32::total_cmp),
        DType::F64 => argsort_typed::<f64>(tensor, dim, descending, f64::total_cmp),
        DType::F16 => argsort_half(tensor, dim, descending, f16::to_f32),
        DType::BF16 => argsort_half(tensor, dim, descending, bf16::to_f32),
        DType::I64 => argsort_typed::<i64>(tensor, dim, descending, Ord::cmp),
        DType::I32 => argsort_typed::<i32>(tensor, dim, descending, Ord::cmp),
        DType::I16 => argsort_typed::<i16>(tensor, dim, descending, Ord::cmp),
        DType::I8 => argsort_typed::<i8>(tensor, dim, descending, Ord::cmp),
        DType::U64 => argsort_typed::<u64>(tensor, dim, descending, Ord::cmp),
        DType::U32 => argsort_typed::<u32>(tensor, dim, descending, Ord::cmp),
        DType::U16 => argsort_typed::<u16>(tensor, dim, descending, Ord::cmp),
        DType::U8 => argsort_typed::<u8>(tensor, dim, descending, Ord::cmp),
        dt => panic!("argsort: unsupported dtype {:?}", dt),
    }
}

// ---------------------------------------------------------------------------
// Typed sort (operates directly on storage)
// ---------------------------------------------------------------------------

fn sort_typed<E: Element + Pod + Copy + Send>(
    tensor: FlexTensor,
    dim: usize,
    descending: bool,
    cmp: fn(&E, &E) -> core::cmp::Ordering,
) -> FlexTensor {
    let tensor = tensor.to_contiguous();
    let shape = tensor.layout().shape().clone();
    let dtype = tensor.dtype();
    if validate_sort_args(&shape, dim) {
        return tensor;
    }

    let mut data: Vec<E> = tensor.storage::<E>().to_vec();

    if shape.num_dims() == 1 {
        if descending {
            data.sort_unstable_by(|a, b| cmp(b, a));
        } else {
            data.sort_unstable_by(cmp);
        }
    } else {
        sort_along_dim(&mut data, &shape, dim, descending, cmp);
    }

    FlexTensor::new(Bytes::from_elems(data), Layout::contiguous(shape), dtype)
}

fn sort_with_indices_typed<E: Element + Pod + Copy + Send>(
    tensor: FlexTensor,
    dim: usize,
    descending: bool,
    cmp: fn(&E, &E) -> core::cmp::Ordering,
) -> (FlexTensor, FlexTensor) {
    let tensor = tensor.to_contiguous();
    let shape = tensor.layout().shape().clone();
    let dtype = tensor.dtype();
    let n = shape.num_elements();
    if validate_sort_args(&shape, dim) {
        let idx = make_index_tensor(Vec::new(), shape.clone());
        return (tensor, idx);
    }

    let src: &[E] = tensor.storage();
    let mut values: Vec<E> = src.to_vec();
    let mut indices: Vec<isize> = vec![0; n];

    if shape.num_dims() == 1 {
        sort_1d_with_indices(&mut values, &mut indices, descending, cmp);
    } else {
        sort_along_dim_with_indices(&mut values, &mut indices, &shape, dim, descending, cmp);
    }

    let idx_tensor = make_index_tensor(indices, shape.clone());
    let val_tensor = FlexTensor::new(Bytes::from_elems(values), Layout::contiguous(shape), dtype);
    (val_tensor, idx_tensor)
}

/// Argsort without materializing sorted values.
fn argsort_typed<E: Element + Pod + Copy + Sync>(
    tensor: FlexTensor,
    dim: usize,
    descending: bool,
    cmp: fn(&E, &E) -> core::cmp::Ordering,
) -> FlexTensor {
    let tensor = tensor.to_contiguous();
    let shape = tensor.layout().shape().clone();
    let n = shape.num_elements();
    if validate_sort_args(&shape, dim) {
        return make_index_tensor(Vec::new(), shape);
    }

    let src: &[E] = tensor.storage();
    let mut indices: Vec<isize> = vec![0; n];

    if shape.num_dims() == 1 {
        let mut idx_vec: Vec<usize> = (0..n).collect();
        if descending {
            idx_vec.sort_unstable_by(|&a, &b| cmp(&src[b], &src[a]));
        } else {
            idx_vec.sort_unstable_by(|&a, &b| cmp(&src[a], &src[b]));
        }
        for (out_i, &orig_i) in idx_vec.iter().enumerate() {
            indices[out_i] = orig_i as isize;
        }
    } else {
        argsort_along_dim(src, &mut indices, &shape, dim, descending, cmp);
    }

    make_index_tensor(indices, shape)
}

/// 1D sort with index tracking, shared by typed and half-precision paths.
fn sort_1d_with_indices<E: Copy>(
    values: &mut [E],
    indices: &mut [isize],
    descending: bool,
    cmp: fn(&E, &E) -> core::cmp::Ordering,
) {
    let n = values.len();
    let mut idx_vec: Vec<usize> = (0..n).collect();
    if descending {
        idx_vec.sort_unstable_by(|&a, &b| cmp(&values[b], &values[a]));
    } else {
        idx_vec.sort_unstable_by(|&a, &b| cmp(&values[a], &values[b]));
    }
    // Apply permutation in one pass using the sorted index order
    let old_values = values.to_vec();
    for (out_i, &orig_i) in idx_vec.iter().enumerate() {
        values[out_i] = old_values[orig_i];
        indices[out_i] = orig_i as isize;
    }
}

/// Sort along a given dimension for N-D tensors.
fn sort_along_dim<E: Copy + Send>(
    data: &mut [E],
    shape: &Shape,
    dim: usize,
    descending: bool,
    cmp: fn(&E, &E) -> core::cmp::Ordering,
) {
    let strides = contiguous_strides(shape);
    let dim_size = shape[dim];
    let dim_stride = strides[dim];
    let num_slices = data.len() / dim_size;

    // Fast path: last dimension (stride==1). Rows are contiguous at
    // offsets `slice_idx * dim_size`, so `chunks_exact_mut(dim_size)`
    // walks them directly. Parallelized with rayon above the threshold.
    if dim_stride == 1 {
        let sort_row = |row: &mut [E]| {
            if descending {
                row.sort_unstable_by(|a, b| cmp(b, a));
            } else {
                row.sort_unstable_by(cmp);
            }
        };

        #[cfg(feature = "rayon")]
        if data.len() >= PARALLEL_THRESHOLD {
            data.par_chunks_exact_mut(dim_size).for_each(sort_row);
            return;
        }

        data.chunks_exact_mut(dim_size).for_each(sort_row);
        return;
    }

    let mut slice_buf: Vec<E> = vec![data[0]; dim_size];

    for slice_idx in 0..num_slices {
        let base = slice_base_offset(slice_idx, shape, &strides, dim);

        for i in 0..dim_size {
            slice_buf[i] = data[base + i * dim_stride];
        }

        if descending {
            slice_buf.sort_unstable_by(|a, b| cmp(b, a));
        } else {
            slice_buf.sort_unstable_by(cmp);
        }

        for i in 0..dim_size {
            data[base + i * dim_stride] = slice_buf[i];
        }
    }
}

/// Sort along a dimension, tracking original indices.
fn sort_along_dim_with_indices<E: Copy + Send>(
    data: &mut [E],
    indices: &mut [isize],
    shape: &Shape,
    dim: usize,
    descending: bool,
    cmp: fn(&E, &E) -> core::cmp::Ordering,
) {
    let strides = contiguous_strides(shape);
    let dim_size = shape[dim];
    let dim_stride = strides[dim];
    let num_slices = data.len() / dim_size;

    // Fast path: last dimension (stride==1). Values and indices rows
    // are both contiguous at `slice_idx * dim_size`, so we can zip
    // matching chunks and avoid the per-row stride arithmetic.
    if dim_stride == 1 {
        let sort_row = |(row, idx_row): (&mut [E], &mut [isize])| {
            let mut pairs: Vec<(usize, E)> = (0..dim_size).map(|i| (i, row[i])).collect();
            if descending {
                pairs.sort_unstable_by(|a, b| cmp(&b.1, &a.1));
            } else {
                pairs.sort_unstable_by(|a, b| cmp(&a.1, &b.1));
            }
            for (i, &(orig_idx, val)) in pairs.iter().enumerate() {
                row[i] = val;
                idx_row[i] = orig_idx as isize;
            }
        };

        #[cfg(feature = "rayon")]
        if data.len() >= PARALLEL_THRESHOLD {
            data.par_chunks_exact_mut(dim_size)
                .zip(indices.par_chunks_exact_mut(dim_size))
                .for_each(sort_row);
            return;
        }

        data.chunks_exact_mut(dim_size)
            .zip(indices.chunks_exact_mut(dim_size))
            .for_each(sort_row);
        return;
    }

    let mut pairs: Vec<(usize, E)> = Vec::with_capacity(dim_size);

    for slice_idx in 0..num_slices {
        let base = slice_base_offset(slice_idx, shape, &strides, dim);

        pairs.clear();
        for i in 0..dim_size {
            pairs.push((i, data[base + i * dim_stride]));
        }

        if descending {
            pairs.sort_unstable_by(|a, b| cmp(&b.1, &a.1));
        } else {
            pairs.sort_unstable_by(|a, b| cmp(&a.1, &b.1));
        }

        for (i, &(orig_idx, val)) in pairs.iter().enumerate() {
            let offset = base + i * dim_stride;
            data[offset] = val;
            indices[offset] = orig_idx as isize;
        }
    }
}

/// Argsort along a dimension without writing sorted values.
fn argsort_along_dim<E: Copy + Sync>(
    data: &[E],
    indices: &mut [isize],
    shape: &Shape,
    dim: usize,
    descending: bool,
    cmp: fn(&E, &E) -> core::cmp::Ordering,
) {
    let strides = contiguous_strides(shape);
    let dim_size = shape[dim];
    let dim_stride = strides[dim];
    let num_slices = data.len() / dim_size;

    // Fast path: last dimension (stride==1). Both input rows and
    // output index rows are contiguous at `slice_idx * dim_size`.
    if dim_stride == 1 {
        let sort_row = |(row, idx_row): (&[E], &mut [isize])| {
            let mut idx_buf: Vec<usize> = (0..dim_size).collect();
            if descending {
                idx_buf.sort_unstable_by(|&a, &b| cmp(&row[b], &row[a]));
            } else {
                idx_buf.sort_unstable_by(|&a, &b| cmp(&row[a], &row[b]));
            }
            for (i, &orig_idx) in idx_buf.iter().enumerate() {
                idx_row[i] = orig_idx as isize;
            }
        };

        #[cfg(feature = "rayon")]
        if data.len() >= PARALLEL_THRESHOLD {
            data.par_chunks_exact(dim_size)
                .zip(indices.par_chunks_exact_mut(dim_size))
                .for_each(sort_row);
            return;
        }

        data.chunks_exact(dim_size)
            .zip(indices.chunks_exact_mut(dim_size))
            .for_each(sort_row);
        return;
    }

    let mut idx_buf: Vec<usize> = (0..dim_size).collect();

    for slice_idx in 0..num_slices {
        let base = slice_base_offset(slice_idx, shape, &strides, dim);

        idx_buf.clear();
        idx_buf.extend(0..dim_size);

        if descending {
            idx_buf.sort_unstable_by(|&a, &b| {
                cmp(&data[base + b * dim_stride], &data[base + a * dim_stride])
            });
        } else {
            idx_buf.sort_unstable_by(|&a, &b| {
                cmp(&data[base + a * dim_stride], &data[base + b * dim_stride])
            });
        }

        for (i, &orig_idx) in idx_buf.iter().enumerate() {
            indices[base + i * dim_stride] = orig_idx as isize;
        }
    }
}

// ---------------------------------------------------------------------------
// Half-precision sort (convert to f32, sort, convert back)
// ---------------------------------------------------------------------------

fn sort_half<H: Element + Pod + Copy>(
    tensor: FlexTensor,
    dim: usize,
    descending: bool,
    to_f32: fn(H) -> f32,
    from_f32: fn(f32) -> H,
) -> FlexTensor {
    let tensor = tensor.to_contiguous();
    let shape = tensor.layout().shape().clone();
    let dtype = tensor.dtype();
    if validate_sort_args(&shape, dim) {
        return tensor;
    }
    let src: &[H] = tensor.storage();
    let mut f32_data: Vec<f32> = src.iter().map(|&v| to_f32(v)).collect();

    if shape.num_dims() == 1 {
        if descending {
            f32_data.sort_unstable_by(|a, b| f32::total_cmp(b, a));
        } else {
            f32_data.sort_unstable_by(f32::total_cmp);
        }
    } else {
        sort_along_dim(&mut f32_data, &shape, dim, descending, f32::total_cmp);
    }

    let result: Vec<H> = f32_data.iter().map(|&v| from_f32(v)).collect();
    FlexTensor::new(Bytes::from_elems(result), Layout::contiguous(shape), dtype)
}

fn sort_with_indices_half<H: Element + Pod + Copy>(
    tensor: FlexTensor,
    dim: usize,
    descending: bool,
    to_f32: fn(H) -> f32,
    from_f32: fn(f32) -> H,
) -> (FlexTensor, FlexTensor) {
    let tensor = tensor.to_contiguous();
    let shape = tensor.layout().shape().clone();
    let dtype = tensor.dtype();
    let n = shape.num_elements();
    if validate_sort_args(&shape, dim) {
        let idx = make_index_tensor(Vec::new(), shape.clone());
        return (tensor, idx);
    }
    let src: &[H] = tensor.storage();
    let mut f32_data: Vec<f32> = src.iter().map(|&v| to_f32(v)).collect();
    let mut indices: Vec<isize> = vec![0; n];

    if shape.num_dims() == 1 {
        sort_1d_with_indices(&mut f32_data, &mut indices, descending, f32::total_cmp);
    } else {
        sort_along_dim_with_indices(
            &mut f32_data,
            &mut indices,
            &shape,
            dim,
            descending,
            f32::total_cmp,
        );
    }

    let result: Vec<H> = f32_data.iter().map(|&v| from_f32(v)).collect();
    let val_tensor = FlexTensor::new(
        Bytes::from_elems(result),
        Layout::contiguous(shape.clone()),
        dtype,
    );
    let idx_tensor = make_index_tensor(indices, shape);
    (val_tensor, idx_tensor)
}

fn argsort_half<H: Element + Pod + Copy>(
    tensor: FlexTensor,
    dim: usize,
    descending: bool,
    to_f32: fn(H) -> f32,
) -> FlexTensor {
    let tensor = tensor.to_contiguous();
    let shape = tensor.layout().shape().clone();
    let n = shape.num_elements();
    if validate_sort_args(&shape, dim) {
        return make_index_tensor(Vec::new(), shape);
    }
    let src: &[H] = tensor.storage();
    let f32_data: Vec<f32> = src.iter().map(|&v| to_f32(v)).collect();
    let mut indices: Vec<isize> = vec![0; n];

    if shape.num_dims() == 1 {
        let mut idx_vec: Vec<usize> = (0..n).collect();
        if descending {
            idx_vec.sort_unstable_by(|&a, &b| f32::total_cmp(&f32_data[b], &f32_data[a]));
        } else {
            idx_vec.sort_unstable_by(|&a, &b| f32::total_cmp(&f32_data[a], &f32_data[b]));
        }
        for (out_i, &orig_i) in idx_vec.iter().enumerate() {
            indices[out_i] = orig_i as isize;
        }
    } else {
        argsort_along_dim(
            &f32_data,
            &mut indices,
            &shape,
            dim,
            descending,
            f32::total_cmp,
        );
    }

    make_index_tensor(indices, shape)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn contiguous_strides(shape: &Shape) -> Vec<usize> {
    crate::layout::contiguous_strides_usize(shape)
}

fn slice_base_offset(slice_idx: usize, shape: &Shape, strides: &[usize], dim: usize) -> usize {
    crate::layout::slice_base_offset(slice_idx, shape, strides, dim)
}

fn make_index_tensor(indices: Vec<isize>, shape: Shape) -> FlexTensor {
    let bytes = Bytes::from_elems(indices);
    FlexTensor::new(bytes, Layout::contiguous(shape), INDEX_DTYPE)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Exercise both below and above the parallel threshold so serial and
    // rayon fast paths in sort_along_dim agree on row-wise sort results.
    fn check_sort_last_dim(rows: usize, cols: usize) {
        let n = rows * cols;
        // Deterministic non-monotonic input with repeats (mirrors bench fill % 1000).
        let src: Vec<f32> = (0..n)
            .map(|i| ((i * 1664525 + 1013904223) % 1000) as f32)
            .collect();

        let mut data = src.clone();
        let shape = Shape::new([rows, cols]);
        sort_along_dim(&mut data, &shape, 1, false, f32::total_cmp);

        for r in 0..rows {
            let row = &data[r * cols..(r + 1) * cols];
            for w in row.windows(2) {
                assert!(w[0] <= w[1], "row {r} not sorted: {:?}", row);
            }
            let mut expected: Vec<f32> = src[r * cols..(r + 1) * cols].to_vec();
            expected.sort_unstable_by(f32::total_cmp);
            assert_eq!(row, expected.as_slice());
        }
    }

    #[test]
    fn sort_along_last_dim_small_serial() {
        // 64*64 = 4K elements, well under PARALLEL_THRESHOLD.
        check_sort_last_dim(64, 64);
    }

    #[test]
    fn sort_along_last_dim_large_parallel() {
        // 1024*1024 = 1M elements, above PARALLEL_THRESHOLD when rayon is on.
        check_sort_last_dim(1024, 1024);
    }

    #[test]
    fn sort_along_last_dim_descending() {
        let mut data: Vec<f32> = (0..4096).map(|i| (i % 17) as f32).collect();
        let shape = Shape::new([128, 32]);
        sort_along_dim(&mut data, &shape, 1, true, f32::total_cmp);
        for r in 0..128 {
            let row = &data[r * 32..(r + 1) * 32];
            for w in row.windows(2) {
                assert!(w[0] >= w[1]);
            }
        }
    }

    #[test]
    fn sort_with_indices_last_dim_roundtrip() {
        let rows = 512;
        let cols = 512; // 256K — straddles threshold equal-case
        let src: Vec<f32> = (0..rows * cols).map(|i| (i as f32 * 0.37).sin()).collect();
        let mut values = src.clone();
        let mut indices = vec![0isize; rows * cols];
        let shape = Shape::new([rows, cols]);
        sort_along_dim_with_indices(&mut values, &mut indices, &shape, 1, false, f32::total_cmp);
        for r in 0..rows {
            let vs = &values[r * cols..(r + 1) * cols];
            let idx_row = &indices[r * cols..(r + 1) * cols];
            let orig = &src[r * cols..(r + 1) * cols];
            for w in vs.windows(2) {
                assert!(f32::total_cmp(&w[0], &w[1]) != core::cmp::Ordering::Greater);
            }
            // Indices must reconstruct the sorted values from the original row.
            for (i, &orig_idx) in idx_row.iter().enumerate() {
                assert_eq!(vs[i], orig[orig_idx as usize]);
            }
        }
    }

    #[test]
    fn argsort_last_dim_matches_permutation() {
        let rows = 200;
        let cols = 1500; // 300K, over threshold with rayon on
        let src: Vec<f32> = (0..rows * cols)
            .map(|i| ((i * 7919) % 997) as f32)
            .collect();
        let mut indices = vec![0isize; rows * cols];
        let shape = Shape::new([rows, cols]);
        argsort_along_dim(&src, &mut indices, &shape, 1, false, f32::total_cmp);
        for r in 0..rows {
            let idx_row = &indices[r * cols..(r + 1) * cols];
            let orig = &src[r * cols..(r + 1) * cols];
            let sorted: Vec<f32> = idx_row.iter().map(|&i| orig[i as usize]).collect();
            for w in sorted.windows(2) {
                assert!(w[0] <= w[1]);
            }
        }
    }
}
