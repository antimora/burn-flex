//! Sort and argsort operations for FlexTensor.
//!
//! Operates directly on storage without TensorData round-trips.

use alloc::vec;
use alloc::vec::Vec;
use burn_backend::{DType, Element};
use burn_std::{Bytes, Shape, bf16, f16};
use bytemuck::Pod;

use crate::{FlexTensor, Layout};

use super::INDEX_DTYPE;

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

fn sort_typed<E: Element + Pod + Copy>(
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

fn sort_with_indices_typed<E: Element + Pod + Copy>(
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
fn argsort_typed<E: Element + Pod + Copy>(
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
fn sort_along_dim<E: Copy>(
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

    // Fast path: last dimension (stride==1), sort contiguous sub-slices in-place
    if dim_stride == 1 {
        for slice_idx in 0..num_slices {
            let base = slice_base_offset(slice_idx, shape, &strides, dim);
            let slice = &mut data[base..base + dim_size];
            if descending {
                slice.sort_unstable_by(|a, b| cmp(b, a));
            } else {
                slice.sort_unstable_by(cmp);
            }
        }
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
fn sort_along_dim_with_indices<E: Copy>(
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

    let mut pairs: Vec<(usize, E)> = Vec::with_capacity(dim_size);

    for slice_idx in 0..num_slices {
        let base = slice_base_offset(slice_idx, shape, &strides, dim);

        pairs.clear();
        if dim_stride == 1 {
            for (i, &val) in data[base..base + dim_size].iter().enumerate() {
                pairs.push((i, val));
            }
        } else {
            for i in 0..dim_size {
                pairs.push((i, data[base + i * dim_stride]));
            }
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
fn argsort_along_dim<E: Copy>(
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

    let mut idx_buf: Vec<usize> = (0..dim_size).collect();

    for slice_idx in 0..num_slices {
        let base = slice_base_offset(slice_idx, shape, &strides, dim);

        idx_buf.clear();
        idx_buf.extend(0..dim_size);

        if dim_stride == 1 {
            let slice = &data[base..base + dim_size];
            if descending {
                idx_buf.sort_unstable_by(|&a, &b| cmp(&slice[b], &slice[a]));
            } else {
                idx_buf.sort_unstable_by(|&a, &b| cmp(&slice[a], &slice[b]));
            }
        } else if descending {
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

fn slice_base_offset(
    slice_idx: usize,
    shape: &Shape,
    strides: &[usize],
    dim: usize,
) -> usize {
    crate::layout::slice_base_offset(slice_idx, shape, strides, dim)
}

fn make_index_tensor(indices: Vec<isize>, shape: Shape) -> FlexTensor {
    let bytes = Bytes::from_elems(indices);
    FlexTensor::new(bytes, Layout::contiguous(shape), INDEX_DTYPE)
}
