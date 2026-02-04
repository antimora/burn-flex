//! Slice operations for EmberTensor.

use alloc::vec;
use alloc::vec::Vec;
use burn_backend::{DType, Element};
use burn_std::{Bytes, Shape, Slice};

use crate::{EmberTensor, Layout};

/// Slice a tensor according to the given slice parameters.
///
/// For positive steps, this is zero-copy (metadata only).
/// For negative steps, data is copied to handle the reversal.
pub fn slice(tensor: EmberTensor, slices: &[Slice]) -> EmberTensor {
    let (new_layout, needs_copy) = tensor.layout().slice(slices);

    if !needs_copy {
        // Zero-copy: just update the layout
        EmberTensor::new(tensor.bytes().clone(), new_layout, tensor.dtype())
    } else {
        // Needs copy due to negative steps
        slice_with_copy(&tensor, slices)
    }
}

/// Slice with data copy (handles negative steps).
fn slice_with_copy(tensor: &EmberTensor, slices: &[Slice]) -> EmberTensor {
    match tensor.dtype() {
        DType::F32 => slice_copy_impl::<f32>(tensor, slices),
        DType::F64 => slice_copy_impl::<f64>(tensor, slices),
        DType::I32 => slice_copy_impl::<i32>(tensor, slices),
        DType::I64 => slice_copy_impl::<i64>(tensor, slices),
        DType::I16 => slice_copy_impl::<i16>(tensor, slices),
        DType::I8 => slice_copy_impl::<i8>(tensor, slices),
        DType::U32 => slice_copy_impl::<u32>(tensor, slices),
        DType::U64 => slice_copy_impl::<u64>(tensor, slices),
        DType::U16 => slice_copy_impl::<u16>(tensor, slices),
        DType::U8 => slice_copy_impl::<u8>(tensor, slices),
        DType::Bool => slice_copy_impl::<u8>(tensor, slices),
        _ => panic!("slice: unsupported dtype {:?}", tensor.dtype()),
    }
}

/// Generic slice implementation with copy.
fn slice_copy_impl<E: Element + bytemuck::Pod + Default>(
    tensor: &EmberTensor,
    slices: &[Slice],
) -> EmberTensor {
    let src = tensor.storage::<E>();
    let src_layout = tensor.layout();
    let ndims = src_layout.num_dims();

    // Calculate output shape and collect normalized slice info
    let mut out_shape = Vec::with_capacity(ndims);
    let mut slice_info: Vec<(usize, usize, isize)> = Vec::with_capacity(ndims); // (start, len, step)

    for dim in 0..ndims {
        let dim_size = src_layout.shape().dims[dim] as isize;

        let slice = if dim < slices.len() {
            &slices[dim]
        } else {
            // Default: full range
            &Slice::new(0, None, 1)
        };

        let step = slice.step;
        let abs_step = step.unsigned_abs();

        if step > 0 {
            // Forward iteration
            let start = normalize_index(slice.start, dim_size);
            let end = match slice.end {
                Some(e) => normalize_index(e, dim_size).min(dim_size as usize),
                None => dim_size as usize,
            };
            let len = if end > start {
                (end - start + abs_step - 1) / abs_step
            } else {
                0
            };
            out_shape.push(len);
            slice_info.push((start, len, step));
        } else {
            // Backward iteration (negative step)
            let start = match slice.start {
                s if s < 0 => (dim_size + s).max(0) as usize,
                s => (s as usize).min((dim_size - 1).max(0) as usize),
            };
            let end = match slice.end {
                Some(e) if e < 0 => (dim_size + e).max(-1) as isize,
                Some(e) => (e as isize - 1).max(-1),
                None => -1,
            };
            let len = if (start as isize) > end {
                ((start as isize - end) as usize + abs_step - 1) / abs_step
            } else {
                0
            };
            out_shape.push(len);
            slice_info.push((start, len, step));
        }
    }

    let out_layout = Layout::contiguous(Shape::from(out_shape.clone()));
    let num_elements = out_layout.num_elements();

    if num_elements == 0 {
        let bytes = Bytes::from_elems::<E>(Vec::new());
        return EmberTensor::new(bytes, out_layout, tensor.dtype());
    }

    // Allocate output
    let mut out_data: Vec<E> = Vec::with_capacity(num_elements);

    // Use recursive iteration for arbitrary dimensions
    let mut indices = vec![0usize; ndims];
    copy_slice_recursive(
        src,
        src_layout,
        &slice_info,
        &mut out_data,
        &mut indices,
        0,
    );

    let bytes = Bytes::from_elems(out_data);
    EmberTensor::new(bytes, out_layout, tensor.dtype())
}

/// Recursively copy sliced elements.
fn copy_slice_recursive<E: Copy>(
    src: &[E],
    src_layout: &Layout,
    slice_info: &[(usize, usize, isize)],
    out: &mut Vec<E>,
    indices: &mut [usize],
    dim: usize,
) {
    let ndims = src_layout.num_dims();

    if dim == ndims {
        // Base case: copy single element
        let src_idx = compute_src_index(src_layout, slice_info, indices);
        out.push(src[src_idx]);
        return;
    }

    let (_, len, _) = slice_info[dim];

    for i in 0..len {
        indices[dim] = i;
        copy_slice_recursive(src, src_layout, slice_info, out, indices, dim + 1);
    }
}

/// Compute source index from output indices and slice info.
fn compute_src_index(layout: &Layout, slice_info: &[(usize, usize, isize)], out_indices: &[usize]) -> usize {
    let mut idx = layout.start_offset();
    for (dim, &out_i) in out_indices.iter().enumerate() {
        let (start, _, step) = slice_info[dim];
        let src_i = if step > 0 {
            start + out_i * step as usize
        } else {
            // Negative step: start from high index, go down
            (start as isize - (out_i as isize) * (-step)) as usize
        };
        idx += src_i * layout.strides()[dim];
    }
    idx
}

/// Normalize a potentially negative index to a positive one.
fn normalize_index(idx: isize, dim_size: isize) -> usize {
    if idx < 0 {
        (dim_size + idx).max(0) as usize
    } else {
        idx as usize
    }
}

/// Assign values to a slice of a tensor.
pub fn slice_assign(
    tensor: EmberTensor,
    slices: &[Slice],
    value: EmberTensor,
) -> EmberTensor {
    match tensor.dtype() {
        DType::F32 => slice_assign_impl::<f32>(tensor, slices, value),
        DType::F64 => slice_assign_impl::<f64>(tensor, slices, value),
        DType::I32 => slice_assign_impl::<i32>(tensor, slices, value),
        DType::I64 => slice_assign_impl::<i64>(tensor, slices, value),
        DType::I16 => slice_assign_impl::<i16>(tensor, slices, value),
        DType::I8 => slice_assign_impl::<i8>(tensor, slices, value),
        DType::U32 => slice_assign_impl::<u32>(tensor, slices, value),
        DType::U64 => slice_assign_impl::<u64>(tensor, slices, value),
        DType::U16 => slice_assign_impl::<u16>(tensor, slices, value),
        DType::U8 => slice_assign_impl::<u8>(tensor, slices, value),
        DType::Bool => slice_assign_impl::<u8>(tensor, slices, value),
        _ => panic!("slice_assign: unsupported dtype {:?}", tensor.dtype()),
    }
}

/// Generic slice assign implementation.
fn slice_assign_impl<E: Element + bytemuck::Pod + Clone>(
    tensor: EmberTensor,
    slices: &[Slice],
    value: EmberTensor,
) -> EmberTensor {
    // Make tensor contiguous
    let mut tensor = tensor.to_contiguous();
    let dst_layout = tensor.layout().clone();
    let ndims = dst_layout.num_dims();

    // Get value data
    let value = value.to_contiguous();
    let val_src: Vec<E> = value.storage::<E>().to_vec();

    // Calculate slice info
    let mut slice_info: Vec<(usize, usize, isize)> = Vec::with_capacity(ndims);

    for dim in 0..ndims {
        let dim_size = dst_layout.shape().dims[dim] as isize;

        let slice = if dim < slices.len() {
            &slices[dim]
        } else {
            &Slice::new(0, None, 1)
        };

        let step = slice.step;
        let abs_step = step.unsigned_abs();

        if step > 0 {
            let start = normalize_index(slice.start, dim_size);
            let end = match slice.end {
                Some(e) => normalize_index(e, dim_size).min(dim_size as usize),
                None => dim_size as usize,
            };
            let len = if end > start {
                (end - start + abs_step - 1) / abs_step
            } else {
                0
            };
            slice_info.push((start, len, step));
        } else {
            let start = match slice.start {
                s if s < 0 => (dim_size + s).max(0) as usize,
                s => (s as usize).min((dim_size - 1).max(0) as usize),
            };
            let end = match slice.end {
                Some(e) if e < 0 => (dim_size + e).max(-1) as isize,
                Some(e) => (e as isize - 1).max(-1),
                None => -1,
            };
            let len = if (start as isize) > end {
                ((start as isize - end) as usize + abs_step - 1) / abs_step
            } else {
                0
            };
            slice_info.push((start, len, step));
        }
    }

    // Get mutable access and assign values
    let dst = tensor.storage_mut::<E>();
    let mut indices = vec![0usize; ndims];
    let mut val_idx = 0usize;
    assign_slice_recursive(
        dst,
        &dst_layout,
        &val_src,
        &slice_info,
        &mut indices,
        0,
        &mut val_idx,
    );

    tensor
}

/// Recursively assign values to slice.
fn assign_slice_recursive<E: Clone>(
    dst: &mut [E],
    dst_layout: &Layout,
    val_src: &[E],
    slice_info: &[(usize, usize, isize)],
    indices: &mut [usize],
    dim: usize,
    val_idx: &mut usize,
) {
    let ndims = dst_layout.num_dims();

    if dim == ndims {
        // Base case: assign single element
        let dst_idx = compute_dst_index(dst_layout, slice_info, indices);
        dst[dst_idx] = val_src[*val_idx].clone();
        *val_idx += 1;
        return;
    }

    let (_, len, _) = slice_info[dim];

    for i in 0..len {
        indices[dim] = i;
        assign_slice_recursive(dst, dst_layout, val_src, slice_info, indices, dim + 1, val_idx);
    }
}

/// Compute destination index from output indices and slice info.
fn compute_dst_index(layout: &Layout, slice_info: &[(usize, usize, isize)], out_indices: &[usize]) -> usize {
    let mut idx = layout.start_offset();
    for (dim, &out_i) in out_indices.iter().enumerate() {
        let (start, _, step) = slice_info[dim];
        let dst_i = if step > 0 {
            start + out_i * step as usize
        } else {
            (start as isize - (out_i as isize) * (-step)) as usize
        };
        idx += dst_i * layout.strides()[dim];
    }
    idx
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_backend::TensorData;

    #[test]
    fn test_slice_basic() {
        // Create a 2x3 tensor: [[0, 1, 2], [3, 4, 5]]
        let data: Vec<f32> = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let tensor = EmberTensor::from_data(TensorData::new(data, [2, 3]));

        // Slice [0:1, 1:3] -> [[1, 2]]
        let slices = vec![
            Slice::new(0, Some(1), 1),
            Slice::new(1, Some(3), 1),
        ];
        let result = slice(tensor, &slices);

        assert_eq!(result.layout().shape().dims, vec![1, 2]);
        let result_data = result.into_data();
        let values: Vec<f32> = bytemuck::cast_slice(&result_data.bytes).to_vec();
        assert_eq!(values, vec![1.0, 2.0]);
    }

    #[test]
    fn test_slice_with_step() {
        // Create a 1D tensor: [0, 1, 2, 3, 4, 5]
        let data: Vec<f32> = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let tensor = EmberTensor::from_data(TensorData::new(data, [6]));

        // Slice [0:6:2] -> [0, 2, 4]
        let slices = vec![Slice::new(0, Some(6), 2)];
        let result = slice(tensor, &slices);

        assert_eq!(result.layout().shape().dims, vec![3]);
        let result_data = result.into_data();
        let values: Vec<f32> = bytemuck::cast_slice(&result_data.bytes).to_vec();
        assert_eq!(values, vec![0.0, 2.0, 4.0]);
    }

    #[test]
    fn test_slice_negative_index() {
        // Create a 1D tensor: [0, 1, 2, 3, 4]
        let data: Vec<f32> = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let tensor = EmberTensor::from_data(TensorData::new(data, [5]));

        // Slice [-3:] -> [2, 3, 4]
        let slices = vec![Slice::new(-3, None, 1)];
        let result = slice(tensor, &slices);

        assert_eq!(result.layout().shape().dims, vec![3]);
        let result_data = result.into_data();
        let values: Vec<f32> = bytemuck::cast_slice(&result_data.bytes).to_vec();
        assert_eq!(values, vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_slice_negative_step() {
        // Create a 1D tensor: [0, 1, 2, 3, 4]
        let data: Vec<f32> = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let tensor = EmberTensor::from_data(TensorData::new(data, [5]));

        // Slice [::-1] -> [4, 3, 2, 1, 0] (reverse)
        let slices = vec![Slice::new(4, None, -1)];
        let result = slice(tensor, &slices);

        assert_eq!(result.layout().shape().dims, vec![5]);
        let result_data = result.into_data();
        let values: Vec<f32> = bytemuck::cast_slice(&result_data.bytes).to_vec();
        assert_eq!(values, vec![4.0, 3.0, 2.0, 1.0, 0.0]);
    }
}
