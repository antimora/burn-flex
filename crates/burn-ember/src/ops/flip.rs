//! Flip operation for reversing tensor elements along axes.

use alloc::vec::Vec;
use burn_backend::{DType, Element};
use burn_std::{Bytes, bf16, f16};

use crate::{EmberTensor, Layout};

/// Flip tensor elements along specified axes.
///
/// This is a copying operation since our layout uses unsigned strides.
pub fn flip(tensor: EmberTensor, axes: &[usize]) -> EmberTensor {
    let dtype = tensor.dtype();
    match dtype {
        DType::F32 => flip_typed::<f32>(tensor, axes, DType::F32),
        DType::F64 => flip_typed::<f64>(tensor, axes, DType::F64),
        DType::F16 => flip_typed::<f16>(tensor, axes, DType::F16),
        DType::BF16 => flip_typed::<bf16>(tensor, axes, DType::BF16),
        DType::I64 => flip_typed::<i64>(tensor, axes, DType::I64),
        DType::I32 => flip_typed::<i32>(tensor, axes, DType::I32),
        DType::I16 => flip_typed::<i16>(tensor, axes, DType::I16),
        DType::I8 => flip_typed::<i8>(tensor, axes, DType::I8),
        DType::U64 => flip_typed::<u64>(tensor, axes, DType::U64),
        DType::U32 => flip_typed::<u32>(tensor, axes, DType::U32),
        DType::U16 => flip_typed::<u16>(tensor, axes, DType::U16),
        DType::U8 => flip_typed::<u8>(tensor, axes, DType::U8),
        DType::Bool => flip_bool(tensor, axes),
        _ => panic!("flip: unsupported dtype {:?}", dtype),
    }
}

fn flip_typed<T>(tensor: EmberTensor, axes: &[usize], dtype: DType) -> EmberTensor
where
    T: Element + bytemuck::Pod + Copy,
{
    let shape = tensor.layout().shape().clone();
    let ndims = shape.num_dims();

    if axes.is_empty() || shape.num_elements() == 0 {
        return tensor.to_contiguous();
    }

    let tensor = tensor.to_contiguous();
    let storage: &[T] = tensor.storage();

    // Create output with elements in flipped order
    let num_elements = shape.num_elements();
    let mut output = Vec::with_capacity(num_elements);

    // Compute strides for output indexing
    let mut strides = vec![1usize; ndims];
    for i in (0..ndims.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape.dims[i + 1];
    }

    // Iterate through all output positions
    for out_idx in 0..num_elements {
        // Convert linear index to multi-dimensional coords
        let mut coords = vec![0usize; ndims];
        let mut remaining = out_idx;
        for d in 0..ndims {
            coords[d] = remaining / strides[d];
            remaining %= strides[d];
        }

        // Flip coordinates along specified axes
        for &axis in axes {
            coords[axis] = shape.dims[axis] - 1 - coords[axis];
        }

        // Convert back to linear index in input
        let in_idx: usize = coords
            .iter()
            .zip(strides.iter())
            .map(|(&c, &s)| c * s)
            .sum();

        output.push(storage[in_idx]);
    }

    EmberTensor::new(
        Bytes::from_elems(output),
        Layout::contiguous(shape),
        dtype,
    )
}

fn flip_bool(tensor: EmberTensor, axes: &[usize]) -> EmberTensor {
    let shape = tensor.layout().shape().clone();
    let ndims = shape.num_dims();

    if axes.is_empty() || shape.num_elements() == 0 {
        return tensor.to_contiguous();
    }

    let tensor = tensor.to_contiguous();
    let storage: &[u8] = tensor.bytes();

    let num_elements = shape.num_elements();
    let mut output = Vec::with_capacity(num_elements);

    let mut strides = vec![1usize; ndims];
    for i in (0..ndims.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape.dims[i + 1];
    }

    for out_idx in 0..num_elements {
        let mut coords = vec![0usize; ndims];
        let mut remaining = out_idx;
        for d in 0..ndims {
            coords[d] = remaining / strides[d];
            remaining %= strides[d];
        }

        for &axis in axes {
            coords[axis] = shape.dims[axis] - 1 - coords[axis];
        }

        let in_idx: usize = coords
            .iter()
            .zip(strides.iter())
            .map(|(&c, &s)| c * s)
            .sum();

        output.push(storage[in_idx]);
    }

    EmberTensor::new(
        Bytes::from_elems(output),
        Layout::contiguous(shape),
        DType::Bool,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_backend::TensorData;

    #[test]
    fn test_flip_1d() {
        let tensor = EmberTensor::from_data(TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0], [4]));
        let flipped = flip(tensor, &[0]);
        let data: Vec<f32> = flipped.into_data().to_vec().unwrap();
        assert_eq!(data, vec![4.0, 3.0, 2.0, 1.0]);
    }

    #[test]
    fn test_flip_2d_axis0() {
        // [[1, 2], [3, 4]] -> [[3, 4], [1, 2]]
        let tensor =
            EmberTensor::from_data(TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0], [2, 2]));
        let flipped = flip(tensor, &[0]);
        let data: Vec<f32> = flipped.into_data().to_vec().unwrap();
        assert_eq!(data, vec![3.0, 4.0, 1.0, 2.0]);
    }

    #[test]
    fn test_flip_2d_axis1() {
        // [[1, 2], [3, 4]] -> [[2, 1], [4, 3]]
        let tensor =
            EmberTensor::from_data(TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0], [2, 2]));
        let flipped = flip(tensor, &[1]);
        let data: Vec<f32> = flipped.into_data().to_vec().unwrap();
        assert_eq!(data, vec![2.0, 1.0, 4.0, 3.0]);
    }

    #[test]
    fn test_flip_2d_both_axes() {
        // [[1, 2], [3, 4]] -> [[4, 3], [2, 1]]
        let tensor =
            EmberTensor::from_data(TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0], [2, 2]));
        let flipped = flip(tensor, &[0, 1]);
        let data: Vec<f32> = flipped.into_data().to_vec().unwrap();
        assert_eq!(data, vec![4.0, 3.0, 2.0, 1.0]);
    }

    #[test]
    fn test_flip_empty_axes() {
        let tensor = EmberTensor::from_data(TensorData::new(vec![1.0f32, 2.0, 3.0], [3]));
        let flipped = flip(tensor, &[]);
        let data: Vec<f32> = flipped.into_data().to_vec().unwrap();
        assert_eq!(data, vec![1.0, 2.0, 3.0]);
    }
}
