//! Mask operations for conditional element replacement.

use alloc::vec::Vec;
use burn_backend::{DType, Element};
use burn_std::{Bytes, bf16, f16};

use crate::{EmberTensor, Layout};

/// Fill tensor elements with a value where mask is true.
///
/// mask_fill(tensor, mask, value) -> tensor with elements replaced where mask is true
pub fn mask_fill<T>(tensor: EmberTensor, mask: EmberTensor, value: T) -> EmberTensor
where
    T: Element + bytemuck::Pod + Copy,
{
    let dtype = tensor.dtype();

    // Broadcast mask to tensor shape if needed
    let (tensor, mask) = crate::ops::expand::broadcast_binary(tensor, mask);

    let tensor = tensor.to_contiguous();
    let mask = mask.to_contiguous();

    let shape = tensor.layout().shape().clone();
    let tensor_data: &[T] = tensor.storage();
    let mask_data: &[u8] = mask.bytes();

    let result: Vec<T> = tensor_data
        .iter()
        .zip(mask_data.iter())
        .map(|(&elem, &m)| if m != 0 { value } else { elem })
        .collect();

    EmberTensor::new(Bytes::from_elems(result), Layout::contiguous(shape), dtype)
}

/// Mask fill for f32.
pub fn mask_fill_f32(tensor: EmberTensor, mask: EmberTensor, value: f32) -> EmberTensor {
    mask_fill(tensor, mask, value)
}

/// Mask fill for f64.
pub fn mask_fill_f64(tensor: EmberTensor, mask: EmberTensor, value: f64) -> EmberTensor {
    mask_fill(tensor, mask, value)
}

/// Mask fill for f16.
pub fn mask_fill_f16(tensor: EmberTensor, mask: EmberTensor, value: f16) -> EmberTensor {
    mask_fill(tensor, mask, value)
}

/// Mask fill for bf16.
pub fn mask_fill_bf16(tensor: EmberTensor, mask: EmberTensor, value: bf16) -> EmberTensor {
    mask_fill(tensor, mask, value)
}

/// Mask fill for i64.
pub fn mask_fill_i64(tensor: EmberTensor, mask: EmberTensor, value: i64) -> EmberTensor {
    mask_fill(tensor, mask, value)
}

/// Mask fill for bool tensors.
pub fn mask_fill_bool(tensor: EmberTensor, mask: EmberTensor, value: bool) -> EmberTensor {
    let (tensor, mask) = crate::ops::expand::broadcast_binary(tensor, mask);

    let tensor = tensor.to_contiguous();
    let mask = mask.to_contiguous();

    let shape = tensor.layout().shape().clone();
    let tensor_data: &[u8] = tensor.bytes();
    let mask_data: &[u8] = mask.bytes();
    let value_u8 = value as u8;

    let result: Vec<u8> = tensor_data
        .iter()
        .zip(mask_data.iter())
        .map(|(&elem, &m)| if m != 0 { value_u8 } else { elem })
        .collect();

    EmberTensor::new(
        Bytes::from_elems(result),
        Layout::contiguous(shape),
        DType::Bool,
    )
}

/// Replace elements from value tensor where mask is true.
///
/// mask_where(tensor, mask, value) -> tensor with elements from value where mask is true
pub fn mask_where<T>(tensor: EmberTensor, mask: EmberTensor, value: EmberTensor) -> EmberTensor
where
    T: Element + bytemuck::Pod + Copy,
{
    let dtype = tensor.dtype();

    // Broadcast all to the same shape
    let target_shape =
        crate::ops::expand::broadcast_shape(tensor.layout().shape(), mask.layout().shape());
    let target_shape = crate::ops::expand::broadcast_shape(&target_shape, value.layout().shape());

    let tensor = if tensor.layout().shape() == &target_shape {
        tensor
    } else {
        crate::ops::expand::expand(tensor, target_shape.clone())
    };
    let mask = if mask.layout().shape() == &target_shape {
        mask
    } else {
        crate::ops::expand::expand(mask, target_shape.clone())
    };
    let value = if value.layout().shape() == &target_shape {
        value
    } else {
        crate::ops::expand::expand(value, target_shape.clone())
    };

    let tensor = tensor.to_contiguous();
    let mask = mask.to_contiguous();
    let value = value.to_contiguous();

    let shape = tensor.layout().shape().clone();
    let tensor_data: &[T] = tensor.storage();
    let mask_data: &[u8] = mask.bytes();
    let value_data: &[T] = value.storage();

    let result: Vec<T> = tensor_data
        .iter()
        .zip(mask_data.iter())
        .zip(value_data.iter())
        .map(|((&t, &m), &v)| if m != 0 { v } else { t })
        .collect();

    EmberTensor::new(Bytes::from_elems(result), Layout::contiguous(shape), dtype)
}

/// Mask where for f32.
pub fn mask_where_f32(tensor: EmberTensor, mask: EmberTensor, value: EmberTensor) -> EmberTensor {
    mask_where::<f32>(tensor, mask, value)
}

/// Mask where for f64.
pub fn mask_where_f64(tensor: EmberTensor, mask: EmberTensor, value: EmberTensor) -> EmberTensor {
    mask_where::<f64>(tensor, mask, value)
}

/// Mask where for f16.
pub fn mask_where_f16(tensor: EmberTensor, mask: EmberTensor, value: EmberTensor) -> EmberTensor {
    mask_where::<f16>(tensor, mask, value)
}

/// Mask where for bf16.
pub fn mask_where_bf16(tensor: EmberTensor, mask: EmberTensor, value: EmberTensor) -> EmberTensor {
    mask_where::<bf16>(tensor, mask, value)
}

/// Mask where for i64.
pub fn mask_where_i64(tensor: EmberTensor, mask: EmberTensor, value: EmberTensor) -> EmberTensor {
    mask_where::<i64>(tensor, mask, value)
}

/// Mask where for bool tensors.
pub fn mask_where_bool(tensor: EmberTensor, mask: EmberTensor, value: EmberTensor) -> EmberTensor {
    let target_shape =
        crate::ops::expand::broadcast_shape(tensor.layout().shape(), mask.layout().shape());
    let target_shape = crate::ops::expand::broadcast_shape(&target_shape, value.layout().shape());

    let tensor = if tensor.layout().shape() == &target_shape {
        tensor
    } else {
        crate::ops::expand::expand(tensor, target_shape.clone())
    };
    let mask = if mask.layout().shape() == &target_shape {
        mask
    } else {
        crate::ops::expand::expand(mask, target_shape.clone())
    };
    let value = if value.layout().shape() == &target_shape {
        value
    } else {
        crate::ops::expand::expand(value, target_shape)
    };

    let tensor = tensor.to_contiguous();
    let mask = mask.to_contiguous();
    let value = value.to_contiguous();

    let shape = tensor.layout().shape().clone();
    let tensor_data: &[u8] = tensor.bytes();
    let mask_data: &[u8] = mask.bytes();
    let value_data: &[u8] = value.bytes();

    let result: Vec<u8> = tensor_data
        .iter()
        .zip(mask_data.iter())
        .zip(value_data.iter())
        .map(|((&t, &m), &v)| if m != 0 { v } else { t })
        .collect();

    EmberTensor::new(
        Bytes::from_elems(result),
        Layout::contiguous(shape),
        DType::Bool,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_backend::TensorData;

    #[test]
    fn test_mask_fill_f32() {
        let tensor = EmberTensor::from_data(TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0], [4]));
        let mask = EmberTensor::from_data(TensorData::new(vec![true, false, true, false], [4]));

        let result = mask_fill_f32(tensor, mask, 0.0);
        let data: Vec<f32> = result.into_data().to_vec().unwrap();
        assert_eq!(data, vec![0.0, 2.0, 0.0, 4.0]);
    }

    #[test]
    fn test_mask_fill_2d() {
        let tensor = EmberTensor::from_data(TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0], [2, 2]));
        let mask = EmberTensor::from_data(TensorData::new(vec![true, false, false, true], [2, 2]));

        let result = mask_fill_f32(tensor, mask, -1.0);
        let data: Vec<f32> = result.into_data().to_vec().unwrap();
        assert_eq!(data, vec![-1.0, 2.0, 3.0, -1.0]);
    }

    #[test]
    fn test_mask_where_f32() {
        let tensor = EmberTensor::from_data(TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0], [4]));
        let mask = EmberTensor::from_data(TensorData::new(vec![true, false, true, false], [4]));
        let value = EmberTensor::from_data(TensorData::new(vec![10.0f32, 20.0, 30.0, 40.0], [4]));

        let result = mask_where_f32(tensor, mask, value);
        let data: Vec<f32> = result.into_data().to_vec().unwrap();
        assert_eq!(data, vec![10.0, 2.0, 30.0, 4.0]);
    }

    #[test]
    fn test_mask_fill_broadcast() {
        // Tensor [2, 3], mask [3] (broadcasts to [2, 3])
        let tensor = EmberTensor::from_data(TensorData::new(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            [2, 3],
        ));
        let mask = EmberTensor::from_data(TensorData::new(vec![true, false, true], [3]));

        let result = mask_fill_f32(tensor, mask, 0.0);
        let data: Vec<f32> = result.into_data().to_vec().unwrap();
        // Columns 0 and 2 should be filled with 0
        assert_eq!(data, vec![0.0, 2.0, 0.0, 0.0, 5.0, 0.0]);
    }
}
