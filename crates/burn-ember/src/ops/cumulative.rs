//! Cumulative operations along a dimension.

use alloc::vec;
use burn_backend::{DType, Element};
use burn_std::Bytes;
use bytemuck::Pod;
use num_traits::Num;

use crate::{EmberTensor, Layout};

/// Cumulative sum along a dimension.
///
/// For each position along `dim`, output contains the sum of all elements
/// from index 0 up to and including that position.
pub fn cumsum<E: Element + Pod + Default + Copy + Num>(
    tensor: EmberTensor,
    dim: usize,
) -> EmberTensor {
    cumulative_op(tensor, dim, E::zero(), |acc, val| acc + val)
}

/// Cumulative product along a dimension.
pub fn cumprod<E: Element + Pod + Default + Copy + Num>(
    tensor: EmberTensor,
    dim: usize,
) -> EmberTensor {
    cumulative_op(tensor, dim, E::one(), |acc, val| acc * val)
}

/// Cumulative minimum along a dimension.
pub fn cummin<E: Element + Pod + Default + Copy + PartialOrd>(
    tensor: EmberTensor,
    dim: usize,
) -> EmberTensor {
    let init = get_max_value::<E>();
    cumulative_op(
        tensor,
        dim,
        init,
        |acc, val| if val < acc { val } else { acc },
    )
}

/// Cumulative maximum along a dimension.
pub fn cummax<E: Element + Pod + Default + Copy + PartialOrd>(
    tensor: EmberTensor,
    dim: usize,
) -> EmberTensor {
    let init = get_min_value::<E>();
    cumulative_op(
        tensor,
        dim,
        init,
        |acc, val| if val > acc { val } else { acc },
    )
}

/// Generic cumulative operation along a dimension.
fn cumulative_op<E: Element + Pod + Default + Copy, F>(
    tensor: EmberTensor,
    dim: usize,
    init: E,
    op: F,
) -> EmberTensor
where
    F: Fn(E, E) -> E,
{
    let tensor = tensor.to_contiguous();
    let shape = tensor.layout().shape().clone();
    let ndims = shape.num_dims();

    assert!(
        dim < ndims,
        "dim {} out of bounds for {} dimensions",
        dim,
        ndims
    );

    let data: &[E] = tensor.storage();
    let total_size = shape.num_elements();
    let mut result = vec![E::default(); total_size];

    // Calculate strides for row-major order
    let mut strides = vec![1usize; ndims];
    for i in (0..ndims - 1).rev() {
        strides[i] = strides[i + 1] * shape.dims[i + 1];
    }

    let dim_size = shape.dims[dim];
    let dim_stride = strides[dim];

    // Calculate the number of "slices" (all dimensions except dim)
    let num_slices = total_size / dim_size;

    // For each slice along all other dimensions
    for slice_idx in 0..num_slices {
        // Calculate the starting position for this slice
        let mut remaining = slice_idx;
        let mut base_idx = 0;

        #[allow(clippy::needless_range_loop)]
        for d in 0..ndims {
            if d == dim {
                continue;
            }
            // Number of complete cycles along dimensions after d (excluding dim)
            let mut cycle_size = 1;
            for dd in (d + 1)..ndims {
                if dd != dim {
                    cycle_size *= shape.dims[dd];
                }
            }
            let coord = remaining / cycle_size;
            remaining %= cycle_size;
            base_idx += coord * strides[d];
        }

        // Scan along the dimension
        let mut acc = init;
        for i in 0..dim_size {
            let idx = base_idx + i * dim_stride;
            acc = op(acc, data[idx]);
            result[idx] = acc;
        }
    }

    let bytes = Bytes::from_elems(result);
    EmberTensor::new(bytes, Layout::contiguous(shape), E::dtype())
}

// Helper functions to get min/max values for different types
fn get_max_value<E: Element + Pod>() -> E {
    // Use bytemuck to transmute appropriate max values
    match E::dtype() {
        DType::F32 => bytemuck::cast(f32::INFINITY),
        DType::F64 => bytemuck::cast(f64::INFINITY),
        DType::I64 => bytemuck::cast(i64::MAX),
        DType::I32 => bytemuck::cast(i32::MAX),
        DType::U8 => bytemuck::cast(u8::MAX),
        _ => panic!("get_max_value: unsupported dtype {:?}", E::dtype()),
    }
}

fn get_min_value<E: Element + Pod>() -> E {
    match E::dtype() {
        DType::F32 => bytemuck::cast(f32::NEG_INFINITY),
        DType::F64 => bytemuck::cast(f64::NEG_INFINITY),
        DType::I64 => bytemuck::cast(i64::MIN),
        DType::I32 => bytemuck::cast(i32::MIN),
        DType::U8 => bytemuck::cast(u8::MIN),
        _ => panic!("get_min_value: unsupported dtype {:?}", E::dtype()),
    }
}

// Type-specific wrappers

pub fn cumsum_f32(tensor: EmberTensor, dim: usize) -> EmberTensor {
    cumsum::<f32>(tensor, dim)
}

pub fn cumsum_f64(tensor: EmberTensor, dim: usize) -> EmberTensor {
    cumsum::<f64>(tensor, dim)
}

pub fn cumsum_i64(tensor: EmberTensor, dim: usize) -> EmberTensor {
    cumsum::<i64>(tensor, dim)
}

pub fn cumprod_f32(tensor: EmberTensor, dim: usize) -> EmberTensor {
    cumprod::<f32>(tensor, dim)
}

pub fn cumprod_f64(tensor: EmberTensor, dim: usize) -> EmberTensor {
    cumprod::<f64>(tensor, dim)
}

pub fn cumprod_i64(tensor: EmberTensor, dim: usize) -> EmberTensor {
    cumprod::<i64>(tensor, dim)
}

pub fn cummin_f32(tensor: EmberTensor, dim: usize) -> EmberTensor {
    cummin::<f32>(tensor, dim)
}

pub fn cummin_f64(tensor: EmberTensor, dim: usize) -> EmberTensor {
    cummin::<f64>(tensor, dim)
}

pub fn cummin_i64(tensor: EmberTensor, dim: usize) -> EmberTensor {
    cummin::<i64>(tensor, dim)
}

pub fn cummax_f32(tensor: EmberTensor, dim: usize) -> EmberTensor {
    cummax::<f32>(tensor, dim)
}

pub fn cummax_f64(tensor: EmberTensor, dim: usize) -> EmberTensor {
    cummax::<f64>(tensor, dim)
}

pub fn cummax_i64(tensor: EmberTensor, dim: usize) -> EmberTensor {
    cummax::<i64>(tensor, dim)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_backend::TensorData;

    #[test]
    fn test_cumsum_1d() {
        // [1, 2, 3, 4] -> [1, 3, 6, 10]
        let tensor = EmberTensor::from_data(TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0], [4]));
        let result = cumsum_f32(tensor, 0);
        let data: Vec<f32> = result.into_data().to_vec().unwrap();
        assert_eq!(data, vec![1.0, 3.0, 6.0, 10.0]);
    }

    #[test]
    fn test_cumsum_2d_dim0() {
        // [[1, 2], [3, 4], [5, 6]] cumsum along dim 0
        // -> [[1, 2], [4, 6], [9, 12]]
        let tensor = EmberTensor::from_data(TensorData::new(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            [3, 2],
        ));
        let result = cumsum_f32(tensor, 0);
        let data: Vec<f32> = result.into_data().to_vec().unwrap();
        assert_eq!(data, vec![1.0, 2.0, 4.0, 6.0, 9.0, 12.0]);
    }

    #[test]
    fn test_cumsum_2d_dim1() {
        // [[1, 2, 3], [4, 5, 6]] cumsum along dim 1
        // -> [[1, 3, 6], [4, 9, 15]]
        let tensor = EmberTensor::from_data(TensorData::new(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            [2, 3],
        ));
        let result = cumsum_f32(tensor, 1);
        let data: Vec<f32> = result.into_data().to_vec().unwrap();
        assert_eq!(data, vec![1.0, 3.0, 6.0, 4.0, 9.0, 15.0]);
    }

    #[test]
    fn test_cumprod_1d() {
        // [1, 2, 3, 4] -> [1, 2, 6, 24]
        let tensor = EmberTensor::from_data(TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0], [4]));
        let result = cumprod_f32(tensor, 0);
        let data: Vec<f32> = result.into_data().to_vec().unwrap();
        assert_eq!(data, vec![1.0, 2.0, 6.0, 24.0]);
    }

    #[test]
    fn test_cummin_1d() {
        // [3, 1, 4, 1, 5] -> [3, 1, 1, 1, 1]
        let tensor = EmberTensor::from_data(TensorData::new(vec![3.0f32, 1.0, 4.0, 1.0, 5.0], [5]));
        let result = cummin_f32(tensor, 0);
        let data: Vec<f32> = result.into_data().to_vec().unwrap();
        assert_eq!(data, vec![3.0, 1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_cummax_1d() {
        // [3, 1, 4, 1, 5] -> [3, 3, 4, 4, 5]
        let tensor = EmberTensor::from_data(TensorData::new(vec![3.0f32, 1.0, 4.0, 1.0, 5.0], [5]));
        let result = cummax_f32(tensor, 0);
        let data: Vec<f32> = result.into_data().to_vec().unwrap();
        assert_eq!(data, vec![3.0, 3.0, 4.0, 4.0, 5.0]);
    }

    #[test]
    fn test_cumsum_i64() {
        let tensor = EmberTensor::from_data(TensorData::new(vec![1i64, 2, 3, 4], [4]));
        let result = cumsum_i64(tensor, 0);
        let data: Vec<i64> = result.into_data().to_vec().unwrap();
        assert_eq!(data, vec![1, 3, 6, 10]);
    }

    #[test]
    fn test_cummax_2d_dim1() {
        // [[1, 3, 2], [4, 2, 5]] cummax along dim 1
        // -> [[1, 3, 3], [4, 4, 5]]
        let tensor = EmberTensor::from_data(TensorData::new(
            vec![1.0f32, 3.0, 2.0, 4.0, 2.0, 5.0],
            [2, 3],
        ));
        let result = cummax_f32(tensor, 1);
        let data: Vec<f32> = result.into_data().to_vec().unwrap();
        assert_eq!(data, vec![1.0, 3.0, 3.0, 4.0, 4.0, 5.0]);
    }
}
