//! Unfold operation for sliding window extraction.

use alloc::vec;
use alloc::vec::Vec;
use burn_backend::Element;
use burn_std::{Bytes, Shape};
use bytemuck::Pod;

use crate::{EmberTensor, Layout};

/// Calculate the number of windows that can be extracted from a dimension.
#[inline]
fn calculate_windows(dim_size: usize, window_size: usize, step: usize) -> usize {
    assert!(step > 0, "step must be positive");
    if dim_size + step < window_size {
        0
    } else {
        (dim_size + step - window_size) / step
    }
}

/// Unfold: extract sliding windows from a tensor along a dimension.
///
/// Given a tensor with shape `[pre..., dim_size, post...]`, extracts windows of
/// `size` elements along dimension `dim`, stepping by `step`.
///
/// Returns a tensor with shape `[pre..., windows, post..., size]` where:
/// - `windows = (dim_size - size + step) / step`
/// - The `size` dimension is appended at the end
pub fn unfold<E: Element + Pod + Default + Copy>(
    tensor: EmberTensor,
    dim: usize,
    size: usize,
    step: usize,
) -> EmberTensor {
    let tensor = tensor.to_contiguous();
    let shape = tensor.layout().shape();
    let ndims = shape.num_dims();

    assert!(dim < ndims, "dim {} out of bounds for {} dimensions", dim, ndims);
    assert!(size > 0, "window size must be positive");
    assert!(step > 0, "step must be positive");
    assert!(
        shape.dims[dim] >= size,
        "dimension {} has size {} which is smaller than window size {}",
        dim, shape.dims[dim], size
    );

    let dim_size = shape.dims[dim];
    let windows = calculate_windows(dim_size, size, step);

    // Build output shape: [pre..., windows, post..., size]
    let mut output_dims: Vec<usize> = Vec::with_capacity(ndims + 1);
    for (d, &s) in shape.dims.iter().enumerate() {
        if d == dim {
            output_dims.push(windows);
        } else {
            output_dims.push(s);
        }
    }
    output_dims.push(size); // Append size at the end

    let output_shape = Shape::from(output_dims);
    let output_size = output_shape.num_elements();

    let tensor_data: &[E] = tensor.storage();

    // Compute input strides
    let input_strides = compute_strides(&shape.dims);

    // Compute output strides
    let output_strides = compute_strides(&output_shape.dims);

    let mut result = vec![E::default(); output_size];

    // For each output position, compute the corresponding input position
    for out_idx in 0..output_size {
        let mut remaining = out_idx;
        let mut in_idx = 0;

        // Process dimensions up to and including the unfolded dimension
        for d in 0..ndims {
            let coord = remaining / output_strides[d];
            remaining %= output_strides[d];

            if d == dim {
                // This is the window index; multiply by step to get starting position
                in_idx += coord * step * input_strides[d];
            } else {
                in_idx += coord * input_strides[d];
            }
        }

        // The last dimension of output is the position within the window
        let window_pos = remaining; // remaining after processing ndims dimensions
        in_idx += window_pos * input_strides[dim];

        result[out_idx] = tensor_data[in_idx];
    }

    let bytes = Bytes::from_elems(result);
    EmberTensor::new(bytes, Layout::contiguous(output_shape), E::dtype())
}

/// Compute row-major strides for a shape.
#[inline]
fn compute_strides(dims: &[usize]) -> Vec<usize> {
    let ndims = dims.len();
    let mut strides = vec![1usize; ndims];
    for i in (0..ndims.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * dims[i + 1];
    }
    strides
}

// Type-specific wrappers

pub fn unfold_f32(tensor: EmberTensor, dim: usize, size: usize, step: usize) -> EmberTensor {
    unfold::<f32>(tensor, dim, size, step)
}

pub fn unfold_f64(tensor: EmberTensor, dim: usize, size: usize, step: usize) -> EmberTensor {
    unfold::<f64>(tensor, dim, size, step)
}

pub fn unfold_bool(tensor: EmberTensor, dim: usize, size: usize, step: usize) -> EmberTensor {
    unfold::<u8>(tensor, dim, size, step)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_backend::TensorData;

    #[test]
    fn test_unfold_1d() {
        // Input: [1, 2, 3, 4, 5] shape [5]
        // Unfold dim=0, size=3, step=1
        // Windows: (5 - 3 + 1) / 1 = 3
        // Output shape: [3, 3]
        // Window 0: [1, 2, 3]
        // Window 1: [2, 3, 4]
        // Window 2: [3, 4, 5]
        let tensor = EmberTensor::from_data(TensorData::new(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0],
            [5],
        ));
        let result = unfold_f32(tensor, 0, 3, 1);
        assert_eq!(result.layout().shape().dims, vec![3, 3]);
        let data: Vec<f32> = result.into_data().to_vec().unwrap();
        assert_eq!(data, vec![1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_unfold_1d_step2() {
        // Input: [1, 2, 3, 4, 5, 6] shape [6]
        // Unfold dim=0, size=3, step=2
        // Windows: (6 - 3 + 2) / 2 = 2
        // Output shape: [2, 3]
        // Window 0: [1, 2, 3]
        // Window 1: [3, 4, 5]
        let tensor = EmberTensor::from_data(TensorData::new(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            [6],
        ));
        let result = unfold_f32(tensor, 0, 3, 2);
        assert_eq!(result.layout().shape().dims, vec![2, 3]);
        let data: Vec<f32> = result.into_data().to_vec().unwrap();
        assert_eq!(data, vec![1.0, 2.0, 3.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_unfold_2d_dim1() {
        // Input: [[1, 2, 3, 4], [5, 6, 7, 8]] shape [2, 4]
        // Unfold dim=1, size=2, step=1
        // Windows: (4 - 2 + 1) / 1 = 3
        // Output shape: [2, 3, 2]
        let tensor = EmberTensor::from_data(TensorData::new(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            [2, 4],
        ));
        let result = unfold_f32(tensor, 1, 2, 1);
        assert_eq!(result.layout().shape().dims, vec![2, 3, 2]);
        let data: Vec<f32> = result.into_data().to_vec().unwrap();
        // Row 0: windows [1,2], [2,3], [3,4]
        // Row 1: windows [5,6], [6,7], [7,8]
        assert_eq!(
            data,
            vec![1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 5.0, 6.0, 6.0, 7.0, 7.0, 8.0]
        );
    }
}
