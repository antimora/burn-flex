//! Expand operation for broadcasting tensors to larger shapes.

use burn_std::Shape;

use crate::{EmberTensor, Layout};

/// Compute the broadcast shape of two tensors.
///
/// Returns the shape that both tensors can be expanded to for element-wise operations.
pub fn broadcast_shape(lhs: &Shape, rhs: &Shape) -> Shape {
    let max_dims = lhs.num_dims().max(rhs.num_dims());
    let mut result = vec![0; max_dims];

    for (i, out) in result.iter_mut().enumerate() {
        let lhs_idx = i as isize + lhs.num_dims() as isize - max_dims as isize;
        let rhs_idx = i as isize + rhs.num_dims() as isize - max_dims as isize;

        let lhs_dim = if lhs_idx >= 0 {
            lhs.dims[lhs_idx as usize]
        } else {
            1
        };
        let rhs_dim = if rhs_idx >= 0 {
            rhs.dims[rhs_idx as usize]
        } else {
            1
        };

        if lhs_dim == rhs_dim {
            *out = lhs_dim;
        } else if lhs_dim == 1 {
            *out = rhs_dim;
        } else if rhs_dim == 1 {
            *out = lhs_dim;
        } else {
            panic!(
                "broadcast_shape: incompatible dimensions {} and {} at position {}",
                lhs_dim, rhs_dim, i
            );
        }
    }

    Shape::from(result)
}

/// Broadcast two tensors to the same shape for binary operations.
pub fn broadcast_binary(lhs: EmberTensor, rhs: EmberTensor) -> (EmberTensor, EmberTensor) {
    let lhs_shape = lhs.layout().shape().clone();
    let rhs_shape = rhs.layout().shape().clone();

    if lhs_shape == rhs_shape {
        return (lhs, rhs);
    }

    let target = broadcast_shape(&lhs_shape, &rhs_shape);

    let lhs_expanded = if lhs_shape == target {
        lhs
    } else {
        expand(lhs, target.clone())
    };
    let rhs_expanded = if rhs_shape == target {
        rhs
    } else {
        expand(rhs, target)
    };

    (lhs_expanded, rhs_expanded)
}

/// Expand a tensor to a larger shape by broadcasting.
///
/// Dimensions of size 1 can be expanded to any size. The result is a view
/// that doesn't copy data - it uses stride 0 for expanded dimensions.
pub fn expand(tensor: EmberTensor, target_shape: Shape) -> EmberTensor {
    // Capture values we need before consuming tensor
    let src_dims = tensor.layout().shape().dims.clone();
    let src_strides = tensor.layout().strides().to_vec();
    let start_offset = tensor.layout().start_offset();
    let dtype = tensor.dtype();

    let src_ndims = src_dims.len();
    let target_ndims = target_shape.num_dims();

    // Prepend 1s to source shape if needed (for broadcasting like [3] -> [2, 3])
    let dim_diff = target_ndims.saturating_sub(src_ndims);

    let mut new_strides = Vec::with_capacity(target_ndims);

    for i in 0..target_ndims {
        let target_dim = target_shape.dims[i];

        if i < dim_diff {
            // New dimension prepended - must be broadcastable from size 1
            new_strides.push(0);
        } else {
            let src_idx = i - dim_diff;
            let src_dim = src_dims[src_idx];
            let src_stride = src_strides[src_idx];

            if src_dim == target_dim {
                // Same size - keep stride
                new_strides.push(src_stride);
            } else if src_dim == 1 {
                // Broadcast dimension - stride becomes 0
                new_strides.push(0);
            } else {
                panic!(
                    "expand: cannot expand dimension {} from {} to {}",
                    i, src_dim, target_dim
                );
            }
        }
    }

    let new_layout = Layout::new(target_shape, new_strides, start_offset);
    EmberTensor::new(tensor.into_bytes(), new_layout, dtype)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_backend::TensorData;

    #[test]
    fn test_expand_1d_to_2d() {
        // [3] -> [2, 3]
        let tensor = EmberTensor::from_data(TensorData::new(vec![1.0f32, 2.0, 3.0], [3]));
        let expanded = expand(tensor, Shape::new([2, 3]));

        assert_eq!(expanded.layout().shape().dims, vec![2, 3]);
        assert_eq!(expanded.layout().strides(), &[0, 1]);
    }

    #[test]
    fn test_expand_broadcast_dim() {
        // [3, 1] -> [3, 4]
        let tensor = EmberTensor::from_data(TensorData::new(vec![1.0f32, 2.0, 3.0], [3, 1]));
        let expanded = expand(tensor, Shape::new([3, 4]));

        assert_eq!(expanded.layout().shape().dims, vec![3, 4]);
        // Original strides for [3, 1] would be [1, 1]
        // After expand to [3, 4], stride for dim 1 becomes 0
        assert_eq!(expanded.layout().strides()[1], 0);
    }

    #[test]
    fn test_expand_same_shape() {
        // [2, 3] -> [2, 3] (no change)
        let tensor = EmberTensor::from_data(TensorData::new(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            [2, 3],
        ));
        let original_strides = tensor.layout().strides().to_vec();
        let expanded = expand(tensor, Shape::new([2, 3]));

        assert_eq!(expanded.layout().shape().dims, vec![2, 3]);
        assert_eq!(expanded.layout().strides(), &original_strides);
    }
}
