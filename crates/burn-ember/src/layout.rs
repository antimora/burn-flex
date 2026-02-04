use alloc::vec;
use alloc::vec::Vec;
use burn_std::{Shape, Slice};

/// Layout describes how to interpret a linear buffer as an N-dimensional tensor.
///
/// Stores shape, strides (in elements), and an optional start offset for views/slices.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Layout {
    shape: Shape,
    strides: Vec<usize>,
    start_offset: usize,
}

impl Layout {
    /// Create a new contiguous layout (row-major/C-order).
    pub fn contiguous(shape: Shape) -> Self {
        let ndims = shape.num_dims();
        let mut strides = vec![1usize; ndims];

        // Compute strides from right to left
        for i in (0..ndims.saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * shape.dims[i + 1];
        }

        Self {
            shape,
            strides,
            start_offset: 0,
        }
    }

    /// Create a layout with explicit strides.
    pub fn new(shape: Shape, strides: Vec<usize>, start_offset: usize) -> Self {
        debug_assert_eq!(shape.num_dims(), strides.len());
        Self {
            shape,
            strides,
            start_offset,
        }
    }

    /// The shape of the tensor.
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// The strides in elements.
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    /// The start offset for views/slices.
    pub fn start_offset(&self) -> usize {
        self.start_offset
    }

    /// Number of dimensions.
    pub fn num_dims(&self) -> usize {
        self.shape.num_dims()
    }

    /// Total number of elements.
    pub fn num_elements(&self) -> usize {
        self.shape.num_elements()
    }

    /// Check if this layout is contiguous (row-major).
    pub fn is_contiguous(&self) -> bool {
        if self.shape.num_dims() == 0 {
            return true;
        }

        let mut expected_stride = 1usize;
        for i in (0..self.shape.num_dims()).rev() {
            if self.strides[i] != expected_stride {
                return false;
            }
            expected_stride *= self.shape.dims[i];
        }
        true
    }

    /// If contiguous, return (start, end) offsets for direct slice access.
    pub fn contiguous_offsets(&self) -> Option<(usize, usize)> {
        if self.is_contiguous() {
            Some((self.start_offset, self.start_offset + self.num_elements()))
        } else {
            None
        }
    }

    /// Transpose: swap two dimensions (zero-copy, metadata only).
    pub fn transpose(&self, dim1: usize, dim2: usize) -> Self {
        let mut dims = self.shape.dims.clone();
        let mut strides = self.strides.clone();
        dims.swap(dim1, dim2);
        strides.swap(dim1, dim2);
        Self {
            shape: Shape::from(dims),
            strides,
            start_offset: self.start_offset,
        }
    }

    /// Narrow/slice along a dimension (zero-copy, metadata only).
    pub fn narrow(&self, dim: usize, start: usize, len: usize) -> Self {
        debug_assert!(
            start + len <= self.shape.dims[dim],
            "narrow: start ({}) + len ({}) exceeds dimension size ({})",
            start,
            len,
            self.shape.dims[dim]
        );
        let mut dims = self.shape.dims.clone();
        dims[dim] = len;
        Self {
            shape: Shape::from(dims),
            strides: self.strides.clone(),
            start_offset: self.start_offset + self.strides[dim] * start,
        }
    }

    /// Apply slices to create a new layout (zero-copy for positive steps).
    ///
    /// Returns `(new_layout, needs_copy)`:
    /// - `needs_copy = false`: Can use zero-copy view with new layout
    /// - `needs_copy = true`: Has negative steps, requires data copy
    pub fn slice(&self, slices: &[Slice]) -> (Self, bool) {
        let ndims = self.num_dims();
        let mut new_dims = self.shape.dims.clone();
        let mut new_strides = self.strides.clone();
        let mut new_offset = self.start_offset;
        let mut needs_copy = false;

        for (dim, slice) in slices.iter().enumerate() {
            if dim >= ndims {
                break;
            }

            let dim_size = self.shape.dims[dim] as isize;
            let stride = self.strides[dim];

            // Normalize start index (handle negative)
            let start = if slice.start < 0 {
                (dim_size + slice.start).max(0) as usize
            } else {
                (slice.start as usize).min(dim_size as usize)
            };

            // Normalize end index (handle negative and None)
            let end = match slice.end {
                Some(e) if e < 0 => (dim_size + e).max(0) as usize,
                Some(e) => (e as usize).min(dim_size as usize),
                None if slice.step > 0 => dim_size as usize,
                None => 0, // For negative step with no end, go to beginning
            };

            let step = slice.step;

            if step > 0 {
                // Positive step: forward iteration
                let len = if end > start {
                    (end - start + step as usize - 1) / step as usize
                } else {
                    0
                };
                new_dims[dim] = len;
                new_strides[dim] = stride * step as usize;
                new_offset += stride * start;
            } else {
                // Negative step: reverse iteration - requires copy
                needs_copy = true;
                let abs_step = (-step) as usize;
                // For negative step, start from higher index going down
                let (actual_start, actual_end) = if slice.end.is_none() {
                    // No end specified: start from slice.start going to 0
                    let s = if slice.start < 0 {
                        (dim_size + slice.start).max(0) as usize
                    } else {
                        (slice.start as usize).min(dim_size.saturating_sub(1) as usize)
                    };
                    (s, 0)
                } else {
                    (start, end)
                };
                let len = if actual_start >= actual_end {
                    (actual_start - actual_end + abs_step) / abs_step
                } else {
                    0
                };
                new_dims[dim] = len;
                new_strides[dim] = stride; // Will be handled during copy
            }
        }

        (
            Self {
                shape: Shape::from(new_dims),
                strides: new_strides,
                start_offset: new_offset,
            },
            needs_copy,
        )
    }

    /// Reshape to a new shape. Only works if contiguous with zero offset.
    ///
    /// Returns None if not contiguous or has non-zero offset (would require data copy).
    pub fn reshape(&self, new_shape: Shape) -> Option<Self> {
        if !self.is_contiguous() || self.start_offset != 0 {
            return None;
        }
        debug_assert_eq!(
            self.num_elements(),
            new_shape.num_elements(),
            "reshape must preserve total elements"
        );
        Some(Self::contiguous(new_shape))
    }

    /// Compute linear index from multi-dimensional indices.
    pub fn index(&self, indices: &[usize]) -> usize {
        debug_assert_eq!(indices.len(), self.num_dims());
        let mut offset = self.start_offset;
        for (i, &idx) in indices.iter().enumerate() {
            offset += idx * self.strides[i];
        }
        offset
    }

    /// Get stride of the innermost (last) dimension.
    /// Returns 1 for contiguous tensors, larger values for transposed.
    pub fn inner_stride(&self) -> usize {
        self.strides.last().copied().unwrap_or(1)
    }

    /// Check if innermost dimension is contiguous (stride == 1).
    /// This enables efficient vectorized inner loops.
    pub fn has_contiguous_inner(&self) -> bool {
        self.inner_stride() == 1
    }

    /// For 2D layouts, get (outer_size, inner_size, outer_stride, inner_stride).
    /// Returns None if not 2D.
    pub fn as_2d_strides(&self) -> Option<(usize, usize, usize, usize)> {
        if self.num_dims() != 2 {
            return None;
        }
        Some((
            self.shape.dims[0],
            self.shape.dims[1],
            self.strides[0],
            self.strides[1],
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_contiguous_layout() {
        let layout = Layout::contiguous(Shape::from(vec![2, 3, 4]));
        assert_eq!(layout.strides(), &[12, 4, 1]);
        assert!(layout.is_contiguous());
    }

    #[test]
    fn test_transpose() {
        let layout = Layout::contiguous(Shape::from(vec![2, 3]));
        let transposed = layout.transpose(0, 1);
        assert_eq!(transposed.shape().dims, vec![3, 2]);
        assert_eq!(transposed.strides(), &[1, 3]);
        assert!(!transposed.is_contiguous());
    }

    #[test]
    fn test_narrow() {
        let layout = Layout::contiguous(Shape::from(vec![4, 4]));
        let narrowed = layout.narrow(0, 1, 2);
        assert_eq!(narrowed.shape().dims, vec![2, 4]);
        assert_eq!(narrowed.start_offset(), 4);
    }

    #[test]
    fn test_contiguous_offsets() {
        let layout = Layout::contiguous(Shape::from(vec![2, 3]));
        assert_eq!(layout.contiguous_offsets(), Some((0, 6)));
    }

    #[test]
    fn test_index() {
        let layout = Layout::contiguous(Shape::from(vec![2, 3]));
        assert_eq!(layout.index(&[0, 0]), 0);
        assert_eq!(layout.index(&[0, 2]), 2);
        assert_eq!(layout.index(&[1, 0]), 3);
        assert_eq!(layout.index(&[1, 2]), 5);
    }
}
