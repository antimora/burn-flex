use alloc::vec;
use alloc::vec::Vec;
use burn_std::Shape;

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
