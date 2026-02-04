//! Bool tensor operations for the Ember backend.

use burn_backend::{
    DType, ExecutionError, TensorData,
    ops::BoolTensorOps,
    tensor::{BoolTensor, Device, FloatTensor, IntTensor},
};
use burn_std::{Shape, Slice};

use crate::{Ember, EmberTensor};

impl BoolTensorOps<Ember> for Ember {
    fn bool_from_data(data: TensorData, _device: &Device<Ember>) -> BoolTensor<Ember> {
        EmberTensor::from_data(data)
    }

    async fn bool_into_data(tensor: BoolTensor<Ember>) -> Result<TensorData, ExecutionError> {
        Ok(tensor.into_data())
    }

    fn bool_device(_tensor: &BoolTensor<Ember>) -> Device<Ember> {
        Default::default()
    }

    fn bool_to_device(tensor: BoolTensor<Ember>, _device: &Device<Ember>) -> BoolTensor<Ember> {
        tensor
    }

    fn bool_reshape(tensor: BoolTensor<Ember>, shape: Shape) -> BoolTensor<Ember> {
        tensor.reshape(shape)
    }

    fn bool_slice(tensor: BoolTensor<Ember>, slices: &[Slice]) -> BoolTensor<Ember> {
        crate::ops::slice::slice(tensor, slices)
    }

    fn bool_empty(shape: Shape, _device: &Device<Ember>) -> BoolTensor<Ember> {
        EmberTensor::empty(shape, DType::Bool)
    }

    fn bool_slice_assign(
        tensor: BoolTensor<Ember>,
        slices: &[Slice],
        value: BoolTensor<Ember>,
    ) -> BoolTensor<Ember> {
        crate::ops::slice::slice_assign(tensor, slices, value)
    }

    fn bool_into_int(_tensor: BoolTensor<Ember>) -> IntTensor<Ember> {
        todo!("bool_into_int")
    }

    fn bool_into_float(_tensor: BoolTensor<Ember>) -> FloatTensor<Ember> {
        todo!("bool_into_float")
    }

    fn bool_swap_dims(tensor: BoolTensor<Ember>, dim1: usize, dim2: usize) -> BoolTensor<Ember> {
        tensor.transpose(dim1, dim2)
    }

    fn bool_permute(_tensor: BoolTensor<Ember>, _axes: &[usize]) -> BoolTensor<Ember> {
        todo!("bool_permute")
    }

    fn bool_flip(_tensor: BoolTensor<Ember>, _axes: &[usize]) -> BoolTensor<Ember> {
        todo!("bool_flip")
    }

    fn bool_equal(lhs: BoolTensor<Ember>, rhs: BoolTensor<Ember>) -> BoolTensor<Ember> {
        use crate::Layout;
        use crate::strided_index::StridedIter;
        use burn_backend::DType;
        use burn_std::Bytes;

        debug_assert_eq!(
            lhs.layout().shape(),
            rhs.layout().shape(),
            "bool_equal: shape mismatch"
        );

        let shape = lhs.layout().shape().clone();
        let lhs_storage: &[u8] = lhs.bytes();
        let rhs_storage: &[u8] = rhs.bytes();

        let result: Vec<u8> = match (
            lhs.layout().contiguous_offsets(),
            rhs.layout().contiguous_offsets(),
        ) {
            (Some((l_start, l_end)), Some((r_start, r_end))) => {
                let l_slice = &lhs_storage[l_start..l_end];
                let r_slice = &rhs_storage[r_start..r_end];
                l_slice
                    .iter()
                    .zip(r_slice)
                    .map(|(&a, &b)| (a == b) as u8)
                    .collect()
            }
            _ => {
                let lhs_iter = StridedIter::new(lhs.layout());
                let rhs_iter = StridedIter::new(rhs.layout());
                lhs_iter
                    .zip(rhs_iter)
                    .map(|(li, ri)| (lhs_storage[li] == rhs_storage[ri]) as u8)
                    .collect()
            }
        };

        let bytes = Bytes::from_elems(result);
        EmberTensor::new(bytes, Layout::contiguous(shape), DType::Bool)
    }

    fn bool_not(tensor: BoolTensor<Ember>) -> BoolTensor<Ember> {
        use crate::Layout;
        use crate::strided_index::StridedIter;
        use burn_backend::DType;
        use burn_std::Bytes;

        let shape = tensor.layout().shape().clone();
        let storage: &[u8] = tensor.bytes();

        let result: Vec<u8> = match tensor.layout().contiguous_offsets() {
            Some((start, end)) => storage[start..end]
                .iter()
                .map(|&v| (v == 0) as u8)
                .collect(),
            None => StridedIter::new(tensor.layout())
                .map(|idx| (storage[idx] == 0) as u8)
                .collect(),
        };

        let bytes = Bytes::from_elems(result);
        EmberTensor::new(bytes, Layout::contiguous(shape), DType::Bool)
    }

    fn bool_and(lhs: BoolTensor<Ember>, rhs: BoolTensor<Ember>) -> BoolTensor<Ember> {
        bool_binary_op(lhs, rhs, |a, b| a & b)
    }

    fn bool_or(lhs: BoolTensor<Ember>, rhs: BoolTensor<Ember>) -> BoolTensor<Ember> {
        bool_binary_op(lhs, rhs, |a, b| a | b)
    }

    fn bool_xor(lhs: BoolTensor<Ember>, rhs: BoolTensor<Ember>) -> BoolTensor<Ember> {
        bool_binary_op(lhs, rhs, |a, b| a ^ b)
    }

    fn bool_expand(tensor: BoolTensor<Ember>, shape: Shape) -> BoolTensor<Ember> {
        crate::ops::expand::expand(tensor, shape)
    }

    // Missing methods
    fn bool_zeros(shape: Shape, device: &Device<Ember>) -> BoolTensor<Ember> {
        Self::bool_empty(shape, device)
    }

    fn bool_ones(_shape: Shape, _device: &Device<Ember>) -> BoolTensor<Ember> {
        todo!("bool_ones")
    }

    fn bool_mask_where(
        _tensor: BoolTensor<Ember>,
        _mask: BoolTensor<Ember>,
        _value: BoolTensor<Ember>,
    ) -> BoolTensor<Ember> {
        todo!("bool_mask_where")
    }

    fn bool_mask_fill(
        _tensor: BoolTensor<Ember>,
        _mask: BoolTensor<Ember>,
        _value: bool,
    ) -> BoolTensor<Ember> {
        todo!("bool_mask_fill")
    }

    fn bool_gather(
        _dim: usize,
        _tensor: BoolTensor<Ember>,
        _indices: IntTensor<Ember>,
    ) -> BoolTensor<Ember> {
        todo!("bool_gather")
    }

    fn bool_scatter_or(
        _dim: usize,
        _tensor: BoolTensor<Ember>,
        _indices: IntTensor<Ember>,
        _value: BoolTensor<Ember>,
    ) -> BoolTensor<Ember> {
        todo!("bool_scatter_or")
    }

    fn bool_equal_elem(lhs: BoolTensor<Ember>, rhs: bool) -> BoolTensor<Ember> {
        use crate::Layout;
        use crate::strided_index::StridedIter;
        use burn_std::Bytes;

        let shape = lhs.layout().shape().clone();
        let storage: &[u8] = lhs.bytes();
        let rhs_val = rhs as u8;

        let result: Vec<u8> = match lhs.layout().contiguous_offsets() {
            Some((start, end)) => storage[start..end]
                .iter()
                .map(|&v| (v == rhs_val) as u8)
                .collect(),
            None => StridedIter::new(lhs.layout())
                .map(|idx| (storage[idx] == rhs_val) as u8)
                .collect(),
        };

        let bytes = Bytes::from_elems(result);
        EmberTensor::new(bytes, Layout::contiguous(shape), DType::Bool)
    }

    fn bool_unfold(
        _tensor: BoolTensor<Ember>,
        _dim: usize,
        _size: usize,
        _step: usize,
    ) -> BoolTensor<Ember> {
        todo!("bool_unfold")
    }
}

fn bool_binary_op<F>(lhs: EmberTensor, rhs: EmberTensor, op: F) -> EmberTensor
where
    F: Fn(u8, u8) -> u8,
{
    use crate::Layout;
    use crate::strided_index::StridedIter;
    use burn_std::Bytes;

    debug_assert_eq!(
        lhs.layout().shape(),
        rhs.layout().shape(),
        "bool_binary_op: shape mismatch"
    );

    let shape = lhs.layout().shape().clone();
    let lhs_storage: &[u8] = lhs.bytes();
    let rhs_storage: &[u8] = rhs.bytes();

    let result: Vec<u8> = match (
        lhs.layout().contiguous_offsets(),
        rhs.layout().contiguous_offsets(),
    ) {
        (Some((l_start, l_end)), Some((r_start, r_end))) => {
            let l_slice = &lhs_storage[l_start..l_end];
            let r_slice = &rhs_storage[r_start..r_end];
            l_slice
                .iter()
                .zip(r_slice)
                .map(|(&a, &b)| op(a, b))
                .collect()
        }
        _ => {
            let lhs_iter = StridedIter::new(lhs.layout());
            let rhs_iter = StridedIter::new(rhs.layout());
            lhs_iter
                .zip(rhs_iter)
                .map(|(li, ri)| op(lhs_storage[li], rhs_storage[ri]))
                .collect()
        }
    };

    let bytes = Bytes::from_elems(result);
    EmberTensor::new(bytes, Layout::contiguous(shape), DType::Bool)
}
