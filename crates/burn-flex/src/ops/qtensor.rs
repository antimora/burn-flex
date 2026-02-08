//! Quantized tensor operations for the Flex backend.

use alloc::vec::Vec;

use burn_backend::{
    DType, ExecutionError, TensorData, TensorMetadata,
    ops::QTensorOps,
    quantization::{
        QuantLevel, QuantScheme, QuantStore, QuantizationParametersPrimitive, QuantizedBytes,
    },
    tensor::{Device, FloatTensor, IntTensor, QuantizedTensor},
};
use burn_std::{Bytes, Shape, Slice};

use crate::{Flex, FlexQTensor, FlexTensor, Layout};

impl QTensorOps<Flex> for Flex {
    fn q_from_data(data: TensorData, _device: &Device<Flex>) -> QuantizedTensor<Flex> {
        let scheme = match data.dtype {
            DType::QFloat(scheme) => scheme,
            _ => panic!("Expected quantized dtype, got {:?}", data.dtype),
        };

        let shape = data.shape.clone();
        let num_elements = data.num_elements();

        let q_bytes = QuantizedBytes {
            bytes: data.into_bytes(),
            scheme,
            num_elements,
        };

        let (values, qparams) = q_bytes.into_vec_i8();
        let tensor_data = TensorData::new(values, shape);
        let tensor = FlexTensor::from_data(tensor_data);

        // Use native storage since we've unpacked to i8
        let scheme = scheme.with_store(QuantStore::Native);

        FlexQTensor {
            tensor,
            scheme,
            scales: qparams.scales,
        }
    }

    fn quantize(
        tensor: FloatTensor<Flex>,
        scheme: &QuantScheme,
        qparams: QuantizationParametersPrimitive<Flex>,
    ) -> QuantizedTensor<Flex> {
        let shape = tensor.shape();
        let tensor = tensor.to_contiguous();
        let float_data: &[f32] = tensor.storage();

        // Extract scales from the qparams tensor
        let scales_tensor = qparams.scales.to_contiguous();
        let scales_data: &[f32] = scales_tensor.storage();
        let scales: Vec<f32> = scales_data.to_vec();

        let (a, b) = scheme.value.range();

        let quantized = match scheme.level {
            QuantLevel::Tensor => {
                let scale = validated_scale(scales[0]);
                float_data
                    .iter()
                    .map(|&x| (x / scale).round().clamp(a, b) as i8)
                    .collect::<Vec<i8>>()
            }
            QuantLevel::Block(block_size) => {
                let block_elems = block_size.num_elements();
                float_data
                    .chunks(block_elems)
                    .zip(scales.iter())
                    .flat_map(|(block, &s)| {
                        let scale = validated_scale(s);
                        block
                            .iter()
                            .map(move |&x| (x / scale).round().clamp(a, b) as i8)
                    })
                    .collect::<Vec<i8>>()
            }
        };

        let bytes = Bytes::from_elems(quantized);
        let layout = Layout::contiguous(shape);
        let qt = FlexTensor::new(bytes, layout, DType::I8);

        // Ensure scales are valid
        let scales = scales.into_iter().map(validated_scale).collect();

        FlexQTensor {
            tensor: qt,
            scheme: scheme.with_store(QuantStore::Native),
            scales,
        }
    }

    fn dequantize(tensor: QuantizedTensor<Flex>) -> FloatTensor<Flex> {
        let shape = tensor.tensor.shape();
        let qt = tensor.tensor.to_contiguous();
        let q_data: &[i8] = qt.storage();

        let dequantized = match tensor.scheme.level {
            QuantLevel::Tensor => {
                let scale = tensor.scales[0];
                q_data
                    .iter()
                    .map(|&x_q| scale * x_q as f32)
                    .collect::<Vec<f32>>()
            }
            QuantLevel::Block(block_size) => {
                let block_elems = block_size.num_elements();
                q_data
                    .chunks(block_elems)
                    .zip(tensor.scales.iter())
                    .flat_map(|(block, &scale)| block.iter().map(move |&x_q| scale * x_q as f32))
                    .collect::<Vec<f32>>()
            }
        };

        let bytes = Bytes::from_elems(dequantized);
        let layout = Layout::contiguous(shape);
        FlexTensor::new(bytes, layout, DType::F32)
    }

    fn q_device(_tensor: &QuantizedTensor<Flex>) -> Device<Flex> {
        Default::default()
    }

    fn q_to_device(tensor: QuantizedTensor<Flex>, _device: &Device<Flex>) -> QuantizedTensor<Flex> {
        tensor
    }

    fn q_reshape(tensor: QuantizedTensor<Flex>, shape: Shape) -> QuantizedTensor<Flex> {
        FlexQTensor {
            tensor: tensor.tensor.reshape(shape),
            scheme: tensor.scheme,
            scales: tensor.scales,
        }
    }

    async fn q_into_data(tensor: QuantizedTensor<Flex>) -> Result<TensorData, ExecutionError> {
        let shape = tensor.tensor.shape();
        let scheme = tensor.scheme;
        let qt = tensor.tensor.to_contiguous();
        let values: Vec<i8> = qt.storage::<i8>().to_vec();

        Ok(TensorData::quantized(
            values,
            shape.dims,
            scheme,
            &tensor.scales,
        ))
    }

    fn q_swap_dims(
        tensor: QuantizedTensor<Flex>,
        dim1: usize,
        dim2: usize,
    ) -> QuantizedTensor<Flex> {
        FlexQTensor {
            tensor: tensor.tensor.transpose(dim1, dim2),
            scheme: tensor.scheme,
            scales: tensor.scales,
        }
    }

    fn q_permute(tensor: QuantizedTensor<Flex>, axes: &[usize]) -> QuantizedTensor<Flex> {
        FlexQTensor {
            tensor: tensor.tensor.permute(axes),
            scheme: tensor.scheme,
            scales: tensor.scales,
        }
    }

    fn q_flip(tensor: QuantizedTensor<Flex>, axes: &[usize]) -> QuantizedTensor<Flex> {
        FlexQTensor {
            tensor: crate::ops::flip::flip(tensor.tensor, axes),
            scheme: tensor.scheme,
            scales: tensor.scales,
        }
    }

    fn q_expand(tensor: QuantizedTensor<Flex>, shape: Shape) -> QuantizedTensor<Flex> {
        FlexQTensor {
            tensor: crate::ops::expand::expand(tensor.tensor, shape),
            scheme: tensor.scheme,
            scales: tensor.scales,
        }
    }

    fn q_select(
        tensor: QuantizedTensor<Flex>,
        dim: usize,
        indices: IntTensor<Flex>,
    ) -> QuantizedTensor<Flex> {
        FlexQTensor {
            tensor: crate::ops::gather_scatter::select::<i8>(tensor.tensor, dim, indices),
            scheme: tensor.scheme,
            scales: tensor.scales,
        }
    }

    fn q_slice(tensor: QuantizedTensor<Flex>, slices: &[Slice]) -> QuantizedTensor<Flex> {
        FlexQTensor {
            tensor: crate::ops::slice::slice(tensor.tensor, slices),
            scheme: tensor.scheme,
            scales: tensor.scales,
        }
    }
}

/// Ensure scale is never zero to avoid division by zero.
fn validated_scale(scale: f32) -> f32 {
    if scale == 0.0 { 0.1 } else { scale }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_backend::{TensorMetadata, quantization::QuantValue};

    #[test]
    fn test_quantize_dequantize_roundtrip() {
        // Create a float tensor
        let values = vec![0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0];
        let tensor = FlexTensor::from_data(TensorData::new(values.clone(), [2, 3]));

        let scheme = QuantScheme::default()
            .with_value(QuantValue::Q8S)
            .with_store(QuantStore::Native);

        // Compute scale: symmetric, so scale = 2 * max(|min|, |max|) / (b - a)
        // max_abs = 5.0, range = 127 - (-127) = 254
        // scale = 2 * 5.0 / 254 = 0.03937008
        let scale: f32 = 2.0 * 5.0 / 254.0;
        let scales_tensor = FlexTensor::from_data(TensorData::new(vec![scale], [1]));

        let qparams = QuantizationParametersPrimitive {
            scales: scales_tensor,
        };

        // Quantize
        let qtensor = Flex::quantize(tensor, &scheme, qparams);
        assert_eq!(qtensor.tensor.shape().dims, vec![2, 3]);
        assert_eq!(qtensor.tensor.dtype(), DType::I8);

        // Check quantized values
        let q_vals: &[i8] = qtensor.tensor.storage();
        // 0 / 0.03937 = 0, 1 / 0.03937 = 25.4 -> 25, etc.
        assert_eq!(q_vals[0], 0);
        assert_eq!(q_vals[1], 25);
        assert_eq!(q_vals[5], 127);

        // Dequantize
        let result = Flex::dequantize(qtensor);
        assert_eq!(result.shape().dims, vec![2, 3]);
        assert_eq!(result.dtype(), DType::F32);

        let result_vals: &[f32] = result.storage();
        // Values should be approximately equal (quantization introduces small errors)
        for (orig, deq) in values.iter().zip(result_vals.iter()) {
            assert!((orig - deq).abs() < 0.05, "orig={orig}, dequantized={deq}");
        }
    }

    #[test]
    fn test_quantize_dequantize_negative_values() {
        let values = vec![-3.0f32, -1.5, 0.0, 1.5, 3.0];
        let tensor = FlexTensor::from_data(TensorData::new(values.clone(), [5]));

        let scheme = QuantScheme::default()
            .with_value(QuantValue::Q8S)
            .with_store(QuantStore::Native);

        let scale: f32 = 2.0 * 3.0 / 254.0;
        let scales_tensor = FlexTensor::from_data(TensorData::new(vec![scale], [1]));

        let qparams = QuantizationParametersPrimitive {
            scales: scales_tensor,
        };

        let qtensor = Flex::quantize(tensor, &scheme, qparams);
        let result = Flex::dequantize(qtensor);
        let result_vals: &[f32] = result.storage();

        for (orig, deq) in values.iter().zip(result_vals.iter()) {
            assert!((orig - deq).abs() < 0.05, "orig={orig}, dequantized={deq}");
        }
    }

    #[test]
    fn test_q_from_data_into_data_roundtrip() {
        // Create quantized TensorData the standard way
        let values = vec![0i8, 25, 51, 76, 102, 127];
        let scale = 0.03937008f32;
        let scheme = QuantScheme::default()
            .with_value(QuantValue::Q8S)
            .with_store(QuantStore::Native);

        let data = TensorData::quantized(values.clone(), [2, 3], scheme, &[scale]);

        // Load into FlexQTensor
        let qtensor = Flex::q_from_data(data, &Default::default());
        assert_eq!(qtensor.tensor.shape().dims, vec![2, 3]);
        assert_eq!(qtensor.scales, vec![scale]);

        // Dequantize and check values
        let float_tensor = Flex::dequantize(qtensor);
        let result: &[f32] = float_tensor.storage();
        assert!((result[0]).abs() < 0.01); // 0 * scale ~ 0
        assert!((result[5] - 5.0).abs() < 0.05); // 127 * scale ~ 5.0
    }

    #[test]
    fn test_quantize_zero_tensor() {
        let values = vec![0.0f32; 4];
        let tensor = FlexTensor::from_data(TensorData::new(values, [4]));

        let scheme = QuantScheme::default()
            .with_value(QuantValue::Q8S)
            .with_store(QuantStore::Native);

        // Scale of 0 should be handled gracefully
        let scales_tensor = FlexTensor::from_data(TensorData::new(vec![0.0f32], [1]));
        let qparams = QuantizationParametersPrimitive {
            scales: scales_tensor,
        };

        let qtensor = Flex::quantize(tensor, &scheme, qparams);
        let q_vals: &[i8] = qtensor.tensor.storage();
        assert_eq!(q_vals, &[0, 0, 0, 0]);
    }

    #[test]
    fn test_q_layout_ops_preserve_scheme() {
        let values = vec![0i8, 25, 51, 76, 102, 127];
        let scale = 0.03937008f32;
        let scheme = QuantScheme::default()
            .with_value(QuantValue::Q8S)
            .with_store(QuantStore::Native);
        let data = TensorData::quantized(values, [2, 3], scheme, &[scale]);
        let qtensor = Flex::q_from_data(data, &Default::default());

        // Reshape
        let reshaped = Flex::q_reshape(qtensor.clone(), Shape::from(vec![3, 2]));
        assert_eq!(reshaped.tensor.shape().dims, vec![3, 2]);
        assert_eq!(reshaped.scales, vec![scale]);

        // Transpose
        let transposed = Flex::q_swap_dims(qtensor.clone(), 0, 1);
        assert_eq!(transposed.tensor.shape().dims, vec![3, 2]);

        // Permute
        let permuted = Flex::q_permute(qtensor.clone(), &[1, 0]);
        assert_eq!(permuted.tensor.shape().dims, vec![3, 2]);

        // Flip
        let flipped = Flex::q_flip(qtensor.clone(), &[0]);
        assert_eq!(flipped.tensor.shape().dims, vec![2, 3]);

        // Expand
        let to_expand = Flex::q_reshape(qtensor.clone(), Shape::from(vec![1, 2, 3]));
        let expanded = Flex::q_expand(to_expand, Shape::from(vec![4, 2, 3]));
        assert_eq!(expanded.tensor.shape().dims, vec![4, 2, 3]);
    }

    #[test]
    fn test_q_select() {
        let values = vec![10i8, 20, 30, 40, 50, 60];
        let scale = 0.1f32;
        let scheme = QuantScheme::default()
            .with_value(QuantValue::Q8S)
            .with_store(QuantStore::Native);
        let data = TensorData::quantized(values, [2, 3], scheme, &[scale]);
        let qtensor = Flex::q_from_data(data, &Default::default());

        // Select row 1
        let indices = FlexTensor::from_data(TensorData::new(vec![1i64], [1]));
        let selected = Flex::q_select(qtensor, 0, indices);
        assert_eq!(selected.tensor.shape().dims, vec![1, 3]);

        let selected_contiguous = selected.tensor.to_contiguous();
        let selected_vals: &[i8] = selected_contiguous.storage();
        assert_eq!(selected_vals, &[40, 50, 60]);
    }

    #[test]
    fn test_q_slice() {
        let values = vec![10i8, 20, 30, 40, 50, 60];
        let scale = 0.1f32;
        let scheme = QuantScheme::default()
            .with_value(QuantValue::Q8S)
            .with_store(QuantStore::Native);
        let data = TensorData::quantized(values, [2, 3], scheme, &[scale]);
        let qtensor = Flex::q_from_data(data, &Default::default());

        // Slice [0:1, 1:3] -> [[20, 30]]
        let slices = vec![Slice::new(0, Some(1), 1), Slice::new(1, Some(3), 1)];
        let sliced = Flex::q_slice(qtensor, &slices);
        assert_eq!(sliced.tensor.shape().dims, vec![1, 2]);
    }

    #[test]
    fn test_per_block_quantize_dequantize() {
        use burn_std::quantization::BlockSize;

        let values = vec![0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let tensor = FlexTensor::from_data(TensorData::new(values.clone(), [8]));

        let block_size = BlockSize::new([4]);
        let scheme = QuantScheme::default()
            .with_value(QuantValue::Q8S)
            .with_level(QuantLevel::Block(block_size))
            .with_store(QuantStore::Native);

        // Block 1: [0, 1, 2, 3] -> max_abs=3, scale = 6/254
        // Block 2: [4, 5, 6, 7] -> max_abs=7, scale = 14/254
        let scale_1: f32 = 2.0 * 3.0 / 254.0;
        let scale_2: f32 = 2.0 * 7.0 / 254.0;
        let scales_tensor = FlexTensor::from_data(TensorData::new(vec![scale_1, scale_2], [2]));

        let qparams = QuantizationParametersPrimitive {
            scales: scales_tensor,
        };

        let qtensor = Flex::quantize(tensor, &scheme, qparams);
        assert_eq!(qtensor.scales.len(), 2);

        let result = Flex::dequantize(qtensor);
        let result_vals: &[f32] = result.storage();

        for (orig, deq) in values.iter().zip(result_vals.iter()) {
            assert!((orig - deq).abs() < 0.1, "orig={orig}, dequantized={deq}");
        }
    }
}
