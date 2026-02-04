//! Module operations for the Ember backend.
//!
//! These operations power neural network modules like convolutions and pooling.

use crate::Ember;
use crate::ops::conv;
use burn_backend::{
    DType,
    ops::{
        ConvOptions, ConvTransposeOptions, DeformConv2dBackward, DeformConvOptions,
        InterpolateOptions, MaxPool2dBackward, MaxPool2dWithIndices, ModuleOps,
    },
    tensor::{FloatTensor, IntTensor},
};

impl ModuleOps<Ember> for Ember {
    fn conv1d(
        x: FloatTensor<Ember>,
        weight: FloatTensor<Ember>,
        bias: Option<FloatTensor<Ember>>,
        options: ConvOptions<1>,
    ) -> FloatTensor<Ember> {
        match x.dtype() {
            DType::F32 => conv::conv1d_f32(x, weight, bias, &options),
            DType::F64 => conv::conv1d_f64(x, weight, bias, &options),
            DType::F16 => conv::conv1d_f16(x, weight, bias, &options),
            DType::BF16 => conv::conv1d_bf16(x, weight, bias, &options),
            dtype => panic!("conv1d: unsupported dtype {:?}", dtype),
        }
    }

    fn conv2d(
        x: FloatTensor<Ember>,
        weight: FloatTensor<Ember>,
        bias: Option<FloatTensor<Ember>>,
        options: ConvOptions<2>,
    ) -> FloatTensor<Ember> {
        match x.dtype() {
            DType::F32 => conv::conv2d_f32(x, weight, bias, &options),
            DType::F64 => conv::conv2d_f64(x, weight, bias, &options),
            DType::F16 => conv::conv2d_f16(x, weight, bias, &options),
            DType::BF16 => conv::conv2d_bf16(x, weight, bias, &options),
            dtype => panic!("conv2d: unsupported dtype {:?}", dtype),
        }
    }

    fn deform_conv2d(
        _x: FloatTensor<Ember>,
        _offset: FloatTensor<Ember>,
        _weight: FloatTensor<Ember>,
        _mask: Option<FloatTensor<Ember>>,
        _bias: Option<FloatTensor<Ember>>,
        _options: DeformConvOptions<2>,
    ) -> FloatTensor<Ember> {
        todo!("deform_conv2d")
    }

    fn deform_conv2d_backward(
        _x: FloatTensor<Ember>,
        _offset: FloatTensor<Ember>,
        _weight: FloatTensor<Ember>,
        _mask: Option<FloatTensor<Ember>>,
        _bias: Option<FloatTensor<Ember>>,
        _output_grad: FloatTensor<Ember>,
        _options: DeformConvOptions<2>,
    ) -> DeformConv2dBackward<Ember> {
        todo!("deform_conv2d_backward")
    }

    fn conv3d(
        x: FloatTensor<Ember>,
        weight: FloatTensor<Ember>,
        bias: Option<FloatTensor<Ember>>,
        options: ConvOptions<3>,
    ) -> FloatTensor<Ember> {
        match x.dtype() {
            DType::F32 => conv::conv3d_f32(x, weight, bias, &options),
            DType::F64 => conv::conv3d_f64(x, weight, bias, &options),
            DType::F16 => conv::conv3d_f16(x, weight, bias, &options),
            DType::BF16 => conv::conv3d_bf16(x, weight, bias, &options),
            dtype => panic!("conv3d: unsupported dtype {:?}", dtype),
        }
    }

    fn conv_transpose2d(
        _x: FloatTensor<Ember>,
        _weight: FloatTensor<Ember>,
        _bias: Option<FloatTensor<Ember>>,
        _options: ConvTransposeOptions<2>,
    ) -> FloatTensor<Ember> {
        todo!("conv_transpose2d")
    }

    fn conv_transpose3d(
        _x: FloatTensor<Ember>,
        _weight: FloatTensor<Ember>,
        _bias: Option<FloatTensor<Ember>>,
        _options: ConvTransposeOptions<3>,
    ) -> FloatTensor<Ember> {
        todo!("conv_transpose3d")
    }

    fn avg_pool2d(
        _x: FloatTensor<Ember>,
        _kernel_size: [usize; 2],
        _stride: [usize; 2],
        _padding: [usize; 2],
        _count_include_pad: bool,
        _divisor_override: bool,
    ) -> FloatTensor<Ember> {
        todo!("avg_pool2d")
    }

    fn avg_pool2d_backward(
        _x: FloatTensor<Ember>,
        _grad: FloatTensor<Ember>,
        _kernel_size: [usize; 2],
        _stride: [usize; 2],
        _padding: [usize; 2],
        _count_include_pad: bool,
        _divisor_override: bool,
    ) -> FloatTensor<Ember> {
        todo!("avg_pool2d_backward")
    }

    fn adaptive_avg_pool2d(_x: FloatTensor<Ember>, _output_size: [usize; 2]) -> FloatTensor<Ember> {
        todo!("adaptive_avg_pool2d")
    }

    fn adaptive_avg_pool2d_backward(
        _x: FloatTensor<Ember>,
        _grad: FloatTensor<Ember>,
    ) -> FloatTensor<Ember> {
        todo!("adaptive_avg_pool2d_backward")
    }

    fn max_pool2d(
        _x: FloatTensor<Ember>,
        _kernel_size: [usize; 2],
        _stride: [usize; 2],
        _padding: [usize; 2],
        _dilation: [usize; 2],
        _ceil_mode: bool,
    ) -> FloatTensor<Ember> {
        todo!("max_pool2d")
    }

    fn max_pool2d_with_indices(
        _x: FloatTensor<Ember>,
        _kernel_size: [usize; 2],
        _stride: [usize; 2],
        _padding: [usize; 2],
        _dilation: [usize; 2],
        _ceil_mode: bool,
    ) -> MaxPool2dWithIndices<Ember> {
        todo!("max_pool2d_with_indices")
    }

    fn max_pool2d_with_indices_backward(
        _x: FloatTensor<Ember>,
        _kernel_size: [usize; 2],
        _stride: [usize; 2],
        _padding: [usize; 2],
        _dilation: [usize; 2],
        _ceil_mode: bool,
        _output_grad: FloatTensor<Ember>,
        _indices: IntTensor<Ember>,
    ) -> MaxPool2dBackward<Ember> {
        todo!("max_pool2d_with_indices_backward")
    }

    fn interpolate(
        _x: FloatTensor<Ember>,
        _output_size: [usize; 2],
        _options: InterpolateOptions,
    ) -> FloatTensor<Ember> {
        todo!("interpolate")
    }

    fn interpolate_backward(
        _x: FloatTensor<Ember>,
        _grad: FloatTensor<Ember>,
        _output_size: [usize; 2],
        _options: InterpolateOptions,
    ) -> FloatTensor<Ember> {
        todo!("interpolate_backward")
    }
}
