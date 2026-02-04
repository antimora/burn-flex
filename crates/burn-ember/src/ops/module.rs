//! Module operations for the Ember backend.
//!
//! These operations power neural network modules like convolutions and pooling.

use crate::Ember;
use crate::ops::{conv, pool};
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
        x: FloatTensor<Ember>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        count_include_pad: bool,
        _divisor_override: bool,
    ) -> FloatTensor<Ember> {
        match x.dtype() {
            DType::F32 => {
                pool::avg_pool2d_f32(x, kernel_size, stride, padding, count_include_pad, false)
            }
            DType::F64 => {
                pool::avg_pool2d_f64(x, kernel_size, stride, padding, count_include_pad, false)
            }
            DType::F16 => {
                pool::avg_pool2d_f16(x, kernel_size, stride, padding, count_include_pad, false)
            }
            DType::BF16 => {
                pool::avg_pool2d_bf16(x, kernel_size, stride, padding, count_include_pad, false)
            }
            dtype => panic!("avg_pool2d: unsupported dtype {:?}", dtype),
        }
    }

    fn avg_pool2d_backward(
        x: FloatTensor<Ember>,
        grad: FloatTensor<Ember>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        count_include_pad: bool,
        _divisor_override: bool,
    ) -> FloatTensor<Ember> {
        match x.dtype() {
            DType::F32 => pool::avg_pool2d_backward_f32(
                x,
                grad,
                kernel_size,
                stride,
                padding,
                count_include_pad,
            ),
            DType::F64 => pool::avg_pool2d_backward_f64(
                x,
                grad,
                kernel_size,
                stride,
                padding,
                count_include_pad,
            ),
            DType::F16 => pool::avg_pool2d_backward_f16(
                x,
                grad,
                kernel_size,
                stride,
                padding,
                count_include_pad,
            ),
            DType::BF16 => pool::avg_pool2d_backward_bf16(
                x,
                grad,
                kernel_size,
                stride,
                padding,
                count_include_pad,
            ),
            dtype => panic!("avg_pool2d_backward: unsupported dtype {:?}", dtype),
        }
    }

    fn adaptive_avg_pool2d(x: FloatTensor<Ember>, output_size: [usize; 2]) -> FloatTensor<Ember> {
        match x.dtype() {
            DType::F32 => pool::adaptive_avg_pool2d_f32(x, output_size),
            DType::F64 => pool::adaptive_avg_pool2d_f64(x, output_size),
            DType::F16 => pool::adaptive_avg_pool2d_f16(x, output_size),
            DType::BF16 => pool::adaptive_avg_pool2d_bf16(x, output_size),
            dtype => panic!("adaptive_avg_pool2d: unsupported dtype {:?}", dtype),
        }
    }

    fn adaptive_avg_pool2d_backward(
        x: FloatTensor<Ember>,
        grad: FloatTensor<Ember>,
    ) -> FloatTensor<Ember> {
        match x.dtype() {
            DType::F32 => pool::adaptive_avg_pool2d_backward_f32(x, grad),
            DType::F64 => pool::adaptive_avg_pool2d_backward_f64(x, grad),
            DType::F16 => pool::adaptive_avg_pool2d_backward_f16(x, grad),
            DType::BF16 => pool::adaptive_avg_pool2d_backward_bf16(x, grad),
            dtype => panic!(
                "adaptive_avg_pool2d_backward: unsupported dtype {:?}",
                dtype
            ),
        }
    }

    fn max_pool2d(
        x: FloatTensor<Ember>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        ceil_mode: bool,
    ) -> FloatTensor<Ember> {
        match x.dtype() {
            DType::F32 => {
                pool::max_pool2d_f32(x, kernel_size, stride, padding, dilation, ceil_mode)
            }
            DType::F64 => {
                pool::max_pool2d_f64(x, kernel_size, stride, padding, dilation, ceil_mode)
            }
            DType::F16 => {
                pool::max_pool2d_f16(x, kernel_size, stride, padding, dilation, ceil_mode)
            }
            DType::BF16 => {
                pool::max_pool2d_bf16(x, kernel_size, stride, padding, dilation, ceil_mode)
            }
            dtype => panic!("max_pool2d: unsupported dtype {:?}", dtype),
        }
    }

    fn max_pool2d_with_indices(
        x: FloatTensor<Ember>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        ceil_mode: bool,
    ) -> MaxPool2dWithIndices<Ember> {
        let (output, indices) = match x.dtype() {
            DType::F32 => pool::max_pool2d_with_indices_f32(
                x,
                kernel_size,
                stride,
                padding,
                dilation,
                ceil_mode,
            ),
            DType::F64 => pool::max_pool2d_with_indices_f64(
                x,
                kernel_size,
                stride,
                padding,
                dilation,
                ceil_mode,
            ),
            DType::F16 => pool::max_pool2d_with_indices_f16(
                x,
                kernel_size,
                stride,
                padding,
                dilation,
                ceil_mode,
            ),
            DType::BF16 => pool::max_pool2d_with_indices_bf16(
                x,
                kernel_size,
                stride,
                padding,
                dilation,
                ceil_mode,
            ),
            dtype => panic!("max_pool2d_with_indices: unsupported dtype {:?}", dtype),
        };
        MaxPool2dWithIndices::new(output, indices)
    }

    fn max_pool2d_with_indices_backward(
        x: FloatTensor<Ember>,
        _kernel_size: [usize; 2],
        _stride: [usize; 2],
        _padding: [usize; 2],
        _dilation: [usize; 2],
        _ceil_mode: bool,
        output_grad: FloatTensor<Ember>,
        indices: IntTensor<Ember>,
    ) -> MaxPool2dBackward<Ember> {
        let x_grad = match x.dtype() {
            DType::F32 => pool::max_pool2d_backward_f32(x, output_grad, indices),
            DType::F64 => pool::max_pool2d_backward_f64(x, output_grad, indices),
            DType::F16 => pool::max_pool2d_backward_f16(x, output_grad, indices),
            DType::BF16 => pool::max_pool2d_backward_bf16(x, output_grad, indices),
            dtype => panic!(
                "max_pool2d_with_indices_backward: unsupported dtype {:?}",
                dtype
            ),
        };
        MaxPool2dBackward::new(x_grad)
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
