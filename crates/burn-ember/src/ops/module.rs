//! Module operations for the Ember backend.
//!
//! These operations power neural network modules like convolutions and pooling.

use crate::Ember;
use crate::ops::{conv, deform_conv, interpolate, pool};
use burn_backend::{
    DType,
    ops::{
        ConvOptions, ConvTransposeOptions, DeformConv2dBackward, DeformConvOptions,
        InterpolateMode, InterpolateOptions, MaxPool2dBackward, MaxPool2dWithIndices, ModuleOps,
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
        x: FloatTensor<Ember>,
        offset: FloatTensor<Ember>,
        weight: FloatTensor<Ember>,
        mask: Option<FloatTensor<Ember>>,
        bias: Option<FloatTensor<Ember>>,
        options: DeformConvOptions<2>,
    ) -> FloatTensor<Ember> {
        match x.dtype() {
            DType::F32 => deform_conv::deform_conv2d_f32(
                x,
                offset,
                weight,
                mask,
                bias,
                options.stride,
                options.padding,
                options.dilation,
                options.weight_groups,
                options.offset_groups,
            ),
            DType::F64 => deform_conv::deform_conv2d_f64(
                x,
                offset,
                weight,
                mask,
                bias,
                options.stride,
                options.padding,
                options.dilation,
                options.weight_groups,
                options.offset_groups,
            ),
            dtype => panic!("deform_conv2d: unsupported dtype {:?}", dtype),
        }
    }

    fn deform_conv2d_backward(
        x: FloatTensor<Ember>,
        offset: FloatTensor<Ember>,
        weight: FloatTensor<Ember>,
        mask: Option<FloatTensor<Ember>>,
        bias: Option<FloatTensor<Ember>>,
        output_grad: FloatTensor<Ember>,
        options: DeformConvOptions<2>,
    ) -> DeformConv2dBackward<Ember> {
        let (x_grad, offset_grad, weight_grad, mask_grad, bias_grad) = match x.dtype() {
            DType::F32 => deform_conv::deform_conv2d_backward_f32(
                x,
                offset,
                weight,
                mask,
                bias,
                output_grad,
                options.stride,
                options.padding,
                options.dilation,
                options.weight_groups,
                options.offset_groups,
            ),
            dtype => panic!("deform_conv2d_backward: unsupported dtype {:?}", dtype),
        };
        DeformConv2dBackward::new(x_grad, offset_grad, weight_grad, mask_grad, bias_grad)
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

    fn conv_transpose1d(
        x: FloatTensor<Ember>,
        weight: FloatTensor<Ember>,
        bias: Option<FloatTensor<Ember>>,
        options: ConvTransposeOptions<1>,
    ) -> FloatTensor<Ember> {
        match x.dtype() {
            DType::F32 => conv::conv_transpose1d_f32(x, weight, bias, &options),
            DType::F64 => conv::conv_transpose1d_f64(x, weight, bias, &options),
            DType::F16 => conv::conv_transpose1d_f16(x, weight, bias, &options),
            DType::BF16 => conv::conv_transpose1d_bf16(x, weight, bias, &options),
            dtype => panic!("conv_transpose1d: unsupported dtype {:?}", dtype),
        }
    }

    fn conv_transpose2d(
        x: FloatTensor<Ember>,
        weight: FloatTensor<Ember>,
        bias: Option<FloatTensor<Ember>>,
        options: ConvTransposeOptions<2>,
    ) -> FloatTensor<Ember> {
        match x.dtype() {
            DType::F32 => conv::conv_transpose2d_f32(x, weight, bias, &options),
            DType::F64 => conv::conv_transpose2d_f64(x, weight, bias, &options),
            DType::F16 => conv::conv_transpose2d_f16(x, weight, bias, &options),
            DType::BF16 => conv::conv_transpose2d_bf16(x, weight, bias, &options),
            dtype => panic!("conv_transpose2d: unsupported dtype {:?}", dtype),
        }
    }

    fn conv_transpose3d(
        x: FloatTensor<Ember>,
        weight: FloatTensor<Ember>,
        bias: Option<FloatTensor<Ember>>,
        options: ConvTransposeOptions<3>,
    ) -> FloatTensor<Ember> {
        match x.dtype() {
            DType::F32 => conv::conv_transpose3d_f32(x, weight, bias, &options),
            DType::F64 => conv::conv_transpose3d_f64(x, weight, bias, &options),
            DType::F16 => conv::conv_transpose3d_f16(x, weight, bias, &options),
            DType::BF16 => conv::conv_transpose3d_bf16(x, weight, bias, &options),
            dtype => panic!("conv_transpose3d: unsupported dtype {:?}", dtype),
        }
    }

    fn avg_pool2d(
        x: FloatTensor<Ember>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        count_include_pad: bool,
        ceil_mode: bool,
    ) -> FloatTensor<Ember> {
        match x.dtype() {
            DType::F32 => pool::avg_pool2d_f32(
                x,
                kernel_size,
                stride,
                padding,
                count_include_pad,
                ceil_mode,
            ),
            DType::F64 => pool::avg_pool2d_f64(
                x,
                kernel_size,
                stride,
                padding,
                count_include_pad,
                ceil_mode,
            ),
            DType::F16 => pool::avg_pool2d_f16(
                x,
                kernel_size,
                stride,
                padding,
                count_include_pad,
                ceil_mode,
            ),
            DType::BF16 => pool::avg_pool2d_bf16(
                x,
                kernel_size,
                stride,
                padding,
                count_include_pad,
                ceil_mode,
            ),
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
        x: FloatTensor<Ember>,
        output_size: [usize; 2],
        options: InterpolateOptions,
    ) -> FloatTensor<Ember> {
        match (options.mode, x.dtype()) {
            (InterpolateMode::Nearest, DType::F32) => {
                interpolate::interpolate_nearest_f32(x, output_size)
            }
            (InterpolateMode::Nearest, DType::F64) => {
                interpolate::interpolate_nearest_f64(x, output_size)
            }
            (InterpolateMode::Nearest, DType::F16) => {
                interpolate::interpolate_nearest_f16(x, output_size)
            }
            (InterpolateMode::Nearest, DType::BF16) => {
                interpolate::interpolate_nearest_bf16(x, output_size)
            }
            (InterpolateMode::Bilinear, DType::F32) => {
                interpolate::interpolate_bilinear_f32(x, output_size)
            }
            (InterpolateMode::Bilinear, DType::F64) => {
                interpolate::interpolate_bilinear_f64(x, output_size)
            }
            (InterpolateMode::Bilinear, DType::F16) => {
                interpolate::interpolate_bilinear_f16(x, output_size)
            }
            (InterpolateMode::Bilinear, DType::BF16) => {
                interpolate::interpolate_bilinear_bf16(x, output_size)
            }
            (InterpolateMode::Bicubic, DType::F32) => {
                interpolate::interpolate_bicubic_f32(x, output_size)
            }
            (InterpolateMode::Bicubic, DType::F64) => {
                interpolate::interpolate_bicubic_f64(x, output_size)
            }
            (InterpolateMode::Bicubic, DType::F16) => {
                interpolate::interpolate_bicubic_f16(x, output_size)
            }
            (InterpolateMode::Bicubic, DType::BF16) => {
                interpolate::interpolate_bicubic_bf16(x, output_size)
            }
            (mode, dtype) => panic!(
                "interpolate: unsupported mode {:?} / dtype {:?}",
                mode, dtype
            ),
        }
    }

    fn interpolate_backward(
        x: FloatTensor<Ember>,
        grad: FloatTensor<Ember>,
        output_size: [usize; 2],
        options: InterpolateOptions,
    ) -> FloatTensor<Ember> {
        match (options.mode, x.dtype()) {
            (InterpolateMode::Nearest, DType::F32) => {
                interpolate::interpolate_nearest_backward_f32(x, grad, output_size)
            }
            (InterpolateMode::Nearest, DType::F64) => {
                interpolate::interpolate_nearest_backward_f64(x, grad, output_size)
            }
            (InterpolateMode::Nearest, DType::F16) => {
                interpolate::interpolate_nearest_backward_f16(x, grad, output_size)
            }
            (InterpolateMode::Nearest, DType::BF16) => {
                interpolate::interpolate_nearest_backward_bf16(x, grad, output_size)
            }
            (InterpolateMode::Bilinear, DType::F32) => {
                interpolate::interpolate_bilinear_backward_f32(x, grad, output_size)
            }
            (InterpolateMode::Bilinear, DType::F64) => {
                interpolate::interpolate_bilinear_backward_f64(x, grad, output_size)
            }
            (InterpolateMode::Bilinear, DType::F16) => {
                interpolate::interpolate_bilinear_backward_f16(x, grad, output_size)
            }
            (InterpolateMode::Bilinear, DType::BF16) => {
                interpolate::interpolate_bilinear_backward_bf16(x, grad, output_size)
            }
            (InterpolateMode::Bicubic, DType::F32) => {
                interpolate::interpolate_bicubic_backward_f32(x, grad, output_size)
            }
            (InterpolateMode::Bicubic, DType::F64) => {
                interpolate::interpolate_bicubic_backward_f64(x, grad, output_size)
            }
            (InterpolateMode::Bicubic, DType::F16) => {
                interpolate::interpolate_bicubic_backward_f16(x, grad, output_size)
            }
            (InterpolateMode::Bicubic, DType::BF16) => {
                interpolate::interpolate_bicubic_backward_bf16(x, grad, output_size)
            }
            (mode, dtype) => {
                panic!(
                    "interpolate_backward: unsupported mode {:?} / dtype {:?}",
                    mode, dtype
                )
            }
        }
    }
}
