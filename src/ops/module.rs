//! Module operations for the Ember backend.
//!
//! These operations power neural network modules like convolutions and pooling.

use crate::Ember;
use burn_backend::{
    ops::{
        ConvOptions, ConvTransposeOptions, DeformConv2dBackward, DeformConvOptions,
        InterpolateOptions, MaxPool2dBackward, MaxPool2dWithIndices, ModuleOps,
    },
    tensor::{FloatTensor, IntTensor},
};

impl ModuleOps<Ember> for Ember {
    fn conv2d(
        _x: FloatTensor<Ember>,
        _weight: FloatTensor<Ember>,
        _bias: Option<FloatTensor<Ember>>,
        _options: ConvOptions<2>,
    ) -> FloatTensor<Ember> {
        todo!("conv2d")
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
        _x: FloatTensor<Ember>,
        _weight: FloatTensor<Ember>,
        _bias: Option<FloatTensor<Ember>>,
        _options: ConvOptions<3>,
    ) -> FloatTensor<Ember> {
        todo!("conv3d")
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
