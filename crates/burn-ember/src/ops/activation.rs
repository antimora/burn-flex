//! Activation function operations for the Ember backend.
//!
//! Each activation is implemented as a single-pass unary operation,
//! replacing the default multi-op compositions from Burn's trait defaults.

use burn_backend::Scalar;
use burn_backend::ops::ActivationOps;
use burn_backend::tensor::FloatTensor;
use num_traits::ToPrimitive;

use crate::Ember;
use crate::ops::binary::binary_op;
use crate::ops::unary::unary_op;

impl ActivationOps<Ember> for Ember {
    fn relu(tensor: FloatTensor<Ember>) -> FloatTensor<Ember> {
        unary_op(tensor, |x: f32| x.max(0.0), |x: f64| x.max(0.0))
    }

    fn relu_backward(output: FloatTensor<Ember>, grad: FloatTensor<Ember>) -> FloatTensor<Ember> {
        // grad * (output > 0): zero the gradient where output was zero
        binary_op(
            output,
            grad,
            |out: f32, g| if out > 0.0 { g } else { 0.0 },
            |out: f64, g| if out > 0.0 { g } else { 0.0 },
        )
    }

    fn leaky_relu(tensor: FloatTensor<Ember>, negative_slope: Scalar) -> FloatTensor<Ember> {
        let ns32 = negative_slope.to_f32().unwrap();
        let ns64 = negative_slope.to_f64().unwrap();
        unary_op(
            tensor,
            move |x: f32| if x >= 0.0 { x } else { ns32 * x },
            move |x: f64| if x >= 0.0 { x } else { ns64 * x },
        )
    }

    fn prelu(tensor: FloatTensor<Ember>, alpha: FloatTensor<Ember>) -> FloatTensor<Ember> {
        // x if x >= 0, alpha * x otherwise
        binary_op(
            tensor,
            alpha,
            |x: f32, a| if x >= 0.0 { x } else { a * x },
            |x: f64, a| if x >= 0.0 { x } else { a * x },
        )
    }

    fn gelu(tensor: FloatTensor<Ember>) -> FloatTensor<Ember> {
        // 0.5 * x * (1 + erf(x / sqrt(2)))
        use crate::ops::unary::{erf_f32, erf_f64};
        let sqrt2_f32: f32 = core::f32::consts::SQRT_2;
        let sqrt2_f64: f64 = core::f64::consts::SQRT_2;
        unary_op(
            tensor,
            move |x: f32| 0.5 * x * (1.0 + erf_f32(x / sqrt2_f32)),
            move |x: f64| 0.5 * x * (1.0 + erf_f64(x / sqrt2_f64)),
        )
    }

    fn gelu_backward(x: FloatTensor<Ember>, grad: FloatTensor<Ember>) -> FloatTensor<Ember> {
        // d/dx[gelu(x)] = 0.5 * (1 + erf(x/sqrt(2))) + x * (1/sqrt(2*pi)) * exp(-x^2/2)
        use crate::ops::unary::{erf_f32, erf_f64};
        let sqrt2_f32: f32 = core::f32::consts::SQRT_2;
        let sqrt2_f64: f64 = core::f64::consts::SQRT_2;
        let inv_sqrt_2pi_f32: f32 = 1.0 / (2.0 * core::f32::consts::PI).sqrt();
        let inv_sqrt_2pi_f64: f64 = 1.0 / (2.0 * core::f64::consts::PI).sqrt();
        binary_op(
            x,
            grad,
            move |x: f32, g| {
                let cdf = 0.5 * (1.0 + erf_f32(x / sqrt2_f32));
                let pdf = inv_sqrt_2pi_f32 * (-0.5 * x * x).exp();
                g * (cdf + x * pdf)
            },
            move |x: f64, g| {
                let cdf = 0.5 * (1.0 + erf_f64(x / sqrt2_f64));
                let pdf = inv_sqrt_2pi_f64 * (-0.5 * x * x).exp();
                g * (cdf + x * pdf)
            },
        )
    }

    fn sigmoid(tensor: FloatTensor<Ember>) -> FloatTensor<Ember> {
        unary_op(tensor, sigmoid_f32, sigmoid_f64)
    }

    fn sigmoid_backward(
        output: FloatTensor<Ember>,
        grad: FloatTensor<Ember>,
    ) -> FloatTensor<Ember> {
        // grad * output * (1 - output)
        binary_op(
            output,
            grad,
            |s: f32, g| g * s * (1.0 - s),
            |s: f64, g| g * s * (1.0 - s),
        )
    }

    fn hard_sigmoid(tensor: FloatTensor<Ember>, alpha: Scalar, beta: Scalar) -> FloatTensor<Ember> {
        let alpha32 = alpha.to_f32().unwrap();
        let beta32 = beta.to_f32().unwrap();
        let alpha64 = alpha.to_f64().unwrap();
        let beta64 = beta.to_f64().unwrap();
        unary_op(
            tensor,
            move |x: f32| (alpha32 * x + beta32).clamp(0.0, 1.0),
            move |x: f64| (alpha64 * x + beta64).clamp(0.0, 1.0),
        )
    }

    fn log_sigmoid(tensor: FloatTensor<Ember>) -> FloatTensor<Ember> {
        // Numerically stable: -softplus(-x) = -log(1 + exp(-x))
        // For x >= 0: -log(1 + exp(-x))  (standard form, exp(-x) is small)
        // For x < 0: x - log(1 + exp(x))  (avoids exp of large positive)
        unary_op(
            tensor,
            |x: f32| {
                if x >= 0.0 {
                    -((-x).exp().ln_1p())
                } else {
                    x - x.exp().ln_1p()
                }
            },
            |x: f64| {
                if x >= 0.0 {
                    -((-x).exp().ln_1p())
                } else {
                    x - x.exp().ln_1p()
                }
            },
        )
    }

    fn log_sigmoid_backward(x: FloatTensor<Ember>, grad: FloatTensor<Ember>) -> FloatTensor<Ember> {
        // d/dx[log_sigmoid(x)] = sigmoid(-x) * (-1) * (-1) = 1 - sigmoid(x) = sigmoid(-x)
        // So: grad * sigmoid(-x)
        binary_op(
            x,
            grad,
            |x: f32, g| g * sigmoid_f32(-x),
            |x: f64, g| g * sigmoid_f64(-x),
        )
    }
}

#[inline]
fn sigmoid_f32(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
fn sigmoid_f64(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use burn_backend::Tolerance;
    use burn_tensor::{Tensor, TensorData, activation};

    use crate::Ember;

    #[test]
    fn test_relu() {
        let t: Tensor<Ember, 1> =
            Tensor::from_data([-2.0f32, -1.0, 0.0, 1.0, 2.0], &Default::default());
        activation::relu(t).into_data().assert_approx_eq::<f32>(
            &TensorData::from([0.0, 0.0, 0.0, 1.0, 2.0]),
            Tolerance::absolute(1e-6),
        );
    }

    #[test]
    fn test_sigmoid() {
        let t: Tensor<Ember, 1> = Tensor::from_data([-10.0f32, 0.0, 10.0], &Default::default());
        // sigmoid(-10) ~ 0, sigmoid(0) = 0.5, sigmoid(10) ~ 1
        activation::sigmoid(t).into_data().assert_approx_eq::<f32>(
            &TensorData::from([0.0, 0.5, 1.0]),
            Tolerance::absolute(1e-3),
        );
    }

    #[test]
    fn test_gelu() {
        let t: Tensor<Ember, 1> = Tensor::from_data([-3.0f32, 0.0, 3.0], &Default::default());
        // gelu(0) = 0, gelu(-3) ~ -0.004, gelu(3) ~ 2.996
        activation::gelu(t).into_data().assert_approx_eq::<f32>(
            &TensorData::from([0.0, 0.0, 3.0]),
            Tolerance::absolute(0.01),
        );
    }

    #[test]
    fn test_leaky_relu() {
        let t: Tensor<Ember, 1> =
            Tensor::from_data([-2.0f32, -1.0, 0.0, 1.0, 2.0], &Default::default());
        activation::leaky_relu(t, 0.01)
            .into_data()
            .assert_approx_eq::<f32>(
                &TensorData::from([-0.02, -0.01, 0.0, 1.0, 2.0]),
                Tolerance::absolute(1e-6),
            );
    }

    #[test]
    fn test_log_sigmoid() {
        let t: Tensor<Ember, 1> = Tensor::from_data([-10.0f32, 0.0, 10.0], &Default::default());
        // log_sigmoid(-10) ~ -10, log_sigmoid(0) = ln(0.5) = -0.6931..., log_sigmoid(10) ~ 0
        activation::log_sigmoid(t)
            .into_data()
            .assert_approx_eq::<f32>(
                &TensorData::from([-10.0, -0.6931472, 0.0]),
                Tolerance::absolute(1e-3),
            );
    }
}
