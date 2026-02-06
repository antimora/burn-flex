/// Comparison operation type.
#[derive(Clone, Copy)]
pub enum CmpOp {
    Gt,
    Ge,
    Lt,
    Le,
    Eq,
    Ne,
}

/// Scalar comparison for f32 slices, output as u8.
#[inline]
pub fn cmp_f32(a: &[f32], b: &[f32], out: &mut [u8], op: CmpOp) {
    for i in 0..a.len() {
        out[i] = match op {
            CmpOp::Gt => (a[i] > b[i]) as u8,
            CmpOp::Ge => (a[i] >= b[i]) as u8,
            CmpOp::Lt => (a[i] < b[i]) as u8,
            CmpOp::Le => (a[i] <= b[i]) as u8,
            CmpOp::Eq => (a[i] == b[i]) as u8,
            CmpOp::Ne => (a[i] != b[i]) as u8,
        };
    }
}

/// Scalar comparison with scalar value.
#[inline]
pub fn cmp_scalar_f32(a: &[f32], scalar: f32, out: &mut [u8], op: CmpOp) {
    for i in 0..a.len() {
        out[i] = match op {
            CmpOp::Gt => (a[i] > scalar) as u8,
            CmpOp::Ge => (a[i] >= scalar) as u8,
            CmpOp::Lt => (a[i] < scalar) as u8,
            CmpOp::Le => (a[i] <= scalar) as u8,
            CmpOp::Eq => (a[i] == scalar) as u8,
            CmpOp::Ne => (a[i] != scalar) as u8,
        };
    }
}

/// Scalar add for f32 slices.
#[inline]
pub fn add_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
    for i in 0..a.len() {
        out[i] = a[i] + b[i];
    }
}

/// Scalar sub for f32 slices.
#[inline]
pub fn sub_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
    for i in 0..a.len() {
        out[i] = a[i] - b[i];
    }
}

/// Scalar mul for f32 slices.
#[inline]
pub fn mul_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
    for i in 0..a.len() {
        out[i] = a[i] * b[i];
    }
}

/// Scalar div for f32 slices.
#[inline]
pub fn div_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
    for i in 0..a.len() {
        out[i] = a[i] / b[i];
    }
}

/// Scalar add with scalar value.
#[inline]
pub fn add_scalar_f32(a: &[f32], scalar: f32, out: &mut [f32]) {
    for i in 0..a.len() {
        out[i] = a[i] + scalar;
    }
}

/// Scalar mul with scalar value.
#[inline]
pub fn mul_scalar_f32(a: &[f32], scalar: f32, out: &mut [f32]) {
    for i in 0..a.len() {
        out[i] = a[i] * scalar;
    }
}

/// Scalar in-place add.
#[inline]
pub fn add_inplace_f32(a: &mut [f32], b: &[f32]) {
    for i in 0..a.len() {
        a[i] += b[i];
    }
}

/// Scalar in-place sub.
#[inline]
pub fn sub_inplace_f32(a: &mut [f32], b: &[f32]) {
    for i in 0..a.len() {
        a[i] -= b[i];
    }
}

/// Scalar in-place mul.
#[inline]
pub fn mul_inplace_f32(a: &mut [f32], b: &[f32]) {
    for i in 0..a.len() {
        a[i] *= b[i];
    }
}

/// Scalar in-place div.
#[inline]
pub fn div_inplace_f32(a: &mut [f32], b: &[f32]) {
    for i in 0..a.len() {
        a[i] /= b[i];
    }
}

/// Scalar boolean NOT: out[i] = !a[i] (0 becomes 1, non-zero becomes 0)
#[inline]
pub fn bool_not_u8(a: &[u8], out: &mut [u8]) {
    for i in 0..a.len() {
        out[i] = (a[i] == 0) as u8;
    }
}

/// Scalar boolean NOT in-place: a[i] = !a[i]
#[inline]
pub fn bool_not_inplace_u8(a: &mut [u8]) {
    for i in 0..a.len() {
        a[i] = (a[i] == 0) as u8;
    }
}

/// Scalar boolean AND: out[i] = a[i] & b[i]
#[inline]
pub fn bool_and_u8(a: &[u8], b: &[u8], out: &mut [u8]) {
    for i in 0..a.len() {
        out[i] = a[i] & b[i];
    }
}

/// Scalar boolean OR: out[i] = a[i] | b[i]
#[inline]
pub fn bool_or_u8(a: &[u8], b: &[u8], out: &mut [u8]) {
    for i in 0..a.len() {
        out[i] = a[i] | b[i];
    }
}

/// Scalar boolean XOR: out[i] = a[i] ^ b[i]
#[inline]
pub fn bool_xor_u8(a: &[u8], b: &[u8], out: &mut [u8]) {
    for i in 0..a.len() {
        out[i] = a[i] ^ b[i];
    }
}

/// Scalar boolean AND in-place: a[i] &= b[i]
#[inline]
pub fn bool_and_inplace_u8(a: &mut [u8], b: &[u8]) {
    for i in 0..a.len() {
        a[i] &= b[i];
    }
}

/// Scalar boolean OR in-place: a[i] |= b[i]
#[inline]
pub fn bool_or_inplace_u8(a: &mut [u8], b: &[u8]) {
    for i in 0..a.len() {
        a[i] |= b[i];
    }
}

/// Scalar boolean XOR in-place: a[i] ^= b[i]
#[inline]
pub fn bool_xor_inplace_u8(a: &mut [u8], b: &[u8]) {
    for i in 0..a.len() {
        a[i] ^= b[i];
    }
}
