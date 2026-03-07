use super::*;
use std::f32::consts::{FRAC_PI_2, FRAC_PI_3, FRAC_PI_4, FRAC_PI_6, PI};

fn ulp_error(computed: f32, expected: f32) -> f32 {
    if computed.is_nan() && expected.is_nan() {
        return 0.0;
    }
    if computed == expected {
        return 0.0;
    }
    if expected.is_infinite() || computed.is_nan() || expected.is_nan() {
        return f32::MAX;
    }
    // Compute ULP as the distance between adjacent floats at expected
    let bits = expected.to_bits() as i32;
    let next = f32::from_bits((bits + 1) as u32);
    let ulp = (next - expected).abs();
    if ulp == 0.0 {
        return f32::MAX;
    }
    (computed - expected).abs() / ulp
}

fn assert_ulp(computed: f32, expected: f32, max_ulp: f32, msg: &str) {
    let err = ulp_error(computed, expected);
    assert!(
        err <= max_ulp,
        "{msg}: computed={computed}, expected={expected}, ulp_error={err} > {max_ulp}"
    );
}

// ============ Vec4 tests ============

#[test]
fn test_vec4_splat() {
    let v = Vec4::splat(3.5);
    assert_eq!(v.0, [3.5; 4]);
}

#[test]
fn test_vec4_zero_one() {
    assert_eq!(Vec4::ZERO.0, [0.0; 4]);
    assert_eq!(Vec4::ONE.0, [1.0; 4]);
}

#[test]
fn test_vec4_add() {
    let a = Vec4([1.0, 2.0, 3.0, 4.0]);
    let b = Vec4([5.0, 6.0, 7.0, 8.0]);
    let c = a + b;
    assert_eq!(c.0, [6.0, 8.0, 10.0, 12.0]);
}

#[test]
fn test_vec4_sub() {
    let a = Vec4([5.0, 6.0, 7.0, 8.0]);
    let b = Vec4([1.0, 2.0, 3.0, 4.0]);
    let c = a - b;
    assert_eq!(c.0, [4.0, 4.0, 4.0, 4.0]);
}

#[test]
fn test_vec4_mul() {
    let a = Vec4([1.0, 2.0, 3.0, 4.0]);
    let b = Vec4([2.0, 3.0, 4.0, 5.0]);
    let c = a * b;
    assert_eq!(c.0, [2.0, 6.0, 12.0, 20.0]);
}

#[test]
fn test_vec4_div() {
    let a = Vec4([10.0, 20.0, 30.0, 40.0]);
    let b = Vec4([2.0, 4.0, 5.0, 8.0]);
    let c = a / b;
    assert_eq!(c.0, [5.0, 5.0, 6.0, 5.0]);
}

#[test]
fn test_vec4_neg() {
    let a = Vec4([1.0, -2.0, 3.0, -4.0]);
    let b = -a;
    assert_eq!(b.0, [-1.0, 2.0, -3.0, 4.0]);
}

#[test]
fn test_vec4_abs() {
    let a = Vec4([-1.0, 2.0, -3.0, 4.0]);
    assert_eq!(a.abs().0, [1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_vec4_sqrt() {
    let a = Vec4([4.0, 9.0, 16.0, 25.0]);
    assert_eq!(a.sqrt().0, [2.0, 3.0, 4.0, 5.0]);
}

#[test]
fn test_vec4_floor() {
    let a = Vec4([1.5, 2.7, -1.3, -2.9]);
    assert_eq!(a.floor().0, [1.0, 2.0, -2.0, -3.0]);
}

#[test]
fn test_vec4_dot() {
    let a = Vec4([1.0, 2.0, 3.0, 4.0]);
    let b = Vec4([5.0, 6.0, 7.0, 8.0]);
    assert_eq!(a.dot(b), 70.0); // 5+12+21+32
}

#[test]
fn test_vec4_mul_add() {
    let a = Vec4([1.0, 2.0, 3.0, 4.0]);
    let b = Vec4([2.0, 3.0, 4.0, 5.0]);
    let c = Vec4([10.0, 20.0, 30.0, 40.0]);
    let r = a.mul_add(b, c);
    assert_eq!(r.0, [12.0, 26.0, 42.0, 60.0]);
}

#[test]
fn test_vec4_add_scalar() {
    let a = Vec4([1.0, 2.0, 3.0, 4.0]);
    let r = a + 10.0;
    assert_eq!(r.0, [11.0, 12.0, 13.0, 14.0]);
}

#[test]
fn test_vec4_sub_scalar() {
    let a = Vec4([10.0, 20.0, 30.0, 40.0]);
    let r = a - 5.0;
    assert_eq!(r.0, [5.0, 15.0, 25.0, 35.0]);
}

#[test]
fn test_vec4_mul_scalar() {
    let a = Vec4([1.0, 2.0, 3.0, 4.0]);
    let r = a * 3.0;
    assert_eq!(r.0, [3.0, 6.0, 9.0, 12.0]);
}

#[test]
fn test_vec4_div_scalar() {
    let a = Vec4([10.0, 20.0, 30.0, 40.0]);
    let r = a / 10.0;
    assert_eq!(r.0, [1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_f32_add_vec4() {
    let a = Vec4([1.0, 2.0, 3.0, 4.0]);
    let r = 10.0 + a;
    assert_eq!(r.0, [11.0, 12.0, 13.0, 14.0]);
}

#[test]
fn test_f32_sub_vec4() {
    let a = Vec4([1.0, 2.0, 3.0, 4.0]);
    let r = 10.0 - a;
    assert_eq!(r.0, [9.0, 8.0, 7.0, 6.0]);
}

#[test]
fn test_f32_mul_vec4() {
    let a = Vec4([1.0, 2.0, 3.0, 4.0]);
    let r = 3.0 * a;
    assert_eq!(r.0, [3.0, 6.0, 9.0, 12.0]);
}

#[test]
fn test_f32_div_vec4() {
    let a = Vec4([1.0, 2.0, 4.0, 5.0]);
    let r = 20.0 / a;
    assert_eq!(r.0, [20.0, 10.0, 5.0, 4.0]);
}

#[test]
fn test_vec4_index() {
    let v = Vec4([10.0, 20.0, 30.0, 40.0]);
    assert_eq!(v[0], 10.0);
    assert_eq!(v[1], 20.0);
    assert_eq!(v[2], 30.0);
    assert_eq!(v[3], 40.0);
}

#[test]
#[should_panic]
fn test_vec4_index_oob() {
    let v = Vec4([1.0; 4]);
    let _ = v[4];
}

#[test]
fn test_vec4_sum() {
    let vecs = vec![
        Vec4([1.0, 2.0, 3.0, 4.0]),
        Vec4([5.0, 6.0, 7.0, 8.0]),
        Vec4([9.0, 10.0, 11.0, 12.0]),
    ];
    let s: Vec4 = vecs.into_iter().sum();
    assert_eq!(s.0, [15.0, 18.0, 21.0, 24.0]);
}

#[test]
fn test_vec4_sum_empty() {
    let vecs: Vec<Vec4> = vec![];
    let s: Vec4 = vecs.into_iter().sum();
    assert_eq!(s, Vec4::ZERO);
}

#[test]
fn test_vec4_from_array() {
    let arr = [1.0f32, 2.0, 3.0, 4.0];
    let v: Vec4 = arr.into();
    assert_eq!(v.0, arr);
}

#[test]
fn test_vec4_into_array() {
    let v = Vec4([1.0, 2.0, 3.0, 4.0]);
    let arr: [f32; 4] = v.into();
    assert_eq!(arr, [1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_vec4_debug() {
    let v = Vec4([1.0, 2.0, 3.0, 4.0]);
    let s = format!("{:?}", v);
    assert!(s.contains("Vec4"));
}

#[test]
fn test_vec4_clone_copy() {
    let a = Vec4([1.0, 2.0, 3.0, 4.0]);
    let b = a;
    let c = a.clone();
    assert_eq!(a, b);
    assert_eq!(a, c);
}

// ============ Vec4 transcendental tests ============

#[test]
fn test_vec4_sin_basic() {
    let vals = [0.0, FRAC_PI_6, FRAC_PI_4, FRAC_PI_2];
    let v = Vec4(vals);
    let r = v.sin();
    for i in 0..4 {
        assert_ulp(r[i], vals[i].sin(), 1.0, &format!("sin({})", vals[i]));
    }
}

#[test]
fn test_vec4_sin_more() {
    let vals = [PI, -FRAC_PI_4, -PI, 3.0 * FRAC_PI_2];
    let v = Vec4(vals);
    let r = v.sin();
    for i in 0..4 {
        assert_ulp(r[i], vals[i].sin(), 1.0, &format!("sin({})", vals[i]));
    }
}

#[test]
fn test_vec4_sin_neg_zero() {
    let v = Vec4([-0.0, 0.0, -0.0, 0.0]);
    let r = v.sin();
    assert!(r[0].is_sign_negative() && r[0] == 0.0);
    assert!(r[1].is_sign_positive() && r[1] == 0.0);
}

#[test]
fn test_vec4_sin_nan_inf() {
    let v = Vec4([f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.0]);
    let r = v.sin();
    assert!(r[0].is_nan());
    assert!(r[1].is_nan());
    assert!(r[2].is_nan());
    assert_eq!(r[3], 0.0);
}

#[test]
fn test_vec4_sin_large() {
    let vals = [100.0, -100.0, 1000.0, -1000.0];
    let v = Vec4(vals);
    let r = v.sin();
    for i in 0..4 {
        assert_ulp(r[i], vals[i].sin(), 1.0, &format!("sin({})", vals[i]));
    }
}

#[test]
fn test_vec4_cos_basic() {
    let vals = [0.0, FRAC_PI_6, FRAC_PI_4, FRAC_PI_2];
    let v = Vec4(vals);
    let r = v.cos();
    for i in 0..4 {
        assert_ulp(r[i], vals[i].cos(), 1.0, &format!("cos({})", vals[i]));
    }
}

#[test]
fn test_vec4_cos_more() {
    let vals = [PI, -FRAC_PI_4, -PI, 3.0 * FRAC_PI_2];
    let v = Vec4(vals);
    let r = v.cos();
    for i in 0..4 {
        assert_ulp(r[i], vals[i].cos(), 1.0, &format!("cos({})", vals[i]));
    }
}

#[test]
fn test_vec4_cos_nan_inf() {
    let v = Vec4([f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.0]);
    let r = v.cos();
    assert!(r[0].is_nan());
    assert!(r[1].is_nan());
    assert!(r[2].is_nan());
    assert_eq!(r[3], 1.0);
}

#[test]
fn test_vec4_cos_large() {
    let vals = [100.0, -100.0, 1000.0, -1000.0];
    let v = Vec4(vals);
    let r = v.cos();
    for i in 0..4 {
        assert_ulp(r[i], vals[i].cos(), 1.0, &format!("cos({})", vals[i]));
    }
}

#[test]
fn test_vec4_exp_basic() {
    let vals = [0.0, 1.0, -1.0, 2.0];
    let v = Vec4(vals);
    let r = v.exp();
    for i in 0..4 {
        assert_ulp(r[i], vals[i].exp(), 1.0, &format!("exp({})", vals[i]));
    }
}

#[test]
fn test_vec4_exp_edge() {
    let v = Vec4([0.0, -200.0, 200.0, f32::NAN]);
    let r = v.exp();
    assert_eq!(r[0], 1.0);
    assert_eq!(r[1], 0.0);
    assert_eq!(r[2], f32::INFINITY);
    assert!(r[3].is_nan());
}

#[test]
fn test_vec4_exp_ln2() {
    let v = Vec4([std::f32::consts::LN_2, -std::f32::consts::LN_2, 0.5, -0.5]);
    let r = v.exp();
    for i in 0..4 {
        assert_ulp(r[i], v[i].exp(), 1.0, &format!("exp({})", v[i]));
    }
}

// ============ Vec8 tests ============

#[test]
fn test_vec8_splat() {
    let v = Vec8::splat(3.5);
    assert_eq!(v.0, [3.5; 8]);
}

#[test]
fn test_vec8_zero_one() {
    assert_eq!(Vec8::ZERO.0, [0.0; 8]);
    assert_eq!(Vec8::ONE.0, [1.0; 8]);
}

#[test]
fn test_vec8_add() {
    let a = Vec8([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let b = Vec8([8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]);
    let c = a + b;
    assert_eq!(c.0, [9.0; 8]);
}

#[test]
fn test_vec8_sub() {
    let a = Vec8([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]);
    let b = Vec8([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let c = a - b;
    assert_eq!(c.0, [9.0, 18.0, 27.0, 36.0, 45.0, 54.0, 63.0, 72.0]);
}

#[test]
fn test_vec8_mul() {
    let a = Vec8([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let b = Vec8([2.0; 8]);
    let c = a * b;
    assert_eq!(c.0, [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]);
}

#[test]
fn test_vec8_div() {
    let a = Vec8([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]);
    let b = Vec8([10.0; 8]);
    let c = a / b;
    assert_eq!(c.0, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
}

#[test]
fn test_vec8_neg() {
    let a = Vec8([1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0]);
    let b = -a;
    assert_eq!(b.0, [-1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0]);
}

#[test]
fn test_vec8_abs() {
    let a = Vec8([-1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0]);
    assert_eq!(a.abs().0, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
}

#[test]
fn test_vec8_sqrt() {
    let a = Vec8([1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0]);
    assert_eq!(a.sqrt().0, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
}

#[test]
fn test_vec8_floor() {
    let a = Vec8([1.5, 2.7, -1.3, -2.9, 0.0, 3.99, -0.01, 100.1]);
    assert_eq!(
        a.floor().0,
        [1.0, 2.0, -2.0, -3.0, 0.0, 3.0, -1.0, 100.0]
    );
}

#[test]
fn test_vec8_dot() {
    let a = Vec8([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let b = Vec8([1.0; 8]);
    assert_eq!(a.dot(b), 36.0);
}

#[test]
fn test_vec8_mul_add() {
    let a = Vec8([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let b = Vec8([2.0; 8]);
    let c = Vec8([10.0; 8]);
    let r = a.mul_add(b, c);
    assert_eq!(r.0, [12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0]);
}

#[test]
fn test_vec8_add_scalar() {
    let a = Vec8([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let r = a + 100.0;
    assert_eq!(r.0, [101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0]);
}

#[test]
fn test_vec8_sub_scalar() {
    let a = Vec8([10.0; 8]);
    let r = a - 3.0;
    assert_eq!(r.0, [7.0; 8]);
}

#[test]
fn test_vec8_mul_scalar() {
    let a = Vec8([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let r = a * 0.5;
    assert_eq!(r.0, [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]);
}

#[test]
fn test_vec8_div_scalar() {
    let a = Vec8([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]);
    let r = a / 10.0;
    assert_eq!(r.0, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
}

#[test]
fn test_f32_add_vec8() {
    let a = Vec8([1.0; 8]);
    let r = 10.0 + a;
    assert_eq!(r.0, [11.0; 8]);
}

#[test]
fn test_f32_sub_vec8() {
    let a = Vec8([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let r = 10.0 - a;
    assert_eq!(r.0, [9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0]);
}

#[test]
fn test_f32_mul_vec8() {
    let a = Vec8([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let r = 3.0 * a;
    assert_eq!(r.0, [3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0]);
}

#[test]
fn test_f32_div_vec8() {
    let a = Vec8([1.0, 2.0, 4.0, 5.0, 10.0, 20.0, 25.0, 50.0]);
    let r = 100.0 / a;
    assert_eq!(r.0, [100.0, 50.0, 25.0, 20.0, 10.0, 5.0, 4.0, 2.0]);
}

#[test]
fn test_vec8_index() {
    let v = Vec8([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]);
    for i in 0..8 {
        assert_eq!(v[i], (i as f32 + 1.0) * 10.0);
    }
}

#[test]
#[should_panic]
fn test_vec8_index_oob() {
    let v = Vec8([1.0; 8]);
    let _ = v[8];
}

#[test]
fn test_vec8_sum() {
    let vecs = vec![
        Vec8([1.0; 8]),
        Vec8([2.0; 8]),
        Vec8([3.0; 8]),
    ];
    let s: Vec8 = vecs.into_iter().sum();
    assert_eq!(s.0, [6.0; 8]);
}

#[test]
fn test_vec8_sum_empty() {
    let vecs: Vec<Vec8> = vec![];
    let s: Vec8 = vecs.into_iter().sum();
    assert_eq!(s, Vec8::ZERO);
}

#[test]
fn test_vec8_from_array() {
    let arr = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let v: Vec8 = arr.into();
    assert_eq!(v.0, arr);
}

#[test]
fn test_vec8_into_array() {
    let v = Vec8([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let arr: [f32; 8] = v.into();
    assert_eq!(arr, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
}

#[test]
fn test_vec8_debug() {
    let v = Vec8([1.0; 8]);
    let s = format!("{:?}", v);
    assert!(s.contains("Vec8"));
}

#[test]
fn test_vec8_clone_copy() {
    let a = Vec8([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let b = a;
    let c = a.clone();
    assert_eq!(a, b);
    assert_eq!(a, c);
}

// ============ Vec8 transcendental tests ============

#[test]
fn test_vec8_sin_basic() {
    let vals = [0.0, FRAC_PI_6, FRAC_PI_4, FRAC_PI_3, FRAC_PI_2, PI, -FRAC_PI_4, -PI];
    let v = Vec8(vals);
    let r = v.sin();
    for i in 0..8 {
        assert_ulp(r[i], vals[i].sin(), 1.0, &format!("sin({})", vals[i]));
    }
}

#[test]
fn test_vec8_sin_neg_zero() {
    let mut vals = [0.0f32; 8];
    vals[0] = -0.0;
    let r = Vec8(vals).sin();
    assert!(r[0].is_sign_negative() && r[0] == 0.0);
}

#[test]
fn test_vec8_sin_nan_inf() {
    let v = Vec8([
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        0.0,
        1.0,
        -1.0,
        2.0,
        -2.0,
    ]);
    let r = v.sin();
    assert!(r[0].is_nan());
    assert!(r[1].is_nan());
    assert!(r[2].is_nan());
    for i in 3..8 {
        assert_ulp(r[i], v[i].sin(), 1.0, &format!("sin({})", v[i]));
    }
}

#[test]
fn test_vec8_sin_large() {
    let vals = [100.0, -100.0, 1000.0, -1000.0, 50.0, -50.0, 200.0, -200.0];
    let v = Vec8(vals);
    let r = v.sin();
    for i in 0..8 {
        assert_ulp(r[i], vals[i].sin(), 1.0, &format!("sin({})", vals[i]));
    }
}

#[test]
fn test_vec8_cos_basic() {
    let vals = [0.0, FRAC_PI_6, FRAC_PI_4, FRAC_PI_3, FRAC_PI_2, PI, -FRAC_PI_4, -PI];
    let v = Vec8(vals);
    let r = v.cos();
    for i in 0..8 {
        assert_ulp(r[i], vals[i].cos(), 1.0, &format!("cos({})", vals[i]));
    }
}

#[test]
fn test_vec8_cos_nan_inf() {
    let v = Vec8([
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        0.0,
        1.0,
        -1.0,
        2.0,
        -2.0,
    ]);
    let r = v.cos();
    assert!(r[0].is_nan());
    assert!(r[1].is_nan());
    assert!(r[2].is_nan());
    for i in 3..8 {
        assert_ulp(r[i], v[i].cos(), 1.0, &format!("cos({})", v[i]));
    }
}

#[test]
fn test_vec8_cos_large() {
    let vals = [100.0, -100.0, 1000.0, -1000.0, 50.0, -50.0, 200.0, -200.0];
    let v = Vec8(vals);
    let r = v.cos();
    for i in 0..8 {
        assert_ulp(r[i], vals[i].cos(), 1.0, &format!("cos({})", vals[i]));
    }
}

#[test]
fn test_vec8_exp_basic() {
    let vals = [0.0, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 10.0];
    let v = Vec8(vals);
    let r = v.exp();
    for i in 0..8 {
        assert_ulp(r[i], vals[i].exp(), 1.0, &format!("exp({})", vals[i]));
    }
}

#[test]
fn test_vec8_exp_edge() {
    let v = Vec8([0.0, -200.0, 200.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 88.0, -88.0]);
    let r = v.exp();
    assert_eq!(r[0], 1.0);
    assert_eq!(r[1], 0.0);
    assert_eq!(r[2], f32::INFINITY);
    assert!(r[3].is_nan());
    assert_eq!(r[4], f32::INFINITY);
    assert_eq!(r[5], 0.0);
    assert_ulp(r[6], 88.0f32.exp(), 1.0, "exp(88)");
    assert_ulp(r[7], (-88.0f32).exp(), 1.0, "exp(-88)");
}

// ============ Exhaustive ULP sweep tests ============

#[test]
fn test_vec4_sin_sweep() {
    let mut max_ulp = 0.0f32;
    let step = 0.001;
    let mut x = -125.0f32;
    while x < 125.0 {
        let v = Vec4([x, x + step, x + 2.0 * step, x + 3.0 * step]);
        let r = v.sin();
        for i in 0..4 {
            let expected = v[i].sin();
            let u = ulp_error(r[i], expected);
            if u > max_ulp {
                max_ulp = u;
            }
        }
        x += 4.0 * step;
    }
    assert!(
        max_ulp <= 1.0,
        "Vec4 sin max ULP error: {max_ulp} > 1.0"
    );
}

#[test]
fn test_vec4_cos_sweep() {
    let mut max_ulp = 0.0f32;
    let step = 0.001;
    let mut x = -125.0f32;
    while x < 125.0 {
        let v = Vec4([x, x + step, x + 2.0 * step, x + 3.0 * step]);
        let r = v.cos();
        for i in 0..4 {
            let expected = v[i].cos();
            let u = ulp_error(r[i], expected);
            if u > max_ulp {
                max_ulp = u;
            }
        }
        x += 4.0 * step;
    }
    assert!(
        max_ulp <= 1.0,
        "Vec4 cos max ULP error: {max_ulp} > 1.0"
    );
}

#[test]
fn test_vec4_exp_sweep() {
    let mut max_ulp = 0.0f32;
    let step = 0.001;
    let mut x = -87.0f32;
    while x < 87.0 {
        let v = Vec4([x, x + step, x + 2.0 * step, x + 3.0 * step]);
        let r = v.exp();
        for i in 0..4 {
            let expected = v[i].exp();
            if expected.is_finite() && expected.is_normal() {
                let u = ulp_error(r[i], expected);
                if u > max_ulp {
                    max_ulp = u;
                }
            }
        }
        x += 4.0 * step;
    }
    assert!(
        max_ulp <= 1.0,
        "Vec4 exp max ULP error: {max_ulp} > 1.0"
    );
}

#[test]
fn test_vec8_sin_sweep() {
    let mut max_ulp = 0.0f32;
    let step = 0.001;
    let mut x = -125.0f32;
    while x < 125.0 {
        let v = Vec8([
            x,
            x + step,
            x + 2.0 * step,
            x + 3.0 * step,
            x + 4.0 * step,
            x + 5.0 * step,
            x + 6.0 * step,
            x + 7.0 * step,
        ]);
        let r = v.sin();
        for i in 0..8 {
            let expected = v[i].sin();
            let u = ulp_error(r[i], expected);
            if u > max_ulp {
                max_ulp = u;
            }
        }
        x += 8.0 * step;
    }
    assert!(
        max_ulp <= 1.0,
        "Vec8 sin max ULP error: {max_ulp} > 1.0"
    );
}

#[test]
fn test_vec8_cos_sweep() {
    let mut max_ulp = 0.0f32;
    let step = 0.001;
    let mut x = -125.0f32;
    while x < 125.0 {
        let v = Vec8([
            x,
            x + step,
            x + 2.0 * step,
            x + 3.0 * step,
            x + 4.0 * step,
            x + 5.0 * step,
            x + 6.0 * step,
            x + 7.0 * step,
        ]);
        let r = v.cos();
        for i in 0..8 {
            let expected = v[i].cos();
            let u = ulp_error(r[i], expected);
            if u > max_ulp {
                max_ulp = u;
            }
        }
        x += 8.0 * step;
    }
    assert!(
        max_ulp <= 1.0,
        "Vec8 cos max ULP error: {max_ulp} > 1.0"
    );
}

#[test]
fn test_vec8_exp_sweep() {
    let mut max_ulp = 0.0f32;
    let step = 0.001;
    let mut x = -87.0f32;
    while x < 87.0 {
        let v = Vec8([
            x,
            x + step,
            x + 2.0 * step,
            x + 3.0 * step,
            x + 4.0 * step,
            x + 5.0 * step,
            x + 6.0 * step,
            x + 7.0 * step,
        ]);
        let r = v.exp();
        for i in 0..8 {
            let expected = v[i].exp();
            if expected.is_finite() && expected.is_normal() {
                let u = ulp_error(r[i], expected);
                if u > max_ulp {
                    max_ulp = u;
                }
            }
        }
        x += 8.0 * step;
    }
    assert!(
        max_ulp <= 1.0,
        "Vec8 exp max ULP error: {max_ulp} > 1.0"
    );
}

// ============ Sin/Cos identity tests ============

#[test]
fn test_vec4_sin_cos_identity() {
    let vals = [0.5, 1.0, 1.5, 2.0];
    let v = Vec4(vals);
    let s = v.sin();
    let c = v.cos();
    for i in 0..4 {
        let sum_sq = s[i] * s[i] + c[i] * c[i];
        assert!(
            (sum_sq - 1.0).abs() < 1e-5,
            "sin^2 + cos^2 != 1 for x={}: got {sum_sq}",
            vals[i]
        );
    }
}

#[test]
fn test_vec8_sin_cos_identity() {
    let vals = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0];
    let v = Vec8(vals);
    let s = v.sin();
    let c = v.cos();
    for i in 0..8 {
        let sum_sq = s[i] * s[i] + c[i] * c[i];
        assert!(
            (sum_sq - 1.0).abs() < 1e-5,
            "sin^2 + cos^2 != 1 for x={}: got {sum_sq}",
            vals[i]
        );
    }
}

// ============ Exp properties ============

#[test]
fn test_vec4_exp_properties() {
    // exp(a) * exp(b) ≈ exp(a+b)
    let a = Vec4([1.0, 2.0, 3.0, -1.0]);
    let b = Vec4([0.5, -0.5, 1.0, -2.0]);
    let ea = a.exp();
    let eb = b.exp();
    let eab = (a + b).exp();
    let prod = ea * eb;
    for i in 0..4 {
        let rel_err = ((prod[i] - eab[i]) / eab[i]).abs();
        assert!(rel_err < 1e-5, "exp(a)*exp(b) != exp(a+b) at i={i}");
    }
}
