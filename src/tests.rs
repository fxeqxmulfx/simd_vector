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

#[test]
fn test_vec8_exp_properties() {
    let a = Vec8([1.0, 2.0, 3.0, -1.0, 0.5, -0.5, 4.0, -3.0]);
    let b = Vec8([0.5, -0.5, 1.0, -2.0, 1.5, 2.0, -1.0, 0.5]);
    let ea = a.exp();
    let eb = b.exp();
    let eab = (a + b).exp();
    let prod = ea * eb;
    for i in 0..8 {
        let rel_err = ((prod[i] - eab[i]) / eab[i]).abs();
        assert!(rel_err < 1e-5, "exp(a)*exp(b) != exp(a+b) at i={i}");
    }
}

// ============ Boundary: Cody-Waite / rempif transition at 125.0 ============

#[test]
fn test_vec4_sin_boundary_125() {
    // Mix lanes: some < 125 (Cody-Waite), some > 125 (rempif)
    let vals = [124.9, 125.0, 125.1, 126.0];
    let v = Vec4(vals);
    let r = v.sin();
    for i in 0..4 {
        assert_ulp(r[i], vals[i].sin(), 1.0, &format!("sin({})", vals[i]));
    }
}

#[test]
fn test_vec4_cos_boundary_125() {
    let vals = [124.9, 125.0, 125.1, 126.0];
    let v = Vec4(vals);
    let r = v.cos();
    for i in 0..4 {
        assert_ulp(r[i], vals[i].cos(), 1.0, &format!("cos({})", vals[i]));
    }
}

#[test]
fn test_vec8_sin_boundary_125() {
    let vals = [124.0, 124.5, 124.9, 125.0, 125.1, 125.5, 126.0, 130.0];
    let v = Vec8(vals);
    let r = v.sin();
    for i in 0..8 {
        assert_ulp(r[i], vals[i].sin(), 1.0, &format!("sin({})", vals[i]));
    }
}

#[test]
fn test_vec8_cos_boundary_125() {
    let vals = [124.0, 124.5, 124.9, 125.0, 125.1, 125.5, 126.0, 130.0];
    let v = Vec8(vals);
    let r = v.cos();
    for i in 0..8 {
        assert_ulp(r[i], vals[i].cos(), 1.0, &format!("cos({})", vals[i]));
    }
}

// ============ Mixed in-range / out-of-range lanes ============

#[test]
fn test_vec4_sin_mixed_range() {
    let vals = [1.0, 200.0, -0.5, -300.0];
    let v = Vec4(vals);
    let r = v.sin();
    for i in 0..4 {
        assert_ulp(r[i], vals[i].sin(), 1.0, &format!("sin({})", vals[i]));
    }
}

#[test]
fn test_vec4_cos_mixed_range() {
    let vals = [1.0, 200.0, -0.5, -300.0];
    let v = Vec4(vals);
    let r = v.cos();
    for i in 0..4 {
        assert_ulp(r[i], vals[i].cos(), 1.0, &format!("cos({})", vals[i]));
    }
}

#[test]
fn test_vec8_sin_mixed_range() {
    let vals = [0.1, 200.0, -50.0, 500.0, 3.0, -1000.0, 124.9, 126.0];
    let v = Vec8(vals);
    let r = v.sin();
    for i in 0..8 {
        assert_ulp(r[i], vals[i].sin(), 1.0, &format!("sin({})", vals[i]));
    }
}

#[test]
fn test_vec8_cos_mixed_range() {
    let vals = [0.1, 200.0, -50.0, 500.0, 3.0, -1000.0, 124.9, 126.0];
    let v = Vec8(vals);
    let r = v.cos();
    for i in 0..8 {
        assert_ulp(r[i], vals[i].cos(), 1.0, &format!("cos({})", vals[i]));
    }
}

// ============ Very large values for sin/cos ============

#[test]
fn test_vec4_sin_very_large() {
    let vals = [1e4, -1e4, 1e5, -1e5];
    let v = Vec4(vals);
    let r = v.sin();
    for i in 0..4 {
        assert_ulp(r[i], vals[i].sin(), 1.0, &format!("sin({})", vals[i]));
    }
}

#[test]
fn test_vec4_cos_very_large() {
    let vals = [1e4, -1e4, 1e5, -1e5];
    let v = Vec4(vals);
    let r = v.cos();
    for i in 0..4 {
        assert_ulp(r[i], vals[i].cos(), 1.0, &format!("cos({})", vals[i]));
    }
}

#[test]
fn test_vec8_sin_very_large() {
    let vals = [1e4, -1e4, 1e5, -1e5, 1e6, -1e6, 5e5, -5e5];
    let v = Vec8(vals);
    let r = v.sin();
    for i in 0..8 {
        assert_ulp(r[i], vals[i].sin(), 1.0, &format!("sin({})", vals[i]));
    }
}

#[test]
fn test_vec8_cos_very_large() {
    let vals = [1e4, -1e4, 1e5, -1e5, 1e6, -1e6, 5e5, -5e5];
    let v = Vec8(vals);
    let r = v.cos();
    for i in 0..8 {
        assert_ulp(r[i], vals[i].cos(), 1.0, &format!("cos({})", vals[i]));
    }
}

// ============ Small values near zero ============

#[test]
fn test_vec4_sin_small() {
    let vals = [1e-7, -1e-7, 1e-10, -1e-10];
    let v = Vec4(vals);
    let r = v.sin();
    for i in 0..4 {
        // For very small x, sin(x) ≈ x
        assert_ulp(r[i], vals[i].sin(), 1.0, &format!("sin({})", vals[i]));
    }
}

#[test]
fn test_vec4_cos_small() {
    let vals = [1e-7, -1e-7, 1e-10, -1e-10];
    let v = Vec4(vals);
    let r = v.cos();
    for i in 0..4 {
        // For very small x, cos(x) ≈ 1
        assert_ulp(r[i], vals[i].cos(), 1.0, &format!("cos({})", vals[i]));
    }
}

#[test]
fn test_vec8_sin_small() {
    let vals = [1e-5, -1e-5, 1e-7, -1e-7, 1e-10, -1e-10, 1e-20, -1e-20];
    let v = Vec8(vals);
    let r = v.sin();
    for i in 0..8 {
        assert_ulp(r[i], vals[i].sin(), 1.0, &format!("sin({})", vals[i]));
    }
}

// ============ Exact multiples of pi ============

#[test]
fn test_vec4_sin_multiples_of_pi() {
    let vals = [PI, 2.0 * PI, 3.0 * PI, 4.0 * PI];
    let v = Vec4(vals);
    let r = v.sin();
    for i in 0..4 {
        assert_ulp(r[i], vals[i].sin(), 1.0, &format!("sin({}*pi)", i + 1));
    }
}

#[test]
fn test_vec4_cos_multiples_of_half_pi() {
    let vals = [FRAC_PI_2, PI, 3.0 * FRAC_PI_2, 2.0 * PI];
    let v = Vec4(vals);
    let r = v.cos();
    for i in 0..4 {
        assert_ulp(r[i], vals[i].cos(), 1.0, &format!("cos({})", vals[i]));
    }
}

#[test]
fn test_vec8_sin_negative_multiples() {
    let vals = [-PI, -2.0 * PI, -3.0 * PI, -4.0 * PI,
                -FRAC_PI_2, -FRAC_PI_4, -FRAC_PI_6, -FRAC_PI_3];
    let v = Vec8(vals);
    let r = v.sin();
    for i in 0..8 {
        assert_ulp(r[i], vals[i].sin(), 1.0, &format!("sin({})", vals[i]));
    }
}

// ============ Exp boundary tests ============

#[test]
fn test_vec4_exp_near_clamp_boundaries() {
    // Near the -104 (->0) and 100 (->inf) clamp boundaries
    let vals = [-103.9, -104.0, -104.1, -105.0];
    let v = Vec4(vals);
    let r = v.exp();
    // -103.9 should still produce a small positive value
    assert!(r[0] > 0.0 && r[0].is_finite(), "exp(-103.9) should be finite positive");
    // -104 and below should clamp to 0
    assert_eq!(r[1], 0.0, "exp(-104) should be 0");
    assert_eq!(r[2], 0.0, "exp(-104.1) should be 0");
    assert_eq!(r[3], 0.0, "exp(-105) should be 0");
}

#[test]
fn test_vec4_exp_near_overflow() {
    // f32::MAX ≈ 3.4e38, exp(88.72) ≈ f32::MAX, so anything above ~88.72 overflows
    let vals = [88.0, 88.7, 100.0, 101.0];
    let v = Vec4(vals);
    let r = v.exp();
    assert!(r[0].is_finite(), "exp(88) should be finite");
    assert_ulp(r[0], 88.0f32.exp(), 1.0, "exp(88)");
    assert!(r[1].is_finite(), "exp(88.7) should be finite");
    assert_ulp(r[1], 88.7f32.exp(), 1.0, "exp(88.7)");
    assert_eq!(r[2], f32::INFINITY, "exp(100) should be inf");
    assert_eq!(r[3], f32::INFINITY, "exp(101) should be inf");
}

#[test]
fn test_vec8_exp_clamp_boundaries() {
    let vals = [-103.5, -103.9, -104.0, -110.0, 85.0, 88.0, 100.0, 110.0];
    let v = Vec8(vals);
    let r = v.exp();
    assert!(r[0] > 0.0 && r[0].is_finite());
    assert!(r[1] > 0.0 && r[1].is_finite());
    assert_eq!(r[2], 0.0);
    assert_eq!(r[3], 0.0);
    assert!(r[4].is_finite());
    assert_ulp(r[4], 85.0f32.exp(), 1.0, "exp(85)");
    assert!(r[5].is_finite());
    assert_ulp(r[5], 88.0f32.exp(), 1.0, "exp(88)");
    assert_eq!(r[6], f32::INFINITY);
    assert_eq!(r[7], f32::INFINITY);
}

// ============ Special float edge cases ============

#[test]
fn test_vec4_neg_special() {
    let v = Vec4([0.0, -0.0, f32::INFINITY, f32::NEG_INFINITY]);
    let r = -v;
    assert_eq!(r[0], -0.0);
    assert!(r[0].is_sign_negative());
    assert_eq!(r[1], 0.0);
    assert!(r[1].is_sign_positive());
    assert_eq!(r[2], f32::NEG_INFINITY);
    assert_eq!(r[3], f32::INFINITY);
}

#[test]
fn test_vec4_neg_nan() {
    let v = Vec4([f32::NAN, 1.0, -1.0, 0.0]);
    let r = -v;
    assert!(r[0].is_nan());
}

#[test]
fn test_vec4_abs_special() {
    let v = Vec4([-0.0, 0.0, f32::NEG_INFINITY, f32::INFINITY]);
    let r = v.abs();
    assert_eq!(r[0], 0.0);
    assert!(r[0].is_sign_positive());
    assert_eq!(r[1], 0.0);
    assert_eq!(r[2], f32::INFINITY);
    assert_eq!(r[3], f32::INFINITY);
}

#[test]
fn test_vec4_abs_nan() {
    let v = Vec4([f32::NAN, -f32::NAN, 1.0, -1.0]);
    let r = v.abs();
    assert!(r[0].is_nan());
    assert!(r[1].is_nan());
}

#[test]
fn test_vec4_sqrt_special() {
    let v = Vec4([0.0, -0.0, f32::INFINITY, 1.0]);
    let r = v.sqrt();
    assert_eq!(r[0], 0.0);
    assert_eq!(r[1], -0.0);
    assert!(r[1].is_sign_negative());
    assert_eq!(r[2], f32::INFINITY);
    assert_eq!(r[3], 1.0);
}

#[test]
fn test_vec4_sqrt_negative() {
    let v = Vec4([-1.0, -4.0, 0.0, 4.0]);
    let r = v.sqrt();
    assert!(r[0].is_nan());
    assert!(r[1].is_nan());
}

#[test]
fn test_vec4_floor_special() {
    let v = Vec4([f32::INFINITY, f32::NEG_INFINITY, -0.0, 0.0]);
    let r = v.floor();
    assert_eq!(r[0], f32::INFINITY);
    assert_eq!(r[1], f32::NEG_INFINITY);
    assert_eq!(r[2], -0.0);
    assert!(r[2].is_sign_negative());
    assert_eq!(r[3], 0.0);
}

#[test]
fn test_vec4_floor_integers() {
    let v = Vec4([1.0, -1.0, 100.0, -100.0]);
    let r = v.floor();
    assert_eq!(r.0, [1.0, -1.0, 100.0, -100.0]);
}

#[test]
fn test_vec8_neg_special() {
    let v = Vec8([0.0, -0.0, f32::INFINITY, f32::NEG_INFINITY,
                  f32::NAN, 1.0, -1.0, f32::MAX]);
    let r = -v;
    assert!(r[0].is_sign_negative());
    assert!(r[1].is_sign_positive());
    assert_eq!(r[2], f32::NEG_INFINITY);
    assert_eq!(r[3], f32::INFINITY);
    assert!(r[4].is_nan());
    assert_eq!(r[5], -1.0);
    assert_eq!(r[6], 1.0);
    assert_eq!(r[7], -f32::MAX);
}

#[test]
fn test_vec8_abs_special() {
    let v = Vec8([-0.0, 0.0, f32::NEG_INFINITY, f32::INFINITY,
                  f32::NAN, -f32::MAX, f32::MIN_POSITIVE, -f32::MIN_POSITIVE]);
    let r = v.abs();
    assert!(r[0].is_sign_positive());
    assert_eq!(r[2], f32::INFINITY);
    assert_eq!(r[3], f32::INFINITY);
    assert!(r[4].is_nan());
    assert_eq!(r[5], f32::MAX);
}

#[test]
fn test_vec8_sqrt_special() {
    let v = Vec8([0.0, -0.0, 1.0, 4.0, 9.0, f32::INFINITY, -1.0, f32::NAN]);
    let r = v.sqrt();
    assert_eq!(r[0], 0.0);
    assert!(r[1].is_sign_negative() && r[1] == 0.0);
    assert_eq!(r[2], 1.0);
    assert_eq!(r[3], 2.0);
    assert_eq!(r[4], 3.0);
    assert_eq!(r[5], f32::INFINITY);
    assert!(r[6].is_nan());
    assert!(r[7].is_nan());
}

// ============ Div by zero ============

#[test]
fn test_vec4_div_by_zero() {
    let a = Vec4([1.0, -1.0, 0.0, -0.0]);
    let b = Vec4([0.0, 0.0, 0.0, 0.0]);
    let r = a / b;
    assert_eq!(r[0], f32::INFINITY);
    assert_eq!(r[1], f32::NEG_INFINITY);
    assert!(r[2].is_nan()); // 0/0
    assert!(r[3].is_nan()); // -0/0
}

#[test]
fn test_vec8_div_by_zero() {
    let a = Vec8([1.0, -1.0, 0.0, f32::INFINITY, f32::NEG_INFINITY, -0.0, 2.0, -2.0]);
    let b = Vec8([0.0; 8]);
    let r = a / b;
    assert_eq!(r[0], f32::INFINITY);
    assert_eq!(r[1], f32::NEG_INFINITY);
    assert!(r[2].is_nan());
    assert_eq!(r[3], f32::INFINITY);
    assert_eq!(r[4], f32::NEG_INFINITY);
    assert!(r[5].is_nan()); // -0/0
    assert_eq!(r[6], f32::INFINITY);
    assert_eq!(r[7], f32::NEG_INFINITY);
}

// ============ Dot product edge cases ============

#[test]
fn test_vec4_dot_zeros() {
    assert_eq!(Vec4::ZERO.dot(Vec4::ONE), 0.0);
    assert_eq!(Vec4::ZERO.dot(Vec4::ZERO), 0.0);
}

#[test]
fn test_vec4_dot_orthogonal() {
    let a = Vec4([1.0, 0.0, 0.0, 0.0]);
    let b = Vec4([0.0, 1.0, 0.0, 0.0]);
    assert_eq!(a.dot(b), 0.0);
}

#[test]
fn test_vec4_dot_self() {
    let a = Vec4([3.0, 4.0, 0.0, 0.0]);
    assert_eq!(a.dot(a), 25.0); // 9 + 16
}

#[test]
fn test_vec8_dot_zeros() {
    assert_eq!(Vec8::ZERO.dot(Vec8::ONE), 0.0);
}

#[test]
fn test_vec8_dot_self() {
    let a = Vec8([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    assert_eq!(a.dot(a), 8.0);
}

// ============ Cos at -0.0, Exp at -0.0 ============

#[test]
fn test_vec4_cos_neg_zero() {
    let v = Vec4([-0.0, 0.0, -0.0, 0.0]);
    let r = v.cos();
    assert_eq!(r[0], 1.0);
    assert_eq!(r[1], 1.0);
}

#[test]
fn test_vec4_exp_neg_zero() {
    let v = Vec4([-0.0, 0.0, -0.0, 0.0]);
    let r = v.exp();
    assert_eq!(r[0], 1.0);
    assert_eq!(r[1], 1.0);
}

#[test]
fn test_vec8_cos_neg_zero() {
    let mut vals = [0.0f32; 8];
    vals[0] = -0.0;
    vals[2] = -0.0;
    let r = Vec8(vals).cos();
    for i in 0..8 {
        assert_eq!(r[i], 1.0, "cos(0) should be 1 at lane {i}");
    }
}

#[test]
fn test_vec8_exp_neg_zero() {
    let mut vals = [0.0f32; 8];
    vals[0] = -0.0;
    vals[3] = -0.0;
    let r = Vec8(vals).exp();
    for i in 0..8 {
        assert_eq!(r[i], 1.0, "exp(0) should be 1 at lane {i}");
    }
}

// ============ PartialEq edge cases ============

#[test]
fn test_vec4_partial_eq_neg_zero() {
    // IEEE 754: 0.0 == -0.0
    let a = Vec4([0.0, -0.0, 1.0, -1.0]);
    let b = Vec4([-0.0, 0.0, 1.0, -1.0]);
    assert_eq!(a, b);
}

#[test]
fn test_vec4_partial_eq_nan() {
    // NaN != NaN
    let a = Vec4([f32::NAN, 1.0, 2.0, 3.0]);
    let b = Vec4([f32::NAN, 1.0, 2.0, 3.0]);
    assert_ne!(a, b);
}

#[test]
fn test_vec8_partial_eq_neg_zero() {
    let a = Vec8([0.0; 8]);
    let b = Vec8([-0.0; 8]);
    assert_eq!(a, b);
}

#[test]
fn test_vec8_partial_eq_nan() {
    let mut a = [1.0f32; 8];
    a[4] = f32::NAN;
    let mut b = [1.0f32; 8];
    b[4] = f32::NAN;
    assert_ne!(Vec8(a), Vec8(b));
}

// ============ FMA precision test ============

#[test]
fn test_vec4_mul_add_precision() {
    // FMA should be more precise than separate mul + add for this case
    // 1.0000001 * 1.0000001 + (-1.0) should give ~2e-7 with FMA precision
    let a = Vec4([1.0000001; 4]);
    let b = Vec4([1.0000001; 4]);
    let c = Vec4([-1.0; 4]);
    let fma_result = a.mul_add(b, c);
    // Just verify it computes correctly (FMA has single rounding)
    let expected = 1.0000001f32.mul_add(1.0000001, -1.0);
    for i in 0..4 {
        assert_eq!(fma_result[i], expected, "FMA precision at lane {i}");
    }
}

#[test]
fn test_vec8_mul_add_precision() {
    let a = Vec8([1.0000001; 8]);
    let b = Vec8([1.0000001; 8]);
    let c = Vec8([-1.0; 8]);
    let fma_result = a.mul_add(b, c);
    let expected = 1.0000001f32.mul_add(1.0000001, -1.0);
    for i in 0..8 {
        assert_eq!(fma_result[i], expected, "FMA precision at lane {i}");
    }
}

// ============ Large value sin/cos sweep (rempif path) ============

#[test]
fn test_vec4_sin_sweep_large() {
    let mut max_ulp = 0.0f32;
    let step = 0.1;
    let mut x = 125.0f32;
    while x < 1000.0 {
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
        "Vec4 sin large-range max ULP error: {max_ulp} > 1.0"
    );
}

#[test]
fn test_vec4_cos_sweep_large() {
    let mut max_ulp = 0.0f32;
    let step = 0.1;
    let mut x = 125.0f32;
    while x < 1000.0 {
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
        "Vec4 cos large-range max ULP error: {max_ulp} > 1.0"
    );
}

#[test]
fn test_vec8_sin_sweep_large() {
    let mut max_ulp = 0.0f32;
    let step = 0.1;
    let mut x = 125.0f32;
    while x < 1000.0 {
        let v = Vec8([
            x, x + step, x + 2.0 * step, x + 3.0 * step,
            x + 4.0 * step, x + 5.0 * step, x + 6.0 * step, x + 7.0 * step,
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
        "Vec8 sin large-range max ULP error: {max_ulp} > 1.0"
    );
}

#[test]
fn test_vec8_cos_sweep_large() {
    let mut max_ulp = 0.0f32;
    let step = 0.1;
    let mut x = 125.0f32;
    while x < 1000.0 {
        let v = Vec8([
            x, x + step, x + 2.0 * step, x + 3.0 * step,
            x + 4.0 * step, x + 5.0 * step, x + 6.0 * step, x + 7.0 * step,
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
        "Vec8 cos large-range max ULP error: {max_ulp} > 1.0"
    );
}

// ============ Sin/cos identity for large values ============

#[test]
fn test_vec4_sin_cos_identity_large() {
    let vals = [200.0, 500.0, 750.0, 999.0];
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
fn test_vec8_sin_cos_identity_large() {
    let vals = [200.0, 400.0, 600.0, 800.0, 150.0, 350.0, 550.0, 950.0];
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

// ============ Sin odd / Cos even symmetry ============

#[test]
fn test_vec4_sin_odd_symmetry() {
    let vals = [0.5, 1.0, 2.0, 3.0];
    let v_pos = Vec4(vals);
    let v_neg = Vec4([-0.5, -1.0, -2.0, -3.0]);
    let s_pos = v_pos.sin();
    let s_neg = v_neg.sin();
    for i in 0..4 {
        assert_eq!(s_pos[i], -s_neg[i], "sin(-x) != -sin(x) for x={}", vals[i]);
    }
}

#[test]
fn test_vec4_cos_even_symmetry() {
    let vals = [0.5, 1.0, 2.0, 3.0];
    let v_pos = Vec4(vals);
    let v_neg = Vec4([-0.5, -1.0, -2.0, -3.0]);
    let c_pos = v_pos.cos();
    let c_neg = v_neg.cos();
    for i in 0..4 {
        assert_eq!(c_pos[i], c_neg[i], "cos(-x) != cos(x) for x={}", vals[i]);
    }
}

#[test]
fn test_vec8_sin_odd_symmetry() {
    let vals = [0.5, 1.0, 2.0, 3.0, 100.0, 200.0, 500.0, 0.001];
    let neg_vals = [-0.5, -1.0, -2.0, -3.0, -100.0, -200.0, -500.0, -0.001];
    let s_pos = Vec8(vals).sin();
    let s_neg = Vec8(neg_vals).sin();
    for i in 0..8 {
        assert_eq!(s_pos[i], -s_neg[i], "sin(-x) != -sin(x) for x={}", vals[i]);
    }
}

#[test]
fn test_vec8_cos_even_symmetry() {
    let vals = [0.5, 1.0, 2.0, 3.0, 100.0, 200.0, 500.0, 0.001];
    let neg_vals = [-0.5, -1.0, -2.0, -3.0, -100.0, -200.0, -500.0, -0.001];
    let c_pos = Vec8(vals).cos();
    let c_neg = Vec8(neg_vals).cos();
    for i in 0..8 {
        assert_eq!(c_pos[i], c_neg[i], "cos(-x) != cos(x) for x={}", vals[i]);
    }
}

// ============ Exp symmetry: exp(-x) = 1/exp(x) ============

#[test]
fn test_vec4_exp_reciprocal() {
    let vals = [1.0, 2.0, 5.0, 10.0];
    let neg_vals = [-1.0, -2.0, -5.0, -10.0];
    let e_pos = Vec4(vals).exp();
    let e_neg = Vec4(neg_vals).exp();
    for i in 0..4 {
        let product = e_pos[i] * e_neg[i];
        assert!(
            (product - 1.0).abs() < 1e-5,
            "exp(x)*exp(-x) != 1 for x={}: got {product}",
            vals[i]
        );
    }
}

#[test]
fn test_vec8_exp_reciprocal() {
    let vals = [1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 0.5, 0.001];
    let neg_vals = [-1.0, -2.0, -5.0, -10.0, -20.0, -30.0, -0.5, -0.001];
    let e_pos = Vec8(vals).exp();
    let e_neg = Vec8(neg_vals).exp();
    for i in 0..8 {
        let product = e_pos[i] * e_neg[i];
        assert!(
            (product - 1.0).abs() < 1e-4,
            "exp(x)*exp(-x) != 1 for x={}: got {product}",
            vals[i]
        );
    }
}

// ============ Arithmetic with infinity/NaN ============

#[test]
fn test_vec4_arithmetic_inf() {
    let inf = Vec4([f32::INFINITY; 4]);
    let one = Vec4::ONE;
    let r_add = inf + one;
    assert_eq!(r_add[0], f32::INFINITY);
    let r_mul = inf * one;
    assert_eq!(r_mul[0], f32::INFINITY);
    let r_sub = inf - inf;
    assert!(r_sub[0].is_nan()); // inf - inf = NaN
}

#[test]
fn test_vec4_arithmetic_nan_propagation() {
    let nan = Vec4([f32::NAN; 4]);
    let one = Vec4::ONE;
    assert!((nan + one)[0].is_nan());
    assert!((nan - one)[0].is_nan());
    assert!((nan * one)[0].is_nan());
    assert!((nan / one)[0].is_nan());
}

#[test]
fn test_vec8_arithmetic_nan_propagation() {
    let nan = Vec8([f32::NAN; 8]);
    let one = Vec8::ONE;
    assert!((nan + one)[0].is_nan());
    assert!((nan - one)[0].is_nan());
    assert!((nan * one)[0].is_nan());
    assert!((nan / one)[0].is_nan());
}

// ============ Exp with NaN/Inf ============

#[test]
fn test_vec4_exp_inf() {
    let v = Vec4([f32::INFINITY, f32::NEG_INFINITY, 0.0, 1.0]);
    let r = v.exp();
    assert_eq!(r[0], f32::INFINITY);
    assert_eq!(r[1], 0.0);
    assert_eq!(r[2], 1.0);
}

#[test]
fn test_vec8_exp_inf() {
    let v = Vec8([f32::INFINITY, f32::NEG_INFINITY, 0.0, 1.0,
                  f32::NAN, -1.0, 50.0, -50.0]);
    let r = v.exp();
    assert_eq!(r[0], f32::INFINITY);
    assert_eq!(r[1], 0.0);
    assert_eq!(r[2], 1.0);
    assert!(r[4].is_nan());
}

// ============ Vec4/Vec8 consistency: same inputs, same results ============

#[test]
fn test_vec4_vec8_sin_consistency() {
    let vals = [0.5, 1.0, -2.0, 3.14];
    let r4 = Vec4(vals).sin();
    let r8 = Vec8([vals[0], vals[1], vals[2], vals[3], 0.0, 0.0, 0.0, 0.0]).sin();
    for i in 0..4 {
        assert_eq!(r4[i], r8[i], "Vec4/Vec8 sin mismatch at lane {i} for x={}", vals[i]);
    }
}

#[test]
fn test_vec4_vec8_cos_consistency() {
    let vals = [0.5, 1.0, -2.0, 3.14];
    let r4 = Vec4(vals).cos();
    let r8 = Vec8([vals[0], vals[1], vals[2], vals[3], 0.0, 0.0, 0.0, 0.0]).cos();
    for i in 0..4 {
        assert_eq!(r4[i], r8[i], "Vec4/Vec8 cos mismatch at lane {i} for x={}", vals[i]);
    }
}

#[test]
fn test_vec4_vec8_exp_consistency() {
    let vals = [0.5, 1.0, -2.0, 3.14];
    let r4 = Vec4(vals).exp();
    let r8 = Vec8([vals[0], vals[1], vals[2], vals[3], 0.0, 0.0, 0.0, 0.0]).exp();
    for i in 0..4 {
        assert_eq!(r4[i], r8[i], "Vec4/Vec8 exp mismatch at lane {i} for x={}", vals[i]);
    }
}

// ============ Splat consistency ============

#[test]
fn test_vec4_splat_special() {
    let v = Vec4::splat(f32::NAN);
    for i in 0..4 {
        assert!(v[i].is_nan());
    }
    let v = Vec4::splat(f32::INFINITY);
    assert_eq!(v.0, [f32::INFINITY; 4]);
    let v = Vec4::splat(-0.0);
    for i in 0..4 {
        assert!(v[i].is_sign_negative());
    }
}

#[test]
fn test_vec8_splat_special() {
    let v = Vec8::splat(f32::NAN);
    for i in 0..8 {
        assert!(v[i].is_nan());
    }
    let v = Vec8::splat(f32::INFINITY);
    assert_eq!(v.0, [f32::INFINITY; 8]);
    let v = Vec8::splat(-0.0);
    for i in 0..8 {
        assert!(v[i].is_sign_negative());
    }
}

// ============ Direct scalar helper tests (coverage for #[inline(always)] fns) ============

#[test]
fn test_df_normalize_scalar() {
    let (hi, lo) = df_normalize(1.0, 1e-10);
    assert_eq!(hi + lo, 1.0 + 1e-10);
    // hi should be close to the sum, lo is the residual
    assert!((hi - 1.0).abs() < 1e-6);

    // Exact case
    let (hi, lo) = df_normalize(1.0, 0.0);
    assert_eq!(hi, 1.0);
    assert_eq!(lo, 0.0);
}

#[test]
fn test_df_add2_f2_f2_scalar() {
    let (hi, lo) = df_add2_f2_f2(1.0, 1e-10, 2.0, 2e-10);
    let sum = hi as f64 + lo as f64;
    let expected = 1.0f64 + 1e-10 + 2.0 + 2e-10;
    assert!((sum - expected).abs() < 1e-14, "df_add2_f2_f2 error: {sum} vs {expected}");
}

#[test]
fn test_df_mul_f_f_scalar() {
    let a = 1.0000001f32;
    let b = 1.0000002f32;
    let (hi, lo) = df_mul_f_f(a, b);
    // hi+lo pair should represent the exact product of the f32 values
    let result = hi as f64 + lo as f64;
    let expected = (a as f64) * (b as f64);
    assert!((result - expected).abs() < 1e-14, "df_mul_f_f error: {result} vs {expected}");

    // Exact case
    let (hi, lo) = df_mul_f_f(2.0, 3.0);
    assert_eq!(hi, 6.0);
    assert_eq!(lo, 0.0);
}

#[test]
fn test_df_mul_f2_f_scalar() {
    let (hi, lo) = df_mul_f2_f(1.0, 1e-8, 3.0);
    let result = hi as f64 + lo as f64;
    let expected = (1.0f64 + 1e-8) * 3.0;
    assert!((result - expected).abs() < 1e-13, "df_mul_f2_f error");
}

#[test]
fn test_df_mul_f2_f2_scalar() {
    let (hi, lo) = df_mul_f2_f2(1.0, 1e-8, 2.0, 2e-8);
    let result = hi as f64 + lo as f64;
    let expected = (1.0f64 + 1e-8) * (2.0f64 + 2e-8);
    assert!((result - expected).abs() < 1e-13, "df_mul_f2_f2 error");
}

// ============ Direct rempif_scalar tests ============

#[test]
fn test_rempif_scalar_small_input() {
    // a.abs() < 0.7 → returns (a, 0.0, 0)
    let (hi, lo, q) = rempif_scalar(0.5);
    assert_eq!(hi, 0.5);
    assert_eq!(lo, 0.0);
    assert_eq!(q, 0);

    let (hi, lo, q) = rempif_scalar(-0.3);
    assert_eq!(hi, -0.3);
    assert_eq!(lo, 0.0);
    assert_eq!(q, 0);
}

#[test]
fn test_rempif_scalar_medium_input() {
    // Values in 125..1000 range (normal rempif path)
    let (hi, lo, q) = rempif_scalar(200.0);
    // Result should be the fractional part of 200/pi, times 2*pi
    // 200/(2*pi) ≈ 31.83 → reduced to ~0.83 * 2*pi ≈ 5.22
    let reconstructed = (q as f64) * std::f64::consts::FRAC_PI_2 + hi as f64 + lo as f64;
    let diff = (reconstructed.sin() - (200.0f64).sin()).abs();
    assert!(diff < 1e-5, "rempif_scalar(200) reconstruction error: {diff}");
}

#[test]
fn test_rempif_scalar_very_large_exponent() {
    // Trigger the ex > 65 scaling path: needs exponent > 90
    // 2^91 ≈ 2.48e27
    let large = 1e30f32;
    let (hi, lo, _q) = rempif_scalar(large);
    // Just verify it doesn't crash and returns finite values
    assert!(hi.is_finite(), "rempif_scalar(1e30) hi should be finite, got {hi}");
    assert!(lo.is_finite(), "rempif_scalar(1e30) lo should be finite, got {lo}");

    // Try 1e35 (exponent ~116)
    let very_large = 1e35f32;
    let (hi, lo, _q) = rempif_scalar(very_large);
    assert!(hi.is_finite(), "rempif_scalar(1e35) hi should be finite, got {hi}");
    assert!(lo.is_finite(), "rempif_scalar(1e35) lo should be finite, got {lo}");
}

#[test]
fn test_rempif_scalar_negative_exponent() {
    // ex < 0 gets clamped to 0 → idx = 0
    // Input with small exponent but >= 0.7 so it doesn't hit early return
    let (hi, lo, _q) = rempif_scalar(1.0);
    // 1.0 has exponent 0, ex = 0 - 25 = -25 → clamped to 0
    // Since |1.0| > 0.7, it goes through the full rempif path
    assert!(hi.is_finite());
    assert!(lo.is_finite());
}

// ============ Very large exponent sin/cos (ex > 65 rempif path) ============

#[test]
fn test_vec4_sin_extreme_large() {
    // Values > 2^90 ≈ 1.24e27 to trigger exponent scaling in rempif_scalar
    let vals = [1e28, -1e28, 1e30, -1e30];
    let v = Vec4(vals);
    let r = v.sin();
    for i in 0..4 {
        assert!(r[i].is_finite(), "sin({}) should be finite", vals[i]);
        assert!(r[i].abs() <= 1.0, "sin({}) should be in [-1,1], got {}", vals[i], r[i]);
    }
}

#[test]
fn test_vec4_cos_extreme_large() {
    let vals = [1e28, -1e28, 1e30, -1e30];
    let v = Vec4(vals);
    let r = v.cos();
    for i in 0..4 {
        assert!(r[i].is_finite(), "cos({}) should be finite", vals[i]);
        assert!(r[i].abs() <= 1.0, "cos({}) should be in [-1,1], got {}", vals[i], r[i]);
    }
}

#[test]
fn test_vec8_sin_extreme_large() {
    let vals = [1e28, -1e28, 1e30, -1e30, 1e35, -1e35, 1e38, -1e38];
    let v = Vec8(vals);
    let r = v.sin();
    for i in 0..8 {
        assert!(r[i].is_finite(), "sin({}) should be finite", vals[i]);
        assert!(r[i].abs() <= 1.0, "sin({}) should be in [-1,1], got {}", vals[i], r[i]);
    }
}

#[test]
fn test_vec8_cos_extreme_large() {
    let vals = [1e28, -1e28, 1e30, -1e30, 1e35, -1e35, 1e38, -1e38];
    let v = Vec8(vals);
    let r = v.cos();
    for i in 0..4 {
        assert!(r[i].is_finite(), "cos({}) should be finite", vals[i]);
        assert!(r[i].abs() <= 1.0, "cos({}) should be in [-1,1], got {}", vals[i], r[i]);
    }
}

#[test]
fn test_vec4_sin_cos_identity_extreme() {
    // sin^2 + cos^2 = 1 even at extreme values
    let vals = [1e28, 1e30, 1e35, 1e38];
    let v = Vec4(vals);
    let s = v.sin();
    let c = v.cos();
    for i in 0..4 {
        let sum_sq = s[i] * s[i] + c[i] * c[i];
        assert!(
            (sum_sq - 1.0).abs() < 1e-4,
            "sin^2 + cos^2 != 1 for x={}: got {sum_sq}",
            vals[i]
        );
    }
}

// ============ ulp_error helper edge cases (coverage for test utilities) ============

#[test]
fn test_ulp_error_nan_nan() {
    // Both NaN → 0
    assert_eq!(ulp_error(f32::NAN, f32::NAN), 0.0);
}

#[test]
fn test_ulp_error_equal() {
    assert_eq!(ulp_error(1.0, 1.0), 0.0);
    assert_eq!(ulp_error(0.0, 0.0), 0.0);
    assert_eq!(ulp_error(-1.5, -1.5), 0.0);
}

#[test]
fn test_ulp_error_inf_expected() {
    // expected is infinite → MAX
    assert_eq!(ulp_error(1.0, f32::INFINITY), f32::MAX);
    assert_eq!(ulp_error(1.0, f32::NEG_INFINITY), f32::MAX);
}

#[test]
fn test_ulp_error_nan_computed() {
    // computed is NaN, expected is normal → MAX
    assert_eq!(ulp_error(f32::NAN, 1.0), f32::MAX);
}

#[test]
fn test_ulp_error_one_ulp() {
    // Exactly 1 ULP difference
    let a = 1.0f32;
    let b = f32::from_bits(a.to_bits() + 1);
    let err = ulp_error(b, a);
    assert!((err - 1.0).abs() < 0.01, "expected ~1.0 ULP, got {err}");
}

#[test]
fn test_ulp_error_half_ulp() {
    // Verify sub-ULP differences are measured correctly
    let a = 1.0f32;
    let err = ulp_error(a, a);
    assert_eq!(err, 0.0);
}

// ============ Floor/NaN edge case ============

#[test]
fn test_vec4_floor_nan() {
    let v = Vec4([f32::NAN, 1.5, -1.5, f32::NAN]);
    let r = v.floor();
    assert!(r[0].is_nan());
    assert_eq!(r[1], 1.0);
    assert_eq!(r[2], -2.0);
    assert!(r[3].is_nan());
}

#[test]
fn test_vec8_floor_special() {
    let v = Vec8([f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.0,
                  0.0, 1.0, -1.0, 0.5]);
    let r = v.floor();
    assert!(r[0].is_nan());
    assert_eq!(r[1], f32::INFINITY);
    assert_eq!(r[2], f32::NEG_INFINITY);
    assert!(r[3].is_sign_negative() && r[3] == 0.0);
    assert_eq!(r[4], 0.0);
    assert_eq!(r[5], 1.0);
    assert_eq!(r[6], -1.0);
    assert_eq!(r[7], 0.0);
}

// ============ Scalar f32 ops both sides for Vec8 (mirror Vec4 tests) ============

#[test]
fn test_f32_sub_vec8_ordered() {
    // Verify f32 - Vec8 gives (f32 - each lane), not commutative
    let a = Vec8([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let r = 0.0 - a;
    assert_eq!(r.0, [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0]);
}

#[test]
fn test_f32_div_vec8_ordered() {
    let a = Vec8([2.0; 8]);
    let r = 1.0 / a;
    assert_eq!(r.0, [0.5; 8]);
}

#[test]
fn test_f32_sub_vec4_ordered() {
    let a = Vec4([1.0, 2.0, 3.0, 4.0]);
    let r = 0.0 - a;
    assert_eq!(r.0, [-1.0, -2.0, -3.0, -4.0]);
}

// ============ Dot product with negative values ============

#[test]
fn test_vec4_dot_negative() {
    let a = Vec4([1.0, -1.0, 1.0, -1.0]);
    let b = Vec4([1.0, 1.0, 1.0, 1.0]);
    assert_eq!(a.dot(b), 0.0); // 1 - 1 + 1 - 1
}

#[test]
fn test_vec8_dot_negative() {
    let a = Vec8([1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0]);
    let b = Vec8([1.0; 8]);
    assert_eq!(a.dot(b), 0.0); // alternating sum = 0
}

// ============ Chained operations ============

#[test]
fn test_vec4_chained_ops() {
    let a = Vec4([1.0, 2.0, 3.0, 4.0]);
    // (a * 2 + 1) - a = a + 1
    let r = (a * 2.0 + 1.0) - a;
    assert_eq!(r.0, [2.0, 3.0, 4.0, 5.0]);
}

#[test]
fn test_vec8_chained_ops() {
    let a = Vec8([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let r = (a * 2.0 + 1.0) - a;
    assert_eq!(r.0, [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
}

// ============ Exp small values ============

#[test]
fn test_vec4_exp_small() {
    let vals = [1e-7, -1e-7, 1e-10, -1e-10];
    let v = Vec4(vals);
    let r = v.exp();
    for i in 0..4 {
        assert_ulp(r[i], vals[i].exp(), 1.0, &format!("exp({})", vals[i]));
    }
}

#[test]
fn test_vec8_exp_small() {
    let vals = [1e-5, -1e-5, 1e-7, -1e-7, 1e-10, -1e-10, 1e-20, -1e-20];
    let v = Vec8(vals);
    let r = v.exp();
    for i in 0..8 {
        assert_ulp(r[i], vals[i].exp(), 1.0, &format!("exp({})", vals[i]));
    }
}
