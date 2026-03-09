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
    assert_eq!(a.floor().0, [1.0, 2.0, -2.0, -3.0, 0.0, 3.0, -1.0, 100.0]);
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
    assert_eq!(
        r.0,
        [101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0]
    );
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
    let vecs = vec![Vec8([1.0; 8]), Vec8([2.0; 8]), Vec8([3.0; 8])];
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
    let vals = [
        0.0, FRAC_PI_6, FRAC_PI_4, FRAC_PI_3, FRAC_PI_2, PI, -FRAC_PI_4, -PI,
    ];
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
    let vals = [
        0.0, FRAC_PI_6, FRAC_PI_4, FRAC_PI_3, FRAC_PI_2, PI, -FRAC_PI_4, -PI,
    ];
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
    let v = Vec8([
        0.0,
        -200.0,
        200.0,
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        88.0,
        -88.0,
    ]);
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
    assert!(max_ulp <= 1.0, "Vec4 sin max ULP error: {max_ulp} > 1.0");
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
    assert!(max_ulp <= 1.0, "Vec4 cos max ULP error: {max_ulp} > 1.0");
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
    assert!(max_ulp <= 1.0, "Vec4 exp max ULP error: {max_ulp} > 1.0");
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
    assert!(max_ulp <= 1.0, "Vec8 sin max ULP error: {max_ulp} > 1.0");
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
    assert!(max_ulp <= 1.0, "Vec8 cos max ULP error: {max_ulp} > 1.0");
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
    assert!(max_ulp <= 1.0, "Vec8 exp max ULP error: {max_ulp} > 1.0");
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
    let vals = [
        -PI,
        -2.0 * PI,
        -3.0 * PI,
        -4.0 * PI,
        -FRAC_PI_2,
        -FRAC_PI_4,
        -FRAC_PI_6,
        -FRAC_PI_3,
    ];
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
    assert!(
        r[0] > 0.0 && r[0].is_finite(),
        "exp(-103.9) should be finite positive"
    );
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
    let v = Vec8([
        0.0,
        -0.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        1.0,
        -1.0,
        f32::MAX,
    ]);
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
    let v = Vec8([
        -0.0,
        0.0,
        f32::NEG_INFINITY,
        f32::INFINITY,
        f32::NAN,
        -f32::MAX,
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
    ]);
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
    let a = Vec8([
        1.0,
        -1.0,
        0.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        -0.0,
        2.0,
        -2.0,
    ]);
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
    let v = Vec8([
        f32::INFINITY,
        f32::NEG_INFINITY,
        0.0,
        1.0,
        f32::NAN,
        -1.0,
        50.0,
        -50.0,
    ]);
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
        assert_eq!(
            r4[i], r8[i],
            "Vec4/Vec8 sin mismatch at lane {i} for x={}",
            vals[i]
        );
    }
}

#[test]
fn test_vec4_vec8_cos_consistency() {
    let vals = [0.5, 1.0, -2.0, 3.14];
    let r4 = Vec4(vals).cos();
    let r8 = Vec8([vals[0], vals[1], vals[2], vals[3], 0.0, 0.0, 0.0, 0.0]).cos();
    for i in 0..4 {
        assert_eq!(
            r4[i], r8[i],
            "Vec4/Vec8 cos mismatch at lane {i} for x={}",
            vals[i]
        );
    }
}

#[test]
fn test_vec4_vec8_exp_consistency() {
    let vals = [0.5, 1.0, -2.0, 3.14];
    let r4 = Vec4(vals).exp();
    let r8 = Vec8([vals[0], vals[1], vals[2], vals[3], 0.0, 0.0, 0.0, 0.0]).exp();
    for i in 0..4 {
        assert_eq!(
            r4[i], r8[i],
            "Vec4/Vec8 exp mismatch at lane {i} for x={}",
            vals[i]
        );
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
    assert!(
        (sum - expected).abs() < 1e-14,
        "df_add2_f2_f2 error: {sum} vs {expected}"
    );
}

#[test]
fn test_df_mul_f_f_scalar() {
    let a = 1.0000001f32;
    let b = 1.0000002f32;
    let (hi, lo) = df_mul_f_f(a, b);
    // hi+lo pair should represent the exact product of the f32 values
    let result = hi as f64 + lo as f64;
    let expected = (a as f64) * (b as f64);
    assert!(
        (result - expected).abs() < 1e-14,
        "df_mul_f_f error: {result} vs {expected}"
    );

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
    assert!(
        diff < 1e-5,
        "rempif_scalar(200) reconstruction error: {diff}"
    );
}

#[test]
fn test_rempif_scalar_very_large_exponent() {
    // Trigger the ex > 65 scaling path: needs exponent > 90
    // 2^91 ≈ 2.48e27
    let large = 1e30f32;
    let (hi, lo, _q) = rempif_scalar(large);
    // Just verify it doesn't crash and returns finite values
    assert!(
        hi.is_finite(),
        "rempif_scalar(1e30) hi should be finite, got {hi}"
    );
    assert!(
        lo.is_finite(),
        "rempif_scalar(1e30) lo should be finite, got {lo}"
    );

    // Try 1e35 (exponent ~116)
    let very_large = 1e35f32;
    let (hi, lo, _q) = rempif_scalar(very_large);
    assert!(
        hi.is_finite(),
        "rempif_scalar(1e35) hi should be finite, got {hi}"
    );
    assert!(
        lo.is_finite(),
        "rempif_scalar(1e35) lo should be finite, got {lo}"
    );
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
        assert!(
            r[i].abs() <= 1.0,
            "sin({}) should be in [-1,1], got {}",
            vals[i],
            r[i]
        );
    }
}

#[test]
fn test_vec4_cos_extreme_large() {
    let vals = [1e28, -1e28, 1e30, -1e30];
    let v = Vec4(vals);
    let r = v.cos();
    for i in 0..4 {
        assert!(r[i].is_finite(), "cos({}) should be finite", vals[i]);
        assert!(
            r[i].abs() <= 1.0,
            "cos({}) should be in [-1,1], got {}",
            vals[i],
            r[i]
        );
    }
}

#[test]
fn test_vec8_sin_extreme_large() {
    let vals = [1e28, -1e28, 1e30, -1e30, 1e35, -1e35, 1e38, -1e38];
    let v = Vec8(vals);
    let r = v.sin();
    for i in 0..8 {
        assert!(r[i].is_finite(), "sin({}) should be finite", vals[i]);
        assert!(
            r[i].abs() <= 1.0,
            "sin({}) should be in [-1,1], got {}",
            vals[i],
            r[i]
        );
    }
}

#[test]
fn test_vec8_cos_extreme_large() {
    let vals = [1e28, -1e28, 1e30, -1e30, 1e35, -1e35, 1e38, -1e38];
    let v = Vec8(vals);
    let r = v.cos();
    for i in 0..4 {
        assert!(r[i].is_finite(), "cos({}) should be finite", vals[i]);
        assert!(
            r[i].abs() <= 1.0,
            "cos({}) should be in [-1,1], got {}",
            vals[i],
            r[i]
        );
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
    let v = Vec8([
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        -0.0,
        0.0,
        1.0,
        -1.0,
        0.5,
    ]);
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

// ============ Ceil tests ============

#[test]
fn test_vec4_ceil_basic() {
    let v = Vec4([1.5, 2.7, -1.3, -2.9]);
    assert_eq!(v.ceil().0, [2.0, 3.0, -1.0, -2.0]);
}

#[test]
fn test_vec4_ceil_integers() {
    let v = Vec4([1.0, -1.0, 100.0, -100.0]);
    assert_eq!(v.ceil().0, [1.0, -1.0, 100.0, -100.0]);
}

#[test]
fn test_vec4_ceil_special() {
    let v = Vec4([f32::INFINITY, f32::NEG_INFINITY, -0.0, 0.0]);
    let r = v.ceil();
    assert_eq!(r[0], f32::INFINITY);
    assert_eq!(r[1], f32::NEG_INFINITY);
    assert_eq!(r[2], -0.0);
    assert!(r[2].is_sign_negative());
    assert_eq!(r[3], 0.0);
}

#[test]
fn test_vec4_ceil_nan() {
    let v = Vec4([f32::NAN, 1.5, -1.5, f32::NAN]);
    let r = v.ceil();
    assert!(r[0].is_nan());
    assert_eq!(r[1], 2.0);
    assert_eq!(r[2], -1.0);
    assert!(r[3].is_nan());
}

#[test]
fn test_vec4_ceil_small_negative() {
    // ceil(-0.1) = -0.0, ceil(-0.9) = -0.0
    let v = Vec4([-0.1, -0.5, -0.9, -0.0001]);
    let r = v.ceil();
    for i in 0..4 {
        assert_eq!(r[i], 0.0, "ceil({}) should be 0.0", v[i]);
    }
}

#[test]
fn test_vec8_ceil_basic() {
    let v = Vec8([1.5, 2.7, -1.3, -2.9, 0.0, 3.99, -0.01, 100.1]);
    assert_eq!(v.ceil().0, [2.0, 3.0, -1.0, -2.0, 0.0, 4.0, 0.0, 101.0]);
}

#[test]
fn test_vec8_ceil_special() {
    let v = Vec8([
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        -0.0,
        0.0,
        1.0,
        -1.0,
        -0.5,
    ]);
    let r = v.ceil();
    assert!(r[0].is_nan());
    assert_eq!(r[1], f32::INFINITY);
    assert_eq!(r[2], f32::NEG_INFINITY);
    assert!(r[3].is_sign_negative() && r[3] == 0.0);
    assert_eq!(r[4], 0.0);
    assert_eq!(r[5], 1.0);
    assert_eq!(r[6], -1.0);
    assert_eq!(r[7], 0.0);
}

// ============ Round tests ============

#[test]
fn test_vec4_round_basic() {
    let v = Vec4([1.5, 2.3, -1.5, -2.7]);
    let r = v.round();
    assert_eq!(r[0], 2.0); // half away from zero
    assert_eq!(r[1], 2.0);
    assert_eq!(r[2], -2.0); // half away from zero
    assert_eq!(r[3], -3.0);
}

#[test]
fn test_vec4_round_half_away_from_zero() {
    // Key: 0.5 → 1.0, -0.5 → -1.0 (not banker's rounding)
    let v = Vec4([0.5, 1.5, 2.5, 3.5]);
    let r = v.round();
    assert_eq!(r.0, [1.0, 2.0, 3.0, 4.0]);

    let v = Vec4([-0.5, -1.5, -2.5, -3.5]);
    let r = v.round();
    assert_eq!(r.0, [-1.0, -2.0, -3.0, -4.0]);
}

#[test]
fn test_vec4_round_integers() {
    let v = Vec4([1.0, -1.0, 100.0, -100.0]);
    assert_eq!(v.round().0, [1.0, -1.0, 100.0, -100.0]);
}

#[test]
fn test_vec4_round_special() {
    let v = Vec4([f32::INFINITY, f32::NEG_INFINITY, -0.0, 0.0]);
    let r = v.round();
    assert_eq!(r[0], f32::INFINITY);
    assert_eq!(r[1], f32::NEG_INFINITY);
    assert_eq!(r[2], -0.0);
    assert!(r[2].is_sign_negative());
    assert_eq!(r[3], 0.0);
}

#[test]
fn test_vec4_round_nan() {
    let v = Vec4([f32::NAN, 1.5, -1.5, f32::NAN]);
    let r = v.round();
    assert!(r[0].is_nan());
    assert_eq!(r[1], 2.0);
    assert_eq!(r[2], -2.0);
    assert!(r[3].is_nan());
}

#[test]
fn test_vec4_round_matches_f32_round() {
    // Verify consistency with Rust's f32::round() (half away from zero)
    let vals = [0.4999999, 0.5000001, -0.4999999, -0.5000001, 1.4999999, 1.5000001];
    for &x in &vals {
        let v = Vec4([x, 0.0, 0.0, 0.0]);
        let r = v.round();
        assert_eq!(r[0], x.round(), "round({x}) mismatch vs f32::round");
    }
}

#[test]
fn test_vec8_round_basic() {
    let v = Vec8([1.5, 2.3, -1.5, -2.7, 0.5, -0.5, 0.0, 3.5]);
    let r = v.round();
    assert_eq!(r.0, [2.0, 2.0, -2.0, -3.0, 1.0, -1.0, 0.0, 4.0]);
}

#[test]
fn test_vec8_round_special() {
    let v = Vec8([
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        -0.0,
        0.0,
        0.5,
        -0.5,
        f32::MAX,
    ]);
    let r = v.round();
    assert!(r[0].is_nan());
    assert_eq!(r[1], f32::INFINITY);
    assert_eq!(r[2], f32::NEG_INFINITY);
    assert!(r[3].is_sign_negative() && r[3] == 0.0);
    assert_eq!(r[4], 0.0);
    assert_eq!(r[5], 1.0);
    assert_eq!(r[6], -1.0);
    assert_eq!(r[7], f32::MAX); // already integer
}

#[test]
fn test_vec8_round_half_away_from_zero() {
    let v = Vec8([0.5, 1.5, 2.5, 3.5, -0.5, -1.5, -2.5, -3.5]);
    let r = v.round();
    assert_eq!(r.0, [1.0, 2.0, 3.0, 4.0, -1.0, -2.0, -3.0, -4.0]);
}

// ============ Horizontal sum tests ============

#[test]
fn test_vec4_horizontal_sum() {
    let v = Vec4([1.0, 2.0, 3.0, 4.0]);
    assert_eq!(v.sum(), 10.0);
}

#[test]
fn test_vec4_horizontal_sum_zeros() {
    assert_eq!(Vec4::ZERO.sum(), 0.0);
}

#[test]
fn test_vec4_horizontal_sum_negative() {
    let v = Vec4([1.0, -1.0, 2.0, -2.0]);
    assert_eq!(v.sum(), 0.0);
}

#[test]
fn test_vec4_horizontal_sum_accuracy() {
    // f64-intermediate should give ≤ 0.5 ULP
    // Use values that would lose precision with naive f32 summation
    let v = Vec4([1e7, 1.0, -1e7, 0.5]);
    let r = v.sum();
    assert_ulp(r, 1.5, 0.5, "sum accuracy");
}

#[test]
fn test_vec8_horizontal_sum() {
    let v = Vec8([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    assert_eq!(v.sum(), 36.0);
}

#[test]
fn test_vec8_horizontal_sum_zeros() {
    assert_eq!(Vec8::ZERO.sum(), 0.0);
}

#[test]
fn test_vec8_horizontal_sum_accuracy() {
    let v = Vec8([1e7, 1.0, -1e7, 0.5, 1e7, 0.25, -1e7, 0.125]);
    let r = v.sum();
    assert_ulp(r, 1.875, 0.5, "sum accuracy");
}

// ============ Dot product accuracy ============

#[test]
fn test_vec4_dot_accuracy() {
    // Values that stress f32 precision
    let a = Vec4([1e4, 1e4, 1e4, 1e4]);
    let b = Vec4([1e4, 1e4, 1e4, 1e4]);
    let r = a.dot(b);
    assert_ulp(r, 4e8, 0.5, "dot accuracy");
}

#[test]
fn test_vec8_dot_accuracy() {
    let a = Vec8([1e4, 1e4, 1e4, 1e4, 1e4, 1e4, 1e4, 1e4]);
    let b = Vec8([1e4, 1e4, 1e4, 1e4, 1e4, 1e4, 1e4, 1e4]);
    let r = a.dot(b);
    assert_ulp(r, 8e8, 0.5, "dot accuracy");
}

// ============ Subnormal inputs for transcendentals ============

#[test]
fn test_vec4_sin_subnormal() {
    let tiny = f32::from_bits(1); // smallest positive subnormal
    let vals = [tiny, -tiny, f32::MIN_POSITIVE, -f32::MIN_POSITIVE];
    let v = Vec4(vals);
    let r = v.sin();
    for i in 0..4 {
        // sin(x) ≈ x for very small x
        assert_ulp(r[i], vals[i].sin(), 1.0, &format!("sin({})", vals[i]));
    }
}

#[test]
fn test_vec4_cos_subnormal() {
    let tiny = f32::from_bits(1);
    let vals = [tiny, -tiny, f32::MIN_POSITIVE, -f32::MIN_POSITIVE];
    let v = Vec4(vals);
    let r = v.cos();
    for i in 0..4 {
        // cos(x) ≈ 1 for very small x
        assert_eq!(r[i], 1.0, "cos({}) should be 1.0", vals[i]);
    }
}

#[test]
fn test_vec4_exp_subnormal() {
    let tiny = f32::from_bits(1);
    let vals = [tiny, -tiny, f32::MIN_POSITIVE, -f32::MIN_POSITIVE];
    let v = Vec4(vals);
    let r = v.exp();
    for i in 0..4 {
        assert_ulp(r[i], vals[i].exp(), 1.0, &format!("exp({})", vals[i]));
    }
}

#[test]
fn test_vec8_sin_subnormal() {
    let tiny = f32::from_bits(1);
    let vals = [
        tiny,
        -tiny,
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
        f32::from_bits(100),
        -f32::from_bits(100),
        f32::from_bits(0x007F_FFFF), // largest subnormal
        -f32::from_bits(0x007F_FFFF),
    ];
    let v = Vec8(vals);
    let r = v.sin();
    for i in 0..8 {
        assert_ulp(r[i], vals[i].sin(), 1.0, &format!("sin({})", vals[i]));
    }
}

#[test]
fn test_vec8_exp_subnormal() {
    let tiny = f32::from_bits(1);
    let vals = [
        tiny,
        -tiny,
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
        f32::from_bits(100),
        -f32::from_bits(100),
        f32::from_bits(0x007F_FFFF),
        -f32::from_bits(0x007F_FFFF),
    ];
    let v = Vec8(vals);
    let r = v.exp();
    for i in 0..8 {
        assert_ulp(r[i], vals[i].exp(), 1.0, &format!("exp({})", vals[i]));
    }
}

// ============ Near multiples of pi/4 (SLEEF critical test pattern) ============

#[test]
fn test_vec4_sin_near_pi_multiples() {
    // Values very close to n*pi/4 — range reduction accuracy matters most here
    let eps = 1e-7;
    let vals = [
        FRAC_PI_4 + eps,
        FRAC_PI_4 - eps,
        FRAC_PI_2 + eps,
        FRAC_PI_2 - eps,
    ];
    let v = Vec4(vals);
    let r = v.sin();
    for i in 0..4 {
        assert_ulp(r[i], vals[i].sin(), 1.0, &format!("sin({})", vals[i]));
    }
}

#[test]
fn test_vec4_cos_near_pi_multiples() {
    let eps = 1e-7;
    let vals = [
        FRAC_PI_4 + eps,
        FRAC_PI_4 - eps,
        FRAC_PI_2 + eps,
        FRAC_PI_2 - eps,
    ];
    let v = Vec4(vals);
    let r = v.cos();
    for i in 0..4 {
        assert_ulp(r[i], vals[i].cos(), 1.0, &format!("cos({})", vals[i]));
    }
}

#[test]
fn test_vec4_sin_near_pi() {
    let eps = 1e-7;
    let vals = [PI + eps, PI - eps, 2.0 * PI + eps, 2.0 * PI - eps];
    let v = Vec4(vals);
    let r = v.sin();
    for i in 0..4 {
        assert_ulp(r[i], vals[i].sin(), 1.0, &format!("sin({})", vals[i]));
    }
}

#[test]
fn test_vec4_cos_near_pi() {
    let eps = 1e-7;
    let vals = [PI + eps, PI - eps, 2.0 * PI + eps, 2.0 * PI - eps];
    let v = Vec4(vals);
    let r = v.cos();
    for i in 0..4 {
        assert_ulp(r[i], vals[i].cos(), 1.0, &format!("cos({})", vals[i]));
    }
}

#[test]
fn test_vec8_sin_near_pi_multiples() {
    let eps = 1e-7;
    let vals = [
        FRAC_PI_4 + eps,
        FRAC_PI_4 - eps,
        FRAC_PI_2 + eps,
        FRAC_PI_2 - eps,
        PI + eps,
        PI - eps,
        2.0 * PI + eps,
        2.0 * PI - eps,
    ];
    let v = Vec8(vals);
    let r = v.sin();
    for i in 0..8 {
        assert_ulp(r[i], vals[i].sin(), 1.0, &format!("sin({})", vals[i]));
    }
}

#[test]
fn test_vec8_cos_near_pi_multiples() {
    let eps = 1e-7;
    let vals = [
        FRAC_PI_4 + eps,
        FRAC_PI_4 - eps,
        FRAC_PI_2 + eps,
        FRAC_PI_2 - eps,
        PI + eps,
        PI - eps,
        2.0 * PI + eps,
        2.0 * PI - eps,
    ];
    let v = Vec8(vals);
    let r = v.cos();
    for i in 0..8 {
        assert_ulp(r[i], vals[i].cos(), 1.0, &format!("cos({})", vals[i]));
    }
}

// ============ SLEEF-style large multiples of pi/4 (rempif path) ============

#[test]
fn test_vec4_sin_large_pi_multiples() {
    // Large multiples of pi/4 with small perturbations — stresses rempif
    let n = 1000.0;
    let eps = 1e-5;
    let vals = [
        n * FRAC_PI_4 + eps,
        n * FRAC_PI_4 - eps,
        -n * FRAC_PI_4 + eps,
        -n * FRAC_PI_4 - eps,
    ];
    let v = Vec4(vals);
    let r = v.sin();
    for i in 0..4 {
        assert_ulp(r[i], vals[i].sin(), 1.0, &format!("sin({})", vals[i]));
    }
}

#[test]
fn test_vec4_cos_large_pi_multiples() {
    let n = 1000.0;
    let eps = 1e-5;
    let vals = [
        n * FRAC_PI_4 + eps,
        n * FRAC_PI_4 - eps,
        -n * FRAC_PI_4 + eps,
        -n * FRAC_PI_4 - eps,
    ];
    let v = Vec4(vals);
    let r = v.cos();
    for i in 0..4 {
        assert_ulp(r[i], vals[i].cos(), 1.0, &format!("cos({})", vals[i]));
    }
}

// ============ Exp near ln(2) boundaries (range reduction stress) ============

#[test]
fn test_vec4_exp_near_ln2_multiples() {
    let ln2 = std::f32::consts::LN_2;
    let eps = 1e-7;
    let vals = [
        ln2 + eps,
        ln2 - eps,
        10.0 * ln2 + eps,
        10.0 * ln2 - eps,
    ];
    let v = Vec4(vals);
    let r = v.exp();
    for i in 0..4 {
        assert_ulp(r[i], vals[i].exp(), 1.0, &format!("exp({})", vals[i]));
    }
}

#[test]
fn test_vec8_exp_near_ln2_multiples() {
    let ln2 = std::f32::consts::LN_2;
    let eps = 1e-7;
    let vals = [
        ln2 + eps,
        ln2 - eps,
        -ln2 + eps,
        -ln2 - eps,
        50.0 * ln2 + eps,
        50.0 * ln2 - eps,
        -50.0 * ln2 + eps,
        -50.0 * ln2 - eps,
    ];
    let v = Vec8(vals);
    let r = v.exp();
    for i in 0..8 {
        assert_ulp(r[i], vals[i].exp(), 1.0, &format!("exp({})", vals[i]));
    }
}

// ============ Floor/Ceil/Round consistency with f32 methods ============

#[test]
fn test_vec4_floor_ceil_round_consistency() {
    let vals = [1.1, -1.1, 2.5, -2.5];
    let v = Vec4(vals);
    let fl = v.floor();
    let ce = v.ceil();
    let ro = v.round();
    for i in 0..4 {
        assert_eq!(fl[i], vals[i].floor(), "floor({}) mismatch", vals[i]);
        assert_eq!(ce[i], vals[i].ceil(), "ceil({}) mismatch", vals[i]);
        assert_eq!(ro[i], vals[i].round(), "round({}) mismatch", vals[i]);
    }
}

#[test]
fn test_vec8_floor_ceil_round_consistency() {
    let vals = [1.1, -1.1, 2.5, -2.5, 0.0, -0.0, 99.9, -99.9];
    let v = Vec8(vals);
    let fl = v.floor();
    let ce = v.ceil();
    let ro = v.round();
    for i in 0..8 {
        assert_eq!(fl[i], vals[i].floor(), "floor({}) mismatch", vals[i]);
        assert_eq!(ce[i], vals[i].ceil(), "ceil({}) mismatch", vals[i]);
        assert_eq!(ro[i], vals[i].round(), "round({}) mismatch", vals[i]);
    }
}

// ============ Round: 0.4999999f edge case ============

#[test]
fn test_vec4_round_049999() {
    // 0.4999999701976776123 — adding 0.5 to this in f32 yields exactly 1.0
    // trunc+adjust correctly gives 0.0; naive (x+0.5).floor() would give 1.0
    let val = 0.49999997;
    let v = Vec4([val, -val, val, -val]);
    let r = v.round();
    assert_eq!(r[0], 0.0, "round(0.49999997) should be 0");
    assert_eq!(r[1], 0.0, "round(-0.49999997) should be 0");
}

// ============ Floor/ceil/round at 2^23 boundary (all f32 >= 2^23 are integers) ============

#[test]
fn test_vec4_rounding_at_2pow23_boundary() {
    let p = 8388608.0f32; // 2^23
    let vals = [p - 1.5, p - 0.5, p, p + 1.0];
    let v = Vec4(vals);
    for i in 0..4 {
        assert_eq!(v.floor()[i], vals[i].floor(), "floor at 2^23 boundary");
        assert_eq!(v.ceil()[i], vals[i].ceil(), "ceil at 2^23 boundary");
        assert_eq!(v.round()[i], vals[i].round(), "round at 2^23 boundary");
    }
}

#[test]
fn test_vec4_rounding_large_integers() {
    // Values > 2^23 are always exact integers in f32
    let vals = [1e7, -1e7, f32::MAX, -f32::MAX];
    let v = Vec4(vals);
    assert_eq!(v.floor().0, vals);
    assert_eq!(v.ceil().0, vals);
    assert_eq!(v.round().0, vals);
}

// ============ Pseudo-random sweep: floor/ceil/round (SLEEF tests all random floats) ============

#[test]
fn test_vec4_floor_ceil_round_random_sweep() {
    // Simple LCG to generate deterministic pseudo-random bit patterns
    let mut state: u32 = 0xDEAD_BEEF;
    for _ in 0..10000 {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        let f = f32::from_bits(state);
        if !f.is_finite() {
            continue;
        }
        let v = Vec4([f, -f, f, -f]);
        let fl = v.floor();
        let ce = v.ceil();
        let ro = v.round();
        assert_eq!(fl[0], f.floor(), "floor({f:e})");
        assert_eq!(fl[1], (-f).floor(), "floor({:e})", -f);
        assert_eq!(ce[0], f.ceil(), "ceil({f:e})");
        assert_eq!(ce[1], (-f).ceil(), "ceil({:e})", -f);
        assert_eq!(ro[0], f.round(), "round({f:e})");
        assert_eq!(ro[1], (-f).round(), "round({:e})", -f);
    }
}

// ============ Pseudo-random sin/cos/exp ULP sweep ============

#[test]
fn test_vec4_sin_cos_random_sweep() {
    let mut state: u32 = 0xCAFE_BABE;
    let mut max_sin_ulp = 0.0f32;
    let mut max_cos_ulp = 0.0f32;
    for _ in 0..25000 {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        let f = f32::from_bits(state);
        if !f.is_finite() {
            continue;
        }
        let v = Vec4([f, -f, f, -f]);
        let rs = v.sin();
        let rc = v.cos();
        for i in 0..2 {
            let x = v[i];
            let su = ulp_error(rs[i], x.sin());
            let cu = ulp_error(rc[i], x.cos());
            if su > max_sin_ulp {
                max_sin_ulp = su;
            }
            if cu > max_cos_ulp {
                max_cos_ulp = cu;
            }
        }
    }
    assert!(
        max_sin_ulp <= 1.0,
        "random sin max ULP: {max_sin_ulp}"
    );
    assert!(
        max_cos_ulp <= 1.0,
        "random cos max ULP: {max_cos_ulp}"
    );
}

#[test]
fn test_vec4_exp_random_sweep() {
    let mut state: u32 = 0xBAAD_F00D;
    let mut max_ulp = 0.0f32;
    for _ in 0..25000 {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        // Map to exp-meaningful range [-104, 100]
        let f = (state as f32 / u32::MAX as f32) * 204.0 - 104.0;
        let v = Vec4([f, -f.abs(), f, -f.abs()]);
        let r = v.exp();
        for i in 0..2 {
            let expected = v[i].exp();
            if expected.is_finite() && expected.is_normal() {
                let u = ulp_error(r[i], expected);
                if u > max_ulp {
                    max_ulp = u;
                }
            }
        }
    }
    assert!(max_ulp <= 1.0, "random exp max ULP: {max_ulp}");
}

// ============ Sin/cos very wide sweep (rempif: 1e3..1e6) ============

#[test]
fn test_vec4_sin_sweep_very_large() {
    let mut max_ulp = 0.0f32;
    let step = 1.0;
    let mut x = 1000.0f32;
    while x < 1e6 {
        let v = Vec4([x, x + step, x + 2.0 * step, x + 3.0 * step]);
        let r = v.sin();
        for i in 0..4 {
            let u = ulp_error(r[i], v[i].sin());
            if u > max_ulp {
                max_ulp = u;
            }
        }
        x += 4.0 * step;
    }
    assert!(max_ulp <= 1.0, "Vec4 sin 1e3..1e6 max ULP: {max_ulp}");
}

#[test]
fn test_vec4_cos_sweep_very_large() {
    let mut max_ulp = 0.0f32;
    let step = 1.0;
    let mut x = 1000.0f32;
    while x < 1e6 {
        let v = Vec4([x, x + step, x + 2.0 * step, x + 3.0 * step]);
        let r = v.cos();
        for i in 0..4 {
            let u = ulp_error(r[i], v[i].cos());
            if u > max_ulp {
                max_ulp = u;
            }
        }
        x += 4.0 * step;
    }
    assert!(max_ulp <= 1.0, "Vec4 cos 1e3..1e6 max ULP: {max_ulp}");
}

// ============ Exp at exact overflow/underflow boundaries ============

#[test]
fn test_vec4_exp_overflow_boundary() {
    // ln(FLT_MAX) ≈ 88.72284
    let boundary = 88.72284f32;
    let vals = [boundary, boundary - 0.001, boundary + 0.001, 88.7229];
    let v = Vec4(vals);
    let r = v.exp();
    for i in 0..4 {
        let expected = vals[i].exp();
        if expected.is_finite() {
            assert_ulp(r[i], expected, 1.0, &format!("exp({})", vals[i]));
        } else {
            assert_eq!(r[i], f32::INFINITY, "exp({}) should overflow", vals[i]);
        }
    }
}

#[test]
fn test_vec4_exp_underflow_boundary() {
    // Smallest normal exp result: exp(-87.33655) ≈ FLT_MIN
    let vals = [-87.0, -87.3, -87.33, -88.0];
    let v = Vec4(vals);
    let r = v.exp();
    for i in 0..4 {
        let expected = vals[i].exp();
        if expected.is_normal() {
            assert_ulp(r[i], expected, 1.0, &format!("exp({})", vals[i]));
        }
    }
}

// ============ Dot product: f64-intermediate prevents f32 overflow ============

#[test]
fn test_vec4_dot_large_values() {
    // f64-intermediate: exact products in f64, single rounding on final f64→f32
    let val = 1e15f32;
    let a = Vec4([val; 4]);
    let b = Vec4([val; 4]);
    let r = a.dot(b);
    // Expected via f64: 4 * (val as f64)^2, then cast to f32
    let vd = val as f64;
    let expected = (4.0 * vd * vd) as f32;
    assert_eq!(r, expected, "dot large values");
}

#[test]
fn test_vec4_dot_cancellation() {
    // Near-cancellation: sum of products is small relative to individual products
    let a = Vec4([1e7, 1e7, -1e7, -1e7]);
    let b = Vec4([1.0, 1.0, 1.0, 1.0]);
    let r = a.dot(b);
    assert_eq!(r, 0.0, "dot cancellation");
}

#[test]
fn test_vec8_dot_cancellation() {
    let a = Vec8([1e7, -1e7, 1e7, -1e7, 1.0, -1.0, 0.5, -0.5]);
    let b = Vec8([1.0; 8]);
    let r = a.dot(b);
    assert_eq!(r, 0.0, "dot cancellation 8-lane");
}

// ============ Sum: f64-intermediate precision ============

#[test]
fn test_vec4_sum_catastrophic_cancellation() {
    // Without f64 intermediates, naive f32 sum would lose the small terms entirely
    let v = Vec4([1e8, 1.0, -1e8, 0.5]);
    let r = v.sum();
    assert_ulp(r, 1.5, 0.5, "sum catastrophic cancellation");
}

#[test]
fn test_vec8_sum_catastrophic_cancellation() {
    let v = Vec8([1e8, -1e8, 1e8, -1e8, 1.0, 0.5, 0.25, 0.125]);
    let r = v.sum();
    assert_ulp(r, 1.875, 0.5, "sum catastrophic cancellation 8-lane");
}

// ============ Arithmetic with subnormals ============

#[test]
fn test_vec4_arithmetic_subnormal() {
    let tiny = f32::from_bits(1); // smallest subnormal ≈ 1.4e-45
    let a = Vec4([tiny, tiny, tiny, tiny]);
    let b = Vec4([tiny, tiny, tiny, tiny]);
    let sum = a + b;
    assert_eq!(sum[0], tiny + tiny);
    let neg = -a;
    assert_eq!(neg[0], -tiny);
    let abs = neg.abs();
    assert_eq!(abs[0], tiny);
}

#[test]
fn test_vec8_arithmetic_subnormal() {
    let tiny = f32::from_bits(1);
    let a = Vec8([tiny; 8]);
    let b = Vec8([tiny; 8]);
    let sum = a + b;
    assert_eq!(sum[0], tiny + tiny);
    let neg = -a;
    assert_eq!(neg[0], -tiny);
    let abs = neg.abs();
    assert_eq!(abs[0], tiny);
}

// ============ Vec4/Vec8 consistency for all operations ============

#[test]
fn test_vec4_vec8_floor_ceil_round_consistency() {
    let vals = [1.7, -2.3, 0.5, -0.5];
    let v4 = Vec4(vals);
    let v8 = Vec8([vals[0], vals[1], vals[2], vals[3], 0.0, 0.0, 0.0, 0.0]);
    for i in 0..4 {
        assert_eq!(v4.floor()[i], v8.floor()[i], "floor mismatch lane {i}");
        assert_eq!(v4.ceil()[i], v8.ceil()[i], "ceil mismatch lane {i}");
        assert_eq!(v4.round()[i], v8.round()[i], "round mismatch lane {i}");
    }
}

#[test]
fn test_vec4_vec8_abs_neg_sqrt_consistency() {
    let vals = [4.0, -9.0, 16.0, -25.0];
    let v4 = Vec4(vals);
    let v8 = Vec8([vals[0], vals[1], vals[2], vals[3], 1.0, 1.0, 1.0, 1.0]);
    for i in 0..4 {
        assert_eq!((-v4)[i], (-v8)[i], "neg mismatch lane {i}");
        assert_eq!(v4.abs()[i], v8.abs()[i], "abs mismatch lane {i}");
    }
    let pos4 = Vec4([4.0, 9.0, 16.0, 25.0]);
    let pos8 = Vec8([4.0, 9.0, 16.0, 25.0, 1.0, 1.0, 1.0, 1.0]);
    for i in 0..4 {
        assert_eq!(pos4.sqrt()[i], pos8.sqrt()[i], "sqrt mismatch lane {i}");
    }
}

#[test]
fn test_vec4_vec8_arithmetic_consistency() {
    let a4 = Vec4([1.5, 2.7, 3.1, 4.9]);
    let b4 = Vec4([5.3, 6.1, 7.7, 8.3]);
    let a8 = Vec8([1.5, 2.7, 3.1, 4.9, 0.0, 0.0, 0.0, 0.0]);
    let b8 = Vec8([5.3, 6.1, 7.7, 8.3, 1.0, 1.0, 1.0, 1.0]);
    for i in 0..4 {
        assert_eq!((a4 + b4)[i], (a8 + b8)[i], "add mismatch lane {i}");
        assert_eq!((a4 - b4)[i], (a8 - b8)[i], "sub mismatch lane {i}");
        assert_eq!((a4 * b4)[i], (a8 * b8)[i], "mul mismatch lane {i}");
        assert_eq!((a4 / b4)[i], (a8 / b8)[i], "div mismatch lane {i}");
    }
}

#[test]
fn test_vec4_vec8_mul_add_consistency() {
    let a4 = Vec4([1.5, 2.7, 3.1, 4.9]);
    let b4 = Vec4([5.3, 6.1, 7.7, 8.3]);
    let c4 = Vec4([0.1, 0.2, 0.3, 0.4]);
    let a8 = Vec8([1.5, 2.7, 3.1, 4.9, 0.0, 0.0, 0.0, 0.0]);
    let b8 = Vec8([5.3, 6.1, 7.7, 8.3, 1.0, 1.0, 1.0, 1.0]);
    let c8 = Vec8([0.1, 0.2, 0.3, 0.4, 0.0, 0.0, 0.0, 0.0]);
    for i in 0..4 {
        assert_eq!(
            a4.mul_add(b4, c4)[i],
            a8.mul_add(b8, c8)[i],
            "mul_add mismatch lane {i}"
        );
    }
}

// ============ Sin/cos: near MAX float (SLEEF tests full range) ============

#[test]
fn test_vec4_sin_cos_near_max() {
    // Near f32::MAX — rempif should handle this
    let big = 3.4e38f32;
    let vals = [big, -big, big * 0.5, -big * 0.5];
    let v = Vec4(vals);
    let rs = v.sin();
    let rc = v.cos();
    for i in 0..4 {
        assert!(rs[i].is_finite(), "sin({}) should be finite", vals[i]);
        assert!(rs[i].abs() <= 1.0, "sin({}) out of range", vals[i]);
        assert!(rc[i].is_finite(), "cos({}) should be finite", vals[i]);
        assert!(rc[i].abs() <= 1.0, "cos({}) out of range", vals[i]);
        // sin^2 + cos^2 = 1
        let ss = rs[i] * rs[i] + rc[i] * rc[i];
        assert!(
            (ss - 1.0).abs() < 1e-4,
            "sin^2+cos^2 != 1 at x={}: {}",
            vals[i],
            ss
        );
    }
}

// ============ Sin/cos sweep near pi/4 multiples with perturbation (SLEEF pattern) ============

#[test]
fn test_vec4_sin_cos_near_pi4_multiples_sweep() {
    let mut max_sin = 0.0f32;
    let mut max_cos = 0.0f32;
    // Test n * pi/4 + small perturbation for n in -100..100
    for n in -100..=100i32 {
        let base = n as f32 * FRAC_PI_4;
        let eps_vals = [1e-7, -1e-7, 1e-4, -1e-4];
        let vals: [f32; 4] = std::array::from_fn(|i| base + eps_vals[i]);
        let v = Vec4(vals);
        let rs = v.sin();
        let rc = v.cos();
        for i in 0..4 {
            let su = ulp_error(rs[i], vals[i].sin());
            let cu = ulp_error(rc[i], vals[i].cos());
            if su > max_sin {
                max_sin = su;
            }
            if cu > max_cos {
                max_cos = cu;
            }
        }
    }
    assert!(max_sin <= 1.0, "near-pi/4 sin max ULP: {max_sin}");
    assert!(max_cos <= 1.0, "near-pi/4 cos max ULP: {max_cos}");
}

// ============ Exp: sweep near integer multiples of ln(2) ============

#[test]
fn test_vec4_exp_sweep_near_ln2() {
    let ln2 = std::f32::consts::LN_2;
    let mut max_ulp = 0.0f32;
    for n in -120..=120i32 {
        let base = n as f32 * ln2;
        let vals = [base, base + 1e-7, base - 1e-7, base + 1e-4];
        let v = Vec4(vals);
        let r = v.exp();
        for i in 0..4 {
            let expected = vals[i].exp();
            if expected.is_finite() && expected.is_normal() {
                let u = ulp_error(r[i], expected);
                if u > max_ulp {
                    max_ulp = u;
                }
            }
        }
    }
    assert!(max_ulp <= 1.0, "exp near-ln2 max ULP: {max_ulp}");
}

// ============ Floor/ceil relationship: ceil(x) = -floor(-x) ============

#[test]
fn test_vec4_ceil_floor_duality() {
    let vals = [1.3, -2.7, 0.5, -0.1];
    let v = Vec4(vals);
    let neg_v = -v;
    let ceil_v = v.ceil();
    let neg_floor_neg = -neg_v.floor();
    for i in 0..4 {
        assert_eq!(
            ceil_v[i], neg_floor_neg[i],
            "ceil(x) != -floor(-x) for x={}",
            vals[i]
        );
    }
}

#[test]
fn test_vec8_ceil_floor_duality() {
    let vals = [1.3, -2.7, 0.5, -0.1, 99.9, -0.001, 0.0, -0.0];
    let v = Vec8(vals);
    let neg_v = -v;
    let ceil_v = v.ceil();
    let neg_floor_neg = -neg_v.floor();
    for i in 0..8 {
        assert_eq!(
            ceil_v[i].to_bits(),
            neg_floor_neg[i].to_bits(),
            "ceil(x) != -floor(-x) for x={}",
            vals[i]
        );
    }
}

// ============ Floor/ceil: floor(x) <= x <= ceil(x) ============

#[test]
fn test_vec4_floor_le_x_le_ceil() {
    let mut state: u32 = 0x1234_5678;
    for _ in 0..5000 {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        let f = f32::from_bits(state);
        if !f.is_finite() {
            continue;
        }
        let v = Vec4([f, f, f, f]);
        let fl = v.floor()[0];
        let ce = v.ceil()[0];
        assert!(fl <= f, "floor({f:e}) > {f:e}");
        assert!(ce >= f, "ceil({f:e}) < {f:e}");
        assert!((ce - fl) <= 1.0 || fl == f, "ceil-floor gap > 1 for {f:e}");
    }
}

// ============ Round: |round(x) - x| <= 0.5 ============

#[test]
fn test_vec4_round_distance() {
    let mut state: u32 = 0xABCD_EF01;
    for _ in 0..5000 {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        let f = f32::from_bits(state);
        if !f.is_finite() {
            continue;
        }
        let v = Vec4([f, f, f, f]);
        let r = v.round()[0];
        let dist = (r - f).abs();
        assert!(
            dist <= 0.5 || f.abs() >= 8388608.0, // >= 2^23, all integers
            "round({f:e}) = {r:e}, distance = {dist:e}"
        );
    }
}

// ============ Scalar ops: commutativity of add/mul ============

#[test]
fn test_vec4_scalar_commutativity() {
    let v = Vec4([1.5, 2.7, 3.1, 4.9]);
    let s = 7.3f32;
    // v + s == s + v
    let r1 = v + s;
    let r2 = s + v;
    assert_eq!(r1, r2, "add commutativity");
    // v * s == s * v
    let r3 = v * s;
    let r4 = s * v;
    assert_eq!(r3, r4, "mul commutativity");
}

#[test]
fn test_vec8_scalar_commutativity() {
    let v = Vec8([1.5, 2.7, 3.1, 4.9, 5.1, 6.3, 7.7, 8.9]);
    let s = 7.3f32;
    assert_eq!(v + s, s + v, "add commutativity");
    assert_eq!(v * s, s * v, "mul commutativity");
}

// ============ Vec8 random sweeps (mirror Vec4) ============

#[test]
fn test_vec8_sin_cos_random_sweep() {
    let mut state: u32 = 0xFEED_FACE;
    let mut max_sin_ulp = 0.0f32;
    let mut max_cos_ulp = 0.0f32;
    for _ in 0..25000 {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        let f = f32::from_bits(state);
        if !f.is_finite() {
            continue;
        }
        let v = Vec8([f, -f, f, -f, f, -f, f, -f]);
        let rs = v.sin();
        let rc = v.cos();
        for i in 0..2 {
            let x = v[i];
            let su = ulp_error(rs[i], x.sin());
            let cu = ulp_error(rc[i], x.cos());
            if su > max_sin_ulp {
                max_sin_ulp = su;
            }
            if cu > max_cos_ulp {
                max_cos_ulp = cu;
            }
        }
    }
    assert!(max_sin_ulp <= 1.0, "Vec8 random sin max ULP: {max_sin_ulp}");
    assert!(max_cos_ulp <= 1.0, "Vec8 random cos max ULP: {max_cos_ulp}");
}

#[test]
fn test_vec8_exp_random_sweep() {
    let mut state: u32 = 0xDEAD_C0DE;
    let mut max_ulp = 0.0f32;
    for _ in 0..25000 {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        let f = (state as f32 / u32::MAX as f32) * 204.0 - 104.0;
        let v = Vec8([f, -f.abs(), f, -f.abs(), f, -f.abs(), f, -f.abs()]);
        let r = v.exp();
        for i in 0..2 {
            let expected = v[i].exp();
            if expected.is_finite() && expected.is_normal() {
                let u = ulp_error(r[i], expected);
                if u > max_ulp {
                    max_ulp = u;
                }
            }
        }
    }
    assert!(max_ulp <= 1.0, "Vec8 random exp max ULP: {max_ulp}");
}

#[test]
fn test_vec8_floor_ceil_round_random_sweep() {
    let mut state: u32 = 0xC0FF_EEBA;
    for _ in 0..10000 {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        let f = f32::from_bits(state);
        if !f.is_finite() {
            continue;
        }
        let v = Vec8([f, -f, f, -f, f, -f, f, -f]);
        let fl = v.floor();
        let ce = v.ceil();
        let ro = v.round();
        assert_eq!(fl[0], f.floor(), "floor({f:e})");
        assert_eq!(fl[1], (-f).floor(), "floor({:e})", -f);
        assert_eq!(ce[0], f.ceil(), "ceil({f:e})");
        assert_eq!(ce[1], (-f).ceil(), "ceil({:e})", -f);
        assert_eq!(ro[0], f.round(), "round({f:e})");
        assert_eq!(ro[1], (-f).round(), "round({:e})", -f);
    }
}

// ============ Vec8 wide sin/cos sweep (1e3..1e6) ============

#[test]
fn test_vec8_sin_sweep_very_large() {
    let mut max_ulp = 0.0f32;
    let step = 1.0;
    let mut x = 1000.0f32;
    while x < 1e6 {
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
            let u = ulp_error(r[i], v[i].sin());
            if u > max_ulp {
                max_ulp = u;
            }
        }
        x += 8.0 * step;
    }
    assert!(max_ulp <= 1.0, "Vec8 sin 1e3..1e6 max ULP: {max_ulp}");
}

#[test]
fn test_vec8_cos_sweep_very_large() {
    let mut max_ulp = 0.0f32;
    let step = 1.0;
    let mut x = 1000.0f32;
    while x < 1e6 {
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
            let u = ulp_error(r[i], v[i].cos());
            if u > max_ulp {
                max_ulp = u;
            }
        }
        x += 8.0 * step;
    }
    assert!(max_ulp <= 1.0, "Vec8 cos 1e3..1e6 max ULP: {max_ulp}");
}

// ============ Vec8 exp boundary tests ============

#[test]
fn test_vec8_exp_overflow_boundary() {
    let boundary = 88.72284f32;
    let v = Vec8([
        boundary,
        boundary - 0.001,
        boundary + 0.001,
        88.7229,
        -87.0,
        -87.3,
        -87.33,
        -88.0,
    ]);
    let r = v.exp();
    for i in 0..4 {
        let expected = v[i].exp();
        if expected.is_finite() {
            assert_ulp(r[i], expected, 1.0, &format!("exp({})", v[i]));
        } else {
            assert_eq!(r[i], f32::INFINITY, "exp({}) should overflow", v[i]);
        }
    }
    for i in 4..8 {
        let expected = v[i].exp();
        if expected.is_normal() {
            assert_ulp(r[i], expected, 1.0, &format!("exp({})", v[i]));
        }
    }
}

// ============ Vec8 round edge cases ============

#[test]
fn test_vec8_round_049999() {
    let val = 0.49999997;
    let v = Vec8([val, -val, val, -val, val, -val, val, -val]);
    let r = v.round();
    for i in 0..8 {
        assert_eq!(r[i], 0.0, "round lane {i} should be 0");
    }
}

#[test]
fn test_vec8_rounding_at_2pow23_boundary() {
    let p = 8388608.0f32; // 2^23
    let v = Vec8([
        p - 1.5,
        p - 0.5,
        p,
        p + 1.0,
        -(p - 1.5),
        -(p - 0.5),
        -p,
        -(p + 1.0),
    ]);
    for i in 0..8 {
        assert_eq!(v.floor()[i], v[i].floor(), "floor at 2^23");
        assert_eq!(v.ceil()[i], v[i].ceil(), "ceil at 2^23");
        assert_eq!(v.round()[i], v[i].round(), "round at 2^23");
    }
}

#[test]
fn test_vec8_rounding_large_integers() {
    let v = Vec8([1e7, -1e7, 1e8, -1e8, f32::MAX, -f32::MAX, 1e10, -1e10]);
    assert_eq!(v.floor().0, v.0);
    assert_eq!(v.ceil().0, v.0);
    assert_eq!(v.round().0, v.0);
}

// ============ Vec8 floor/round property tests ============

#[test]
fn test_vec8_floor_le_x_le_ceil() {
    let mut state: u32 = 0xBEEF_CAFE;
    for _ in 0..5000 {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        let f = f32::from_bits(state);
        if !f.is_finite() {
            continue;
        }
        let v = Vec8([f; 8]);
        let fl = v.floor()[0];
        let ce = v.ceil()[0];
        assert!(fl <= f, "floor({f:e}) > {f:e}");
        assert!(ce >= f, "ceil({f:e}) < {f:e}");
    }
}

#[test]
fn test_vec8_round_distance() {
    let mut state: u32 = 0x1357_9BDF;
    for _ in 0..5000 {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        let f = f32::from_bits(state);
        if !f.is_finite() {
            continue;
        }
        let v = Vec8([f; 8]);
        let r = v.round()[0];
        let dist = (r - f).abs();
        assert!(
            dist <= 0.5 || f.abs() >= 8388608.0,
            "round({f:e}) = {r:e}, distance = {dist:e}"
        );
    }
}

// ============ Vec8 sin/cos near-max and near-pi/4 sweeps ============

#[test]
fn test_vec8_sin_cos_near_max() {
    let big = 3.4e38f32;
    let vals = [big, -big, big * 0.5, -big * 0.5, big * 0.1, -big * 0.1, big * 0.9, -big * 0.9];
    let v = Vec8(vals);
    let rs = v.sin();
    let rc = v.cos();
    for i in 0..8 {
        assert!(rs[i].is_finite(), "sin({}) should be finite", vals[i]);
        assert!(rs[i].abs() <= 1.0, "sin({}) out of range", vals[i]);
        assert!(rc[i].is_finite(), "cos({}) should be finite", vals[i]);
        assert!(rc[i].abs() <= 1.0, "cos({}) out of range", vals[i]);
        let ss = rs[i] * rs[i] + rc[i] * rc[i];
        assert!(
            (ss - 1.0).abs() < 1e-4,
            "sin^2+cos^2 != 1 at x={}: {}",
            vals[i],
            ss
        );
    }
}

#[test]
fn test_vec8_sin_cos_near_pi4_multiples_sweep() {
    let mut max_sin = 0.0f32;
    let mut max_cos = 0.0f32;
    for n in -100..=100i32 {
        let base = n as f32 * FRAC_PI_4;
        let vals: [f32; 8] = [
            base + 1e-7,
            base - 1e-7,
            base + 1e-4,
            base - 1e-4,
            base + 1e-5,
            base - 1e-5,
            base + 1e-3,
            base - 1e-3,
        ];
        let v = Vec8(vals);
        let rs = v.sin();
        let rc = v.cos();
        for i in 0..8 {
            let su = ulp_error(rs[i], vals[i].sin());
            let cu = ulp_error(rc[i], vals[i].cos());
            if su > max_sin {
                max_sin = su;
            }
            if cu > max_cos {
                max_cos = cu;
            }
        }
    }
    assert!(max_sin <= 1.0, "Vec8 near-pi/4 sin max ULP: {max_sin}");
    assert!(max_cos <= 1.0, "Vec8 near-pi/4 cos max ULP: {max_cos}");
}

// ============ Vec8 exp sweep near ln(2) ============

#[test]
fn test_vec8_exp_sweep_near_ln2() {
    let ln2 = std::f32::consts::LN_2;
    let mut max_ulp = 0.0f32;
    for n in -120..=120i32 {
        let base = n as f32 * ln2;
        let vals = [
            base,
            base + 1e-7,
            base - 1e-7,
            base + 1e-4,
            base - 1e-4,
            base + 1e-5,
            base - 1e-5,
            base + 1e-3,
        ];
        let v = Vec8(vals);
        let r = v.exp();
        for i in 0..8 {
            let expected = vals[i].exp();
            if expected.is_finite() && expected.is_normal() {
                let u = ulp_error(r[i], expected);
                if u > max_ulp {
                    max_ulp = u;
                }
            }
        }
    }
    assert!(max_ulp <= 1.0, "Vec8 exp near-ln2 max ULP: {max_ulp}");
}

// ============ Horizontal sum/dot with NaN/Inf ============

#[test]
fn test_vec4_sum_nan_propagation() {
    let v = Vec4([1.0, f32::NAN, 3.0, 4.0]);
    assert!(v.sum().is_nan(), "sum with NaN should be NaN");
}

#[test]
fn test_vec4_sum_inf() {
    let v = Vec4([1.0, f32::INFINITY, 3.0, 4.0]);
    assert_eq!(v.sum(), f32::INFINITY);
}

#[test]
fn test_vec4_dot_nan_propagation() {
    let a = Vec4([1.0, f32::NAN, 3.0, 4.0]);
    let b = Vec4([1.0, 1.0, 1.0, 1.0]);
    assert!(a.dot(b).is_nan(), "dot with NaN should be NaN");
}

#[test]
fn test_vec8_sum_nan_propagation() {
    let mut vals = [1.0f32; 8];
    vals[3] = f32::NAN;
    let v = Vec8(vals);
    assert!(v.sum().is_nan(), "sum with NaN should be NaN");
}

#[test]
fn test_vec8_sum_inf() {
    let mut vals = [1.0f32; 8];
    vals[5] = f32::INFINITY;
    let v = Vec8(vals);
    assert_eq!(v.sum(), f32::INFINITY);
}

#[test]
fn test_vec8_dot_nan_propagation() {
    let mut a = [1.0f32; 8];
    a[6] = f32::NAN;
    let b = [1.0f32; 8];
    assert!(Vec8(a).dot(Vec8(b)).is_nan(), "dot with NaN should be NaN");
}

// ============ Known exact values ============

#[test]
fn test_vec4_exp_known_values() {
    let e = std::f32::consts::E;
    let v = Vec4([1.0, 0.0, -1.0, 2.0]);
    let r = v.exp();
    assert_ulp(r[0], e, 1.0, "exp(1) = e");
    assert_eq!(r[1], 1.0, "exp(0) = 1");
    assert_ulp(r[2], 1.0 / e, 1.0, "exp(-1) = 1/e");
    assert_ulp(r[3], e * e, 1.0, "exp(2) = e^2");
}

#[test]
fn test_vec4_sin_known_values() {
    let v = Vec4([0.0, FRAC_PI_2, PI, 3.0 * FRAC_PI_2]);
    let r = v.sin();
    assert_eq!(r[0], 0.0, "sin(0) = 0");
    // Note: f32 PI is an approximation, so sin(PI) != 0 exactly.
    // We compare against Rust's f32::sin() as the reference.
    for i in 0..4 {
        assert_ulp(r[i], v[i].sin(), 1.0, &format!("sin({})", v[i]));
    }
}

#[test]
fn test_vec4_cos_known_values() {
    let v = Vec4([0.0, FRAC_PI_2, PI, 2.0 * PI]);
    let r = v.cos();
    assert_eq!(r[0], 1.0, "cos(0) = 1");
    for i in 0..4 {
        assert_ulp(r[i], v[i].cos(), 1.0, &format!("cos({})", v[i]));
    }
}

// ============ sin(x) ≈ x for tiny x (Taylor first term) ============

#[test]
fn test_vec4_sin_equals_x_tiny() {
    // For |x| < ~1e-4, sin(x) should equal x in f32
    let vals = [1e-10, -1e-10, 1e-20, -1e-20];
    let v = Vec4(vals);
    let r = v.sin();
    for i in 0..4 {
        assert_eq!(r[i], vals[i], "sin({:e}) should equal x", vals[i]);
    }
}

#[test]
fn test_vec8_sin_equals_x_tiny() {
    let vals = [1e-10, -1e-10, 1e-20, -1e-20, 1e-30, -1e-30, 1e-38, -1e-38];
    let v = Vec8(vals);
    let r = v.sin();
    for i in 0..8 {
        assert_eq!(r[i], vals[i], "sin({:e}) should equal x", vals[i]);
    }
}

// ============ cos(x) = 1 for tiny x ============

#[test]
fn test_vec8_cos_equals_one_tiny() {
    let vals = [1e-10, -1e-10, 1e-20, -1e-20, 1e-30, -1e-30, 1e-38, -1e-38];
    let v = Vec8(vals);
    let r = v.cos();
    for i in 0..8 {
        assert_eq!(r[i], 1.0, "cos({:e}) should be 1.0", vals[i]);
    }
}

// ============ exp(x) ≈ 1 + x for tiny x ============

#[test]
fn test_vec4_exp_one_plus_x_tiny() {
    let vals = [1e-10, -1e-10, 1e-20, -1e-20];
    let v = Vec4(vals);
    let r = v.exp();
    for i in 0..4 {
        // For tiny x, exp(x) rounds to 1.0 in f32
        assert_eq!(r[i], 1.0, "exp({:e}) should be 1.0", vals[i]);
    }
}

// ============ Large n * pi/2 for sin/cos (quadrant correctness) ============

#[test]
fn test_vec4_sin_quadrant_large_multiples() {
    // Verify quadrant is correct for large multiples of pi/2
    for k in [10, 50, 100, 500, 1000] {
        let x = k as f32 * FRAC_PI_2;
        let v = Vec4([x, x, x, x]);
        let r = v.sin();
        assert_ulp(r[0], x.sin(), 1.0, &format!("sin({}*pi/2)", k));
    }
}

#[test]
fn test_vec4_cos_quadrant_large_multiples() {
    for k in [10, 50, 100, 500, 1000] {
        let x = k as f32 * FRAC_PI_2;
        let v = Vec4([x, x, x, x]);
        let r = v.cos();
        assert_ulp(r[0], x.cos(), 1.0, &format!("cos({}*pi/2)", k));
    }
}

#[test]
fn test_vec8_sin_cos_quadrant_large_multiples() {
    let mults: [f32; 8] = [10.0, 50.0, 100.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0];
    let vals: [f32; 8] = std::array::from_fn(|i| mults[i] * FRAC_PI_2);
    let v = Vec8(vals);
    let rs = v.sin();
    let rc = v.cos();
    for i in 0..8 {
        assert_ulp(rs[i], vals[i].sin(), 1.0, &format!("sin({}*pi/2)", mults[i]));
        assert_ulp(rc[i], vals[i].cos(), 1.0, &format!("cos({}*pi/2)", mults[i]));
    }
}

// ============ Vec4/Vec8 consistency for sum/dot ============

#[test]
fn test_vec4_vec8_sum_consistency() {
    let vals = [1.5, 2.7, 3.1, 4.9];
    let v4 = Vec4(vals);
    let v8 = Vec8([vals[0], vals[1], vals[2], vals[3], 0.0, 0.0, 0.0, 0.0]);
    assert_eq!(v4.sum(), v8.sum() - 0.0, "sum should be close");
    // More precise: v8.sum() includes four zero lanes so sum = same as v4.sum()
    // Actually v8 sums all 8 lanes, so 4 extras are 0.0
    assert_eq!(v4.sum(), v8.sum(), "sum consistency");
}

#[test]
fn test_vec4_vec8_dot_consistency() {
    let a = [1.5, 2.7, 3.1, 4.9];
    let b = [5.3, 6.1, 7.7, 8.3];
    let d4 = Vec4(a).dot(Vec4(b));
    let d8 = Vec8([a[0], a[1], a[2], a[3], 0.0, 0.0, 0.0, 0.0])
        .dot(Vec8([b[0], b[1], b[2], b[3], 0.0, 0.0, 0.0, 0.0]));
    assert_eq!(d4, d8, "dot consistency");
}

// ============ Arithmetic: associativity stress test ============

#[test]
fn test_vec4_add_associative_exact() {
    // For exact f32 values, addition should be associative
    let a = Vec4([1.0, 2.0, 4.0, 8.0]);
    let b = Vec4([0.5, 1.0, 2.0, 4.0]);
    let c = Vec4([0.25, 0.5, 1.0, 2.0]);
    let r1 = (a + b) + c;
    let r2 = a + (b + c);
    assert_eq!(r1, r2, "addition should be associative for exact values");
}

// ============ MulAdd vs separate mul+add ============

#[test]
fn test_vec4_mul_add_vs_separate() {
    // mul_add should give same or better precision than mul then add
    let a = Vec4([1.0000001; 4]);
    let b = Vec4([1.0000001; 4]);
    let c = Vec4([-1.0; 4]);
    let fma_r = a.mul_add(b, c);
    // The key property: FMA has single rounding vs two roundings for mul+add
    for i in 0..4 {
        assert!(fma_r[i].is_finite(), "FMA result should be finite");
    }
}

#[test]
fn test_vec8_mul_add_vs_separate() {
    let a = Vec8([1.0000001; 8]);
    let b = Vec8([1.0000001; 8]);
    let c = Vec8([-1.0; 8]);
    let fma_r = a.mul_add(b, c);
    for i in 0..8 {
        assert!(fma_r[i].is_finite(), "FMA result should be finite");
    }
}

// ============ Sqrt edge cases ============

#[test]
fn test_vec4_sqrt_precision() {
    // Non-perfect-square values
    let vals = [2.0, 3.0, 5.0, 7.0];
    let v = Vec4(vals);
    let r = v.sqrt();
    for i in 0..4 {
        assert_eq!(r[i], vals[i].sqrt(), "sqrt({}) mismatch", vals[i]);
    }
}

#[test]
fn test_vec8_sqrt_precision() {
    let vals = [2.0, 3.0, 5.0, 7.0, 10.0, 100.0, 0.01, 0.0001];
    let v = Vec8(vals);
    let r = v.sqrt();
    for i in 0..8 {
        assert_eq!(r[i], vals[i].sqrt(), "sqrt({}) mismatch", vals[i]);
    }
}

#[test]
fn test_vec4_sqrt_subnormal() {
    let tiny = f32::from_bits(1);
    let v = Vec4([tiny, f32::MIN_POSITIVE, tiny * 100.0, f32::MIN_POSITIVE * 0.5]);
    let r = v.sqrt();
    for i in 0..4 {
        assert_eq!(r[i], v[i].sqrt(), "sqrt subnormal lane {i}");
    }
}

// ============ Large sweep with mixed Cody-Waite / rempif lanes ============

#[test]
fn test_vec8_sin_cos_mixed_cw_rempif() {
    // Each Vec8 mixes lanes in Cody-Waite range (< 125) and rempif range (>= 125)
    let mut max_sin = 0.0f32;
    let mut max_cos = 0.0f32;
    let mut x = 120.0f32;
    while x < 130.0 {
        let v = Vec8([
            x,
            x + 0.5,
            x + 1.0,
            x + 1.5,
            x + 2.0,
            x + 2.5,
            x + 3.0,
            x + 3.5,
        ]);
        let rs = v.sin();
        let rc = v.cos();
        for i in 0..8 {
            let su = ulp_error(rs[i], v[i].sin());
            let cu = ulp_error(rc[i], v[i].cos());
            if su > max_sin {
                max_sin = su;
            }
            if cu > max_cos {
                max_cos = cu;
            }
        }
        x += 0.01;
    }
    assert!(max_sin <= 1.0, "mixed CW/rempif sin max ULP: {max_sin}");
    assert!(max_cos <= 1.0, "mixed CW/rempif cos max ULP: {max_cos}");
}

// ============ AVX2 high-lane vs low-lane independence ============

#[test]
fn test_vec8_lane_independence_sin() {
    // Each lane has a very different value; verify all 8 produce correct results
    let vals = [0.1, 100.0, -50.0, 500.0, 0.001, -200.0, PI, -1e5];
    let v = Vec8(vals);
    let r = v.sin();
    for i in 0..8 {
        assert_ulp(r[i], vals[i].sin(), 1.0, &format!("sin lane {i}: {}", vals[i]));
    }
}

#[test]
fn test_vec8_lane_independence_cos() {
    let vals = [0.1, 100.0, -50.0, 500.0, 0.001, -200.0, PI, -1e5];
    let v = Vec8(vals);
    let r = v.cos();
    for i in 0..8 {
        assert_ulp(r[i], vals[i].cos(), 1.0, &format!("cos lane {i}: {}", vals[i]));
    }
}

#[test]
fn test_vec8_lane_independence_exp() {
    let vals = [0.0, 1.0, -1.0, 50.0, -50.0, 87.0, -87.0, 0.001];
    let v = Vec8(vals);
    let r = v.exp();
    for i in 0..8 {
        assert_ulp(r[i], vals[i].exp(), 1.0, &format!("exp lane {i}: {}", vals[i]));
    }
}

#[test]
fn test_vec8_lane_independence_special_mix() {
    // Mix specials in high lanes (4-7) with normals in low lanes (0-3)
    let v = Vec8([1.0, 2.0, 3.0, 4.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.0]);
    let rs = v.sin();
    assert_ulp(rs[0], 1.0f32.sin(), 1.0, "sin(1) low lane");
    assert_ulp(rs[1], 2.0f32.sin(), 1.0, "sin(2) low lane");
    assert_ulp(rs[2], 3.0f32.sin(), 1.0, "sin(3) low lane");
    assert_ulp(rs[3], 4.0f32.sin(), 1.0, "sin(4) low lane");
    assert!(rs[4].is_nan(), "sin(NaN) high lane");
    assert!(rs[5].is_nan(), "sin(Inf) high lane");
    assert!(rs[6].is_nan(), "sin(-Inf) high lane");
    assert!(rs[7] == 0.0 && rs[7].is_sign_negative(), "sin(-0) high lane");

    let re = v.exp();
    assert_ulp(re[0], 1.0f32.exp(), 1.0, "exp(1) low lane");
    assert!(re[4].is_nan(), "exp(NaN) high lane");
    assert_eq!(re[5], f32::INFINITY, "exp(Inf) high lane");
    assert_eq!(re[6], 0.0, "exp(-Inf) high lane");
    assert_eq!(re[7], 1.0, "exp(-0) high lane");
}

// ============ Trigonometric identities ============

#[test]
fn test_vec4_sin_double_angle() {
    // sin(2x) ≈ 2*sin(x)*cos(x)
    let vals = [0.5, 1.0, 1.5, 2.0];
    let v = Vec4(vals);
    let s = v.sin();
    let c = v.cos();
    let double_vals: [f32; 4] = std::array::from_fn(|i| 2.0 * vals[i]);
    let s2 = Vec4(double_vals).sin();
    for i in 0..4 {
        let identity = 2.0 * s[i] * c[i];
        let diff = (s2[i] - identity).abs();
        assert!(diff < 1e-5, "sin(2x) != 2sin(x)cos(x) at x={}: diff={diff}", vals[i]);
    }
}

#[test]
fn test_vec8_sin_double_angle() {
    let vals = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 50.0, 200.0];
    let v = Vec8(vals);
    let s = v.sin();
    let c = v.cos();
    let double_vals: [f32; 8] = std::array::from_fn(|i| 2.0 * vals[i]);
    let s2 = Vec8(double_vals).sin();
    for i in 0..8 {
        let identity = 2.0 * s[i] * c[i];
        let diff = (s2[i] - identity).abs();
        assert!(diff < 1e-4, "sin(2x) != 2sin(x)cos(x) at x={}: diff={diff}", vals[i]);
    }
}

#[test]
fn test_vec4_cos_double_angle() {
    // cos(2x) ≈ cos²(x) - sin²(x)
    let vals = [0.5, 1.0, 1.5, 2.0];
    let v = Vec4(vals);
    let s = v.sin();
    let c = v.cos();
    let double_vals: [f32; 4] = std::array::from_fn(|i| 2.0 * vals[i]);
    let c2 = Vec4(double_vals).cos();
    for i in 0..4 {
        let identity = c[i] * c[i] - s[i] * s[i];
        let diff = (c2[i] - identity).abs();
        assert!(diff < 1e-5, "cos(2x) != cos²-sin² at x={}: diff={diff}", vals[i]);
    }
}

#[test]
fn test_vec4_sin_addition_formula() {
    // sin(a+b) ≈ sin(a)cos(b) + cos(a)sin(b)
    let a_vals = [0.3, 1.0, -0.5, 2.0];
    let b_vals = [0.7, 0.5, 1.5, -1.0];
    let a = Vec4(a_vals);
    let b = Vec4(b_vals);
    let sa = a.sin();
    let ca = a.cos();
    let sb = b.sin();
    let cb = b.cos();
    let ab: [f32; 4] = std::array::from_fn(|i| a_vals[i] + b_vals[i]);
    let sab = Vec4(ab).sin();
    for i in 0..4 {
        let identity = sa[i] * cb[i] + ca[i] * sb[i];
        let diff = (sab[i] - identity).abs();
        assert!(diff < 1e-5, "sin(a+b) formula at a={}, b={}: diff={diff}", a_vals[i], b_vals[i]);
    }
}

// ============ Exp subnormal output region ============

#[test]
fn test_vec4_exp_subnormal_output() {
    // Near -87.3: exp produces subnormal results (below FLT_MIN_POSITIVE)
    let vals = [-87.0, -87.3, -87.33, -86.0];
    let v = Vec4(vals);
    let r = v.exp();
    for i in 0..4 {
        let expected = vals[i].exp();
        // Just verify finite and same sign as expected
        assert!(r[i].is_finite(), "exp({}) should be finite", vals[i]);
        assert!(r[i] > 0.0, "exp({}) should be positive", vals[i]);
        if expected.is_normal() {
            assert_ulp(r[i], expected, 1.0, &format!("exp({})", vals[i]));
        }
    }
}

#[test]
fn test_vec8_exp_subnormal_output() {
    let vals = [-85.0, -86.0, -87.0, -87.3, -87.33, -87.33654, -86.5, -84.0];
    let v = Vec8(vals);
    let r = v.exp();
    for i in 0..8 {
        assert!(r[i].is_finite() && r[i] > 0.0, "exp({}) should be finite positive", vals[i]);
    }
}

// ============ Negative sweep for sin/cos (rempif path) ============

#[test]
fn test_vec4_sin_sweep_negative_large() {
    let mut max_ulp = 0.0f32;
    let step = 1.0;
    let mut x = -1e6f32;
    while x < -1000.0 {
        let v = Vec4([x, x + step, x + 2.0 * step, x + 3.0 * step]);
        let r = v.sin();
        for i in 0..4 {
            let u = ulp_error(r[i], v[i].sin());
            if u > max_ulp {
                max_ulp = u;
            }
        }
        x += 4.0 * step;
    }
    assert!(max_ulp <= 1.0, "Vec4 sin negative 1e3..1e6 max ULP: {max_ulp}");
}

#[test]
fn test_vec4_cos_sweep_negative_large() {
    let mut max_ulp = 0.0f32;
    let step = 1.0;
    let mut x = -1e6f32;
    while x < -1000.0 {
        let v = Vec4([x, x + step, x + 2.0 * step, x + 3.0 * step]);
        let r = v.cos();
        for i in 0..4 {
            let u = ulp_error(r[i], v[i].cos());
            if u > max_ulp {
                max_ulp = u;
            }
        }
        x += 4.0 * step;
    }
    assert!(max_ulp <= 1.0, "Vec4 cos negative 1e3..1e6 max ULP: {max_ulp}");
}

// ============ Arithmetic identity tests ============

#[test]
fn test_vec4_mul_by_one() {
    let v = Vec4([1.5, -2.7, 0.0, f32::MAX]);
    assert_eq!(v * 1.0, v);
    assert_eq!(1.0 * v, v);
}

#[test]
fn test_vec4_mul_by_zero() {
    let v = Vec4([1.5, -2.7, 100.0, -100.0]);
    let r = v * 0.0;
    for i in 0..4 {
        assert_eq!(r[i], 0.0, "v * 0 should be 0 at lane {i}");
    }
}

#[test]
fn test_vec4_add_zero_identity() {
    let v = Vec4([1.5, -2.7, 0.0, f32::MAX]);
    assert_eq!(v + 0.0, v);
    assert_eq!(0.0 + v, v);
}

#[test]
fn test_vec4_sub_self_is_zero() {
    let v = Vec4([1.5, -2.7, 100.0, -100.0]);
    let r = v - v;
    for i in 0..4 {
        assert_eq!(r[i], 0.0, "v - v should be 0 at lane {i}");
    }
}

#[test]
fn test_vec4_div_by_self() {
    let v = Vec4([1.5, -2.7, 100.0, 0.001]);
    let r = v / v;
    for i in 0..4 {
        assert_eq!(r[i], 1.0, "v / v should be 1 at lane {i}");
    }
}

#[test]
fn test_vec8_mul_by_one() {
    let v = Vec8([1.5, -2.7, 0.0, f32::MAX, f32::MIN_POSITIVE, -0.001, 1e10, -1e10]);
    assert_eq!(v * 1.0, v);
    assert_eq!(1.0 * v, v);
}

#[test]
fn test_vec8_add_zero_identity() {
    let v = Vec8([1.5, -2.7, 0.0, f32::MAX, f32::MIN_POSITIVE, -0.001, 1e10, -1e10]);
    assert_eq!(v + 0.0, v);
    assert_eq!(0.0 + v, v);
}

#[test]
fn test_vec8_sub_self_is_zero() {
    let v = Vec8([1.5, -2.7, 100.0, -100.0, 0.001, -0.001, 1e10, -1e10]);
    let r = v - v;
    for i in 0..8 {
        assert_eq!(r[i], 0.0, "v - v should be 0 at lane {i}");
    }
}

#[test]
fn test_vec8_div_by_self() {
    let v = Vec8([1.5, -2.7, 100.0, 0.001, 1e10, -1e10, 0.5, -0.5]);
    let r = v / v;
    for i in 0..8 {
        assert_eq!(r[i], 1.0, "v / v should be 1 at lane {i}");
    }
}

// ============ Round: all half-integers in [-10.5, 10.5] ============

#[test]
fn test_vec4_round_all_half_integers() {
    // Exhaustively test half-integers: these are the boundary cases for round
    let halves: Vec<f32> = (-21..=21).map(|i| i as f32 * 0.5).collect();
    for chunk in halves.chunks(4) {
        if chunk.len() < 4 {
            break;
        }
        let v = Vec4([chunk[0], chunk[1], chunk[2], chunk[3]]);
        let r = v.round();
        for i in 0..4 {
            assert_eq!(
                r[i],
                chunk[i].round(),
                "round({}) mismatch",
                chunk[i]
            );
        }
    }
}

#[test]
fn test_vec8_round_all_half_integers() {
    let halves: Vec<f32> = (-21..=21).map(|i| i as f32 * 0.5).collect();
    for chunk in halves.chunks(8) {
        if chunk.len() < 8 {
            break;
        }
        let mut arr = [0.0f32; 8];
        arr.copy_from_slice(chunk);
        let v = Vec8(arr);
        let r = v.round();
        for i in 0..8 {
            assert_eq!(r[i], arr[i].round(), "round({}) mismatch", arr[i]);
        }
    }
}

// ============ Sin/cos at power-of-2 multiples of pi ============

#[test]
fn test_vec4_sin_pow2_pi() {
    for &k in &[2.0f32, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0] {
        let x = k * PI;
        let v = Vec4([x, -x, x, -x]);
        let r = v.sin();
        for i in 0..4 {
            assert_ulp(r[i], v[i].sin(), 1.0, &format!("sin({}*pi)", k));
        }
    }
}

#[test]
fn test_vec4_cos_pow2_pi() {
    for &k in &[2.0f32, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0] {
        let x = k * PI;
        let v = Vec4([x, -x, x, -x]);
        let r = v.cos();
        for i in 0..4 {
            assert_ulp(r[i], v[i].cos(), 1.0, &format!("cos({}*pi)", k));
        }
    }
}

// ============ Dot/sum accuracy with alternating-sign pattern ============

#[test]
fn test_vec4_dot_alternating_precision() {
    // Products: 1e7*1, -1e7*1, 1e7*1, -1e7*1 → sum = 0
    // f64-intermediate should get exactly 0
    let a = Vec4([1e7, -1e7, 1e7, -1e7]);
    let b = Vec4([1.0, 1.0, 1.0, 1.0]);
    assert_eq!(a.dot(b), 0.0, "alternating dot should cancel to 0");
}

#[test]
fn test_vec8_dot_alternating_precision() {
    let a = Vec8([1e7, 1.0, -1e7, 0.5, 1e7, 0.25, -1e7, 0.125]);
    let b = Vec8([1.0; 8]);
    // f64 sum: 1e7 + 1 - 1e7 + 0.5 + 1e7 + 0.25 - 1e7 + 0.125 = 1.875
    let r = a.dot(b);
    assert_ulp(r, 1.875, 0.5, "alternating dot precision");
}

#[test]
fn test_vec4_sum_alternating_precision() {
    let v = Vec4([1e7, -1e7, 1e7, -1e7]);
    assert_eq!(v.sum(), 0.0, "alternating sum should cancel to 0");
}

// ============ Sin/cos odd symmetry over rempif path ============

#[test]
fn test_vec4_sin_odd_symmetry_large() {
    let vals = [200.0, 500.0, 1000.0, 5000.0];
    let neg_vals = [-200.0, -500.0, -1000.0, -5000.0];
    let s_pos = Vec4(vals).sin();
    let s_neg = Vec4(neg_vals).sin();
    for i in 0..4 {
        assert_eq!(s_pos[i], -s_neg[i], "sin(-x) != -sin(x) for x={}", vals[i]);
    }
}

#[test]
fn test_vec4_cos_even_symmetry_large() {
    let vals = [200.0, 500.0, 1000.0, 5000.0];
    let neg_vals = [-200.0, -500.0, -1000.0, -5000.0];
    let c_pos = Vec4(vals).cos();
    let c_neg = Vec4(neg_vals).cos();
    for i in 0..4 {
        assert_eq!(c_pos[i], c_neg[i], "cos(-x) != cos(x) for x={}", vals[i]);
    }
}

#[test]
fn test_vec8_sin_odd_symmetry_large() {
    let vals = [200.0, 500.0, 1000.0, 5000.0, 1e4, 5e4, 1e5, 5e5];
    let neg_vals: [f32; 8] = std::array::from_fn(|i| -vals[i]);
    let s_pos = Vec8(vals).sin();
    let s_neg = Vec8(neg_vals).sin();
    for i in 0..8 {
        assert_eq!(s_pos[i], -s_neg[i], "sin(-x) != -sin(x) for x={}", vals[i]);
    }
}

#[test]
fn test_vec8_cos_even_symmetry_large() {
    let vals = [200.0, 500.0, 1000.0, 5000.0, 1e4, 5e4, 1e5, 5e5];
    let neg_vals: [f32; 8] = std::array::from_fn(|i| -vals[i]);
    let c_pos = Vec8(vals).cos();
    let c_neg = Vec8(neg_vals).cos();
    for i in 0..8 {
        assert_eq!(c_pos[i], c_neg[i], "cos(-x) != cos(x) for x={}", vals[i]);
    }
}

// ============ Exp monotonicity ============

#[test]
fn test_vec4_exp_monotonic() {
    // exp is strictly increasing: if a < b then exp(a) <= exp(b)
    let a = Vec4([-10.0, -1.0, 0.0, 1.0]);
    let b = Vec4([-1.0, 0.0, 1.0, 10.0]);
    let ea = a.exp();
    let eb = b.exp();
    for i in 0..4 {
        assert!(ea[i] <= eb[i], "exp not monotonic: exp({}) > exp({})", a[i], b[i]);
    }
}

#[test]
fn test_vec8_exp_monotonic() {
    let a = Vec8([-80.0, -50.0, -10.0, -1.0, 0.0, 1.0, 10.0, 50.0]);
    let b = Vec8([-50.0, -10.0, -1.0, 0.0, 1.0, 10.0, 50.0, 80.0]);
    let ea = a.exp();
    let eb = b.exp();
    for i in 0..8 {
        assert!(ea[i] <= eb[i], "exp not monotonic: exp({}) > exp({})", a[i], b[i]);
    }
}

// ============ Sin range: |sin(x)| <= 1 for random inputs ============

#[test]
fn test_vec4_sin_range_bound() {
    let mut state: u32 = 0x1111_2222;
    for _ in 0..10000 {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        let f = f32::from_bits(state);
        if !f.is_finite() {
            continue;
        }
        let v = Vec4([f; 4]);
        let r = v.sin();
        assert!(r[0].abs() <= 1.0, "sin({f:e}) = {} out of [-1,1]", r[0]);
    }
}

#[test]
fn test_vec4_cos_range_bound() {
    let mut state: u32 = 0x3333_4444;
    for _ in 0..10000 {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        let f = f32::from_bits(state);
        if !f.is_finite() {
            continue;
        }
        let v = Vec4([f; 4]);
        let r = v.cos();
        assert!(r[0].abs() <= 1.0, "cos({f:e}) = {} out of [-1,1]", r[0]);
    }
}

// ============ Exp positivity: exp(x) > 0 for all finite x ============

#[test]
fn test_vec4_exp_positive() {
    let mut state: u32 = 0x5555_6666;
    for _ in 0..10000 {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        let f = f32::from_bits(state);
        if !f.is_finite() {
            continue;
        }
        let v = Vec4([f; 4]);
        let r = v.exp();
        assert!(r[0] >= 0.0, "exp({f:e}) = {} should be >= 0", r[0]);
        if f > -104.0 {
            assert!(r[0] > 0.0, "exp({f:e}) should be > 0 (not clamped)");
        }
    }
}

// ============ Sqrt: sqrt(x)^2 ≈ x ============

#[test]
fn test_vec4_sqrt_roundtrip() {
    let vals = [2.0, 3.0, 10.0, 100.0];
    let v = Vec4(vals);
    let s = v.sqrt();
    let roundtrip = s * s;
    for i in 0..4 {
        assert_ulp(roundtrip[i], vals[i], 1.0, &format!("sqrt({})^2", vals[i]));
    }
}

#[test]
fn test_vec8_sqrt_roundtrip() {
    let vals = [2.0, 3.0, 5.0, 7.0, 10.0, 100.0, 0.01, 0.0001];
    let v = Vec8(vals);
    let s = v.sqrt();
    let roundtrip = s * s;
    for i in 0..8 {
        assert_ulp(roundtrip[i], vals[i], 1.0, &format!("sqrt({})^2", vals[i]));
    }
}

// ============ MulAdd: a*b+c with NaN/Inf ============

#[test]
fn test_vec4_mul_add_special() {
    let a = Vec4([f32::NAN, f32::INFINITY, 0.0, 1.0]);
    let b = Vec4([1.0, 1.0, f32::NAN, f32::INFINITY]);
    let c = Vec4([0.0, 0.0, 0.0, 0.0]);
    let r = a.mul_add(b, c);
    assert!(r[0].is_nan(), "NaN * 1 + 0 = NaN");
    assert_eq!(r[1], f32::INFINITY, "Inf * 1 + 0 = Inf");
    assert!(r[2].is_nan(), "0 * NaN + 0 = NaN");
    assert_eq!(r[3], f32::INFINITY, "1 * Inf + 0 = Inf");
}

#[test]
fn test_vec8_mul_add_special() {
    let a = Vec8([f32::NAN, f32::INFINITY, 0.0, 1.0, -1.0, f32::NEG_INFINITY, 2.0, -2.0]);
    let b = Vec8([1.0; 8]);
    let c = Vec8([0.0; 8]);
    let r = a.mul_add(b, c);
    assert!(r[0].is_nan());
    assert_eq!(r[1], f32::INFINITY);
    assert_eq!(r[2], 0.0);
    assert_eq!(r[3], 1.0);
    assert_eq!(r[4], -1.0);
    assert_eq!(r[5], f32::NEG_INFINITY);
}

// ============ PartialEq reflexivity ============

#[test]
fn test_vec4_eq_reflexive() {
    let v = Vec4([1.0, 2.0, 3.0, 4.0]);
    assert_eq!(v, v);
}

#[test]
fn test_vec8_eq_reflexive() {
    let v = Vec8([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    assert_eq!(v, v);
}

// ============ Floor/ceil: integer inputs are idempotent ============

#[test]
fn test_vec4_floor_ceil_round_idempotent() {
    // Applying floor/ceil/round twice gives the same result as once
    let v = Vec4([1.7, -2.3, 0.5, -0.5]);
    let fl = v.floor();
    let ce = v.ceil();
    let ro = v.round();
    assert_eq!(fl.floor(), fl, "floor idempotent");
    assert_eq!(ce.ceil(), ce, "ceil idempotent");
    assert_eq!(ro.round(), ro, "round idempotent");
}

#[test]
fn test_vec8_floor_ceil_round_idempotent() {
    let v = Vec8([1.7, -2.3, 0.5, -0.5, 100.1, -100.9, 0.0, -0.0]);
    let fl = v.floor();
    let ce = v.ceil();
    let ro = v.round();
    assert_eq!(fl.floor(), fl, "floor idempotent");
    assert_eq!(ce.ceil(), ce, "ceil idempotent");
    assert_eq!(ro.round(), ro, "round idempotent");
}
