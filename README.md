# simd_vector

SIMD vector types for x86-64 in pure stable Rust.

- `Vec4` — 4×f32, SSE4.1 + FMA3
- `Vec8` — 8×f32, AVX2 + FMA3

## Features

- Arithmetic: `+`, `-`, `*`, `/` (vec×vec, vec×f32, f32×vec)
- `splat`, `abs`, `neg`, `sqrt`, `floor`, `ceil`, `round`, `sum`, `dot`, `mul_add` (FMA)
- `Sum`, `Index`, `From`/`Into` array, `Clone`, `Copy`, `Debug`, `PartialEq`
- Transcendentals: `sin`, `cos`, `exp` — ported from [SLEEF](https://github.com/shibatch/sleef)

## Accuracy

| Function | ULP error | Notes |
|----------|-----------|-------|
| `+`, `-`, `*`, `/` | 0.0 | IEEE 754 correctly rounded |
| `neg` | 0.0 | exact bit flip |
| `abs` | 0.0 | exact bit mask |
| `floor` | 0.0 | exact (IEEE 754 `roundps`) |
| `ceil` | 0.0 | exact (IEEE 754 `roundps`) |
| `round` | 0.0 | exact (half away from zero, matches `f32::round`) |
| `splat` | 0.0 | exact broadcast |
| `sum` | ≤ 0.5 | f64 intermediate: exact tree sum, single rounding on f64→f32 |
| `dot` | ≤ 0.5 | f64 intermediate: exact products + tree sum, single rounding on f64→f32 |
| `mul_add` | ≤ 0.5 | single FMA instruction |
| `sqrt` | 0.0 | IEEE 754 correctly rounded (`sqrtps`) |
| `sin` | ≤ 1.0 | SLEEF `xsinf_u1` — Cody-Waite + Payne-Hanek + double-float polynomial |
| `cos` | ≤ 1.0 | SLEEF `xcosf_u1` — Cody-Waite + Payne-Hanek + double-float polynomial |
| `exp` | ≤ 1.0 | SLEEF `xexpf` — ln(2) range reduction + degree-6 polynomial + ldexp |

### Transcendental implementation details

- **Range reduction** for `sin`/`cos`: Cody-Waite (3-constant) for |x| < 125, Payne-Hanek table-based (`rempif`) for larger arguments
- **Double-float arithmetic**: FMA-based error-free transformations for high precision in the polynomial evaluation
- **Edge cases**: NaN/Inf propagation, sin(-0) = -0, exp(-inf) = 0, exp(inf) = inf
- Polynomial coefficients, constants, and the 416-entry `Sleef_rempitabsp` table are taken directly from [SLEEF](https://github.com/shibatch/sleef)

## Required CPU features

SSE4.1, AVX2, FMA3. Enabled globally via `.cargo/config.toml`:

```toml
[build]
rustflags = ["-C", "target-feature=+sse4.1,+avx2,+fma"]
```

## Installation

```
cargo add simd_vector
```

## Usage

```rust
use simd_vector::{Vec4, Vec8};

let a = Vec4([1.0, 2.0, 3.0, 4.0]);
let b = Vec4([5.0, 6.0, 7.0, 8.0]);
let c = a + b;              // [6.0, 8.0, 10.0, 12.0]
let d = a.dot(b);           // 70.0
let s = a.sin();            // [0.8415, 0.9093, 0.1411, -0.7568]
let e = Vec8::splat(1.0).exp(); // [2.71828, 2.71828, ...] (8 lanes)
```

## Tests

345 tests covering all operations, edge cases (NaN, Inf, -0.0, subnormals), sampled ULP sweep verification, trigonometric identities, arithmetic properties, and AVX2 lane independence.

```
cargo test
```

Coverage (via `cargo llvm-cov`):

```
Filename   Regions  Missed  Cover   Functions  Missed  Cover   Lines  Missed  Cover
lib.rs     142      0       100.00% 6          0       100.00% 61     0       100.00%
vec4.rs    892      0       100.00% 48         0       100.00% 437    0       100.00%
vec8.rs    915      0       100.00% 48         0       100.00% 441    0       100.00%
TOTAL      1949     0       100.00% 102        0       100.00% 939    0       100.00%
```

## License

MIT
