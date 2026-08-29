#pragma once

// On Windows, math.h needs to be included with _USE_MATH_DEFINES defined to
// access constants such as M_SQRT2 and M_2_SQRTPI.
#ifdef _WIN32
#define _USE_MATH_DEFINES
#include <cmath>
#include <math.h>
#endif // _WIN32

#include <ATen/cpu/vec/vec.h>
#include <c10/util/BFloat16.h> // For c10::is_reduced_floating_point_v.

#if defined(__riscv_v_intrinsic) && __riscv_v_intrinsic >= 12000
#include <riscv_vector.h>
#endif

namespace at::native {
inline namespace CPU_CAPABILITY {
constexpr double kGeluBeta = M_SQRT2 * M_2_SQRTPI * 0.5;
constexpr double kGeluKappa = 0.044715;

#if defined(__riscv_v_intrinsic) && __riscv_v_intrinsic >= 12000
// RVV fast tanh approximation for GELU
// tanh(x) ≈ x * (3 + x^2) / (3 + 3*x^2) for good accuracy
static inline vfloat32m2_t rvv_fast_tanh_f32m2(vfloat32m2_t x, size_t vl) {
  vfloat32m2_t x2 = __riscv_vfmul_vv_f32m2(x, x, vl);
  vfloat32m2_t three = __riscv_vfmv_v_f_f32m2(3.0f, vl);
  vfloat32m2_t three_x2 = __riscv_vfmul_vv_f32m2(three, x2, vl);
  
  vfloat32m2_t numerator = __riscv_vfadd_vv_f32m2(three, x2, vl);
  numerator = __riscv_vfmul_vv_f32m2(x, numerator, vl);
  
  vfloat32m2_t denominator = __riscv_vfadd_vv_f32m2(three, three_x2, vl);
  
  return __riscv_vfdiv_vv_f32m2(numerator, denominator, vl);
}

// RVV-optimized GELU with tanh approximation for float32 contiguous tensors
static vfloat32m2_t rvv_gelu_approximated_tanh_f32m2(vfloat32m2_t x, size_t vl) {
  vfloat32m2_t kGeluBetaVec = __riscv_vfmv_v_f_f32m2((float)kGeluBeta, vl);
  vfloat32m2_t kGeluKappaVec = __riscv_vfmv_v_f_f32m2((float)kGeluKappa, vl);
  vfloat32m2_t half = __riscv_vfmv_v_f_f32m2(0.5f, vl);
  vfloat32m2_t one = __riscv_vfmv_v_f_f32m2(1.0f, vl);
  
  vfloat32m2_t x_cube = __riscv_vfmul_vv_f32m2(x, __riscv_vfmul_vv_f32m2(x, x, vl), vl);
  vfloat32m2_t inner = __riscv_vfadd_vv_f32m2(x, __riscv_vfmul_vv_f32m2(kGeluKappaVec, x_cube, vl), vl);
  inner = __riscv_vfmul_vv_f32m2(kGeluBetaVec, inner, vl);
  
  vfloat32m2_t tanh_inner = rvv_fast_tanh_f32m2(inner, vl);
  vfloat32m2_t one_plus_tanh = __riscv_vfadd_vv_f32m2(one, tanh_inner, vl);
  
  return __riscv_vfmul_vv_f32m2(half, __riscv_vfmul_vv_f32m2(x, one_plus_tanh, vl), vl);
}
#endif

template <typename T>
using reduced_fp_to_float_t = std::conditional_t<c10::is_reduced_floating_point_v<T>, float, T>;

template <typename T, std::enable_if_t<c10::is_reduced_floating_point_v<T>, bool> = true>
float reduced_fp_to_float(T x) {
  return float(x);
}

template <typename T, std::enable_if_t<!c10::is_reduced_floating_point_v<T>, bool> = true>
T reduced_fp_to_float(T x) {
  return x;
}

template <typename T>
T scalar_gelu_approximated_with_tanh(T x) {
  using opmath_t = reduced_fp_to_float_t<T>;
  auto x_float = reduced_fp_to_float(x);
  auto x_cube = x_float * x_float * x_float;
  auto inner = opmath_t(kGeluBeta) * (x_float + opmath_t(kGeluKappa) * x_cube);
  return opmath_t(0.5) * x_float * (opmath_t(1) + std::tanh(inner));
}

template <typename T, std::enable_if_t<!c10::is_reduced_floating_point_v<T>, bool> = true>
vec::Vectorized<T> vectorized_gelu_approximated_with_tanh(vec::Vectorized<T> x) {
  const vec::Vectorized<T> kPointFiveVec(T(0.5));
  const vec::Vectorized<T> kOneVec(T(1));
  const vec::Vectorized<T> kGeluBetaVec((T(kGeluBeta)));
  const vec::Vectorized<T> kGeluKappaVec((T(kGeluKappa)));
  auto x_cube = x * x * x;
  vec::Vectorized<T> inner_vec = kGeluBetaVec * (x + kGeluKappaVec * x_cube);
  return kPointFiveVec * x * (kOneVec + inner_vec.tanh());
}

template <typename T, std::enable_if_t<c10::is_reduced_floating_point_v<T>, bool> = true>
vec::Vectorized<T> vectorized_gelu_approximated_with_tanh(vec::Vectorized<T> x) {
  auto [x0, x1] = at::vec::convert_to_float<T>(x);
  return at::vec::convert_from_float<T>(
      vectorized_gelu_approximated_with_tanh(x0),
      vectorized_gelu_approximated_with_tanh(x1));
}


template <typename T>
T scalar_gelu(T x) {
  using opmath_t = reduced_fp_to_float_t<T>;
  const auto kAlpha = opmath_t(M_SQRT1_2);
  return reduced_fp_to_float(x) * opmath_t(0.5) * (opmath_t(1) + std::erf(reduced_fp_to_float(x) * kAlpha));
}

template<typename T, std::enable_if_t<!c10::is_reduced_floating_point_v<T>, bool> = true>
vec::Vectorized<T> vectorized_gelu(vec::Vectorized<T> x) {
  const vec::Vectorized<T> kAlphaVec(T(M_SQRT1_2));
  const vec::Vectorized<T> kOneVec(T(1));
  const vec::Vectorized<T> kPointFiveVec(T(0.5));
  return x * kPointFiveVec * (kOneVec + (x * kAlphaVec).erf());
}

template<typename T, std::enable_if_t<c10::is_reduced_floating_point_v<T>, bool> = true>
vec::Vectorized<T> vectorized_gelu(vec::Vectorized<T> x) {
  auto [x0, x1] = at::vec::convert_to_float<T>(x);
  return at::vec::convert_from_float<T>(vectorized_gelu(x0), vectorized_gelu(x1));
}

} // namespace CPU_CAPABILITY
} // namespace at::native
