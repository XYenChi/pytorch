#pragma once

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>

#include <cstdint>
#include <cstring>
#include <type_traits>

namespace c10 {

// Note: Explicit implementation of copysign for Half and BFloat16
// is needed to workaround g++-7/8 crash on aarch64, but also makes
// copysign faster for the half-precision types
template <typename T, typename U>
inline auto copysign(const T& a, const U& b) {
#if defined(__riscv)
  // Workaround for buggy std::copysign on some RISC-V toolchains/emulators.
  // Use bit manipulation to copy the sign bit, matching the approach already
  // used for Half and BFloat16. Using if constexpr inside the template
  // guarantees the workaround is always selected for float/double, avoiding
  // any overload resolution ambiguity with non-template overloads.
  if constexpr (std::is_same_v<float, std::decay_t<T>> &&
                std::is_same_v<float, std::decay_t<U>>) {
    uint32_t a_bits, b_bits;
    std::memcpy(&a_bits, &a, sizeof(a));
    std::memcpy(&b_bits, &b, sizeof(b));
    a_bits = (a_bits & 0x7FFFFFFFU) | (b_bits & 0x80000000U);
    std::decay_t<T> result;
    std::memcpy(&result, &a_bits, sizeof(result));
    return result;
  } else if constexpr (std::is_same_v<double, std::decay_t<T>> &&
                       std::is_same_v<double, std::decay_t<U>>) {
    uint64_t a_bits, b_bits;
    std::memcpy(&a_bits, &a, sizeof(a));
    std::memcpy(&b_bits, &b, sizeof(b));
    a_bits = (a_bits & 0x7FFFFFFFFFFFFFFFULL) | (b_bits & 0x8000000000000000ULL);
    std::decay_t<T> result;
    std::memcpy(&result, &a_bits, sizeof(result));
    return result;
  } else {
    return std::copysign(a, b);
  }
#else
  return std::copysign(a, b);
#endif
}

// Implement copysign for half precision floats using bit ops
// Sign is the most significant bit for both half and bfloat16 types
inline c10::Half copysign(c10::Half a, c10::Half b) {
  return c10::Half((a.x & 0x7fff) | (b.x & 0x8000), c10::Half::from_bits());
}

inline c10::BFloat16 copysign(c10::BFloat16 a, c10::BFloat16 b) {
  return c10::BFloat16(
      (a.x & 0x7fff) | (b.x & 0x8000), c10::BFloat16::from_bits());
}

} // namespace c10
