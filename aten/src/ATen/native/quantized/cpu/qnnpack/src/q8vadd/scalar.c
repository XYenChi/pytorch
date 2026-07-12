/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <stddef.h>
#include <stdint.h>

#include <qnnpack/q8vadd.h>
#include <qnnpack/scalar-utils.h>


void pytorch_q8vadd_ukernel__scalar(
    size_t n,
    const uint8_t* a,
    const uint8_t* b,
    uint8_t* y,
    const union pytorch_qnnp_add_quantization_params params[restrict static 1])
{
  const int32_t zp_product = params->scalar.zero_point_product;
  const int32_t a_mul = (int32_t) params->scalar.a_multiplier;
  const int32_t b_mul = (int32_t) params->scalar.b_multiplier;
  const uint32_t shift = params->scalar.shift;
  const int32_t remainder_mask = params->scalar.remainder_mask;
  const int32_t remainder_threshold = params->scalar.remainder_threshold;
  const int32_t y_zero_point = params->scalar.y_zero_point;
  const int32_t y_max = params->scalar.y_max;
  const int32_t y_min = params->scalar.y_min;

  while (n-- != 0) {
    const int32_t va = (int32_t) (uint32_t) *a++;
    const int32_t vb = (int32_t) (uint32_t) *b++;
    int32_t acc = zp_product + va * a_mul + vb * b_mul;
    const int32_t rem = (acc & remainder_mask) - (int32_t) (acc < 0);
    int32_t out = asr_s32(acc, shift) + (int32_t) (rem > remainder_threshold);
    out += y_zero_point;
    if (out < y_min) out = y_min;
    if (out > y_max) out = y_max;
    *y++ = (uint8_t) out;
  }
}
