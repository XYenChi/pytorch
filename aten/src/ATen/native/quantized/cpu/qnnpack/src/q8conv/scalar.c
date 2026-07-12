/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Scalar (portable C) implementation of q8conv microkernel.
 * Layout matches pack_q8conv_w with nr=8, kr=1.
 */

#include <stddef.h>
#include <stdint.h>

#include <qnnpack/q8conv.h>
#include <qnnpack/scalar-utils.h>


void pytorch_q8conv_ukernel_4x8__scalar(
    size_t mr,
    size_t nr,
    size_t kc,
    size_t ks,
    const uint8_t** restrict a,
    const void* restrict w,
    uint8_t* restrict c,
    size_t c_stride,
    size_t output_channel_index,
    const union pytorch_qnnp_conv_quantization_params quantization_params[restrict static 1])
{
  const int32_t* packed_bias = (const int32_t*) w;
  int32_t vacc[4][8];
  for (size_t i = 0; i < 4; i++) {
    for (size_t j = 0; j < 8; j++) {
      vacc[i][j] = packed_bias[j];
    }
  }
  const uint8_t* packed_w = (const uint8_t*) ((uintptr_t) w + 8 * sizeof(int32_t));

  const int32_t input_zero_point =
      quantization_params->scalar.input_zero_point;

  for (size_t p = 0; p < ks; p++) {
    const uint8_t* a0 = a[0];
    const uint8_t* a1 = a[1];
    const uint8_t* a2 = a[2];
    const uint8_t* a3 = a[3];
    a += 4;
    for (size_t kk = 0; kk < kc; kk++) {
      const int32_t va0 = (int32_t) a0[kk] - input_zero_point;
      const int32_t va1 = (int32_t) a1[kk] - input_zero_point;
      const int32_t va2 = (int32_t) a2[kk] - input_zero_point;
      const int32_t va3 = (int32_t) a3[kk] - input_zero_point;
      for (size_t j = 0; j < 8; j++) {
        const int32_t vb = (int32_t) packed_w[j] -
            (int32_t) quantization_params->scalar.kernel_zero_points[
                output_channel_index + j];
        vacc[0][j] += va0 * vb;
        vacc[1][j] += va1 * vb;
        vacc[2][j] += va2 * vb;
        vacc[3][j] += va3 * vb;
      }
      packed_w += 8;
    }
  }

  for (size_t i = 0; i < mr; i++) {
    for (size_t j = 0; j < nr; j++) {
      c[i * c_stride + j] = pytorch_scalar_requantize_precise(
          vacc[i][j],
          quantization_params->scalar.requantization_scales[
              output_channel_index + j],
          (uint8_t) quantization_params->scalar.output_zero_point,
          (uint8_t)(quantization_params->scalar.output_zero_point +
              quantization_params->scalar.output_min_less_zero_point),
          (uint8_t)(quantization_params->scalar.output_zero_point +
              quantization_params->scalar.output_max_less_zero_point));
    }
  }
}
