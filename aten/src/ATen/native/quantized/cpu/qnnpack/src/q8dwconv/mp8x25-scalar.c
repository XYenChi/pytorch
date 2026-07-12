/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <stddef.h>
#include <stdint.h>

#include <qnnpack/q8dwconv.h>
#include <qnnpack/scalar-utils.h>

void pytorch_q8dwconv_ukernel_mp8x25__scalar(
    size_t channels,
    size_t output_width,
    const uint8_t** input,
    const void* weights,
    int32_t* outacc32,
    uint8_t* output,
    size_t input_stride,
    size_t output_increment,
    const union pytorch_qnnp_conv_quantization_params
        quantization_params[restrict static 1]) {
  (void)outacc32;

  const int32_t input_zero_point =
      quantization_params->scalar.input_zero_point;
  const uint8_t output_zero_point =
      (uint8_t)quantization_params->scalar.output_zero_point;
  const uint8_t output_min =
      (uint8_t)(quantization_params->scalar.output_zero_point +
          quantization_params->scalar.output_min_less_zero_point);
  const uint8_t output_max =
      (uint8_t)(quantization_params->scalar.output_zero_point +
          quantization_params->scalar.output_max_less_zero_point);

  const size_t c_stride = (channels + 7) & ~(size_t)7;
  const size_t chunk1_offset = 14 * c_stride;
  const size_t chunk2_offset = 24 * c_stride;

  do {
    const uint8_t* w0 = (const uint8_t*)weights;
    const uint8_t* w1 = (const uint8_t*)weights + chunk1_offset;
    const uint8_t* w2 = (const uint8_t*)weights + chunk2_offset;

    const uint8_t* rows[25];
    for (size_t p = 0; p < 25; p++) {
      rows[p] = input[p];
    }

    size_t c = channels;
    size_t channel_offset = 0;
    for (; c >= 8; c -= 8) {
      const int32_t* bias = (const int32_t*)w0;
      const uint8_t* pw0 = w0 + 8 * sizeof(int32_t);
      const uint8_t* pw1 = w1;
      const uint8_t* pw2 = w2;

      for (size_t cc = 0; cc < 8; cc++) {
        const size_t channel = channel_offset + cc;
        const int32_t kernel_zero_point =
            quantization_params->scalar.kernel_zero_points[channel];
        int32_t acc = bias[cc];

        for (size_t p = 0; p < 10; p++) {
          acc += ((int32_t)rows[p][cc] - input_zero_point) *
              ((int32_t)pw0[p * 8 + cc] - kernel_zero_point);
        }
        for (size_t p = 0; p < 10; p++) {
          acc += ((int32_t)rows[10 + p][cc] - input_zero_point) *
              ((int32_t)pw1[p * 8 + cc] - kernel_zero_point);
        }
        for (size_t p = 0; p < 5; p++) {
          acc += ((int32_t)rows[20 + p][cc] - input_zero_point) *
              ((int32_t)pw2[p * 8 + cc] - kernel_zero_point);
        }

        output[cc] = pytorch_scalar_requantize_precise(
            acc,
            quantization_params->scalar.requantization_scales[channel],
            output_zero_point,
            output_min,
            output_max);
      }

      for (size_t p = 0; p < 25; p++) {
        rows[p] += 8;
      }
      output += 8;
      w0 += 8 * sizeof(int32_t) + 10 * 8;
      w1 += 10 * 8;
      w2 += 5 * 8;
      channel_offset += 8;
    }

    if (c != 0) {
      const int32_t* bias = (const int32_t*)w0;
      const uint8_t* pw0 = w0 + 8 * sizeof(int32_t);
      const uint8_t* pw1 = w1;
      const uint8_t* pw2 = w2;

      for (size_t cc = 0; cc < c; cc++) {
        const size_t channel = channel_offset + cc;
        const int32_t kernel_zero_point =
            quantization_params->scalar.kernel_zero_points[channel];
        int32_t acc = bias[cc];

        for (size_t p = 0; p < 10; p++) {
          acc += ((int32_t)rows[p][cc] - input_zero_point) *
              ((int32_t)pw0[p * 8 + cc] - kernel_zero_point);
        }
        for (size_t p = 0; p < 10; p++) {
          acc += ((int32_t)rows[10 + p][cc] - input_zero_point) *
              ((int32_t)pw1[p * 8 + cc] - kernel_zero_point);
        }
        for (size_t p = 0; p < 5; p++) {
          acc += ((int32_t)rows[20 + p][cc] - input_zero_point) *
              ((int32_t)pw2[p * 8 + cc] - kernel_zero_point);
        }

        output[cc] = pytorch_scalar_requantize_precise(
            acc,
            quantization_params->scalar.requantization_scales[channel],
            output_zero_point,
            output_min,
            output_max);
      }
      output += c;
    }

    input = (const uint8_t**)((uintptr_t)input + input_stride);
    output = (uint8_t*)((uintptr_t)output + output_increment);
  } while (--output_width != 0);
}
