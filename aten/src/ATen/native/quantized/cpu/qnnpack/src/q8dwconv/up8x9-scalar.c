/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Scalar (portable C) implementation of q8dwconv up8x9.
 * Layout matches pack_q8dw_w with cr=8:
 *   For each cr-block: int32_t bias[8], then for each of 9 positions: uint8_t w[8]
 * Bias has -izp*sum_w pre-applied; we compute acc = pbias + sum i*(w-kzp).
 */

#include <stddef.h>
#include <stdint.h>

#include <qnnpack/q8dwconv.h>
#include <qnnpack/scalar-utils.h>


void pytorch_q8dwconv_ukernel_up8x9__scalar(
    size_t channels,
    size_t output_width,
    const uint8_t** input,
    const void* weights,
    uint8_t* output,
    size_t input_stride,
    size_t output_increment,
    const union pytorch_qnnp_conv_quantization_params quantization_params[restrict static 1])
{
  const int32_t input_zero_point =
      quantization_params->scalar.input_zero_point;

  do {
    const uint8_t* i0 = input[0];
    const uint8_t* i1 = input[1];
    const uint8_t* i2 = input[2];
    const uint8_t* i3 = input[3];
    const uint8_t* i4 = input[4];
    const uint8_t* i5 = input[5];
    const uint8_t* i6 = input[6];
    const uint8_t* i7 = input[7];
    const uint8_t* i8 = input[8];
    input = (const uint8_t**) ((uintptr_t) input + input_stride);

    const uint8_t* w = (const uint8_t*) weights;
    size_t c = channels;
    size_t channel_offset = 0;
    /* cr-block size is 8. We iterate channel-by-channel within the block. */
    for (; c >= 8; c -= 8) {
      const int32_t* pbias = (const int32_t*) w;
      const uint8_t* pw = w + 8 * sizeof(int32_t);
      for (size_t cc = 0; cc < 8; cc++) {
        const size_t channel = channel_offset + cc;
        const int32_t kernel_zero_point =
            quantization_params->scalar.kernel_zero_points[channel];
        int32_t acc = pbias[cc];
        acc += ((int32_t) i0[cc] - input_zero_point) *
            ((int32_t) pw[0 * 8 + cc] - kernel_zero_point);
        acc += ((int32_t) i1[cc] - input_zero_point) *
            ((int32_t) pw[1 * 8 + cc] - kernel_zero_point);
        acc += ((int32_t) i2[cc] - input_zero_point) *
            ((int32_t) pw[2 * 8 + cc] - kernel_zero_point);
        acc += ((int32_t) i3[cc] - input_zero_point) *
            ((int32_t) pw[3 * 8 + cc] - kernel_zero_point);
        acc += ((int32_t) i4[cc] - input_zero_point) *
            ((int32_t) pw[4 * 8 + cc] - kernel_zero_point);
        acc += ((int32_t) i5[cc] - input_zero_point) *
            ((int32_t) pw[5 * 8 + cc] - kernel_zero_point);
        acc += ((int32_t) i6[cc] - input_zero_point) *
            ((int32_t) pw[6 * 8 + cc] - kernel_zero_point);
        acc += ((int32_t) i7[cc] - input_zero_point) *
            ((int32_t) pw[7 * 8 + cc] - kernel_zero_point);
        acc += ((int32_t) i8[cc] - input_zero_point) *
            ((int32_t) pw[8 * 8 + cc] - kernel_zero_point);

        output[cc] = pytorch_scalar_requantize_precise(
            acc,
            quantization_params->scalar.requantization_scales[channel],
            (uint8_t) quantization_params->scalar.output_zero_point,
            (uint8_t)(quantization_params->scalar.output_zero_point +
                quantization_params->scalar.output_min_less_zero_point),
            (uint8_t)(quantization_params->scalar.output_zero_point +
                quantization_params->scalar.output_max_less_zero_point));
      }
      i0 += 8; i1 += 8; i2 += 8; i3 += 8; i4 += 8;
      i5 += 8; i6 += 8; i7 += 8; i8 += 8;
      output += 8;
      channel_offset += 8;
      w += 8 * sizeof(int32_t) + 9 * 8;
    }
    /* Tail: c channels remaining (c < 8). */
    if (c != 0) {
      const int32_t* pbias = (const int32_t*) w;
      const uint8_t* pw = w + 8 * sizeof(int32_t);
      for (size_t cc = 0; cc < c; cc++) {
        const size_t channel = channel_offset + cc;
        const int32_t kernel_zero_point =
            quantization_params->scalar.kernel_zero_points[channel];
        int32_t acc = pbias[cc];
        acc += ((int32_t) i0[cc] - input_zero_point) *
            ((int32_t) pw[0 * 8 + cc] - kernel_zero_point);
        acc += ((int32_t) i1[cc] - input_zero_point) *
            ((int32_t) pw[1 * 8 + cc] - kernel_zero_point);
        acc += ((int32_t) i2[cc] - input_zero_point) *
            ((int32_t) pw[2 * 8 + cc] - kernel_zero_point);
        acc += ((int32_t) i3[cc] - input_zero_point) *
            ((int32_t) pw[3 * 8 + cc] - kernel_zero_point);
        acc += ((int32_t) i4[cc] - input_zero_point) *
            ((int32_t) pw[4 * 8 + cc] - kernel_zero_point);
        acc += ((int32_t) i5[cc] - input_zero_point) *
            ((int32_t) pw[5 * 8 + cc] - kernel_zero_point);
        acc += ((int32_t) i6[cc] - input_zero_point) *
            ((int32_t) pw[6 * 8 + cc] - kernel_zero_point);
        acc += ((int32_t) i7[cc] - input_zero_point) *
            ((int32_t) pw[7 * 8 + cc] - kernel_zero_point);
        acc += ((int32_t) i8[cc] - input_zero_point) *
            ((int32_t) pw[8 * 8 + cc] - kernel_zero_point);

        output[cc] = pytorch_scalar_requantize_precise(
            acc,
            quantization_params->scalar.requantization_scales[channel],
            (uint8_t) quantization_params->scalar.output_zero_point,
            (uint8_t)(quantization_params->scalar.output_zero_point +
                quantization_params->scalar.output_min_less_zero_point),
            (uint8_t)(quantization_params->scalar.output_zero_point +
                quantization_params->scalar.output_max_less_zero_point));
      }
      output += c;
    }
    output = (uint8_t*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}
