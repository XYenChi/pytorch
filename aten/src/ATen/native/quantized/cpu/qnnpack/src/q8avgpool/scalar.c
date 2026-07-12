/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Scalar (portable C) implementations of q8avgpool microkernels.
 */

#include <stddef.h>
#include <stdint.h>

#include <qnnpack/q8avgpool.h>
#include <qnnpack/scalar-utils.h>


static inline uint8_t scalar_avgpool_requantize(
    int32_t acc,
    const union pytorch_qnnp_avgpool_quantization_params* quantization_params) {
  return pytorch_scalar_requantize_precise(
      acc,
      quantization_params->scalar.scale,
      (uint8_t) quantization_params->scalar.output_zero_point,
      quantization_params->scalar.output_min,
      quantization_params->scalar.output_max);
}

/*
 * up8xm: called when channels < kr (==8). nrows == ks, no padding needed.
 * Sum ks real input rows per output pixel.
 */
void pytorch_q8avgpool_ukernel_up8xm__scalar(
    size_t n,
    size_t ks,
    size_t kc,
    const uint8_t** input,
    const uint8_t* zero,
    uint8_t* output,
    size_t input_increment,
    size_t output_increment,
    const union pytorch_qnnp_avgpool_quantization_params quantization_params[restrict static 1])
{
  const int32_t bias = quantization_params->scalar.bias;

  (void) zero;
  do {
    for (size_t c = 0; c < kc; c++) {
      int32_t acc = bias;
      for (size_t p = 0; p < ks; p++) {
        acc += (int32_t) (uint32_t) input[p][c];
      }
      output[c] = scalar_avgpool_requantize(acc, quantization_params);
    }
    output = (uint8_t*) ((uintptr_t) output + kc + output_increment);
    input = (const uint8_t**) ((uintptr_t) input + input_increment);
  } while (--n != 0);
}


/*
 * up8x9: channels >= kr and pooling_size <= mr (==9).
 * Sum exactly 9 rows; pad missing rows with `zero` pointer.
 */
void pytorch_q8avgpool_ukernel_up8x9__scalar(
    size_t n,
    size_t ks,
    size_t kc,
    const uint8_t** input,
    const uint8_t* zero,
    uint8_t* output,
    size_t input_increment,
    size_t output_increment,
    const union pytorch_qnnp_avgpool_quantization_params quantization_params[restrict static 1])
{
  const int32_t bias = quantization_params->scalar.bias;

  do {
    const uint8_t* rows[9];
    for (size_t p = 0; p < 9; p++) {
      rows[p] = input[p];
    }
    if (ks < 2) rows[1] = zero;
    if (ks <= 2) rows[2] = zero;
    if (ks < 4) rows[3] = zero;
    if (ks <= 4) rows[4] = zero;
    if (ks < 6) rows[5] = zero;
    if (ks <= 6) rows[6] = zero;
    if (ks < 8) rows[7] = zero;
    if (ks <= 8) rows[8] = zero;

    for (size_t c = 0; c < kc; c++) {
      int32_t acc = bias;
      for (size_t p = 0; p < 9; p++) {
        acc += (int32_t) (uint32_t) rows[p][c];
      }
      output[c] = scalar_avgpool_requantize(acc, quantization_params);
    }

    input = (const uint8_t**) ((uintptr_t) input + input_increment);
    output = (uint8_t*) ((uintptr_t) output + kc + output_increment);
  } while (--n != 0);
}


/*
 * mp8x9p8q: channels >= kr and pooling_size > mr.
 * Process first 9 rows, then chunks of 8 rows, then tail 8 rows with
 * zero substitution. Accumulates per-channel in `buffer`.
 */
void pytorch_q8avgpool_ukernel_mp8x9p8q__scalar(
    size_t n,
    size_t ks,
    size_t kc,
    const uint8_t** input,
    const uint8_t* zero,
    int32_t* buffer,
    uint8_t* output,
    size_t input_increment,
    size_t output_increment,
    const union pytorch_qnnp_avgpool_quantization_params quantization_params[restrict static 1])
{
  const int32_t bias = quantization_params->scalar.bias;

  do {
    /* First block: 9 real rows (ks > 9 guaranteed) */
    {
      const uint8_t* rows[9];
      for (size_t p = 0; p < 9; p++) {
        rows[p] = *input++;
      }
      for (size_t c = 0; c < kc; c++) {
        int32_t acc = bias;
        for (size_t p = 0; p < 9; p++) {
          acc += (int32_t) (uint32_t) rows[p][c];
        }
        buffer[c] = acc;
      }
    }

    /* Middle: full chunks of 8 rows */
    size_t m;
    for (m = ks - 9; m > 8; m -= 8) {
      const uint8_t* rows[8];
      for (size_t p = 0; p < 8; p++) {
        rows[p] = *input++;
      }
      for (size_t c = 0; c < kc; c++) {
        int32_t acc = buffer[c];
        for (size_t p = 0; p < 8; p++) {
          acc += (int32_t) (uint32_t) rows[p][c];
        }
        buffer[c] = acc;
      }
    }

    /* Tail: 8 entries, substitute zero for unused (based on m in [1,8]) */
    {
      const uint8_t* rows[8];
      for (size_t p = 0; p < 8; p++) {
        rows[p] = input[p];
      }
      if (m < 2) rows[1] = zero;
      if (m <= 2) rows[2] = zero;
      if (m < 4) rows[3] = zero;
      if (m <= 4) rows[4] = zero;
      if (m < 6) rows[5] = zero;
      if (m <= 6) rows[6] = zero;
      if (m != 8) rows[7] = zero;

      for (size_t c = 0; c < kc; c++) {
        int32_t acc = buffer[c];
        for (size_t p = 0; p < 8; p++) {
          acc += (int32_t) (uint32_t) rows[p][c];
        }
        output[c] = scalar_avgpool_requantize(acc, quantization_params);
      }
    }

    input = (const uint8_t**) ((uintptr_t) input + input_increment);
    output = (uint8_t*) ((uintptr_t) output + kc + output_increment);
  } while (--n != 0);
}
