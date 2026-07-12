/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Scalar (portable C) implementations of q8gavgpool microkernels.
 * Sums m rows of n channel uint8 values, then quantizes.
 */

#include <stddef.h>
#include <stdint.h>

#include <qnnpack/q8gavgpool.h>
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

void pytorch_q8gavgpool_ukernel_up8xm__scalar(
    size_t m,
    size_t n,
    const uint8_t* x,
    size_t x_stride,
    const uint8_t* zero,
    uint8_t* y,
    const union pytorch_qnnp_avgpool_quantization_params quantization_params[restrict static 1])
{
  const int32_t bias = quantization_params->scalar.bias;

  for (size_t c = 0; c < n; c++) {
    int32_t acc = bias;
    const uint8_t* row = x;
    for (size_t i = 0; i < m; i++) {
      acc += (int32_t) (uint32_t) row[c];
      row = (const uint8_t*) ((uintptr_t) row + x_stride);
    }
    y[c] = scalar_avgpool_requantize(acc, quantization_params);
  }
  (void) zero;
}


void pytorch_q8gavgpool_ukernel_up8x7__scalar(
    size_t m,
    size_t n,
    const uint8_t* x,
    size_t x_stride,
    const uint8_t* zero,
    uint8_t* y,
    const union pytorch_qnnp_avgpool_quantization_params quantization_params[restrict static 1])
{
  pytorch_q8gavgpool_ukernel_up8xm__scalar(
      m, n, x, x_stride, zero, y, quantization_params);
}


void pytorch_q8gavgpool_ukernel_mp8x7p7q__scalar(
    size_t m,
    size_t n,
    const uint8_t* x,
    size_t x_stride,
    const uint8_t* zero,
    int32_t* buffer,
    uint8_t* y,
    const union pytorch_qnnp_avgpool_quantization_params quantization_params[restrict static 1])
{
  (void) buffer;
  pytorch_q8gavgpool_ukernel_up8xm__scalar(
      m, n, x, x_stride, zero, y, quantization_params);
}
