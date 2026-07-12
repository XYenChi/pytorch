/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <stddef.h>
#include <stdint.h>

#include <qnnpack/u8clamp.h>


void pytorch_u8clamp_ukernel__scalar(
    size_t n,
    const uint8_t* x,
    uint8_t* y,
    const union pytorch_qnnp_u8_clamping_params params[restrict static 1])
{
  const int32_t ymin = params->scalar.output_min;
  const int32_t ymax = params->scalar.output_max;
  while (n-- != 0) {
    int32_t v = (int32_t) (uint32_t) *x++;
    if (v < ymin) v = ymin;
    if (v > ymax) v = ymax;
    *y++ = (uint8_t) v;
  }
}
