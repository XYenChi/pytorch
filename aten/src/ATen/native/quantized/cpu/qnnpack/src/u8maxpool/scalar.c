/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Scalar (portable C) implementation of u8maxpool microkernels.
 */

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#include <qnnpack/u8maxpool.h>


/*
 * sub16: channels < 16. Per output pixel: for each of kc channels, take max
 * over `ks` input rows; clamp to [ymin, ymax]. Consumes `ks` input pointers
 * via *input++ to match the SSE2 contract (operator-run computes
 * input_increment assuming the kernel consumed `pooling_size` pointers).
 */
void pytorch_u8maxpool_ukernel_sub16__scalar(
    size_t n,
    size_t ks,
    size_t kc,
    const uint8_t** input,
    uint8_t* output,
    size_t input_increment,
    size_t output_increment,
    const union pytorch_qnnp_u8_clamping_params params[restrict static 1])
{
  const uint8_t ymin = params->scalar.output_min;
  const uint8_t ymax = params->scalar.output_max;
  do {
    const uint8_t** rows = input;
    input += ks;
    for (size_t c = 0; c < kc; c++) {
      uint8_t m = 0;
      for (size_t p = 0; p < ks; p++) {
        const uint8_t v = rows[p][c];
        if (v > m) m = v;
      }
      if (m < ymin) m = ymin;
      if (m > ymax) m = ymax;
      output[c] = m;
    }
    input = (const uint8_t**) ((uintptr_t) input + input_increment);
    output = (uint8_t*) ((uintptr_t) output + kc + output_increment);
  } while (--n != 0);
}


/*
 * 16x9p8q: channels >= 16. Processes first 9 input rows, then chunks of 8
 * rows until ks rows have been covered, substituting the first row pointer
 * for any unused slots so that max() is unaffected. Pointer advancement
 * matches the SSE2 reference (consumes mp_adj = round_up(doz(ks,9),8)+9
 * pointers per output pixel).
 */
void pytorch_u8maxpool_ukernel_16x9p8q__scalar(
    size_t n,
    size_t ks,
    size_t kc,
    const uint8_t** input,
    uint8_t* output,
    size_t input_increment,
    size_t output_increment,
    const union pytorch_qnnp_u8_clamping_params params[restrict static 1])
{
  const uint8_t ymin = params->scalar.output_min;
  const uint8_t ymax = params->scalar.output_max;
  do {
    /* First block: 9 pointers; substitute i0 for slots beyond ks. */
    const uint8_t* r[9];
    for (size_t p = 0; p < 9; p++) {
      r[p] = *input++;
    }
    if (ks < 2) r[1] = r[0];
    if (ks <= 2) r[2] = r[0];
    if (ks < 4) r[3] = r[0];
    if (ks <= 4) r[4] = r[0];
    if (ks < 6) r[5] = r[0];
    if (ks <= 6) r[6] = r[0];
    if (ks < 8) r[7] = r[0];
    if (ks <= 8) r[8] = r[0];

    for (size_t c = 0; c < kc; c++) {
      uint8_t m = 0;
      for (size_t p = 0; p < 9; p++) {
        const uint8_t v = r[p][c];
        if (v > m) m = v;
      }
      if (m < ymin) m = ymin;
      if (m > ymax) m = ymax;
      output[c] = m;
    }

    /* Trailing blocks of 8 rows each, while there is real data left. */
    for (ptrdiff_t mleft = (ptrdiff_t) ks - 9; mleft > 0; mleft -= 8) {
      const uint8_t* t[8];
      for (size_t p = 0; p < 8; p++) {
        t[p] = *input++;
      }
      if (mleft < 2) t[1] = t[0];
      if (mleft <= 2) t[2] = t[0];
      if (mleft < 4) t[3] = t[0];
      if (mleft <= 4) t[4] = t[0];
      if (mleft < 6) t[5] = t[0];
      if (mleft <= 6) t[6] = t[0];
      if (mleft < 8) t[7] = t[0];

      for (size_t c = 0; c < kc; c++) {
        uint8_t m = output[c];
        for (size_t p = 0; p < 8; p++) {
          const uint8_t v = t[p][c];
          if (v > m) m = v;
        }
        if (m < ymin) m = ymin;
        if (m > ymax) m = ymax;
        output[c] = m;
      }
    }

    input = (const uint8_t**) ((uintptr_t) input + input_increment);
    output = (uint8_t*) ((uintptr_t) output + kc + output_increment);
  } while (--n != 0);
}
