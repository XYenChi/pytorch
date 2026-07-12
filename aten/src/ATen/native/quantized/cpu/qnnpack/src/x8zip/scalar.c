/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <stddef.h>
#include <stdint.h>

#include <qnnpack/x8zip.h>


void pytorch_qnnp_x8zip_x2__scalar(size_t n, const void* input, void* output) {
  const uint8_t* x = (const uint8_t*) input;
  const uint8_t* y = x + n;
  uint8_t* o = (uint8_t*) output;
  while (n-- != 0) {
    o[0] = *x++;
    o[1] = *y++;
    o += 2;
  }
}

void pytorch_qnnp_x8zip_x3__scalar(size_t n, const void* input, void* output) {
  const uint8_t* x = (const uint8_t*) input;
  const uint8_t* y = x + n;
  const uint8_t* z = y + n;
  uint8_t* o = (uint8_t*) output;
  while (n-- != 0) {
    o[0] = *x++;
    o[1] = *y++;
    o[2] = *z++;
    o += 3;
  }
}

void pytorch_qnnp_x8zip_x4__scalar(size_t n, const void* input, void* output) {
  const uint8_t* x = (const uint8_t*) input;
  const uint8_t* y = x + n;
  const uint8_t* z = y + n;
  const uint8_t* w = z + n;
  uint8_t* o = (uint8_t*) output;
  while (n-- != 0) {
    o[0] = *x++;
    o[1] = *y++;
    o[2] = *z++;
    o[3] = *w++;
    o += 4;
  }
}

void pytorch_qnnp_x8zip_xm__scalar(size_t n, size_t m, const void* input, void* output) {
  const uint8_t* in = (const uint8_t*) input;
  uint8_t* o = (uint8_t*) output;
  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < m; j++) {
      o[j] = in[j * n + i];
    }
    o += m;
  }
}
