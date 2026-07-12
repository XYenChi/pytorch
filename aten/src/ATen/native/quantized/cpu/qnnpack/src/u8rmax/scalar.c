/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <stddef.h>
#include <stdint.h>

#include <qnnpack/u8rmax.h>


uint8_t pytorch_u8rmax_ukernel__scalar(size_t n, const uint8_t* x) {
  uint8_t m = 0;
  while (n-- != 0) {
    const uint8_t v = *x++;
    if (v > m) m = v;
  }
  return m;
}
