#ifdef USE_XNNPACK

#include <ATen/native/xnnpack/Common.h>
#include <ATen/native/xnnpack/Engine.h>
#include <ATen/native/utils/Factory.h>

namespace at::native::xnnpack {

bool use_channel_shuffle(
    const Tensor& input,
    const int64_t groups) {
    return false;
}

Tensor channel_shuffle(
    const Tensor& input,
    const int64_t groups) {
  TORCH_CHECK(false, "XNNPACK channel_shuffle is unavailable in this XNNPACK fork");
}

} // namespace at::native::xnnpack

#endif /* USE_XNNPACK */
