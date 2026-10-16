// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <array>
#include <string>
#include <vector>
namespace Garnet::Image {
struct NativeImageConfig {
  int patch = 16, merge = 2, temporal = 2, pixels = 65536;
  std::array<float, 3> mean{.5f, .5f, .5f}, stddev{.5f, .5f, .5f};
};
struct NativeImageResult {
  std::vector<float> patches;
  int gridH = 0, gridW = 0, features = 0;
  double decodeMs = 0, resizeMs = 0, packMs = 0;
};
NativeImageResult PrepareNativeImage(const std::string &,
                                     const NativeImageConfig &);
} // namespace Garnet::Image
