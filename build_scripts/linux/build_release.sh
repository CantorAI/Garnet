#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
build_dir="${repo_root}/out/build/linux-Release"
xlang3_root="${XLANG3_ROOT:-${repo_root}/../xlang3}"
tensorrt_root="${GARNET_TENSORRT_ROOT:-${repo_root}/../ThirdPartySDK/TensorRT}"
target="${1:-garnet}"

cmake -S "${repo_root}" -B "${build_dir}" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_RUNTIME_OUTPUT_DIRECTORY="${build_dir}/bin" \
  -DCMAKE_LIBRARY_OUTPUT_DIRECTORY="${build_dir}/bin" \
  -DXLANG3_ROOT="${xlang3_root}" \
  -DGARNET_TENSORRT_ROOT="${tensorrt_root}"

cmake --build "${build_dir}" --target "${target}"
