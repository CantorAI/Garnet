#!/bin/bash
# SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

echo "Garnet currently requires CUDA and cannot be built natively on macOS." >&2
exit 1
