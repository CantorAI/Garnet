# SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
# SPDX-License-Identifier: Apache-2.0

import garnet

T = garnet.tensor()


@T.fusion(name="tiny_matmul", role="integration")
def forward(left, right):
    return left @ right
