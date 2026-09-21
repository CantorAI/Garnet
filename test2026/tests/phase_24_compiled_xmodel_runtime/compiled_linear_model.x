# SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
# SPDX-License-Identifier: Apache-2.0

from garnet import garnet

T = garnet.tensor()


@T.fusion()
def Model(x, weight):
    return x * T.binary_op("linear") * weight
