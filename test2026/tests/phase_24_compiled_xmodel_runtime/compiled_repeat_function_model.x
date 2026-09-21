# SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
# SPDX-License-Identifier: Apache-2.0

from garnet import garnet

T = garnet.tensor()


def DoubleBlock(x):
    return x + x


@T.fusion()
def Model(x):
    output = x
    for layer_index in range(5):
        output = DoubleBlock(output)
    return output
