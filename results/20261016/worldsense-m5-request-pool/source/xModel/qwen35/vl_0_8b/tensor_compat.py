"""Pass tensor operator attributes through XLang3's supported kwargs path."""

import garnet


class TensorCompat:
    def __init__(self, native_tensor):
        self._native_tensor = native_tensor

    def __getattr__(self, name):
        return getattr(self._native_tensor, name)

    def unary_op(self, name, **attributes):
        return self._native_tensor.unary_op(name, **attributes)

    def binary_op(self, name, **attributes):
        return self._native_tensor.binary_op(name, **attributes)


def tensor():
    return TensorCompat(garnet.tensor())
