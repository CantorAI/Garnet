"""Pass tensor operator attributes through XLang3's supported kwargs path."""

import garnet


class TensorCompat:
    def __init__(self, native_tensor):
        self._native_tensor = native_tensor

    def __getattr__(self, name):
        return getattr(self._native_tensor, name)

    def unary_op(self, name, **attributes):
        # The preserved runtime's direct property-call fast path rejects the
        # native factory. Materializing the callable uses its supported path.
        factory = self._native_tensor.unary_op
        return factory(name, **attributes)

    def binary_op(self, name, **attributes):
        factory = self._native_tensor.binary_op
        return factory(name, **attributes)


def tensor():
    return TensorCompat(garnet.tensor())
