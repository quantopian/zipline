"""
Filter that produces the most most recently-known value of a boolean-valued
Column.
"""
from zipline.utils.numpy_utils import bool_dtype

from .filter import CustomFilter
from ..mixins import SingleInputMixin


class Latest(SingleInputMixin, CustomFilter):
    """
    Filter producing the most recently-known value of `inputs[0]` on each day.
    """
    window_length = 1

    def compute(self, today, assets, out, data):
        out[:] = data[-1]

    def _validate(self):
        if self.inputs[0].dtype != bool_dtype:
            raise TypeError(
                "{name} expected an input of dtype bool, "
                "but got {not_bool} instead.".format(
                    name=type(self).__name__,
                    not_bool=self.inputs[0].dtype,
                )
            )
        super(Latest, self)._validate()
