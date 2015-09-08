"""
Factor that produces the most most recently-known value of Column.
"""
from .factor import Factor
from ..term import SingleInputMixin


class Latest(SingleInputMixin, Factor):
    """
    Factor producing the most recently-known value of `inputs[0]` on each day.
    """
    window_length = 0

    def _compute(self, arrays, mask):
        return arrays[0]
