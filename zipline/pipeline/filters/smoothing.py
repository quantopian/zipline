"""
Filters that apply smoothing operations on other filters.

These are generally useful for controlling/minimizing turnover on existing
Filters.
"""
from .filter import CustomFilter


class All(CustomFilter):
    """
    A Filter requiring that assets produce True for ``window_length``
    consecutive days.

    **Default Inputs:** None

    **Default Window Length:** None
    """

    def compute(self, today, assets, out, arg):
        out[:] = arg.sum(axis=0) == self.window_length


class Any(CustomFilter):
    """
    A Filter requiring that assets produce True for at least one day in the
    last ``window_length`` days.

    **Default Inputs:** None

    **Default Window Length:** None
    """

    def compute(self, today, assets, out, arg):
        out[:] = arg.sum(axis=0) > 0


class AtLeastN(CustomFilter):
    """
    A Filter requiring that assets produce True for at least N days in the
    last ``window_length`` days.

    **Default Inputs:** None

    **Default Window Length:** None
    """

    params = ("N",)

    def compute(self, today, assets, out, arg, N):
        out[:] = arg.sum(axis=0) >= N
