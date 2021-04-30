from interface import implements

from .base import FXRateReader


class ExplodingFXRateReader(implements(FXRateReader)):
    """An FXRateReader that raises an error when used.

    This is useful for testing contexts where FX rates aren't actually needed.
    """

    def get_rates(self, rate, quote, bases, dts):
        raise AssertionError("FX rates requested unexpectedly!")
