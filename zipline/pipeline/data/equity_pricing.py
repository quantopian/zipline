"""
Dataset representing OHLCV data.
"""
from zipline.utils.numpy_utils import float64_dtype

from ..domain import USEquities
from .dataset import Column, DataSet


class EquityPricing(DataSet):
    """
    Dataset representing daily trading prices and volumes.
    """
    open = Column(float64_dtype)
    high = Column(float64_dtype)
    low = Column(float64_dtype)
    close = Column(float64_dtype)
    volume = Column(float64_dtype)


# Backwards compat alias.
USEquityPricing = EquityPricing.specialize(USEquities)
