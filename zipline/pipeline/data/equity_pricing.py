"""
Dataset representing OHLCV data.
"""
from zipline.utils.numpy_utils import float64_dtype

from .dataset import Column, DataSet


class USEquityPricing(DataSet):
    """
    Dataset representing daily trading prices and volumes.
    """
    open = Column(float64_dtype)
    high = Column(float64_dtype)
    low = Column(float64_dtype)
    close = Column(float64_dtype)
    volume = Column(float64_dtype)
