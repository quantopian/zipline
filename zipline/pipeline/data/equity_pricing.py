from numpy import float64, uint32

from .dataset import Column, DataSet


class USEquityPricing(DataSet):

    open = Column(float64)
    high = Column(float64)
    low = Column(float64)
    close = Column(float64)
    volume = Column(uint32)
