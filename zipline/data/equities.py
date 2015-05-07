from numpy import (
    float32,
    uint32,
)

from zipline.data.dataset import (
    Column,
    DataSet,
)


class USEquityPricing(DataSet):

    open = Column(float32)
    high = Column(float32)
    low = Column(float32)
    close = Column(float32)
    volume = Column(uint32)
