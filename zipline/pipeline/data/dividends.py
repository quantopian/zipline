"""
Dataset representing dates of upcoming dividends.
"""
from zipline.utils.numpy_utils import datetime64ns_dtype, float64_dtype

from .dataset import Column, DataSet


class DividendsByExDate(DataSet):
    next_date = Column(datetime64ns_dtype)
    previous_date = Column(datetime64ns_dtype)
    next_amount = Column(float64_dtype)
    previous_amount = Column(float64_dtype)


class DividendsByPayDate(DataSet):
    next_date = Column(datetime64ns_dtype)
    previous_date = Column(datetime64ns_dtype)
    next_amount = Column(float64_dtype)
    previous_amount = Column(float64_dtype)


class DividendsByAnnouncementDate(DataSet):
    previous_announcement_date = Column(datetime64ns_dtype)
    previous_amount = Column(float64_dtype)
