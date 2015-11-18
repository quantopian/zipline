"""
Dataset representing dates of upcoming earnings.
"""
from zipline.utils.numpy_utils import datetime64ns_dtype

from .dataset import Column, DataSet


class EarningsCalendar(DataSet):
    """
    Dataset representing dates of upcoming or recently announced earnings.
    """
    next_announcement = Column(datetime64ns_dtype)
    previous_announcement = Column(datetime64ns_dtype)

    # TODO: Provide categorical columns for when during the day the
    # announcement occurred.
