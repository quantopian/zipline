"""
Dataset representing dates of upcoming earnings.
"""
from zipline.utils.numpy_utils import datetime64ns_dtype, float64_dtype

from .dataset import Column, DataSet


class CashBuybackAuthorizations(DataSet):
    """
    Dataset representing dates of recently announced buyback authorization.
    """
    previous_value = Column(float64_dtype)
    previous_announcement_date = Column(datetime64ns_dtype)


class ShareBuybackAuthorizations(DataSet):
    previous_share_count = Column(float64_dtype)
    previous_announcement_date = Column(datetime64ns_dtype)
