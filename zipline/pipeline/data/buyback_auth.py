"""
Dataset representing dates of upcoming earnings.
"""
from zipline.utils.numpy_utils import datetime64ns_dtype, float64_dtype

from .dataset import Column, DataSet


class BuybackAuthorizations(DataSet):
    """
    Dataset representing dates of recently announced buyback authorization.
    """
    previous_buyback_value = Column(float64_dtype)
    previous_buyback_share_count = Column(float64_dtype)
    previous_buyback_value_announcement = Column(datetime64ns_dtype)
    previous_buyback_share_count_announcement = Column(datetime64ns_dtype)

