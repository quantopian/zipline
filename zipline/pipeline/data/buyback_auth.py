"""
Datasets representing dates of recently announced buyback authorizations.
"""
from zipline.utils.numpy_utils import datetime64ns_dtype, float64_dtype

from .dataset import Column, DataSet


class CashBuybackAuthorizations(DataSet):
    """
    Dataset representing dates of recently announced cash buyback
    authorizations.
    """
    cash_amount = Column(float64_dtype)
    announcement_date = Column(datetime64ns_dtype)


class ShareBuybackAuthorizations(DataSet):
    """
    Dataset representing dates of recently announced share buyback
    authorizations.
    """
    share_count = Column(float64_dtype)
    announcement_date = Column(datetime64ns_dtype)
