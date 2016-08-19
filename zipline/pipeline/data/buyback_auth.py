"""
Datasets representing dates of recently announced buyback authorizations.
"""
from zipline.utils.numpy_utils import (
    datetime64ns_dtype,
    float64_dtype,
    categorical_dtype
)

from .dataset import Column, DataSet


class BuybackAuthorizations(DataSet):
    """
    Dataset representing dates of recently announced cash buyback
    authorizations.
    """
    previous_amount = Column(float64_dtype)
    previous_date = Column(datetime64ns_dtype)
    previous_unit = Column(categorical_dtype, missing_value=None)
    previous_type = Column(categorical_dtype, missing_value=None)
