"""
Datasets for testing use.

Loaders for datasets in this file can be found in
zipline.pipeline.data.testing.
"""
from .dataset import Column, DataSet
from zipline.utils.numpy_utils import (
    bool_dtype,
    float64_dtype,
    datetime64ns_dtype,
    int64_dtype,
)


class TestingDataSet(DataSet):

    bool_col = Column(dtype=bool_dtype)
    float_col = Column(dtype=float64_dtype)
    datetime_col = Column(dtype=datetime64ns_dtype)
    int_col = Column(dtype=int64_dtype)
