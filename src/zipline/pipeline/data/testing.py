"""
Datasets for testing use.

Loaders for datasets in this file can be found in
zipline.pipeline.data.testing.
"""
from .dataset import Column, DataSet
from zipline.utils.numpy_utils import (
    bool_dtype,
    categorical_dtype,
    float64_dtype,
    datetime64ns_dtype,
    int64_dtype,
)


class TestingDataSet(DataSet):
    # Tell nose this isn't a test case.
    __test__ = False

    bool_col = Column(dtype=bool_dtype, missing_value=False)
    bool_col_default_True = Column(dtype=bool_dtype, missing_value=True)

    float_col = Column(dtype=float64_dtype)

    datetime_col = Column(dtype=datetime64ns_dtype)

    int_col = Column(dtype=int64_dtype, missing_value=0)

    categorical_col = Column(dtype=categorical_dtype)
    categorical_default_explicit_None = Column(
        dtype=categorical_dtype,
        missing_value=None,
    )
    categorical_default_NULL_string = Column(
        dtype=categorical_dtype,
        missing_value="<<NULL>>",
    )
