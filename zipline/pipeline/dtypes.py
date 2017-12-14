from zipline.utils.numpy_utils import (
    bool_dtype,
    datetime64ns_dtype,
    float64_dtype,
    int64_dtype,
    object_dtype,
)

CLASSIFIER_DTYPES = frozenset({object_dtype, int64_dtype})
FACTOR_DTYPES = frozenset({datetime64ns_dtype, float64_dtype, int64_dtype})
FILTER_DTYPES = frozenset({bool_dtype})
