"""
Utilities for creating public APIs (e.g. argument validation decorators).
"""
from zipline.utils.input_validation import preprocess


def restrict_to_dtype(dtype, message_template):
    """
    A factory for decorators that restrict Term methods to only be callable on
    Terms with a specific dtype.

    This is conceptually similar to
    zipline.utils.input_validation.expect_dtypes, but provides more flexibility
    for providing error messages that are specifically targeting Term methods.

    Parameters
    ----------
    dtype : numpy.dtype
        The dtype on which the decorated method may be called.
    message_template : str
        A template for the error message to be raised.
        `message_template.format` will be called with keyword arguments
        `method_name`, `expected_dtype`, and `received_dtype`.

    Examples
    --------
    @restrict_to_dtype(
        dtype=float64_dtype,
        message_template=(
            "{method_name}() was called on a factor of dtype {received_dtype}."
            "{method_name}() requires factors of dtype{expected_dtype}."

        ),
    )
    def some_factor_method(self, ...):
        self.stuff_that_requires_being_float64(...)
    """
    def processor(term_method, _, term_instance):
        term_dtype = term_instance.dtype
        if term_dtype != dtype:
            raise TypeError(
                message_template.format(
                    method_name=term_method.__name__,
                    expected_dtype=dtype.name,
                    received_dtype=term_dtype,
                )
            )
        return term_instance
    return preprocess(self=processor)
