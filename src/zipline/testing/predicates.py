from collections import OrderedDict

# from contextlib import contextmanager
import datetime
from functools import partial

import numpy as np
import pandas as pd
from pandas.testing import (
    assert_frame_equal,
    assert_series_equal,
    assert_index_equal,
)
from itertools import zip_longest
from toolz import keyfilter
import toolz.curried.operator as op

from zipline.assets import Asset
from zipline.dispatch import dispatch
from zipline.lib.adjustment import Adjustment
from zipline.lib.labelarray import LabelArray
from zipline.testing.core import ensure_doctest
from zipline.utils.compat import getargspec, mappingproxy
from zipline.utils.formatting import s
from zipline.utils.functional import dzip_exact, instance
from zipline.utils.math_utils import tolerant_equals
from zipline.utils.numpy_utils import (
    assert_array_compare,
    compare_datetime_arrays,
)


@instance
@ensure_doctest
class wildcard:
    """An object that compares equal to any other object.

    This is useful when using :func:`~zipline.testing.predicates.assert_equal`
    with a large recursive structure and some fields to be ignored.

    Examples
    --------
    >>> wildcard == 5
    True
    >>> wildcard == 'ayy'
    True

    # reflected
    >>> 5 == wildcard
    True
    >>> 'ayy' == wildcard
    True
    """

    @staticmethod
    def __eq__(other):
        return True

    @staticmethod
    def __ne__(other):
        return False

    def __repr__(self):
        return "<%s>" % type(self).__name__


class instance_of:
    """An object that compares equal to any instance of a given type or types.

    Parameters
    ----------
    types : type or tuple[type]
        The types to compare equal to.
    exact : bool, optional
        Only compare equal to exact instances, not instances of subclasses?
    """

    def __init__(self, types, exact=False):
        if not isinstance(types, tuple):
            types = (types,)

        for type_ in types:
            if not isinstance(type_, type):
                raise TypeError("types must be a type or tuple of types")

        self.types = types
        self.exact = exact

    def __eq__(self, other):
        if self.exact:
            return type(other) in self.types

        return isinstance(other, self.types)

    def __ne__(self, other):
        return not self == other

    def __repr__(self):
        typenames = tuple(t.__name__ for t in self.types)
        return "%s(%s%s)" % (
            type(self).__name__,
            (typenames[0] if len(typenames) == 1 else "(%s)" % ", ".join(typenames)),
            ", exact=True" if self.exact else "",
        )


def keywords(func):
    """Get the argument names of a function

    >>> def f(x, y=2):
    ...     pass

    >>> keywords(f)
    ['x', 'y']

    Notes
    -----
    Taken from odo.utils
    """
    if isinstance(func, type):
        return keywords(func.__init__)
    elif isinstance(func, partial):
        return keywords(func.func)
    return getargspec(func).args


def filter_kwargs(f, kwargs):
    """Return a dict of valid kwargs for `f` from a subset of `kwargs`

    Examples
    --------
    >>> def f(a, b=1, c=2):
    ...     return a + b + c
    ...
    >>> raw_kwargs = dict(a=1, b=3, d=4)
    >>> f(**raw_kwargs)
    Traceback (most recent call last):
        ...
    TypeError: f() got an unexpected keyword argument 'd'
    >>> kwargs = filter_kwargs(f, raw_kwargs)
    >>> f(**kwargs)
    6

    Notes
    -----
    Taken from odo.utils
    """
    return keyfilter(op.contains(keywords(f)), kwargs)


def _fmt_path(path):
    """Format the path for final display.

    Parameters
    ----------
    path : iterable of str
        The path to the values that are not equal.

    Returns
    -------
    fmtd : str
        The formatted path to put into the error message.
    """
    if not path:
        return ""
    return "path: _" + "".join(path)


def _fmt_msg(msg):
    """Format the message for final display.

    Parameters
    ----------
    msg : str
        The message to show to the user to provide additional context.

    returns
    -------
    fmtd : str
        The formatted message to put into the error message.
    """
    if not msg:
        return ""
    return msg + "\n"


def make_assert_equal_assertion_error(assertion_message, path, msg):
    """Create an assertion error formatted for use in ``assert_equal``.

    Parameters
    ----------
    assertion_message : str
        The concrete reason for the failure.
    path : tuple[str]
        The path leading up to the failure.
    msg : str
        The user supplied message.

    Returns
    -------
    exception_instance : AssertionError
        The new exception instance.

    Notes
    -----
    This doesn't raise the exception, it only returns it.
    """
    return AssertionError(
        "%s%s\n%s"
        % (
            _fmt_msg(msg),
            assertion_message,
            _fmt_path(path),
        ),
    )


@dispatch(object, object)
def assert_equal(result, expected, path=(), msg="", **kwargs):
    """Assert that two objects are equal using the ``==`` operator.

    Parameters
    ----------
    result : object
        The result that came from the function under test.
    expected : object
        The expected result.

    Raises
    ------
    AssertionError
        Raised when ``result`` is not equal to ``expected``.
    """
    if result != expected:
        raise make_assert_equal_assertion_error(
            "%s != %s" % (result, expected),
            path,
            msg,
        )


@assert_equal.register(float, float)
def assert_float_equal(
    result,
    expected,
    path=(),
    msg="",
    float_rtol=10e-7,
    float_atol=10e-7,
    float_equal_nan=True,
    **kwargs,
):
    assert tolerant_equals(
        result,
        expected,
        rtol=float_rtol,
        atol=float_atol,
        equal_nan=float_equal_nan,
    ), "%s%s != %s with rtol=%s and atol=%s%s\n%s" % (
        _fmt_msg(msg),
        result,
        expected,
        float_rtol,
        float_atol,
        (" (with nan != nan)" if not float_equal_nan else ""),
        _fmt_path(path),
    )


def _check_sets(result, expected, msg, path, type_):
    """Compare two sets. This is used to check dictionary keys and sets.

    Parameters
    ----------
    result : set
    expected : set
    msg : str
    path : tuple
    type : str
        The type of an element. For dict we use ``'key'`` and for set we use
        ``'element'``.
    """
    if result != expected:
        if result > expected:
            diff = result - expected
            msg = "extra %s in result: %r" % (s(type_, diff), diff)
        elif result < expected:
            diff = expected - result
            msg = "result is missing %s: %r" % (s(type_, diff), diff)
        else:
            in_result = result - expected
            in_expected = expected - result
            msg = "%s only in result: %s\n%s only in expected: %s" % (
                s(type_, in_result),
                in_result,
                s(type_, in_expected),
                in_expected,
            )
        raise AssertionError(
            "%ss do not match\n%s%s"
            % (
                type_,
                _fmt_msg(msg),
                _fmt_path(path),
            ),
        )


@assert_equal.register(dict, dict)
def assert_dict_equal(result, expected, path=(), msg="", **kwargs):
    _check_sets(
        result.keys(),
        expected.keys(),
        msg,
        path + (".keys()",),
        "key",
    )

    failures = []
    for k, (resultv, expectedv) in dzip_exact(result, expected).items():
        try:
            assert_equal(
                resultv,
                expectedv,
                path=path + ("[%r]" % (k,),),
                msg=msg,
                **kwargs,
            )
        except AssertionError as e:
            failures.append(str(e))

    if failures:
        raise AssertionError("\n===\n".join(failures))


@assert_equal.register(mappingproxy, mappingproxy)
def asssert_mappingproxy_equal(result, expected, path=(), msg="", **kwargs):
    # mappingproxies compare like dict but shouldn't compare to dicts
    _check_sets(
        set(result),
        set(expected),
        msg,
        path + (".keys()",),
        "key",
    )

    failures = []
    for k, resultv in result.items():
        # we know this exists because of the _check_sets call above
        expectedv = expected[k]

        try:
            assert_equal(
                resultv,
                expectedv,
                path=path + ("[%r]" % (k,),),
                msg=msg,
                **kwargs,
            )
        except AssertionError as e:
            failures.append(str(e))

    if failures:
        raise AssertionError("\n".join(failures))


@assert_equal.register(OrderedDict, OrderedDict)
def assert_ordereddict_equal(result, expected, path=(), **kwargs):
    assert_sequence_equal(
        result.items(), expected.items(), path=path + (".items()",), **kwargs
    )


@assert_equal.register(list, list)
@assert_equal.register(tuple, tuple)
def assert_sequence_equal(result, expected, path=(), msg="", **kwargs):
    result_len = len(result)
    expected_len = len(expected)
    assert result_len == expected_len, "%s%s lengths do not match: %d != %d\n%s" % (
        _fmt_msg(msg),
        type(result).__name__,
        result_len,
        expected_len,
        _fmt_path(path),
    )
    for n, (resultv, expectedv) in enumerate(zip(result, expected)):
        assert_equal(resultv, expectedv, path=path + ("[%d]" % n,), msg=msg, **kwargs)


@assert_equal.register(set, set)
def assert_set_equal(result, expected, path=(), msg="", **kwargs):
    _check_sets(
        result,
        expected,
        msg,
        path,
        "element",
    )


@assert_equal.register(np.ndarray, np.ndarray)
def assert_array_equal(
    result,
    expected,
    path=(),
    msg="",
    array_verbose=True,
    array_decimal=None,
    **kwargs,
):
    result_dtype = result.dtype
    expected_dtype = expected.dtype

    if result_dtype.kind in "mM" and expected_dtype.kind in "mM":
        assert result_dtype == expected_dtype, (
            "\nType mismatch:\n\n"
            "result dtype: %s\n"
            "expected dtype: %s\n%s" % (result_dtype, expected_dtype, _fmt_path(path))
        )

        f = partial(
            assert_array_compare,
            compare_datetime_arrays,
            header="Arrays are not equal",
        )
    elif array_decimal is not None and expected_dtype.kind not in {"O", "S"}:
        f = partial(
            np.testing.assert_array_almost_equal,
            decimal=array_decimal,
        )
    else:
        f = np.testing.assert_array_equal

    try:
        f(
            result,
            expected,
            verbose=array_verbose,
            err_msg=msg,
        )
    except AssertionError as exc:
        raise AssertionError("\n".join((str(exc), _fmt_path(path)))) from exc


@assert_equal.register(LabelArray, LabelArray)
def assert_labelarray_equal(result, expected, path=(), **kwargs):
    assert_equal(
        result.categories,
        expected.categories,
        path=path + (".categories",),
        **kwargs,
    )
    assert_equal(
        result.as_int_array(),
        expected.as_int_array(),
        path=path + (".as_int_array()",),
        **kwargs,
    )


def _register_assert_equal_wrapper(type_, assert_eq):
    """Register a new check for an ndframe object.

    Parameters
    ----------
    type_ : type
        The class to register an ``assert_equal`` dispatch for.
    assert_eq : callable[type_, type_]
        The function which checks that if the two ndframes are equal.

    Returns
    -------
    assert_ndframe_equal : callable[type_, type_]
        The wrapped function registered with ``assert_equal``.
    """

    @assert_equal.register(type_, type_)
    def assert_ndframe_equal(result, expected, path=(), msg="", **kwargs):
        try:
            assert_eq(result, expected, **filter_kwargs(assert_eq, kwargs))
        except AssertionError as exc:
            raise AssertionError(
                _fmt_msg(msg) + "\n".join((str(exc), _fmt_path(path))),
            ) from exc

    return assert_ndframe_equal


assert_frame_equal = _register_assert_equal_wrapper(
    pd.DataFrame,
    assert_frame_equal,
)
assert_series_equal = _register_assert_equal_wrapper(
    pd.Series,
    assert_series_equal,
)
assert_index_equal = _register_assert_equal_wrapper(
    pd.Index,
    assert_index_equal,
)


@assert_equal.register(pd.Categorical, pd.Categorical)
def assert_categorical_equal(result, expected, path=(), msg="", **kwargs):
    assert_equal(
        result.categories,
        expected.categories,
        path=path + (".categories",),
        msg=msg,
        **kwargs,
    )
    assert_equal(
        result.codes, expected.codes, path=path + (".codes",), msg=msg, **kwargs
    )


@assert_equal.register(Adjustment, Adjustment)
def assert_adjustment_equal(result, expected, path=(), **kwargs):
    for attr in ("first_row", "last_row", "first_col", "last_col", "value"):
        assert_equal(
            getattr(result, attr),
            getattr(expected, attr),
            path=path + ("." + attr,),
            **kwargs,
        )


@assert_equal.register(
    (datetime.datetime, np.datetime64),
    (datetime.datetime, np.datetime64),
)
def assert_timestamp_and_datetime_equal(
    result,
    expected,
    path=(),
    msg="",
    allow_datetime_coercions=False,
    compare_nat_equal=True,
    **kwargs,
):
    """
    Branch for comparing python datetime (which includes pandas Timestamp) and
    np.datetime64 as equal.

    Returns raises unless ``allow_datetime_coercions`` is passed as True.
    """
    assert allow_datetime_coercions or type(result) == type(expected), (
        "%sdatetime types (%s, %s) don't match and "
        "allow_datetime_coercions was not set.\n%s"
        % (
            _fmt_msg(msg),
            type(result),
            type(expected),
            _fmt_path(path),
        )
    )

    if isinstance(result, pd.Timestamp) and isinstance(expected, pd.Timestamp):
        assert_equal(result.tz, expected.tz, path=path + (".tz",), msg=msg, **kwargs)

    result = pd.Timestamp(result)
    expected = pd.Timestamp(expected)
    if compare_nat_equal and pd.isnull(result) and pd.isnull(expected):
        return

    assert_equal.dispatch(object, object)(
        result, expected, path=path, msg=msg, **kwargs
    )


@assert_equal.register(slice, slice)
def assert_slice_equal(result, expected, path=(), msg=""):
    diff_start = (
        ("starts are not equal: %s != %s" % (result.start, result.stop))
        if result.start != expected.start
        else ""
    )
    diff_stop = (
        ("stops are not equal: %s != %s" % (result.stop, result.stop))
        if result.stop != expected.stop
        else ""
    )
    diff_step = (
        ("steps are not equal: %s != %s" % (result.step, result.stop))
        if result.step != expected.step
        else ""
    )
    diffs = diff_start, diff_stop, diff_step

    assert not any(diffs), "%s%s\n%s" % (
        _fmt_msg(msg),
        "\n".join(filter(None, diffs)),
        _fmt_path(path),
    )


@assert_equal.register(Asset, Asset)
def assert_asset_equal(result, expected, path=(), msg="", **kwargs):
    if type(result) is not type(expected):
        raise AssertionError(
            "%sresult type differs from expected type: %s is not %s\n%s",
            _fmt_msg(msg),
            type(result).__name__,
            type(expected).__name__,
            _fmt_path(path),
        )

    assert_equal(
        result.to_dict(),
        expected.to_dict(),
        path=path + (".to_dict()",),
        msg=msg,
        **kwargs,
    )


def assert_isidentical(result, expected, msg=""):
    assert result.isidentical(expected), "%s%s is not identical to %s" % (
        _fmt_msg(msg),
        result,
        expected,
    )


def assert_messages_equal(result, expected, msg=""):
    """Assertion helper for comparing very long strings (e.g. error messages)."""
    # The arg here is "keepends" which keeps trailing newlines (which
    # matters for checking trailing whitespace). You can't pass keepends by
    # name :(.
    left_lines = result.splitlines(True)
    right_lines = expected.splitlines(True)
    iter_lines = enumerate(zip_longest(left_lines, right_lines))
    for line, (ll, rl) in iter_lines:
        if ll != rl:
            col = index_of_first_difference(ll, rl)
            raise AssertionError(
                "{msg}Messages differ on line {line}, col {col}:"
                "\n{ll!r}\n!=\n{rl!r}".format(
                    msg=_fmt_msg(msg), line=line, col=col, ll=ll, rl=rl
                )
            )


def index_of_first_difference(left, right):
    """Get the index of the first difference between two strings."""
    difflocs = (i for (i, (lc, rc)) in enumerate(zip_longest(left, right)) if lc != rc)
    try:
        return next(difflocs)
    except StopIteration as exc:
        raise ValueError("Left was equal to right!") from exc
