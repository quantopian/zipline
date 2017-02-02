from contextlib import contextmanager
import datetime
from functools import partial
import inspect
import re

from nose.tools import (  # noqa
    assert_almost_equal,
    assert_almost_equals,
    assert_dict_contains_subset,
    assert_false,
    assert_greater,
    assert_greater_equal,
    assert_in,
    assert_is,
    assert_is_instance,
    assert_is_none,
    assert_is_not,
    assert_is_not_none,
    assert_less,
    assert_less_equal,
    assert_multi_line_equal,
    assert_not_almost_equal,
    assert_not_almost_equals,
    assert_not_equal,
    assert_not_equals,
    assert_not_in,
    assert_not_is_instance,
    assert_raises,
    assert_raises_regexp,
    assert_regexp_matches,
    assert_true,
    assert_tuple_equal,
)
import numpy as np
import pandas as pd
from pandas.util.testing import (
    assert_frame_equal,
    assert_panel_equal,
    assert_series_equal,
    assert_index_equal,
)
from six import iteritems, viewkeys, PY2
from toolz import dissoc, keyfilter
import toolz.curried.operator as op

from zipline.testing.core import ensure_doctest
from zipline.dispatch import dispatch
from zipline.lib.adjustment import Adjustment
from zipline.lib.labelarray import LabelArray
from zipline.utils.functional import dzip_exact, instance
from zipline.utils.math_utils import tolerant_equals


@instance
@ensure_doctest
class wildcard(object):
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
        return '<%s>' % type(self).__name__
    __str__ = __repr__


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
    return inspect.getargspec(func).args


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


def _s(word, seq, suffix='s'):
    """Adds a suffix to ``word`` if some sequence has anything other than
    exactly one element.

    word : str
        The string to add the suffix to.
    seq : sequence
        The sequence to check the length of.
    suffix : str, optional.
        The suffix to add to ``word``

    Returns
    -------
    maybe_plural : str
        ``word`` with ``suffix`` added if ``len(seq) != 1``.
    """
    return word + (suffix if len(seq) != 1 else '')


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
        return ''
    return 'path: _' + ''.join(path)


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
        return ''
    return msg + '\n'


def _safe_cls_name(cls):
    try:
        return cls.__name__
    except AttributeError:
        return repr(cls)


def assert_is_subclass(subcls, cls, msg=''):
    """Assert that ``subcls`` is a subclass of ``cls``.

    Parameters
    ----------
    subcls : type
        The type to check.
    cls : type
        The type to check ``subcls`` against.
    msg : str, optional
        An extra assertion message to print if this fails.
    """
    assert issubclass(subcls, cls), (
        '%s is not a subclass of %s\n%s' % (
            _safe_cls_name(subcls),
            _safe_cls_name(cls),
            msg,
        )
    )


def assert_regex(result, expected, msg=''):
    """Assert that ``expected`` matches the result.

    Parameters
    ----------
    result : str
        The string to search.
    expected : str or compiled regex
        The pattern to search for in ``result``.
    msg : str, optional
        An extra assertion message to print if this fails.
    """
    assert re.search(expected, result), (
        '%s%r not found in %r' % (_fmt_msg(msg), expected, result)
    )


@contextmanager
def assert_raises_regex(exc, pattern, msg=''):
    """Assert that some exception is raised in a context and that the message
    matches some pattern.

    Parameters
    ----------
    exc : type or tuple[type]
        The exception type or types to expect.
    pattern : str or compiled regex
        The pattern to search for in the str of the raised exception.
    msg : str, optional
        An extra assertion message to print if this fails.
    """
    try:
        yield
    except exc as e:
        assert re.search(pattern, str(e)), (
            '%s%r not found in %r' % (_fmt_msg(msg), pattern, str(e))
        )
    else:
        raise AssertionError('%s%s was not raised' % (_fmt_msg(msg), exc))


@dispatch(object, object)
def assert_equal(result, expected, path=(), msg='', **kwargs):
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
    assert result == expected, '%s%s != %s\n%s' % (
        _fmt_msg(msg),
        result,
        expected,
        _fmt_path(path),
    )


@assert_equal.register(float, float)
def assert_float_equal(result,
                       expected,
                       path=(),
                       msg='',
                       float_rtol=10e-7,
                       float_atol=10e-7,
                       float_equal_nan=True,
                       **kwargs):
    assert tolerant_equals(
        result,
        expected,
        rtol=float_rtol,
        atol=float_atol,
        equal_nan=float_equal_nan,
    ), '%s%s != %s with rtol=%s and atol=%s%s\n%s' % (
        _fmt_msg(msg),
        result,
        expected,
        float_rtol,
        float_atol,
        (' (with nan != nan)' if not float_equal_nan else ''),
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
            msg = 'extra %s in result: %r' % (_s(type_, diff), diff)
        elif result < expected:
            diff = expected - result
            msg = 'result is missing %s: %r' % (_s(type_, diff), diff)
        else:
            in_result = result - expected
            in_expected = expected - result
            msg = '%s only in result: %s\n%s only in expected: %s' % (
                _s(type_, in_result),
                in_result,
                _s(type_, in_expected),
                in_expected,
            )
        raise AssertionError(
            '%s%ss do not match\n%s' % (
                _fmt_msg(msg),
                type_,
                _fmt_path(path),
            ),
        )


@assert_equal.register(dict, dict)
def assert_dict_equal(result, expected, path=(), msg='', **kwargs):
    _check_sets(
        viewkeys(result),
        viewkeys(expected),
        msg,
        path + ('.%s()' % ('viewkeys' if PY2 else 'keys'),),
        'key',
    )

    failures = []
    for k, (resultv, expectedv) in iteritems(dzip_exact(result, expected)):
        try:
            assert_equal(
                resultv,
                expectedv,
                path=path + ('[%r]' % k,),
                msg=msg,
                **kwargs
            )
        except AssertionError as e:
            failures.append(str(e))

    if failures:
        raise AssertionError('\n'.join(failures))


@assert_equal.register(list, list)
@assert_equal.register(tuple, tuple)
def assert_sequence_equal(result, expected, path=(), msg='', **kwargs):
    result_len = len(result)
    expected_len = len(expected)
    assert result_len == expected_len, (
        '%s%s lengths do not match: %d != %d\n%s' % (
            _fmt_msg(msg),
            type(result).__name__,
            result_len,
            expected_len,
            _fmt_path(path),
        )
    )
    for n, (resultv, expectedv) in enumerate(zip(result, expected)):
        assert_equal(
            resultv,
            expectedv,
            path=path + ('[%d]' % n,),
            msg=msg,
            **kwargs
        )


@assert_equal.register(set, set)
def assert_set_equal(result, expected, path=(), msg='', **kwargs):
    _check_sets(
        result,
        expected,
        msg,
        path,
        'element',
    )


@assert_equal.register(np.ndarray, np.ndarray)
def assert_array_equal(result,
                       expected,
                       path=(),
                       msg='',
                       array_verbose=True,
                       array_decimal=None,
                       **kwargs):
    f = (
        np.testing.assert_array_equal
        if array_decimal is None else
        partial(np.testing.assert_array_almost_equal, decimal=array_decimal)
    )
    try:
        f(
            result,
            expected,
            verbose=array_verbose,
            err_msg=msg,
        )
    except AssertionError as e:
        raise AssertionError('\n'.join((str(e), _fmt_path(path))))


@assert_equal.register(LabelArray, LabelArray)
def assert_labelarray_equal(result, expected, path=(), **kwargs):
    assert_equal(
        result.categories,
        expected.categories,
        path=path + ('.categories',),
        **kwargs
    )
    assert_equal(
        result.as_int_array(),
        expected.as_int_array(),
        path=path + ('.as_int_array()',),
        **kwargs
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
    def assert_ndframe_equal(result, expected, path=(), msg='', **kwargs):
        try:
            assert_eq(
                result,
                expected,
                **filter_kwargs(assert_eq, kwargs)
            )
        except AssertionError as e:
            raise AssertionError(
                _fmt_msg(msg) + '\n'.join((str(e), _fmt_path(path))),
            )

    return assert_ndframe_equal


assert_frame_equal = _register_assert_equal_wrapper(
    pd.DataFrame,
    assert_frame_equal,
)
assert_panel_equal = _register_assert_equal_wrapper(
    pd.Panel,
    assert_panel_equal,
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
def assert_categorical_equal(result, expected, path=(), msg='', **kwargs):
    assert_equal(
        result.categories,
        expected.categories,
        path=path + ('.categories',),
        msg=msg,
        **kwargs
    )
    assert_equal(
        result.codes,
        expected.codes,
        path=path + ('.codes',),
        msg=msg,
        **kwargs
    )


@assert_equal.register(Adjustment, Adjustment)
def assert_adjustment_equal(result, expected, path=(), **kwargs):
    for attr in ('first_row', 'last_row', 'first_col', 'last_col', 'value'):
        assert_equal(
            getattr(result, attr),
            getattr(expected, attr),
            path=path + ('.' + attr,),
            **kwargs
        )


@assert_equal.register(
    (datetime.datetime, np.datetime64),
    (datetime.datetime, np.datetime64),
)
def assert_timestamp_and_datetime_equal(result,
                                        expected,
                                        path=(),
                                        msg='',
                                        allow_datetime_coercions=False,
                                        compare_nat_equal=True,
                                        **kwargs):
    """
    Branch for comparing python datetime (which includes pandas Timestamp) and
    np.datetime64 as equal.

    Returns raises unless ``allow_datetime_coercions`` is passed as True.
    """
    assert allow_datetime_coercions or type(result) == type(expected), (
        "%sdatetime types (%s, %s) don't match and "
        "allow_datetime_coercions was not set.\n%s" % (
            _fmt_msg(msg),
            type(result),
            type(expected),
            _fmt_path(path),
        )
    )

    result = pd.Timestamp(result)
    expected = pd.Timestamp(result)
    if compare_nat_equal and pd.isnull(result) and pd.isnull(expected):
        return

    assert_equal.dispatch(object, object)(
        result,
        expected,
        path=path,
        **kwargs
    )


@assert_equal.register(slice, slice)
def assert_slice_equal(result, expected, path=(), msg=''):
    diff_start = (
        ('starts are not equal: %s != %s' % (result.start, result.stop))
        if result.start != expected.start else
        ''
    )
    diff_stop = (
        ('stops are not equal: %s != %s' % (result.stop, result.stop))
        if result.stop != expected.stop else
        ''
    )
    diff_step = (
        ('steps are not equal: %s != %s' % (result.step, result.stop))
        if result.step != expected.step else
        ''
    )
    diffs = diff_start, diff_stop, diff_step

    assert not any(diffs), '%s%s\n%s' % (
        _fmt_msg(msg),
        '\n'.join(filter(None, diffs)),
        _fmt_path(path),
    )


def assert_isidentical(result, expected, msg=''):
    assert result.isidentical(expected), (
        '%s%s is not identical to %s' % (_fmt_msg(msg), result, expected)
    )


try:
    # pull the dshape cases in
    from datashape.util.testing import assert_dshape_equal
except ImportError:
    pass
else:
    assert_equal.funcs.update(
        dissoc(assert_dshape_equal.funcs, (object, object)),
    )
