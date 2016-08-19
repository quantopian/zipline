from functools import partial
import inspect

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
    assert_sequence_equal,
    assert_set_equal,
    assert_true,
    assert_tuple_equal,
)
import numpy as np
import pandas as pd
from pandas.util.testing import assert_frame_equal
from six import iteritems, viewkeys, PY2
from toolz import dissoc, keyfilter
import toolz.curried.operator as op

from zipline.dispatch import dispatch
from zipline.lib.adjustment import Adjustment
from zipline.utils.functional import dzip_exact
from zipline.utils.math_utils import tolerant_equals


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


@assert_equal.register(dict, dict)
def assert_dict_equal(result, expected, path=(), msg='', **kwargs):
    if path is None:
        path = ()

    result_keys = viewkeys(result)
    expected_keys = viewkeys(expected)
    if result_keys != expected_keys:
        if result_keys > expected_keys:
            diff = result_keys - expected_keys
            msg = 'extra %s in result: %r' % (_s('key', diff), diff)
        elif result_keys < expected_keys:
            diff = expected_keys - result_keys
            msg = 'result is missing %s: %r' % (_s('key', diff), diff)
        else:
            sym = result_keys ^ expected_keys
            in_result = sym - expected_keys
            in_expected = sym - result_keys
            msg = '%s only in result: %s\n%s only in expected: %s' % (
                _s('key', in_result),
                in_result,
                _s('key', in_expected),
                in_expected,
            )
        raise AssertionError(
            '%sdict keys do not match\n%s' % (
                _fmt_msg(msg),
                _fmt_path(path + ('.%s()' % ('viewkeys' if PY2 else 'keys'),)),
            ),
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


@assert_equal.register(list, list)  # noqa
def assert_list_equal(result, expected, path=(), msg='', **kwargs):
    result_len = len(result)
    expected_len = len(expected)
    assert result_len == expected_len, (
        '%slist lengths do not match: %d != %d\n%s' % (
            _fmt_msg(msg),
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


@assert_equal.register(pd.DataFrame, pd.DataFrame)
def assert_dataframe_equal(result, expected, path=(), msg='', **kwargs):
    try:
        assert_frame_equal(
            result,
            expected,
            **filter_kwargs(assert_frame_equal, kwargs)
        )
    except AssertionError as e:
        raise AssertionError(
            _fmt_msg(msg) + '\n'.join((str(e), _fmt_path(path))),
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


try:
    # pull the dshape cases in
    from datashape.util.testing import assert_dshape_equal
except ImportError:
    pass
else:
    assert_equal.funcs.update(
        dissoc(assert_dshape_equal.funcs, (object, object)),
    )
