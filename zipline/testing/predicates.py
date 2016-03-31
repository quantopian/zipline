from six import iteritems, viewkeys, PY2

from zipline.dispatch import dispatch
from zipline.utils.functional import dzip_exact


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


@dispatch(object, object)
def assert_equal(result, expected, path=(), **kwargs):
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
    assert result == expected, '%s != %s\n%s' % (
        result,
        expected,
        _fmt_path(path),
    )


@assert_equal.register(dict, dict)
def assert_dict_equal(result, expected, path=(), **kwargs):
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
            'dict keys do not match\n%s\n%s' % (
                msg,
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
                **kwargs
            )
        except AssertionError as e:
            failures.append(str(e))

    if failures:
        raise AssertionError('\n'.join(failures))


@assert_equal.register(list, list)  # noqa
def assert_list_equal(result, expected, path=(), **kwargs):
    result_len = len(result)
    expected_len = len(expected)
    assert result_len == expected_len, (
        'list lengths do not match: %d != %d\n%s' %
        result_len,
        expected_len,
        _fmt_path(path),
    )

    for n, (resultv, expectedv) in enumerate(zip(result, expected)):
        assert_equal(
            resultv,
            expectedv,
            path=path + ('[%d]' % n,),
            **kwargs
        )


try:
    # pull the dshape cases in
    from datashape.util.testing import assert_dshape_equal
except ImportError:
    pass
else:
    assert_equal.funcs.update(assert_dshape_equal.funcs)
