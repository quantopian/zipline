from six.moves import map, zip
from toolz import curry


@curry
def apply(f, *args, **kwargs):
    """Apply a function to arguments.

    Parameters
    ----------
    f : callable
        The function to call.
    *args, **kwargs
    **kwargs
        Arguments to feed to the callable.

    Returns
    -------
    a : any
        The result of ``f(*args, **kwargs)``

    Examples
    --------
    >>> from toolz.curried.operator import add, sub
    >>> fs = add(1), sub(1)
    >>> tuple(map(apply, fs, (1, 2)))
    (2, -1)

    Class decorator
    >>> instance = apply
    >>> @instance
    ... class obj:
    ...     def f(self):
    ...         return 'f'
    ...
    >>> obj.f()
    'f'
    >>> issubclass(obj, object)
    Traceback (most recent call last):
        ...
    TypeError: issubclass() arg 1 must be a class
    >>> isinstance(obj, type)
    False

    See Also
    --------
    unpack_apply
    mapply
    """
    return f(*args, **kwargs)


# Alias for use as a class decorator.
instance = apply


def mapall(funcs, seq):
    """
    Parameters
    ----------
    funcs : iterable[function]
        Sequence of functions to map over `seq`.
    seq : iterable
        Sequence over which to map funcs.

    Yields
    ------
    elem : object
        Concatenated result of mapping each ``func`` over ``seq``.

    Example
    -------
    >>> list(mapall([lambda x: x + 1, lambda x: x - 1], [1, 2, 3]))
    [2, 3, 4, 0, 1, 2]
    """
    for func in funcs:
        for elem in seq:
            yield func(elem)


def _gen_unzip(it, elem_len):
    """Helper for unzip which checks the lengths of each element in it.

    Parameters
    ----------
    it : iterable[tuple]
        An iterable of tuples. ``unzip`` should map ensure that these are
        already tuples.
    elem_len : int or None
        The expected element length. If this is None it is infered from the
        length of the first element.

    Yields
    ------
    elem : tuple
        Each element of ``it``.

    Raises
    ------
    ValueError
        Raised when the lengths do not match the ``elem_len``.
    """
    elem = next(it)
    first_elem_len = len(elem)

    if elem_len is not None and elem_len != first_elem_len:
        raise ValueError(
            'element at index 0 was length %d, expected %d' % (
                first_elem_len,
                elem_len,
            )
        )
    else:
        elem_len = first_elem_len

    yield elem
    for n, elem in enumerate(it, 1):
        if len(elem) != elem_len:
            raise ValueError(
                'element at index %d was length %d, expected %d' % (
                    n,
                    len(elem),
                    elem_len,
                ),
            )
        yield elem


def unzip(seq, elem_len=None):
    """Unzip a length n sequence of length m sequences into m seperate length
    n sequences.

    Parameters
    ----------
    seq : iterable[iterable]
        The sequence to unzip.
    elem_len : int, optional
        The expected length of each element of ``seq``. If not provided this
        will be infered from the length of the first element of ``seq``. This
        can be used to ensure that code like: ``a, b = unzip(seq)`` does not
        fail even when ``seq`` is empty.

    Returns
    -------
    seqs : iterable[iterable]
        The new sequences pulled out of the first iterable.

    Raises
    ------
    ValueError
        Raised when ``seq`` is empty and ``elem_len`` is not provided.
        Raised when elements of ``seq`` do not match the given ``elem_len`` or
        the length of the first element of ``seq``.

    Examples
    --------
    >>> seq = [('a', 1), ('b', 2), ('c', 3)]
    >>> cs, ns = unzip(seq)
    >>> cs
    ('a', 'b', 'c')
    >>> ns
    (1, 2, 3)

    # checks that the elements are the same length
    >>> seq = [('a', 1), ('b', 2), ('c', 3, 'extra')]
    >>> cs, ns = unzip(seq)
    Traceback (most recent call last):
      ...
    ValueError: element at index 2 was length 3, expected 2

    # allows an explicit element length instead of infering
    >>> seq = [('a', 1, 'extra'), ('b', 2), ('c', 3)]
    >>> cs, ns = unzip(seq, 2)
    Traceback (most recent call last):
      ...
    ValueError: element at index 0 was length 3, expected 2

    # handles empty sequences when a length is given
    >>> cs, ns = unzip([], elem_len=2)
    >>> cs == ns == ()
    True

    Notes
    -----
    This function will force seq to completion.
    """
    ret = tuple(zip(*_gen_unzip(map(tuple, seq), elem_len)))
    if ret:
        return ret

    if elem_len is None:
        raise ValueError("cannot unzip empty sequence without 'elem_len'")
    return ((),) * elem_len
