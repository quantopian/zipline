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
