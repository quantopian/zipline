import operator as op

from six import PY2
from toolz import peek

from zipline.utils.functional import foldr


if PY2:
    class range(object):
        """Lazy range object with constant time containment check.

        The arguments are the same as ``range``.
        """
        __slots__ = 'start', 'stop', 'step'

        def __init__(self, stop, *args):
            if len(args) > 2:
                raise TypeError(
                    'range takes at most 3 arguments (%d given)' % len(args)
                )

            if not args:
                self.start = 0
                self.stop = stop
                self.step = 1
            else:
                self.start = stop
                self.stop = args[0]
                try:
                    self.step = args[1]
                except IndexError:
                    self.step = 1

            if self.step == 0:
                raise ValueError('range step must not be zero')

        def __iter__(self):
            """
            Examples
            --------
            >>> list(range(1))
            [0]
            >>> list(range(5))
            [0, 1, 2, 3, 4]
            >>> list(range(1, 5))
            [1, 2, 3, 4]
            >>> list(range(0, 5, 2))
            [0, 2, 4]
            >>> list(range(5, 0, -1))
            [5, 4, 3, 2, 1]
            >>> list(range(5, 0, 1))
            []
            """
            n = self.start
            stop = self.stop
            step = self.step
            cmp_ = op.lt if step > 0 else op.gt
            while cmp_(n, stop):
                yield n
                n += step

        _ops = (
            (op.gt, op.ge),
            (op.le, op.lt),
        )

        def __contains__(self, other, _ops=_ops):
            # Algorithm taken from CPython
            # Objects/rangeobject.c:range_contains_long
            start = self.start
            step = self.step
            cmp_start, cmp_stop = _ops[step > 0]
            return (
                cmp_start(start, other) and
                cmp_stop(other, self.stop) and
                (other - start) % step == 0
            )

        del _ops

        def __len__(self):
            """
            Examples
            --------
            >>> len(range(1))
            1
            >>> len(range(5))
            5
            >>> len(range(1, 5))
            4
            >>> len(range(0, 5, 2))
            3
            >>> len(range(5, 0, -1))
            5
            >>> len(range(5, 0, 1))
            0
            """
            # Algorithm taken from CPython
            # rangeobject.c:compute_range_length
            step = self.step

            if step > 0:
                low = self.start
                high = self.stop
            else:
                low = self.stop
                high = self.start
                step = -step

            if low >= high:
                return 0

            return (high - low - 1) // step + 1

        def __repr__(self):
            return '%s(%s, %s%s)' % (
                type(self).__name__,
                self.start,
                self.stop,
                (', ' + str(self.step)) if self.step != 1 else '',
            )

        def __hash__(self):
            return hash((type(self), self.start, self.stop, self.step))

        def __eq__(self, other):
            """
            Examples
            --------
            >>> range(1) == range(1)
            True
            >>> range(0, 5, 2) == range(0, 5, 2)
            True
            >>> range(5, 0, -2) == range(5, 0, -2)
            True

            >>> range(1) == range(2)
            False
            >>> range(0, 5, 2) == range(0, 5, 3)
            False
            """
            return all(
                getattr(self, attr) == getattr(other, attr)
                for attr in self.__slots__
            )
else:
    range = range


def from_tuple(tup):
    """Convert a tuple into a range with error handling.

    Parameters
    ----------
    tup : tuple (len 2 or 3)
        The tuple to turn into a range.

    Returns
    -------
    range : range
        The range from the tuple.

    Raises
    ------
    ValueError
        Raised when the tuple length is not 2 or 3.
    """
    if len(tup) not in (2, 3):
        raise ValueError(
            'tuple must contain 2 or 3 elements, not: %d (%r' % (
                len(tup),
                tup,
            ),
        )
    return range(*tup)


def maybe_from_tuple(tup_or_range):
    """Convert a tuple into a range but pass ranges through silently.

    This is useful to ensure that input is a range so that attributes may
    be accessed with `.start`, `.stop` or so that containment checks are
    constant time.

    Parameters
    ----------
    tup_or_range : tuple or range
        A tuple to pass to from_tuple or a range to return.

    Returns
    -------
    range : range
        The input to convert to a range.

    Raises
    ------
    ValueError
        Raised when the input is not a tuple or a range. ValueError is also
        raised if the input is a tuple whose length is not 2 or 3.
    """
    if isinstance(tup_or_range, tuple):
        return from_tuple(tup_or_range)
    elif isinstance(tup_or_range, range):
        return tup_or_range

    raise ValueError(
        'maybe_from_tuple expects a tuple or range, got %r: %r' % (
            type(tup_or_range).__name__,
            tup_or_range,
        ),
    )


def _check_steps(a, b):
    """Check that the steps of ``a`` and ``b`` are both 1.

    Parameters
    ----------
    a : range
        The first range to check.
    b : range
        The second range to check.

    Raises
    ------
    ValueError
        Raised when either step is not 1.
    """
    if a.step != 1:
        raise ValueError('a.step must be equal to 1, got: %s' % a.step)
    if b.step != 1:
        raise ValueError('b.step must be equal to 1, got: %s' % b.step)


def overlap(a, b):
    """Check if  two ranges overlap.

    Parameters
    ----------
    a : range
        The first range.
    b : range
        The second range.

    Returns
    -------
    overlaps : bool
        Do these ranges overlap.

    Notes
    -----
    This function does not support ranges with step != 1.
    """
    _check_steps(a, b)
    return a.stop >= b.start and b.stop >= a.start


def merge(a, b):
    """Merge two ranges with step == 1.

    Parameters
    ----------
    a : range
        The first range.
    b : range
        The second range.
    """
    _check_steps(a, b)
    return range(min(a.start, b.start), max(a.stop, b.stop))


def _combine(n, rs):
    """helper for ``_group_ranges``
    """
    try:
        r, rs = peek(rs)
    except StopIteration:
        yield n
        return

    if overlap(n, r):
        yield merge(n, r)
        next(rs)
        for r in rs:
            yield r
    else:
        yield n
        for r in rs:
            yield r


def group_ranges(ranges):
    """Group any overlapping ranges into a single range.

    Parameters
    ----------
    ranges : iterable[ranges]
        A sorted sequence of ranges to group.

    Returns
    -------
    grouped : iterable[ranges]
        A sorted sequence of ranges with overlapping ranges merged together.
    """
    return foldr(_combine, ranges, ())


def sorted_diff(rs, ss):
    try:
        r, rs = peek(rs)
    except StopIteration:
        return

    try:
        s, ss = peek(ss)
    except StopIteration:
        for r in rs:
            yield r
        return

    rtup = (r.start, r.stop)
    stup = (s.start, s.stop)
    if rtup == stup:
        next(rs)
        next(ss)
    elif rtup < stup:
        yield next(rs)
    else:
        next(ss)

    for t in sorted_diff(rs, ss):
        yield t


def intersecting_ranges(ranges):
    """Return any ranges that intersect.

    Parameters
    ----------
    ranges : iterable[ranges]
        A sequence of ranges to check for intersections.

    Returns
    -------
    intersections : iterable[ranges]
        A sequence of all of the ranges that intersected in ``ranges``.

    Examples
    --------
    >>> ranges = [range(0, 1), range(2, 5), range(4, 7)]
    >>> list(intersecting_ranges(ranges))
    [range(2, 5), range(4, 7)]

    >>> ranges = [range(0, 1), range(2, 3)]
    >>> list(intersecting_ranges(ranges))
    []

    >>> ranges = [range(0, 1), range(1, 2)]
    >>> list(intersecting_ranges(ranges))
    [range(0, 1), range(1, 2)]
    """
    ranges = sorted(ranges, key=op.attrgetter('start'))
    return sorted_diff(ranges, group_ranges(ranges))
