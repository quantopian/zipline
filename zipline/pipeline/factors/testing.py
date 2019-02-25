import numpy as np

from zipline.testing.predicates import assert_equal
from .factor import CustomFactor


class IDBox(object):
    """A wrapper that hashs to the id of the underlying object and compares
    equality on the id of the underlying.

    Parameters
    ----------
    ob : any
        The object to wrap.

    Attributes
    ----------
    ob : any
        The object being wrapped.

    Notes
    -----
    This is useful for storing non-hashable values in a set or dict.
    """
    def __init__(self, ob):
        self.ob = ob

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        if not isinstance(other, IDBox):
            return NotImplemented

        return id(self.ob) == id(other.ob)


class CheckWindowsFactor(CustomFactor):
    """A custom factor that makes assertions about the lookback windows that
    it gets passed.

    Parameters
    ----------
    input_ : Term
        The input term to the factor.
    window_length : int
        The length of the lookback window.
    expected_windows : dict[int, dict[pd.Timestamp, np.ndarray]]
        For each asset, for each day, what the expected lookback window is.

    Notes
    -----
    The output of this factor is the same as ``Latest``. Any assets or days
    not in ``expected_windows`` are not checked.
    """
    params = ('expected_windows',)

    def __new__(cls, input_, window_length, expected_windows):
        return super(CheckWindowsFactor, cls).__new__(
            cls,
            inputs=[input_],
            dtype=input_.dtype,
            window_length=window_length,
            expected_windows=frozenset(
                (k, IDBox(v)) for k, v in expected_windows.items()
            ),
        )

    def compute(self, today, assets, out, input_, expected_windows):
        for asset, expected_by_day in expected_windows:
            expected_by_day = expected_by_day.ob

            col_ix = np.searchsorted(assets, asset)
            if assets[col_ix] != asset:
                raise AssertionError('asset %s is not in the window' % asset)

            try:
                expected = expected_by_day[today]
            except KeyError:
                pass
            else:
                expected = np.array(expected)
                actual = input_[:, col_ix]
                assert_equal(actual, expected)

        # output is just latest
        out[:] = input_[-1]
