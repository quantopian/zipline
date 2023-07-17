import numpy as np

from zipline.pipeline.factors.factor import CustomFactor
from zipline.pipeline.classifiers.classifier import CustomClassifier
from zipline.utils.idbox import IDBox

from .predicates import assert_equal


class CheckWindowsMixin:
    params = ("expected_windows",)

    def compute(self, today, assets, out, input_, expected_windows):
        for asset, expected_by_day in expected_windows:
            expected_by_day = expected_by_day.ob

            col_ix = np.searchsorted(assets, asset)
            if assets[col_ix] != asset:
                raise AssertionError("asset %s is not in the window" % asset)

            try:
                expected = expected_by_day[today]
            except KeyError:
                pass
            else:
                expected = np.asanyarray(expected)
                actual = input_[:, col_ix]
                assert_equal(
                    actual,
                    expected,
                    array_decimal=(6 if expected.dtype.kind == "f" else None),
                )

        # output is just latest
        out[:] = input_[-1]


class CheckWindowsClassifier(CheckWindowsMixin, CustomClassifier):
    """A custom classifier that makes assertions about the lookback windows that
    it gets passed.

    Parameters
    ----------
    input_ : Term
        The input term to the classifier.
    window_length : int
        The length of the lookback window.
    expected_windows : dict[int, dict[pd.Timestamp, np.ndarray]]
        For each asset, for each day, what the expected lookback window is.

    Notes
    -----
    The output of this classifier is the same as ``Latest``. Any assets or days
    not in ``expected_windows`` are not checked.
    """

    def __new__(cls, input_, window_length, expected_windows):
        if input_.dtype.kind == "V":
            dtype = np.dtype("O")
        else:
            dtype = input_.dtype

        return super(CheckWindowsClassifier, cls).__new__(
            cls,
            inputs=[input_],
            dtype=dtype,
            window_length=window_length,
            expected_windows=frozenset(
                (k, IDBox(v)) for k, v in expected_windows.items()
            ),
        )


class CheckWindowsFactor(CheckWindowsMixin, CustomFactor):
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
