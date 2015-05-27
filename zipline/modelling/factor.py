"""
factor.py
"""
from zipline.modelling.computable import Term


class Factor(Term):
    """
    A transformation yielding a timeseries of scalar values associated with an
    Asset.
    """
    pass


class TestFactor(Factor):
    """
    Base class for testing that asserts all inputs are correctly shaped.
    """

    def compute_from_windows(self, windows, outbuf, dates, assets):

        for idx, _ in enumerate(dates):
            result = self.compute(*(next(w) for w in windows))
            assert result.shape == (len(assets),)
            outbuf[idx] = result

        for window in windows:
            try:
                next(window)
            except StopIteration:
                pass
            else:
                raise AssertionError("window %s was not exhausted" % window)
        return outbuf
