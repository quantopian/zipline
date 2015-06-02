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
        assert self.window_length > 0
        for idx, _ in enumerate(dates):
            result = self.from_windows(*(next(w) for w in windows))
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

    def compute_from_arrays(self, arrays, outbuf, dates, assets):
        assert self.window_length == 0
        for array in arrays:
            assert array.shape == len(dates), len(assets) == outbuf.shape
        outbuf[:] = self.from_arrays(*arrays)
