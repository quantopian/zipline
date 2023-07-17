"""
Loaders for zipline.pipeline.data.testing datasets.
"""
from .synthetic import EyeLoader, SeededRandomLoader
from ..data.testing import TestingDataSet


def make_eye_loader(dates, sids):
    """
    Make a PipelineLoader that emits np.eye arrays for the columns in
    ``TestingDataSet``.
    """
    return EyeLoader(TestingDataSet.columns, dates, sids)


def make_seeded_random_loader(seed, dates, sids, columns=TestingDataSet.columns):
    """
    Make a PipelineLoader that emits random arrays seeded with `seed` for the
    columns in ``TestingDataSet``.
    """
    return SeededRandomLoader(seed, columns, dates, sids)
