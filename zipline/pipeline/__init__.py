from __future__ import print_function
from zipline.assets import AssetFinder

from .classifiers import Classifier, CustomClassifier
from .engine import SimplePipelineEngine
from .factors import Factor, CustomFactor
from .filters import Filter, CustomFilter
from .term import Term
from .graph import TermGraph
from .pipeline import Pipeline
from .loaders import USEquityPricingLoader


def engine_from_files(daily_bar_path,
                      adjustments_path,
                      asset_db_path,
                      calendar,
                      warmup_assets=False):
    """
    Construct a SimplePipelineEngine from local filesystem resources.

    Parameters
    ----------
    daily_bar_path : str
        Path to pass to `BcolzDailyBarReader`.
    adjustments_path : str
        Path to pass to SQLiteAdjustmentReader.
    asset_db_path : str
        Path to pass to `AssetFinder`.
    calendar : pd.DatetimeIndex
        Calendar to use for the loader.
    warmup_assets : bool, optional
        Whether or not to populate AssetFinder caches.  This can speed up
        initial latency on subsequent pipeline runs, at the cost of extra
        memory consumption.  Default is False
    """
    loader = USEquityPricingLoader.from_files(daily_bar_path, adjustments_path)

    if not asset_db_path.startswith("sqlite:"):
        asset_db_path = "sqlite:///" + asset_db_path
    asset_finder = AssetFinder(asset_db_path)
    if warmup_assets:
        results = asset_finder.retrieve_all(asset_finder.sids)
        print("Warmed up %d assets." % len(results))

    return SimplePipelineEngine(
        lambda _: loader,
        calendar,
        asset_finder,
    )


__all__ = (
    'Classifier',
    'CustomFactor',
    'CustomFilter',
    'CustomClassifier',
    'engine_from_files',
    'Factor',
    'Filter',
    'Pipeline',
    'SimplePipelineEngine',
    'Term',
    'TermGraph',
)
