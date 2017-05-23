from __future__ import print_function
from zipline.assets import AssetFinder
from zipline.utils.calendars import get_calendar
from zipline.utils.date_utils import compute_date_range_chunks
from zipline.utils.pandas_utils import categorical_df_concat

from .classifiers import Classifier, CustomClassifier
from .engine import SimplePipelineEngine
from .factors import Factor, CustomFactor
from .filters import Filter, CustomFilter
from .term import Term
from .graph import ExecutionPlan, TermGraph
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
    asset_finder = AssetFinder(asset_db_path)
    if warmup_assets:
        results = asset_finder.retrieve_all(asset_finder.sids)
        print("Warmed up %d assets." % len(results))

    return SimplePipelineEngine(
        lambda _: loader,
        calendar,
        asset_finder,
    )


def run_chunked_pipeline(engine, pipeline, start_date, end_date, chunksize):
    """Run a pipeline to collect the results.

    Parameters
    ----------
    engine : Engine
        The pipeline engine.
    pipeline : Pipeline
        The pipeline to run.
    start_date : pd.Timestamp
        The start date to run the pipeline for.
    end_date : pd.Timestamp
        The end date to run the pipeline for.
    chunksize : int or None
        The number of days to execute at a time. If this is None, all the days
        will be run at once.

    Returns
    -------
    results : pd.DataFrame
        The results for each output term in the pipeline.
    """
    ranges = compute_date_range_chunks(
        get_calendar('NYSE'),
        start_date,
        end_date,
        chunksize,
    )
    chunks = [engine.run_pipeline(pipeline, s, e) for s, e in ranges]

    return categorical_df_concat(chunks, inplace=True)


__all__ = (
    'Classifier',
    'CustomFactor',
    'CustomFilter',
    'CustomClassifier',
    'engine_from_files',
    'ExecutionPlan',
    'Factor',
    'Filter',
    'Pipeline',
    'SimplePipelineEngine',
    'run_chunked_pipeline',
    'Term',
    'TermGraph',
)
