import pandas as pd

from zipline.data.bundles import load
from zipline.pipeline import Pipeline, SimplePipelineEngine
from zipline.pipeline.loaders import USEquityPricingLoader
from zipline.pipeline.factors import AverageDollarVolume, MACDSignal

from trading_calendars import get_calendar

# Fetch previously ingested data from the 'quandl' bundle.
data = load('quandl')

# Create a pipeline that computes a MACD Signal line for the 5 most liquid
# assets by average dollar volume.
pipeline = Pipeline({
    'macd': MACDSignal(),
}, screen=AverageDollarVolume(window_length=200).top(5))

# The USEquityPricingLoader class can provide data for the USEquityPricing
# dataset, which is needed to compute beta.
pricing_loader = USEquityPricingLoader(
    data.equity_daily_bar_reader,
    data.adjustment_reader,
)


# The SimplePipelineEngine needs to be told what loader should be used for each
# dataset referenced by a given Pipeline. We provide that information by
# passing a function that takes a column and returns the loader to use to fetch
# data for that column.
#
# In this example, the only dataset we're interested is USEquityPricing, so our
# ``get_loader`` function just always returns the loader.
def get_loader(column):
    return pricing_loader


engine = SimplePipelineEngine(
    get_loader,
    get_calendar('NYSE').all_sessions,
    data.asset_finder,
)

# NOTE: These dates need to be trading days.
start_date = pd.Timestamp('2016-01-04', tz='UTC')
end_date = pd.Timestamp('2017-01-03', tz='UTC')

# Run the pipeline.
result = engine.run_pipeline(pipeline, start_date, end_date)

# Print results for the first three days.
print(result.head(15))
