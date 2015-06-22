import shelve
import time

import pandas as pd

import numpy as np

import bcolz

from zipline.data.equities import USEquityPricing
from zipline.data.ffc.loaders.us_equity_pricing import \
    USEquityPricingLoader


def check_result(table, prices, asset_ix, asset):
    query = '(sid == {0}) & (day >= {1}) & (day <= {2})'.format(
        asset,
        pd.Timestamp(start_date, tz='UTC').strftime('%s'),
        pd.Timestamp(end_date, tz='UTC').strftime('%s'))
    raw_result = table[query]['close'][:]
    where_nan = raw_result == 0
    result = raw_result.astype(np.float32) * 0.001
    result = result[~where_nan]
    from_loader = prices[:, asset_ix]
    from_loader = from_loader[~np.isnan(from_loader)]
    try:
        np.testing.assert_allclose(result, from_loader,
                                   rtol=0.01,
                                   atol=0.01)
    except Exception as err:
        raise Exception("asset_ix={0} asset={1} err=\n{2}".format(
            asset_ix, asset, err))


if __name__ == "__main__":
    """
    This shim runs load_adjusted_array over a bcolz dataset generated
    internally.

    Used to test both speed of load_adjusted_array, and data correctness.

    Unit tests of the 6 different cases outlined in _load_adjusted_array
    may replace this
    """
    import zipline.finance.trading
    env = zipline.finance.trading.TradingEnvironment.instance()

    min_date = pd.Timestamp('2002-01-02', tz='UTC')
    td = env.trading_days

    start_date = '2003-01-01'
    end_date = '2003-12-31'
    mask = (td >= start_date) & (td <= end_date)

    dates = env.trading_days[mask]

    # generated from start_pos.keys()[0:8000]
    d = shelve.open('./daily_equity_index.shelf')
    assets = np.array(sorted(d['start_pos'].keys())[0:8000], dtype=np.uint32)

    loader = USEquityPricingLoader("./equity_daily_bars.bcolz",
                                   "./daily_equity_index.shelf",
                                   trading_days=td[td >= min_date])

    before = time.time()
    result = loader.load_adjusted_array(
        [USEquityPricing.close, USEquityPricing.volume],
        dates,
        assets,
    )
    after = time.time()
    duration = after - before
    print "time in load_adjusted_array={0}".format(duration)

    prices = np.array(result[0].data)
    volumes = np.array(result[1].data)

    print prices
    print volumes

    table = bcolz.open("./equity_daily_bars.bcolz",)
    for i, asset in enumerate(assets):
        print "checking index={0} and asset={1}".format(i, asset)
        check_result(table, prices, i, asset)
