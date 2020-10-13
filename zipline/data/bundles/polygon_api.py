import bs4 as bs
import alpaca_trade_api as tradeapi
import csv
from datetime import datetime as dt
from datetime import timedelta
import numpy as np
from os import listdir, mkdir, remove
from os.path import exists, isfile, join
from pathlib import Path
import pandas as pd
import pickle
import requests
from trading_calendars import register_calendar
# from trading_calendars.exchange_calendar_binance import BinanceExchangeCalendar
import yaml
from zipline.data.bundles import core as bundles
from dateutil.parser import parse as date_parse

user_home = str(Path.home())
custom_data_path = join(user_home, '.zipline/custom_data')

CLIENT: tradeapi.REST = None


def initialize_client():
    global CLIENT
    with open("alpaca.yaml", mode='r') as f:
        o = yaml.safe_load(f)
        key = o["key_id"]
        secret = o["secret"]
    CLIENT = tradeapi.REST(key_id=key,
                           secret_key=secret)

ASSETS = None
def list_assets():
    global ASSETS
    if not ASSETS:
        # ASSETS = CLIENT.list_assets()
        ASSETS = [_.ticker for _ in CLIENT.polygon.all_tickers()]

    return ASSETS[:20]


def tickers():
    """
    Save Binance trading pair tickers to a pickle file
    Return a list of trading ticker pairs
    """
    cmc_binance_url = 'https://coinmarketcap.com/exchanges/binance/'
    response = requests.get(cmc_binance_url)
    if response.ok:
        soup = bs.BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table', {'id': 'exchange-markets'})
        ticker_pairs = []

        for row in table.findAll('tr')[1:]:
            ticker_pair = row.findAll('td')[2].text
            ticker_pairs.append(ticker_pair.strip().replace('/', ''))

    if not exists(custom_data_path):
        mkdir(custom_data_path)

    with open(join(custom_data_path, 'binance_ticker_pairs.pickle'), 'wb') as f:
        pickle.dump(ticker_pairs, f)

    return ticker_pairs


def tickers_generator():
    """
    Return a tuple (sid, ticker_pair)
    """
    tickers_file = join(custom_data_path, 'binance_ticker_pairs.pickle')
    if not isfile(tickers_file):
        ticker_pairs = list_assets()

    else:
        with open(tickers_file, 'rb') as f:
            ticker_pairs = pickle.load(f)[:]

    return (tuple((sid, ticker)) for sid, ticker in enumerate(ticker_pairs))


def iso_date(date_str):
    """
    this method will make sure that dates are formatted properly
    as with isoformat
    :param date_str:
    :return: YYYY-MM-DD date formatted
    """
    return date_parse(date_str).date().isoformat()


def get_aggs_from_polygon(dataname,
                          dtbegin,
                          dtend,
                          granularity,
                          compression):
    """
    so polygon has a much more convenient api for this than alpaca because
    we could insert the compression in to the api call and we don't need to
    resample it. but, at this point in time, something is not working
    properly and data is returned in segments. meaning, we have patches of
    missing data. e.g we request data from 2020-03-01 to 2020-07-01 and we
    get something like this: 2020-03-01:2020-03-15, 2020-06-25:2020-07-01
    so that makes life difficult.. there's no way to know which patch will
    be returned and which one we should try to get again.
    so the solution must be, ask data in segments. I select an arbitrary
    time window of 2 weeks, and split the calls until we get all required
    data
    """
    def _clear_out_of_market_hours(df):
        """
        only interested in samples between 9:30, 16:00 NY time
        """
        return df.between_time("09:30", "16:00")

    if granularity == 'day':
        cdl = CLIENT.polygon.historic_agg_v2(
            dataname,
            compression,
            granularity,
            _from=iso_date(dtbegin.isoformat()),
            to=iso_date(dtend.isoformat())).df
    else:
        cdl = pd.DataFrame()
        segment_start = dtbegin
        segment_end = segment_start + timedelta(weeks=2) if \
            dtend - dtbegin >= timedelta(weeks=2) else dtend
        while segment_end <= dtend and dtend not in cdl.index:
            response = CLIENT.polygon.historic_agg_v2(
                dataname,
                compression,
                granularity,
                _from=iso_date(segment_start.isoformat()),
                to=iso_date(segment_end.isoformat()))
            # No result from the server, most likely error
            if response.df.shape[0] == 0 and cdl.shape[0] == 0:
                raise Exception("received empty response")
            temp = response.df
            cdl = pd.concat([cdl, temp])
            cdl = cdl[~cdl.index.duplicated()]
            segment_start = segment_end
            segment_end = segment_start + timedelta(weeks=2) if \
                dtend - dtbegin >= timedelta(weeks=2) else dtend
        cdl = _clear_out_of_market_hours(cdl)
    return cdl


def df_generator(interval):
    start = '2020-7-14'  # Binance launch date
    end = dt.utcnow().strftime('%Y-%m-%d')  # Current day

    for item in list_assets():
        try:
            sid = item[0]
            ticker_pair = item[1]
            df = pd.DataFrame(
                columns=['date', 'open', 'high', 'low', 'close', 'volume'])

            symbol = ticker_pair
            print(symbol, interval)
            asset_name = ticker_pair
            exchange = 'Binance'

            klines = CLIENT.get_historical_klines_generator(
                ticker_pair, interval, start, end)

            for kline in klines:
                line = kline[:]
                del line[6:]
                # Make a real copy of kline
                # Binance API forbids the change of open time
                line[0] = np.datetime64(line[0], 'ms')
                line[0] = pd.Timestamp(line[0], 'ms')
                df.loc[len(df)] = line

            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df = df.astype({'open': 'float64', 'high': 'float64',
                            'low': 'float64', 'close': 'float64', 'volume': 'float64'})

            start_date = df.index[0]
            end_date = df.index[-1]
            first_traded = start_date
            auto_close_date = end_date + pd.Timedelta(days=1)

            # Check if there is any missing session; skip the ticker pair otherwise
            if interval == '1d' and len(df.index) - 1 != pd.Timedelta(end_date - start_date).days:
                # print('Missing sessions found in {}. Skip importing'.format(ticker_pair))
                continue
            elif interval == '1m' and timedelta(minutes=(len(df.index) + 60)) != end_date - start_date:
                # print('Missing sessions found in {}. Skip importing'.format(ticker_pair))
                continue

            yield (sid, df), symbol, asset_name, start_date, end_date, first_traded, auto_close_date, exchange
        except Exception as e:
            print(f"error while processig {ticker_pair}: {e}")


def metadata_df():
    metadata_dtype = [
        ('symbol', 'object'),
        ('asset_name', 'object'),
        ('start_date', 'datetime64[ns]'),
        ('end_date', 'datetime64[ns]'),
        ('first_traded', 'datetime64[ns]'),
        ('auto_close_date', 'datetime64[ns]'),
        ('exchange', 'object'), ]
    metadata_df = pd.DataFrame(
        np.empty(len(list_assets()), dtype=metadata_dtype))

    return metadata_df


@bundles.register('binance_api', calendar_name="24/7", minutes_per_day=1440)
def api_to_bundle(interval=['1m']):
    def ingest(environ,
               asset_db_writer,
               minute_bar_writer,
               daily_bar_writer,
               adjustment_writer,
               calendar,
               start_session,
               end_session,
               cache,
               show_progress,
               output_dir
               ):


        def minute_data_generator():
            return (sid_df for (sid_df, *metadata.iloc[sid_df[0]]) in df_generator(interval='1m'))

        def daily_data_generator():
            return (sid_df for (sid_df, *metadata.iloc[sid_df[0]]) in df_generator(interval='1d'))
        for _interval in interval:
            metadata = metadata_df()
            if _interval == '1d':
                daily_bar_writer.write(
                    daily_data_generator(), show_progress=True)
            elif _interval == '1m':
                minute_bar_writer.write(
                    minute_data_generator(), show_progress=True)

            # Drop the ticker rows which have missing sessions in their data sets
            metadata.dropna(inplace=True)

            asset_db_writer.write(equities=metadata)
            print(metadata)
            adjustment_writer.write()

    return ingest


if __name__ == '__main__':
    from zipline.data.bundles import register
    from zipline.data import bundles as bundles_module
    import os

    initialize_client()

    register(
        'polygon_api',
        # api_to_bundle(interval=['1d', '1m']),
        # api_to_bundle(interval=['1m']),
        api_to_bundle(interval=['1d']),
        calendar_name='24/7',
    )
    assets_version = ((),)[0]  # just a weird way to create an empty tuple
    bundles_module.ingest(
        "polygon_api",
        os.environ,
        pd.Timestamp.utcnow(),
        assets_version,
        True,
    )
