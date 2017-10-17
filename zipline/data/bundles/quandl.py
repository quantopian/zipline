"""
Module for building a complete daily dataset from Quandl's WIKI dataset.
"""
from io import BytesIO
import tarfile
from zipfile import ZipFile

from click import progressbar
from logbook import Logger
import pandas as pd
import requests
from six.moves.urllib.parse import urlencode

from zipline.utils.calendars import register_calendar_alias

from . import core as bundles

log = Logger(__name__)
seconds_per_call = (pd.Timedelta('10 minutes') / 2000).total_seconds()
ONE_MEGABYTE = 1024 * 1024
QUANDL_METADATA_URL = (
    'https://www.quandl.com/api/v3/datatables/WIKI/PRICES.csv?'
)


def format_metadata_url(api_key):
    """Build the query URL for the Quandl WIKI metadata.
    """
    query_params = [('api_key', api_key), ('qopts.export', 'true')]

    return (
        QUANDL_METADATA_URL + urlencode(query_params)
    )


def fetch_data_table(api_key, show_progress):
    ''' Import WIKI Prices data table from Quandl
    '''
    if show_progress:
        print('Downloading WIKI metadata.')

    metadata = pd.read_csv(
        format_metadata_url(api_key)
    )
    # Extract link from metadata and download zip file.
    table_url = metadata.loc[0, 'file.link']
    if show_progress:
        raw_file = download_with_progress(
            table_url,
            chunk_size=ONE_MEGABYTE,
            label="Downloading WIKI Prices table from Quandl"
        )
    else:
        raw_file = download_without_progress(table_url)

    with ZipFile(raw_file) as zip_file:
        file_names = zip_file.namelist()
        wiki_prices = file_names.pop()
        table_file = zip_file.open(wiki_prices)
        if show_progress:
            print('Parsing raw data.')
        data_table = pd.read_csv(
            table_file,
            parse_dates=['date'],
            usecols=[
                'ticker',
                'date',
                'open',
                'high',
                'low',
                'close',
                'volume',
                'ex-dividend',
                'split_ratio'
            ],
            na_values=['NA']
        ).rename(
            columns={
                'ticker': 'symbol',
                'ex-dividend': 'ex_dividend'
            }
        )
        table_file.close()

    return data_table


def gen_asset_metadata(data, show_progress):
    if show_progress:
        print('Generating asset metadata.')

    symbols = data['symbol'].unique()
    asset_metadata = {}

    asset_metadata['symbol'] = symbols
    data.set_index('symbol', inplace=True)
    asset_metadata['start_date'] = \
        [data.loc[asset, 'date'].min() for asset in symbols]
    asset_metadata['end_date'] = \
        [data.loc[asset, 'date'].max() for asset in symbols]

    asset_metadata = pd.DataFrame.from_dict(asset_metadata)
    asset_metadata['exchange'] = 'QUANDL'
    asset_metadata['auto_close_date'] = \
        asset_metadata['end_date'].values + pd.Timedelta(days=1)
    return asset_metadata


def parse_asset_splits(data, show_progress):
    if show_progress:
        print('Parsing split data.')

    split_ratios = data.split_ratio
    return pd.DataFrame({
        'ratio': 1.0 / split_ratios[split_ratios != 1],
        'effective_date': data.date,
        'sid': pd.factorize(data.symbol)[0]
    })


def parse_asset_dividends(data, show_progress):
    if show_progress:
        print('Parsing dividend data.')

    divs = data.ex_dividend
    return pd.DataFrame({
        'amount': divs[divs != 0],
        'ex_date': data.date,
        'sid': pd.factorize(data.symbol)[0],
        'record_date': pd.NaT,
        'declared_date': pd.NaT,
        'pay_date': pd.NaT
    })


def parse_asset_data(data,
                     calendar,
                     start_session,
                     end_session,
                     symbol_map):
    sessions = calendar.sessions_in_range(start_session, end_session)
    for asset_id, symbol in symbol_map.iteritems():
        asset_data = data[data['symbol'] == symbol].drop(
            'symbol', axis=1
        ).reindex(
            sessions.tz_localize(None)
        ).fillna(0.0)
        yield asset_id, asset_data


@bundles.register('quandl')
def quandl_bundle(environ,
                  asset_db_writer,
                  minute_bar_writer,
                  daily_bar_writer,
                  adjustment_writer,
                  calendar,
                  start_session,
                  end_session,
                  cache,
                  show_progress,
                  output_dir):
    """Build a zipline data bundle from the Quandl WIKI dataset.
    """
    api_key = environ.get('QUANDL_API_KEY')
    raw_data = fetch_data_table(
        api_key,
        show_progress
    )
    asset_metadata = gen_asset_metadata(
        raw_data[['symbol', 'date']],
        show_progress
    )
    asset_db_writer.write(asset_metadata)

    symbol_map = asset_metadata.symbol
    raw_data.set_index('date', inplace=True)
    daily_bar_writer.write(
        parse_asset_data(
            raw_data,
            calendar,
            start_session,
            end_session,
            symbol_map
        ),
        show_progress=show_progress
    )

    raw_data.reset_index(inplace=True)

    adjustment_writer.write(
        splits=parse_asset_splits(
            raw_data[['symbol', 'date', 'split_ratio']],
            show_progress=show_progress
        ),
        dividends=parse_asset_dividends(
            raw_data[['symbol', 'date', 'ex_dividend']],
            show_progress=show_progress
        )
    )


def download_with_progress(url, chunk_size, **progress_kwargs):
    """
    Download streaming data from a URL, printing progress information to the
    terminal.

    Parameters
    ----------
    url : str
        A URL that can be understood by ``requests.get``.
    chunk_size : int
        Number of bytes to read at a time from requests.
    **progress_kwargs
        Forwarded to click.progressbar.

    Returns
    -------
    data : BytesIO
        A BytesIO containing the downloaded data.
    """
    resp = requests.get(url, stream=True)
    resp.raise_for_status()

    total_size = int(resp.headers['content-length'])
    data = BytesIO()
    with progressbar(length=total_size, **progress_kwargs) as pbar:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            data.write(chunk)
            pbar.update(len(chunk))

    data.seek(0)
    return data


def download_without_progress(url):
    """
    Download data from a URL, returning a BytesIO containing the loaded data.

    Parameters
    ----------
    url : str
        A URL that can be understood by ``requests.get``.

    Returns
    -------
    data : BytesIO
        A BytesIO containing the downloaded data.
    """
    resp = requests.get(url)
    resp.raise_for_status()
    return BytesIO(resp.content)


QUANTOPIAN_QUANDL_URL = (
    'https://s3.amazonaws.com/quantopian-public-zipline-data/quandl'
)


@bundles.register('quantopian-quandl', create_writers=False)
def quantopian_quandl_bundle(environ,
                             asset_db_writer,
                             minute_bar_writer,
                             daily_bar_writer,
                             adjustment_writer,
                             calendar,
                             start_session,
                             end_session,
                             cache,
                             show_progress,
                             output_dir):
    if show_progress:
        data = download_with_progress(
            QUANTOPIAN_QUANDL_URL,
            chunk_size=ONE_MEGABYTE,
            label="Downloading Bundle: quantopian-quandl",
        )
    else:
        data = download_without_progress(QUANTOPIAN_QUANDL_URL)

    with tarfile.open('r', fileobj=data) as tar:
        if show_progress:
            print("Writing data to %s." % output_dir)
        tar.extractall(output_dir)


register_calendar_alias("QUANDL", "NYSE")
