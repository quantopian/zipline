"""
Module for building a complete daily dataset from Quandl's WIKI dataset.
"""
from io import BytesIO
import tarfile
from zipfile import ZipFile
import warnings

from click import progressbar
from logbook import Logger
import pandas as pd
import requests
from six.moves.urllib.parse import urlencode

from zipline.utils.calendars import register_calendar_alias
from zipline.utils.deprecate import deprecated
from . import core as bundles
import numpy as np

log = Logger(__name__)
warnings.simplefilter('once', DeprecationWarning)

ONE_MEGABYTE = 1024 * 1024
QUANDL_DATA_URL = (
    'https://www.quandl.com/api/v3/datatables/WIKI/PRICES.csv?'
)


def format_metadata_url(api_key):
    """ Build the query URL for the Quandl WIKI metadata.
    """
    query_params = [('api_key', api_key), ('qopts.export', 'true')]

    return (
        QUANDL_DATA_URL + urlencode(query_params)
    )


def load_data_table(file,
                    index_col,
                    show_progress=False):
    """ Load data table from zip file provided by Quandl.
    """
    with ZipFile(file) as zip_file:
        file_names = zip_file.namelist()
        assert len(file_names) == 1, "Expected a single file from Quandl."
        wiki_prices = file_names.pop()
        table_file = zip_file.open(wiki_prices)
        if show_progress:
            log.info('Parsing raw data.')
        data_table = pd.read_csv(
            table_file,
            parse_dates=['date'],
            index_col=index_col,
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

    data_table['symbol'] = data_table['symbol'].astype('category')
    return data_table


def fetch_data_table(api_key,
                     show_progress,
                     retries):
    """ Fetch WIKI Prices data table from Quandl
    """
    for _ in range(retries):
        try:
            if show_progress:
                log.info('Downloading WIKI metadata.')

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

            return load_data_table(
                file=raw_file,
                index_col=None,
                show_progress=show_progress,
            )

        except Exception:
            log.exception("Exception raised reading Quandl data. Retrying.")

    else:
        raise ValueError(
            "Failed to download Quandl data after %d attempts." % (retries)
        )


def gen_asset_metadata(data, show_progress):
    if show_progress:
        log.info('Generating asset metadata.')

    asset_metadata = data.groupby(
        by='symbol'
    ).agg(
        {'date': [np.min, np.max]}
    ).reset_index()
    asset_metadata['start_date'] = asset_metadata.date.amin
    asset_metadata['end_date'] = asset_metadata.date.amax
    del asset_metadata['date']
    asset_metadata.columns = asset_metadata.columns.get_level_values(0)

    asset_metadata['exchange'] = 'QUANDL'
    asset_metadata['auto_close_date'] = \
        asset_metadata['end_date'].values + pd.Timedelta(days=1)
    return asset_metadata


def parse_splits(data, show_progress):
    if show_progress:
        log.info('Parsing split data.')

    split_ratios = data.split_ratio
    return pd.DataFrame({
        'ratio': 1.0 / split_ratios[split_ratios != 1],
        'effective_date': data.date,
        'sid': pd.factorize(data.symbol)[0]
    }).dropna()


def parse_dividends(data, show_progress):
    if show_progress:
        log.info('Parsing dividend data.')

    divs = data.ex_dividend
    df = pd.DataFrame({
        'amount': divs[divs != 0],
        'ex_date': data.date,
        'sid': pd.factorize(data.symbol)[0]
    }).dropna()
    df['record_date'] = df['declared_date'] = df['pay_date'] = pd.NaT
    return df


def parse_pricing_and_vol(data,
                          sessions,
                          symbol_map):
    for asset_id, symbol in symbol_map.iteritems():
        asset_data = data.xs(
            symbol,
            level=1
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
    """
    quandl_bundl builds a data bundle using Quandl's WIKI Prices dataset.

    For more information on Quandl's API and how to obtain an API key,
    please visit https://docs.quandl.com/docs#section-authentication
    """
    raw_data = fetch_data_table(
        environ.get('QUANDL_API_KEY'),
        show_progress,
        environ.get('QUANDL_DOWNLOAD_ATTEMPTS', 5)
    )
    asset_metadata = gen_asset_metadata(
        raw_data[['symbol', 'date']],
        show_progress
    )
    asset_db_writer.write(asset_metadata)

    symbol_map = asset_metadata.symbol
    sessions = calendar.sessions_in_range(start_session, end_session)

    raw_data.set_index(['date', 'symbol'], inplace=True)
    daily_bar_writer.write(
        parse_pricing_and_vol(
            raw_data,
            sessions,
            symbol_map
        ),
        show_progress=show_progress
    )

    raw_data.reset_index(inplace=True)

    adjustment_writer.write(
        splits=parse_splits(
            raw_data[['symbol', 'date', 'split_ratio']],
            show_progress=show_progress
        ),
        dividends=parse_dividends(
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
@deprecated(
    'quantopian-quandl has been deprecated and '
    'will be removed in a future release.'
)
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
            log.info("Writing data to %s." % output_dir)
        tar.extractall(output_dir)


register_calendar_alias("QUANDL", "NYSE")
