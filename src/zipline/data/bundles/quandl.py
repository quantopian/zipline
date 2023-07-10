"""
Module for building a complete daily dataset from Quandl's WIKI dataset.
"""
from io import BytesIO
import tarfile
from zipfile import ZipFile

from click import progressbar
import logging
import pandas as pd
import requests
from urllib.parse import urlencode
from zipline.utils.calendar_utils import register_calendar_alias

from . import core as bundles
import numpy as np

log = logging.getLogger(__name__)

ONE_MEGABYTE = 1024 * 1024
QUANDL_DATA_URL = "https://www.quandl.com/api/v3/datatables/WIKI/PRICES.csv?"


def format_metadata_url(api_key):
    """Build the query URL for Quandl WIKI Prices metadata."""
    query_params = [("api_key", api_key), ("qopts.export", "true")]

    return QUANDL_DATA_URL + urlencode(query_params)


def load_data_table(file, index_col, show_progress=False):
    """Load data table from zip file provided by Quandl."""
    with ZipFile(file) as zip_file:
        file_names = zip_file.namelist()
        assert len(file_names) == 1, "Expected a single file from Quandl."
        wiki_prices = file_names.pop()
        with zip_file.open(wiki_prices) as table_file:
            if show_progress:
                log.info("Parsing raw data.")
            data_table = pd.read_csv(
                table_file,
                parse_dates=["date"],
                index_col=index_col,
                usecols=[
                    "ticker",
                    "date",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "ex-dividend",
                    "split_ratio",
                ],
            )

    data_table.rename(
        columns={
            "ticker": "symbol",
            "ex-dividend": "ex_dividend",
        },
        inplace=True,
        copy=False,
    )
    return data_table


def fetch_data_table(api_key, show_progress, retries):
    """Fetch WIKI Prices data table from Quandl"""
    for _ in range(retries):
        try:
            if show_progress:
                log.info("Downloading WIKI metadata.")

            metadata = pd.read_csv(format_metadata_url(api_key))
            # Extract link from metadata and download zip file.
            table_url = metadata.loc[0, "file.link"]
            if show_progress:
                raw_file = download_with_progress(
                    table_url,
                    chunk_size=ONE_MEGABYTE,
                    label="Downloading WIKI Prices table from Quandl",
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
        log.info("Generating asset metadata.")

    data = data.groupby(by="symbol").agg({"date": [np.min, np.max]})
    data.reset_index(inplace=True)
    data["start_date"] = data.date[np.min.__name__]
    data["end_date"] = data.date[np.max.__name__]
    del data["date"]
    data.columns = data.columns.get_level_values(0)

    data["exchange"] = "QUANDL"
    data["auto_close_date"] = data["end_date"].values + pd.Timedelta(days=1)
    return data


def parse_splits(data, show_progress):
    if show_progress:
        log.info("Parsing split data.")

    data["split_ratio"] = 1.0 / data.split_ratio
    data.rename(
        columns={
            "split_ratio": "ratio",
            "date": "effective_date",
        },
        inplace=True,
        copy=False,
    )
    return data


def parse_dividends(data, show_progress):
    if show_progress:
        log.info("Parsing dividend data.")

    data["record_date"] = data["declared_date"] = data["pay_date"] = pd.NaT
    data.rename(
        columns={
            "ex_dividend": "amount",
            "date": "ex_date",
        },
        inplace=True,
        copy=False,
    )
    return data


def parse_pricing_and_vol(data, sessions, symbol_map):
    for asset_id, symbol in symbol_map.items():
        asset_data = (
            data.xs(symbol, level=1).reindex(sessions.tz_localize(None)).fillna(0.0)
        )
        yield asset_id, asset_data


@bundles.register("quandl")
def quandl_bundle(
    environ,
    asset_db_writer,
    minute_bar_writer,
    daily_bar_writer,
    adjustment_writer,
    calendar,
    start_session,
    end_session,
    cache,
    show_progress,
    output_dir,
):
    """
    quandl_bundle builds a daily dataset using Quandl's WIKI Prices dataset.

    For more information on Quandl's API and how to obtain an API key,
    please visit https://docs.quandl.com/docs#section-authentication
    """
    api_key = environ.get("QUANDL_API_KEY")
    if api_key is None:
        raise ValueError(
            "Please set your QUANDL_API_KEY environment variable and retry."
        )

    raw_data = fetch_data_table(
        api_key, show_progress, environ.get("QUANDL_DOWNLOAD_ATTEMPTS", 5)
    )
    asset_metadata = gen_asset_metadata(raw_data[["symbol", "date"]], show_progress)

    exchanges = pd.DataFrame(
        data=[["QUANDL", "QUANDL", "US"]],
        columns=["exchange", "canonical_name", "country_code"],
    )
    asset_db_writer.write(equities=asset_metadata, exchanges=exchanges)

    symbol_map = asset_metadata.symbol
    sessions = calendar.sessions_in_range(start_session, end_session)

    raw_data.set_index(["date", "symbol"], inplace=True)
    daily_bar_writer.write(
        parse_pricing_and_vol(raw_data, sessions, symbol_map),
        show_progress=show_progress,
    )

    raw_data.reset_index(inplace=True)
    raw_data["symbol"] = raw_data["symbol"].astype("category")
    raw_data["sid"] = raw_data.symbol.cat.codes
    adjustment_writer.write(
        splits=parse_splits(
            raw_data[
                [
                    "sid",
                    "date",
                    "split_ratio",
                ]
            ].loc[raw_data.split_ratio != 1],
            show_progress=show_progress,
        ),
        dividends=parse_dividends(
            raw_data[
                [
                    "sid",
                    "date",
                    "ex_dividend",
                ]
            ].loc[raw_data.ex_dividend != 0],
            show_progress=show_progress,
        ),
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

    total_size = int(resp.headers["content-length"])
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


QUANTOPIAN_QUANDL_URL = "https://s3.amazonaws.com/quantopian-public-zipline-data/quandl"


@bundles.register("quantopian-quandl", create_writers=False)
def quantopian_quandl_bundle(
    environ,
    asset_db_writer,
    minute_bar_writer,
    daily_bar_writer,
    adjustment_writer,
    calendar,
    start_session,
    end_session,
    cache,
    show_progress,
    output_dir,
):
    if show_progress:
        data = download_with_progress(
            QUANTOPIAN_QUANDL_URL,
            chunk_size=ONE_MEGABYTE,
            label="Downloading Bundle: quantopian-quandl",
        )
    else:
        data = download_without_progress(QUANTOPIAN_QUANDL_URL)

    with tarfile.open("r", fileobj=data) as tar:
        if show_progress:
            log.info("Writing data to %s.", output_dir)
        tar.extractall(output_dir)


register_calendar_alias("QUANDL", "NYSE")
