"""
Module for building a complete dataset from local directory with csv files.
"""
import os
import sys

import logging
import numpy as np
import pandas as pd
from zipline.utils.calendar_utils import register_calendar_alias
from zipline.utils.cli import maybe_show_progress

from . import core as bundles

handler = logging.StreamHandler()
# handler = logging.StreamHandler(sys.stdout, format_string=" | {record.message}")
logger = logging.getLogger(__name__)
logger.handlers.append(handler)


def csvdir_equities(tframes=None, csvdir=None):
    """
    Generate an ingest function for custom data bundle
    This function can be used in ~/.zipline/extension.py
    to register bundle with custom parameters, e.g. with
    a custom trading calendar.

    Parameters
    ----------
    tframes: tuple, optional
        The data time frames, supported timeframes: 'daily' and 'minute'
    csvdir : string, optional, default: CSVDIR environment variable
        The path to the directory of this structure:
        <directory>/<timeframe1>/<symbol1>.csv
        <directory>/<timeframe1>/<symbol2>.csv
        <directory>/<timeframe1>/<symbol3>.csv
        <directory>/<timeframe2>/<symbol1>.csv
        <directory>/<timeframe2>/<symbol2>.csv
        <directory>/<timeframe2>/<symbol3>.csv

    Returns
    -------
    ingest : callable
        The bundle ingest function

    Examples
    --------
    This code should be added to ~/.zipline/extension.py
    .. code-block:: python
       from zipline.data.bundles import csvdir_equities, register
       register('custom-csvdir-bundle',
                csvdir_equities(["daily", "minute"],
                '/full/path/to/the/csvdir/directory'))
    """

    return CSVDIRBundle(tframes, csvdir).ingest


class CSVDIRBundle:
    """
    Wrapper class to call csvdir_bundle with provided
    list of time frames and a path to the csvdir directory
    """

    def __init__(self, tframes=None, csvdir=None):
        self.tframes = tframes
        self.csvdir = csvdir

    def ingest(
        self,
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
        csvdir_bundle(
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
            self.tframes,
            self.csvdir,
        )


@bundles.register("csvdir")
def csvdir_bundle(
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
    tframes=None,
    csvdir=None,
):
    """
    Build a zipline data bundle from the directory with csv files.
    """
    if not csvdir:
        csvdir = environ.get("CSVDIR")
        if not csvdir:
            raise ValueError("CSVDIR environment variable is not set")

    if not os.path.isdir(csvdir):
        raise ValueError("%s is not a directory" % csvdir)

    if not tframes:
        tframes = set(["daily", "minute"]).intersection(os.listdir(csvdir))

        if not tframes:
            raise ValueError(
                "'daily' and 'minute' directories " "not found in '%s'" % csvdir
            )

    divs_splits = {
        "divs": pd.DataFrame(
            columns=[
                "sid",
                "amount",
                "ex_date",
                "record_date",
                "declared_date",
                "pay_date",
            ]
        ),
        "splits": pd.DataFrame(columns=["sid", "ratio", "effective_date"]),
    }
    for tframe in tframes:
        ddir = os.path.join(csvdir, tframe)

        symbols = sorted(
            item.split(".csv")[0] for item in os.listdir(ddir) if ".csv" in item
        )
        if not symbols:
            raise ValueError("no <symbol>.csv* files found in %s" % ddir)

        dtype = [
            ("start_date", "datetime64[ns]"),
            ("end_date", "datetime64[ns]"),
            ("auto_close_date", "datetime64[ns]"),
            ("symbol", "object"),
        ]
        metadata = pd.DataFrame(np.empty(len(symbols), dtype=dtype))

        if tframe == "minute":
            writer = minute_bar_writer
        else:
            writer = daily_bar_writer

        writer.write(
            _pricing_iter(ddir, symbols, metadata, divs_splits, show_progress),
            show_progress=show_progress,
        )

        # Hardcode the exchange to "CSVDIR" for all assets and (elsewhere)
        # register "CSVDIR" to resolve to the NYSE calendar, because these
        # are all equities and thus can use the NYSE calendar.
        metadata["exchange"] = "CSVDIR"

        asset_db_writer.write(equities=metadata)

        divs_splits["divs"]["sid"] = divs_splits["divs"]["sid"].astype(int)
        divs_splits["splits"]["sid"] = divs_splits["splits"]["sid"].astype(int)
        adjustment_writer.write(
            splits=divs_splits["splits"], dividends=divs_splits["divs"]
        )


def _pricing_iter(csvdir, symbols, metadata, divs_splits, show_progress):
    with maybe_show_progress(
        symbols, show_progress, label="Loading custom pricing data: "
    ) as it:
        # using scandir instead of listdir can be faster
        files = os.scandir(csvdir)
        # building a dictionary of filenames
        # NOTE: if there are duplicates it will arbitrarily pick the latest found
        fnames = {f.name.split(".")[0]: f.name for f in files if f.is_file()}

        for sid, symbol in enumerate(it):
            logger.debug(f"{symbol}: sid {sid}")
            fname = fnames.get(symbol, None)

            if fname is None:
                raise ValueError(f"{symbol}.csv file is not in {csvdir}")

            # NOTE: read_csv can also read compressed csv files
            dfr = pd.read_csv(
                os.path.join(csvdir, fname),
                parse_dates=[0],
                index_col=0,
            ).sort_index()

            start_date = dfr.index[0]
            end_date = dfr.index[-1]

            # The auto_close date is the day after the last trade.
            ac_date = end_date + pd.Timedelta(days=1)
            metadata.iloc[sid] = start_date, end_date, ac_date, symbol

            if "split" in dfr.columns:
                tmp = 1.0 / dfr[dfr["split"] != 1.0]["split"]
                split = pd.DataFrame(
                    data=tmp.index.tolist(), columns=["effective_date"]
                )
                split["ratio"] = tmp.tolist()
                split["sid"] = sid

                splits = divs_splits["splits"]
                index = pd.Index(
                    range(splits.shape[0], splits.shape[0] + split.shape[0])
                )
                split.set_index(index, inplace=True)
                divs_splits["splits"] = pd.concat([splits, split], axis=0)

            if "dividend" in dfr.columns:
                # ex_date   amount  sid record_date declared_date pay_date
                tmp = dfr[dfr["dividend"] != 0.0]["dividend"]
                div = pd.DataFrame(data=tmp.index.tolist(), columns=["ex_date"])
                div["record_date"] = pd.NaT
                div["declared_date"] = pd.NaT
                div["pay_date"] = pd.NaT
                div["amount"] = tmp.tolist()
                div["sid"] = sid

                divs = divs_splits["divs"]
                ind = pd.Index(range(divs.shape[0], divs.shape[0] + div.shape[0]))
                div.set_index(ind, inplace=True)
                divs_splits["divs"] = pd.concat([divs, div], axis=0)

            yield sid, dfr


register_calendar_alias("CSVDIR", "NYSE")
