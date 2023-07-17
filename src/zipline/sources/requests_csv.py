from abc import ABC, abstractmethod
from collections import namedtuple
import hashlib
from textwrap import dedent
import warnings

import logging
import numpy
import pandas as pd
import datetime

import requests
from io import StringIO
from zipline.errors import MultipleSymbolsFound, SymbolNotFound, ZiplineError
from zipline.protocol import DATASOURCE_TYPE, Event
from zipline.assets import Equity

logger = logging.getLogger("Requests Source Logger")


def roll_dts_to_midnight(dts, trading_day):
    if len(dts) == 0:
        return dts

    return (
        pd.DatetimeIndex(
            (dts.tz_convert("US/Eastern") - pd.Timedelta(hours=16)).date,
            tz="UTC",
        )
        + trading_day
    )


class FetcherEvent(Event):
    pass


class FetcherCSVRedirectError(ZiplineError):
    msg = dedent(
        """\
        Attempt to fetch_csv from a redirected url. {url}
        must be changed to {new_url}
        """
    )

    def __init__(self, *args, **kwargs):
        self.url = kwargs["url"]
        self.new_url = kwargs["new_url"]
        self.extra = kwargs["extra"]

        super(FetcherCSVRedirectError, self).__init__(*args, **kwargs)


# The following optional arguments are supported for
# requests backed data sources.
# see https://requests.readthedocs.io/en/latest/api/#main-interface
# for a full list.
ALLOWED_REQUESTS_KWARGS = {"params", "headers", "auth", "cert"}

# The following optional arguments are supported for pandas' read_csv
# function, and may be passed as kwargs to the datasource below.
# see https://pandas.pydata.org/
# pandas-docs/stable/generated/pandas.io.parsers.read_csv.html
ALLOWED_READ_CSV_KWARGS = {
    "sep",
    "dialect",
    "doublequote",
    "escapechar",
    "quotechar",
    "quoting",
    "skipinitialspace",
    "lineterminator",
    "header",
    "index_col",
    "names",
    "prefix",
    "skiprows",
    "skipfooter",
    "skip_footer",
    "na_values",
    "true_values",
    "false_values",
    "delimiter",
    "converters",
    "dtype",
    "delim_whitespace",
    "as_recarray",
    "na_filter",
    "compact_ints",
    "use_unsigned",
    "buffer_lines",
    "warn_bad_lines",
    "error_bad_lines",
    "keep_default_na",
    "thousands",
    "comment",
    "decimal",
    "keep_date_col",
    "nrows",
    "chunksize",
    "encoding",
    "usecols",
}

SHARED_REQUESTS_KWARGS = {
    "stream": True,
    "allow_redirects": False,
}


def mask_requests_args(url, validating=False, params_checker=None, **kwargs):
    requests_kwargs = {
        key: val for (key, val) in kwargs.items() if key in ALLOWED_REQUESTS_KWARGS
    }
    if params_checker is not None:
        url, s_params = params_checker(url)
        if s_params:
            if "params" in requests_kwargs:
                requests_kwargs["params"].update(s_params)
            else:
                requests_kwargs["params"] = s_params

    # Giving the connection 30 seconds. This timeout does not
    # apply to the download of the response body.
    # (Note that Quandl links can take >10 seconds to return their
    # first byte on occasion)
    requests_kwargs["timeout"] = 1.0 if validating else 30.0
    requests_kwargs.update(SHARED_REQUESTS_KWARGS)

    request_pair = namedtuple("RequestPair", ("requests_kwargs", "url"))
    return request_pair(requests_kwargs, url)


class PandasCSV(ABC):
    def __init__(
        self,
        pre_func,
        post_func,
        asset_finder,
        trading_day,
        start_date,
        end_date,
        date_column,
        date_format,
        timezone,
        symbol,
        mask,
        symbol_column,
        data_frequency,
        country_code,
        **kwargs,
    ):
        self.start_date = start_date
        self.end_date = end_date
        self.date_column = date_column
        self.date_format = date_format
        self.timezone = timezone
        self.mask = mask
        self.symbol_column = symbol_column or "symbol"
        self.data_frequency = data_frequency
        self.country_code = country_code

        invalid_kwargs = set(kwargs) - ALLOWED_READ_CSV_KWARGS
        if invalid_kwargs:
            raise TypeError(
                "Unexpected keyword arguments: %s" % invalid_kwargs,
            )

        self.pandas_kwargs = self.mask_pandas_args(kwargs)

        self.symbol = symbol

        self.finder = asset_finder
        self.trading_day = trading_day

        self.pre_func = pre_func
        self.post_func = post_func

    @property
    def fields(self):
        return self.df.columns.tolist()

    def get_hash(self):
        return self.namestring

    @abstractmethod
    def fetch_data(self):
        return

    @staticmethod
    def parse_date_str_series(
        format_str, tz, date_str_series, data_frequency, trading_day
    ):
        """
        Efficient parsing for a 1d Pandas/numpy object containing string
        representations of dates.

        Note: pd.to_datetime is significantly faster when no format string is
        passed, and in pandas 0.12.0 the %p strptime directive is not correctly
        handled if a format string is explicitly passed, but AM/PM is handled
        properly if format=None.

        Moreover, we were previously ignoring this parameter unintentionally
        because we were incorrectly passing it as a positional.  For all these
        reasons, we ignore the format_str parameter when parsing datetimes.
        """

        # Explicitly ignoring this parameter.  See note above.
        if format_str is not None:
            logger.warning(
                "The 'format_str' parameter to fetch_csv is deprecated. "
                "Ignoring and defaulting to pandas default date parsing."
            )
            format_str = None

        tz_str = str(tz)
        if tz_str == str(datetime.timezone.utc):
            parsed = pd.to_datetime(
                date_str_series.values,
                # format=format_str,
                utc=True,
                errors="coerce",
            )
        else:
            parsed = (
                pd.to_datetime(
                    date_str_series.values,
                    format=format_str,
                    errors="coerce",
                )
                .tz_localize(tz_str)
                .tz_convert("UTC")
            )

        if data_frequency == "daily":
            parsed = roll_dts_to_midnight(parsed, trading_day)
        return parsed

    def mask_pandas_args(self, kwargs):
        pandas_kwargs = {
            key: val for (key, val) in kwargs.items() if key in ALLOWED_READ_CSV_KWARGS
        }
        if "usecols" in pandas_kwargs:
            usecols = pandas_kwargs["usecols"]
            if usecols and self.date_column not in usecols:
                # make a new list so we don't modify user's,
                # and to ensure it is mutable
                with_date = list(usecols)
                with_date.append(self.date_column)
                pandas_kwargs["usecols"] = with_date

        # No strings in the 'symbol' column should be interpreted as NaNs
        pandas_kwargs.setdefault("keep_default_na", False)
        pandas_kwargs.setdefault("na_values", {"symbol": []})

        return pandas_kwargs

    def _lookup_unconflicted_symbol(self, symbol):
        """
        Attempt to find a unique asset whose symbol is the given string.

        If multiple assets have held the given symbol, return a 0.

        If no asset has held the given symbol, return a  NaN.
        """
        try:
            uppered = symbol.upper()
        except AttributeError:
            # The mapping fails because symbol was a non-string
            return numpy.nan

        try:
            return self.finder.lookup_symbol(
                uppered,
                as_of_date=None,
                country_code=self.country_code,
            )
        except MultipleSymbolsFound:
            # Fill conflicted entries with zeros to mark that they need to be
            # resolved by date.
            return 0
        except SymbolNotFound:
            # Fill not found entries with nans.
            return numpy.nan

    def load_df(self):
        df = self.fetch_data()

        if self.pre_func:
            df = self.pre_func(df)

        # Batch-convert the user-specifed date column into timestamps.
        df["dt"] = self.parse_date_str_series(
            self.date_format,
            self.timezone,
            df[self.date_column],
            self.data_frequency,
            self.trading_day,
        ).values

        # ignore rows whose dates we couldn't parse
        df = df[df["dt"].notnull()]

        if self.symbol is not None:
            df["sid"] = self.symbol
        elif self.finder:
            df.sort_values(by=self.symbol_column, inplace=True)

            # Pop the 'sid' column off of the DataFrame, just in case the user
            # has assigned it, and throw a warning
            try:
                df.pop("sid")
                warnings.warn(
                    "Assignment of the 'sid' column of a DataFrame is "
                    "not supported by Fetcher. The 'sid' column has been "
                    "overwritten.",
                    category=UserWarning,
                    stacklevel=2,
                )
            except KeyError:
                # There was no 'sid' column, so no warning is necessary
                pass

            # Fill entries for any symbols that don't require a date to
            # uniquely identify.  Entries for which multiple securities exist
            # are replaced with zeroes, while entries for which no asset
            # exists are replaced with NaNs.
            unique_symbols = df[self.symbol_column].unique()
            sid_series = pd.Series(
                data=map(self._lookup_unconflicted_symbol, unique_symbols),
                index=unique_symbols,
                name="sid",
            )
            df = df.join(sid_series, on=self.symbol_column)

            # Fill any zero entries left in our sid column by doing a lookup
            # using both symbol and the row date.
            conflict_rows = df[df["sid"] == 0]
            for row_idx, row in conflict_rows.iterrows():
                try:
                    asset = (
                        self.finder.lookup_symbol(
                            row[self.symbol_column],
                            # Replacing tzinfo here is necessary because of the
                            # timezone metadata bug described below.
                            row["dt"].replace(tzinfo=datetime.tzinfo.utc),
                            country_code=self.country_code,
                            # It's possible that no asset comes back here if our
                            # lookup date is from before any asset held the
                            # requested symbol.  Mark such cases as NaN so that
                            # they get dropped in the next step.
                        )
                        or numpy.nan
                    )
                except SymbolNotFound:
                    asset = numpy.nan

                # Assign the resolved asset to the cell
                df.iloc[row_idx, df.columns.get_loc("sid")] = asset

            # Filter out rows containing symbols that we failed to find.
            length_before_drop = len(df)
            df = df[df["sid"].notnull()]
            no_sid_count = length_before_drop - len(df)
            if no_sid_count:
                logger.warning(
                    "Dropped %s rows from fetched csv.",
                    no_sid_count,
                    extra={"syslog": True},
                )
        else:
            df["sid"] = df["symbol"]

        # Dates are localized to UTC when they come out of
        # parse_date_str_series, but we need to re-localize them here because
        # of a bug that wasn't fixed until
        # https://github.com/pydata/pandas/pull/7092.
        # We should be able to remove the call to tz_localize once we're on
        # pandas 0.14.0

        # We don't set 'dt' as the index until here because the Symbol parsing
        # operations above depend on having a unique index for the dataframe,
        # and the 'dt' column can contain multiple dates for the same entry.
        df.drop_duplicates(["sid", "dt"])
        df.set_index(["dt"], inplace=True)
        df = df.tz_localize("UTC")
        df.sort_index(inplace=True)

        cols_to_drop = [self.date_column]
        if self.symbol is None:
            cols_to_drop.append(self.symbol_column)
        df = df[df.columns.drop(cols_to_drop)]

        if self.post_func:
            df = self.post_func(df)

        return df

    def __iter__(self):
        asset_cache = {}
        for dt, series in self.df.iterrows():
            if dt < self.start_date:
                continue

            if dt > self.end_date:
                return

            event = FetcherEvent()
            # when dt column is converted to be the dataframe's index
            # the dt column is dropped. So, we need to manually copy
            # dt into the event.
            event.dt = dt
            for k, v in series.iteritems():
                # convert numpy integer types to
                # int. This assumes we are on a 64bit
                # platform that will not lose information
                # by casting.
                # TODO: this is only necessary on the
                # amazon qexec instances. would be good
                # to figure out how to use the numpy dtypes
                # without this check and casting.
                if isinstance(v, numpy.integer):
                    v = int(v)

                setattr(event, k, v)

            # If it has start_date, then it's already an Asset
            # object from asset_for_symbol, and we don't have to
            # transform it any further. Checking for start_date is
            # faster than isinstance.
            if event.sid in asset_cache:
                event.sid = asset_cache[event.sid]
            elif hasattr(event.sid, "start_date"):
                # Clone for user algo code, if we haven't already.
                asset_cache[event.sid] = event.sid
            elif self.finder and isinstance(event.sid, int):
                asset = self.finder.retrieve_asset(event.sid, default_none=True)
                if asset:
                    # Clone for user algo code.
                    event.sid = asset_cache[asset] = asset
                elif self.mask:
                    # When masking drop all non-mappable values.
                    continue
                elif self.symbol is None:
                    # If the event's sid property is an int we coerce
                    # it into an Equity.
                    event.sid = asset_cache[event.sid] = Equity(event.sid)

            event.type = DATASOURCE_TYPE.CUSTOM
            event.source_id = self.namestring
            yield event


class PandasRequestsCSV(PandasCSV):
    # maximum 100 megs to prevent DDoS
    MAX_DOCUMENT_SIZE = (1024 * 1024) * 100

    # maximum number of bytes to read in at a time
    CONTENT_CHUNK_SIZE = 4096

    def __init__(
        self,
        url,
        pre_func,
        post_func,
        asset_finder,
        trading_day,
        start_date,
        end_date,
        date_column,
        date_format,
        timezone,
        symbol,
        mask,
        symbol_column,
        data_frequency,
        country_code,
        special_params_checker=None,
        **kwargs,
    ):
        # Peel off extra requests kwargs, forwarding the remaining kwargs to
        # the superclass.
        # Also returns possible https updated url if sent to http quandl ds
        # If url hasn't changed, will just return the original.
        self._requests_kwargs, self.url = mask_requests_args(
            url, params_checker=special_params_checker, **kwargs
        )

        remaining_kwargs = {
            k: v for k, v in kwargs.items() if k not in self.requests_kwargs
        }

        self.namestring = type(self).__name__

        super(PandasRequestsCSV, self).__init__(
            pre_func,
            post_func,
            asset_finder,
            trading_day,
            start_date,
            end_date,
            date_column,
            date_format,
            timezone,
            symbol,
            mask,
            symbol_column,
            data_frequency,
            country_code=country_code,
            **remaining_kwargs,
        )

        self.fetch_size = None
        self.fetch_hash = None

        self.df = self.load_df()

        self.special_params_checker = special_params_checker

    @property
    def requests_kwargs(self):
        return self._requests_kwargs

    def fetch_url(self, url):
        info = "checking {url} with {params}"
        logger.info(info.format(url=url, params=self.requests_kwargs))
        # setting decode_unicode=True sometimes results in a
        # UnicodeEncodeError exception, so instead we'll use
        # pandas logic for decoding content
        try:
            response = requests.get(url, **self.requests_kwargs)
        except requests.exceptions.ConnectionError as exc:
            raise Exception("Could not connect to %s" % url) from exc

        if not response.ok:
            raise Exception("Problem reaching %s" % url)
        elif response.is_redirect:
            # On the offchance we don't catch a redirect URL
            # in validation, this will catch it.
            new_url = response.headers["location"]
            raise FetcherCSVRedirectError(
                url=url,
                new_url=new_url,
                extra={"old_url": url, "new_url": new_url},
            )

        content_length = 0
        logger.info(
            "{} connection established in {:.1f} seconds".format(
                url, response.elapsed.total_seconds()
            )
        )

        # use the decode_unicode flag to ensure that the output of this is
        # a string, and not bytes.
        for chunk in response.iter_content(
            self.CONTENT_CHUNK_SIZE, decode_unicode=True
        ):
            if content_length > self.MAX_DOCUMENT_SIZE:
                raise Exception("Document size too big.")
            if chunk:
                content_length += len(chunk)
                yield chunk

        return

    def fetch_data(self):
        # create a data frame directly from the full text of
        # the response from the returned file-descriptor.
        data = self.fetch_url(self.url)
        fd = StringIO()

        if isinstance(data, str):
            fd.write(data)
        else:
            for chunk in data:
                fd.write(chunk)

        self.fetch_size = fd.tell()

        fd.seek(0)

        try:
            # see if pandas can parse csv data
            frames = pd.read_csv(fd, **self.pandas_kwargs)

            frames_hash = hashlib.md5(str(fd.getvalue()).encode("utf-8"))
            self.fetch_hash = frames_hash.hexdigest()
        except pd.parser.CParserError as exc:
            # could not parse the data, raise exception
            raise Exception("Error parsing remote CSV data.") from exc
        finally:
            fd.close()

        return frames
