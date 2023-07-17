import numpy as np
import pandas as pd
import toolz.curried.operator as op
from os.path import (
    dirname,
    join,
    realpath,
)

from zipline.utils.calendar_utils import get_calendar
from zipline.data.bundles import ingest, load, bundles
from zipline.data.bundles.quandl import format_metadata_url, load_data_table
from zipline.lib.adjustment import Float64Multiply
from zipline.testing import (
    tmp_dir,
    patch_read_csv,
)
from zipline.testing.fixtures import (
    ZiplineTestCase,
    WithResponses,
)

from zipline.utils.functional import apply
from zipline.testing.github_actions import skip_on

TEST_RESOURCE_PATH = join(
    dirname(dirname(dirname(realpath(__file__)))),
    "resources",  # zipline_repo/tests
)


class QuandlBundleTestCase(WithResponses, ZiplineTestCase):
    symbols = "AAPL", "BRK_A", "MSFT", "ZEN"
    start_date = pd.Timestamp("2014-01")
    end_date = pd.Timestamp("2015-01")
    bundle = bundles["quandl"]
    calendar = get_calendar(bundle.calendar_name)
    api_key = "IamNotaQuandlAPIkey"
    columns = "open", "high", "low", "close", "volume"

    def _expected_data(self, asset_finder):
        sids = {
            symbol: asset_finder.lookup_symbol(
                symbol,
                None,
            ).sid
            for symbol in self.symbols
        }

        # Load raw data from quandl test resources.
        data = load_data_table(
            file=join(TEST_RESOURCE_PATH, "quandl_samples", "QUANDL_ARCHIVE.zip"),
            index_col="date",
        )
        data["sid"] = pd.factorize(data.symbol)[0]

        all_ = data.set_index(
            "sid",
            append=True,
        ).unstack()

        # fancy list comprehension with statements
        @list
        @apply
        def pricing():
            for column in self.columns:
                vs = all_[column].values
                if column == "volume":
                    vs = np.nan_to_num(vs)
                yield vs

        # the first index our written data will appear in the files on disk
        start_idx = (
            self.calendar.sessions.get_indexer([self.start_date], "ffill")[0] + 1
        )

        # convert an index into the raw dataframe into an index into the
        # final data
        i = op.add(start_idx)

        def expected_dividend_adjustment(idx, symbol):
            sid = sids[symbol]
            return (
                1
                - all_.iloc[idx]["ex_dividend", sid] / all_.iloc[idx - 1]["close", sid]
            )

        adjustments = [
            {
                i(24): [
                    Float64Multiply(
                        first_row=0,
                        last_row=i(24),
                        first_col=sids["AAPL"],
                        last_col=sids["AAPL"],
                        value=expected_dividend_adjustment(24, "AAPL"),
                    )
                ],
                i(87): [
                    Float64Multiply(
                        first_row=0,
                        last_row=i(87),
                        first_col=sids["AAPL"],
                        last_col=sids["AAPL"],
                        value=expected_dividend_adjustment(87, "AAPL"),
                    )
                ],
                i(150): [
                    Float64Multiply(
                        first_row=0,
                        last_row=i(150),
                        first_col=sids["AAPL"],
                        last_col=sids["AAPL"],
                        value=expected_dividend_adjustment(150, "AAPL"),
                    )
                ],
                i(214): [
                    Float64Multiply(
                        first_row=0,
                        last_row=i(214),
                        first_col=sids["AAPL"],
                        last_col=sids["AAPL"],
                        value=expected_dividend_adjustment(214, "AAPL"),
                    )
                ],
                i(31): [
                    Float64Multiply(
                        first_row=0,
                        last_row=i(31),
                        first_col=sids["MSFT"],
                        last_col=sids["MSFT"],
                        value=expected_dividend_adjustment(31, "MSFT"),
                    )
                ],
                i(90): [
                    Float64Multiply(
                        first_row=0,
                        last_row=i(90),
                        first_col=sids["MSFT"],
                        last_col=sids["MSFT"],
                        value=expected_dividend_adjustment(90, "MSFT"),
                    )
                ],
                i(158): [
                    Float64Multiply(
                        first_row=0,
                        last_row=i(158),
                        first_col=sids["MSFT"],
                        last_col=sids["MSFT"],
                        value=expected_dividend_adjustment(158, "MSFT"),
                    )
                ],
                i(222): [
                    Float64Multiply(
                        first_row=0,
                        last_row=i(222),
                        first_col=sids["MSFT"],
                        last_col=sids["MSFT"],
                        value=expected_dividend_adjustment(222, "MSFT"),
                    )
                ],
                # splits
                i(108): [
                    Float64Multiply(
                        first_row=0,
                        last_row=i(108),
                        first_col=sids["AAPL"],
                        last_col=sids["AAPL"],
                        value=1.0 / 7.0,
                    )
                ],
            },
        ] * (len(self.columns) - 1) + [
            # volume
            {
                i(108): [
                    Float64Multiply(
                        first_row=0,
                        last_row=i(108),
                        first_col=sids["AAPL"],
                        last_col=sids["AAPL"],
                        value=7.0,
                    )
                ],
            }
        ]
        return pricing, adjustments

    @skip_on(PermissionError)
    def test_bundle(self):
        with open(
            join(TEST_RESOURCE_PATH, "quandl_samples", "QUANDL_ARCHIVE.zip"),
            "rb",
        ) as quandl_response:
            self.responses.add(
                self.responses.GET,
                "https://file_url.mock.quandl",
                body=quandl_response.read(),
                content_type="application/zip",
                status=200,
            )

        url_map = {
            format_metadata_url(self.api_key): join(
                TEST_RESOURCE_PATH,
                "quandl_samples",
                "metadata.csv.gz",
            )
        }

        zipline_root = self.enter_instance_context(tmp_dir()).path
        environ = {
            "ZIPLINE_ROOT": zipline_root,
            "QUANDL_API_KEY": self.api_key,
        }

        with patch_read_csv(url_map):
            ingest("quandl", environ=environ)

        bundle = load("quandl", environ=environ)
        sids = 0, 1, 2, 3
        assert set(bundle.asset_finder.sids) == set(sids)

        sessions = self.calendar.sessions
        actual = bundle.equity_daily_bar_reader.load_raw_arrays(
            self.columns,
            sessions[sessions.get_indexer([self.start_date], "bfill")[0]],
            sessions[sessions.get_indexer([self.end_date], "ffill")[0]],
            sids,
        )
        expected_pricing, expected_adjustments = self._expected_data(
            bundle.asset_finder,
        )
        np.testing.assert_array_almost_equal(actual, expected_pricing, decimal=2)

        adjs_for_cols = bundle.adjustment_reader.load_pricing_adjustments(
            self.columns,
            sessions,
            pd.Index(sids),
        )

        for column, adjustments, expected in zip(
            self.columns, adjs_for_cols, expected_adjustments
        ):
            assert adjustments == expected, column
