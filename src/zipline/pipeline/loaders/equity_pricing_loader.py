# Copyright 2015 Quantopian, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from collections import defaultdict

from interface import implements
from numpy import iinfo, uint32, multiply

from zipline.data.fx import ExplodingFXRateReader
from zipline.lib.adjusted_array import AdjustedArray
from zipline.utils.numpy_utils import repeat_first_axis

from .base import PipelineLoader
from .utils import shift_dates
from ..data.equity_pricing import EquityPricing

UINT32_MAX = iinfo(uint32).max


class EquityPricingLoader(implements(PipelineLoader)):
    """A PipelineLoader for loading daily OHLCV data.

    Parameters
    ----------
    raw_price_reader : zipline.data.session_bars.SessionBarReader
        Reader providing raw prices.
    adjustments_reader : zipline.data.adjustments.SQLiteAdjustmentReader
        Reader providing price/volume adjustments.
    fx_reader : zipline.data.fx.FXRateReader
       Reader providing currency conversions.
    """

    def __init__(self, raw_price_reader, adjustments_reader, fx_reader):
        self.raw_price_reader = raw_price_reader
        self.adjustments_reader = adjustments_reader
        self.fx_reader = fx_reader

    @classmethod
    def without_fx(cls, raw_price_reader, adjustments_reader):
        """
        Construct an EquityPricingLoader without support for fx rates.

        The returned loader will raise an error if requested to load
        currency-converted columns.

        Parameters
        ----------
        raw_price_reader : zipline.data.session_bars.SessionBarReader
            Reader providing raw prices.
        adjustments_reader : zipline.data.adjustments.SQLiteAdjustmentReader
            Reader providing price/volume adjustments.

        Returns
        -------
        loader : EquityPricingLoader
            A loader that can only provide currency-naive data.
        """
        return cls(
            raw_price_reader=raw_price_reader,
            adjustments_reader=adjustments_reader,
            fx_reader=ExplodingFXRateReader(),
        )

    def load_adjusted_array(self, domain, columns, dates, sids, mask):
        # load_adjusted_array is called with dates on which the user's algo
        # will be shown data, which means we need to return the data that would
        # be known at the **start** of each date. We assume that the latest
        # data known on day N is the data from day (N - 1), so we shift all
        # query dates back by a trading session.
        sessions = domain.sessions()
        shifted_dates = shift_dates(sessions, dates[0], dates[-1], shift=1)

        ohlcv_cols, currency_cols = self._split_column_types(columns)
        del columns  # From here on we should use ohlcv_cols or currency_cols.
        ohlcv_colnames = [c.name for c in ohlcv_cols]

        raw_ohlcv_arrays = self.raw_price_reader.load_raw_arrays(
            ohlcv_colnames,
            shifted_dates[0],
            shifted_dates[-1],
            sids,
        )

        # Currency convert raw_arrays in place if necessary. We use shifted
        # dates to load currency conversion rates to make them line up with
        # dates used to fetch prices.
        self._inplace_currency_convert(
            ohlcv_cols,
            raw_ohlcv_arrays,
            shifted_dates,
            sids,
        )

        adjustments = self.adjustments_reader.load_pricing_adjustments(
            ohlcv_colnames,
            dates,
            sids,
        )

        out = {}
        for c, c_raw, c_adjs in zip(ohlcv_cols, raw_ohlcv_arrays, adjustments):
            out[c] = AdjustedArray(
                c_raw.astype(c.dtype),
                c_adjs,
                c.missing_value,
            )

        for c in currency_cols:
            codes_1d = self.raw_price_reader.currency_codes(sids)
            codes = repeat_first_axis(codes_1d, len(dates))
            out[c] = AdjustedArray(
                codes,
                adjustments={},
                missing_value=None,
            )

        return out

    @property
    def currency_aware(self):
        # Tell the pipeline engine that this loader supports currency
        # conversion if we have a non-dummy fx rates reader.
        return not isinstance(self.fx_reader, ExplodingFXRateReader)

    def _inplace_currency_convert(self, columns, arrays, dates, sids):
        """
        Currency convert raw data loaded for ``column``.

        Parameters
        ----------
        columns : list[zipline.pipeline.data.BoundColumn]
            List of columns whose raw data has been loaded.
        arrays : list[np.array]
            List of arrays, parallel to ``columns`` containing data for the
            column.
        dates : pd.DatetimeIndex
            Labels for rows of ``arrays``. These are the dates that should
            be used to fetch fx rates for conversion.
        sids : np.array[int64]
            Labels for columns of ``arrays``.

        Returns
        -------
        None

        Side Effects
        ------------
        Modifies ``arrays`` in place by applying currency conversions.
        """
        # Group columns by currency conversion spec.
        by_spec = defaultdict(list)
        for column, array in zip(columns, arrays):
            by_spec[column.currency_conversion].append(array)

        # Nothing to do for terms with no currency conversion.
        by_spec.pop(None, None)
        if not by_spec:
            return

        fx_reader = self.fx_reader
        base_currencies = self.raw_price_reader.currency_codes(sids)

        # Columns with the same conversion spec will use the same multipliers.
        for spec, arrays in by_spec.items():
            rates = fx_reader.get_rates(
                rate=spec.field,
                quote=spec.currency.code,
                bases=base_currencies,
                dts=dates,
            )
            for arr in arrays:
                multiply(arr, rates, out=arr)

    def _split_column_types(self, columns):
        """Split out currency columns from OHLCV columns.

        Parameters
        ----------
        columns : list[zipline.pipeline.data.BoundColumn]
            Columns to be loaded by ``load_adjusted_array``.

        Returns
        -------
        ohlcv_columns : list[zipline.pipeline.data.BoundColumn]
            Price and volume columns from ``columns``.
        currency_columns : list[zipline.pipeline.data.BoundColumn]
            Currency code column from ``columns``, if present.
        """
        currency_name = EquityPricing.currency.name

        ohlcv = []
        currency = []
        for c in columns:
            if c.name == currency_name:
                currency.append(c)
            else:
                ohlcv.append(c)

        return ohlcv, currency


# Backwards compat alias.
USEquityPricingLoader = EquityPricingLoader
