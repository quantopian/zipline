"""
HDF5-backed FX Rates

The classes defined in this file produce and consume foreign exchange rates
stored in HDF5 files.

An FX Rate file contains one or more "rates", each of which contains a
collection of arrays exchange rates describing how to convert from "base"
currencies to "quote" currencies.

The specific set of rates contained in a particular file is arbitrary and
user-defined. We infer no particular semantics from their names, other than
that they are distinct rates. Pipeline API users can select the rate that
should be used for a currency conversion via ``BoundColumn.fx`` method.

Examples of possible rate names might be things like "bid", "mid", and "ask",
or "london_close", "tokyo_close", "nyse_close".

HDF5 FX Rate File Schema
------------------------

An FX rate file has two kinds of nodes: **data nodes** and **index nodes**.

Data nodes are 2-dimensional arrays of numerical exchange rate values. Data
nodes are stored at paths of the form /data/{rate}/{quote_currency}/rates. The
path to a data node encodes two pieces of information: the "rate" (e.g. 'mid',
'bid', 'ask') of the node, and the quote currency of the node. All data nodes
in a file have the same shape.

Index nodes are 1-dimensional arrays containing row- and column-labels for the
file's data nodes. There are two index nodes in a file, at /index/dts and
/index/currencies.

**Example:**

A file containing fields ["rate1", "rate2"] and currencies ["USD", "CAD",
"EUR"] has the following schema:

.. code-block:: none

   /data
     /field1
       /USD
         /rates
       /CAD
         /rates
       /EUR
         /rates
     /field2
       /USD
         /rates
       /CAD
         /rates
       /EUR
         /rates
   /index
     /dts
     /currencies

/index/dts
^^^^^^^^^^

This node contains a 1-dimensional array of int64s. When interpreted as an
np.datetime64[ns], each entry in the array defines the date label for the
corresponding row of a data node.

/index/currencies
^^^^^^^^^^^^^^^^^

This node contains a 1-dimensional array of length-3 strings. Each entry in the
array defines the the label for the corresponding column of a data node.

/data/{rate}/{quote_currency}/rates
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Nodes at these locations are data nodes. They store 2-dimensional arrays of
exchange rates.

Represented as a DataFrame with appropriate indices, the data at
/data/rate1/USD/rates might look like this::

                    USD       EUR       CAD
   2014-01-01  1.000000  0.808477  0.442329
   2014-01-02  1.000000  2.235770  0.817352
   2014-01-03  1.000000  0.378603  1.181822
   2014-01-04  1.000000  0.090983  0.198182
   2014-01-05  1.000000  1.135781  0.536609

Each column of the array contains exchange rates mapping from currency of the
column's label to the quote encoded in the node's path (USD in this
example). The label for column i in a data node is the ith element of
/index/currencies.

Each row of the array contains exchange rates for a given date. The label for
row i in a data node is the ith element of /index/dts.
"""
from interface import implements
import h5py
from logbook import Logger
import numpy as np
import pandas as pd

from zipline.utils.memoize import lazyval

from .base import FXRateReader

INDEX = 'index'
DATA = 'data'
CURRENCIES = 'currencies'
DTS = 'dts'
RATES = 'rates'
DEFAULT = 'default'

log = Logger(__name__)


class HDF5FXRateReader(implements(FXRateReader)):
    """An FXRateReader backed by HDF5.

    Parameters
    ----------
    group : h5py.Group
        Top-level group written by an :class:`HDF5FXRateWriter`.
    """

    def __init__(self, group, default_rate=None):
        self._group = group
        self._default_rate = default_rate

    @classmethod
    def from_path(cls, path):
        """Construct from a file path.

        Parameters
        ----------
        path : str
            Path to an HDF5 fx rates file.
        """
        return cls(h5py.File(path))

    @lazyval
    def dts(self):
        """Row labels for rate groups.
        """
        return self._group[INDEX][DTS][:].astype('M8[ns]')

    @lazyval
    def currencies(self):
        """Column labels for rate groups.
        """
        return pd.Index(self._group[INDEX][CURRENCIES][:])

    def get_rates(self, rate, quote, bases, dts):
        """Get rates to convert ``bases`` into ``quote``.
        """
        dts = dts.values
        self._check_dts(self.dts, dts)

        if rate == DEFAULT:
            if self._default_rate is None:
                raise ValueError(
                    "FX rates not available for rate={}. "
                    "No default rate was configured.".format(DEFAULT)
                )

        row_ixs = self.dts.searchsorted(dts, side='right') - 1
        col_ixs = self.currencies.get_indexer(bases)

        return self._read_rate_block(rate, quote, row_ixs, col_ixs)

    def _read_rate_block(self, rate, quote, row_ixs, col_ixs):
        try:
            dataset = self._group[DATA][rate][quote][RATES]
        except KeyError:
            raise ValueError(
                "FX rates not available for rate={}, quote_currency={}."
                .format(rate, quote)
            )

        # There aren't many columns in the output array, so it's easier to pull
        # all columns and reindex in memory. For rows, however, a quick and
        # easy optimization is to pull just the slice from min(row_ixs) to
        # max(row_ixs).
        min_row = row_ixs[0]
        max_row = row_ixs[-1]
        rows = dataset[min_row:max_row + 1]  # +1 to be inclusive of end

        return rows[row_ixs - min_row][:, col_ixs]

    def _check_dts(self, stored, requested):
        """Validate that requested dates are in bounds for what we have stored.
        """
        request_start, request_end = requested[[0, -1]]
        data_start, data_end = stored[[0, -1]]

        if request_start < data_start:
            raise ValueError(
                "Requested fx rates starting at {}, but data starts at {}"
                .format(request_start, data_start)
            )

        if request_end > data_end:
            raise ValueError(
                "Requested fx rates ending at {}, but data ends at {}"
                .format(request_end, data_end)
            )


class HDF5FXRateWriter(object):
    """Writer class for HDF5 files consumed by HDF5FXRateReader.
    """
    def __init__(self, group):
        self._group = group

    def write(self, dts, currencies, data):
        """Write data to the file.

        Parameters
        ----------
        dts : pd.DatetimeIndex
            Index of row labels for rates to be written.
        currencies : np.array[object]
            Array of column labels for rates to be written.
        data : iterator[(str, str, np.array[float64])]
            Iterator of (rate, quote_currency, array) tuples. Each array
            should be of shape ``(len(dts), len(currencies))``, and should
            contain a table of rates where each column is a timeseries of rates
            mapping its column label's currency to ``quote_currency``.
        """
        self._write_index_group(dts, currencies)
        self._write_data_group(dts, currencies, data)

    def _write_index_group(self, dts, currencies):
        """Write content of /index.
        """
        index_group = self._group.create_group(INDEX)

        self._log_writing(INDEX, DTS)
        index_group.create_dataset(DTS, data=dts.astype('int64'))

        self._log_writing(INDEX, CURRENCIES)
        index_group.create_dataset(CURRENCIES, data=currencies.astype('S3'))

    def _write_data_group(self, dts, currencies, data):
        """Write content of /data.
        """
        data_group = self._group.create_group(DATA)

        expected_shape = (len(dts), len(currencies))
        for rate, quote, array in data:
            if array.shape != expected_shape:
                raise ValueError(
                    "Unexpected shape for rate={}, quote={}."
                    "\nExpected shape: {}. Got {}."
                    .format(rate, quote, expected_shape, array.shape)
                )

            self._log_writing(DATA, rate, quote)
            target = data_group.require_group('/'.join((rate, quote)))
            target.create_dataset(RATES, data=array)

    def _log_writing(self, *path):
        log.debug("Writing {}", '/'.join(path))
