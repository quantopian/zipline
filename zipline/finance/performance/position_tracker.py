from __future__ import division

import logbook
import numpy as np
import pandas as pd
from pandas.lib import checknull
try:
    # optional cython based OrderedDict
    from cyordereddict import OrderedDict
except ImportError:
    from collections import OrderedDict
from six import iteritems, itervalues

from zipline.protocol import Event, DATASOURCE_TYPE
from zipline.finance.slippage import Transaction
from zipline.utils.serialization_utils import (
    VERSION_LABEL
)

import zipline.protocol as zp
from zipline.assets import (
    Equity, Future
)
from zipline.errors import PositionTrackerMissingAssetFinder
from . position import positiondict

log = logbook.Logger('Performance')


class PositionTracker(object):

    def __init__(self, asset_finder):
        self.asset_finder = asset_finder

        # sid => position object
        self.positions = positiondict()
        # Arrays for quick calculations of positions value
        self._position_amounts = OrderedDict()
        self._position_last_sale_prices = OrderedDict()
        self._position_value_multipliers = OrderedDict()
        self._position_exposure_multipliers = OrderedDict()
        self._position_payout_multipliers = OrderedDict()
        self._unpaid_dividends = pd.DataFrame(
            columns=zp.DIVIDEND_PAYMENT_FIELDS,
        )
        self._positions_store = zp.Positions()

        # Dict, keyed on dates, that contains lists of close position events
        # for any Assets in this tracker's positions
        self._auto_close_position_sids = {}

    def _update_asset(self, sid):
        try:
            self._position_value_multipliers[sid]
            self._position_exposure_multipliers[sid]
            self._position_payout_multipliers[sid]
        except KeyError:
            # Check if there is an AssetFinder
            if self.asset_finder is None:
                raise PositionTrackerMissingAssetFinder()

            # Collect the value multipliers from applicable sids
            asset = self.asset_finder.retrieve_asset(sid)
            if isinstance(asset, Equity):
                self._position_value_multipliers[sid] = 1
                self._position_exposure_multipliers[sid] = 1
                self._position_payout_multipliers[sid] = 0
            if isinstance(asset, Future):
                self._position_value_multipliers[sid] = 0
                self._position_exposure_multipliers[sid] = \
                    asset.contract_multiplier
                self._position_payout_multipliers[sid] = \
                    asset.contract_multiplier
                # Futures are closed on their notice_date
                if asset.notice_date:
                    self._insert_auto_close_position_date(
                        dt=asset.notice_date,
                        sid=sid
                    )
                # If the Future does not have a notice_date, it will be closed
                # on its expiration_date
                elif asset.expiration_date:
                    self._insert_auto_close_position_date(
                        dt=asset.expiration_date,
                        sid=sid
                    )

    def _insert_auto_close_position_date(self, dt, sid):
        """
        Inserts the given SID in to the list of positions to be auto-closed by
        the given dt.

        Parameters
        ----------
        dt : pandas.Timestamp
            The date before-which the given SID will be auto-closed
        sid : int
            The SID of the Asset to be auto-closed
        """
        self._auto_close_position_sids.setdefault(dt, set()).add(sid)

    def auto_close_position_events(self, next_trading_day):
        """
        Generates CLOSE_POSITION events for any SIDs whose auto-close date is
        before or equal to the given date.

        Parameters
        ----------
        next_trading_day : pandas.Timestamp
            The time before-which certain Assets need to be closed

        Yields
        ------
        Event
            A close position event for any sids that should be closed before
            the next_trading_day parameter
        """
        past_asset_end_dates = set()

        # Check the auto_close_position_dates dict for SIDs to close
        for date, sids in self._auto_close_position_sids.items():
            if date > next_trading_day:
                continue
            past_asset_end_dates.add(date)

            for sid in sids:
                # Yield a CLOSE_POSITION event
                event = Event({
                    'dt': date,
                    'type': DATASOURCE_TYPE.CLOSE_POSITION,
                    'sid': sid,
                })
                yield event

        # Clear out past dates
        while past_asset_end_dates:
            self._auto_close_position_sids.pop(past_asset_end_dates.pop())

    def update_last_sale(self, event):
        # NOTE, PerformanceTracker already vetted as TRADE type
        sid = event.sid
        if sid not in self.positions:
            return 0

        price = event.price

        if checknull(price):
            return 0

        pos = self.positions[sid]
        old_price = pos.last_sale_price
        pos.last_sale_date = event.dt
        pos.last_sale_price = price
        self._position_last_sale_prices[sid] = price

        # Calculate cash adjustment on assets with multipliers
        return ((price - old_price) * self._position_payout_multipliers[sid]
                * pos.amount)

    def update_positions(self, positions):
        # update positions in batch
        self.positions.update(positions)
        for sid, pos in iteritems(positions):
            self._position_amounts[sid] = pos.amount
            self._position_last_sale_prices[sid] = pos.last_sale_price
            self._update_asset(sid)

    def update_position(self, sid, amount=None, last_sale_price=None,
                        last_sale_date=None, cost_basis=None):
        pos = self.positions[sid]

        if amount is not None:
            pos.amount = amount
            self._position_amounts[sid] = amount
            self._position_values = None  # invalidate cache
            self._update_asset(sid=sid)
        if last_sale_price is not None:
            pos.last_sale_price = last_sale_price
            self._position_last_sale_prices[sid] = last_sale_price
            self._position_values = None  # invalidate cache
        if last_sale_date is not None:
            pos.last_sale_date = last_sale_date
        if cost_basis is not None:
            pos.cost_basis = cost_basis

    def execute_transaction(self, txn):
        # Update Position
        # ----------------
        sid = txn.sid
        position = self.positions[sid]
        position.update(txn)
        self._position_amounts[sid] = position.amount
        self._position_last_sale_prices[sid] = position.last_sale_price
        self._update_asset(sid)

    def handle_commission(self, commission):
        # Adjust the cost basis of the stock if we own it
        if commission.sid in self.positions:
            self.positions[commission.sid].\
                adjust_commission_cost_basis(commission)

    @property
    def position_values(self):
        iter_amount_price_multiplier = zip(
            itervalues(self._position_amounts),
            itervalues(self._position_last_sale_prices),
            itervalues(self._position_value_multipliers),
        )
        return [
            price * amount * multiplier for
            price, amount, multiplier in iter_amount_price_multiplier
        ]

    @property
    def position_exposures(self):
        iter_amount_price_multiplier = zip(
            itervalues(self._position_amounts),
            itervalues(self._position_last_sale_prices),
            itervalues(self._position_exposure_multipliers),
        )
        return [
            price * amount * multiplier for
            price, amount, multiplier in iter_amount_price_multiplier
        ]

    def calculate_positions_value(self):
        if len(self.position_values) == 0:
            return np.float64(0)

        return sum(self.position_values)

    def calculate_positions_exposure(self):
        if len(self.position_exposures) == 0:
            return np.float64(0)

        return sum(self.position_exposures)

    def _longs_count(self):
        return sum(1 for i in self.position_exposures if i > 0)

    def _long_exposure(self):
        return sum(i for i in self.position_exposures if i > 0)

    def _long_value(self):
        return sum(i for i in self.position_values if i > 0)

    def _shorts_count(self):
        return sum(1 for i in self.position_exposures if i < 0)

    def _short_exposure(self):
        return sum(i for i in self.position_exposures if i < 0)

    def _short_value(self):
        return sum(i for i in self.position_values if i < 0)

    def _gross_exposure(self):
        return self._long_exposure() + abs(self._short_exposure())

    def _gross_value(self):
        return self._long_value() + abs(self._short_value())

    def _net_exposure(self):
        return self.calculate_positions_exposure()

    def _net_value(self):
        return self.calculate_positions_value()

    def handle_split(self, split):
        if split.sid in self.positions:
            # Make the position object handle the split. It returns the
            # leftover cash from a fractional share, if there is any.
            position = self.positions[split.sid]
            leftover_cash = position.handle_split(split)
            self._position_amounts[split.sid] = position.amount
            self._position_last_sale_prices[split.sid] = \
                position.last_sale_price
            self._update_asset(split.sid)
            return leftover_cash

    def _maybe_earn_dividend(self, dividend):
        """
        Take a historical dividend record and return a Series with fields in
        zipline.protocol.DIVIDEND_FIELDS (plus an 'id' field) representing
        the cash/stock amount we are owed when the dividend is paid.
        """
        if dividend['sid'] in self.positions:
            return self.positions[dividend['sid']].earn_dividend(dividend)
        else:
            return zp.dividend_payment()

    def earn_dividends(self, dividend_frame):
        """
        Given a frame of dividends whose ex_dates are all the next trading day,
        calculate and store the cash and/or stock payments to be paid on each
        dividend's pay date.
        """
        earned = dividend_frame.apply(self._maybe_earn_dividend, axis=1)\
                               .dropna(how='all')
        if len(earned) > 0:
            # Store the earned dividends so that they can be paid on the
            # dividends' pay_dates.
            self._unpaid_dividends = pd.concat(
                [self._unpaid_dividends, earned],
            )

    def _maybe_pay_dividend(self, dividend):
        """
        Take a historical dividend record, look up any stored record of
        cash/stock we are owed for that dividend, and return a Series
        with fields drawn from zipline.protocol.DIVIDEND_PAYMENT_FIELDS.
        """
        try:
            unpaid_dividend = self._unpaid_dividends.loc[dividend['id']]
            return unpaid_dividend
        except KeyError:
            return zp.dividend_payment()

    def pay_dividends(self, dividend_frame):
        """
        Given a frame of dividends whose pay_dates are all the next trading
        day, grant the cash and/or stock payments that were calculated on the
        given dividends' ex dates.
        """
        payments = dividend_frame.apply(self._maybe_pay_dividend, axis=1)\
                                 .dropna(how='all')

        # Mark these dividends as paid by dropping them from our unpaid
        # table.
        self._unpaid_dividends.drop(payments.index)

        # Add stock for any stock dividends paid.  Again, the values here may
        # be negative in the case of short positions.
        stock_payments = payments[payments['payment_sid'].notnull()]
        for _, row in stock_payments.iterrows():
            stock = row['payment_sid']
            share_count = row['share_count']
            # note we create a Position for stock dividend if we don't
            # already own the asset
            position = self.positions[stock]

            position.amount += share_count
            self._position_amounts[stock] = position.amount
            self._position_last_sale_prices[stock] = position.last_sale_price
            self._update_asset(stock)

        # Add cash equal to the net cash payed from all dividends.  Note that
        # "negative cash" is effectively paid if we're short an asset,
        # representing the fact that we're required to reimburse the owner of
        # the stock for any dividends paid while borrowing.
        net_cash_payment = payments['cash_amount'].fillna(0).sum()
        return net_cash_payment

    def maybe_create_close_position_transaction(self, event):
        if not self._position_amounts.get(event.sid):
            return None
        if 'price' in event:
            price = event.price
        else:
            price = self._position_last_sale_prices[event.sid]
        txn = Transaction(
            sid=event.sid,
            amount=(-1 * self._position_amounts[event.sid]),
            dt=event.dt,
            price=price,
            commission=0,
            order_id=0
        )
        return txn

    def get_positions(self):

        positions = self._positions_store

        for sid, pos in iteritems(self.positions):

            if pos.amount == 0:
                # Clear out the position if it has become empty since the last
                # time get_positions was called.  Catching the KeyError is
                # faster than checking `if sid in positions`, and this can be
                # potentially called in a tight inner loop.
                try:
                    del positions[sid]
                except KeyError:
                    pass
                continue

            # Note that this will create a position if we don't currently have
            # an entry
            position = positions[sid]
            position.amount = pos.amount
            position.cost_basis = pos.cost_basis
            position.last_sale_price = pos.last_sale_price
        return positions

    def get_positions_list(self):
        positions = []
        for sid, pos in iteritems(self.positions):
            if pos.amount != 0:
                positions.append(pos.to_dict())
        return positions

    def __getstate__(self):
        state_dict = {}

        state_dict['asset_finder'] = self.asset_finder
        state_dict['positions'] = dict(self.positions)
        state_dict['unpaid_dividends'] = self._unpaid_dividends

        # Asset-finder dependent dicts must be serialized
        state_dict['position_value_multipliers'] = \
            serialize_ordered_dict(self._position_value_multipliers)
        state_dict['position_exposure_multipliers'] = \
            serialize_ordered_dict(self._position_exposure_multipliers)
        state_dict['position_payout_multipliers'] = \
            serialize_ordered_dict(self._position_payout_multipliers)
        state_dict['auto_close_position_sids'] = self._auto_close_position_sids

        STATE_VERSION = 3
        state_dict[VERSION_LABEL] = STATE_VERSION
        return state_dict

    def __setstate__(self, state):
        OLDEST_SUPPORTED_STATE = 3
        version = state.pop(VERSION_LABEL)

        if version < OLDEST_SUPPORTED_STATE:
            raise BaseException("PositionTracker saved state is too old.")

        self.asset_finder = state['asset_finder']
        self.positions = positiondict()
        # note that positions_store is temporary and gets regened from
        # .positions
        self._positions_store = zp.Positions()

        self._unpaid_dividends = state['unpaid_dividends']

        # AssetFinder-dependent dicts are de-serialized
        self._position_value_multipliers = \
            deserialize_ordered_dict(state['position_value_multipliers'])
        self._position_exposure_multipliers = \
            deserialize_ordered_dict(state['position_exposure_multipliers'])
        self._position_payout_multipliers = \
            deserialize_ordered_dict(state['position_payout_multipliers'])
        self._auto_close_position_sids = state['auto_close_position_sids']

        # Arrays for quick calculations of positions value
        self._position_amounts = OrderedDict()
        self._position_last_sale_prices = OrderedDict()

        # Update positions is called without a finder
        self.update_positions(state['positions'])


def serialize_ordered_dict(ordered_dict):
    """
    Converts an OrderedDict in to a list of key/value pair tuples
    """
    return [(key, value) for key, value in ordered_dict.items()]


def deserialize_ordered_dict(serialized_ordered_dict):
    """
    Converts a list of key/value pair tuples in to an OrderedDict
    """
    result = OrderedDict()
    for key, value in serialized_ordered_dict:
        result[key] = value
    return result
