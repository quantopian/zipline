import pytz
import math
import logbook
import datetime

import zipline.protocol as zp
from zipline.protocol import SIMULATION_STYLE

log = logbook.Logger('Transaction Simulator')

class TransactionSimulator(object):
    UPDATER = True

    def __init__(self, open_orders, style=SIMULATION_STYLE.PARTIAL_VOLUME):
        self.open_orders                = open_orders
        self.txn_count                  = 0
        self.trade_window               = datetime.timedelta(seconds=30)
        self.orderTTL                   = datetime.timedelta(days=1)
        self.commission                 = 0.03

        if not style or style == SIMULATION_STYLE.PARTIAL_VOLUME:
            self.apply_trade_to_open_orders = self.simulate_with_partial_volume
        elif style == SIMULATION_STYLE.BUY_ALL:
            self.apply_trade_to_open_orders =  self.simulate_buy_all
        elif style == SIMULATION_STYLE.FIXED_SLIPPAGE:
            self.apply_trade_to_open_orders = self.simulate_with_fixed_cost
        elif style == SIMULATION_STYLE.NOOP:
            self.apply_trade_to_open_orders = self.simulate_noop

    def update(self, event):
        event.TRANSACTION = None
        if event.type == zp.DATASOURCE_TYPE.TRADE:
            event.TRANSACTION = self.apply_trade_to_open_orders(event)
        return event
        
    def simulate_buy_all(self, event):
        txn = self.create_transaction(
            event.sid,
            event.volume,
            event.price,
            event.dt,
            1
        )
        return txn

    def simulate_noop(self, event):
        return None

    def simulate_with_fixed_cost(self, event):
        if self.open_orders.has_key(event.sid):
            orders = self.open_orders[event.sid]
            orders = sorted(orders, key=lambda o: o.dt)
        else:
            return None

        amount = 0
        for order in orders:
            amount += order.amount

        if(amount == 0):
            return

        direction = amount / math.fabs(amount)

        txn = self.create_transaction(
            event.sid,
            amount,
            event.price + 0.10, # Magic constant?
            event.dt,
            direction
        )

        self.open_orders[event.sid] = []

        return txn

    def simulate_with_partial_volume(self, event):
        if(event.volume == 0):
            #there are zero volume events bc some stocks trade
            #less frequently than once per minute.
            return None

        if self.open_orders.has_key(event.sid):
            orders = self.open_orders[event.sid]
            orders = sorted(orders, key=lambda o: o.dt)
        else:
            return None

        dt = event.dt
        expired = []
        total_order = 0
        simulated_amount = 0
        simulated_impact = 0.0
        direction = 1.0
        for order in orders:

            if(order.dt < event.dt):

                # orders are only good on the day they are issued
                if order.dt.day < event.dt.day:
                    continue

                open_amount = order.amount - order.filled

                if(open_amount != 0):
                    direction = open_amount / math.fabs(open_amount)
                else:
                    direction = 1

                desired_order = total_order + open_amount

                volume_share = direction * (desired_order) / event.volume
                if volume_share > .25:
                    volume_share = .25
                simulated_amount = int(volume_share * event.volume * direction)
                simulated_impact = (volume_share)**2 * .1 * direction * event.price

                order.filled += (simulated_amount - total_order)
                total_order = simulated_amount

                # we cap the volume share at 25% of a trade
                if volume_share == .25:
                    break

        orders = [ x for x in orders if abs(x.amount - x.filled) > 0 and x.dt.day >= event.dt.day]

        self.open_orders[event.sid] = orders


        if simulated_amount != 0:
            return self.create_transaction(
                event.sid,
                simulated_amount,
                event.price + simulated_impact,
                dt.replace(tzinfo = pytz.utc),
                direction
            )

    def create_transaction(self, sid, amount, price, dt, direction):
        self.txn_count += 1
        txn = {'sid'            : sid,
                'amount'        : int(amount),
                'dt'            : dt,
                'price'         : price,
                'commission'    : self.commission * amount * direction
                }
        return zp.ndict(txn)


class TradingEnvironment(object):

    def __init__(
        self,
        benchmark_returns,
        treasury_curves,
        period_start    = None,
        period_end      = None,
        capital_base    = None,
        max_drawdown    = None
    ):

        self.trading_days = []
        self.trading_day_map = {}
        self.treasury_curves = treasury_curves
        self.benchmark_returns = benchmark_returns
        self.period_start = period_start
        self.period_end = period_end
        self.capital_base = capital_base
        self.period_trading_days = None
        self.max_drawdown = max_drawdown

        for bm in benchmark_returns:
            self.trading_days.append(bm.date)
            self.trading_day_map[bm.date] = bm

        self.first_open = self.calculate_first_open()
        self.last_close = self.calculate_last_close()

    def calculate_first_open(self):
        """
        Finds the first trading day on or after self.period_start.
        """
        first_open  = self.period_start
        one_day      = datetime.timedelta(days=1)

        while not self.is_trading_day(first_open):
            first_open = first_open + one_day

        first_open = self.set_NYSE_time(first_open, 9, 30)
        return first_open

    def calculate_last_close(self):
        """
        Finds the last trading day on or before self.period_end
        """
        last_close  = self.period_end
        one_day     = datetime.timedelta(days=1)

        while not self.is_trading_day(last_close):
            last_close = last_close - one_day

        last_close = self.set_NYSE_time(last_close, 16, 00)

        return last_close

    #TODO: add other exchanges and timezones...
    def set_NYSE_time(self, dt, hour, minute):
        naive = datetime.datetime(
            year=dt.year,
            month=dt.month,
            day=dt.day
        )
        local = pytz.timezone ('US/Eastern')
        local_dt = naive.replace (tzinfo = local)
        # set the clock to the opening bell in NYC time.
        local_dt = local_dt.replace(hour=hour, minute=minute)
        # convert to UTC
        utc_dt = local_dt.astimezone (pytz.utc)
        return utc_dt

    def normalize_date(self, test_date):
        return datetime.datetime(
            year=test_date.year,
            month=test_date.month,
            day=test_date.day,
            tzinfo=pytz.utc
        )

    @property
    def days_in_period(self):
        """return the number of trading days within the period [start, end)"""
        assert(self.period_start != None)
        assert(self.period_end != None)

        if self.period_trading_days == None:
            self.period_trading_days = []
            for date in self.trading_days:
                if date > self.period_end:
                    break
                if date >= self.period_start:
                    self.period_trading_days.append(date)


        return len(self.period_trading_days)

    def is_market_hours(self, test_date):
        if not self.is_trading_day(test_date):
            return False

        mkt_open = self.set_NYSE_time(test_date, 9, 30)
        #TODO: half days?
        mkt_close = self.set_NYSE_time(test_date, 16, 00)

        return test_date >= mkt_open and test_date <= mkt_close

    def is_trading_day(self, test_date):
        dt = self.normalize_date(test_date)
        return self.trading_day_map.has_key(dt)

    def get_benchmark_daily_return(self, test_date):
        date = self.normalize_date(test_date)
        if self.trading_day_map.has_key(date):
            return self.trading_day_map[date].returns
        else:
            return 0.0
