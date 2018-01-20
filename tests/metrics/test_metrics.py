import numpy as np
import pandas as pd

from zipline import api
from zipline.assets import Equity, Future
from zipline.assets.synthetic import make_commodity_future_info
from zipline.data.data_portal import DataPortal
from zipline.data.resample import MinuteResampleSessionBarReader
from zipline.testing import (
    parameter_space,
    prices_generating_returns,
    simulate_minutes_for_day,
)
from zipline.testing.fixtures import (
    WithMakeAlgo,
    WithConstantEquityMinuteBarData,
    WithConstantFutureMinuteBarData,
    WithWerror,
    ZiplineTestCase,
)
from zipline.testing.predicates import assert_equal, wildcard


def T(cs):
    return pd.Timestamp(cs, tz='utc')


def portfolio_snapshot(p):
    """Extract all of the fields from the portfolio as a new dictionary.
    """
    fields = (
        'cash_flow',
        'starting_cash',
        'portfolio_value',
        'pnl',
        'returns',
        'cash',
        'positions',
        'positions_value',
        'positions_exposure',
    )
    return {field: getattr(p, field) for field in fields}


class TestConstantPrice(WithConstantEquityMinuteBarData,
                        WithConstantFutureMinuteBarData,
                        WithMakeAlgo,
                        WithWerror,
                        ZiplineTestCase):
    EQUITY_DAILY_BAR_SOURCE_FROM_MINUTE = True
    FUTURE_DAILY_BAR_SOURCE_FROM_MINUTE = True

    ASSET_FINDER_EQUITY_SIDS = [ord('A')]

    EQUITY_MINUTE_CONSTANT_LOW = 1.0
    EQUITY_MINUTE_CONSTANT_OPEN = 1.0
    EQUITY_MINUTE_CONSTANT_CLOSE = 1.0
    EQUITY_MINUTE_CONSTANT_HIGH = 1.0
    EQUITY_MINUTE_CONSTANT_VOLUME = 100.0

    FUTURE_MINUTE_CONSTANT_LOW = 1.0
    FUTURE_MINUTE_CONSTANT_OPEN = 1.0
    FUTURE_MINUTE_CONSTANT_CLOSE = 1.0
    FUTURE_MINUTE_CONSTANT_HIGH = 1.0
    FUTURE_MINUTE_CONSTANT_VOLUME = 100.0

    START_DATE = T('2014-01-06')
    END_DATE = T('2014-01-10')

    # note: class attributes after this do not configure fixtures, they are
    # just used in this test suite

    # we use a contract multiplier to make sure we are correctly calculating
    # exposure as price * multiplier
    future_contract_multiplier = 2

    # this is the expected exposure for a position of one contract
    future_constant_exposure = (
        FUTURE_MINUTE_CONSTANT_CLOSE * future_contract_multiplier
    )

    @classmethod
    def make_futures_info(cls):
        return make_commodity_future_info(
            first_sid=ord('Z'),
            root_symbols=['Z'],
            years=[cls.START_DATE.year],
            multiplier=cls.future_contract_multiplier,
        )

    @classmethod
    def init_class_fixtures(cls):
        super(TestConstantPrice, cls).init_class_fixtures()

        cls.equity = cls.asset_finder.retrieve_asset(
            cls.asset_finder.equities_sids[0],
        )
        cls.future = cls.asset_finder.retrieve_asset(
            cls.asset_finder.futures_sids[0],
        )

        cls.trading_minutes = pd.Index(
            cls.trading_calendar.minutes_for_sessions_in_range(
                cls.START_DATE,
                cls.END_DATE,
            ),
        )
        cls.closes = pd.Index(
            cls.trading_calendar.session_closes_in_range(
                cls.START_DATE,
                cls.END_DATE,
            ),
        )
        cls.closes.name = None

    def test_nop(self):
        perf = self.run_algorithm()

        zeros = pd.Series(0.0, index=self.closes)
        all_zero_fields = [
            'algorithm_period_return',
            'benchmark_period_return',
            'benchmark_returns',
            'capital_used',
            'downside_risk',
            'excess_return',
            'long_exposure',
            'long_value',
            'longs_count',
            'max_drawdown',
            'max_leverage',
            'short_exposure',
            'short_value',
            'shorts_count',
            'treasury_period_return',
        ]

        for field in all_zero_fields:
            assert_equal(
                perf[field],
                zeros,
                check_names=False,
                check_dtype=False,
                msg=field,
            )

        nan_then_zero = pd.Series(0.0, index=self.closes)
        nan_then_zero[0] = float('nan')
        nan_then_zero_fields = (
            'algo_volatility',
            'algorithm_volatility',
            'benchmark_volatility',
        )
        for field in nan_then_zero_fields:
            assert_equal(
                perf[field],
                nan_then_zero,
                check_names=False,
                msg=field,
            )

        empty_lists = pd.Series([[]] * len(self.closes), self.closes)
        empty_list_fields = (
            'orders',
            'positions',
            'transactions',
        )
        for field in empty_list_fields:
            assert_equal(
                perf[field],
                empty_lists,
                check_names=False,
                msg=field,
            )

    @parameter_space(
        direction=['long', 'short'],
        # checking the portfolio forces a sync; we want to ensure that the
        # perf packets are correct even without explicitly requesting the
        # portfolio every day. we also want to test that ``context.portfolio``
        # produces the expected values when queried mid-simulation
        check_portfolio_during_simulation=[True, False],
    )
    def test_equity_slippage(self,
                             direction,
                             check_portfolio_during_simulation):
        if direction not in ('long', 'short'):
            raise ValueError(
                'direction must be either long or short, got: %r' % direction,
            )

        # the number of shares to order, this will be filled one share at a
        # time
        shares = 100

        # random values in the range [0, 5) rounded to 3 decimal points
        st = np.random.RandomState(1868655980)
        per_fill_slippage = st.uniform(0, 5, shares).round(3)

        if direction == 'short':
            per_fill_slippage = -per_fill_slippage
            shares = -shares

        slippage_iter = iter(per_fill_slippage)

        class TestingSlippage(api.slippage.SlippageModel):
            @staticmethod
            def process_order(data, order):
                return (
                    self.EQUITY_MINUTE_CONSTANT_CLOSE + next(slippage_iter),
                    1 if direction == 'long' else -1,
                )

        if check_portfolio_during_simulation:
            portfolio_snapshots = {}

            def check_portfolio(context):
                # force the portfolio even on the first bar
                portfolio = context.portfolio
                portfolio_snapshots[api.get_datetime()] = portfolio_snapshot(
                    portfolio,
                )

                if context.bar_count < 1:
                    assert_equal(portfolio.positions, {})
                    return

                expected_amount = min(context.bar_count, 100)
                if direction == 'short':
                    expected_amount = -expected_amount

                expected_position = {
                    'asset': self.equity,
                    'last_sale_date': api.get_datetime(),
                    'last_sale_price': self.EQUITY_MINUTE_CONSTANT_CLOSE,
                    'amount': expected_amount,
                    'cost_basis': (
                        self.EQUITY_MINUTE_CONSTANT_CLOSE +
                        per_fill_slippage[:context.bar_count].mean()
                    ),
                }
                expected_positions = {self.equity: [expected_position]}

                positions = {
                    asset: [{k: getattr(p, k) for k in expected_position}]
                    for asset, p in portfolio.positions.items()
                }

                assert_equal(positions, expected_positions)
        else:
            def check_portfolio(context):
                pass

        def initialize(context):
            api.set_slippage(TestingSlippage())
            api.set_commission(api.commission.NoCommission())
            context.bar_count = 0

        def handle_data(context, data):
            if context.bar_count == 0:
                api.order(self.equity, shares)

            check_portfolio(context)
            context.bar_count += 1

        perf = self.run_algorithm(
            initialize=initialize,
            handle_data=handle_data,
        )

        first_day_returns = -(
            abs(per_fill_slippage.sum()) / self.SIM_PARAMS_CAPITAL_BASE
        )
        expected_returns = pd.Series(0.0, index=self.closes)
        expected_returns.iloc[0] = first_day_returns

        assert_equal(
            perf['returns'],
            expected_returns,
            check_names=False,
        )

        expected_cumulative_returns = pd.Series(
            first_day_returns,
            index=self.closes,
        )

        assert_equal(
            perf['algorithm_period_return'],
            expected_cumulative_returns,
            check_names=False,
        )

        first_day_capital_used = -(
            shares * self.EQUITY_MINUTE_CONSTANT_CLOSE +
            abs(per_fill_slippage.sum())
        )
        expected_capital_used = pd.Series(0.0, index=self.closes)
        expected_capital_used.iloc[0] = first_day_capital_used

        assert_equal(
            perf['capital_used'],
            expected_capital_used,
            check_names=False,
        )

        if not check_portfolio_during_simulation:
            return

        portfolio_snapshots = pd.DataFrame.from_dict(
            portfolio_snapshots,
            orient='index',
        )

        # each minute our cash flow is the share filled (if any) plus the
        # slippage for that minute
        minutely_cash_flow = pd.Series(0.0, index=self.trading_minutes)
        minutely_cash_flow[1:abs(shares) + 1] = (
            -(per_fill_slippage + self.EQUITY_MINUTE_CONSTANT_CLOSE)
            if direction == 'long' else
            (per_fill_slippage + self.EQUITY_MINUTE_CONSTANT_CLOSE)
        )
        expected_cash_flow = minutely_cash_flow.cumsum()

        assert_equal(
            portfolio_snapshots['cash_flow'],
            expected_cash_flow,
            check_names=False,
        )

        # Our pnl should just be the cost of the slippage incurred. This is
        # because we trade from cash into a position which holds 100% of its
        # value, but we lose the slippage on the way into that position.
        minutely_pnl = pd.Series(0.0, index=self.trading_minutes)
        minutely_pnl[1:abs(shares) + 1] = -np.abs(per_fill_slippage)
        expected_pnl = minutely_pnl.cumsum()

        assert_equal(
            portfolio_snapshots['pnl'],
            expected_pnl,
            check_names=False,
        )

        # the divisor is capital base because this is cumulative returns
        expected_returns = expected_pnl / self.SIM_PARAMS_CAPITAL_BASE

        assert_equal(
            portfolio_snapshots['returns'],
            expected_returns,
            check_names=False,
        )

    @parameter_space(
        direction=['long', 'short'],
        # checking the portfolio forces a sync; we want to ensure that the
        # perf packets are correct even without explicitly requesting the
        # portfolio every day. we also want to test that ``context.portfolio``
        # produces the expected values when queried mid-simulation
        check_portfolio_during_simulation=[True, False],
    )
    def test_equity_commissions(self,
                                direction,
                                check_portfolio_during_simulation):
        if direction not in ('long', 'short'):
            raise ValueError(
                'direction must be either long or short, got: %r' % direction,
            )

        shares = 100

        # random values in the range [0, 5) rounded to 3 decimal points
        st = np.random.RandomState(1868655980)
        per_fill_commission = st.uniform(0, 5, shares).round(3)
        commission_iter = iter(per_fill_commission)

        if direction == 'short':
            shares = -shares

        class SplitOrderButIncurNoSlippage(api.slippage.SlippageModel):
            """This model fills 1 share at a time, but otherwise fills with no
            penalty.
            """
            @staticmethod
            def process_order(data, order):
                return (
                    self.EQUITY_MINUTE_CONSTANT_CLOSE,
                    1 if direction == 'long' else -1,
                )

        class TestingCommission(api.commission.CommissionModel):
            @staticmethod
            def calculate(order, transaction):
                return next(commission_iter)

        if check_portfolio_during_simulation:
            portfolio_snapshots = {}

            def check_portfolio(context):
                # force the portfolio even on the first bar
                portfolio = context.portfolio
                portfolio_snapshots[api.get_datetime()] = portfolio_snapshot(
                    portfolio,
                )

                if context.bar_count < 1:
                    assert_equal(portfolio.positions, {})
                    return

                expected_amount = min(context.bar_count, 100)
                if direction == 'short':
                    expected_amount = -expected_amount

                expected_position = {
                    'asset': self.equity,
                    'last_sale_date': api.get_datetime(),
                    'last_sale_price': self.EQUITY_MINUTE_CONSTANT_CLOSE,
                    'amount': expected_amount,
                    'cost_basis': (
                        self.EQUITY_MINUTE_CONSTANT_CLOSE +
                        np.copysign(
                            per_fill_commission[:context.bar_count].mean(),
                            expected_amount,
                        )
                    ),
                }
                expected_positions = {self.equity: [expected_position]}

                positions = {
                    asset: [{k: getattr(p, k) for k in expected_position}]
                    for asset, p in portfolio.positions.items()
                }

                assert_equal(positions, expected_positions)
        else:
            def check_portfolio(context):
                pass

        def initialize(context):
            api.set_slippage(SplitOrderButIncurNoSlippage())
            api.set_commission(TestingCommission())
            context.bar_count = 0

        def handle_data(context, data):
            if context.bar_count == 0:
                api.order(self.equity, shares)

            check_portfolio(context)
            context.bar_count += 1

        perf = self.run_algorithm(
            initialize=initialize,
            handle_data=handle_data,
        )

        first_day_returns = -(
            abs(per_fill_commission.sum()) / self.SIM_PARAMS_CAPITAL_BASE
        )
        expected_returns = pd.Series(0.0, index=self.closes)
        expected_returns.iloc[0] = first_day_returns

        assert_equal(
            perf['returns'],
            expected_returns,
            check_names=False,
        )

        expected_cumulative_returns = pd.Series(
            first_day_returns,
            index=self.closes,
        )

        assert_equal(
            perf['algorithm_period_return'],
            expected_cumulative_returns,
            check_names=False,
        )

        first_day_capital_used = -(
            shares * self.EQUITY_MINUTE_CONSTANT_CLOSE +
            per_fill_commission.sum()
        )
        expected_capital_used = pd.Series(0.0, index=self.closes)
        expected_capital_used.iloc[0] = first_day_capital_used

        assert_equal(
            perf['capital_used'],
            expected_capital_used,
            check_names=False,
        )

        if not check_portfolio_during_simulation:
            return

        portfolio_snapshots = pd.DataFrame.from_dict(
            portfolio_snapshots,
            orient='index',
        )

        # each minute our cash flow is the share filled (if any) plus the
        # commission for that minute
        minutely_cash_flow = pd.Series(0.0, index=self.trading_minutes)
        minutely_cash_flow[1:abs(shares) + 1] = (
            -(self.EQUITY_MINUTE_CONSTANT_CLOSE + per_fill_commission)
            if direction == 'long' else
            (self.EQUITY_MINUTE_CONSTANT_CLOSE - per_fill_commission)
        )
        expected_cash_flow = minutely_cash_flow.cumsum()

        assert_equal(
            portfolio_snapshots['cash_flow'],
            expected_cash_flow,
            check_names=False,
        )

        # Our pnl should just be the cost of the commission incurred. This is
        # because we trade from cash into a position which holds 100% of its
        # value, but we lose the commission on the way into that position.
        minutely_pnl = pd.Series(0.0, index=self.trading_minutes)
        minutely_pnl[1:abs(shares) + 1] = -per_fill_commission
        expected_pnl = minutely_pnl.cumsum()

        assert_equal(
            portfolio_snapshots['pnl'],
            expected_pnl,
            check_names=False,
        )

        # the divisor is capital base because this is cumulative returns
        expected_returns = expected_pnl / self.SIM_PARAMS_CAPITAL_BASE

        assert_equal(
            portfolio_snapshots['returns'],
            expected_returns,
            check_names=False,
        )

    @parameter_space(
        direction=['long', 'short'],
        # checking the portfolio forces a sync; we want to ensure that the
        # perf packets are correct even without explicitly requesting the
        # portfolio every day. we also want to test that ``context.portfolio``
        # produces the expected values when queried mid-simulation
        check_portfolio_during_simulation=[True, False],
    )
    def test_equity_single_position(self,
                                    direction,
                                    check_portfolio_during_simulation):
        if direction not in ('long', 'short'):
            raise ValueError(
                'direction must be either long or short, got: %r' % direction,
            )

        shares = 1 if direction == 'long' else -1

        def initialize(context):
            api.set_benchmark(self.equity)

            api.set_slippage(api.slippage.NoSlippage())
            api.set_commission(api.commission.NoCommission())

            context.first_bar = True

        if check_portfolio_during_simulation:
            portfolio_snapshots = {}

            def check_portfolio(context, first_bar):
                portfolio = context.portfolio
                portfolio_snapshots[api.get_datetime()] = portfolio_snapshot(
                    portfolio,
                )

                positions = portfolio.positions
                if first_bar:
                    assert_equal(positions, {})
                    return

                assert_equal(positions.keys(), [self.equity])
                position = positions[self.equity]
                assert_equal(position.last_sale_date, api.get_datetime())
                assert_equal(position.amount, shares)
                assert_equal(
                    position.last_sale_price,
                    self.EQUITY_MINUTE_CONSTANT_CLOSE,
                )
                assert_equal(position.asset, self.equity)
                assert_equal(
                    position.cost_basis,
                    self.EQUITY_MINUTE_CONSTANT_CLOSE,
                )
        else:
            def check_portfolio(context, first_bar):
                pass

        def handle_data(context, data):
            first_bar = context.first_bar
            if first_bar:
                api.order(self.equity, shares)
                context.first_bar = False

            # take the snapshot after the order; ordering does not affect
            # the portfolio on the bar of the order, only the following bars
            check_portfolio(context, first_bar)

        perf = self.run_algorithm(
            initialize=initialize,
            handle_data=handle_data,
        )

        zeros = pd.Series(0.0, index=self.closes)
        all_zero_fields = [
            'algorithm_period_return',
            'benchmark_period_return',
            'benchmark_returns',
            'downside_risk',
            'excess_return',
            'max_drawdown',
            'treasury_period_return',
        ]
        if direction == 'long':
            all_zero_fields.extend((
                'short_value',
                'shorts_count',
            ))
        else:
            all_zero_fields.extend((
                'long_value',
                'longs_count',
            ))
        for field in all_zero_fields:
            assert_equal(
                perf[field],
                zeros,
                check_names=False,
                check_dtype=False,
                msg=field,
            )

        ones = pd.Series(1, index=self.closes)
        if direction == 'long':
            count_field = 'longs_count'
        else:
            count_field = 'shorts_count'

        assert_equal(
            perf[count_field],
            ones,
            check_names=False,
            msg=field,
        )

        if direction == 'long':
            expected_exposure = pd.Series(
                self.EQUITY_MINUTE_CONSTANT_CLOSE,
                index=self.closes,
            )
            for field in 'long_value', 'long_exposure':
                assert_equal(
                    perf[field],
                    expected_exposure,
                    check_names=False,
                )
        else:
            expected_exposure = pd.Series(
                -self.EQUITY_MINUTE_CONSTANT_CLOSE,
                index=self.closes,
            )
            for field in 'short_value', 'short_exposure':
                assert_equal(
                    perf[field],
                    expected_exposure,
                    check_names=False,
                )

        nan_then_zero = pd.Series(0.0, index=self.closes)
        nan_then_zero[0] = float('nan')
        nan_then_zero_fields = (
            'algo_volatility',
            'algorithm_volatility',
            'benchmark_volatility',
        )
        for field in nan_then_zero_fields:
            assert_equal(
                perf[field],
                nan_then_zero,
                check_names=False,
                check_dtype=False,
                msg=field,
            )

        capital_base_series = pd.Series(
            self.SIM_PARAMS_CAPITAL_BASE,
            index=self.closes,
        )

        # with no commissions, slippage, or returns our portfolio value stays
        # constant (at the capital base)
        assert_equal(
            perf['portfolio_value'],
            capital_base_series,
            check_names=False,
        )

        # leverage is gross market exposure / current notional capital
        # gross market exposure is
        # sum(long_exposure) + sum(abs(short_exposure))
        # current notional capital is the current portfolio value
        expected_max_leverage = (
            # we are exposed to only one share, the portfolio value is the
            # capital_base because we have no commissions, slippage, or
            # returns
            self.EQUITY_MINUTE_CONSTANT_CLOSE / capital_base_series
        )
        assert_equal(
            perf['max_leverage'],
            expected_max_leverage,
            check_names=False,
        )

        expected_cash = capital_base_series.copy()
        if direction == 'long':
            # we purchased one share on the first day
            cash_modifier = -self.EQUITY_MINUTE_CONSTANT_CLOSE
        else:
            # we sold one share on the first day
            cash_modifier = +self.EQUITY_MINUTE_CONSTANT_CLOSE

        expected_cash[1:] += cash_modifier

        assert_equal(
            perf['starting_cash'],
            expected_cash,
            check_names=False,
        )

        expected_cash[0] += cash_modifier
        assert_equal(
            perf['ending_cash'],
            expected_cash,
            check_names=False,
        )

        # we purchased one share on the first day
        expected_capital_used = pd.Series(0.0, index=self.closes)
        expected_capital_used[0] += cash_modifier

        assert_equal(
            perf['capital_used'],
            expected_capital_used,
            check_names=False,
        )

        # we hold one share so our positions exposure is that one share's price
        expected_position_exposure = pd.Series(
            -cash_modifier,
            index=self.closes,
        )
        for field in 'ending_value', 'ending_exposure':
            # for equities, position value and position exposure are the same
            assert_equal(
                perf[field],
                expected_position_exposure,
                check_names=False,
                msg=field,
            )

        # we don't start with any positions; the first day has no starting
        # exposure
        expected_position_exposure[0] = 0
        for field in 'starting_value', 'starting_exposure':
            # for equities, position value and position exposure are the same
            assert_equal(
                perf[field],
                expected_position_exposure,
                check_names=False,
                msg=field,
            )

        assert_equal(
            perf['trading_days'],
            pd.Series(
                np.arange(len(self.closes)) + 1,
                index=self.closes,
            ),
            check_names=False,
        )

        all_nan = pd.Series(np.nan, index=self.closes)
        all_nan_fields = (
            'alpha',
            'beta',
            'sortino',
        )
        for field in all_nan_fields:
            assert_equal(
                perf[field],
                all_nan,
                check_names=False,
                msg=field,
            )

        orders = perf['orders']

        expected_single_order = {
            'amount': shares,
            'commission': 0.0,
            'created': T('2014-01-06 14:31'),
            'dt': T('2014-01-06 14:32'),
            'filled': shares,
            'id': wildcard,
            'limit': None,
            'limit_reached': False,
            'reason': None,
            'sid': self.equity,
            'status': 1,
            'stop': None,
            'stop_reached': False
        }

        # we only order on the first day
        expected_orders = (
            [[expected_single_order]] +
            [[]] * (len(self.closes) - 1)
        )

        assert_equal(
            orders.tolist(),
            expected_orders,
            check_names=False,
        )
        assert_equal(
            orders.index,
            self.closes,
            check_names=False,
        )

        transactions = perf['transactions']

        expected_single_transaction = {
            'amount': shares,
            'commission': None,
            'dt': T('2014-01-06 14:32'),
            'order_id': wildcard,
            'price': 1.0,
            'sid': self.equity,
        }

        # since we only order on the first day, we should only transact on the
        # first day
        expected_transactions = (
            [[expected_single_transaction]] +
            [[]] * (len(self.closes) - 1)
        )

        assert_equal(
            transactions.tolist(),
            expected_transactions,
        )
        assert_equal(
            transactions.index,
            self.closes,
            check_names=False,
        )

        if not check_portfolio_during_simulation:
            return

        portfolio_snapshots = pd.DataFrame.from_dict(
            portfolio_snapshots,
            orient='index',
        )

        expected_cash = pd.Series(
            self.SIM_PARAMS_CAPITAL_BASE,
            index=self.trading_minutes,
        )
        if direction == 'long':
            expected_cash.iloc[1:] -= self.EQUITY_MINUTE_CONSTANT_CLOSE
        else:
            expected_cash.iloc[1:] += self.EQUITY_MINUTE_CONSTANT_CLOSE
        assert_equal(
            portfolio_snapshots['cash'],
            expected_cash,
            check_names=False,
        )

        expected_portfolio_capital_used = pd.Series(
            cash_modifier,
            index=self.trading_minutes,
        )
        expected_portfolio_capital_used[0] = 0.0
        expected_capital_used[0] = 0
        assert_equal(
            portfolio_snapshots['cash_flow'],
            expected_portfolio_capital_used,
            check_names=False,
        )

        zero_minutes = pd.Series(0.0, index=self.trading_minutes)
        for field in 'pnl', 'returns':
            assert_equal(
                portfolio_snapshots[field],
                zero_minutes,
                check_names=False,
                msg=field,
            )

        reindex_columns = sorted(
            set(portfolio_snapshots.columns) - {
                'starting_cash',
                'cash_flow',
                'pnl',
                'returns',
                'positions',
            },
        )
        minute_reindex = perf.rename(
            columns={
                'capital_used': 'cash_flow',
                'ending_cash': 'cash',
                'ending_exposure': 'positions_exposure',
                'ending_value': 'positions_value',
            },
        )[reindex_columns].reindex(
            self.trading_minutes,
            method='bfill',
        )

        first_minute = self.trading_minutes[0]
        # the first minute should have the default values because we haven't
        # done anything yet
        minute_reindex.loc[first_minute, 'cash'] = (
            self.SIM_PARAMS_CAPITAL_BASE
        )
        minute_reindex.loc[
            first_minute,
            ['positions_exposure', 'positions_value'],
        ] = 0

        assert_equal(
            portfolio_snapshots[reindex_columns],
            minute_reindex,
            check_names=False,
        )

    @parameter_space(
        direction=['long', 'short'],
        # checking the portfolio forces a sync; we want to ensure that the
        # perf packets are correct even without explicitly requesting the
        # portfolio every day. we also want to test that ``context.portfolio``
        # produces the expected values when queried mid-simulation
        check_portfolio_during_simulation=[True, False],
    )
    def test_future_single_position(self,
                                    direction,
                                    check_portfolio_during_simulation):
        if direction not in ('long', 'short'):
            raise ValueError(
                'direction must be either long or short, got: %r' % direction,
            )

        if direction == 'long':
            contracts = 1
            expected_exposure = self.future_constant_exposure
        else:
            contracts = -1
            expected_exposure = -self.future_constant_exposure

        def initialize(context):
            api.set_benchmark(self.equity)

            api.set_slippage(us_futures=api.slippage.NoSlippage())
            api.set_commission(us_futures=api.commission.NoCommission())

            context.first_bar = True

        if check_portfolio_during_simulation:
            portfolio_snapshots = {}

            def check_portfolio(context, first_bar):
                portfolio = context.portfolio
                portfolio_snapshots[api.get_datetime()] = portfolio_snapshot(
                    portfolio,
                )

                positions = portfolio.positions
                if first_bar:
                    assert_equal(positions, {})
                    return

                assert_equal(positions.keys(), [self.future])
                position = positions[self.future]
                assert_equal(position.last_sale_date, api.get_datetime())
                assert_equal(position.amount, contracts)
                assert_equal(
                    position.last_sale_price,
                    self.FUTURE_MINUTE_CONSTANT_CLOSE,
                )
                assert_equal(position.asset, self.future)
                assert_equal(
                    position.cost_basis,
                    self.FUTURE_MINUTE_CONSTANT_CLOSE,
                )
        else:
            def check_portfolio(context, first_bar):
                pass

        def handle_data(context, data):
            first_bar = context.first_bar
            if first_bar:
                api.order(self.future, contracts)
                context.first_bar = False

            # take the snapshot after the order; ordering does not affect
            # the portfolio on the bar of the order, only the following bars
            check_portfolio(context, first_bar)

        perf = self.run_algorithm(
            initialize=initialize,
            handle_data=handle_data,
        )

        zeros = pd.Series(0.0, index=self.closes)
        all_zero_fields = [
            'algorithm_period_return',
            'benchmark_period_return',
            'benchmark_returns',
            'downside_risk',
            'excess_return',
            'max_drawdown',
            'treasury_period_return',

            # futures contracts have no value, just exposure
            'starting_value',
            'ending_value',
            'long_value',
            'short_value',
        ]
        if direction == 'long':
            all_zero_fields.extend((
                'short_value',
                'shorts_count',
            ))
        else:
            all_zero_fields.extend((
                'long_value',
                'longs_count',
            ))

        for field in all_zero_fields:
            assert_equal(
                perf[field],
                zeros,
                check_names=False,
                check_dtype=False,
                msg=field,
            )

        ones = pd.Series(1, index=self.closes)
        count_field = direction + 's_count'
        assert_equal(
            perf[count_field],
            ones,
            check_names=False,
            msg=count_field,
        )

        expected_exposure_series = pd.Series(
            expected_exposure,
            index=self.closes,
        )
        exposure_field = direction + '_exposure'
        assert_equal(
            perf[exposure_field],
            expected_exposure_series,
            check_names=False,
            msg=exposure_field,
        )

        nan_then_zero = pd.Series(0.0, index=self.closes)
        nan_then_zero[0] = float('nan')
        nan_then_zero_fields = (
            'algo_volatility',
            'algorithm_volatility',
            'benchmark_volatility',
        )
        for field in nan_then_zero_fields:
            assert_equal(
                perf[field],
                nan_then_zero,
                check_names=False,
                check_dtype=False,
                msg=field,
            )

        # with no commissions, entering or exiting a future position does not
        # affect your cash
        capital_base_series = pd.Series(
            self.SIM_PARAMS_CAPITAL_BASE,
            index=self.closes,
        )

        # leverage is gross market exposure / current notional capital
        # gross market exposure is
        # sum(long_exposure) + sum(abs(short_exposure))
        # current notional capital is the current portfolio value
        expected_max_leverage = (
            self.future_constant_exposure / capital_base_series
        )
        assert_equal(
            perf['max_leverage'],
            expected_max_leverage,
            check_names=False,
        )

        # with no commissions, slippage, or returns our portfolio value stays
        # constant (at the capital base)
        for field in 'starting_cash', 'ending_cash', 'portfolio_value':
            assert_equal(
                perf[field],
                capital_base_series,
                check_names=False,
                msg=field,
            )

        # with no commissions, entering or exiting a future position does not
        # affect your cash; thus no capital gets used
        expected_capital_used = pd.Series(0.0, index=self.closes)

        assert_equal(
            perf['capital_used'],
            expected_capital_used,
            check_names=False,
        )

        # we hold one contract so our positions exposure is that one
        # contract's price
        expected_position_exposure = pd.Series(
            expected_exposure,
            index=self.closes,
        )
        assert_equal(
            perf['ending_exposure'],
            expected_position_exposure,
            check_names=False,
            check_dtype=False,
        )

        # we don't start with any positions; the first day has no starting
        # exposure
        expected_position_exposure[0] = 0
        assert_equal(
            perf['starting_exposure'],
            expected_position_exposure,
            check_names=False,
        )

        assert_equal(
            perf['trading_days'],
            pd.Series(
                np.arange(len(self.closes)) + 1,
                index=self.closes,
            ),
            check_names=False,
        )

        all_nan = pd.Series(np.nan, index=self.closes)
        all_nan_fields = (
            'alpha',
            'beta',
            'sortino',
        )
        for field in all_nan_fields:
            assert_equal(
                perf[field],
                all_nan,
                check_names=False,
                msg=field,
            )

        orders = perf['orders']

        # we only order on the first day
        expected_orders = [
            [{
                'amount': contracts,
                'commission': 0.0,
                'created': T('2014-01-06 14:31'),
                'dt': T('2014-01-06 14:32'),
                'filled': contracts,
                'id': wildcard,
                'limit': None,
                'limit_reached': False,
                'reason': None,
                'sid': self.future,
                'status': 1,
                'stop': None,
                'stop_reached': False
            }],
        ] + [[]] * (len(self.closes) - 1)

        assert_equal(
            orders.tolist(),
            expected_orders,
            check_names=False,
        )
        assert_equal(
            orders.index,
            self.closes,
            check_names=False,
        )

        transactions = perf['transactions']

        # since we only order on the first day, we should only transact on the
        # first day
        expected_transactions = [
            [{
                'amount': contracts,
                'commission': None,
                'dt': T('2014-01-06 14:32'),
                'order_id': wildcard,
                'price': 1.0,
                'sid': self.future,
            }],
        ] + [[]] * (len(self.closes) - 1)

        assert_equal(
            transactions.tolist(),
            expected_transactions,
            check_names=False,
        )
        assert_equal(
            transactions.index,
            self.closes,
            check_names=False,
        )

        if not check_portfolio_during_simulation:
            return

        portfolio_snapshots = pd.DataFrame.from_dict(
            portfolio_snapshots,
            orient='index',
        )

        expected_starting_cash = pd.Series(
            self.SIM_PARAMS_CAPITAL_BASE,
            index=self.trading_minutes,
        )
        assert_equal(
            portfolio_snapshots['starting_cash'],
            expected_starting_cash,
            check_names=False,
        )

        zero_minutes = pd.Series(0.0, index=self.trading_minutes)
        for field in 'pnl', 'returns', 'cash_flow':
            assert_equal(
                portfolio_snapshots[field],
                zero_minutes,
                check_names=False,
                msg=field,
            )

        reindex_columns = sorted(
            set(portfolio_snapshots.columns) - {
                'starting_cash',
                'cash_flow',
                'pnl',
                'returns',
                'positions',
            },
        )
        minute_reindex = perf.rename(
            columns={
                'capital_used': 'cash_flow',
                'ending_cash': 'cash',
                'ending_exposure': 'positions_exposure',
                'ending_value': 'positions_value',
            },
        )[reindex_columns].reindex(
            self.trading_minutes,
            method='bfill',
        )

        first_minute = self.trading_minutes[0]
        # the first minute should have the default values because we haven't
        # done anything yet
        minute_reindex.loc[first_minute, 'cash'] = (
            self.SIM_PARAMS_CAPITAL_BASE
        )
        minute_reindex.loc[
            first_minute,
            ['positions_exposure', 'positions_value'],
        ] = 0

        assert_equal(
            portfolio_snapshots[reindex_columns],
            minute_reindex,
            check_names=False,
        )


class TestFixedReturns(WithMakeAlgo, WithWerror, ZiplineTestCase):
    EQUITY_DAILY_BAR_SOURCE_FROM_MINUTE = True
    FUTURE_DAILY_BAR_SOURCE_FROM_MINUTE = True

    START_DATE = T('2014-01-06')
    END_DATE = T('2014-01-10')

    # note: class attributes after this do not configure fixtures, they are
    # just used in this test suite

    # we use a contract multiplier to make sure we are correctly calculating
    # exposure as price * multiplier
    future_contract_multiplier = 2

    asset_start_price = 100
    asset_daily_returns = np.array([
        +0.02,  # up 2%
        -0.02,  # down 2%, this should give us less value that we started with
        +0.00,  # no returns
        +0.04,  # up 4%
    ])
    asset_daily_close = prices_generating_returns(
        asset_daily_returns,
        asset_start_price,
    )
    asset_daily_volume = 100000

    @classmethod
    def init_class_fixtures(cls):
        super(TestFixedReturns, cls).init_class_fixtures()

        cls.equity = cls.asset_finder.retrieve_asset(
            cls.asset_finder.equities_sids[0],
        )
        cls.future = cls.asset_finder.retrieve_asset(
            cls.asset_finder.futures_sids[0],
        )

        cls.equity_minutes = pd.Index(
            cls.trading_calendars[Equity].minutes_for_sessions_in_range(
                cls.START_DATE,
                cls.END_DATE,
            ),
        )
        cls.equity_closes = pd.Index(
            cls.trading_calendars[Equity].session_closes_in_range(
                cls.START_DATE,
                cls.END_DATE,
            ),
        )
        cls.equity_closes.name = None

        futures_cal = cls.trading_calendars[Future]
        cls.future_minutes = pd.Index(
            futures_cal.execution_minutes_for_sessions_in_range(
                cls.START_DATE,
                cls.END_DATE,
            ),

        )
        cls.future_closes = pd.Index(
            futures_cal.execution_time_from_close(
                futures_cal.session_closes_in_range(
                    cls.START_DATE,
                    cls.END_DATE,
                ),
            ),
        )
        cls.future_closes.name = None

        cls.future_opens = pd.Index(
            futures_cal.execution_time_from_open(
                futures_cal.session_opens_in_range(
                    cls.START_DATE,
                    cls.END_DATE,
                ),
            ),
        )
        cls.future_opens.name = None

    def init_instance_fixtures(self):
        super(TestFixedReturns, self).init_instance_fixtures()

        if self.DATA_PORTAL_FIRST_TRADING_DAY is None:
            if self.DATA_PORTAL_USE_MINUTE_DATA:
                self.DATA_PORTAL_FIRST_TRADING_DAY = (
                    self.bcolz_future_minute_bar_reader.first_trading_day
                )
            elif self.DATA_PORTAL_USE_DAILY_DATA:
                self.DATA_PORTAL_FIRST_TRADING_DAY = (
                    self.bcolz_future_daily_bar_reader.first_trading_day
                )

        self.futures_data_portal = DataPortal(
            self.asset_finder,
            self.trading_calendars[Future],
            first_trading_day=self.DATA_PORTAL_FIRST_TRADING_DAY,
            equity_daily_reader=(
                self.bcolz_equity_daily_bar_reader
                if self.DATA_PORTAL_USE_DAILY_DATA else
                None
            ),
            equity_minute_reader=(
                self.bcolz_equity_minute_bar_reader
                if self.DATA_PORTAL_USE_MINUTE_DATA else
                None
            ),
            adjustment_reader=(
                self.adjustment_reader
                if self.DATA_PORTAL_USE_ADJUSTMENTS else
                None
            ),
            future_minute_reader=(
                self.bcolz_future_minute_bar_reader
                if self.DATA_PORTAL_USE_MINUTE_DATA else
                None
            ),
            future_daily_reader=(
                MinuteResampleSessionBarReader(
                    self.bcolz_future_minute_bar_reader.trading_calendar,
                    self.bcolz_future_minute_bar_reader)
                if self.DATA_PORTAL_USE_MINUTE_DATA else None
            ),
            last_available_session=self.DATA_PORTAL_LAST_AVAILABLE_SESSION,
            last_available_minute=self.DATA_PORTAL_LAST_AVAILABLE_MINUTE,
            minute_history_prefetch_length=(
                self.DATA_PORTAL_MINUTE_HISTORY_PREFETCH
            ),
            daily_history_prefetch_length=(
                self.DATA_PORTAL_DAILY_HISTORY_PREFETCH
            ),
        )

    @classmethod
    def make_futures_info(cls):
        return make_commodity_future_info(
            first_sid=ord('Z'),
            root_symbols=['Z'],
            years=[cls.START_DATE.year],
            multiplier=cls.future_contract_multiplier,
        )

    @classmethod
    def _make_minute_bar_data(cls, calendar, sids):
        daily_close = cls.asset_daily_close
        daily_open = daily_close - 1
        daily_high = daily_close + 1
        daily_low = daily_close - 2
        random_state = np.random.RandomState(seed=1337)

        data = pd.concat(
            [
                simulate_minutes_for_day(
                    o,
                    h,
                    l,
                    c,
                    cls.asset_daily_volume,
                    trading_minutes=len(calendar.minutes_for_session(session)),
                    random_state=random_state,
                )
                for o, h, l, c, session in zip(
                    daily_open,
                    daily_high,
                    daily_low,
                    daily_close,
                    calendar.sessions_in_range(cls.START_DATE, cls.END_DATE),
                )
            ],
            ignore_index=True,
        )
        data.index = calendar.minutes_for_sessions_in_range(
            cls.START_DATE,
            cls.END_DATE,
        )

        for sid in sids:
            yield sid, data

    @classmethod
    def make_equity_minute_bar_data(cls):
        return cls._make_minute_bar_data(
            cls.trading_calendars[Equity],
            cls.asset_finder.equities_sids,
        )

    @classmethod
    def make_future_minute_bar_data(cls):
        return cls._make_minute_bar_data(
            cls.trading_calendars[Future],
            cls.asset_finder.futures_sids,
        )

    @parameter_space(
        direction=['long', 'short'],
        # checking the portfolio forces a sync; we want to ensure that the
        # perf packets are correct even without explicitly requesting the
        # portfolio every day. we also want to test that ``context.portfolio``
        # produces the expected values when queried mid-simulation
        check_portfolio_during_simulation=[True, False],
    )
    def test_equity_single_position(self,
                                    direction,
                                    check_portfolio_during_simulation):
        if direction not in ('long', 'short'):
            raise ValueError(
                'direction must be either long or short, got: %r' % direction,
            )

        shares = 1 if direction == 'long' else -1

        expected_fill_price = self.data_portal.get_scalar_asset_spot_value(
            self.equity,
            'close',
            # we expect to kill in the second bar of the first day
            self.equity_minutes[1],
            'minute',
        )

        def initialize(context):
            api.set_benchmark(self.equity)

            api.set_slippage(api.slippage.NoSlippage())
            api.set_commission(api.commission.NoCommission())

            context.first_bar = True

        if check_portfolio_during_simulation:
            portfolio_snapshots = {}

            def check_portfolio(data, context, first_bar):
                portfolio = context.portfolio
                portfolio_snapshots[api.get_datetime()] = portfolio_snapshot(
                    portfolio,
                )

                positions = portfolio.positions
                if first_bar:
                    assert_equal(positions, {})
                    return

                assert_equal(positions.keys(), [self.equity])
                position = positions[self.equity]
                assert_equal(position.last_sale_date, api.get_datetime())
                assert_equal(position.amount, shares)
                assert_equal(
                    position.last_sale_price,
                    data.current(self.equity, 'close'),
                )
                assert_equal(position.asset, self.equity)
                assert_equal(
                    position.cost_basis,
                    expected_fill_price,
                )
        else:
            def check_portfolio(data, context, first_bar):
                pass

        def handle_data(context, data):
            first_bar = context.first_bar
            if first_bar:
                api.order(self.equity, shares)
                context.first_bar = False

            # take the snapshot after the order; ordering does not affect
            # the portfolio on the bar of the order, only the following bars
            check_portfolio(data, context, first_bar)

        perf = self.run_algorithm(
            initialize=initialize,
            handle_data=handle_data,
        )

        zeros = pd.Series(0.0, index=self.equity_closes)
        all_zero_fields = [
            'excess_return',
            'treasury_period_return',
        ]
        if direction == 'long':
            all_zero_fields.extend((
                'short_value',
                'shorts_count',
            ))
        else:
            all_zero_fields.extend((
                'long_value',
                'longs_count',
            ))
        for field in all_zero_fields:
            assert_equal(
                perf[field],
                zeros,
                check_names=False,
                check_dtype=False,
                msg=field,
            )

        ones = pd.Series(1, index=self.equity_closes)
        if direction == 'long':
            count_field = 'longs_count'
        else:
            count_field = 'shorts_count'

        assert_equal(
            perf[count_field],
            ones,
            check_names=False,
            msg=field,
        )

        if direction == 'long':
            expected_exposure = pd.Series(
                self.asset_daily_close,
                index=self.equity_closes,
            )
            exposure_fields = 'long_value', 'long_exposure'
        else:
            expected_exposure = pd.Series(
                -self.asset_daily_close,
                index=self.equity_closes,
            )
            exposure_fields = 'short_value', 'short_exposure'

        for field in exposure_fields:
            assert_equal(
                perf[field],
                expected_exposure,
                check_names=False,
                msg=field,
            )

        if direction == 'long':
            delta = self.asset_daily_close - expected_fill_price
        else:
            delta = -self.asset_daily_close + expected_fill_price
        expected_portfolio_value = pd.Series(
            self.SIM_PARAMS_CAPITAL_BASE + delta,
            index=self.equity_closes,
        )

        assert_equal(
            perf['portfolio_value'],
            expected_portfolio_value,
            check_names=False,
        )

        capital_base_series = pd.Series(
            self.SIM_PARAMS_CAPITAL_BASE,
            index=self.equity_closes,
        )

        # leverage is gross market exposure / current notional capital
        # gross market exposure is
        # sum(long_exposure) + sum(abs(short_exposure))
        # current notional capital is the current portfolio value
        expected_max_leverage = np.maximum.accumulate(
            expected_exposure.abs() / expected_portfolio_value,
        )
        assert_equal(
            perf['max_leverage'],
            expected_max_leverage,
            check_names=False,
        )

        expected_cash = capital_base_series.copy()
        if direction == 'long':
            # we purchased one share on the first day
            cash_modifier = -expected_fill_price
        else:
            # we sold one share on the first day
            cash_modifier = +expected_fill_price

        expected_cash[1:] += cash_modifier

        assert_equal(
            perf['starting_cash'],
            expected_cash,
            check_names=False,
        )

        expected_cash[0] += cash_modifier
        assert_equal(
            perf['ending_cash'],
            expected_cash,
            check_names=False,
        )

        # we purchased one share on the first day
        expected_capital_used = pd.Series(0.0, index=self.equity_closes)
        expected_capital_used[0] += cash_modifier

        assert_equal(
            perf['capital_used'],
            expected_capital_used,
            check_names=False,
        )

        for field in 'ending_value', 'ending_exposure':
            # for equities, position value and position exposure are the same
            assert_equal(
                perf[field],
                expected_exposure,
                check_names=False,
                msg=field,
            )

        # we don't start with any positions; the first day has no starting
        # exposure
        expected_starting_exposure = expected_exposure.shift(1)
        expected_starting_exposure[0] = 0.0
        for field in 'starting_value', 'starting_exposure':
            # for equities, position value and position exposure are the same
            assert_equal(
                perf[field],
                expected_starting_exposure,
                check_names=False,
                msg=field,
            )

        assert_equal(
            perf['trading_days'],
            pd.Series(
                np.arange(len(self.equity_closes)) + 1,
                index=self.equity_closes,
            ),
            check_names=False,
        )

        orders = perf['orders']

        expected_single_order = {
            'amount': shares,
            'commission': 0.0,
            'created': T('2014-01-06 14:31'),
            'dt': T('2014-01-06 14:32'),
            'filled': shares,
            'id': wildcard,
            'limit': None,
            'limit_reached': False,
            'reason': None,
            'sid': self.equity,
            'status': 1,
            'stop': None,
            'stop_reached': False
        }

        # we only order on the first day
        expected_orders = (
            [[expected_single_order]] +
            [[]] * (len(self.equity_closes) - 1)
        )

        assert_equal(
            orders.tolist(),
            expected_orders,
            check_names=False,
        )
        assert_equal(
            orders.index,
            self.equity_closes,
            check_names=False,
        )

        transactions = perf['transactions']

        expected_single_transaction = {
            'amount': shares,
            'commission': None,
            'dt': T('2014-01-06 14:32'),
            'order_id': wildcard,
            'price': self.data_portal.get_scalar_asset_spot_value(
                self.equity,
                'close',
                T('2014-01-06 14:32'),
                'minute',
            ),
            'sid': self.equity,
        }

        # since we only order on the first day, we should only transact on the
        # first day
        expected_transactions = (
            [[expected_single_transaction]] +
            [[]] * (len(self.equity_closes) - 1)
        )

        assert_equal(
            transactions.tolist(),
            expected_transactions,
        )
        assert_equal(
            transactions.index,
            self.equity_closes,
            check_names=False,
        )

        if not check_portfolio_during_simulation:
            return

        portfolio_snapshots = pd.DataFrame.from_dict(
            portfolio_snapshots,
            orient='index',
        )

        expected_starting_cash = pd.Series(
            self.SIM_PARAMS_CAPITAL_BASE,
            index=self.equity_minutes,
        )
        assert_equal(
            portfolio_snapshots['starting_cash'],
            expected_starting_cash,
            check_names=False,
        )

        expected_portfolio_capital_used = pd.Series(
            cash_modifier,
            index=self.equity_minutes,
        )
        expected_portfolio_capital_used[0] = 0.0
        expected_capital_used[0] = 0
        assert_equal(
            portfolio_snapshots['cash_flow'],
            expected_portfolio_capital_used,
            check_names=False,
        )

        minute_prices = self.data_portal.get_history_window(
            [self.equity],
            self.equity_minutes[-1],
            len(self.equity_minutes),
            '1m',
            'close',
            'minute',
        )[self.equity]

        expected_pnl = minute_prices.diff()
        # we don't enter the position until the second minute
        expected_pnl.iloc[:2] = 0.0
        expected_pnl = expected_pnl.cumsum()

        if direction == 'short':
            expected_pnl = -expected_pnl

        assert_equal(
            portfolio_snapshots['pnl'],
            expected_pnl,
            check_names=False,
        )

        expected_portfolio_value = self.SIM_PARAMS_CAPITAL_BASE + expected_pnl
        assert_equal(
            portfolio_snapshots['portfolio_value'],
            expected_portfolio_value,
            check_names=False,
        )

        expected_returns = (
            portfolio_snapshots['portfolio_value'] /
            self.SIM_PARAMS_CAPITAL_BASE
        ) - 1
        assert_equal(
            portfolio_snapshots['returns'],
            expected_returns,
            check_names=False,
        )

        expected_exposure = minute_prices.copy()
        # we don't enter the position until the second minute
        expected_exposure.iloc[0] = 0.0
        if direction == 'short':
            expected_exposure = -expected_exposure

        for field in 'positions_value', 'positions_exposure':
            assert_equal(
                portfolio_snapshots[field],
                expected_exposure,
                check_names=False,
                msg=field,
            )

    @parameter_space(
        direction=['long', 'short'],
        # checking the portfolio forces a sync; we want to ensure that the
        # perf packets are correct even without explicitly requesting the
        # portfolio every day. we also want to test that ``context.portfolio``
        # produces the expected values when queried mid-simulation
        check_portfolio_during_simulation=[True, False],
    )
    def test_future_single_position(self,
                                    direction,
                                    check_portfolio_during_simulation):
        if direction not in ('long', 'short'):
            raise ValueError(
                'direction must be either long or short, got: %r' % direction,
            )

        contracts = 1 if direction == 'long' else -1

        expected_fill_price = (
            self.futures_data_portal.get_scalar_asset_spot_value(
                self.future,
                'close',
                # we expect to kill in the second bar of the first day
                self.future_minutes[1],
                'minute',
            )
        )

        future_execution_close_prices = pd.Series(
            [
                self.futures_data_portal.get_scalar_asset_spot_value(
                    self.future,
                    'close',
                    execution_close_minute,
                    'minute',
                )
                for execution_close_minute in self.future_closes
            ],
            index=self.future_closes,
        )

        future_execution_open_prices = pd.Series(
            [
                self.futures_data_portal.get_scalar_asset_spot_value(
                    self.future,
                    'close',
                    execution_open_minute,
                    'minute',
                )
                for execution_open_minute in self.future_opens
            ],
            index=self.future_opens,
        )

        def initialize(context):
            api.set_benchmark(self.equity)

            api.set_slippage(us_futures=api.slippage.NoSlippage())
            api.set_commission(us_futures=api.commission.NoCommission())

            context.first_bar = True

        if check_portfolio_during_simulation:
            portfolio_snapshots = {}

            def check_portfolio(data, context, first_bar):
                portfolio = context.portfolio
                portfolio_snapshots[api.get_datetime()] = portfolio_snapshot(
                    portfolio,
                )

                positions = portfolio.positions
                if first_bar:
                    assert_equal(positions, {})
                    return

                assert_equal(positions.keys(), [self.future])
                position = positions[self.future]
                assert_equal(position.last_sale_date, api.get_datetime())
                assert_equal(position.amount, contracts)
                assert_equal(
                    position.last_sale_price,
                    data.current(self.future, 'close'),
                )
                assert_equal(position.asset, self.future)
                assert_equal(
                    position.cost_basis,
                    expected_fill_price,
                )
        else:
            def check_portfolio(data, context, first_bar):
                pass

        def handle_data(context, data):
            first_bar = context.first_bar
            if first_bar:
                api.order(self.future, contracts)
                context.first_bar = False

            # take the snapshot after the order; ordering does not affect
            # the portfolio on the bar of the order, only the following bars
            check_portfolio(data, context, first_bar)

        perf = self.run_algorithm(
            initialize=initialize,
            handle_data=handle_data,
            trading_calendar=self.trading_calendars[Future],
            data_portal=self.futures_data_portal,
        )

        zeros = pd.Series(0.0, index=self.future_closes)
        all_zero_fields = [
            'excess_return',
            'treasury_period_return',
            'short_value',
            'long_value',
            'starting_value',
            'ending_value',
        ]
        if direction == 'long':
            all_zero_fields.append('shorts_count')
        else:
            all_zero_fields.append('longs_count')

        for field in all_zero_fields:
            assert_equal(
                perf[field],
                zeros,
                check_names=False,
                check_dtype=False,
                msg=field,
            )

        ones = pd.Series(1, index=self.future_closes)
        if direction == 'long':
            count_field = 'longs_count'
        else:
            count_field = 'shorts_count'

        assert_equal(
            perf[count_field],
            ones,
            check_names=False,
            msg=field,
        )

        expected_exposure = pd.Series(
            future_execution_close_prices * self.future_contract_multiplier,
            index=self.future_closes,
        )
        exposure_field = 'long_exposure'
        if direction == 'short':
            exposure_field = 'short_exposure'
            expected_exposure = -expected_exposure

        assert_equal(
            perf[exposure_field],
            expected_exposure,
            check_names=False,
            msg=exposure_field,
            check_dtype=False,
        )

        if direction == 'long':
            delta = future_execution_close_prices - expected_fill_price
        else:
            delta = -future_execution_close_prices + expected_fill_price

        expected_portfolio_value = pd.Series(
            (
                self.SIM_PARAMS_CAPITAL_BASE +
                self.future_contract_multiplier * delta
            ),
            index=self.future_closes,
        )

        assert_equal(
            perf['portfolio_value'],
            expected_portfolio_value,
            check_names=False,
        )

        # leverage is gross market exposure / current notional capital
        # gross market exposure is
        # sum(long_exposure) + sum(abs(short_exposure))
        # current notional capital is the current portfolio value
        expected_max_leverage = np.maximum.accumulate(
            expected_exposure.abs() / expected_portfolio_value,
        )
        assert_equal(
            perf['max_leverage'],
            expected_max_leverage,
            check_names=False,
        )

        expected_cashflow = pd.Series(
            (
                self.future_contract_multiplier *
                (future_execution_close_prices - expected_fill_price)
            ),
            index=self.future_closes,
        )
        if direction == 'short':
            expected_cashflow = -expected_cashflow

        expected_cash = self.SIM_PARAMS_CAPITAL_BASE + expected_cashflow
        assert_equal(
            perf['ending_cash'],
            expected_cash,
            check_names=False,
        )

        delta = (
            self.future_contract_multiplier *
            (future_execution_open_prices - expected_fill_price)
        )
        if direction == 'short':
            delta = -delta

        # NOTE: this seems really wrong to me: we should report the cash
        # as of the start of the session, not the cash at the end of the
        # previous session
        expected_starting_cash = expected_cash.shift(1)
        expected_starting_cash.iloc[0] = self.SIM_PARAMS_CAPITAL_BASE

        assert_equal(
            perf['starting_cash'],
            expected_starting_cash,
            check_names=False,
        )

        assert_equal(
            perf['capital_used'],
            perf['ending_cash'] - perf['starting_cash'],
            check_names=False,
        )

        # for equities, position value and position exposure are the same
        assert_equal(
            perf['ending_exposure'],
            expected_exposure,
            check_names=False,
            msg=field,
        )

        # we don't start with any positions; the first day has no starting
        # exposure
        expected_starting_exposure = expected_exposure.shift(1)
        expected_starting_exposure[0] = 0.0
        assert_equal(
            perf['starting_exposure'],
            expected_starting_exposure,
            check_names=False,
            msg=field,
        )

        assert_equal(
            perf['trading_days'],
            pd.Series(
                np.arange(len(self.future_closes)) + 1,
                index=self.future_closes,
            ),
            check_names=False,
        )

        orders = perf['orders']

        expected_single_order = {
            'amount': contracts,
            'commission': 0.0,
            'created': self.future_minutes[0],
            'dt': self.future_minutes[1],
            'filled': contracts,
            'id': wildcard,
            'limit': None,
            'limit_reached': False,
            'reason': None,
            'sid': self.future,
            'status': 1,
            'stop': None,
            'stop_reached': False
        }

        # we only order on the first day
        expected_orders = (
            [[expected_single_order]] +
            [[]] * (len(self.future_closes) - 1)
        )

        assert_equal(
            orders.tolist(),
            expected_orders,
            check_names=False,
        )
        assert_equal(
            orders.index,
            self.future_closes,
            check_names=False,
        )

        transactions = perf['transactions']

        dt = self.future_minutes[1]
        expected_single_transaction = {
            'amount': contracts,
            'commission': None,
            'dt': dt,
            'order_id': wildcard,
            'price': self.futures_data_portal.get_scalar_asset_spot_value(
                self.future,
                'close',
                dt,
                'minute',
            ),
            'sid': self.future,
        }

        # since we only order on the first day, we should only transact on the
        # first day
        expected_transactions = (
            [[expected_single_transaction]] +
            [[]] * (len(self.future_closes) - 1)
        )

        assert_equal(
            transactions.tolist(),
            expected_transactions,
        )
        assert_equal(
            transactions.index,
            self.future_closes,
            check_names=False,
        )

        if not check_portfolio_during_simulation:
            return

        portfolio_snapshots = pd.DataFrame.from_dict(
            portfolio_snapshots,
            orient='index',
        )

        expected_starting_cash = pd.Series(
            self.SIM_PARAMS_CAPITAL_BASE,
            index=self.future_minutes,
        )
        assert_equal(
            portfolio_snapshots['starting_cash'],
            expected_starting_cash,
            check_names=False,
        )

        execution_minute_prices = pd.Series(
            [
                self.futures_data_portal.get_scalar_asset_spot_value(
                    self.future,
                    'close',
                    minute,
                    'minute',
                )
                for minute in self.future_minutes
            ],
            index=self.future_minutes,
        )
        expected_portfolio_capital_used = (
            self.future_contract_multiplier *
            (execution_minute_prices - expected_fill_price)
        )
        if direction == 'short':
            expected_portfolio_capital_used = -expected_portfolio_capital_used

        # we don't execute until the second minute; then cash adjustments begin
        expected_portfolio_capital_used.iloc[:2] = 0.0
        assert_equal(
            portfolio_snapshots['cash_flow'],
            expected_portfolio_capital_used,
            check_names=False,
        )

        all_minutes = (
            self.trading_calendars[Future].minutes_for_sessions_in_range(
                self.START_DATE,
                self.END_DATE,
            )
        )
        valid_minutes = all_minutes[
            all_minutes.slice_indexer(
                self.future_minutes[1],
                self.future_minutes[-1],
            )
        ]
        minute_prices = self.futures_data_portal.get_history_window(
            [self.future],
            self.future_minutes[-1],
            len(valid_minutes) + 1,
            '1m',
            'close',
            'minute',
        )[self.future]

        raw_pnl = minute_prices.diff()
        # we don't execute until the second minute; then cash adjustments begin
        raw_pnl.iloc[:2] = 0.0
        raw_pnl = raw_pnl.cumsum() * self.future_contract_multiplier

        expected_pnl = raw_pnl.reindex(self.future_minutes)
        if direction == 'short':
            expected_pnl = -expected_pnl

        assert_equal(
            portfolio_snapshots['pnl'],
            expected_pnl,
            check_names=False,
        )

        expected_portfolio_value = self.SIM_PARAMS_CAPITAL_BASE + expected_pnl
        assert_equal(
            portfolio_snapshots['portfolio_value'],
            expected_portfolio_value,
            check_names=False,
        )

        expected_returns = (
            portfolio_snapshots['portfolio_value'] /
            self.SIM_PARAMS_CAPITAL_BASE
        ) - 1
        assert_equal(
            portfolio_snapshots['returns'],
            expected_returns,
            check_names=False,
        )

        expected_exposure = (
            minute_prices.copy() * self.future_contract_multiplier
        ).reindex(self.future_minutes)
        # we don't enter the position until the second minute
        expected_exposure.iloc[0] = 0.0
        if direction == 'short':
            expected_exposure = -expected_exposure

        assert_equal(
            portfolio_snapshots['positions_exposure'],
            expected_exposure,
            check_names=False,
        )

        expected_value = pd.Series(0.0, index=self.future_minutes)
        assert_equal(
            portfolio_snapshots['positions_value'],
            expected_value,
            check_names=False,
            check_dtype=False,
        )
