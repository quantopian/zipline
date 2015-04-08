import talib
from zipline.api import record, order_target, history, add_history
import dateutil
import logging
from zipline.utils.factory import load_from_yahoo
from zipline.finance.slippage import FixedSlippage
from zipline.algorithm import TradingAlgorithm
from zipline.finance import commission

logging.basicConfig(level=logging.DEBUG)


# initialize algorithm
def initialize(context):
    logging.debug('enter initialize')
    context.set_slippage(FixedSlippage())
    context.set_commission(commission.PerTrade(cost=5))

    context.LOW_RSI = initialize.low_RSI
    context.HIGH_RSI = initialize.high_RSI
    context.rsi_window = initialize.rsi_window
    add_history(context.rsi_window, '1d', 'price')
    context.i = 0
    context.invested = False

# default parameters for algorithm
initialize.rsi_window = 15
initialize.low_RSI = 30
initialize.high_RSI = 70


# Will be called on every trade event for the securities you specify.
def handle_data(context, data):
    logging.debug('enter handle_data')
    context.i += 1
    if context.i < context.rsi_window:
        return

    # get the last RSI value
    prices = history(context.rsi_window, '1d', 'price')
    sec_rsi = talib.RSI(
        prices[context.security].values,
        timeperiod=context.rsi_window - 1)

    # buy and sell flags
    buy = False
    sell = False

    if sec_rsi[-1] < context.LOW_RSI and not context.invested:
        # RSI under 30 indicates oversold, time to buy
        order_target(context.security, 1000)
        logging.debug('Buying {}'.format(context.security))
        context.invested = True
        buy = True

    elif sec_rsi[-1] > context.HIGH_RSI and context.invested:
        # RSI over 70 indicates overbought, sell everything
        order_target(context.security, 0)
        logging.debug('Selling {}'.format(context.security))
        context.invested = False
        sell = True

    # record data for each time increment
    record(secRSI=sec_rsi[-1],
           price=data[context.security].price,
           buy=buy,
           sell=sell)
    logging.info(context.portfolio.cash)


def run_algorithm(
        security='AAPL',
        start_date='20100101',
        end_date='20150101',
        initial_cash=100000,
        rsi_window=15,
        low_RSI=30,
        high_RSI=70):
    logging.debug('run_algorithm begin')
    # dates
    start = dateutil.parser.parse(start_date)
    end = dateutil.parser.parse(end_date)

    # get data from yahoo
    data = load_from_yahoo(stocks=[security], indexes={}, start=start, end=end)
    logging.debug('done loading from yahoo. {} {} {}'.format(
        security, start_date, end_date))

    # create and run algorithm
    algo = TradingAlgorithm(
        initialize=initialize,
        handle_data=handle_data,
        capital_base=initial_cash)
    algo.security = security
    initialize.low_RSI = low_RSI
    initialize.high_RSI = high_RSI
    initialize.rsi_window = rsi_window
    logging.debug('starting to run algo...')
    results = algo.run(data).dropna()
    logging.debug('done running algo')
    return results


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # run algorithm and get results
    results = run_algorithm(
        security='AAPL',
        start_date='20100101',
        end_date='20150101',
        initial_cash=100000,
        rsi_window=15,
        low_RSI=30,
        high_RSI=70)

    # get s&p500 and nasdaq indexes
    index_data = load_from_yahoo(
        stocks=['^gspc', '^ixic'],
        indexes={},
        start=results.index[0],
        end=results.index[-1])

    # portfolio value, stock holdings and S&P 500 index
    fig = plt.figure(figsize=(12, 6))
    ax11 = fig.add_subplot(311)
    ax12, ax13 = ax11.twinx(), ax11.twinx()
    ax13.spines['right'].set_position(('axes', 1.07))
    ax11.set_ylabel('portfolio value', color='blue')
    ax12.set_ylabel('holdings', color='green')
    ax13.set_ylabel('S&P 500', color='red')

    # portfolio value
    ax11.plot(results.index, results.portfolio_value, color='blue')

    # holdings (number of stocks owned)
    holdings = [0 if t == [] else t[0]['amount'] for t in results.positions]
    ax12.plot(results.index, holdings, color='green')
    ax12.set_ylim([min(holdings) - 30, max(holdings) + 30])

    # index
    ax13.plot(index_data.index, index_data['^gspc'], color='red')

    # algo visualization
    ax21 = fig.add_subplot(312)
    ax21.set_ylabel('stock price', color='blue')
    ax22 = ax21.twinx()
    ax22.set_ylabel('rsi', color='red')

    # stock
    ax21.plot(results.index, results.price, color='blue')

    # add sell and buy flags on top of stock price
    ax21.plot(
        results.ix[results.buy].index,
        results.price[results.buy],
        '^',
        markersize=10,
        color='green')
    ax21.plot(
        results.ix[results.sell].index,
        results.price[results.sell],
        'v',
        markersize=10,
        color='red')

    # rsi value
    ax22.plot(results.index, results.secRSI, color='red')
    # add lines to show under- and over value indicator
    ax22.plot([results.index[0], results.index[-1]], [30, 30], 'k-')
    ax22.plot([results.index[0], results.index[-1]], [70, 70], 'k-')

    # portfolio value, stock value and index in percentage
    ax31 = fig.add_subplot(313)
    ax32, ax33 = ax31.twinx(), ax31.twinx()  # share x for other plots
    ax31.set_ylabel('algo %', color='blue')
    ax32.set_ylabel('snp index %', color='green')
    ax33.set_ylabel('stock %', color='red')
    ax33.spines['right'].set_position(('axes', 1.07))

    # portfolio value
    ax31.plot(
        results.index,
        results.portfolio_value / results.portfolio_value[0] * 100 - 100,
        color='blue')

    # index
    ax32.plot(
        index_data.index,
        index_data['^gspc'] / index_data['^gspc'][0] * 100 - 100,
        color='green')

    # stock value
    ax33.plot(
        results.index,
        results.price /
        results.price[0] * 100 - 100,
        color='red')

    plt.show()
