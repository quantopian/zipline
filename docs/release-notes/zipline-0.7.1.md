# Zipline 0.7.1 Release Notes

## Bug Fixes (BUG)

* Fix a bug where the reported returns could sharply dip for random periods of
  time. [PR378](https://github.com/quantopian/zipline/pull/378)

## Enhancements (ENH)

* Account object: Adds an account object to conext to track information about
  the trading account. [PR396](https://github.com/quantopian/zipline/pull/396)

  > Example:

  > ```
  > context.account.settled_cash
  > ```

  > Returns the settled cash value that is stored on the account object. This
  > value is updated accordingly as the algorithm is run.

* HistoryContainer can now grow
  dynamically. [PR412](https://github.com/quantopian/zipline/pull/412)

  > Calls to `history` will now be able to increase the size or change the shape
  > of the history container to be able to service the call. `add_history` now
  > acts as a preformance hint to pre-allocate sufficient space in the
  > container. This change is backwards compatible with `history`, all existing
  > algorithms should continue to work as intended.

* Simple transforms ported from quantopian and use history.
  [PR429](https://github.com/quantopian/zipline/pull/429)

  > SIDData now has methods for:

  > - `stddev`
  > - `mavg`
  > - `vwap`
  > - `returns`

  > These methods, except for `returns`, accept a number of days. If you are
  > running with minute data, then this will calculate the number of minutes in
  > those days, accounting for early closes and the current time and apply the
  > transform over the set of minutes. `returns` takes no parameters and will
  > return the daily returns of the given security.

  > Example:
  > ```
  > # The standard deviation of the price in the last 3 days.
  > data[security].stdev(3)
  > ```

* New fields in Performance Period
[PR464](https://github.com/quantopian/zipline/pull/464)

  > Performance Period has new fields accessible in return value of to_dict:

  > - gross leverage
  > - net leverage
  > - short exposure
  > - long exposure
  > - shorts count
  > - longs count

* Allow order_percent to work with various market values (by Jeremiah Lowin)
[PR477](https://github.com/quantopian/zipline/pull/477)

    > Currently, `order_percent()` and `order_target_percent()` both operate as a percentage of `self.portfolio.portfolio_value`. This PR lets them operate as percentages of other important MVs.

    > Also adds `context.get_market_value()`, which enables this functionality.

    > For example:
    > ```python
    > # this is how it works today (and this still works)
    > # put 50% of my portfolio in AAPL
    > order_percent('AAPL', 0.5)
    > # note that if this were a fully invested portfolio, it would become 150% levered.

    > # take half of my available cash and buy AAPL
    > order_percent('AAPL', 0.5, percent_of='cash')

    > # rebalance my short position, as a percentage of my current short book
    > order_target_percent('MSFT', 0.1, percent_of='shorts')

    > # rebalance within a custom group of stocks
    > tech_stocks = ('AAPL', 'MSFT', 'GOOGL')
    > tech_filter = lambda p: p.sid in tech_stocks
    > for stock in tech_stocks:
    >    order_target_percent(stock, 1/3, percent_of_fn=tech_filter)
    > ```

* Forward arguments from __init__ to the user-defined initialize().
  [PR456](https://github.com/quantopian/zipline/pull/456)

  >  If you used the new way of creating an algorithm by defining an
  `initialize()` and a `handle_data()` function that you pass to
  `TradingAlgorithm` it was not possible to externally set variables in `initialize`. This is quite an impediment to parameter optimization where you want to be able to run the `TradingAlgorithm` many times passing in different parameter values that you set in `initialize()`.

  > Example:
  > ```python
  > def initialize(context, param=0):
  >     context.param = param
  > def handle_data(context, data):
  >     # use param in some way
  >     ...
  > # Instantiate algorithm, setting param to 3.
  > algo = zipline.TradingAlgorithm(initialize,
                                    handle_data,
                                    param=3)
  > perf = algo.run(data)
  > ```