# Zipline 0.6.2 Release Notes

**Highlights**

* Command line interface to run algorithms directly.

* IPython Magic %%zipline that runs algorithm defined in an IPython
  notebook cell.

* API methods for building safeguards against runaway ordering and undesired
  short positions.


## Enhancements (ENH)

* CLI: Adds a CLI and IPython magic for zipline. [PR325](https://github.com/quantopian/zipline/pull/325)

  > Example:

  > ```
  > python run_algo.py -f dual_moving_avg.py --symbols AAPL --start 2011-1-1 --end 2012-1-1 -o dma.pickle
  > ```

  > Grabs the data from yahoo finance, runs the file
  dual_moving_avg.py (and looks for `dual_moving_avg_analyze.py`
  which, if found, will be executed after the algorithm has been run),
  and outputs the perf `DataFrame` to `dma.pickle`.

* IPython magic command (at the top of an IPython notebook cell). [PR325](https://github.com/quantopian/zipline/pull/325)

   > ```
   > %%zipline --symbols AAPL --start 2011-1-1 --end 2012-1-1 -o perf
   > ```

   > Does the same as above except instead of executing the file looks
   > for the algorithm in the cell and instead of outputting the perf df
   > to a file, creates a variable in the namespace called perf.

* Adds Trading Controls to the algorithm API. [PR329](https://github.com/quantopian/zipline/pull/329)

   > The following functions are now available on ```TradingAlgorithm``` and for algo scripts:
   >   - `set_max_order_size(self, sid=None, max_shares=None, max_notional=None)`
           - Set a limit on the absolute magnitude, in shares and/or total
             dollar value, of any single order placed by this algorithm for a
             given sid. If `sid` is None, then the rule is applied to any order
             placed by the algorithm.
           - Example:

                     def initialize(context):
                          # Algorithm will raise an exception if we attempt to place an
                          # order which would cause us to hold more than 10 shares
                          # or 1000 dollars worth of sid(24).
                          set_max_order_size(sid(24), max_shares=10, max_notional=1000.0)

   >   - `set_max_position_size(self, sid=None, max_shares=None, max_notional=None)`
           - Set a limit on the absolute magnitude, in either shares or dollar
             value, of any position held by the algorithm for a given sid. If `sid`
             is None, then the rule is applied to any position held by the
             algorithm.
           - Example:

                     def initialize(context):
                         # Algorithm will raise an exception if we attempt to order more than
                         # 10 shares or 1000 dollars worth of sid(24) in a single order.
                         set_max_order_size(sid(24), max_shares=10, max_notional=1000.0)

   >   - `set_max_order_count(self, max_count)`
           - Set a limit on the number of orders that can be placed by the
             algorithm in a single trading day.
           - Example:

                     def initialize(context):
                         # Algorithm will raise an exception if more than 50 orders are placed in a day.
                         set_max_order_count(50)

   >   - `set_long_only(self)`
           - Set a rule specifying that the algorithm may not hold short positions.
           - Example:

                     def initialize(context):
                         # Algorithm will raise an exception if it attempts to place
                         # an order that would cause it to hold a short position.
                         set_long_only()

* Adds an `all_api_methods` classmethod on `TradingAlgorithm` that returns a
  list of all `TradingAlgorithm` API methods. [PR333](https://github.com/quantopian/zipline/pull/333)


## Bug Fixes (BUG)

* Fix alignment of trading days and open and closes in trading environment.
  [PR331](https://github.com/quantopian/zipline/pull/331)

## Performance (PERF)

## Maintenance and Refactorings (MAINT)

## Build (BLD)

# Contributors
