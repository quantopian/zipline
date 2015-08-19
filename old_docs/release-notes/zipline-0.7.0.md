# Zipline 0.7.0 Release Notes

**Highlights**

* Command line interface to run algorithms directly.

* IPython Magic %%zipline that runs algorithm defined in an IPython
  notebook cell.

* API methods for building safeguards against runaway ordering and undesired
  short positions.

* New history() function to get a moving DataFrame of past market data
  (replaces BatchTransform).

* A new [beginner tutorial](http://nbviewer.ipython.org/github/quantopian/zipline/blob/master/docs/tutorial.ipynb).


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

* Expanded record() functionality for dynamic naming. [PR325](https://github.com/quantopian/zipline/pull/355)

   > The record() function can now take positional args before the kwargs.
   > All original usage and functionality is the same, but now these
   > extra usages will work:
   >

                     name = 'Dynamically_Generated_String'
                     record( name, value, ... )
                     record( name, value1, 'name2', value2, name3=value3, name4=value4 )

   > The requirements are simply that the poritional args occur only before the
   > kwargs.

* history() has been ported from Quantopian to Zipline and provides moving window of market data. [PR345](https://github.com/quantopian/zipline/pull/345) and [PR357](https://github.com/quantopian/zipline/pull/357)

    > history() replaces BatchTransform. It is faster, works for minute level data and has a superior interface.
    > To use it, call `add_history()` inside of `initialize()` and then receive a pandas `DataFrame` by calling
    > history() from inside `handle_data()`. Check out the [tutorial](http://nbviewer.ipython.org/github/quantopian/zipline/blob/master/docs/tutorial.ipynb) and an [example](https://github.com/quantopian/zipline/blob/master/zipline/examples/dual_moving_average.py).

* history() now supports `1m` window lengths [PR345](https://github.com/quantopian/zipline/pull/345)

## Bug Fixes (BUG)

* Fix alignment of trading days and open and closes in trading environment.
  [PR331](https://github.com/quantopian/zipline/pull/331)

* RollingPanel fix when adding/dropping new fields [PR349](https://github.com/quantopian/zipline/pull/349)

## Performance (PERF)

## Maintenance and Refactorings (MAINT)

* Removed undocumented and untested HDF5 and CSV data sources. [PR267](https://github.com/quantopian/zipline/issues/267)

* Refactor sim_params [PR352](https://github.com/quantopian/zipline/pull/352)

* Refactoring of history [PR340](https://github.com/quantopian/zipline/pull/340)

## Build (BLD)

* The following dependencies have been updated (zipline might work with other versions too):
```diff
-pytz==2013.9
-numpy==1.8.0
+pytz==2014.4
+numpy==1.8.1

+scipy==0.12.0
+patsy==0.2.1
+statsmodels==0.5.0
-six==1.5.2
+six==1.6.1

-Cython==0.20
-TA-Lib==0.4.8
+Cython==0.20.1
+Cython==0.20.1
+--allow-external TA-Lib --allow-unverified TA-Lib TA-Lib==0.4.8

-requests==2.2.0
+requests==2.3.0

-nose==1.3.0
+nose==1.3.3
-xlrd==0.9.2
+xlrd==0.9.3

-pep8==1.4.6
-pyflakes==0.7.3
-pip-tools==0.3.4
+pep8==1.5.7
+pyflakes==0.8.1

-scipy==0.13.2
-tornado==3.2
-pyparsing==2.0.1
-patsy==0.2.1
-statsmodels==0.4.3
+tornado==3.2.1
+pyparsing==2.0.2

q-Markdown==2.3.1
+Markdown==2.4.1
```

# Contributors
    38  Scott Sanderson
    29  Thomas Wiecki
    26  Eddie Hebert
     6  Delaney Granizo-Mackenzie
     3  David Edwards
     3  Richard Frank
     2  Jonathan Kamens
     1  Pankaj Garg
     1  Tony Lambiris
     1  fawce
