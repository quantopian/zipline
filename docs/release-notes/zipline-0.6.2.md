# Zipline 0.6.2 Release Notes

**Highlights**

* Command line interface to run algorithms directly.

* IPython Magic %%zipline that runs algorithm defined in an IPython
  notebook cell.

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
   %%zipline --symbols AAPL --start 2011-1-1 --end 2012-1-1 -o perf
   ```

   > Does the same as above except instead of executing the file looks
   for the algorithm in the cell and instead of outputting the perf df
   to a file, creates a variable in the namespace called perf.

## Bug Fixes (BUG)

## Performance (PERF)

## Maintenance and Refactorings (MAINT)

## Build (BLD)

# Contributors
