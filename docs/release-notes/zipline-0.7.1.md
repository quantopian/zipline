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

