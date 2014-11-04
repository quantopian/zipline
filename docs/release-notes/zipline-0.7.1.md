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

  Calls to `history` will now be able to increase the size or change the shape
  of the history container to be able to service the call. `add_history` now
  acts as a preformance hint to pre-allocate sufficient space in the
  container. This change is backwards compatible with `history`, all existing
  algorithms should continue to work as intended.
