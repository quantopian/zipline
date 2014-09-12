# Zipline 0.7.1 Release Notes

## Bug Fixes (BUG)

* Fix a bug where the reported returns could sharply dip for random periods of time.
  [PR378](https://github.com/quantopian/zipline/pull/378)

## Enhancements (ENH)

* Account object: Adds an account object to conext to track information about the trading account. [PR396](https://github.com/quantopian/zipline/pull/396)

  > Example:

  > ```
  > context.account.settled_cash
  > ```

  > Returns the settled cash value that is stored on the account object. This value is updated accordingly as the algorithm is run.