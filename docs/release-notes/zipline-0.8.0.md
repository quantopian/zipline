# Zipline 0.8.0 Release Notes

## Highlights

  * New documentation system with a new website at [zipline.io](http://www.zipline.io)
  * Major performance enhancements.
  * Dynamic history.

## Bug Fixes (BUG)

### Fix a bug where the reported returns could sharply dip for random periods of time. [PR378](https://github.com/quantopian/zipline/pull/378)

## Enhancements (ENH)

### Account object: Adds an account object to conext to track information about the trading account. [PR396](https://github.com/quantopian/zipline/pull/396)

  > Example:

  > ```
  > context.account.settled_cash
  > ```

  > Returns the settled cash value that is stored on the account object. This
  > value is updated accordingly as the algorithm is run.

### HistoryContainer can now grow dynamically. [PR412](https://github.com/quantopian/zipline/pull/412)

  > Calls to `history` will now be able to increase the size or change the shape
  > of the history container to be able to service the call. `add_history` now
  > acts as a preformance hint to pre-allocate sufficient space in the
  > container. This change is backwards compatible with `history`, all existing
  > algorithms should continue to work as intended.

### Simple transforms ported from quantopian and use history. [PR429](https://github.com/quantopian/zipline/pull/429)

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
```python
# The standard deviation of the price in the last 3 days.
data[security].stdev(3)
```

### New fields in Performance Period [PR464](https://github.com/quantopian/zipline/pull/464)

  > Performance Period has new fields accessible in return value of to_dict:

  > - gross leverage
  > - net leverage
  > - short exposure
  > - long exposure
  > - shorts count
  > - longs count


### Allow order_percent to work with various market values (by Jeremiah Lowin) [PR477](https://github.com/quantopian/zipline/pull/477)

  > Currently, `order_percent()` and `order_target_percent()` both operate as a percentage of `self.portfolio.portfolio_value`. This PR lets them operate as percentages of other important MVs.

  > Also adds `context.get_market_value()`, which enables this functionality.

  > For example:
```python
# this is how it works today (and this still works)
# put 50% of my portfolio in AAPL
order_percent('AAPL', 0.5)
# note that if this were a fully invested portfolio, it would become 150% levered.

# take half of my available cash and buy AAPL
order_percent('AAPL', 0.5, percent_of='cash')

# rebalance my short position, as a percentage of my current short book
order_target_percent('MSFT', 0.1, percent_of='shorts')

# rebalance within a custom group of stocks
tech_stocks = ('AAPL', 'MSFT', 'GOOGL')
tech_filter = lambda p: p.sid in tech_stocks
for stock in tech_stocks:
   order_target_percent(stock, 1/3, percent_of_fn=tech_filter)
```

### Major performance enhancements to history (by Dale Jung) [PR488](https://github.com/quantopian/zipline/commit/38e8d5214d46f089020703712dc6b3f4f6ee084d)

## Contributors

The following people have contributed to this release, ordered by numbers of commit:
```
   349  Eddie Hebert
   213  Thomas Wiecki
   134  fawce
   119  Jeremiah Lowin
    97  Dale Jung
    52  David Edwards
    45  Joe Jevnik
    45  Scott Sanderson
    32  Delaney Granizo-Mackenzie
    30  Richard Frank
    27  Ryan Day
    22  Ben McCann
    22  Jonathan Kamens
    19  twiecki
    14  Colin Alexander
    11  Tony Worm
    10  John Ricklefs
     8  Brian Cappello
     7  Brian Fink
     7  Moises Trovo
     7  Wes McKinney
     6  David Stephens
     6  Mete Atamel
     5  Elektra58
     5  Seong Lee
     4  Jason Kölker
     4  Mark Dunne
     4  Suminda Dharmasena
     4  Tobias Brandt
     4  llllllllll
     3  Chen Huang
     3  Jamie Kirkpatrick
     3  Jean Bredeche
     3  Luke Schiefelbein
     3  Matti Hanninen
     3  Nicholas Pezolano
     2  Aidan
     2  Martin Dengler
     2  Peter Cawthron
     2  Philipp Kosel
     2  jbredeche
     2  stanh
     1  Aaron Marz
     1  Ben
     1  Corey Farwell
     1  Ian Levesque
     1  Jeremi Joslin
     1  Justin Graves
     1  Michael Schatzow
     1  Pankaj Garg
     1  Stan
     1  Sébastien Drouyer
     1  The Gitter Badger
     1  Tony Lambiris
     1  cowmoo
```
