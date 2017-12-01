Trading Calendar Tutorial
-------------------------

What is a Trading Calendar?
~~~~~~~~~~~~~~~~~~~~~~~~~~~
A trading calendar represents the timing information of a single market exchange. The timing information is made up of two parts: sessions, and opens/closes. This is represented by the Zipline :func:`~zipline.utils.trading_calendar.TradingCalendar` class, and is used as the parent class for all new `TradingCalendar`s

A session represents a contiguous set of minutes, and has a label that is midnight UTC. It is important to note that a session label should not be considered a specific point in time, and that midnight UTC is just being used for convenience.

Why Should You Care About Trading Calendars?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let's say you want to buy a share of some equity on Tuesday, and then sell it on Saturday. If the exchange in which you're trading that equity is not open on Saturday, then in reality it would not be possible to trade that equity at that time, and you would have to wait until some other number of days past Saturday. It would then be unreasonable to allow your trading algorithm to also place a trade on Saturday.

In order for you to backtest your strategy, the dates in that are accounted for in your `data bundle`__ and the dates in your `TradingCalendar` should match up; if the dates don't match up, then you you're going to see some errors along the way. This holds for both minutely and daily data. 

__ http://www.zipline.io/bundles.html

The TradingCalendar Class
~~~~~~~~~~~~~~~~~~~~~~~~~

The `TradingCalendar` class has many properties we should be thinking about if we were to build our own `TradingCalendar` for an exchange. These include properties such as:

  - Name of the Exchange
  - Timezone
  - Open Time
  - Close Time
  - Regular & Adhoc Holidays
	- Special Opens & Closes

And several others. If you'd like to see all of the properties and methods available to you through the `TradingCalendar` API, please take a look at the `API Reference`__

__ http://www.zipline.io/appendix.html#trading-calendar-api

Now we'll take a look at the London Stock Exchange Calendar :func:`~zipline.utils.calendars.LSEExchangeCalendar` as an example below:

.. code-block:: python

   class LSEExchangeCalendar(TradingCalendar):
   """
   Exchange calendar for the London Stock Exchange

   Open Time: 8:00 AM, GMT
   Close Time: 4:30 PM, GMT

   Regularly-Observed Holidays:
   - New Years Day (observed on first business day on/after)
   - Good Friday
   - Easter Monday
   - Early May Bank Holiday (first Monday in May)
   - Spring Bank Holiday (last Monday in May)
     - Summer Bank Holiday (last Monday in May)
     - Christmas Day
     - Dec. 27th (if Christmas is on a weekend)
     - Boxing Day
     - Dec. 28th (if Boxing Day is on a weekend)
   """

   @property
   def name(self):
       return "LSE"

   @property
   def tz(self):
       return timezone('Europe/London')

   @property
   def open_time(self):
       return time(8, 1)

   @property
   def close_time(self):
       return time(16, 30)

   @property
   def regular_holidays(self):
       return HolidayCalendar([
           LSENewYearsDay,
           GoodFriday,
           EasterMonday,
           MayBank,
           SpringBank,
           SummerBank,
           Christmas,
           WeekendChristmas,
           BoxingDay,
           WeekendBoxingDay
       ])

You can create the `Holiday` objects mentioned in `def regular_holidays(self)` through the `pandas`__ module, `pandas.tseries.holiday.Holiday`, and also take a look at the `LSEExchangeCalendar`__ code as an example.

__ http://pandas.pydata.org/pandas-docs/stable/
__ https://github.com/quantopian/zipline/blob/master/zipline/utils/calendars/exchange_calendar_lse.py



Building a Custom Trading Calendar
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now we'll build our own custom trading calendar. This calendar will be used for trading assets that can be traded on a 24/7 exchange calendar.

First we'll start off by importing some modules that will be useful to us.

.. code-block:: python

	 # for setting our open and close times
   from datetime import time
	 # for setting our start and end sessions
   import pandas as pd
	 # for setting which days of the week we trade on
   from pandas.tseries.offsets import CustomBusinessDay
	 # for setting our timezone
	 from pytz import timezone

	 # for creating and registering our calendar
	 from zipline.utils.calendars import (
	     register_calendar,
			 TradingCalendar
	 )
	 from zipline.utils.memoize import lazyval


And now we'll actually build this calendar, which we'll call `TFSExchangeCalendar`:

.. code-block:: python

   class TFSExchangeCalendar(TradingCalendar):
	     """
			 An exchange calendar for trading assets 24/7

			 Open Time: 12AM, UTC
			 Close Time: 11:59PM, UTC
			 """

			 @property
			 def name(self):
			     """
					 The name of the exchange, which Zipline will look for
					 when we run our algorithm and pass TFS to
					 the --trading-calendar CLI flag
					 """
			     return "TFS"

			 @property
			 def tz(self):
			     """
					 The timezone in which we'll be running our algorithm
					 """
			     return timezone("UTC")

			 @property
			 def open_time(self):
				   """
					 The time in which our exchange will open each day
					 """
					 return time(0, 0)

			 @property
			 def close_time(self):
			     """
					 The time in which our exchange will close each day
					 """
					 return time(23, 59)

			@lazyval
			def day(self):
			    """
					The days on which our exchange will be open
					"""
					weekmask = "Mon Tue Wed Thu Fri Sat Sun"
					return CustomBusinessDay(
					    weekmask=weekmask
					)


And that's it! We now have an exchange calendar for trading assets 24/7. In order for you to run your algorithm with this calendar, you'll need have a data bundle in which your assets have dates that run through all days of the week. You can read about how to make your own data bundle in the `Writing a New Bundle`__ documentation, or use the `csvdir bundle`__ for creating a bundle from CSV files.

__ http://www.zipline.io/bundles.html#writing-a-new-bundle
__ https://github.com/quantopian/zipline/blob/master/zipline/data/bundles/csvdir.py
