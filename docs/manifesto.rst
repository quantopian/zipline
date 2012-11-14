********************
Quantopian Manifesto
********************

Wall Street's culture was born in an age of information scarcity.
Hoarding information and keeping secrets were the norm. The world has changed.
Today's world is defined by information that wants to be free. The new
scarcity is people: people with the talent and drive to wring insight from all
of that data.

Quantopian's mission is to attract the world's algorithmic and
financial talent. We want to attract today's quants, and we want to
attract talent that hasn't yet had the opportunity to be a quant.  We
want to bring this talent together, provide them with the tools that they
require, and help them build a community.  First and foremost, our community is
rooted in openness and sharing. Members share code, know-how, and data.
Quantopian sets the tone by providing open-sourced code, discussing our
techniques, and supplying the historical data needed for algorithmic investing.

By educating more people about statistical arbitrage and data mining for
finance, we aim to dispense with the secrecy and raise the state of the art.
Rather than hoard data, we relentlessly push data to our community. We want to
diversify the data that can be mined, and permit our members to explore as much
as they like. Our members' success in analyzing and investing will help
us draw more data and more members to our community. Every individual's
success will also help other Quantopians.

The Evolution of Algorithmic Finance 
====================================

Charting 
--------

Algorithmic finance originated as chart reading. Chartists would look for
certain patterns in price history charts. The patterns were always graced with
artfully chosen names like 'head and shoulders,' 'spinning top', or 'morning
star'. Chart reading looks a lot like palm reading, and for the skeptics among
us the similarities don't end with appearances. Still, chart reading is an
attempt to infer the balance of buying and selling appetites in the markets
from a stock's history. Viewed that way, chart reading pursues the noble goal
of prediction. Charting is so common that certain events can trigger market
responses, possibly because so many participants infer the same meaning from a
stock's price chart.

Technical Analysis 
------------------

Analysis grew more sophisticated as chartists gave way to
computer scientists writing algorithms. These algorithms have more scientific
sounding names like Moving Averages, Volume Weighted Moving Averages, Bollinger
Bands, Relative Strength Indicators, and Pearson's Correlation
Coefficient. Building technical analysis algorithms looks a lot like modern
statistics, and the optimists among us would say the similarities run deep.
Technical analysts take algorithmic approaches to the same concept: inferring
future behavior from trailing data. In addition to greater sophistication,
technical analysts can also test their algorithms over historic data. Imperfect
to be sure, but a giant leap from staring at a chart.

Reasonable people can disagree about the 'correctness' of
inferring future events from past behavior. Rather than dwell on that question,
we choose to point out a different limitation of both charting and modern
technical analysis: **both interpret the movement of a single stock in
isolation**.  This limitation is both a blessing and a curse.

On the one hand, there is little room for sophisticated statistics or machine
learning when you have just a single time series for both your signal and your
prediction target.

On the other, technical analysis can still be intuitive, which makes it easier
to get acquainted with the idea of automated trading. Often there is a mental
leap for people to make from understanding the interpretation of a price series
to issuing orders. Because the signals are easy to understand, technical
analysis makes for a good initial learning experience to explore risk and
performance evaluation as well as order management: the price going above its
30 day moving average is something you can visualize. So, you can focus your
attention on the financial and trading aspects of the problem.

Statistical Arbitrage 
---------------------

Statistical Arbitrage is the grandchild of chart reading.
Like technical analysis it relies on algorithms and statistics, but it departs
in one very significant way: 'stat arb' looks for relationships
among many stocks. The challenge with stat arb is twofold:

* visualizing the relationships can be quite difficult, since the relationships
  can have high dimensionality 
* the data processing load is quite high - a simple linear regression for all
  stocks results in 32 million individual regressions. Assuming a 10-day
  window, that can be 320 million individual calculations.  To prepare,
  backtest, and trade a stat arb strategy required both familiarity with the
  mechanics of trading, knowledge of statistics, and a strong computer science
  background.

As stat arb matured, the competition to find stat arb strategies that work
became a two part race:

1.  execute the trades faster 
2.  find new ways to identify relationships within
    market data 

We think the pursuit of faster trades reached diminishing returns when the
market hit sub-millisecond trade execution. We think that the resulting high
level of liquidity is a good thing, but we agree with Thomas Petterffy that
`pursuing even faster trades
<http://www.npr.org/blogs/money/2012/08/27/159992076/a-father-of-high-speed-trading-thinks-we-should-slow-down>`_
"has absolutely no social value".

Finding new relationships in the market data is possible and more important now
than ever. In the summer of 2007, there was a sudden meltdown in quantitative
trading firms. Subsequent analysis points to quants crowding into the same
arbitrage bets, and an unforeseen fund liquidation driving all the quants to
unwind those bets concurrently.  We believe finding new relationships should
permit investments with lower correlation and lower risks.

Algorithmic Investing and the Future 
====================================

A revolution in market understanding happens next. We want Quantopian to enable
more quants than all of Wall Street combined. We want quants, new and old, to
explore and share new ways to view the market. We want to clear away the
obstacles that have so far kept all but a few from doing algorithmic investing
by:

* simulating with clean, high-quality market data for free 
* easy access to markets through trusted brokers 
* providing a robust, flexible open-source backtester to permit evaluation and
  iteration of algorithms 
* supporting a community that fosters the exchange of knowledge, ideas, code
  solutions, and data sources 

The community will find new ways to identify market opportunities. It may take
the form of new, non-market data sources, like news feeds or Twitter. It may be
new algorithmic techniques. Most likely, it will be something we
haven't heard of yet: your idea. The one you keep coming back to. The
idea you couldn't test without data. The idea that needs backtesting,
and iteration, and encouragement from other quants.

Do you want to unleash your idea? This is your chance. `Come hack Wall Street
<http://www.quantopian.com>`_.
