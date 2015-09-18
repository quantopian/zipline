=======================
Creating Custom Factors
=======================

This is the third of 3 example algos showing the new modeling API.

In this example, we build on the lessons of the first two examples. Here we
show how to:
- Compute custome factors
- Use fundamentals and pricing data to calculate factors
- Use percentile_between to filter the universe

.. code-block:: Python

    # Custom factors can be created. First you have to import CustomFactor.
    from zipline.modelling.factor import CustomFactor
    from zipline.modelling.factor.technical import VWAP, SimpleMovingAverage
    from zipline.data.equities import USEquityPricing

    # Custom factors are defined as a class object, outside of initialize or
    # handle data
    class MarketCap(CustomFactor):
        # You can chose to specify a default set of inputs and window length.
        # If you do not specify the default, they will have to be passed in,
        # as shown in the sma and vwap examples below.

        # You do not currently have to import fundamentals data.
        # You do have use a big F when using it as an input (these things will change)
        inputs = [USEquityPricing.close, Fundamentals.valuation.shares_outstanding]
        window_length = 1

        # The compute function is where the magic happens.
        # It has 4 required inputs, self, today, assets and out.
        # Any inputs specified above also need to be included here.
        # Compute gets passed numpy arrays determined by the declared inputs
        def compute(self, today, assets, out, close, shares):

            # Here we are computing an updated market cap using the most recent price and
            # number of shares outstanding.
            out[:] = close[-1] * shares[-1]


    def initialize(context):

        sma_short = SimpleMovingAverage(inputs=[USEquityPricing.close], window_length=30)
        add_factor(sma_short, 'sma_short')

        sma_long = SimpleMovingAverage(inputs=[USEquityPricing.close], window_length=100)
        add_factor(sma_long, 'sma_long')

        sma_val = sma_short/sma_long
        add_factor(sma_val, 'sma_val')

        # Custom formulae are declared exactly as built in factor.
        mkt_cap = MarketCap()
        add_factor(mkt_cap, 'mkt_cap')

        add_factor(sma_val.rank(), 'sma_rank')

        # Percentile between can be used to automatically filter for
        # assets that fell within the specified percentile range each day
        add_filter(mkt_cap.percentile_between(50, 100))

    def before_trading_start(context, data):
        context.my_factors = data.factors
        print "Number of securities returned = %s" % len(context.my_factors)

        context.short_list = data.factors.sort(['sma_rank'], ascending=True).iloc[:200]
        context.long_list = data.factors.sort(['sma_rank'], ascending=True).iloc[-200:]

        update_universe(context.long_list.index.union(context.short_list.index))

    def handle_data(context, data):

        print "SHORT LIST"
        print len(context.short_list)
        log.info("\n" + str(context.short_list.sort(['sma_rank'], ascending=True).head()))
        print "LONG LIST"
        print len(context.long_list)
        log.info("\n" + str(context.long_list.sort(['sma_rank'], ascending=False).head()))
