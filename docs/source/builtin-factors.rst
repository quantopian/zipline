======================
Using Built-In Factors
======================

In this algo, you will learn about using built in factors (name TBD), how to
add and filter by them, and how to access your factors on the data object.

.. code-block:: Python

    from zipline.modelling.factor.technical import SimpleMovingAverage
    # Pricing data, needs to be imported to be used in your factors.
    from zipline.data.equities import USEquityPricing

    # In initialize, you declare which factors and filters you want.
    def initialize(context):

        # Calling a factor indicates you would like it computed.
        sma_short = SimpleMovingAverage(inputs=[USEquityPricing.close], window_length=10)

        # Factors return a scalar value, you can add this value to the data object.
        # Factors will be calculated for the entire universe of securities.
        add_factor(sma_short, 'sma_short')

        # Filter what is included on the data object
        # from the entire universe to those securities you want included.
        # Factors can be use in filters.
        add_filter(sma_short > 0)

    # The data object is now included in before_trading_starts.
    # The results of your factors and filters are stored as a dataframe at data.factors.
    def before_trading_start(context, data):
        # Your factors and filters are computed the first time data.factors is called.
        # Their computation will be optimized for maximum performance,
        # which is possible because they are all declared in initialize.
        context.my_factors = data.factors

        # In before trading starts you can use the information to set your universe
        # Here we are just logging the information
        print "Number of securities returned = %s" % len(context.my_factors)
        log.info("\n" + str(context.my_factors.sort(['sma_short'], ascending=False).head(5)))

    def handle_data(context, data):
        order(sid(24), 50)
