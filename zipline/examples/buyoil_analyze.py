import matplotlib.pyplot as plt


def analyze(context, perf):
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    perf.portfolio_value.plot(ax=ax1)
    ax1.set_ylabel('portfolio value in $')

    ax2 = fig.add_subplot(212)
    perf['CLJ15'].plot(ax=ax2)
    perf['CLK15'].plot(ax=ax2)
    perf['AAPL'].plot(ax=ax2)

    perf_trans = perf.ix[[t != [] for t in perf.transactions]]
    jbuys = perf_trans.ix[[t[0]['amount'] > 0
                           for t in perf_trans.transactions
                           if t[0]['sid']==context.clj]]
    jsells = perf_trans.ix[[t[0]['amount'] < 0
                            for t in perf_trans.transactions
                            if t[0]['sid']==context.clj]]
    kbuys = perf_trans.ix[[t[0]['amount'] > 0
                           for t in perf_trans.transactions
                           if t[0]['sid']==context.clk]]
    ksells = perf_trans.ix[[t[0]['amount'] < 0
                            for t in perf_trans.transactions
                            if t[0]['sid']==context.clk]]
    abuys = perf_trans.ix[[t[0]['amount'] > 0
                           for t in perf_trans.transactions
                           if t[0]['sid']==context.aapl]]
    asells = perf_trans.ix[[t[0]['amount'] < 0
                            for t in perf_trans.transactions
                            if t[0]['sid']==context.aapl]]

    ax2.plot(jbuys.index, perf.CLJ15.ix[jbuys.index],
             '^', markersize=10, color='m')
    ax2.plot(jsells.index, perf.CLJ15.ix[jsells.index],
             'v', markersize=10, color='k')
    ax2.plot(kbuys.index, perf.CLK15.ix[kbuys.index],
             '^', markersize=10, color='m')
    ax2.plot(ksells.index, perf.CLK15.ix[ksells.index],
             'v', markersize=10, color='k')
    ax2.plot(abuys.index, perf.AAPL.ix[abuys.index],
             '^', markersize=10, color='m')
    ax2.plot(asells.index, perf.AAPL.ix[asells.index],
             'v', markersize=10, color='k')
    ax2.set_ylabel('price in $')
    plt.legend(loc=0)
    plt.show()
