import datetime
import pytz
import msgpack
import random
import zipline.util as qutil
import zipline.finance.risk as risk
import zipline.protocol as zp

def load_market_data():
    fp_bm = open("./zipline/test/benchmark.msgpack", "rb")
    bm_map = msgpack.loads(fp_bm.read())
    bm_returns = []
    for epoch, returns in bm_map.iteritems():
        bm_returns.append(risk.daily_return(date=datetime.datetime.fromtimestamp(epoch).replace(hour=0, minute=0, second=0, tzinfo=pytz.utc), returns=returns))
    bm_returns = sorted(bm_returns, key=lambda(x): x.date) 
    fp_tr = open("./zipline/test/treasury_curves.msgpack", "rb")
    tr_map = msgpack.loads(fp_tr.read())
    tr_curves = {}
    for epoch, curve in tr_map.iteritems():
        tr_curves[datetime.datetime.fromtimestamp(epoch).replace(hour=0, minute=0, second=0, tzinfo=pytz.utc)] = curve
        
    return bm_returns, tr_curves
    

def create_trade(sid, price, amount, datetime):
    row = zp.namedict({
        'source_id' : "test_factory",
        'type'      : zp.DATASOURCE_TYPE.TRADE,
        'sid'       : sid,
        'dt'        : datetime,
        'price'     : price,
        'volume'    : amount
    })
    return row

def create_trade_history(sid, prices, amounts, start_time, interval, trading_calendar):
    i = 0
    trades = []
    current = start_time.replace(tzinfo = pytz.utc)

    for price, amount in zip(prices, amounts):

        if(trading_calendar.is_trading_day(current)):
            trade = create_trade(sid, price, amount, current)
            trades.append(trade)

            current = current + interval
        else:
            current = current + datetime.timedelta(days=1)

    return trades

def create_txn(sid, price, amount, datetime, btrid=None):
    txn = zp.namedict({
        'sid':sid,
        'amount':amount, 
        'dt':datetime,
        'price':price, 
    })
    return txn

def create_txn_history(sid, priceList, amtList, startTime, interval, trading_calendar):
    txns = []
    current = startTime

    for price, amount in zip(priceList, amtList):

        if trading_calendar.is_trading_day(current):
            txns.append(create_txn(sid, price, amount, current))
            current = current + interval

        else:
            current = current + datetime.timedelta(days=1)

    return txns


def create_returns(daycount, start, trading_calendar):
    i = 0
    test_range = []
    current = start.replace(tzinfo=pytz.utc)
    one_day = datetime.timedelta(days = 1)
    while i < daycount: 
        i += 1
        r = risk.daily_return(current, random.random())
        test_range.append(r)
        current = current + one_day
    return [ x for x in test_range if(trading_calendar.is_trading_day(x.date)) ]
    

def create_returns_from_range(start, end, trading_calendar):
    current = start.replace(tzinfo=pytz.utc)
    end = end.replace(tzinfo=pytz.utc)
    one_day = datetime.timedelta(days = 1)
    test_range = []
    i = 0
    while current <= end: 
        current = current + one_day
        if(not trading_calendar.is_trading_day(current)):
            continue
        r = risk.daily_return(current, random.random())
        i += 1
        test_range.append(r)

    return test_range
    
def create_returns_from_list(returns, start, trading_calendar):
    current = start.replace(tzinfo=pytz.utc)
    one_day = datetime.timedelta(days = 1)
    test_range = []
    i = 0
    while len(test_range) < len(returns): 
        if(trading_calendar.is_trading_day(current)):
            r = risk.daily_return(current, returns[i])
            i += 1
            test_range.append(r)
        current = current + one_day
    return sorted(test_range, key=lambda(x):x.date)

