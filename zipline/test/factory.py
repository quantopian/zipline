"""
Factory functions to prepare useful data for tests.
"""
import pytz
import msgpack
import random

from datetime import datetime, timedelta
import zipline.util as qutil
import zipline.finance.risk as risk
import zipline.protocol as zp
from zipline.sources import SpecificEquityTrades
from zipline.finance.trading import TradingEnvironment

def load_market_data():
    fp_bm = open("./zipline/test/benchmark.msgpack", "rb")
    bm_map = msgpack.loads(fp_bm.read())
    bm_returns = []
    for epoch, returns in bm_map.iteritems():
        event_dt = datetime.fromtimestamp(epoch)
        event_dt = event_dt.replace(
            hour=0, 
            minute=0, 
            second=0, 
            tzinfo=pytz.utc
        )
        
        daily_return = risk.DailyReturn(date=event_dt, returns=returns)
        bm_returns.append(daily_return)
    bm_returns = sorted(bm_returns, key=lambda(x): x.date) 
    fp_tr = open("./zipline/test/treasury_curves.msgpack", "rb")
    tr_map = msgpack.loads(fp_tr.read())
    tr_curves = {}
    for epoch, curve in tr_map.iteritems():
        tr_dt = datetime.fromtimestamp(epoch)
        tr_dt = tr_dt.replace(hour=0, minute=0, second=0, tzinfo=pytz.utc)
        tr_curves[tr_dt] = curve
        
    return bm_returns, tr_curves
    
def create_trading_environment():
    """Construct a complete environment with reasonable defaults"""
    benchmark_returns, treasury_curves = load_market_data()

    start = datetime.strptime("01/01/2006","%m/%d/%Y")
    start = start.replace(tzinfo=pytz.utc)
    trading_environment = TradingEnvironment(
        benchmark_returns,
        treasury_curves,
        period_start = start,
        capital_base = 100000.0
    )
    
    return trading_environment
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
            current = current + timedelta(days=1)

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
            current = current + timedelta(days=1)

    return txns


def create_returns(daycount, start, trading_calendar):
    i = 0
    test_range = []
    current = start.replace(tzinfo=pytz.utc)
    one_day = timedelta(days = 1)
    while i < daycount: 
        i += 1
        r = risk.DailyReturn(current, random.random())
        test_range.append(r)
        current = current + one_day
    return [ x for x in test_range if(trading_calendar.is_trading_day(x.date)) ]
    

def create_returns_from_range(start, end, trading_calendar):
    current = start.replace(tzinfo=pytz.utc)
    end = end.replace(tzinfo=pytz.utc)
    one_day = timedelta(days = 1)
    test_range = []
    i = 0
    while current <= end: 
        current = current + one_day
        if(not trading_calendar.is_trading_day(current)):
            continue
        r = risk.DailyReturn(current, random.random())
        i += 1
        test_range.append(r)

    return test_range
    
def create_returns_from_list(returns, start, trading_calendar):
    current = start.replace(tzinfo=pytz.utc)
    one_day = timedelta(days = 1)
    test_range = []
    i = 0
    while len(test_range) < len(returns): 
        if(trading_calendar.is_trading_day(current)):
            r = risk.DailyReturn(current, returns[i])
            i += 1
            test_range.append(r)
        current = current + one_day
    return sorted(test_range, key=lambda(x):x.date)

def create_daily_trade_source(sids, trade_count, trading_environment):
    """
    creates trade_count trades for each sid in sids list. 
    first trade will be on trading_environment.period_start, and daily 
    thereafter for each sid. Thus, two sids should result in two trades per 
    day. 
    
    Important side-effect: trading_environment.period_end will be modified
    to match the day of the final trade. 
    """
    trade_history = []
    for sid in sids:
        price = [10.1] * trade_count
        volume = [100] * trade_count
        start_date = trading_environment.period_start
        trade_time_increment = timedelta(days=1)

        generated_trades = create_trade_history( 
            sid, 
            price, 
            volume, 
            start_date, 
            trade_time_increment, 
            trading_environment 
        )
        
        trade_history.extend(generated_trades)
        
    trade_history = sorted(trade_history, key=lambda(x): x.dt)
    
    #set the trading environment's end to same dt as the last trade in the
    #history.
    trading_environment.period_end = trade_history[-1].dt
    
    source = SpecificEquityTrades("flat", trade_history)
    return source
        