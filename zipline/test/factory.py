import datetime
import pytz
from algorithm.quantoenv import *
from algorithm.quantomodels import *
from algorithm.hostedalgorithm import *
from algorithm.risk import *

def createReturns(daycount, start):
    i = 0
    test_range = []
    current = start.replace(tzinfo=pytz.utc)
    one_day = datetime.timedelta(days = 1)
    while i < daycount: 
        i += 1
        r = daily_return(current, random.random())
        test_range.append(r)
        current = current + one_day
    return [ x for x in test_range if(trading_calendar.is_trading_day(x.date)) ]

def createReturnsFromRange(start, end):
    current = start.replace(tzinfo=pytz.utc)
    end = end.replace(tzinfo=pytz.utc)
    one_day = datetime.timedelta(days = 1)
    test_range = []
    i = 0
    while current <= end: 
        current = current + one_day
        if(not trading_calendar.is_trading_day(current)):
            continue
        r = daily_return(current, random.random())
        i += 1
        test_range.append(r)
        
    return test_range
def createReturnsFromList(returns, start):
    current = start.replace(tzinfo=pytz.utc)
    one_day = datetime.timedelta(days = 1)
    test_range = []
    i = 0
    while len(test_range) < len(returns): 
        if(trading_calendar.is_trading_day(current)):
            r = daily_return(current, returns[i])
            i += 1
            test_range.append(r)
        current = current + one_day
    return test_range
    

def createAlgo(filename):
    algo = Algorithm()
    algo.code = getCodeFromFile(filename)
    algo.title = filename
    algo._id = pymongo.objectid.ObjectId()
    hostedAlgo = HostedAlgorithm(algo)
    return hostedAlgo       
    
def getCodeFromFile(filename):
    rVal = None
    with open('./test/algo_samples/' + filename, 'r') as f:
        rVal = f.read()
    return rVal
    
                          
def createTrade(sid, price, amount, datetime):
    row = {}
    row['sid'] = sid
    row['dt'] = datetime
    row['price'] = price
    row['volume'] = amount
    row['exchange_code'] = "fake exchange"
    db = getTickDB()
    db.equity.trades.minute.insert(row,safe=True)
    dw = DocWrap()
    dw.store = row
    return dw
 
def create_trade_history(sid, prices, amounts, start_time, interval):
    i = 0
    trades = []
    current = start_time
    while i < len(prices):
        if(trading_calendar.is_trading_day(current)):  
            trades.append(createTrade(sid, priceList[i], amtList[i], current))
            current = current + interval
            i += 1
        else:
            current = current + datetime.timedelta(days=1)
        
    return trades 
    
def createTxn(sid, price, amount, datetime, btrid=None):
    txn = Transaction(sid=sid, amount=amount, dt = datetime, 
                      price=price, transaction_cost=-1*price*amount)
    return txn
    
def createTxnHistory(sid, priceList, amtList, startTime, interval):
    i = 0
    txns = []
    current = startTime
    while i < len(priceList):
        if(trading_calendar.is_trading_day(current)): 
            txns.append(createTxn(sid,priceList[i],amtList[i], current))
            current = current + interval
            i += 1
        else:
            current = current + datetime.timedelta(days=1) 
    return txns