import datetime
import pytz
import zipline.util as qutil
import zipline.finance.risk as risk

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
    return [ x for x in test_range if(risk.trading_calendar.is_trading_day(x.date)) ]

def createReturnsFromRange(start, end):
    current = start.replace(tzinfo=pytz.utc)
    end = end.replace(tzinfo=pytz.utc)
    one_day = datetime.timedelta(days = 1)
    test_range = []
    i = 0
    while current <= end: 
        current = current + one_day
        if(not risk.trading_calendar.is_trading_day(current)):
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
        if(risk.trading_calendar.is_trading_day(current)):
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
    
                          
def create_trade(sid, price, amount, datetime):
    row = {}
    row['source_id'] = "test_factory"
    row['type'] = "TRADE"
    row['sid'] = sid
    row['dt'] = datetime
    row['price'] = price
    row['volume'] = amount
    return row
 
def create_trade_history(sid, prices, amounts, start_time, interval):
    i = 0
    trades = []
    current = start_time.replace(tzinfo = pytz.utc)
    while i < len(prices):
        if(risk.trading_calendar.is_trading_day(current)):  
            trades.append(create_trade(sid, prices[i], amounts[i], current))
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
        if(risk.trading_calendar.is_trading_day(current)): 
            txns.append(createTxn(sid,priceList[i],amtList[i], current))
            current = current + interval
            i += 1
        else:
            current = current + datetime.timedelta(days=1) 
    return txns