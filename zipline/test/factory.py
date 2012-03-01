import datetime
import pytz
import zipline.util as qutil
import zipline.finance.risk as risk
import zipline.protocol as zp

                    
def create_trade(sid, price, amount, datetime):
    row = {}
    row['source_id'] = "test_factory"
    row['type'] = zp.DATASOURCE_TYPE.TRADE
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