import tornado.auth
import tornado.httpserver
import tornado.ioloop
from tornado.options import define, options
import tornado.web
import pymongo
import bson
import hashlib
import base64
import uuid
import os
import logging
import datetime
import random

import qbt_server

MINUTE_COUNT=390

define("user_email", default="qbt@quantopian.com", help="email address for qbt user")
define("password", default="foobar", help="password for qbt user")

def db_main():
    tornado.options.parse_command_line()
    connection, db = qbt_server.connect_db()
    
    #create a user for testing
    salt, encrypted_password = qbt_server.encrypt_password(None, options.password)
    
    if not db.users.find_one({'email':options.user_email}):
        db.users.insert({'email':options.user_email, 'encrypted_password':encrypted_password, 'salt':salt})
    
    #create one mythical company
    if not db.company_info.find_one({'sid':133}):
        db.company_info.insert({'sid':133, "exchange" : "NEW YORK STOCK EXCHANGE", "symbol" : "JHF", "first date" : "01/04/1993", "last date" : "10/01/2008", "sid" : 133, "industry code" : "130A", "company name" : "JACK INC"})    
    
    #create one mythical company
    if not db.company_info.find_one({'sid':134}):
        db.company_info.insert({'sid':134, "exchange" : "NEW YORK STOCK EXCHANGE", "symbol" : "RCF", "first date" : "01/04/1993", "last date" : "10/01/2008", "sid" : 134, "industry code" : "130A", "company name" : "ROCCO INC"})    
        
    #create minute equity data collection and populate with a day of random data
    prices = {133:25.0,134:45.0} #sid, initial price.
    if not db.equity.trades.minute.find().count() == MINUTE_COUNT * len(prices):
        db.equity.trades.minute.drop()
        trade_start = datetime.datetime.now()
        minute = datetime.timedelta(minutes=1)

        for i in range(MINUTE_COUNT):
            for sid,price in prices.iteritems():
                price = price + random.uniform(-0.05,0.05)
                db.equity.trades.minute.insert({'sid':sid, 'dt':trade_start + (minute * i),'price':price, 'volume':random.randrange(100,10000,100)})

if __name__ == "__main__":
    db_main()
