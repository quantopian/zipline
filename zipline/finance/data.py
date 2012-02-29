import sys
import logging
import datetime
import sys
import os
import pymongo
import csv
import re
import copy
import datetime
import time
import pytz
import shutil
import urllib
import subprocess
from pymongo import ASCENDING, DESCENDING
from zipline.daemon import Daemon
import zipline.util as qutil
import zipline.db as db
import host_settings

class FinancialDataLoader():
    """
    Load trade and quote data from tickdata extracts into the db.
    Dates and times in the extracts must be in GMT.
    
    All data extract files are expected to be in $HOME/fdl/. The expected directory layout is::
        /benchmark.csv      -- this will be created from yahoo data each time load_bench_marks is run
        /interest_rates.csv --
    """
    BATCH_SIZE = 100

    def __init__(self):
        self.conn, self.db = db.DbConnection.get()
        self.data_file_path = os.environ['HOME'] + "/fdl/" 
        subprocess.call("mkdir {data_dir}".format(data_dir=self.data_file_path), shell=True)
        self.last_bm_close = None
        
    def load_bench_marks(self):
        """Fetches the S&P end of day pricing history from yahoo, loads it to db.bench_marks"""
        start = time.time()
        start_date = datetime.datetime(year=1950, month=1, day=3)
        end_date = datetime.datetime.utcnow()
        file_path = self.data_file_path + "benchmark.csv"
        fp = open(file_path + ".tmp", "wb")
        
        #create benchmark files
        #^GSPC 19500103
        query = {}
        query['s'] = "^GSPC" #the s&p 500
        query['d'] = end_date.month - 1 # end_date month, zero indexed
        query['e'] = end_date.day # end_date day str(int(todate[6:8])) #day
        query['f'] = end_date.year #end_date year str(int(todate[0:4]))
        query['g'] = "d" #daily frequency
        query['a'] = start_date.month - 1 #start_date month, zero indexed
        query['b'] = start_date.day #start_date day 
        query['c'] = start_date.year #start_date year
        
        #print query
        params = urllib.urlencode(query)
        params += "&ignore=.csv"

        url = "http://ichart.yahoo.com/table.csv?%s" % params
        qutil.LOGGER.info("fetching {url}".format(url=url))
        f = urllib.urlopen(url)
        fp.write(f.read())
        fp.close()
        qutil.LOGGER.info("fetched {url} Reversing.".format(url=url))
        
        tmp_file = file_path + ".tmp"
        reversed_tmp_file = file_path + ".rev"
        
        rcode = subprocess.call("tac {oldfile} > {newfile}".format(oldfile=tmp_file, newfile=reversed_tmp_file), shell=True)
        #on mac, there is no tac command, so use tail -r (which isn't available on debian)
        if rcode != 0:
            rcode = subprocess.call("tail -r {oldfile} > {newfile}".format(oldfile=tmp_file, newfile=reversed_tmp_file), shell=True)
        
        #tail -1 benchmark.csv.rev > benchmark.csv
        subprocess.call("echo \"date,open,high,low,close,volume,adj_close\" > {result}".format(newfile=reversed_tmp_file, result=self.data_file_path), shell=True)
        #sed '$d' < ~/fdl/benchmark.csv.rev >> ~/fdl/benchmark.csv   
        subprocess.call("sed '$d' < {newfile} >> {result}".format(newfile=reversed_tmp_file, result=self.data_file_path), shell=True)
        #clean up working files
        subprocess.call("rm {tmp} {reversed}".format(tmp=tmp_file, reversed=reversed_tmp_file), shell=True)
        
        #load the records into mongodb
        self.db.bench_marks.drop()
        qutil.LOGGER.info("processing benchmark info")
        self.parse_file(self.db.bench_marks, 
                        self.bench_mark_cb, 
                        file_path, 
                        ['date','open','high','low','close','volume','adj_close'], 
                        None,
                        0)
        qutil.LOGGER.info("benchmark info complete")
        total = time.time() - start
        qutil.LOGGER.info("%d seconds to load benchmark history" % total)
        
    def load_treasuries(self):
        """fetches data from the treasury.gov yield curve website, and populates the treasury_curves table.
        
        to explore data available from the treasury:
        http://www.treasury.gov/resource-center/data-chart-center/interest-rates/Pages/TextView.aspx?data=yield
        
        to fetch xml of all daily yield curves:
        http://data.treasury.gov/feed.svc/DailyTreasuryYieldCurveRateData
        """
        
        from xml.dom.minidom import parse
        self.db.treasury_curves.drop()
        path = os.path.join(self.data_file_path + "all_treasury_rates.xml")
        #download all data to local filesystem
        subprocess.call("curl http://data.treasury.gov/feed.svc/DailyTreasuryYieldCurveRateData > {path}".format(path=path), shell=True)
        dom = parse(path)
        
            
        entries = dom.getElementsByTagName("entry")
        for entry in entries:
            curve = {}
            curve['tid'] = self.get_node_value(entry, "d:Id")
            
            curve['date'] = self.get_treasury_date(self.get_node_value(entry, "d:NEW_DATE"))
            curve['1month'] = self.get_treasury_rate(self.get_node_value(entry, "d:BC_1MONTH"))
            curve['3month'] = self.get_treasury_rate(self.get_node_value(entry, "d:BC_3MONTH"))
            curve['6month'] = self.get_treasury_rate(self.get_node_value(entry, "d:BC_6MONTH"))
            curve['1year'] = self.get_treasury_rate(self.get_node_value(entry, "d:BC_1YEAR"))
            curve['2year'] = self.get_treasury_rate(self.get_node_value(entry, "d:BC_2YEAR"))
            curve['3year'] = self.get_treasury_rate(self.get_node_value(entry, "d:BC_3YEAR"))
            curve['5year'] = self.get_treasury_rate(self.get_node_value(entry, "d:BC_5YEAR"))
            curve['7year'] = self.get_treasury_rate(self.get_node_value(entry, "d:BC_7YEAR"))
            curve['10year'] = self.get_treasury_rate(self.get_node_value(entry, "d:BC_10YEAR"))
            curve['20year'] = self.get_treasury_rate(self.get_node_value(entry, "d:BC_20YEAR"))
            curve['30year'] = self.get_treasury_rate(self.get_node_value(entry, "d:BC_30YEAR"))
            self.db.treasury_curves.insert(curve, True)
    
    def get_treasury_date(self, dstring):
        return datetime.datetime.strptime(dstring.split("T")[0], '%Y-%m-%d') 
        
    def get_treasury_rate(self, string_val):
        val = self.guarded_conversion(float, string_val, None)
        if val != None:
            val = round(val / 100.0, 4)
        return val
    def get_node_value(self, entry_node, tag_name):
       return self.get_xml_text(entry_node.getElementsByTagName(tag_name)[0].childNodes)
       
    def get_xml_text(self, nodelist):
        rc = []
        for node in nodelist:
            if node.nodeType == node.TEXT_NODE:
               rc.append(node.data)
             
        return ''.join(rc) 
        
    def purge_quotes(self):
        self.db.equity.quotes.drop()
        
    def purge_trades(self):
        self.db.equity.trades.drop()
               
    def load_quotes(self):
        start = time.time()
        qutil.LOGGER.info("processing equity quotes")
        self.load_events(self.db.equity.quotes,
                   self.quoteRowCallback,
                   self.data_file_path + "2008/Quotes/DATA",
                   ['trade_date', 'trade_time','exchange_code','bid_price','ask_price', 'bid_size','ask_size'])
        qutil.LOGGER.info("quotes complete")
        total = time.time() - start
        qutil.LOGGER.info("%d seconds to update equity quotes" % total)
        
        
    def load_trades(self):
        start = time.time()
        qutil.LOGGER.info("processing equity minute bars")
        self.load_events(self.db.equity.trades.minute,
                   self.trade_cb,
                   os.path.join(self.data_file_path, "2008/Trades/MINUTE_DATA"),
                   ['trade_date','trade_time','price', 'volume'])
        qutil.LOGGER.info("minute trades complete")
        total = time.time() - start
        qutil.LOGGER.info("%d seconds to recreate equity trades" % total)
    
    def load_hourly_trades(self):
        start = time.time()
        qutil.LOGGER.info("processing equity hour bars")
        self.load_events(self.db.equity.trades.hourly,
                   self.trade_cb,
                   os.path.join(self.data_file_path, "2008/Trades/HOURLY_DATA"),
                   ['trade_date','trade_time','price','volume'])
        qutil.LOGGER.info("hourly trades complete")
        total = time.time() - start
        qutil.LOGGER.info("%d seconds to recreate equity trades" % total)

        
    def load_daily_close(self):
        start = time.time()
        qutil.LOGGER.info("processing equity daily close")
        self.load_events(self.db.equity.trades.daily,
                   self.trade_cb,
                   os.path.join(self.data_file_path, "2008/Trades/DAILY_DATA"),
                   ['trade_date','price', 'volume'])
        qutil.LOGGER.info("daily close complete")
        total = time.time() - start
        qutil.LOGGER.info("%d seconds to recreate equity trades" % total)
        
    def ensure_indexes(self):

        #ensure indexes on minute trades
        qutil.LOGGER.info("ensuring (+datetime, +sid) index on trades.minute")
        self.db.equity.trades.minute.ensure_index([("dt",ASCENDING),("sid",ASCENDING)],background=True) 
        qutil.LOGGER.info("(+datetime, +sid) index on trades.minute ready")
        
        #ensure indexes for hourly trades
        qutil.LOGGER.info("ensuring (sid, +datetime) index on trades.hourly")
        self.db.equity.trades.hourly.ensure_index([("dt",ASCENDING),("sid",ASCENDING)],background=True) 
        qutil.LOGGER.info("(sid, +datetime) index on trades.hourly ready")
        
        #ensure indexes for daily trades
        qutil.LOGGER.info("ensuring (+datetime,+sid) index on trades.daily")
        self.db.equity.trades.daily.ensure_index([("dt",ASCENDING),("sid",ASCENDING)],background=True) 
        qutil.LOGGER.info("(+datetime,+sid) index on trades.daily ready")
         
        #ensure indexes for orders and transactions
        qutil.LOGGER.info("ensuring (+backtestid) index on orders")
        self.db.orders.ensure_index([("back_test_run_id",ASCENDING)],background=True) 
        qutil.LOGGER.info("(+backtestid) index on orders ready")
        
        qutil.LOGGER.info("ensuring (+backtestid, +datetime) index on orders")
        self.db.orders.ensure_index([("back_test_run_id",ASCENDING),("dt",ASCENDING)],background=True) 
        qutil.LOGGER.info("(+backtestid, +datetime) index on orders ready")
        
        qutil.LOGGER.info("ensuring (+backtestid) index on orders")
        self.db.transactions.ensure_index([("back_test_run_id",ASCENDING)],background=True) 
        qutil.LOGGER.info("(+backtestid) index on orders ready")
        
        qutil.LOGGER.info("ensuring (+backtestid) index on transactions")
        self.db.transactions.ensure_index([("back_test_run_id",ASCENDING),("dt",ASCENDING)],background=True) 
        qutil.LOGGER.info("(+backtestid) index on transactions ready")
        
        #indexes for benchmarks and treasuries
        qutil.LOGGER.info("ensuring (+date) index on treasury_curves")
        self.db.treasury_curves.ensure_index([("date",ASCENDING)],background=True) 
        qutil.LOGGER.info(" (+date) index on treasury_curves ready")
        
        qutil.LOGGER.info("ensuring (-date) index on treasury_curves")
        self.db.treasury_curves.ensure_index([("date",DESCENDING)],background=True) 
        qutil.LOGGER.info(" (-date) index on treasury_curves ready")
        
        qutil.LOGGER.info("ensuring (+date) index on bench_marks")
        self.db.bench_marks.ensure_index([("date",ASCENDING)],background=True) 
        qutil.LOGGER.info(" (+date) index on bench_marks ready")
        
        qutil.LOGGER.info("ensuring (+symbol, +date) index on bench_marks")
        self.db.bench_marks.ensure_index([("symbol",ASCENDING),("date",ASCENDING)],background=True) 
        qutil.LOGGER.info(" (+symbol, +date) index on bench_marks ready")
            
    def load_security_info(self):
        start = time.time()
        qutil.LOGGER.info("processing company info")
        
        sourceFile = os.path.join(self.data_file_path, "2008/Trades/MINUTE_DATA/CompanyInfo/CompanyInfo.asc")
        self.db.securities.drop()
        self.parse_file(self.db.securities, 
                        self.security_cb, 
                        sourceFile, 
                        ['symbol','file name','company name','CUSIP','exchange','industry code','first date','last date','company id'], 
                        None,
                        0)
        qutil.LOGGER.info("company info complete")
        total = time.time() - start
        qutil.LOGGER.info("%d seconds to recreate equity trades" % total)
        
    
        
    def load_events(self, collection, rowCallBack, dataDirectory, csvFields):
        id_counter = 0
        listing = os.listdir(dataDirectory)
        processedDir = os.path.join(dataDirectory,"processed")
        if not os.path.exists(processedDir):
            os.mkdir(processedDir)
        for curFile in listing:
            if os.path.isdir(os.path.join(dataDirectory,curFile)):
                continue
            start = time.time()
            if id_counter == 0: #this is the first file we are processing, so we want to ensure we don't duplicate records
                minDateTime = self.get_latest_entry_for_sid(self.get_sid_from_filename(curFile),collection)
            else: 
                minDateTime = None #this isn't the first file, so don't bother querying
            rowCount, totalCount = self.parse_file(collection, rowCallBack, os.path.join(dataDirectory,curFile), csvFields, minDateTime, id_counter)
            id_counter = id_counter + rowCount
            parseTime = time.time() - start
            qutil.LOGGER.info("{time} seconds to parse and load {rowCount} records of {totalCount} from {file}. {rate} records/second".
                                    format(time = parseTime, rowCount=rowCount, totalCount=totalCount, file=curFile, rate = rowCount/parseTime))
            #we successfully processed the file without an exception, move it to the processed folder
            #qutil.LOGGER.info("moving data file to {newpath}".format(newpath=os.path.join(processedDir,curFile)))
            shutil.move(os.path.join(dataDirectory,curFile),os.path.join(processedDir,curFile))
       
    def parse_file(self, collection, rowCallBack, curFile, pFieldnames, minDateTime, id_counter):
        """Parses the given file into the collection. Returns tuple of the rows committed, rows in csvfile"""
        
        qutil.LOGGER.debug("processing {fn}".format(fn=curFile))
        cur_id = id_counter
        rowCount = 0
        csvRowCount = 0
        with open(curFile, 'rb') as f:
            reader = csv.DictReader(f,fieldnames=pFieldnames)
            header = False
            
            if csv.Sniffer().has_header(f.read(1024)):
                header = True
            f.seek(0)
            
            if header:
                reader.next()
            try:
                rows = []
                for row in reader:
                    #row['_id'] = cur_id
                    cur_id = cur_id + 1
                    csvRowCount += 1
                    utcDT, dt = self.get_event_datetime(row)
                    #only add rows that are after the mindate for the current sid.
                    if(minDateTime != None and dt <= minDateTime): 
                        continue
                    if(dt != None):
                        row['dt'] = dt
                    if('company id' not in pFieldnames):
                        company_id = self.get_sid_from_filename(curFile)
                        if(company_id):
                            row['sid'] = int(company_id)
                    if not rowCallBack(curFile, row):
                        continue
                    rows.append(row)
                    rowCount+=1
                    if(len(rows) >= self.BATCH_SIZE):
                        collection.insert(rows, safe=True)
                        rows = []
                if(len(rows) > 0):
                    collection.insert(rows, safe=True)
                    rows = None
            except csv.Error, e:
                sys.exit('file %s, line %d: %s' % (curFile, reader.line_num, e))
        return rowCount, csvRowCount

    def trade_cb(self, curFile, row):
        row['price'] = self.guarded_conversion(float,row['price'])
        row['volume'] = self.guarded_conversion(self.safe_int,row['volume'])   
        return True

    def bench_mark_cb(self, curFile, row):
        row['symbol'] = "GSPC"
        row['volume'] = self.guarded_conversion(int,row['volume'])
        row['open'] = self.guarded_conversion(float,row['open'])
        row['high'] = self.guarded_conversion(float,row['high'])
        row['low'] = self.guarded_conversion(float,row['low'])
        row['close'] = self.guarded_conversion(float,row['close'])
        row['adj_close'] = self.guarded_conversion(float,row['adj_close'])
        row['date'] = datetime.datetime.strptime(row['date'], '%Y-%m-%d')
        if self.last_bm_close == None:
            row['returns'] = (row['close'] - row['open'])/row['open']
        else:
            row['returns'] = (row['close'] - self.last_bm_close) / self.last_bm_close
        self.last_bm_close = row['close']
        return True
        
    def security_cb(self, curFile, row):
        """source columns: ['symbol','file name','company name','CUSIP','exchange','industry code','first date','last date','company id']"""
        row['sid'] = self.guarded_conversion(int,row['company id'])
        del(row['company id'])
        row['start_date'] = self.guarded_conversion(self.date_conversion, row['first date'])
        del(row['first date'])
        row['end_date'] = self.guarded_conversion(self.date_conversion, row['last date'])
        del(row['last date'])
        row['symbol'] = self.verify_symbol_in_filename(row['symbol'], row['file name'])
        del(row['file name'])
        row['company_name'] = row['company name']
        del(row['company name'])
        return True

    def guarded_conversion(self, conversion, strVal, default = None):
        if(strVal == None or strVal == ""):
            return default
        return conversion(strVal)
        
    def safe_int(self,str):
        """casts the string to a float to handle the occassionaly decimal point in int fields from data providers."""
        f = float(str)
        i = int(f)
        return i
        
    def date_conversion(self, dateStr):
        dt = datetime.datetime.strptime(dateStr, '%m/%d/%Y')
        dt = dt.replace (tzinfo = pytz.utc)
        return dt
        
    def verify_symbol_in_filename(self, symbol, file_name):
        if(symbol == file_name):
            return symbol
        
        parts = file_name.split('_')
        if(len(parts) == 2):
            return file_name
        else:
            raise Exception("found a mismatch between symbol and filename, but no underscore.")
            
    def get_event_datetime(self, row):
        """python 2.5 doesn't support %f for setting the microseconds, so this override is necessary.
            a significant side effect - the trade date and trade time elements are removed from this dictionary. done to 
            avoid storing the source fields in the db.
        """
        if row.has_key('trade_date') and row.has_key('trade_time'):    
            value = row['trade_date'] + "-" + row['trade_time']
            dt = datetime.datetime.strptime(value.split(".")[0], '%m/%d/%Y-%H:%M:%S') 
            dt = dt.replace(microsecond=int(value.split(".")[1]+"000"))
            del row['trade_date']
            del row['trade_time']
        elif row.has_key('trade_date'):
            dt = datetime.datetime.strptime(row['trade_date'],'%m/%d/%Y')
            del row['trade_date']
        else:
            return None, None
            
        utcDT = quantoenv.getUTCFromExchangeTime(dt) #store everything in UTC
        return utcDT, dt
        
    def get_sid_from_filename(self, filename):    
        
        regexp = r"(?P<company_id>[0-9]+)([.]csv)"
        result = re.search(regexp,filename)
        if(result):
            companyID = int(result.group('company_id'))
            return companyID       
        else:
            return None
            
    def get_latest_entry_for_sid(self, sid, collection):
        """checks given collection for the most recent record for the given sid."""
        results = collection.find(fields=["dt"],
                                    spec={"sid":sid},
                                    sort=[("dt",DESCENDING)],
                                    limit=1,
                                    as_class=quantoenv.DocWrap)
        
        if(results.count() > 0):
            return results[0].dt
        else:
            return datetime.datetime.min



class DataLoader(Daemon):
    """A daemon process that manages the data in the finance database."""
    
    def __init__(self, pidfile, operation):
        self.operation = operation
        self.pidfile = pidfile
        self.stdin = '/dev/null'
        self.stdout = '/dev/null'
        self.stderr = '/dev/null'
        
    def run(self):
        qutil.LOGGER.info("running operation: {op}".format(op=self.operation))
        try:
            fdl = FinancialDataLoader()
            if(self.operation == 'pt'):
                qutil.LOGGER.info("Purging trades from database!")
                fdl.purge_trades()
            elif(self.operation == 'ei'):
                qutil.LOGGER.info("Ensuring indexes.")
                fdl.ensure_indexes()
            elif(self.operation == 'lt'):  
                qutil.LOGGER.info("Loading trades into database.")
                fdl.loadTrades()
            elif(self.operation == 'lh'):  
                qutil.LOGGER.info("Loading trades into database.")
                fdl.load_hourly_trades()
            elif(self.operation == 'ld'):  
                qutil.LOGGER.info("Loading trades into database.")
                fdl.load_daily_close()  
            elif(self.operation == 'si'):
                qutil.LOGGER.info("Loading security info into database.")
                fdl.load_security_info()
            elif(self.operation == 'tr'):
                qutil.LOGGER.info("Loading US Treasury rates into database.")
                fdl.load_treasuries()
            elif(self.operation == 'bm'):
                qutil.LOGGER.info("loading benchmark data into database.")
                fdl.load_bench_marks()
            else:
                qutil.LOGGER.warning("Unknown command for load data: {op}.".format(op=self.operation))
            qutil.LOGGER.info("Finished.")
        except:
            qutil.LOGGER.exception("exiting load_data due to unexpected exception.")
        finally:
            logging.shutdown()


