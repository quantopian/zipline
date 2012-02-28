import datetime
import sys
import zipline.util as qutil
from zipline.finance.data import DataLoader

def print_usage():
    print """
        Usage is:
        python loaddata.py (pt | lt | lh | ld | ei | bm | si | help) 

        pt - purge trade collection from the db
        lt - load trades (minute bars) to the db
        lh - load trades (hour bars) to the db
        ld - load trades (daily close) to the db
        ei - ensure all indexes on all collections in tick and algo db
        tr - load treasury rates
        bm - load benchmark data
        si - load security info (sid, symbol, qualifier)
        help - display this message
        """

    
if __name__ == "__main__":  
    
    if len(sys.argv) == 2:
        qutil.configure_logging()
        operation = sys.argv[1]     
        if(operation not in['pt','lt','lh','ld','ei','si', 'tr','bm'] or operation == 'help'):
            print_usage()
        else:
            ts = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            pidfile = "/tmp/loaddata-{stamp}.pid".format(stamp=ts)
            daemon = DataLoader(pidfile,operation)
            qutil.LOGGER.info("DataLoader starting.")
            daemon.run()
            sys.exit(0)
    else:
        print_usage()
        sys.exit(2)
