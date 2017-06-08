import pandas as pd;
from zipline.data import loader
from zipline.algorithm import TradingAlgorithm
from datetime import datetime
import pytz
class AlgoRunnerIterator(object):
    '''
    This class is scheduling Trading Algorithm to run multiple times 
    '''
    
    def __init__(self, timeConfigPath, initialize, handle_data, stockNames):
        '''
        Constructor
        '''
        self.timeConfigPath = timeConfigPath;
        self.stockName = stockNames;
        self.timeDF = self._getTimeScheduleTable();
        self.initialize = initialize    
        self.handle_data = handle_data
        self.resetIndex();
        
    def hasNext(self):
        return self.index < self.timeDF.shape[0];
    
    def algoRunNext(self):
        if(self.hasNext()):
            currentSeriesValue = self.timeDF.iloc[self.index];
            self.index += 1;
            start, end = self._getStartEndTimeFromCurrentSeries(currentSeriesValue);
            algo = TradingAlgorithm(initialize=self.initialize, 
                        handle_data=self.handle_data, identifiers=self.stockName)
            data = loader.load_bars_from_yahoo(stocks=self.stockName,
                            start=start, end=end);
            return algo.run(data);
        else:
            raise ValueError;
        
    def setInitialize(self, initialize):
        self.initialize= initialize;
        self.resetIndex();
        
    def setHandleData(self, handle_data):
        self.handle_data= handle_data;
        self.resetIndex();
        
    def resetIndex(self):
        self.index = 0;
        
    def _getTimeScheduleTable(self):
        return pd.read_csv(self.timeConfigPath, comment='#'); 
        
    def _strToDatetime(self, dateStr):
        utc=pytz.UTC
        strAsDatetime = datetime.strptime(dateStr, '%Y-%m-%d')
        return utc.localize(strAsDatetime)

    def _getStartEndTimeFromCurrentSeries(self, currentSeriesValue):
        startDateStr = currentSeriesValue['start']
        endDateStr = currentSeriesValue['end']
        startDate = self._strToDatetime(startDateStr)
        endDate = self._strToDatetime(endDateStr)
        return startDate, endDate