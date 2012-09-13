from zipline.gens.transform import EventWindowBatch

class CovEventWindow(EventWindowBatch):
    def get_value(self, prices, volumes):
        return prices.cov()