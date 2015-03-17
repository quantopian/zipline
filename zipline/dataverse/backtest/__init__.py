from zipline.dataverse.dataverse import BaseDataverse

from .history import BacktestHistoryContainer
from .bardata import BacktestBarData, BacktestSIDData


class BacktestDataverse(BaseDataverse):
    history_container_class = BacktestHistoryContainer
    siddata_class = BacktestSIDData
    bardata_class = BacktestBarData

    def get_history_container(self, *args, **kwargs):
        kwargs['dataverse'] = self
        return self.history_container_class(*args, **kwargs)

    def get_source(self, source, overwrite_sim_params=True):
        self.source = source
        return super(BacktestDataverse, self).get_source(source)

    def update_universe(self, event):
        # we do not update universe i.e. BarData
        pass
