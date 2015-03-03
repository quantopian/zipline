import zipline.protocol as zp
from zipline.history.history_container import HistoryContainer


class BaseDataverse(object):
    """
    Cross-sectional logic that handles all universe and data activities.
    This includes things like:
        Data sources
        history calls
        BarData
        SIDData
        etc

    The conception being that implementation that differs between live and
    backtesting should exist in one conceptual unit.

    OBVs this is a work in progresss.
    """
    siddata_class = zp.SIDData
    bardata_class = zp.BarData
    history_container_class = HistoryContainer

    def get_bar_data(self):
        return self.bardata_class()

    def get_history_container(self, *args, **kwargs):
        return self.history_container_class(*args, **kwargs)
