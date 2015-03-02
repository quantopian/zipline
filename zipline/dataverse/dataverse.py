import zipline.protocol as zp


class DataVerse(object):
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
    siddata_cls = zp.SIDData
    bardata_cls = zp.BarData

    def get_bar_data(self):
        return self.bardata_cls()
