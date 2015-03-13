import warnings

import pandas as pd
import zipline.protocol as zp
from zipline.history.history_container import HistoryContainer
from zipline.sources.data_frame_source import (
    DataFrameSource,
    DataPanelSource
)


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

    def __init__(self):
        self.current_data = self.bardata_class()

    def get_bar_data(self):
        return self.current_data

    def get_history_container(self, *args, **kwargs):
        return self.history_container_class(*args, **kwargs)

    def get_source(self, source, overwrite_sim_params=True):
        if isinstance(source, list):
            if overwrite_sim_params:
                warnings.warn("""List of sources passed, will not attempt to extract sids, and start and end
 dates. Make sure to set the correct fields in sim_params passed to
 __init__().""", UserWarning)
                overwrite_sim_params = False
        elif isinstance(source, pd.DataFrame):
            # if DataFrame provided, wrap in DataFrameSource
            source = DataFrameSource(source)
        elif isinstance(source, pd.Panel):
            source = DataPanelSource(source)
        return source

    def update_universe(self, event):
        self.current_data.update_sid(event)
