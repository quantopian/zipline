"""
Commonly used messaging components.
"""

import logging

import zipline.protocol as zp
from zipline.core.component import Component
from zipline.protocol import COMPONENT_TYPE

LOGGER = logging.getLogger('ZiplineLogger')

class DataSource(Component):
    """
    Abstract baseclass for data sources. Subclass and implement send_all -
    usually this means looping through all records in a store, converting
    to a dict, and calling send(map).

    Every datasource has a dict property to hold filters::
        - key -- name of the filter, e.g. SID
        - value -- a primitive representing the filter. e.g. a list of ints.

    Modify the datasource's filters via the set_filter(name, value)
    """

    def init(self, source_id):
        self.id = source_id
        self.filter = {}
        self.cur_event = None

    def set_filter(self, name, value):
        self.filter[name] = value

    @property
    def get_id(self):
        """
        Returns this component id, this is fixed at a class
        level. This should not and cannot be contingent on
        arguments to the init function. Examples:

            - "TradeDataSource"
            - "RandomEquityTrades"
            - "SpecificEquityTrades"

        """
        raise NotImplementedError

    @property
    def get_type(self):
        return COMPONENT_TYPE.SOURCE

    def open(self):
        self.data_socket = self.connect_data()

    def send(self, event):
        """
        Emit data.
        """
        assert isinstance(event, zp.ndict)

        event['source_id'] = self.get_id
        event['type'] = self.get_type

        try:
            ds_frame = self.frame(event)
        except zp.INVALID_DATASOURCE_FRAME as exc:
            return self.signal_exception(exc)

        self.data_socket.send(ds_frame)

    def frame(self, event):
        return zp.DATASOURCE_FRAME(event)
