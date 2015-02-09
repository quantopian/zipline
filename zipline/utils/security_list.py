import os.path
import pytz
import pandas as pd
from datetime import datetime
from os import listdir

DATE_FORMAT = "%Y%m%d"
import zipline
zipline_dir = os.path.join(*zipline.__path__)
SECURITY_LISTS_DIR = os.path.join(zipline_dir, 'resources', 'security_lists')


def loopback(symbol, *args, **kwargs):
    return symbol


class SecurityList(object):

    def __init__(self, lookup_func, data, current_date_func):
        """
        lookup_func: function that takes a string symbol and a date and
        returns a Security object.
        data: a nested dictionary:
            knowledge_date -> lookup_date ->
              {add: [symbol list], 'delete': []}, delete: [symbol list]}
        current_date_func: function taking no parameters, returning
            current datetime
        """
        self.lookup_func = lookup_func
        self.data = data
        self._cache = {}
        self._knowledge_dates = self.make_knowledge_dates(self.data)
        self.current_date = current_date_func
        self.count = 0
        self._current_set = set()

    def make_knowledge_dates(self, data):
        knowledge_dates = sorted(
            [pd.Timestamp(k) for k in data.keys()])
        return knowledge_dates

    def __iter__(self):
        return iter(self.restricted_list)

    def __contains__(self, item):
        return item in self.restricted_list

    @property
    def restricted_list(self):

        cd = self.current_date()
        for kd in self._knowledge_dates:
            if cd < kd:
                break
            if kd in self._cache:
                self._current_set = self._cache[kd]
                continue

            for effective_date, changes in iter(self.data[kd].items()):
                self.update_current(
                    effective_date,
                    changes['add'],
                    self._current_set.add
                )

                self.update_current(
                    effective_date,
                    changes['delete'],
                    self._current_set.remove
                )

            self._cache[kd] = self._current_set
        return self._current_set

    def update_current(self, effective_date, symbols, change_func):
        for symbol in symbols:
            sid = self.lookup_func(
                symbol,
                as_of_date=effective_date
            )
            change_func(sid)


class SecurityListSet(object):

    def __init__(self, current_date_func, lookup_func=None):
        # provide a cut point to substitute other security
        # list implementations.
        self.sl_constructor = SecurityList
        if lookup_func is None:
            self.lookup_func = loopback
        else:
            self.lookup_func = lookup_func
        self.current_date_func = current_date_func
        self._leveraged_etf = None

    @property
    def leveraged_etf_list(self):
        if self._leveraged_etf is None:
            self._leveraged_etf = self.sl_constructor(
                self.lookup_func,
                load_from_directory('leveraged_etf_list'),
                self.current_date_func
            )
        return self._leveraged_etf


def load_from_directory(list_name):
    """
    To resolve the symbol in the LEVERAGED_ETF list,
    the date on which the symbol was in effect is needed.

    Furthermore, to maintain a point in time record of our own maintenance
    of the restricted list, we need a knowledge date. Thus, restricted lists
    are dictionaries of datetime->symbol lists.
    new symbols should be entered as a new knowledge date entry.

    This method assumes a directory structure of:
    SECURITY_LISTS_DIR/listname/knowledge_date/lookup_date/add.txt
    SECURITY_LISTS_DIR/listname/knowledge_date/lookup_date/delete.txt

    The return value is a dictionary with:
    knowledge_date -> lookup_date ->
       {add: [symbol list], 'delete': [symbol list]}
    """
    data = {}
    dir_path = os.path.join(SECURITY_LISTS_DIR, list_name)
    for kd_name in listdir(dir_path):
        kd = datetime.strptime(kd_name, DATE_FORMAT).replace(
            tzinfo=pytz.utc)
        data[kd] = {}
        kd_path = os.path.join(dir_path, kd_name)
        for ld_name in listdir(kd_path):
            ld = datetime.strptime(ld_name, DATE_FORMAT).replace(
                tzinfo=pytz.utc)
            data[kd][ld] = {}
            ld_path = os.path.join(kd_path, ld_name)
            for fname in listdir(ld_path):
                fpath = os.path.join(ld_path, fname)
                with open(fpath) as f:
                    symbols = f.read().splitlines()
                    data[kd][ld][fname] = symbols

    return data
