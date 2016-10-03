import warnings
from datetime import datetime
from os import listdir
import os.path

import pandas as pd
import pytz
import zipline

from zipline.errors import SymbolNotFound
from zipline.finance.asset_restrictions import SecurityListRestrictions
from zipline.zipline_warnings import ZiplineDeprecationWarning


DATE_FORMAT = "%Y%m%d"
zipline_dir = os.path.dirname(zipline.__file__)
SECURITY_LISTS_DIR = os.path.join(zipline_dir, 'resources', 'security_lists')


class SecurityList(object):

    def __init__(self, data, current_date_func, asset_finder):
        """
        data: a nested dictionary:
            knowledge_date -> lookup_date ->
              {add: [symbol list], 'delete': []}, delete: [symbol list]}
        current_date_func: function taking no parameters, returning
            current datetime
        """
        self.data = data
        self._cache = {}
        self._knowledge_dates = self.make_knowledge_dates(self.data)
        self.current_date = current_date_func
        self.count = 0
        self._current_set = set()
        self.asset_finder = asset_finder

    def make_knowledge_dates(self, data):
        knowledge_dates = sorted(
            [pd.Timestamp(k) for k in data.keys()])
        return knowledge_dates

    def __iter__(self):
        warnings.warn(
            'Iterating over security_lists is deprecated. Use '
            '`for sid in <security_list>.current_securities(dt)` instead.',
            category=ZiplineDeprecationWarning,
            stacklevel=2
        )
        return iter(self.current_securities(self.current_date()))

    def __contains__(self, item):
        warnings.warn(
            'Evaluating inclusion in security_lists is deprecated. Use '
            '`sid in <security_list>.current_securities(dt)` instead.',
            category=ZiplineDeprecationWarning,
            stacklevel=2
        )
        return item in self.current_securities(self.current_date())

    def current_securities(self, dt):
        for kd in self._knowledge_dates:
            if dt < kd:
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
            try:
                asset = self.asset_finder.lookup_symbol(
                    symbol,
                    as_of_date=effective_date
                )
            # Pass if no Asset exists for the symbol
            except SymbolNotFound:
                continue
            change_func(asset.sid)


class SecurityListSet(object):
    # provide a cut point to substitute other security
    # list implementations.
    security_list_type = SecurityList

    def __init__(self, current_date_func, asset_finder):
        self.current_date_func = current_date_func
        self.asset_finder = asset_finder
        self._leveraged_etf = None

    @property
    def leveraged_etf_list(self):
        if self._leveraged_etf is None:
            self._leveraged_etf = self.security_list_type(
                load_from_directory('leveraged_etf_list'),
                self.current_date_func,
                asset_finder=self.asset_finder
            )
        return self._leveraged_etf

    @property
    def restrict_leveraged_etfs(self):
        return SecurityListRestrictions(self.leveraged_etf_list)


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
