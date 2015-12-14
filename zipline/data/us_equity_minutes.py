# Copyright 2015 Quantopian, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import bcolz
import json
import os
import pandas as pd

MINUTES_PER_DAY = 390

METADATA_FILENAME = 'metadata.json'


class BcolzMinuteBarReader(object):

    def __init__(self, rootdir, sid_path_func=None):
        self.rootdir = rootdir

        metadata = self._get_metadata()

        self.first_trading_day = pd.Timestamp(
            metadata['first_trading_day'], tz='UTC')

        self._sid_path_func = sid_path_func

        self._carrays = {
            'open': {},
            'high': {},
            'low': {},
            'close': {},
            'volume': {},
            'sid': {},
            'dt': {},
        }

    def _get_metadata(self):
        with open(os.path.join(self.rootdir, METADATA_FILENAME)) as fp:
            return json.load(fp)

    def _get_ctable(self, asset):
        sid = int(asset)
        if self._sid_path_func is not None:
            path = self._sid_path_func(self.rootdir, sid)
        else:
            path = "{0}/{1}.bcolz".format(self.rootdir, sid)

        return bcolz.open(path, mode='r')

    def _find_position_of_minute(self, minute_dt):
        """
        Internal method that returns the position of the given minute in the
        list of every trading minute since market open of the first trading
        day.

        IMPORTANT: This method assumes every day is 390 minutes long, even
        early closes.  Our minute bcolz files are generated like this to
        support fast lookup.

        ex. this method would return 2 for 1/2/2002 9:32 AM Eastern, if
        1/2/2002 is the first trading day of the dataset.

        Parameters
        ----------
        minute_dt: pd.Timestamp
            The minute whose position should be calculated.

        Returns
        -------
        The position of the given minute in the list of all trading minutes
        since market open on the first trading day.
        """
        NotImplementedError

    def _open_minute_file(self, field, asset):
        sid_str = str(int(asset))

        try:
            carray = self._carrays[field][sid_str]
        except KeyError:
            carray = self._carrays[field][sid_str] = \
                self._get_ctable(asset)[field]

        return carray
