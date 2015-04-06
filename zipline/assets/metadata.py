#
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


import pandas as pd
import math
from zipline.errors import ConsumeAssetMetaDataError


class AssetMetaData(object):

    cache = {}
    fields = ("sid",
              "asset_type",
              "symbol",
              "asset_name",
              "start_date",
              "end_date",
              "first_traded",
              "exchange",
              "notice_date",
              "expiration_date",
              "contract_multiplier")

    def __init__(self, data=None):
        if data is not None:
            self.consume_metadata(data)

    def __iter__(self):
        return self.cache.__iter__()

    def _insert_dataframe(self, dataframe):
        for identifier, row in dataframe.iterrows():
            self.insert_metadata(identifier, **row)

    def _insert_dict(self, dict):
        for sid, entry in dict.items():
            self.insert_metadata(sid, entry)

    def read(self):
        return self.cache.items()

    def retrieve_metadata(self, identifier):
        return self.cache.get(identifier)

    def insert_metadata(self, identifier, **kwargs):
        entry = self.retrieve_metadata(identifier)
        if entry is None:
            entry = {}

        for key, value in kwargs.items():
            # Do not accept invalid fields
            if key not in self.fields:
                continue
            # Do not accept Nones
            if value is None:
                continue
            # Do not accept nans from dataframes
            if isinstance(value, float) and math.isnan(value):
                continue
            entry[key] = value

        self.cache[identifier] = entry

    def consume_metadata(self, metadata):
        """
        Consumes the provided metadata in to this AssetMetaData object. The
        existing values in this AssetMetaData will be overwritten when there
        is a conflict.
        :param metadata: The metadata to be consumed
        """
        if isinstance(metadata, AssetMetaData):
            for identifier in metadata:
                self.insert_metadata(identifier)
        elif isinstance(metadata, pd.DataFrame):
            self._insert_dataframe(metadata)
        elif isinstance(metadata, dict):
            self._insert_dict(metadata)
        else:
            raise ConsumeAssetMetaDataError(obj=metadata)

    def consume_data_source(self, source):
        if hasattr(source, 'identifiers'):
            for identifier in source.identifiers:
                if self.retrieve_metadata(identifier) is None:
                    self.insert_metadata(identifier=identifier)