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


from zipline.sources.data_source import DataSource
import pandas as pd

from zipline.gens.utils import hash_args

from sqlalchemy.orm import sessionmaker
from sqlalchemy import func


class SqlSource(DataSource):

    def __init__(self, engine, object_orm, **kwargs):

        self.engine = engine
        self.object_orm = object_orm
        Session = sessionmaker(bind=engine)
        self.session = Session()

        sid_query = self.session.query(object_orm.sid).distinct().all()
        sids = [int(i.sid) for i in sid_query]
        self.sids = kwargs.get('sids', sids)

        qry = self.session.query(func.min(object_orm.index).label('start'),
                                 func.max(object_orm.index).label('end'))
        res = qry.one()
        self.start = pd.Timestamp(res.start).tz_localize('UTC')
        self.end = pd.Timestamp(res.end).tz_localize('UTC')

        # Hash_value for downstream sorting.
        self.arg_string = hash_args(engine, object_orm, **kwargs)

        self._raw_data = None

    @property
    def mapping(self):
        return {
            'dt': (lambda x: x, 'dt'),
            'sid': (lambda x: x, 'sid'),
            'price': (float, 'price'),
            'volume': (int, 'volume'),
        }

    @property
    def instance_hash(self):
        return self.arg_string

    def raw_data_gen(self):
        for sid in self.sids:
            qry = self.session.query(self.object_orm).filter_by(sid=sid)
            for row in qry:
                event = {
                    'dt': pd.Timestamp(row.index).tz_localize('UTC'),
                    'sid': row.sid,
                    'price': row.price,
                    # Just chose something large
                    # if no volume available.
                    'volume': 1e9
                    }
                yield event

    @property
    def raw_data(self):
        if not self._raw_data:
            self._raw_data = self.raw_data_gen()
        return self._raw_data
