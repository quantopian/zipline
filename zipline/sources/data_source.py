from abc import (
    ABCMeta,
    abstractproperty
)

from six import with_metaclass

from zipline.protocol import DATASOURCE_TYPE
from zipline.protocol import Event


class DataSource(with_metaclass(ABCMeta)):

    @property
    def event_type(self):
        return DATASOURCE_TYPE.TRADE

    @property
    def mapping(self):
        """
        Mappings of the form:
        target_key: (mapping_function, source_key)
        """
        return {}

    @abstractproperty
    def raw_data(self):
        """
        An iterator that yields the raw datasource,
        in chronological order of data, one event at a time.
        """
        NotImplemented

    @abstractproperty
    def instance_hash(self):
        """
        A hash that represents the unique args to the source.
        """
        pass

    def get_hash(self):
        return self.__class__.__name__ + "-" + self.instance_hash

    def apply_mapping(self, raw_row):
        """
        Override this to hand craft conversion of row.
        """
        row = {}
        row.update({'type': self.event_type})
        row.update({target: mapping_func(raw_row[source_key])
                    for target, (mapping_func, source_key)
                    in self.mapping.items()})
        row.update({'source_id': self.get_hash()})
        return row

    @property
    def mapped_data(self):
        for row in self.raw_data:
            yield Event(self.apply_mapping(row))

    def __iter__(self):
        return self

    def next(self):
        return self.mapped_data.next()

    def __next__(self):
        return next(self.mapped_data)
