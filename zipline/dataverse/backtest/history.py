from zipline.history.history_container import (
    HistoryContainer,
    HistoryContainerDelta
)
from zipline.utils.munge import ffill


class HistoryFrame(object):
    def __init__(self):
        pass


class HistoryChunker(object):
    """
    Purpose of this is to break up dataverse into chunks for memory and
    saner computation during non-static universes

    We don't assume that the history calls will be done on every dt, which
    is a case that could be further optimized by just yielding sequential
    frames.

    However, we do get get_loc speed ups by search a smaller subset. We can
    also walk the minimum search space since dt will also be increasing.

    If we treat the entire dataset as one big chunk, this should be the
    same as the original BacktestHistoryContainer. Outside of optimizations
    that could have been applied to both.
    """
    def __init__(self, values, index, window, ffill=True, last_values=None):
        self.values = values
        self.index = index
        self.window = window
        self.ffill = ffill
        self.last_values = last_values

    def __iter__(self):
        """
        Coroutine to generate ndarray slices.
        """
        values = self.values
        index = self.index
        get_loc = index.get_loc
        window = self.window

        if self.ffill:
            values = ffill(values)

        vals = None
        while True:
            dt = (yield vals)
            # TODO walk a starting loc_index to limit search space.
            # in most cases the next loc will be right after the last one
            loc = get_loc(dt)
            start = max(loc - window, 0)
            sl = slice(start, loc)
            vals = values[sl]


class BacktestHistoryContainer(HistoryContainer):
    def __init__(self, *args, **kwargs):
        self.dataverse = kwargs.pop('dataverse')
        # need to make this cache easier to update
        # maybe move to chunking?
        source = self.dataverse.raw_source
        values = source.values
        self.values = values
        self.source = source
        self.index = source.major_axis
        self.container_cache = {}
        self.chunker_cache = {}
        self.chunk_size = kwargs.pop('chunksize', 390)

    def get_history(self, history_spec, algo_dt):
        try:
            data = self.container_cache[history_spec]
            vals = self.grab_chunk(history_spec, algo_dt)
            block = data._data.blocks[0]
            if vals.flags.f_contiguous != block.values.flags.f_contiguous:
                vals = vals.T
            data._data.blocks[0].values = vals
            data._item_cache.clear()
        except KeyError:
            # quick and dirty way to get that initial DataFrame
            loc = self.index.get_loc(algo_dt)
            start = max(loc - history_spec.bar_count, 0)
            sl = slice(start, loc)
            data = self.source.ix[:, sl, history_spec.field]
            self.container_cache[history_spec] = data
        return data

    def grab_chunk(self, history_spec, algo_dt):
        try:
            chunker = self.chunker_cache[history_spec]
            return chunker.send(algo_dt)
        except KeyError:
            # TODO have chunker send better exaustion error
            # either no chunker or chunker exausted
            loc = self.index.get_loc(algo_dt)
            end = min(loc + self.chunk_size, len(self.index))
            sl = slice(loc, end)

            minor_loc = self.source.minor_axis.get_loc(history_spec.field)
            values = self.values[:, sl, minor_loc]
            index = self.index[sl]
            window = history_spec.bar_count
            ffill = history_spec.ffill
            chunker = HistoryChunker(values, index, window, ffill=ffill)
            chunker = iter(chunker)

            self.chunker_cache[history_spec] = chunker
            chunker.send(None)  # warmup
            return chunker.send(algo_dt)

    def ensure_spec(self, spec, dt, bar_data):
        """
        Ensure that this container has enough space to hold the data for the
        given spec. This returns a HistoryContainerDelta to represent the
        changes in shape that the container made to support the new
        HistorySpec.
        """
        updated = {}
        return HistoryContainerDelta(**updated)

    def update(self, data, algo_dt):
        # no need to update history container
        pass
