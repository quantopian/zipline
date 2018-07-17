from zipline.pipeline.data import BoundColumn, DataSet


class PipelineDispatcher(object):
    """Helper class for building a dispatching function for a PipelineLoader.

    Parameters
    ----------
    loaders : dict[BoundColumn or DataSet -> PipelineLoader]
        Map from columns or datasets to pipeline loader for those objects.
    """
    def __init__(self, loaders):
        self._column_loaders = {}
        for data, pl in loaders.items():
            if isinstance(data, BoundColumn):
                    self._column_loaders[data] = pl
            elif issubclass(data, DataSet):
                for c in data.columns:
                    self._column_loaders[c] = pl
            else:
                raise TypeError("%s is neither a BoundColumn "
                                "nor a DataSet" % data)

    def __call__(self, column):
        if column in self._column_loaders:
            return self._column_loaders[column]
        else:
            raise LookupError("No pipeline loader registered for %s" % column)
