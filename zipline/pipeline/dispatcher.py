from zipline.pipeline.data import BoundColumn, DataSet
from zipline.pipeline.loaders.base import PipelineLoader
from zipline.utils.compat import mappingproxy


class PipelineDispatcher(object):
    """Helper class for building a dispatching function for a PipelineLoader.

    Parameters
    ----------
    column_loaders : dict[BoundColumn -> PipelineLoader]
        Map from columns to pipeline loader for those columns.
    dataset_loaders : dict[DataSet -> PipelineLoader]
        Map from datasets to pipeline loader for those datasets.
    """
    def __init__(self, column_loaders=None, dataset_loaders=None):
        self._column_loaders = column_loaders if column_loaders \
                                                 is not None else {}
        self.column_loaders = mappingproxy(self._column_loaders)
        if dataset_loaders is not None:
            for dataset, pl in dataset_loaders:
                self.register(dataset, pl)

    def __call__(self, column):
        if column in self._column_loaders:
            return self._column_loaders[column]
        else:
            raise LookupError("No pipeline loader registered for %s", column)

    def register(self, data, pl):
        """Register a given PipelineLoader to a column or columns of a dataset

        Parameters
        ----------
        data : BoundColumn or DataSet
            The column or dataset for which to register the PipelineLoader
        pl : PipelineLoader
            The PipelineLoader to register for the column or dataset columns
        """
        assert isinstance(pl, PipelineLoader)

        # make it so that in either case nothing will happen if the column is
        # already registered, allowing users to register their own loaders
        # early on in extensions
        if isinstance(data, BoundColumn):
            if data not in self._column_loaders:
                self._column_loaders[data] = pl
        elif issubclass(data, DataSet):
            for c in data.columns:
                if c not in self._column_loaders:
                    self._column_loaders[c] = pl
        else:
            raise TypeError("Data provided is neither a BoundColumn "
                            "nor a DataSet")


global_pipeline_dispatcher = PipelineDispatcher()
register_pipeline_loader = global_pipeline_dispatcher.register
