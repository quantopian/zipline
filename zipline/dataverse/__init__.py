import pandas as pd

from .dataverse import BaseDataverse
from . import backtest

BacktestDataverse = backtest.BacktestDataverse
Dataverse = BacktestDataverse
Dataverse = BaseDataverse


class ProxyDataverse(object):
    """
    ProxyDataverse waits until source is set to determine which Dataverse
    is used. Currently is only recognizes pandas frames/panels as being
    BacktestDataverse compatible.

    Note, this is a bit wonky since the data we are using is not known
    until run is called. Algorithm is in need of a good refactor.
    """

    def __init__(self):
        self.dataverse = BaseDataverse()

    def get_source(self, source, overwrite_sim_params=True):
        # note, we default to BaseDataverse because things like simfactory
        # call set_sources instead of run.
        if isinstance(source, (pd.Panel, pd.DataFrame)):
            self.dataverse = BacktestDataverse()

        return self.dataverse.get_source(
            source, overwrite_sim_params=overwrite_sim_params
        )

    def get_history_container(self, *args, **kwargs):
        # Special case this method call since it is accessed in Algorithm
        # init before dataverse is set.
        return self.dataverse.get_history_container(*args, **kwargs)

    def __getattr__(self, name):
        if hasattr(self.dataverse, name):
            return getattr(self.dataverse, name)
        raise AttributeError()


__all__ = [
    'BaseDataverse',
    'BacktestDataverse',
    'Dataverse',
    'ProxyDataverse'
]
