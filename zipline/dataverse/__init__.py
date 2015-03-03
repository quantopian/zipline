from .dataverse import BaseDataverse
from .backtest import BacktestDataverse

Dataverse = BaseDataverse
Dataverse = BacktestDataverse

__all__ = [
    'Dataverse',
]
