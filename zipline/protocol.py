from utils.protocol_utils import Enum, ndict, namelookup

# Datasource type should completely determine the other fields of a
# message with its type.
DATASOURCE_TYPE = Enum(
    'AS_TRADED_EQUITY',
    'MERGER',
    'SPLIT',
    'DIVIDEND',
    'TRADE',
    'EMPTY',
    'DONE'
)


#Transform type needs to be a ndict to facilitate merging.
TRANSFORM_TYPE = ndict({
    'PASSTHROUGH' : 'PASSTHROUGH',
    'EMPTY'       : ''
})


FINANCE_COMPONENT = namelookup({
    'TRADING_CLIENT'   : 'TRADING_CLIENT',
    'PORTFOLIO_CLIENT' : 'PORTFOLIO_CLIENT',
})


# the simulation style enumerates the available transaction simulation
# strategies.
SIMULATION_STYLE  = Enum(
    'PARTIAL_VOLUME',
    'BUY_ALL',
    'FIXED_SLIPPAGE',
    'NOOP'
)
