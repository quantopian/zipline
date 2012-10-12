from zipline import ndict
# ---------------------
# Error Messages.
# User facing.
# ---------------------

ERRORS = ndict({

    # Raised if a user script calls the override_slippage magic
    # with a slipage object that isn't a VolumeShareSlippage or
    # FixedSlipapge
    'UNSUPPORTED_SLIPPAGE_MODEL':
"You attempted to override slippage with an unsupported class. \
Please use VolumeShareSlippage or FixedSlippage.",

    # Raised if a users script calls override_slippage magic
    # after the initialize method has returned.
    'OVERRIDE_SLIPPAGE_POST_INIT':
"You attempted to override slippage after the simulation has \
started. You may only call override_slippage in your initialize \
method.",

    # Raised if a user script calls the override_commission magic
    # with a commission object that isn't a PerShare or
    # PerTrade commission
    'UNSUPPORTED_COMMISSION_MODEL':
"You attempted to override commission with an unsupported class. \
Please use PerShare or PerTrade.",

    # Raised if a users script calls override_commission magic
    # after the initialize method has returned.
    'OVERRIDE_COMMISSION_POST_INIT':
"You attempted to override commission after the simulation has \
started. You may only call override_commission in your initialize \
method.",


})
