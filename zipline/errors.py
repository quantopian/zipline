class ZiplineError(Exception):
    msg = None

    def __init__(self, *args, **kwargs):
        self.lineno = kwargs.get('lineno', None)
        self.offset = kwargs.get('offset', None)
        self.file = kwargs.get('file', None)

        self.args = args
        self.kwargs = kwargs
        self.message = str(self)

    def __str__(self):
        msg = self.msg.format(**self.kwargs)
        return msg

    __unicode__ = __str__
    __repr__ = __str__


class WrongDataForTransform(ZiplineError):
    """
    Raised whenever a rolling transform is called on an event that
    does not have the necessary properties.
    """
    msg = "{transform} requires {fields}. Event cannot be processed."


class UnsupportedSlippageModel(ZiplineError):
    """
    Raised if a user script calls the override_slippage magic
    with a slipage object that isn't a VolumeShareSlippage or
    FixedSlipapge
    """
    msg = """
You attempted to override slippage with an unsupported class. \
Please use VolumeShareSlippage or FixedSlippage.
""".strip()


class OverrideSlippagePostInit(ZiplineError):
    # Raised if a users script calls override_slippage magic
    # after the initialize method has returned.
    msg = """
You attempted to override slippage after the simulation has \
started. You may only call override_slippage in your initialize \
method.
""".strip()


class UnsupportedCommissionModel(ZiplineError):
    # Raised if a user script calls the override_commission magic
    # with a commission object that isn't a PerShare or
    # PerTrade commission
    msg = """
You attempted to override commission with an unsupported class. \
Please use PerShare or PerTrade.
""".strip()


class OverrideCommissionPostInit(ZiplineError):
    # Raised if a users script calls override_commission magic
    # after the initialize method has returned.
    msg = """
You attempted to override commission after the simulation has \
started. You may only call override_commission in your initialize \
method.
""".strip()
