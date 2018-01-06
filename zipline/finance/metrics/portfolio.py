from zipline.protocol import Portfolio


class PortfolioFieldMetric(object):
    """A metric which just tracks a field of the
    :class:`~zipline.protocol.Portfolio`.
    """
    def __init__(self, field):
        self._field = field

    def start_of_simulation(self, first_session, last_session, capital_base):
        self._capital_base = capital_base
