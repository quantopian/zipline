from collections import defaultdict

class Returns(object):
    """
    Class that maintains a dictionary from sids to the event
    representing the most recent closing price.
    """
    def __init__(self, days = 1):
        self.days = days
        self.mapping = defaultdict(self._create)

    def update(self, event):
        """
        Update and return the calculated returns for this event's sid.
        """
        sid_returns = self.mapping[event.sid].update(event)
        return sid_returns

    def _create(self):
        return ReturnsFromPriorClose(self.days)

class ReturnsFromPriorClose(object):
    """
    Calculates a security's returns since the previous close, using the
    current price.
    """

    def __init__(self):
        self.last_close = None
        self.last_event = None
        self.returns = 0.0

    def update(self, event):
        if self.last_close:
            change = event.price - self.last_close.price
            self.returns = change / self.last_close.price

        if self.last_event:
            if self.last_event.dt.day != event.dt.day:
                # the current event is from the day after
                # the last event. Therefore the last event was
                # the last close
                self.last_close = self.last_event

        # the current event is now the last_event
        self.last_event = event
