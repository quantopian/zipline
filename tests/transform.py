from zipline.messaging import BaseTransform
from zipline.protocol import COMPONENT_TYPE

class DivideByZeroTransform(BaseTransform):
    """
    A transform that fails.
    """

    def __init__(self, name):
        BaseTransform.__init__(self, "PASSTHROUGH")
        self.state['name'] = name
        self.init()

    def init(self):
        pass

    @property
    def get_type(self):
        return COMPONENT_TYPE.CONDUIT

    def transform(self, event):
        return { 'value': 0/0 }
