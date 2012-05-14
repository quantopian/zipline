import zipline.protocol as zp
from zipline.protocol import CONTROL_PROTOCOL, COMPONENT_TYPE, \
    COMPONENT_STATE, CONTROL_FRAME, CONTROL_UNFRAME

class PassthroughTransform(BaseTransform):
    """
    A bypass transform which is also an identity transform::

            +-------+
        +---|   f   |--->
            +-------+
        +------id------->

    """

    def __init__(self, **kwargs):
        BaseTransform.__init__(self, "PASSTHROUGH")
        self.init(**kwargs)

    def init(self, **kwargs):
        pass

    @property
    def get_type(self):
        return COMPONENT_TYPE.CONDUIT

    #TODO, could save some cycles by skipping the _UNFRAME call and just setting value to original msg string.
    def transform(self, event):
        return {'name':zp.TRANSFORM_TYPE.PASSTHROUGH, 'value': zp.FEED_FRAME(event) }
