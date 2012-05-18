from zipline.transforms import BaseTransform
from zipline.protocol import FEED_FRAME, TRANSFORM_TYPE

class PassthroughTransform(BaseTransform):
    """
    A bypass transform passes data through unchanged.
    """

    def init(self):
        self.state = { 'name': 'PASSTHROUGH' }

    #TODO, could save some cycles by skipping the _UNFRAME call
    # and just setting value to original msg string.
    def transform(self, event):
        return {
            'name'  : TRANSFORM_TYPE.PASSTHROUGH,
            'value' : FEED_FRAME(event)
        }
