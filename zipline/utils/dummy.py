
class UnexpectedAttributeAssignment(Exception):
    pass


class DummyMapping(object):
    """
    Dummy object used to provide a mapping interface for singular values.
    """
    def __init__(self, value):
        self._value = value

    def __getitem__(self, key):
        return self._value


class DummyPortfolio(object):
    """
    Dummy portfolio object which raises an exception when attempting to set any
    of its attributes.
    """
    def __setattr__(self, name, value):
        raise UnexpectedAttributeAssignment(
            'Setting new values to portfolio attributes is not allowed.'
        )
