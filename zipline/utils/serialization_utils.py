# Label for the serialization version field in the state returned by
# __getstate__.
VERSION_LABEL = '_stateversion_'


class SerializeableZiplineObject(object):
    """
    This class implements the basic set and get state methods used for
    serialization. It also serves as a demarkation of which objects we
    serialize.
    """

    def __getstate__(self):
        """
        Many get_state methods need this one line of code.
        This method deduplicates the code calls.
        """
        state_dict = \
            {k: v for k, v in self.__dict__.iteritems()
                if not k.startswith('_')}
        return state_dict

    def __setstate__(self, state):
        """
        Many objects require only this code.
        """
        self.__dict__.update(state)

    # =====================================================
    # These are helper methods for some problem data types.
    # =====================================================

    def _defaultdict_list_get_state(self, d):
        return {
            '__original.type__': 'encoded.defaultdict_list',
            'as_dict': dict(d)
        }

    def _defaultdict_ordered_get_state(self, d):
        return {
            '__original.type__': 'encoded.defaultdict_ordered',
            'as_dict': dict(d)
        }

    def _positiondict_get_state(self, d):
        return {
            '__original.type__': 'encoded.positiondict',
            'as_dict': dict(d)
        }

    def _positions_get_state(self, d):
        return {
            '__original.type__': 'encoded.Positions',
            'as_dict': dict(d)
        }
