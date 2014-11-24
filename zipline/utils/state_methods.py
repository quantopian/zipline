def _defaultdict_list_get_state(d):
    return {
        '__defaultdict_list__': True,
        'as_dict': dict(d)
    }


def _defaultdict_ordered_get_state(d):
    return {
        '__defaultdict_ordered__': True,
        'as_dict': dict(d)
    }


def _positiondict_get_state(d):
    return {
        '__positiondict__': True,
        'as_dict': dict(d)
    }


def _positions_get_state(d):
    return {
        '__Positions__': True,
        'as_dict': dict(d)
    }
