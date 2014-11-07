def _defaultdict_list_get_state(d):
    return {
        '__defaultdict_list__': True,
        'as_dict': dict(d)
    }
