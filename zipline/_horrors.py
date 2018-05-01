"""
Quarantine area for legacy code that we can't yet kill, but don't want
to have to reason about when reading code on a day to day basis.
"""
from six import exec_


def _XXX_normalize_algorithm_namespace(namespace,
                                       algo_source,
                                       algo_filename,
                                       initialize,
                                       handle_data,
                                       before_trading_start,
                                       analyze):
    """
    """
    if namespace is None:
        out = {}
    else:
        out = namespace.copy()
    del namespace  # prevent accidental aliasing bugs.

    if algo_source is not None:
        exec_(algo_source, out)

    for name, func in (('initialize', initialize),
                       ('handle_data', handle_data),
                       ('before_trading_start', before_trading_start),
                       ('analyze', analyze)):
        if func is not None:
            if name in out:
                raise ValueError("{} specified more than once!", name)
            out[name] = func
        else:
            out.setdefault(name, noop)

    return out


def noop(*args, **kwargs):
    pass
