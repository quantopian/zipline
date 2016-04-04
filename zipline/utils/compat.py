from six import PY2


if PY2:
    from functools32 import lru_cache  # noqa
else:
    from functools import lru_cache  # noqa
