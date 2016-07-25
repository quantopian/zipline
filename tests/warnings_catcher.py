from warnings import catch_warnings, WarningMessage


class WarningsCatcher(catch_warnings):
    """
    Subclass of warnings.catch_warnings that takes a list of warning types to
    ignore.
    """
    def __init__(self, types_to_ignore=None):
        super(WarningsCatcher, self).__init__(record=True)

        self._types_to_ignore = set(types_to_ignore or [])

    def __enter__(self):
        if self._entered:
            raise RuntimeError("Cannot enter %r twice" % self)
        self._entered = True
        self._filters = self._module.filters
        self._module.filters = self._filters[:]
        self._showwarning = self._module.showwarning
        if self._record:
            log = []

            def showwarning(*args, **kwargs):
                if args[1] in self._types_to_ignore:
                    return
                log.append(WarningMessage(*args, **kwargs))

            self._module.showwarning = showwarning
            return log
        else:
            return None
