from functools import partial

from zipline.utils.compat import mappingproxy


def _make_metrics_set_core():
    """Create a family of metrics sets functions that read from the same
    metrics set mapping.

    Returns
    -------
    metrics_sets : mappingproxy
        The mapping of metrics sets to load functions.
    register : callable
        The function which registers new metrics sets in the ``metrics_sets``
        mapping.
    unregister : callable
        The function which deregisters metrics sets from the ``metrics_sets``
        mapping.
    load : callable
        The function which loads the ingested metrics sets back into memory.
    """
    _metrics_sets = {}
    # Expose _metrics_sets through a proxy so that users cannot mutate this
    # accidentally. Users may go through `register` to update this which will
    # warn when trampling another metrics set.
    metrics_sets = mappingproxy(_metrics_sets)

    def register(name, function=None):
        """Register a new metrics set.

        Parameters
        ----------
        name : str
            The name of the metrics set
        function : callable
            The callable which produces the metrics set.

        Notes
        -----
        This may be used as a decorator if only ``name`` is passed.

        See Also
        --------
        zipline.finance.metrics.get_metrics_set
        zipline.finance.metrics.unregister_metrics_set
        """
        if function is None:
            # allow as decorator with just name.
            return partial(register, name)

        if name in _metrics_sets:
            raise ValueError("metrics set %r is already registered" % name)

        _metrics_sets[name] = function

        return function

    def unregister(name):
        """Unregister an existing metrics set.

        Parameters
        ----------
        name : str
            The name of the metrics set

        See Also
        --------
        zipline.finance.metrics.register_metrics_set
        """
        try:
            del _metrics_sets[name]
        except KeyError as exc:
            raise ValueError(
                "metrics set %r was not already registered" % name,
            ) from exc

    def load(name):
        """Return an instance of the metrics set registered with the given name.

        Returns
        -------
        metrics : set[Metric]
            A new instance of the metrics set.

        Raises
        ------
        ValueError
            Raised when no metrics set is registered to ``name``
        """
        try:
            function = _metrics_sets[name]
        except KeyError as exc:
            raise ValueError(
                "no metrics set registered as %r, options are: %r"
                % (
                    name,
                    sorted(_metrics_sets),
                ),
            ) from exc

        return function()

    return metrics_sets, register, unregister, load


metrics_sets, register, unregister, load = _make_metrics_set_core()
