import click
import pandas as pd

from .context_tricks import CallbackManager


def maybe_show_progress(it, show_progress, **kwargs):
    """Optionally show a progress bar for the given iterator.

    Parameters
    ----------
    it : iterable
        The underlying iterator.
    show_progress : bool
        Should progress be shown.
    **kwargs
        Forwarded to the click progress bar.

    Returns
    -------
    itercontext : context manager
        A context manager whose enter is the actual iterator to use.

    Examples
    --------
    .. code-block:: python

       with maybe_show_progress([1, 2, 3], True) as ns:
            for n in ns:
                ...
    """
    if show_progress:
        return click.progressbar(it, **kwargs)

    # context manager that just return `it` when we enter it
    return CallbackManager(lambda it=it: it)


class _DatetimeParam(click.ParamType):
    def __init__(self, tz=None):
        self.tz = tz

    def parser(self, value):
        return pd.Timestamp(value, tz=self.tz)

    @property
    def name(self):
        return type(self).__name__.upper()

    def convert(self, value, param, ctx):
        try:
            return self.parser(value)
        except ValueError:
            self.fail(
                '%s is not a valid %s' % (value, self.name.lower()),
                param,
                ctx,
            )


class Timestamp(_DatetimeParam):
    """A click parameter that parses the value into pandas.Timestamp objects.

    Parameters
    ----------
    tz : timezone-coercable, optional
        The timezone to parse the string as.
        By default the timezone will be infered from the string or naiive.
    """


class Date(_DatetimeParam):
    """A click parameter that parses the value into datetime.date objects.

    Parameters
    ----------
    tz : timezone-coercable, optional
        The timezone to parse the string as.
        By default the timezone will be infered from the string or naiive.
    as_timestamp : bool, optional
        If True, return the value as a pd.Timestamp object normalized to
        midnight.
    """
    def __init__(self, tz=None, as_timestamp=False):
        super(Date, self).__init__(tz=tz)
        self.as_timestamp = as_timestamp

    def parser(self, value):
        ts = super(Date, self).parser(value)
        return ts.normalize() if self.as_timestamp else ts.date()


class Time(_DatetimeParam):
    """A click parameter that parses the value into timetime.time objects.

    Parameters
    ----------
    tz : timezone-coercable, optional
        The timezone to parse the string as.
        By default the timezone will be infered from the string or naiive.
    """
    def parser(self, value):
        return super(Time, self).parser(value).time()


class Timedelta(_DatetimeParam):
    """A click parameter that parses values into pd.Timedelta objects.

    Parameters
    ----------
    unit : {'D', 'h', 'm', 's', 'ms', 'us', 'ns'}, optional
        Denotes the unit of the input if the input is an integer.
    """
    def __init__(self, unit='ns'):
        self.unit = unit

    def parser(self, value):
        return pd.Timedelta(value, unit=self.unit)
