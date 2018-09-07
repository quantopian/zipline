"""
This module defines the interface and implementations of Pipeline domains.

A domain represents a set of labels for the arrays computed by a Pipeline.
Currently, this means that a domain defines two things:

1. A calendar defining the dates to which the pipeline's inputs and outputs
   should be aligned. The calendar is represented concretely by a pandas
   DatetimeIndex.

2. The set of assets that the pipeline should compute over. Right now, the only
   supported way of representing this set is with a two-character country code
   describing the country of assets over which the pipeline should compute. In
   the future, we expect to expand this functionality to include more general
   concepts.
"""
import datetime
from textwrap import dedent

from interface import implements, Interface
import pandas as pd
import pytz

from trading_calendars import get_calendar

from zipline.country import CountryCode
from zipline.utils.input_validation import expect_types
from zipline.utils.memoize import lazyval
from zipline.utils.pandas_utils import days_at_time


class IDomain(Interface):
    """Domain interface.
    """
    def all_sessions(self):
        """
        Get all trading sessions for the calendar of this domain.

        This determines the row labels of Pipeline outputs for pipelines run on
        this domain.

        Returns
        -------
        sessions : pd.DatetimeIndex
            An array of all session labels for this domain.
        """

    @property
    def country_code(self):
        """The country code for this domain.

        Returns
        -------
        code : str
            The two-character country iso3166 country code for this domain.
        """

    def data_query_cutoff_for_sessions(self, sessions):
        """Compute the data query cutoff time for the given sessions.

        Parameters
        ----------
        sessions : pd.DatetimeIndex
            The sessions to get the data query cutoff times for. This index
            will contain all midnight UTC values.

        Returns
        -------
        data_query_cutoff : pd.DatetimeIndex
            The sessions at the given data query time.
        """


Domain = implements(IDomain)
Domain.__doc__ = """
A domain represents a set of labels for the arrays computed by a Pipeline.

A domain defines two things:

1. A calendar defining the dates to which the pipeline's inputs and outputs
   should be aligned. The calendar is represented concretely by a pandas
   DatetimeIndex.

2. The set of assets that the pipeline should compute over. Right now, the only
   supported way of representing this set is with a two-character country code
   describing the country of assets over which the pipeline should compute. In
   the future, we expect to expand this functionality to include more general
   concepts.
"""
Domain.__name__ = "Domain"
Domain.__qualname__ = "zipline.pipeline.domain.Domain"


class GenericDomain(Domain):
    """Special singleton class used to represent generic DataSets and Columns.
    """
    def all_sessions(self):
        raise NotImplementedError("Can't get sessions for generic domain.")

    @property
    def country_code(self):
        raise NotImplementedError("Can't get country code for generic domain.")

    def data_query_cutoff_for_sessions(self, sessions):
        raise NotImplementedError(
            "Can't compute data query cutoff times for generic domain.",
        )

    def __repr__(self):
        return "GENERIC"


GENERIC = GenericDomain()


class EquityCalendarDomain(Domain):
    """
    An equity domain whose sessions are defined by a named TradingCalendar.

    Parameters
    ----------
    country_code : str
        ISO-3166 two-letter country code of the domain
    calendar_name : str
        Name of the calendar, to be looked by by trading_calendar.get_calendar.
    data_query_offset : datetime.timedelta
         The offset from market open when data should no longer be considered
         available for a session. For example, a ``data_query_offset`` of
         ``-datetime.timedelta(minutes=45)`` means that the data must have
         been available at least 45 minutes prior to market open for it to
         appear in the pipeline input for the given session.
    """
    @expect_types(
        country_code=str,
        calendar_name=str,
        __funcname='EquityCountryDomain',
    )
    def __init__(self,
                 country_code,
                 calendar_name,
                 data_query_offset=-datetime.timedelta(minutes=45)):
        self._country_code = country_code
        self._calendar_name = calendar_name
        self._data_query_offset = (
            # add one minute because `open_time` is actually the open minute
            # label which is one minute _after_ market open...
            data_query_offset - datetime.timedelta(minutes=1)
        )
        if data_query_offset >= datetime.timedelta(0):
            raise ValueError(
                'data must be ready before market open (offset must be < 0)',
            )

    @property
    def country_code(self):
        return self._country_code

    @lazyval
    def calendar(self):
        return get_calendar(self._calendar_name)

    def all_sessions(self):
        return self.calendar.all_sessions

    def data_query_cutoff_for_sessions(self, sessions):
        opens = self.calendar.opens.loc[sessions]
        missing_mask = pd.isnull(opens)
        if missing_mask.any():
            missing_days = sessions[missing_mask.values]
            raise ValueError(
                'cannot resolve data query time for sessions that are not on'
                ' the %s calendar:\n%s' % (
                    self.calendar.name,
                    missing_days,
                ),
            )

        return pd.DatetimeIndex(opens + self._data_query_offset, tz='UTC')

    def __repr__(self):
        return "EquityCalendarDomain({!r}, {!r})".format(
            self.country_code, self._calendar_name,
        )


US_EQUITIES = EquityCalendarDomain(CountryCode.UNITED_STATES, 'NYSE')
CA_EQUITIES = EquityCalendarDomain(CountryCode.CANADA, 'TSX')
GB_EQUITIES = EquityCalendarDomain(CountryCode.UNITED_KINGDOM, 'LSE')


def infer_domain(terms):
    """
    Infer the domain from a collection of terms.

    The algorithm for inferring domains is as follows:

    - If all input terms have a domain of GENERIC, the result is GENERIC.

    - If there is exactly one non-generic domain in the input terms, the result
      is that domain.

    - Otherwise, an AmbiguousDomain error is raised.

    Parameters
    ----------
    terms : iterable[zipline.pipeline.term.Term]

    Returns
    -------
    inferred : Domain or NotSpecified

    Raises
    ------
    AmbiguousDomain
        Raised if more than one concrete domain is present in the input terms.
    """
    domains = {t.domain for t in terms}
    num_domains = len(domains)

    if num_domains == 0:
        return GENERIC
    elif num_domains == 1:
        return domains.pop()
    elif num_domains == 2 and GENERIC in domains:
        domains.remove(GENERIC)
        return domains.pop()
    else:
        # Remove GENERIC if it's present before raising. Showing it to the user
        # is confusing because it doesn't contribute to the error.
        domains.discard(GENERIC)
        raise AmbiguousDomain(sorted(domains, key=repr))


def bulleted_list(items):
    """Format a bulleted list of values.
    """
    return "\n".join(map("  - {}".format, items))


# This would be better if we provided more context for which domains came from
# which terms.
class AmbiguousDomain(Exception):
    """
    Raised when we attempt to infer a domain from a collection of mixed terms.
    """
    _TEMPLATE = dedent(
        """\
        Found terms with conflicting domains:
        {domains}"""
    )

    def __init__(self, domains):
        self.domains = domains

    def __str__(self):
        return self._TEMPLATE.format(domains=bulleted_list(self.domains))


class EquitySessionDomain(Domain):
    """A domain built directly from an index of sessions.

    Mostly useful for testing.

    Parameters
    ----------
    sessions : pd.DatetimeIndex
        Sessions to use as output labels for pipelines run on this domain.
    country_code : str
        ISO 3166 country code of equities to be used with this domain.
    data_query_time : datetime.time, optional
        The time of day when data should no longer be considered available for
        a session.
    data_query_date_offset : int, optional
        The number of days to add to the session label before applying the
        ``data_query_time``. This can be used to express that the cutoff time
        for a session falls on a different calendar day from the session label.
    """
    @expect_types(
        sessions=pd.DatetimeIndex,
        country_code=str,
        __funcname='EquitySessionDomain',
    )
    def __init__(self,
                 sessions,
                 country_code,
                 data_query_time=None,
                 data_query_date_offset=0):
        self._country_code = country_code
        self._sessions = sessions

        if data_query_time is None:
            data_query_time = datetime.time(0, 0, tzinfo=pytz.timezone('UTC'))
        self._data_query_time = data_query_time
        self._data_query_date_offset = data_query_date_offset

    @property
    def country_code(self):
        return self._country_code

    def all_sessions(self):
        return self._sessions

    def data_query_cutoff_for_sessions(self, sessions):
        return days_at_time(
            sessions,
            self._data_query_time,
            self._data_query_time.tzinfo or 'UTC',
            self._data_query_date_offset,
        )
