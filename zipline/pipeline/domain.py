"""
Domains
-------

TODO_SS
"""
from interface import implements, Interface

from trading_calendars import get_calendar

from zipline.country import CountryCode
from zipline.utils.memoize import lazyval

from .sentinels import NotSpecified


class IDomain(Interface):
    """Domain interface.
    """

    @property
    def name(self):
        """User-facing name for this domain.
        """

    @property
    def country_code(self):
        """Country code for assets on this domain.
        """

    # TODO_SS: The original design for domains was to have them return a
    # TradingCalendar, but we have a bunch of tests in test_blaze that use very
    # short session indices that I don't know how to port to using a proper
    # TradingCalendar.
    #
    # Is there a strong reason to prefer just exposing the calendar
    # vs. exposing the sessions? If so, what do we do about the blaze tests?
    def get_sessions(self):
        """Get all trading sessions for the calendar of this domain.
        """


# Create a base class so that we can do type-based validation of user input.
Domain = implements(IDomain)
Domain.__name__ = 'Domain'


# TODO: Better name for this?
# TODO: Do we want/need memoization for this?
class StandardDomain(Domain):
    """TODO_SS
    """
    def __init__(self, name, country_code, calendar_name):
        self._name = name
        self._country_code = country_code
        self._calendar_name = calendar_name

    @property
    def name(self):
        return self._name

    @property
    def country_code(self):
        return self._country_code

    def get_sessions(self):
        return self.calendar.all_sessions

    @lazyval
    def calendar(self):
        return get_calendar(self._calendar_name)

    def __repr__(self):
        return "{}(country={!r}, calendar={!r})".format(
            self.name,
            self.country_code,
            self._calendar_name,
        )


# TODO: Is this the casing convention we want for domains?
USEquities = StandardDomain('USEquities', CountryCode.UNITED_STATES, 'NYSE')
CanadaEquities = StandardDomain('CanadaEquities', CountryCode.CANADA, 'TSX')
# XXX: The actual country code for this is GB. Should we use that for the name
# here?
UKEquities = StandardDomain('UKEquities', CountryCode.UNITED_KINGDOM, 'LSE')


def infer_domain(terms):
    """
    Infer the domain from a collection of terms.

    The algorithm for inferring domains is as follows:

    - If all input terms have a domain of NotSpecified, the result is also
      NotSpecified.

    - If there is exactly one non-NotSpecified domain in the input terms, the
      result is that domain.

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
    domains = {NotSpecified}
    for t in terms:
        domains.update(t.domain)

    if len(domains) == 1:
        return NotSpecified
    elif len(domains) == 2:
        domains.remove(NotSpecified)
        return domains.pop()
    else:
        domains.remove(NotSpecified)
        raise AmbiguousDomain(sorted(domains, key=lambda d: d.country_code))


class AmbiguousDomain(Exception):
    """
    Raised when we attempt to infer a domain from a collection of mixed terms.
    """


class SessionDomain(Domain):
    """TODO_SS
    """

    def __init__(self, name, sessions, country_code):
        self._name = name
        self._country_code = country_code
        self._sessions = sessions

    @property
    def name(self):
        return self._name

    @property
    def country_code(self):
        return self._country_code

    def get_sessions(self):
        return self._sessions


class ExplicitCalendarDomain(Domain):
    """
    A domain that takes an explicit calendar instance at construction time
    rather than a name. This allows for greater control of the start/end dates
    of the calendar, which is sometimes useful for testing.

    TODO_SS:
    """
    def __init__(self, name, country_code, calendar):
        self._name = name
        self._country_code = country_code
        self._calendar = calendar

    @property
    def name(self):
        return self._name

    @property
    def country_code(self):
        return self._country_code

    def get_sessions(self):
        return self._calendar.sessions
