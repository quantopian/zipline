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

from interface import default, implements, Interface
import numpy as np
import pandas as pd
import pytz

from zipline.utils.calendar_utils import get_calendar

from zipline.country import CountryCode
from zipline.utils.formatting import bulleted_list
from zipline.utils.input_validation import expect_types, optional
from zipline.utils.memoize import lazyval
from zipline.utils.pandas_utils import days_at_time


class IDomain(Interface):
    """Domain interface."""

    def sessions(self):
        """Get all trading sessions for the calendar of this domain.

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
            Timestamp of the last minute for which data should be considered
            "available" on each session.
        """

    @default
    def roll_forward(self, dt):
        """Given a date, align it to the calendar of the pipeline's domain.

        Parameters
        ----------
        dt : pd.Timestamp

        Returns
        -------
        pd.Timestamp
        """
        dt = pd.Timestamp(dt)
        trading_days = self.sessions()
        try:
            return trading_days[trading_days.searchsorted(dt)]
        except IndexError as exc:
            raise ValueError(
                f"Date {dt.date()} was past the last session for domain {self}. "
                f"The last session for this domain is {trading_days[-1].date()}."
            ) from exc


Domain = implements(IDomain)
Domain.__doc__ = """A domain represents a set of labels for the arrays computed by a Pipeline.

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
    """Special singleton class used to represent generic DataSets and Columns."""

    def sessions(self):
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
    """An equity domain whose sessions are defined by a named TradingCalendar.

    Parameters
    ----------
    country_code : str
        ISO-3166 two-letter country code of the domain
    calendar_name : str
        Name of the calendar, to be looked by by trading_calendar.get_calendar.
    data_query_offset : np.timedelta64
         The offset from market open when data should no longer be considered
         available for a session. For example, a ``data_query_offset`` of
         ``-np.timedelta64(45, 'm')`` means that the data must have
         been available at least 45 minutes prior to market open for it to
         appear in the pipeline input for the given session.
    """

    @expect_types(
        country_code=str,
        calendar_name=str,
        __funcname="EquityCountryDomain",
    )
    def __init__(
        self, country_code, calendar_name, data_query_offset=-np.timedelta64(45, "m")
    ):
        self._country_code = country_code
        self.calendar_name = calendar_name
        self._data_query_offset = (
            # add one minute because `open_time` is actually the open minute
            # label which is one minute _after_ market open...
            data_query_offset
            - np.timedelta64(1, "m")
        )
        if data_query_offset >= datetime.timedelta(0):
            raise ValueError(
                "data must be ready before market open (offset must be < 0)",
            )

    @property
    def country_code(self):
        return self._country_code

    @lazyval
    def calendar(self):
        return get_calendar(self.calendar_name)

    def sessions(self):
        return self.calendar.sessions

    def data_query_cutoff_for_sessions(self, sessions):
        opens = self.calendar.first_minutes.reindex(sessions)
        missing_mask = pd.isnull(opens)
        if missing_mask.any():
            missing_days = sessions[missing_mask]
            raise ValueError(
                "cannot resolve data query time for sessions that are not on"
                f" the {self.calendar_name} calendar:\n{missing_days}"
            )

        return pd.DatetimeIndex(opens) + self._data_query_offset

    def __repr__(self):
        return "EquityCalendarDomain({!r}, {!r})".format(
            self.country_code,
            self.calendar_name,
        )


AR_EQUITIES = EquityCalendarDomain(CountryCode.ARGENTINA, "XBUE")
AT_EQUITIES = EquityCalendarDomain(CountryCode.AUSTRIA, "XWBO")
AU_EQUITIES = EquityCalendarDomain(CountryCode.AUSTRALIA, "XASX")
BE_EQUITIES = EquityCalendarDomain(CountryCode.BELGIUM, "XBRU")
BR_EQUITIES = EquityCalendarDomain(CountryCode.BRAZIL, "BVMF")
CA_EQUITIES = EquityCalendarDomain(CountryCode.CANADA, "XTSE")
CH_EQUITIES = EquityCalendarDomain(CountryCode.SWITZERLAND, "XSWX")
CL_EQUITIES = EquityCalendarDomain(CountryCode.CHILE, "XSGO")
CN_EQUITIES = EquityCalendarDomain(CountryCode.CHINA, "XSHG")
CO_EQUITIES = EquityCalendarDomain(CountryCode.COLOMBIA, "XBOG")
CZ_EQUITIES = EquityCalendarDomain(CountryCode.CZECH_REPUBLIC, "XPRA")
DE_EQUITIES = EquityCalendarDomain(CountryCode.GERMANY, "XFRA")
DK_EQUITIES = EquityCalendarDomain(CountryCode.DENMARK, "XCSE")
ES_EQUITIES = EquityCalendarDomain(CountryCode.SPAIN, "XMAD")
FI_EQUITIES = EquityCalendarDomain(CountryCode.FINLAND, "XHEL")
FR_EQUITIES = EquityCalendarDomain(CountryCode.FRANCE, "XPAR")
GB_EQUITIES = EquityCalendarDomain(CountryCode.UNITED_KINGDOM, "XLON")
GR_EQUITIES = EquityCalendarDomain(CountryCode.GREECE, "ASEX")
HK_EQUITIES = EquityCalendarDomain(CountryCode.HONG_KONG, "XHKG")
HU_EQUITIES = EquityCalendarDomain(CountryCode.HUNGARY, "XBUD")
ID_EQUITIES = EquityCalendarDomain(CountryCode.INDONESIA, "XIDX")
IE_EQUITIES = EquityCalendarDomain(CountryCode.IRELAND, "XDUB")
IN_EQUITIES = EquityCalendarDomain(CountryCode.INDIA, "XBOM")
IT_EQUITIES = EquityCalendarDomain(CountryCode.ITALY, "XMIL")
JP_EQUITIES = EquityCalendarDomain(CountryCode.JAPAN, "XTKS")
KR_EQUITIES = EquityCalendarDomain(CountryCode.SOUTH_KOREA, "XKRX")
MX_EQUITIES = EquityCalendarDomain(CountryCode.MEXICO, "XMEX")
MY_EQUITIES = EquityCalendarDomain(CountryCode.MALAYSIA, "XKLS")
NL_EQUITIES = EquityCalendarDomain(CountryCode.NETHERLANDS, "XAMS")
NO_EQUITIES = EquityCalendarDomain(CountryCode.NORWAY, "XOSL")
NZ_EQUITIES = EquityCalendarDomain(CountryCode.NEW_ZEALAND, "XNZE")
PE_EQUITIES = EquityCalendarDomain(CountryCode.PERU, "XLIM")
PH_EQUITIES = EquityCalendarDomain(CountryCode.PHILIPPINES, "XPHS")
PK_EQUITIES = EquityCalendarDomain(CountryCode.PAKISTAN, "XKAR")
PL_EQUITIES = EquityCalendarDomain(CountryCode.POLAND, "XWAR")
PT_EQUITIES = EquityCalendarDomain(CountryCode.PORTUGAL, "XLIS")
RU_EQUITIES = EquityCalendarDomain(CountryCode.RUSSIA, "XMOS")
SE_EQUITIES = EquityCalendarDomain(CountryCode.SWEDEN, "XSTO")
SG_EQUITIES = EquityCalendarDomain(CountryCode.SINGAPORE, "XSES")
TH_EQUITIES = EquityCalendarDomain(CountryCode.THAILAND, "XBKK")
TR_EQUITIES = EquityCalendarDomain(CountryCode.TURKEY, "XIST")
TW_EQUITIES = EquityCalendarDomain(CountryCode.TAIWAN, "XTAI")
US_EQUITIES = EquityCalendarDomain(CountryCode.UNITED_STATES, "XNYS")
ZA_EQUITIES = EquityCalendarDomain(CountryCode.SOUTH_AFRICA, "XJSE")

BUILT_IN_DOMAINS = [
    AR_EQUITIES,
    AT_EQUITIES,
    AU_EQUITIES,
    BE_EQUITIES,
    BR_EQUITIES,
    CA_EQUITIES,
    CH_EQUITIES,
    CL_EQUITIES,
    CN_EQUITIES,
    CO_EQUITIES,
    CZ_EQUITIES,
    DE_EQUITIES,
    DK_EQUITIES,
    ES_EQUITIES,
    FI_EQUITIES,
    FR_EQUITIES,
    GB_EQUITIES,
    GR_EQUITIES,
    HK_EQUITIES,
    HU_EQUITIES,
    ID_EQUITIES,
    IE_EQUITIES,
    IN_EQUITIES,
    IT_EQUITIES,
    JP_EQUITIES,
    KR_EQUITIES,
    MX_EQUITIES,
    MY_EQUITIES,
    NL_EQUITIES,
    NO_EQUITIES,
    NZ_EQUITIES,
    PE_EQUITIES,
    PH_EQUITIES,
    PK_EQUITIES,
    PL_EQUITIES,
    PT_EQUITIES,
    RU_EQUITIES,
    SE_EQUITIES,
    SG_EQUITIES,
    TH_EQUITIES,
    TR_EQUITIES,
    TW_EQUITIES,
    US_EQUITIES,
    ZA_EQUITIES,
]


def infer_domain(terms):
    """Infer the domain from a collection of terms.

    The algorithm for inferring domains is as follows:

    - If all input terms have a domain of GENERIC, the result is GENERIC.

    - If there is exactly one non-generic domain in the input terms, the result
      is that domain.

    - Otherwise, an AmbiguousDomain error is raised.

    Parameters
    ----------
    terms : iterable[zipline.pipeline.Term]

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


# This would be better if we provided more context for which domains came from
# which terms.
class AmbiguousDomain(Exception):
    """Raised when we attempt to infer a domain from a collection of mixed terms."""

    _TEMPLATE = dedent(
        """\
        Found terms with conflicting domains:
        {domains}"""
    )

    def __init__(self, domains):
        self.domains = domains

    def __str__(self):
        return self._TEMPLATE.format(
            domains=bulleted_list(self.domains, indent=2),
        )


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
        data_query_time=optional(datetime.time),
        data_query_date_offset=int,
        __funcname="EquitySessionDomain",
    )
    def __init__(
        self,
        sessions,
        country_code,
        data_query_time=None,
        data_query_date_offset=0,
    ):
        self._country_code = country_code
        self._sessions = sessions

        if data_query_time is None:
            data_query_time = datetime.time(0, 0, tzinfo=pytz.timezone("UTC"))

        if data_query_time.tzinfo is None:
            raise ValueError("data_query_time cannot be tz-naive")

        self._data_query_time = data_query_time
        self._data_query_date_offset = data_query_date_offset

    @property
    def country_code(self):
        return self._country_code

    def sessions(self):
        return self._sessions

    def data_query_cutoff_for_sessions(self, sessions):
        return days_at_time(
            sessions,
            self._data_query_time,
            self._data_query_time.tzinfo,
            self._data_query_date_offset,
        )
