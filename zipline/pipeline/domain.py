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
from functools import reduce
from textwrap import dedent

from interface import implements, Interface, default
import numpy as np
import pandas as pd
import pytz

from trading_calendars import get_calendar

from zipline import country
from zipline.errors import NoFurtherDataError
from zipline.utils.formatting import bulleted_list
from zipline.utils.input_validation import expect_types, optional
from zipline.utils.memoize import lazyval
from zipline.utils.pandas_utils import days_at_time
from zipline.utils.sentinel import sentinel

_GENERIC = sentinel('_GENERIC')


class ITimeDimension(Interface):
    """Interface for the calendar dimension of a domain.
    """

    def all_sessions(self):
        """
        Get all trading sessions for the time component of the domain.

        Returns
        -------
        sessions : pd.DatetimeIndex
            An array of all session labels for this domain.
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
        data_query_cutoffs : pd.DatetimeIndex
            Timestamp of the last minute for which data should be considered
            "available" on each session.
        """

    def __eq__(self, other):
        """Check if self is equal to other.
        """

    @default
    def __ne__(self, other):
        """Check if self is not equal to other
        """
        return not (self == other)


class NamedCalendar(implements(ITimeDimension)):
    """A time dimension that corresponds to a daily trading_calendars calendar.

    Parameters
    ----------
    calendar_name : str
        Name of the calendar, to be looked by by trading_calendar.get_calendar.
    data_query_offset : np.timedelta64
         The offset from market open when data should no longer be considered
         available for a session. For example, a ``data_query_offset`` of
         ``-np.timedelta64(45, 'm')`` means that the data must have
         been available at least 45 minutes prior to market open for it to
         appear in the pipeline input for the given session.
    """
    @expect_types(calendar_name=str, __funcname='NamedCalendar')
    def __init__(self,
                 calendar_name,
                 data_query_offset=-np.timedelta64(45, 'm')):
        self._calendar_name = calendar_name
        self._data_query_offset = (
            # add one minute because `open_time` is actually the open minute
            # label which is one minute _after_ market open...
            data_query_offset - np.timedelta64(1, 'm')
        )
        if data_query_offset >= datetime.timedelta(0):
            raise ValueError(
                'data must be ready before market open (offset must be < 0)',
            )

    @lazyval
    def calendar(self):
        return get_calendar(self.calendar_name)

    def all_sessions(self):
        return self.calendar.all_sessions

    def data_query_cutoff_for_sessions(self, sessions):
        opens = self.calendar.opens.loc[sessions].values
        missing_mask = pd.isnull(opens)
        if missing_mask.any():
            missing_days = sessions[missing_mask]
            raise ValueError(
                'cannot resolve data query time for sessions that are not on'
                ' the %s calendar:\n%s' % (
                    self.calendar.name,
                    missing_days,
                ),
            )

        return pd.DatetimeIndex(opens + self._data_query_offset, tz='UTC')

    def __repr__(self):
        return "{}({!r})".format(type(self).__name__, self._calendar_name)

    def __eq__(self, other):
        if not isinstance(other, NamedCalendar):
            return False
        return (
            self._calendar_name == other._calendar_name
            and self._data_query_offset == other._data_query_offset
        )


class AdHocCalendar(implements(ITimeDimension)):
    """A calendar dimension build directly an index of session labels.

    This is mostly useful for testing.

    Parameters
    ----------
    sessions : pd.DatetimeIndex
        Sessions to use as output labels for pipelines run on this domain.
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
        data_query_time=optional(datetime.time),
        data_query_date_offset=int,
        __funcname='AdHocCalendarDimension',
    )
    def __init__(self,
                 sessions,
                 data_query_time=None,
                 data_query_date_offset=0):
        self._sessions = sessions

        if data_query_time is None:
            data_query_time = datetime.time(0, 0, tzinfo=pytz.timezone('UTC'))

        if data_query_time.tzinfo is None:
            raise ValueError("data_query_time cannot be tz-naive")

        self._data_query_time = data_query_time
        self._data_query_date_offset = data_query_date_offset

    def all_sessions(self):
        return self._sessions

    def data_query_cutoff_for_sessions(self, sessions):
        return days_at_time(
            sessions,
            self._data_query_time,
            self._data_query_time.tzinfo,
            self._data_query_date_offset,
        )

    def __eq__(self, other):
        return self is other


class IEntityDimension(Interface):
    """Interface for the entity dimension of a domain.
    """

    def intersect(self, other):
        """Compute the intersection of this entity dimension with another.

        Parameters
        ----------
        other : implements(IEntityDimension)
            TODO

        Returns
        -------
        dim : implements(IEntityDimension)
            TODO
        """

    def lifetimes(self, asset_finder, sessions):
        """
        Compute a DataFrame representing entity lifetimes for the specified
        date range.

        Parameters
        ----------
        asset_finder : zipline.assets.AssetFinder
            Asset database for the simulation.
        sessions : pd.DatetimeIndex
            Pipeline execution dates for which lifetimes are needed.

        Returns
        -------
        lifetimes : pd.DataFrame
            DataFrame of bools, indexed by ``sessions``, containing a column
            for each entity that exists for at least one day in ``sessions``.
            Each ``(session, entity)`` location indicates whether ``entity``
            should be included in pipeline outputs for ``session``.
        """

    @default
    def empty(self):
        return False

    def __eq__(self, other):
        pass

    @default
    def __ne__(self, other):
        return not (self == other)


class GenericDimension(implements(ITimeDimension, IEntityDimension)):
    """A 'generic' dimension, encompassing all entities and all dates.

    TODO: Explain how this works.
    """

    def intersect(self, other):
        return other

    def lifetimes(self, asset_finder, sessions):
        # TODO: Better error.
        raise Exception("Can't get lifetimes from generic dimension.")

    def all_sessions(self):
        raise Exception("Can't get all_sessions from generic dimension.")

    def data_query_cutoff_for_sessions(self, sessions):
        raise Exception(
            "Can't get data_query_cutoff_for_sesions from generic dimension."
        )

    def __repr__(self):
        return "GENERIC"

    def __eq__(self, other):
        return isinstance(other, GenericDimension)

    # We need to provide this b/c interface can't tell that the defaults
    # provided by ITimeDimension and IEntityDimension are the same.
    def __ne__(self, other):
        return not (self == other)


_GENERIC_DIMENSION = GenericDimension()


class EmptyDimension(implements(IEntityDimension)):
    """An empty entity dimension. Contains no entities.
    """

    def empty(self):
        return True

    def intersect(self, other):
        return self

    def lifetimes(self, asset_finder, sessions):
        # TODO: This could also return an empty dataframe? Is that ever useful?
        raise Exception("Can't get lifetimes from empty dimension.")

    def __repr__(self):
        return "EMPTY"

    def __eq__(self, other):
        return isinstance(other, EmptyDimension)


_EMPTY_DIMENSION = EmptyDimension()


class SingleMarketEquities(implements(IEntityDimension)):
    """An entity dimension for equities trading in a single market.

    Parameters
    ----------
    country : zipline.country.Country
        Country in which equities on this domain trade.
    """
    def __init__(self, country):
        self._country = country

    def lifetimes(self, asset_finder, sessions):
        return asset_finder.lifetimes(
            sessions,
            include_start_date=False,
            country_codes=[self._country.code],
        )

    def intersect(self, other):
        if self == other:
            return self
        return _EMPTY_DIMENSION

    def __eq__(self, other):
        if not isinstance(other, SingleMarketEquities):
            return False
        return self._country == other._country

    def __repr__(self):
        return "{}({!r})".format(type(self).__name__, self._country.code)


class WorldCurrencies(implements(IEntityDimension)):
    """An entity dimension for world currencies.
    """

    def intersect(self, other):
        if self == other:
            return self
        return _EMPTY_DIMENSION

    def lifetimes(self, asset_finder, sessions):
        return pd.DataFrame(
            index=sessions,
            columns=sorted(country.ALL_CURRENCIES),
            # TODO: Model currencies being created and destroyed?
            data=True,
        )

    def __eq__(self, other):
        return isinstance(other, WorldCurrencies)

    def __repr__(self):
        return "CURRENCIES".format(type(self).__name__)


_CURRENCY_DIMENSION = WorldCurrencies()


class Domain(object):
    """
    A Pipeline domain.

    A domain defines two things:

    1. A calendar defining the dates to which the pipeline's inputs and outputs
       should be aligned. The calendar is represented concretely by a pandas
       DatetimeIndex.
    2. The set of entities that the pipeline should compute over.

    Parameters
    ----------
    time_dimension : implements(ITimeDimension)
        Dimension describing the time component of the domain.
    entity_dimension : implements(IEntityDimension) or None
        Dimension describing the entity component of the domain.
    """

    def __init__(self, time_dimension, entity_dimension):
        self._time_dim = time_dimension
        self._entity_dim = entity_dimension

    def data_query_cutoff_for_sessions(self, sessions):
        """Compute the data query cutoff time for the given sessions.

        Parameters
        ----------
        sessions : pd.DatetimeIndex
            The sessions to get the data query cutoff times for. This index
            will contain all midnight UTC values.

        Returns
        -------
        data_query_cutoffs : pd.DatetimeIndex
            Timestamp of the last minute for which data should be considered
            "available" on each session.
        """
        return self._time_dim.data_query_cutoff_for_sessions(sessions)

    def lifetimes(self, asset_finder, start, end, extra_rows):
        """
        Compute a DataFrame representing entity lifetimes for the specified
        date range.

        Parameters
        ----------
        asset_finder : zipline.assets.AssetFinder
            Asset database for the simulation.
        start : pd.Timestamp
            Start date of period for which lifetimes should be computed.
        end : pd.Timestamp
            End date of the period for which lifetimes should be computed.
        pre_start_delta : int
            Number of dates prior to ``start`` that should be included in
            output lifetimes.

        Returns
        -------
        lifetimes : pd.DataFrame
            DataFrame of bools, indexed by ``sessions``, containing a column
            for each entity that exists for at least one day in ``sessions``.
            Each ``(session, entity)`` location indicates whether ``entity``
            should be included in pipeline outputs for ``session``.
        """
        sessions = self._time_dim.all_sessions()
        if start not in sessions:
            raise ValueError(
                "Pipeline start date ({}) is not a trading session for "
                "domain {}.".format(start, self)
            )
        elif end not in sessions:
            raise ValueError(
                "Pipeline end date {} is not a trading session for "
                "domain {}.".format(end, self)
            )

        start_idx, end_idx = sessions.slice_locs(start, end)
        if start_idx < extra_rows:
            raise NoFurtherDataError.from_lookback_window(
                initial_message="Insufficient data to compute Pipeline:",
                first_date=sessions[0],
                lookback_start=start,
                lookback_length=extra_rows,
            )

        out_sessions = sessions[start_idx - extra_rows:end_idx]

        return self._entity_dim.lifetimes(asset_finder, out_sessions)

    def roll_forward(self, dt):
        """
        Given a date, roll forward to the next session on this domain's time
        dimension.

        Parameters
        ----------
        dt : pd.Timestamp

        Returns
        -------
        pd.Timestamp
        """
        dt = pd.Timestamp(dt, tz='UTC')

        trading_days = self.all_sessions()
        try:
            return trading_days[trading_days.searchsorted(dt)]
        except IndexError:
            raise ValueError(
                "Date {} was past the last session for domain {}. "
                "The last session for this domain is {}.".format(
                    dt.date(),
                    self,
                    trading_days[-1].date()
                )
            )

    @property
    def ndim(self):
        """TODO
        """
        if self._entity_dim is None:
            return 1
        return 2

    def __repr__(self):
        return "Domain({!r}, {!r})".format(self._time_dim, self._entity_dim)

    def __eq__(self, other):
        return (
            self._time_dim == other._time_dim
            and self._entity_dim == other._entity_dim
        )

    def __ne__(self, other):
        return not self == other


def equity_domain(country, calendar_code):
    return Domain(
        NamedCalendar(calendar_code),
        SingleMarketEquities(country),
    )


AR_EQUITIES = equity_domain(country.ARGENTINA, 'XBUE')
AT_EQUITIES = equity_domain(country.AUSTRIA, 'XWBO')
AU_EQUITIES = equity_domain(country.AUSTRALIA, 'XASX')
BE_EQUITIES = equity_domain(country.BELGIUM, 'XBRU')
BR_EQUITIES = equity_domain(country.BRAZIL, 'BVMF')
CA_EQUITIES = equity_domain(country.CANADA, 'XTSE')
CH_EQUITIES = equity_domain(country.SWITZERLAND, 'XSWX')
CL_EQUITIES = equity_domain(country.CHILE, 'XSGO')
CN_EQUITIES = equity_domain(country.CHINA, 'XSHG')
CO_EQUITIES = equity_domain(country.COLOMBIA, 'XBOG')
CZ_EQUITIES = equity_domain(country.CZECH_REPUBLIC, 'XPRA')
DE_EQUITIES = equity_domain(country.GERMANY, 'XFRA')
DK_EQUITIES = equity_domain(country.DENMARK, 'XCSE')
ES_EQUITIES = equity_domain(country.SPAIN, 'XMAD')
FI_EQUITIES = equity_domain(country.FINLAND, 'XHEL')
FR_EQUITIES = equity_domain(country.FRANCE, 'XPAR')
GB_EQUITIES = equity_domain(country.UNITED_KINGDOM, 'XLON')
GR_EQUITIES = equity_domain(country.GREECE, 'ASEX')
HK_EQUITIES = equity_domain(country.HONG_KONG, 'XHKG')
HU_EQUITIES = equity_domain(country.HUNGARY, 'XBUD')
ID_EQUITIES = equity_domain(country.INDONESIA, 'XIDX')
IE_EQUITIES = equity_domain(country.IRELAND, 'XDUB')
IN_EQUITIES = equity_domain(country.INDIA, "XBOM")
IT_EQUITIES = equity_domain(country.ITALY, 'XMIL')
JP_EQUITIES = equity_domain(country.JAPAN, 'XTKS')
KR_EQUITIES = equity_domain(country.SOUTH_KOREA, 'XKRX')
MX_EQUITIES = equity_domain(country.MEXICO, 'XMEX')
MY_EQUITIES = equity_domain(country.MALAYSIA, 'XKLS')
NL_EQUITIES = equity_domain(country.NETHERLANDS, 'XAMS')
NO_EQUITIES = equity_domain(country.NORWAY, 'XOSL')
NZ_EQUITIES = equity_domain(country.NEW_ZEALAND, 'XNZE')
PE_EQUITIES = equity_domain(country.PERU, 'XLIM')
PH_EQUITIES = equity_domain(country.PHILIPPINES, 'XPHS')
PK_EQUITIES = equity_domain(country.PAKISTAN, 'XKAR')
PL_EQUITIES = equity_domain(country.POLAND, 'XWAR')
PT_EQUITIES = equity_domain(country.PORTUGAL, 'XLIS')
RU_EQUITIES = equity_domain(country.RUSSIA, 'XMOS')
SE_EQUITIES = equity_domain(country.SWEDEN, 'XSTO')
SG_EQUITIES = equity_domain(country.SINGAPORE, 'XSES')
TH_EQUITIES = equity_domain(country.THAILAND, 'XBKK')
TR_EQUITIES = equity_domain(country.TURKEY, 'XIST')
TW_EQUITIES = equity_domain(country.TAIWAN, 'XTAI')
US_EQUITIES = equity_domain(country.UNITED_STATES, 'XNYS')
ZA_EQUITIES = equity_domain(country.SOUTH_AFRICA, 'XJSE')

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

GENERIC = Domain(_GENERIC_DIMENSION, _GENERIC_DIMENSION)
CURRENCIES = Domain(
    _GENERIC_DIMENSION,
    _CURRENCY_DIMENSION,
)


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
    else:
        time_dim = _choose_dimension({d._time_dim for d in domains})
        entity_dim = _choose_dimension({d._entity_dim for d in domains})
        return Domain(time_dim, entity_dim)


def _choose_dimension(dimensions):
    """Choose a dimension from a collection of terms in a pipeline.

    Parameters
    ----------
    dimensions : set[object]
    """
    result = reduce(
        lambda x, y: x.intersect(y),
        dimensions,
    )

    if result.empty():
        raise AmbiguousDomain(dimensions)

    return result


class IncompatibleDimensions(Exception):
    """Raised when we fail to determine a dimension from pipeline inputs.
    """
    _TEMPLATE = """\
    Found terms with incompatible dimensions:
    {dims}"""

    def __init__(self, dims):
        self.dims = dims

    def __str__(self):
        return self._TEMPLATE.format(dims=bulleted_list(self.dims, indent=2))
