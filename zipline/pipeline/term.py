"""
Base class for Filters, Factors and Classifiers
"""
from abc import ABCMeta, abstractproperty
from bisect import insort
from collections import Mapping
from weakref import WeakValueDictionary

from numpy import (
    array,
    dtype as dtype_class,
    ndarray,
    searchsorted,
)
from six import with_metaclass

from zipline.assets import Asset
from zipline.errors import (
    DTypeNotSpecified,
    InvalidOutputName,
    NonExistentAssetInTimeFrame,
    NonSliceableTerm,
    NonWindowSafeInput,
    NotDType,
    TermInputsNotSpecified,
    TermOutputsEmpty,
    UnsupportedDType,
    WindowLengthNotSpecified,
)
from zipline.lib.adjusted_array import can_represent_dtype
from zipline.lib.labelarray import LabelArray
from zipline.utils.input_validation import expect_types
from zipline.utils.memoize import lazyval
from zipline.utils.numpy_utils import (
    bool_dtype,
    categorical_dtype,
    datetime64ns_dtype,
    default_missing_value_for_dtype,
)
from zipline.utils.sharedoc import (
    templated_docstring,
    PIPELINE_ALIAS_NAME_DOC,
    PIPELINE_DOWNSAMPLING_FREQUENCY_DOC,
)

from .downsample_helpers import expect_downsample_frequency
from .sentinels import NotSpecified


class Term(with_metaclass(ABCMeta, object)):
    """
    Base class for terms in a Pipeline API compute graph.
    """
    # These are NotSpecified because a subclass is required to provide them.
    dtype = NotSpecified
    domain = NotSpecified
    missing_value = NotSpecified

    # Subclasses aren't required to provide `params`.  The default behavior is
    # no params.
    params = ()

    # Determines if a term is safe to be used as a windowed input.
    window_safe = False

    # The dimensions of the term's output (1D or 2D).
    ndim = 2

    _term_cache = WeakValueDictionary()

    def __new__(cls,
                domain=domain,
                dtype=dtype,
                missing_value=missing_value,
                window_safe=NotSpecified,
                ndim=NotSpecified,
                # params is explicitly not allowed to be passed to an instance.
                *args,
                **kwargs):
        """
        Memoized constructor for Terms.

        Caching previously-constructed Terms is useful because it allows us to
        only compute equivalent sub-expressions once when traversing a Pipeline
        dependency graph.

        Caching previously-constructed Terms is **sane** because terms and
        their inputs are both conceptually immutable.
        """
        # Subclasses can set override these class-level attributes to provide
        # default values.
        if domain is NotSpecified:
            domain = cls.domain
        if dtype is NotSpecified:
            dtype = cls.dtype
        if missing_value is NotSpecified:
            missing_value = cls.missing_value
        if ndim is NotSpecified:
            ndim = cls.ndim
        if window_safe is NotSpecified:
            window_safe = cls.window_safe

        dtype, missing_value = validate_dtype(
            cls.__name__,
            dtype,
            missing_value,
        )
        params = cls._pop_params(kwargs)

        identity = cls._static_identity(
            domain=domain,
            dtype=dtype,
            missing_value=missing_value,
            window_safe=window_safe,
            ndim=ndim,
            params=params,
            *args, **kwargs
        )

        try:
            return cls._term_cache[identity]
        except KeyError:
            new_instance = cls._term_cache[identity] = \
                super(Term, cls).__new__(cls)._init(
                    domain=domain,
                    dtype=dtype,
                    missing_value=missing_value,
                    window_safe=window_safe,
                    ndim=ndim,
                    params=params,
                    *args, **kwargs
                )
            return new_instance

    @classmethod
    def _pop_params(cls, kwargs):
        """
        Pop entries from the `kwargs` passed to cls.__new__ based on the values
        in `cls.params`.

        Parameters
        ----------
        kwargs : dict
            The kwargs passed to cls.__new__.

        Returns
        -------
        params : list[(str, object)]
            A list of string, value pairs containing the entries in cls.params.

        Raises
        ------
        TypeError
            Raised if any parameter values are not passed or not hashable.
        """
        params = cls.params
        if not isinstance(params, Mapping):
            params = {k: NotSpecified for k in params}
        param_values = []
        for key, default_value in params.items():
            try:
                value = kwargs.pop(key, default_value)
                if value is NotSpecified:
                    raise KeyError(key)

                # Check here that the value is hashable so that we fail here
                # instead of trying to hash the param values tuple later.
                hash(value)
            except KeyError:
                raise TypeError(
                    "{typename} expected a keyword parameter {name!r}.".format(
                        typename=cls.__name__,
                        name=key
                    )
                )
            except TypeError:
                # Value wasn't hashable.
                raise TypeError(
                    "{typename} expected a hashable value for parameter "
                    "{name!r}, but got {value!r} instead.".format(
                        typename=cls.__name__,
                        name=key,
                        value=value,
                    )
                )

            param_values.append((key, value))
        return tuple(param_values)

    def __init__(self, *args, **kwargs):
        """
        Noop constructor to play nicely with our caching __new__.  Subclasses
        should implement _init instead of this method.

        When a class' __new__ returns an instance of that class, Python will
        automatically call __init__ on the object, even if a new object wasn't
        actually constructed.  Because we memoize instances, we often return an
        object that was already initialized from __new__, in which case we
        don't want to call __init__ again.

        Subclasses that need to initialize new instances should override _init,
        which is guaranteed to be called only once.
        """
        pass

    @expect_types(key=Asset)
    def __getitem__(self, key):
        if isinstance(self, LoadableTerm):
            raise NonSliceableTerm(term=self)
        return Slice(self, key)

    @classmethod
    def _static_identity(cls,
                         domain,
                         dtype,
                         missing_value,
                         window_safe,
                         ndim,
                         params):
        """
        Return the identity of the Term that would be constructed from the
        given arguments.

        Identities that compare equal will cause us to return a cached instance
        rather than constructing a new one.  We do this primarily because it
        makes dependency resolution easier.

        This is a classmethod so that it can be called from Term.__new__ to
        determine whether to produce a new instance.
        """
        return (cls, domain, dtype, missing_value, window_safe, ndim, params)

    def _init(self, domain, dtype, missing_value, window_safe, ndim, params):
        """
        Parameters
        ----------
        domain : object
            Unused placeholder.
        dtype : np.dtype
            Dtype of this term's output.
        params : tuple[(str, hashable)]
            Tuple of key/value pairs of additional parameters.
        """
        self.domain = domain
        self.dtype = dtype
        self.missing_value = missing_value
        self.window_safe = window_safe
        self.ndim = ndim

        for name, value in params:
            if hasattr(self, name):
                raise TypeError(
                    "Parameter {name!r} conflicts with already-present"
                    " attribute with value {value!r}.".format(
                        name=name,
                        value=getattr(self, name),
                    )
                )
            # TODO: Consider setting these values as attributes and replacing
            # the boilerplate in NumericalExpression, Rank, and
            # PercentileFilter.

        self.params = dict(params)

        # Make sure that subclasses call super() in their _validate() methods
        # by setting this flag.  The base class implementation of _validate
        # should set this flag to True.
        self._subclass_called_super_validate = False
        self._validate()
        assert self._subclass_called_super_validate, (
            "Term._validate() was not called.\n"
            "This probably means that you overrode _validate"
            " without calling super()."
        )
        del self._subclass_called_super_validate

        return self

    def _validate(self):
        """
        Assert that this term is well-formed.  This should be called exactly
        once, at the end of Term._init().
        """
        # mark that we got here to enforce that subclasses overriding _validate
        # call super().
        self._subclass_called_super_validate = True

    def compute_extra_rows(self,
                           all_dates,
                           start_date,
                           end_date,
                           min_extra_rows):
        """
        Calculate the number of extra rows needed to compute ``self``.

        Must return at least ``min_extra_rows``, and the default implementation
        is to just return ``min_extra_rows``.  This is overridden by
        downsampled terms to ensure that the first date computed is a
        recomputation date.

        Parameters
        ----------
        all_dates : pd.DatetimeIndex
            The trading sessions against which ``self`` will be computed.
        start_date : pd.Timestamp
            The first date for which final output is requested.
        end_date : pd.Timestamp
            The last date for which final output is requested.
        min_extra_rows : int
            The minimum number of extra rows required of ``self``, as
            determined by other terms that depend on ``self``.

        Returns
        -------
        extra_rows : int
            The number of extra rows to compute.  Must be at least
            ``min_extra_rows``.
        """
        return min_extra_rows

    @abstractproperty
    def inputs(self):
        """
        A tuple of other Terms needed as direct inputs for this Term.
        """
        raise NotImplementedError('inputs')

    @abstractproperty
    def windowed(self):
        """
        Boolean indicating whether this term is a trailing-window computation.
        """
        raise NotImplementedError('windowed')

    @abstractproperty
    def mask(self):
        """
        A Filter representing asset/date pairs to include while
        computing this Term. (True means include; False means exclude.)
        """
        raise NotImplementedError('mask')

    @abstractproperty
    def dependencies(self):
        """
        A dictionary mapping terms that must be computed before `self` to the
        number of extra rows needed for those terms.
        """
        raise NotImplementedError('dependencies')


class AssetExists(Term):
    """
    Pseudo-filter describing whether or not an asset existed on a given day.
    This is the default mask for all terms that haven't been passed a mask
    explicitly.

    This is morally a Filter, in the sense that it produces a boolean value for
    every asset on every date.  We don't subclass Filter, however, because
    `AssetExists` is computed directly by the PipelineEngine.

    This term is guaranteed to be available as an input for any term computed
    by SimplePipelineEngine.run_pipeline().

    See Also
    --------
    zipline.assets.AssetFinder.lifetimes
    """
    dtype = bool_dtype
    dataset = None
    inputs = ()
    dependencies = {}
    mask = None
    windowed = False

    def __repr__(self):
        return "AssetExists()"

    def _compute(self, today, assets, out):
        raise NotImplementedError(
            "AssetExists cannot be computed directly."
            " Check your PipelineEngine configuration."
        )


class InputDates(Term):
    """
    1-Dimensional term providing date labels for other term inputs.

    This term is guaranteed to be available as an input for any term computed
    by SimplePipelineEngine.run_pipeline().
    """
    ndim = 1
    dataset = None
    dtype = datetime64ns_dtype
    inputs = ()
    dependencies = {}
    mask = None
    windowed = False
    window_safe = True

    def __repr__(self):
        return "InputDates()"

    def _compute(self, today, assets, out):
        raise NotImplementedError(
            "InputDates cannot be computed directly."
            " Check your PipelineEngine configuration."
        )


class LoadableTerm(Term):
    """
    A Term that should be loaded from an external resource by a PipelineLoader.

    This is the base class for :class:`zipline.pipeline.data.BoundColumn`.
    """
    windowed = False
    inputs = ()

    @lazyval
    def dependencies(self):
        return {self.mask: 0}


class ComputableTerm(Term):
    """
    A Term that should be computed from a tuple of inputs.

    This is the base class for :class:`zipline.pipeline.Factor`,
    :class:`zipline.pipeline.Filter`, and :class:`zipline.pipeline.Classifier`.
    """
    inputs = NotSpecified
    outputs = NotSpecified
    window_length = NotSpecified
    mask = NotSpecified

    def __new__(cls,
                inputs=inputs,
                outputs=outputs,
                window_length=window_length,
                mask=mask,
                *args, **kwargs):

        if inputs is NotSpecified:
            inputs = cls.inputs

        # Having inputs = NotSpecified is an error, but we handle it later
        # in self._validate rather than here.
        if inputs is not NotSpecified:
            # Allow users to specify lists as class-level defaults, but
            # normalize to a tuple so that inputs is hashable.
            inputs = tuple(inputs)

        if outputs is NotSpecified:
            outputs = cls.outputs
        if outputs is not NotSpecified:
            outputs = tuple(outputs)

        if mask is NotSpecified:
            mask = cls.mask
        if mask is NotSpecified:
            mask = AssetExists()

        if window_length is NotSpecified:
            window_length = cls.window_length

        return super(ComputableTerm, cls).__new__(
            cls,
            inputs=inputs,
            outputs=outputs,
            mask=mask,
            window_length=window_length,
            *args, **kwargs
        )

    def _init(self, inputs, outputs, window_length, mask, *args, **kwargs):
        self.inputs = inputs
        self.outputs = outputs
        self.window_length = window_length
        self.mask = mask
        return super(ComputableTerm, self)._init(*args, **kwargs)

    @classmethod
    def _static_identity(cls,
                         inputs,
                         outputs,
                         window_length,
                         mask,
                         *args,
                         **kwargs):
        return (
            super(ComputableTerm, cls)._static_identity(*args, **kwargs),
            inputs,
            outputs,
            window_length,
            mask,
        )

    def _validate(self):
        super(ComputableTerm, self)._validate()

        if self.inputs is NotSpecified:
            raise TermInputsNotSpecified(termname=type(self).__name__)

        if self.outputs is NotSpecified:
            pass
        elif not self.outputs:
            raise TermOutputsEmpty(termname=type(self).__name__)
        else:
            # Raise an exception if there are any naming conflicts between the
            # term's output names and certain attributes.
            disallowed_names = [
                attr for attr in dir(ComputableTerm)
                if not attr.startswith('_')
            ]

            # The name 'compute' is an added special case that is disallowed.
            # Use insort to add it to the list in alphabetical order.
            insort(disallowed_names, 'compute')

            for output in self.outputs:
                if output.startswith('_') or output in disallowed_names:
                    raise InvalidOutputName(
                        output_name=output,
                        termname=type(self).__name__,
                        disallowed_names=disallowed_names,
                    )

        if self.window_length is NotSpecified:
            raise WindowLengthNotSpecified(termname=type(self).__name__)

        if self.mask is NotSpecified:
            # This isn't user error, this is a bug in our code.
            raise AssertionError("{term} has no mask".format(term=self))

        if self.window_length:
            for child in self.inputs:
                if not child.window_safe:
                    raise NonWindowSafeInput(parent=self, child=child)

    def _compute(self, inputs, dates, assets, mask):
        """
        Subclasses should implement this to perform actual computation.

        This is named ``_compute`` rather than just ``compute`` because
        ``compute`` is reserved for user-supplied functions in
        CustomFilter/CustomFactor/CustomClassifier.
        """
        raise NotImplementedError()

    @lazyval
    def windowed(self):
        """
        Whether or not this term represents a trailing window computation.

        If term.windowed is truthy, its compute_from_windows method will be
        called with instances of AdjustedArray as inputs.

        If term.windowed is falsey, its compute_from_baseline will be called
        with instances of np.ndarray as inputs.
        """
        return (
            self.window_length is not NotSpecified
            and self.window_length > 0
        )

    @lazyval
    def dependencies(self):
        """
        The number of extra rows needed for each of our inputs to compute this
        term.
        """
        extra_input_rows = max(0, self.window_length - 1)
        out = {}
        for term in self.inputs:
            out[term] = extra_input_rows
        out[self.mask] = 0
        return out

    @expect_types(data=ndarray)
    def postprocess(self, data):
        """
        Called with an result of ``self``, unravelled (i.e. 1-dimensional)
        after any user-defined screens have been applied.

        This is mostly useful for transforming the dtype of an output, e.g., to
        convert a LabelArray into a pandas Categorical.

        The default implementation is to just return data unchanged.
        """
        return data

    def to_workspace_value(self, result, assets):
        """
        Called with a column of the result of a pipeline. This needs to put
        the data into a format that can be used in a workspace to continue
        doing computations.

        Parameters
        ----------
        result : pd.Series
            A multiindexed series with (dates, assets) whose values are the
            results of running this pipeline term over the dates.
        assets : pd.Index
            All of the assets being requested. This allows us to correctly
            shape the workspace value.

        Returns
        -------
        workspace_value : array-like
            An array like value that the engine can consume.
        """
        return result.unstack().fillna(self.missing_value).reindex(
            columns=assets,
            fill_value=self.missing_value,
        ).values

    def _downsampled_type(self, *args, **kwargs):
        """
        The expression type to return from self.downsample().
        """
        raise NotImplementedError(
            "downsampling is not yet implemented "
            "for instances of %s." % type(self).__name__
        )

    @expect_downsample_frequency
    @templated_docstring(frequency=PIPELINE_DOWNSAMPLING_FREQUENCY_DOC)
    def downsample(self, frequency):
        """
        Make a term that computes from ``self`` at lower-than-daily frequency.

        Parameters
        ----------
        {frequency}
        """
        return self._downsampled_type(term=self, frequency=frequency)

    def _aliased_type(self, *args, **kwargs):
        """
        The expression type to return from self.alias().
        """
        raise NotImplementedError(
            "alias is not yet implemented "
            "for instances of %s." % type(self).__name__
        )

    @templated_docstring(name=PIPELINE_ALIAS_NAME_DOC)
    def alias(self, name):
        """
        Make a term from ``self`` that names the expression.

        Parameters
        ----------
        {name}

        Returns
        -------
        aliased : Aliased
            ``self`` with a name.

        Notes
        -----
        This is useful for giving a name to a numerical or boolean expression.
        """
        return self._aliased_type(term=self, name=name)

    def __repr__(self):
        return (
            "{type}({inputs}, window_length={window_length})"
        ).format(
            type=type(self).__name__,
            inputs=self.inputs,
            window_length=self.window_length,
        )


class Slice(ComputableTerm):
    """
    Term for extracting a single column of a another term's output.

    Parameters
    ----------
    term : zipline.pipeline.term.Term
        The term from which to extract a column of data.
    asset : zipline.assets.Asset
        The asset corresponding to the column of `term` to be extracted.

    Notes
    -----
    Users should rarely construct instances of `Slice` directly. Instead, they
    should construct instances via indexing, e.g. `MyFactor()[Asset(24)]`.
    """
    def __new__(cls, term, asset):
        return super(Slice, cls).__new__(
            cls,
            asset=asset,
            inputs=[term],
            window_length=0,
            mask=term.mask,
            dtype=term.dtype,
            missing_value=term.missing_value,
            window_safe=term.window_safe,
            ndim=1,
        )

    def __repr__(self):
        return "{type}({parent_term}, column={asset})".format(
            type=type(self).__name__,
            parent_term=type(self.inputs[0]).__name__,
            asset=self._asset,
        )

    def _init(self, asset, *args, **kwargs):
        self._asset = asset
        return super(Slice, self)._init(*args, **kwargs)

    @classmethod
    def _static_identity(cls, asset, *args, **kwargs):
        return (super(Slice, cls)._static_identity(*args, **kwargs), asset)

    def _compute(self, windows, dates, assets, mask):
        asset = self._asset
        asset_column = searchsorted(assets.values, asset.sid)
        if assets[asset_column] != asset.sid:
            raise NonExistentAssetInTimeFrame(
                asset=asset, start_date=dates[0], end_date=dates[-1],
            )

        # Return a 2D array with one column rather than a 1D array of the
        # column.
        return windows[0][:, [asset_column]]

    @property
    def _downsampled_type(self):
        raise NotImplementedError(
            'downsampling of slices is not yet supported'
        )


def validate_dtype(termname, dtype, missing_value):
    """
    Validate a `dtype` and `missing_value` passed to Term.__new__.

    Ensures that we know how to represent ``dtype``, and that missing_value
    is specified for types without default missing values.

    Returns
    -------
    validated_dtype, validated_missing_value : np.dtype, any
        The dtype and missing_value to use for the new term.

    Raises
    ------
    DTypeNotSpecified
        When no dtype was passed to the instance, and the class doesn't
        provide a default.
    NotDType
        When either the class or the instance provides a value not
        coercible to a numpy dtype.
    NoDefaultMissingValue
        When dtype requires an explicit missing_value, but
        ``missing_value`` is NotSpecified.
    """
    if dtype is NotSpecified:
        raise DTypeNotSpecified(termname=termname)

    try:
        dtype = dtype_class(dtype)
    except TypeError:
        raise NotDType(dtype=dtype, termname=termname)

    if not can_represent_dtype(dtype):
        raise UnsupportedDType(dtype=dtype, termname=termname)

    if missing_value is NotSpecified:
        missing_value = default_missing_value_for_dtype(dtype)

    try:
        if (dtype == categorical_dtype):
            # This check is necessary because we use object dtype for
            # categoricals, and numpy will allow us to promote numerical
            # values to object even though we don't support them.
            _assert_valid_categorical_missing_value(missing_value)

        # For any other type, we can check if the missing_value is safe by
        # making an array of that value and trying to safely convert it to
        # the desired type.
        # 'same_kind' allows casting between things like float32 and
        # float64, but not str and int.
        array([missing_value]).astype(dtype=dtype, casting='same_kind')
    except TypeError as e:
        raise TypeError(
            "Missing value {value!r} is not a valid choice "
            "for term {termname} with dtype {dtype}.\n\n"
            "Coercion attempt failed with: {error}".format(
                termname=termname,
                value=missing_value,
                dtype=dtype,
                error=e,
            )
        )

    return dtype, missing_value


def _assert_valid_categorical_missing_value(value):
    """
    Check that value is a valid categorical missing_value.

    Raises a TypeError if the value is cannot be used as the missing_value for
    a categorical_dtype Term.
    """
    label_types = LabelArray.SUPPORTED_SCALAR_TYPES
    if not isinstance(value, label_types):
        raise TypeError(
            "Categorical terms must have missing values of type "
            "{types}.".format(
                types=' or '.join([t.__name__ for t in label_types]),
            )
        )
