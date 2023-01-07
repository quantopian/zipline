"""
Tests for Term.
"""
from collections import Counter
from itertools import product

from toolz import assoc
import pandas as pd

from zipline.assets import Asset, ExchangeInfo
from zipline.errors import (
    DTypeNotSpecified,
    InvalidOutputName,
    NonWindowSafeInput,
    NotDType,
    TermInputsNotSpecified,
    NonPipelineInputs,
    TermOutputsEmpty,
    UnsupportedDType,
    WindowLengthNotSpecified,
)
from zipline.pipeline import (
    Classifier,
    CustomClassifier,
    CustomFactor,
    Factor,
    Filter,
    ExecutionPlan,
)
from zipline.pipeline.data import Column, DataSet
from zipline.pipeline.data.testing import TestingDataSet
from zipline.pipeline.domain import US_EQUITIES
from zipline.pipeline.expression import NUMEXPR_MATH_FUNCS
from zipline.pipeline.factors import RecarrayField
from zipline.pipeline.sentinels import NotSpecified
from zipline.pipeline.term import AssetExists, LoadableTerm
from zipline.testing.fixtures import WithTradingSessions, ZiplineTestCase
from zipline.testing.predicates import assert_equal
from zipline.utils.numpy_utils import (
    bool_dtype,
    categorical_dtype,
    complex128_dtype,
    datetime64ns_dtype,
    float64_dtype,
    int64_dtype,
    NoDefaultMissingValue,
)
import pytest
import re


class SomeDataSet(DataSet):
    foo = Column(float64_dtype)
    bar = Column(float64_dtype)
    buzz = Column(float64_dtype)


class SubDataSet(SomeDataSet):
    pass


class SubDataSetNewCol(SomeDataSet):
    qux = Column(float64_dtype)


class SomeFactor(Factor):
    dtype = float64_dtype
    window_length = 5
    inputs = [SomeDataSet.foo, SomeDataSet.bar]


SomeFactorAlias = SomeFactor


class SomeOtherFactor(Factor):
    dtype = float64_dtype
    window_length = 5
    inputs = [SomeDataSet.bar, SomeDataSet.buzz]


class DateFactor(Factor):
    dtype = datetime64ns_dtype
    window_length = 5
    inputs = [SomeDataSet.bar, SomeDataSet.buzz]


class NoLookbackFactor(Factor):
    dtype = float64_dtype
    window_length = 0


class GenericCustomFactor(CustomFactor):
    dtype = float64_dtype
    window_length = 5
    inputs = [SomeDataSet.foo]


class MultipleOutputs(CustomFactor):
    dtype = float64_dtype
    window_length = 5
    inputs = [SomeDataSet.foo, SomeDataSet.bar]
    outputs = ["alpha", "beta"]

    def some_method(self):
        return


class GenericFilter(Filter):
    dtype = bool_dtype
    window_length = 0
    inputs = []


class GenericClassifier(Classifier):
    dtype = categorical_dtype
    window_length = 0
    inputs = []


def gen_equivalent_factors():
    """
    Return an iterator of SomeFactor instances that should all be the same
    object.
    """
    yield SomeFactor()
    yield SomeFactor(inputs=NotSpecified)
    yield SomeFactor(SomeFactor.inputs)
    yield SomeFactor(inputs=SomeFactor.inputs)
    yield SomeFactor([SomeDataSet.foo, SomeDataSet.bar])
    yield SomeFactor(window_length=SomeFactor.window_length)
    yield SomeFactor(window_length=NotSpecified)
    yield SomeFactor(
        [SomeDataSet.foo, SomeDataSet.bar],
        window_length=NotSpecified,
    )
    yield SomeFactor(
        [SomeDataSet.foo, SomeDataSet.bar],
        window_length=SomeFactor.window_length,
    )
    yield SomeFactorAlias()


def to_dict(a_list):
    """
    Convert a list to a dict with keys drawn from '0', '1', '2', ...

    Examples
    --------
    >>> to_dict([2, 3, 4])  # doctest: +SKIP
    {'0': 2, '1': 3, '2': 4}
    """
    return dict(zip(map(str, range(len(a_list))), a_list))


class DependencyResolutionTestCase(WithTradingSessions, ZiplineTestCase):

    TRADING_CALENDAR_STRS = ("NYSE",)
    START_DATE = pd.Timestamp("2014-01-02")
    END_DATE = pd.Timestamp("2014-12-31")

    execution_plan_start = pd.Timestamp("2014-06-01", tz="UTC")
    execution_plan_end = pd.Timestamp("2014-06-30", tz="UTC")

    DOMAIN = US_EQUITIES

    def check_dependency_order(self, ordered_terms):
        seen = set()

        for term in ordered_terms:
            for dep in term.dependencies:
                # LoadableTerms should be specialized do the domain of
                # execution when emitted by an execution plan.
                if isinstance(dep, LoadableTerm):
                    assert dep.specialize(self.DOMAIN) in seen
                else:
                    assert dep in seen

            seen.add(term)

    def make_execution_plan(self, terms):
        return ExecutionPlan(
            domain=self.DOMAIN,
            terms=terms,
            start_date=self.execution_plan_start,
            end_date=self.execution_plan_end,
        )

    def test_single_factor(self):
        """
        Test dependency resolution for a single factor.
        """

        def check_output(graph):

            resolution_order = list(graph.ordered())

            # Loadable terms should get specialized during graph construction.
            specialized_foo = SomeDataSet.foo.specialize(self.DOMAIN)
            specialized_bar = SomeDataSet.foo.specialize(self.DOMAIN)

            assert len(resolution_order) == 4
            self.check_dependency_order(resolution_order)
            assert AssetExists() in resolution_order
            assert specialized_foo in resolution_order
            assert specialized_bar in resolution_order
            assert SomeFactor() in resolution_order

            assert graph.graph.nodes[specialized_foo]["extra_rows"] == 4
            assert graph.graph.nodes[specialized_bar]["extra_rows"] == 4

        for foobar in gen_equivalent_factors():
            check_output(self.make_execution_plan(to_dict([foobar])))

    def test_single_factor_instance_args(self):
        """
        Test dependency resolution for a single factor with arguments passed to
        the constructor.
        """
        bar, buzz = SomeDataSet.bar, SomeDataSet.buzz

        factor = SomeFactor([bar, buzz], window_length=5)
        graph = self.make_execution_plan(to_dict([factor]))

        resolution_order = list(graph.ordered())

        # SomeFactor, its inputs, and AssetExists()
        assert len(resolution_order) == 4
        self.check_dependency_order(resolution_order)
        assert AssetExists() in resolution_order
        assert graph.extra_rows[AssetExists()] == 4

        # LoadableTerms should be specialized to our domain in the execution
        # order.
        assert bar.specialize(self.DOMAIN) in resolution_order
        assert buzz.specialize(self.DOMAIN) in resolution_order

        # ComputableTerms don't yet have a notion of specialization, so they
        # shouldn't appear unchanged in the execution order.
        assert SomeFactor([bar, buzz], window_length=5) in resolution_order

        assert graph.extra_rows[bar.specialize(self.DOMAIN)] == 4
        assert graph.extra_rows[bar.specialize(self.DOMAIN)] == 4

    def test_reuse_loadable_terms(self):
        """
        Test that raw inputs only show up in the dependency graph once.
        """
        f1 = SomeFactor([SomeDataSet.foo, SomeDataSet.bar])
        f2 = SomeOtherFactor([SomeDataSet.bar, SomeDataSet.buzz])

        graph = self.make_execution_plan(to_dict([f1, f2]))
        resolution_order = list(graph.ordered())

        # bar should only appear once.
        assert len(resolution_order) == 6
        assert len(set(resolution_order)) == 6
        self.check_dependency_order(resolution_order)

    def test_disallow_recursive_lookback(self):

        with pytest.raises(NonWindowSafeInput):
            SomeFactor(inputs=[SomeFactor(), SomeDataSet.foo])

    def test_window_safety_one_window_length(self):
        """
        Test that window safety problems are only raised if
        the parent factor has window length greater than 1
        """
        with pytest.raises(NonWindowSafeInput):
            SomeFactor(inputs=[SomeOtherFactor()])

        SomeFactor(inputs=[SomeOtherFactor()], window_length=1)


class TestObjectIdentity:
    def assertSameObject(self, *objs):
        first = objs[0]
        for obj in objs:
            assert first is obj

    def assertDifferentObjects(self, *objs):
        id_counts = Counter(map(id, objs))
        ((most_common_id, count),) = id_counts.most_common(1)
        if count > 1:
            dupe = [o for o in objs if id(o) == most_common_id][0]
            self.fail("%s appeared %d times in %s" % (dupe, count, objs))

    def test_instance_caching(self):

        self.assertSameObject(*gen_equivalent_factors())
        assert SomeFactor(window_length=SomeFactor.window_length + 1) is SomeFactor(
            window_length=SomeFactor.window_length + 1
        )

        assert SomeFactor(dtype=float64_dtype) is SomeFactor(dtype=float64_dtype)

        assert SomeFactor(
            inputs=[SomeFactor.inputs[1], SomeFactor.inputs[0]]
        ) is SomeFactor(inputs=[SomeFactor.inputs[1], SomeFactor.inputs[0]])

        mask = SomeFactor() + SomeOtherFactor()
        assert SomeFactor(mask=mask) is SomeFactor(mask=mask)

    def test_instance_caching_multiple_outputs(self):
        assert MultipleOutputs() is MultipleOutputs()
        assert MultipleOutputs() is MultipleOutputs(outputs=MultipleOutputs.outputs)
        assert MultipleOutputs(
            outputs=[
                MultipleOutputs.outputs[1],
                MultipleOutputs.outputs[0],
            ],
        ) is MultipleOutputs(
            outputs=[
                MultipleOutputs.outputs[1],
                MultipleOutputs.outputs[0],
            ],
        )

        # Ensure that both methods of accessing our outputs return the same
        # things.
        multiple_outputs = MultipleOutputs()
        alpha, beta = MultipleOutputs()
        assert alpha is multiple_outputs.alpha
        assert beta is multiple_outputs.beta

    def test_instance_caching_of_slices(self):
        my_asset = Asset(
            1,
            exchange_info=ExchangeInfo("TEST FULL", "TEST", "US"),
        )

        f = GenericCustomFactor()
        f_slice = f[my_asset]
        assert f_slice is type(f_slice)(GenericCustomFactor(), my_asset)

        filt = GenericFilter()
        filt_slice = filt[my_asset]
        assert filt_slice is type(filt_slice)(GenericFilter(), my_asset)

        c = GenericClassifier()
        c_slice = c[my_asset]
        assert c_slice is type(c_slice)(GenericClassifier(), my_asset)

    def test_instance_non_caching(self):

        f = SomeFactor()

        # Different window_length.
        assert f is not SomeFactor(window_length=SomeFactor.window_length + 1)

        # Different dtype
        assert f is not SomeFactor(dtype=datetime64ns_dtype)

        # Reordering inputs changes semantics.
        assert f is not SomeFactor(inputs=[SomeFactor.inputs[1], SomeFactor.inputs[0]])

    def test_instance_non_caching_redefine_class(self):

        orig_foobar_instance = SomeFactorAlias()

        class SomeFactor(Factor):
            dtype = float64_dtype
            window_length = 5
            inputs = [SomeDataSet.foo, SomeDataSet.bar]

        assert orig_foobar_instance is not SomeFactor()

    def test_instance_non_caching_multiple_outputs(self):
        multiple_outputs = MultipleOutputs()

        # Different outputs.
        assert MultipleOutputs() is not MultipleOutputs(outputs=["beta", "gamma"])

        # Reordering outputs.
        assert multiple_outputs is not MultipleOutputs(
            outputs=[
                MultipleOutputs.outputs[1],
                MultipleOutputs.outputs[0],
            ],
        )

        # Different factors sharing an output name should produce different
        # RecarrayField factors.
        orig_beta = multiple_outputs.beta
        beta, gamma = MultipleOutputs(outputs=["beta", "gamma"])
        assert beta is not orig_beta

    def test_instance_caching_binops(self):
        f = SomeFactor()
        g = SomeOtherFactor()
        for lhs, rhs in product([f, g], [f, g]):
            assert (lhs + rhs) is (lhs + rhs)
            assert (lhs - rhs) is (lhs - rhs)
            assert (lhs * rhs) is (lhs * rhs)
            assert (lhs / rhs) is (lhs / rhs)
            assert (lhs**rhs) is (lhs**rhs)

        assert (1 + rhs) is (1 + rhs)
        assert (rhs + 1) is (rhs + 1)

        assert (1 - rhs) is (1 - rhs)
        assert (rhs - 1) is (rhs - 1)

        assert (2 * rhs) is (2 * rhs)
        assert (rhs * 2) is (rhs * 2)

        assert (2 / rhs) is (2 / rhs)
        assert (rhs / 2) is (rhs / 2)

        assert (2**rhs) is (2**rhs)
        assert (rhs**2) is (rhs**2)

        assert (f + g) + (f + g) is (f + g) + (f + g)

    def test_instance_caching_unary_ops(self):
        f = SomeFactor()
        assert -f is -f
        assert --f is --f
        assert ---f is ---f

    def test_instance_caching_math_funcs(self):
        f = SomeFactor()
        for funcname in NUMEXPR_MATH_FUNCS:
            method = getattr(f, funcname)
            assert method() is method()

    def test_instance_caching_grouped_transforms(self):
        f = SomeFactor()
        c = GenericClassifier()
        m = GenericFilter()

        for meth in f.demean, f.zscore, f.rank:
            assert meth() is meth()
            assert meth(groupby=c) is meth(groupby=c)
            assert meth(mask=m) is meth(mask=m)
            assert meth(groupby=c, mask=m) is meth(groupby=c, mask=m)

    class SomeFactorParameterized(SomeFactor):
        params = ("a", "b")

    def test_parameterized_term(self):

        f = self.SomeFactorParameterized(a=1, b=2)
        assert f.params == {"a": 1, "b": 2}

        g = self.SomeFactorParameterized(a=1, b=3)
        h = self.SomeFactorParameterized(a=2, b=2)
        self.assertDifferentObjects(f, g, h)

        f2 = self.SomeFactorParameterized(a=1, b=2)
        f3 = self.SomeFactorParameterized(b=2, a=1)
        self.assertSameObject(f, f2, f3)

        assert f.params["a"] == 1
        assert f.params["b"] == 2
        assert f.window_length == SomeFactor.window_length
        assert f.inputs == tuple(SomeFactor.inputs)

    def test_parameterized_term_non_hashable_arg(self):
        err_msg = (
            "SomeFactorParameterized expected a hashable value "
            "for parameter 'a', but got [] instead."
        )
        with pytest.raises(TypeError, match=re.escape(err_msg)):
            self.SomeFactorParameterized(a=[], b=1)

        err_msg = (
            "SomeFactorParameterized expected a hashable value "
            "for parameter 'b', but got [] instead."
        )
        with pytest.raises(TypeError, match=re.escape(err_msg)):
            self.SomeFactorParameterized(a=1, b=[])
        err_msg = (
            r"SomeFactorParameterized expected a hashable value "
            r"for parameter '(a|b)', but got \[\] instead\."
        )
        with pytest.raises(TypeError, match=err_msg):
            self.SomeFactorParameterized(a=[], b=[])

    def test_parameterized_term_default_value(self):
        defaults = {"a": "default for a", "b": "default for b"}

        class F(Factor):
            params = defaults

            inputs = (SomeDataSet.foo,)
            dtype = "f8"
            window_length = 5

        assert_equal(F().params, defaults)
        assert_equal(F(a="new a").params, assoc(defaults, "a", "new a"))
        assert_equal(F(b="new b").params, assoc(defaults, "b", "new b"))
        assert_equal(
            F(a="new a", b="new b").params,
            {"a": "new a", "b": "new b"},
        )

    def test_parameterized_term_default_value_with_not_specified(self):
        defaults = {"a": "default for a", "b": NotSpecified}

        class F(Factor):
            params = defaults

            inputs = (SomeDataSet.foo,)
            dtype = "f8"
            window_length = 5

        pattern = r"F expected a keyword parameter 'b'\."
        with pytest.raises(TypeError, match=pattern):
            F()
        with pytest.raises(TypeError, match=pattern):
            F(a="new a")

        assert_equal(F(b="new b").params, assoc(defaults, "b", "new b"))
        assert_equal(
            F(a="new a", b="new b").params,
            {"a": "new a", "b": "new b"},
        )

    def test_bad_input(self):
        class SomeFactor(Factor):
            dtype = float64_dtype

        class SomeFactorDefaultInputs(SomeFactor):
            inputs = (SomeDataSet.foo, SomeDataSet.bar)

        class SomeFactorDefaultLength(SomeFactor):
            window_length = 10

        class SomeFactorNoDType(SomeFactor):
            window_length = 10
            inputs = (SomeDataSet.foo,)
            dtype = NotSpecified

        with pytest.raises(TermInputsNotSpecified):
            SomeFactor(window_length=1)

        with pytest.raises(TermInputsNotSpecified):
            SomeFactorDefaultLength()

        with pytest.raises(NonPipelineInputs):
            SomeFactor(window_length=1, inputs=[2])

        with pytest.raises(WindowLengthNotSpecified):
            SomeFactor(inputs=(SomeDataSet.foo,))

        with pytest.raises(WindowLengthNotSpecified):
            SomeFactorDefaultInputs()

        with pytest.raises(DTypeNotSpecified):
            SomeFactorNoDType()

        with pytest.raises(NotDType):
            SomeFactor(dtype=1)

        with pytest.raises(NoDefaultMissingValue):
            SomeFactor(dtype=int64_dtype)

        with pytest.raises(UnsupportedDType):
            SomeFactor(dtype=complex128_dtype)

        with pytest.raises(TermOutputsEmpty):
            MultipleOutputs(outputs=[])

    def test_bad_output_access(self):
        with pytest.raises(
            AttributeError, match="'SomeFactor' object has no attribute 'not_an_attr'"
        ):
            SomeFactor().not_an_attr

        mo = MultipleOutputs()
        expected = (
            "Instance of MultipleOutputs has no output named 'not_an_attr'. "
            "Possible choices are: \\('alpha', 'beta'\\)."
        )
        with pytest.raises(AttributeError, match=expected):
            mo.not_an_attr

        with pytest.raises(
            ValueError, match="GenericCustomFactor does not have multiple outputs."
        ):
            alpha, beta = GenericCustomFactor()

        # Public method, user-defined method.
        # Accessing these attributes should return the output, not the method.
        conflicting_output_names = ["zscore", "some_method"]

        mo = MultipleOutputs(outputs=conflicting_output_names)
        for name in conflicting_output_names:
            assert isinstance(getattr(mo, name), RecarrayField)

        # Non-callable attribute, private method, special method.
        disallowed_output_names = ["inputs", "_init", "__add__"]

        for name in disallowed_output_names:
            with pytest.raises(InvalidOutputName):
                GenericCustomFactor(outputs=[name])

    def test_require_super_call_in_validate(self):
        class MyFactor(Factor):
            inputs = ()
            dtype = float64_dtype
            window_length = 0

            def _validate(self):
                "Woops, I didn't call super()!"

        err_msg = (
            "Term._validate() was not called.\n"
            "This probably means that you overrode _validate"
            " without calling super()."
        )
        with pytest.raises(AssertionError, match=re.escape(err_msg)):
            MyFactor()

    def test_latest_on_different_dtypes(self):
        factor_dtypes = (float64_dtype, datetime64ns_dtype)
        for column in TestingDataSet.columns:
            if column.dtype == bool_dtype:
                assert isinstance(column.latest, Filter)
            elif column.dtype == int64_dtype or column.dtype.kind in ("O", "S", "U"):
                assert isinstance(column.latest, Classifier)
            elif column.dtype in factor_dtypes:
                assert isinstance(column.latest, Factor)
            else:
                self.fail("Unknown dtype %s for column %s" % (column.dtype, column))
            # These should be the same value, plus this has the convenient
            # property of correctly handling `NaN`.
            assert column.missing_value is column.latest.missing_value

    def test_failure_timing_on_bad_dtypes(self):

        # Just constructing a bad column shouldn't fail.
        Column(dtype=int64_dtype)

        expected_msg = "Failed to create Column with name 'bad_column'"
        with pytest.raises(NoDefaultMissingValue, match=expected_msg):

            class BadDataSet(DataSet):
                bad_column = Column(dtype=int64_dtype)
                float_column = Column(dtype=float64_dtype)
                int_column = Column(dtype=int64_dtype, missing_value=3)

        Column(dtype=complex128_dtype)
        with pytest.raises(UnsupportedDType):

            class BadDataSetComplex(DataSet):
                bad_column = Column(dtype=complex128_dtype)
                float_column = Column(dtype=float64_dtype)
                int_column = Column(dtype=int64_dtype, missing_value=3)


class TestSubDataSet:
    def test_subdataset(self):
        some_dataset_map = {column.name: column for column in SomeDataSet.columns}
        sub_dataset_map = {column.name: column for column in SubDataSet.columns}
        assert {column.name for column in SomeDataSet.columns} == {
            column.name for column in SubDataSet.columns
        }
        for k, some_dataset_column in some_dataset_map.items():
            sub_dataset_column = sub_dataset_map[k]
            assert some_dataset_column is not sub_dataset_column, (
                "subclass column %r should not have the same identity as"
                " the parent" % k
            )
            assert some_dataset_column.dtype == sub_dataset_column.dtype, (
                "subclass column %r should have the same dtype as the parent" % k
            )

    def test_add_column(self):
        some_dataset_map = {column.name: column for column in SomeDataSet.columns}
        sub_dataset_new_col_map = {
            column.name: column for column in SubDataSetNewCol.columns
        }
        sub_col_names = {column.name for column in SubDataSetNewCol.columns}

        # check our extra col
        assert "qux" in sub_col_names
        assert sub_dataset_new_col_map["qux"].dtype == float64_dtype

        assert {column.name for column in SomeDataSet.columns} == sub_col_names - {
            "qux"
        }
        for k, some_dataset_column in some_dataset_map.items():
            sub_dataset_column = sub_dataset_new_col_map[k]
            assert some_dataset_column is not sub_dataset_column, (
                "subclass column %r should not have the same identity as"
                " the parent" % k
            )
            assert some_dataset_column.dtype == sub_dataset_column.dtype, (
                "subclass column %r should have the same dtype as the parent" % k
            )

    @pytest.mark.parametrize(
        "dtype_, outputs_", [(categorical_dtype, ("a",)), (int64_dtype, ("a", "b"))]
    )
    def test_reject_multi_output_classifiers(self, dtype_, outputs_):
        """
        Multi-output CustomClassifiers don't work because they use special
        output allocation for string arrays.
        """

        class SomeClassifier(CustomClassifier):
            dtype = dtype_
            window_length = 5
            inputs = [SomeDataSet.foo, SomeDataSet.bar]
            outputs = outputs_
            missing_value = dtype_.type("123")

        expected_error = (
            f"SomeClassifier does not support custom outputs, "
            f"but received custom outputs={outputs_}."
        )
        with pytest.raises(ValueError, match=re.escape(expected_error)):
            SomeClassifier()

    def test_unreasonable_missing_values(self):

        for base_type, dtype_, bad_mv in (
            (Factor, float64_dtype, "ayy"),
            (Filter, bool_dtype, "lmao"),
            (Classifier, int64_dtype, "lolwut"),
            (Classifier, categorical_dtype, 7),
        ):

            class SomeTerm(base_type):
                inputs = ()
                window_length = 0
                missing_value = bad_mv
                dtype = dtype_

            prefix = (
                "^Missing value {mv!r} is not a valid choice "
                "for term SomeTerm with dtype {dtype}.\n\n"
                "Coercion attempt failed with:"
            ).format(mv=bad_mv, dtype=dtype_)

            with pytest.raises(TypeError, match=prefix):
                SomeTerm()
