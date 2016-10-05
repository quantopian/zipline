"""
Tests for Term.
"""
from collections import Counter
from itertools import product
from unittest import TestCase

from toolz import assoc
import pandas as pd

from zipline.assets import Asset
from zipline.errors import (
    DTypeNotSpecified,
    InvalidOutputName,
    NonWindowSafeInput,
    NotDType,
    TermInputsNotSpecified,
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
from zipline.pipeline.expression import NUMEXPR_MATH_FUNCS
from zipline.pipeline.factors import RecarrayField
from zipline.pipeline.sentinels import NotSpecified
from zipline.pipeline.term import AssetExists, Slice
from zipline.testing import parameter_space
from zipline.testing.fixtures import WithTradingSessions, ZiplineTestCase
from zipline.testing.predicates import (
    assert_equal,
    assert_raises,
    assert_raises_regex,
    assert_regex,
)
from zipline.utils.numpy_utils import (
    bool_dtype,
    categorical_dtype,
    complex128_dtype,
    datetime64ns_dtype,
    float64_dtype,
    int64_dtype,
    NoDefaultMissingValue,
)


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
    outputs = ['alpha', 'beta']

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


def to_dict(l):
    """
    Convert a list to a dict with keys drawn from '0', '1', '2', ...

    Example
    -------
    >>> to_dict([2, 3, 4])  # doctest: +SKIP
    {'0': 2, '1': 3, '2': 4}
    """
    return dict(zip(map(str, range(len(l))), l))


class DependencyResolutionTestCase(WithTradingSessions, ZiplineTestCase):

    TRADING_CALENDAR_STRS = ('NYSE',)
    START_DATE = pd.Timestamp('2014-01-02', tz='UTC')
    END_DATE = pd.Timestamp('2014-12-31', tz='UTC')

    execution_plan_start = pd.Timestamp('2014-06-01', tz='UTC')
    execution_plan_end = pd.Timestamp('2014-06-30', tz='UTC')

    def check_dependency_order(self, ordered_terms):
        seen = set()

        for term in ordered_terms:
            for dep in term.dependencies:
                self.assertIn(dep, seen)

            seen.add(term)

    def make_execution_plan(self, terms):
        return ExecutionPlan(
            terms,
            self.nyse_sessions,
            self.execution_plan_start,
            self.execution_plan_end,
        )

    def test_single_factor(self):
        """
        Test dependency resolution for a single factor.
        """
        def check_output(graph):

            resolution_order = list(graph.ordered())

            self.assertEqual(len(resolution_order), 4)
            self.check_dependency_order(resolution_order)
            self.assertIn(AssetExists(), resolution_order)
            self.assertIn(SomeDataSet.foo, resolution_order)
            self.assertIn(SomeDataSet.bar, resolution_order)
            self.assertIn(SomeFactor(), resolution_order)

            self.assertEqual(
                graph.graph.node[SomeDataSet.foo]['extra_rows'],
                4,
            )
            self.assertEqual(
                graph.graph.node[SomeDataSet.bar]['extra_rows'],
                4,
            )

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
        self.assertEqual(len(resolution_order), 4)
        self.check_dependency_order(resolution_order)
        self.assertIn(AssetExists(), resolution_order)
        self.assertEqual(graph.extra_rows[AssetExists()], 4)

        self.assertIn(bar, resolution_order)
        self.assertIn(buzz, resolution_order)
        self.assertIn(SomeFactor([bar, buzz], window_length=5),
                      resolution_order)
        self.assertEqual(graph.extra_rows[bar], 4)
        self.assertEqual(graph.extra_rows[buzz], 4)

    def test_reuse_loadable_terms(self):
        """
        Test that raw inputs only show up in the dependency graph once.
        """
        f1 = SomeFactor([SomeDataSet.foo, SomeDataSet.bar])
        f2 = SomeOtherFactor([SomeDataSet.bar, SomeDataSet.buzz])

        graph = self.make_execution_plan(to_dict([f1, f2]))
        resolution_order = list(graph.ordered())

        # bar should only appear once.
        self.assertEqual(len(resolution_order), 6)
        self.assertEqual(len(set(resolution_order)), 6)
        self.check_dependency_order(resolution_order)

    def test_disallow_recursive_lookback(self):

        with self.assertRaises(NonWindowSafeInput):
            SomeFactor(inputs=[SomeFactor(), SomeDataSet.foo])


class ObjectIdentityTestCase(TestCase):

    def assertSameObject(self, *objs):
        first = objs[0]
        for obj in objs:
            self.assertIs(first, obj)

    def assertDifferentObjects(self, *objs):
        id_counts = Counter(map(id, objs))
        ((most_common_id, count),) = id_counts.most_common(1)
        if count > 1:
            dupe = [o for o in objs if id(o) == most_common_id][0]
            self.fail("%s appeared %d times in %s" % (dupe, count, objs))

    def test_instance_caching(self):

        self.assertSameObject(*gen_equivalent_factors())
        self.assertIs(
            SomeFactor(window_length=SomeFactor.window_length + 1),
            SomeFactor(window_length=SomeFactor.window_length + 1),
        )

        self.assertIs(
            SomeFactor(dtype=float64_dtype),
            SomeFactor(dtype=float64_dtype),
        )

        self.assertIs(
            SomeFactor(inputs=[SomeFactor.inputs[1], SomeFactor.inputs[0]]),
            SomeFactor(inputs=[SomeFactor.inputs[1], SomeFactor.inputs[0]]),
        )

        mask = SomeFactor() + SomeOtherFactor()
        self.assertIs(SomeFactor(mask=mask), SomeFactor(mask=mask))

    def test_instance_caching_multiple_outputs(self):
        self.assertIs(MultipleOutputs(), MultipleOutputs())
        self.assertIs(
            MultipleOutputs(),
            MultipleOutputs(outputs=MultipleOutputs.outputs),
        )
        self.assertIs(
            MultipleOutputs(
                outputs=[
                    MultipleOutputs.outputs[1], MultipleOutputs.outputs[0],
                ],
            ),
            MultipleOutputs(
                outputs=[
                    MultipleOutputs.outputs[1], MultipleOutputs.outputs[0],
                ],
            ),
        )

        # Ensure that both methods of accessing our outputs return the same
        # things.
        multiple_outputs = MultipleOutputs()
        alpha, beta = MultipleOutputs()
        self.assertIs(alpha, multiple_outputs.alpha)
        self.assertIs(beta, multiple_outputs.beta)

    def test_instance_caching_of_slices(self):
        my_asset = Asset(1, exchange="TEST")

        f = GenericCustomFactor()
        f_slice = f[my_asset]
        self.assertIs(f_slice, Slice(GenericCustomFactor(), my_asset))

        f = GenericFilter()
        f_slice = f[my_asset]
        self.assertIs(f_slice, Slice(GenericFilter(), my_asset))

        c = GenericClassifier()
        c_slice = c[my_asset]
        self.assertIs(c_slice, Slice(GenericClassifier(), my_asset))

    def test_instance_non_caching(self):

        f = SomeFactor()

        # Different window_length.
        self.assertIsNot(
            f,
            SomeFactor(window_length=SomeFactor.window_length + 1),
        )

        # Different dtype
        self.assertIsNot(
            f,
            SomeFactor(dtype=datetime64ns_dtype)
        )

        # Reordering inputs changes semantics.
        self.assertIsNot(
            f,
            SomeFactor(inputs=[SomeFactor.inputs[1], SomeFactor.inputs[0]]),
        )

    def test_instance_non_caching_redefine_class(self):

        orig_foobar_instance = SomeFactorAlias()

        class SomeFactor(Factor):
            dtype = float64_dtype
            window_length = 5
            inputs = [SomeDataSet.foo, SomeDataSet.bar]

        self.assertIsNot(orig_foobar_instance, SomeFactor())

    def test_instance_non_caching_multiple_outputs(self):
        multiple_outputs = MultipleOutputs()

        # Different outputs.
        self.assertIsNot(
            MultipleOutputs(), MultipleOutputs(outputs=['beta', 'gamma']),
        )

        # Reordering outputs.
        self.assertIsNot(
            multiple_outputs,
            MultipleOutputs(
                outputs=[
                    MultipleOutputs.outputs[1], MultipleOutputs.outputs[0],
                ],
            ),
        )

        # Different factors sharing an output name should produce different
        # RecarrayField factors.
        orig_beta = multiple_outputs.beta
        beta, gamma = MultipleOutputs(outputs=['beta', 'gamma'])
        self.assertIsNot(beta, orig_beta)

    def test_instance_caching_binops(self):
        f = SomeFactor()
        g = SomeOtherFactor()
        for lhs, rhs in product([f, g], [f, g]):
            self.assertIs((lhs + rhs), (lhs + rhs))
            self.assertIs((lhs - rhs), (lhs - rhs))
            self.assertIs((lhs * rhs), (lhs * rhs))
            self.assertIs((lhs / rhs), (lhs / rhs))
            self.assertIs((lhs ** rhs), (lhs ** rhs))

        self.assertIs((1 + rhs), (1 + rhs))
        self.assertIs((rhs + 1), (rhs + 1))

        self.assertIs((1 - rhs), (1 - rhs))
        self.assertIs((rhs - 1), (rhs - 1))

        self.assertIs((2 * rhs), (2 * rhs))
        self.assertIs((rhs * 2), (rhs * 2))

        self.assertIs((2 / rhs), (2 / rhs))
        self.assertIs((rhs / 2), (rhs / 2))

        self.assertIs((2 ** rhs), (2 ** rhs))
        self.assertIs((rhs ** 2), (rhs ** 2))

        self.assertIs((f + g) + (f + g), (f + g) + (f + g))

    def test_instance_caching_unary_ops(self):
        f = SomeFactor()
        self.assertIs(-f, -f)
        self.assertIs(--f, --f)
        self.assertIs(---f, ---f)

    def test_instance_caching_math_funcs(self):
        f = SomeFactor()
        for funcname in NUMEXPR_MATH_FUNCS:
            method = getattr(f, funcname)
            self.assertIs(method(), method())

    def test_instance_caching_grouped_transforms(self):
        f = SomeFactor()
        c = GenericClassifier()
        m = GenericFilter()

        for meth in f.demean, f.zscore, f.rank:
            self.assertIs(meth(), meth())
            self.assertIs(meth(groupby=c), meth(groupby=c))
            self.assertIs(meth(mask=m), meth(mask=m))
            self.assertIs(meth(groupby=c, mask=m), meth(groupby=c, mask=m))

    class SomeFactorParameterized(SomeFactor):
        params = ('a', 'b')

    def test_parameterized_term(self):

        f = self.SomeFactorParameterized(a=1, b=2)
        self.assertEqual(f.params, {'a': 1, 'b': 2})

        g = self.SomeFactorParameterized(a=1, b=3)
        h = self.SomeFactorParameterized(a=2, b=2)
        self.assertDifferentObjects(f, g, h)

        f2 = self.SomeFactorParameterized(a=1, b=2)
        f3 = self.SomeFactorParameterized(b=2, a=1)
        self.assertSameObject(f, f2, f3)

        self.assertEqual(f.params['a'], 1)
        self.assertEqual(f.params['b'], 2)
        self.assertEqual(f.window_length, SomeFactor.window_length)
        self.assertEqual(f.inputs, tuple(SomeFactor.inputs))

    def test_parameterized_term_non_hashable_arg(self):
        with assert_raises(TypeError) as e:
            self.SomeFactorParameterized(a=[], b=1)
        assert_equal(
            str(e.exception),
            "SomeFactorParameterized expected a hashable value for parameter"
            " 'a', but got [] instead.",
        )

        with assert_raises(TypeError) as e:
            self.SomeFactorParameterized(a=1, b=[])
        assert_equal(
            str(e.exception),
            "SomeFactorParameterized expected a hashable value for parameter"
            " 'b', but got [] instead.",
        )

        with assert_raises(TypeError) as e:
            self.SomeFactorParameterized(a=[], b=[])
        assert_regex(
            str(e.exception),
            r"SomeFactorParameterized expected a hashable value for parameter"
            r" '(a|b)', but got \[\] instead\.",
        )

    def test_parameterized_term_default_value(self):
        defaults = {'a': 'default for a', 'b': 'default for b'}

        class F(Factor):
            params = defaults

            inputs = (SomeDataSet.foo,)
            dtype = 'f8'
            window_length = 5

        assert_equal(F().params, defaults)
        assert_equal(F(a='new a').params, assoc(defaults, 'a', 'new a'))
        assert_equal(F(b='new b').params, assoc(defaults, 'b', 'new b'))
        assert_equal(
            F(a='new a', b='new b').params,
            {'a': 'new a', 'b': 'new b'},
        )

    def test_parameterized_term_default_value_with_not_specified(self):
        defaults = {'a': 'default for a', 'b': NotSpecified}

        class F(Factor):
            params = defaults

            inputs = (SomeDataSet.foo,)
            dtype = 'f8'
            window_length = 5

        pattern = r"F expected a keyword parameter 'b'\."
        with assert_raises_regex(TypeError, pattern):
            F()
        with assert_raises_regex(TypeError, pattern):
            F(a='new a')

        assert_equal(F(b='new b').params, assoc(defaults, 'b', 'new b'))
        assert_equal(
            F(a='new a', b='new b').params,
            {'a': 'new a', 'b': 'new b'},
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

        with self.assertRaises(TermInputsNotSpecified):
            SomeFactor(window_length=1)

        with self.assertRaises(TermInputsNotSpecified):
            SomeFactorDefaultLength()

        with self.assertRaises(WindowLengthNotSpecified):
            SomeFactor(inputs=(SomeDataSet.foo,))

        with self.assertRaises(WindowLengthNotSpecified):
            SomeFactorDefaultInputs()

        with self.assertRaises(DTypeNotSpecified):
            SomeFactorNoDType()

        with self.assertRaises(NotDType):
            SomeFactor(dtype=1)

        with self.assertRaises(NoDefaultMissingValue):
            SomeFactor(dtype=int64_dtype)

        with self.assertRaises(UnsupportedDType):
            SomeFactor(dtype=complex128_dtype)

        with self.assertRaises(TermOutputsEmpty):
            MultipleOutputs(outputs=[])

    def test_bad_output_access(self):
        with self.assertRaises(AttributeError) as e:
            SomeFactor().not_an_attr

        errmsg = str(e.exception)
        self.assertEqual(
            errmsg, "'SomeFactor' object has no attribute 'not_an_attr'",
        )

        mo = MultipleOutputs()
        with self.assertRaises(AttributeError) as e:
            mo.not_an_attr

        errmsg = str(e.exception)
        expected = (
            "Instance of MultipleOutputs has no output named 'not_an_attr'."
            " Possible choices are: ('alpha', 'beta')."
        )
        self.assertEqual(errmsg, expected)

        with self.assertRaises(ValueError) as e:
            alpha, beta = GenericCustomFactor()

        errmsg = str(e.exception)
        self.assertEqual(
            errmsg, "GenericCustomFactor does not have multiple outputs.",
        )

        # Public method, user-defined method.
        # Accessing these attributes should return the output, not the method.
        conflicting_output_names = ['zscore', 'some_method']

        mo = MultipleOutputs(outputs=conflicting_output_names)
        for name in conflicting_output_names:
            self.assertIsInstance(getattr(mo, name), RecarrayField)

        # Non-callable attribute, private method, special method.
        disallowed_output_names = ['inputs', '_init', '__add__']

        for name in disallowed_output_names:
            with self.assertRaises(InvalidOutputName):
                GenericCustomFactor(outputs=[name])

    def test_require_super_call_in_validate(self):

        class MyFactor(Factor):
            inputs = ()
            dtype = float64_dtype
            window_length = 0

            def _validate(self):
                "Woops, I didn't call super()!"

        with self.assertRaises(AssertionError) as e:
            MyFactor()

        errmsg = str(e.exception)
        self.assertEqual(
            errmsg,
            "Term._validate() was not called.\n"
            "This probably means that you overrode _validate"
            " without calling super()."
        )

    def test_latest_on_different_dtypes(self):
        factor_dtypes = (float64_dtype, datetime64ns_dtype)
        for column in TestingDataSet.columns:
            if column.dtype == bool_dtype:
                self.assertIsInstance(column.latest, Filter)
            elif (column.dtype == int64_dtype
                  or column.dtype.kind in ('O', 'S', 'U')):
                self.assertIsInstance(column.latest, Classifier)
            elif column.dtype in factor_dtypes:
                self.assertIsInstance(column.latest, Factor)
            else:
                self.fail(
                    "Unknown dtype %s for column %s" % (column.dtype, column)
                )
            # These should be the same value, plus this has the convenient
            # property of correctly handling `NaN`.
            self.assertIs(column.missing_value, column.latest.missing_value)

    def test_failure_timing_on_bad_dtypes(self):

        # Just constructing a bad column shouldn't fail.
        Column(dtype=int64_dtype)
        with self.assertRaises(NoDefaultMissingValue) as e:
            class BadDataSet(DataSet):
                bad_column = Column(dtype=int64_dtype)
                float_column = Column(dtype=float64_dtype)
                int_column = Column(dtype=int64_dtype, missing_value=3)

        self.assertTrue(
            str(e.exception.args[0]).startswith(
                "Failed to create Column with name 'bad_column'"
            )
        )

        Column(dtype=complex128_dtype)
        with self.assertRaises(UnsupportedDType):
            class BadDataSetComplex(DataSet):
                bad_column = Column(dtype=complex128_dtype)
                float_column = Column(dtype=float64_dtype)
                int_column = Column(dtype=int64_dtype, missing_value=3)


class SubDataSetTestCase(TestCase):
    def test_subdataset(self):
        some_dataset_map = {
            column.name: column for column in SomeDataSet.columns
        }
        sub_dataset_map = {
            column.name: column for column in SubDataSet.columns
        }
        self.assertEqual(
            {column.name for column in SomeDataSet.columns},
            {column.name for column in SubDataSet.columns},
        )
        for k, some_dataset_column in some_dataset_map.items():
            sub_dataset_column = sub_dataset_map[k]
            self.assertIsNot(
                some_dataset_column,
                sub_dataset_column,
                'subclass column %r should not have the same identity as'
                ' the parent' % k,
            )
            self.assertEqual(
                some_dataset_column.dtype,
                sub_dataset_column.dtype,
                'subclass column %r should have the same dtype as the parent' %
                k,
            )

    def test_add_column(self):
        some_dataset_map = {
            column.name: column for column in SomeDataSet.columns
        }
        sub_dataset_new_col_map = {
            column.name: column for column in SubDataSetNewCol.columns
        }
        sub_col_names = {column.name for column in SubDataSetNewCol.columns}

        # check our extra col
        self.assertIn('qux', sub_col_names)
        self.assertEqual(
            sub_dataset_new_col_map['qux'].dtype,
            float64_dtype,
        )

        self.assertEqual(
            {column.name for column in SomeDataSet.columns},
            sub_col_names - {'qux'},
        )
        for k, some_dataset_column in some_dataset_map.items():
            sub_dataset_column = sub_dataset_new_col_map[k]
            self.assertIsNot(
                some_dataset_column,
                sub_dataset_column,
                'subclass column %r should not have the same identity as'
                ' the parent' % k,
            )
            self.assertEqual(
                some_dataset_column.dtype,
                sub_dataset_column.dtype,
                'subclass column %r should have the same dtype as the parent' %
                k,
            )

    @parameter_space(
        dtype_=[categorical_dtype, int64_dtype],
        outputs_=[('a',), ('a', 'b')],
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
            missing_value = dtype_.type('123')

        expected_error = (
            "SomeClassifier does not support custom outputs, "
            "but received custom outputs={outputs}.".format(outputs=outputs_)
        )

        with self.assertRaises(ValueError) as e:
            SomeClassifier()
        self.assertEqual(str(e.exception), expected_error)

        with self.assertRaises(ValueError) as e:
            SomeClassifier()
        self.assertEqual(str(e.exception), expected_error)

    def test_unreasonable_missing_values(self):

        for base_type, dtype_, bad_mv in ((Factor, float64_dtype, 'ayy'),
                                          (Filter, bool_dtype, 'lmao'),
                                          (Classifier, int64_dtype, 'lolwut'),
                                          (Classifier, categorical_dtype, 7)):
            class SomeTerm(base_type):
                inputs = ()
                window_length = 0
                missing_value = bad_mv
                dtype = dtype_

            with self.assertRaises(TypeError) as e:
                SomeTerm()

            prefix = (
                "^Missing value {mv!r} is not a valid choice "
                "for term SomeTerm with dtype {dtype}.\n\n"
                "Coercion attempt failed with:"
            ).format(mv=bad_mv, dtype=dtype_)

            self.assertRegexpMatches(str(e.exception), prefix)
