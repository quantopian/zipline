"""
Tests for Term.
"""
from itertools import product
from unittest import TestCase

from networkx import topological_sort
from numpy import (
    float32,
    uint32,
    uint8,
)

from zipline.data.dataset import (
    Column,
    DataSet,
)
from zipline.errors import (
    InputTermNotAtomic,
    TermInputsNotSpecified,
    WindowLengthNotSpecified,
)
from zipline.modelling.engine import build_dependency_graph
from zipline.modelling.factor import Factor
from zipline.modelling.expression import NUMEXPR_MATH_FUNCS


class SomeDataSet(DataSet):

    foo = Column(float32)
    bar = Column(uint32)
    buzz = Column(uint8)


class SomeFactor(Factor):
    window_length = 5
    inputs = [SomeDataSet.foo, SomeDataSet.bar]


class NoLookbackFactor(Factor):
    window_length = 0


class SomeOtherFactor(Factor):
    window_length = 5
    inputs = [SomeDataSet.bar, SomeDataSet.buzz]


SomeFactorAlias = SomeFactor


def gen_equivalent_factors():
    """
    Return an iterator of SomeFactor instances that should all be the same
    object.
    """
    yield SomeFactor()
    yield SomeFactor(inputs=None)
    yield SomeFactor(SomeFactor.inputs)
    yield SomeFactor(inputs=SomeFactor.inputs)
    yield SomeFactor([SomeDataSet.foo, SomeDataSet.bar])
    yield SomeFactor(window_length=SomeFactor.window_length)
    yield SomeFactor(window_length=None)
    yield SomeFactor([SomeDataSet.foo, SomeDataSet.bar], window_length=None)
    yield SomeFactor(
        [SomeDataSet.foo, SomeDataSet.bar],
        window_length=SomeFactor.window_length,
    )
    yield SomeFactorAlias()


class DependencyResolutionTestCase(TestCase):

    def setup(self):
        pass

    def teardown(self):
        pass

    def test_single_factor(self):
        """
        Test dependency resolution for a single factor.
        """

        build_dependency_graph([SomeFactor()])

        def check_output(graph):

            resolution_order = topological_sort(graph)

            self.assertEqual(len(resolution_order), 3)
            self.assertEqual(
                set([resolution_order[0], resolution_order[1]]),
                set([SomeDataSet.foo, SomeDataSet.bar]),
            )
            self.assertEqual(resolution_order[-1], SomeFactor())
            self.assertEqual(graph.node[SomeDataSet.foo]['extra_rows'], 4)
            self.assertEqual(graph.node[SomeDataSet.bar]['extra_rows'], 4)

        for foobar in gen_equivalent_factors():
            check_output(build_dependency_graph([foobar]))

    def test_single_factor_instance_args(self):
        """
        Test dependency resolution for a single factor with arguments passed to
        the constructor.
        """
        graph = build_dependency_graph(
            [SomeFactor([SomeDataSet.bar, SomeDataSet.buzz], window_length=5)]
        )
        resolution_order = topological_sort(graph)

        self.assertEqual(len(resolution_order), 3)
        self.assertEqual(
            set([resolution_order[0], resolution_order[1]]),
            set([SomeDataSet.bar, SomeDataSet.buzz]),
        )
        self.assertEqual(
            resolution_order[-1],
            SomeFactor([SomeDataSet.bar, SomeDataSet.buzz], window_length=5),
        )
        self.assertEqual(graph.node[SomeDataSet.bar]['extra_rows'], 4)
        self.assertEqual(graph.node[SomeDataSet.buzz]['extra_rows'], 4)

    def test_reuse_atomic_terms(self):
        """
        Test that raw inputs only show up in the dependency graph once.
        """
        f1 = SomeFactor([SomeDataSet.foo, SomeDataSet.bar])
        f2 = SomeOtherFactor([SomeDataSet.bar, SomeDataSet.buzz])

        graph = build_dependency_graph([f1, f2])
        resolution_order = topological_sort(graph)

        # bar should only appear once.
        self.assertEqual(len(resolution_order), 5)
        indices = {
            term: resolution_order.index(term)
            for term in resolution_order
        }

        # Verify that f1's dependencies will be computed before f1.
        self.assertLess(indices[SomeDataSet.foo], indices[f1])
        self.assertLess(indices[SomeDataSet.bar], indices[f1])

        # Verify that f2's dependencies will be computed before f2.
        self.assertLess(indices[SomeDataSet.bar], indices[f2])
        self.assertLess(indices[SomeDataSet.buzz], indices[f2])

    def test_disallow_recursive_lookback(self):

        with self.assertRaises(InputTermNotAtomic):
            SomeFactor(inputs=[SomeFactor(), SomeDataSet.foo])


class ObjectIdentityTestCase(TestCase):

    def assertSameObject(self, *objs):
        first = objs[0]
        for obj in objs:
            self.assertIs(first, obj)

    def test_instance_caching(self):

        self.assertSameObject(*gen_equivalent_factors())
        self.assertIs(
            SomeFactor(window_length=SomeFactor.window_length + 1),
            SomeFactor(window_length=SomeFactor.window_length + 1),
        )

        self.assertIs(
            SomeFactor(dtype=int),
            SomeFactor(dtype=int),
        )

        self.assertIs(
            SomeFactor(inputs=[SomeFactor.inputs[1], SomeFactor.inputs[0]]),
            SomeFactor(inputs=[SomeFactor.inputs[1], SomeFactor.inputs[0]]),
        )

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
            SomeFactor(dtype=int)
        )

        # Reordering inputs changes semantics.
        self.assertIsNot(
            f,
            SomeFactor(inputs=[SomeFactor.inputs[1], SomeFactor.inputs[0]]),
        )

    def test_instance_non_caching_redefine_class(self):

        orig_foobar_instance = SomeFactorAlias()

        class SomeFactor(Factor):
            window_length = 5
            inputs = [SomeDataSet.foo, SomeDataSet.bar]

        self.assertIsNot(orig_foobar_instance, SomeFactor())

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

    def test_bad_input(self):

        class SomeFactor(Factor):
            pass

        class SomeFactorDefaultInputs(Factor):
            inputs = (SomeDataSet.foo, SomeDataSet.bar)

        class SomeFactorDefaultLength(Factor):
            window_length = 10

        with self.assertRaises(TermInputsNotSpecified):
            SomeFactor(window_length=1)

        with self.assertRaises(TermInputsNotSpecified):
            SomeFactorDefaultLength()

        with self.assertRaises(WindowLengthNotSpecified):
            SomeFactor(inputs=(SomeDataSet.foo,))

        with self.assertRaises(WindowLengthNotSpecified):
            SomeFactorDefaultInputs()
