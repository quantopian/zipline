"""
Tests for the FFC API.
"""
# TODO: Rename this shit.
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
# from zipline.modelling.classifier import Classifier
from zipline.modelling.engine import build_dependency_graph
from zipline.modelling.factor import Factor
# from zipline.modelling.filter import Filter


class SomeDataSet(DataSet):

    foo = Column(float32)
    bar = Column(uint32)
    buzz = Column(uint8)


class SomeFactor(Factor):
    lookback = 5
    inputs = [SomeDataSet.foo, SomeDataSet.bar]


class SomeOtherFactor(Factor):
    lookback = 5
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
    yield SomeFactor(lookback=SomeFactor.lookback)
    yield SomeFactor(lookback=None)
    yield SomeFactor([SomeDataSet.foo, SomeDataSet.bar], lookback=None)
    yield SomeFactor(
        [SomeDataSet.foo, SomeDataSet.bar],
        lookback=SomeFactor.lookback,
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

        build_dependency_graph([], [], [SomeFactor()])

        def check_output(graph):

            resolution_order = topological_sort(graph)

            self.assertEqual(len(resolution_order), 3)
            self.assertEqual(
                set([resolution_order[0], resolution_order[1]]),
                set([SomeDataSet.foo, SomeDataSet.bar]),
            )
            self.assertEqual(resolution_order[-1], SomeFactor())
            self.assertEqual(graph.node[SomeDataSet.foo]['lookback'], 5)
            self.assertEqual(graph.node[SomeDataSet.bar]['lookback'], 5)

        for foobar in gen_equivalent_factors():
            check_output(build_dependency_graph([], [], [foobar]))

    def test_single_factor_instance_args(self):
        """
        Test dependency resolution for a single factor with arguments passed to
        the constructor.
        """
        graph = build_dependency_graph(
            [], [],
            [SomeFactor([SomeDataSet.bar, SomeDataSet.buzz], lookback=5)]
        )
        resolution_order = topological_sort(graph)

        self.assertEqual(len(resolution_order), 3)
        self.assertEqual(
            set([resolution_order[0], resolution_order[1]]),
            set([SomeDataSet.bar, SomeDataSet.buzz]),
        )
        self.assertEqual(
            resolution_order[-1],
            SomeFactor([SomeDataSet.bar, SomeDataSet.buzz], lookback=5),
        )
        self.assertEqual(graph.node[SomeDataSet.bar]['lookback'], 5)
        self.assertEqual(graph.node[SomeDataSet.buzz]['lookback'], 5)

    def test_reuse_atomic_terms(self):
        """
        Test that raw inputs only show up in the dependency graph once.
        """
        f1 = SomeFactor([SomeDataSet.foo, SomeDataSet.bar])
        f2 = SomeOtherFactor([SomeDataSet.bar, SomeDataSet.buzz])

        graph = build_dependency_graph([], [], [f1, f2])
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

    def test_factor_with_self_as_argument(self):
        """
        Test that an instance of a factor can be passed as an input to another
        factor.
        """
        f1 = SomeFactor()
        f2 = SomeFactor([f1, SomeDataSet.foo])

        graph = build_dependency_graph([], [], [f1, f2])
        resolution_order = topological_sort(graph)

        self.assertEqual(
            set(resolution_order[:2]),
            set([SomeDataSet.foo, SomeDataSet.bar]),
        )
        self.assertEqual(resolution_order[2:], [f1, f2])


class ObjectIdentityTestCase(TestCase):

    def assertSameObject(self, *objs):
        first = objs[0]
        for obj in objs:
            self.assertIs(first, obj)

    def test_instance_caching(self):

        self.assertSameObject(*gen_equivalent_factors())
        self.assertIs(
            SomeFactor(lookback=SomeFactor.lookback + 1),
            SomeFactor(lookback=SomeFactor.lookback + 1),
        )

    def test_instance_non_caching(self):

        self.assertIsNot(
            SomeFactor(),
            SomeFactor(lookback=SomeFactor.lookback + 1),
        )
        self.assertIsNot(
            SomeFactor(lookback=SomeFactor.lookback),
            SomeFactor(lookback=SomeFactor.lookback + 1),
        )

        # Reordering inputs changes semantics.
        self.assertIsNot(
            SomeFactor(inputs=[SomeFactor.inputs[0], SomeFactor.inputs[1]]),
            SomeFactor(inputs=[SomeFactor.inputs[1], SomeFactor.inputs[0]]),
        )

    def test_instance_non_caching_redefine_class(self):

        orig_foobar_instance = SomeFactorAlias()

        class SomeFactor(Factor):
            lookback = 5
            inputs = [SomeDataSet.foo, SomeDataSet.bar]

        self.assertIsNot(orig_foobar_instance, SomeFactor())
