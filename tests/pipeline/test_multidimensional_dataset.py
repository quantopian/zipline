from collections import OrderedDict
import itertools
from textwrap import dedent

from nose_parameterized import parameterized
import numpy as np

from zipline.pipeline.data import (
    Column,
    DataSetFamily,
    DataSetFamilySlice,
)
from zipline.testing import ZiplineTestCase
from zipline.testing.predicates import (
    assert_equal,
    assert_is,
    assert_is_not,
    assert_is_subclass,
    assert_raises_str,
)


class TestDataSetFamily(ZiplineTestCase):
    def test_repr(self):
        class MD1(DataSetFamily):
            extra_dims = [('dim_0', [])]

        expected_repr = (
            "<DataSetFamily: 'MD1', extra_dims=['dim_0']>"
        )
        assert_equal(repr(MD1), expected_repr)

        class MD2(DataSetFamily):
            extra_dims = [('dim_0', []), ('dim_1', [])]

        expected_repr = (
            "<DataSetFamily: 'MD2', extra_dims=['dim_0', 'dim_1']>"
        )
        assert_equal(repr(MD2), expected_repr)

        class MD3(DataSetFamily):
            extra_dims = [('dim_1', []), ('dim_0', [])]

        expected_repr = (
            "<DataSetFamily: 'MD3', extra_dims=['dim_1', 'dim_0']>"
        )
        assert_equal(repr(MD3), expected_repr)

    def test_cache(self):
        class MD1(DataSetFamily):
            extra_dims = [('dim_0', ['a', 'b', 'c'])]

        class MD2(DataSetFamily):
            extra_dims = [('dim_0', ['a', 'b', 'c'])]

        MD1Slice = MD1.slice(dim_0='a')
        MD2Slice = MD2.slice(dim_0='a')

        assert_equal(MD1Slice.extra_coords, MD2Slice.extra_coords)
        assert_is_not(MD1Slice, MD2Slice)

    def test_empty_extra_dims(self):
        msg = (
            "DataSetFamily must be defined with non-empty extra_dims,"
            " or with `_abstract = True`"
        )
        with assert_raises_str(ValueError, msg):
            class NoExtraDims(DataSetFamily):
                pass

        with assert_raises_str(ValueError, msg):
            class EmptyExtraDims(DataSetFamily):
                extra_dims = []

        class AbstractParent(DataSetFamily):
            _abstract = True

        with assert_raises_str(ValueError, msg):
            class NoExtraDimsChild(AbstractParent):
                pass

        with assert_raises_str(ValueError, msg):
            class EmptyExtraDimsChild(AbstractParent):
                extra_dims = []

        class AbstractChild(AbstractParent):
            _abstract = True

        class Child(AbstractParent):
            extra_dims = [
                ('dim_0', {'a', 'b', 'c'}),
                ('dim_1', {'d', 'e', 'f'}),
            ]

    def spec(*cs):
        return (cs,)

    @parameterized.expand([
        spec(
            ('dim_0', range(10))
        ),
        spec(
            ('dim_0', range(10)),
            ('dim_1', range(10, 15)),
        ),
        spec(
            ('dim_0', range(10)),
            ('dim_1', range(10, 15)),
            ('dim_2', range(5, 15)),
        ),
        spec(
            ('dim_0', range(6)),
            ('dim_1', {'a', 'b', 'c'}),
            ('dim_2', range(5, 15)),
            ('dim_3', {'b', 'c', 'e'}),
        ),
    ])
    def test_valid_slice(self, dims_spec):
        class MD(DataSetFamily):
            extra_dims = dims_spec

            f8 = Column('f8')
            i8 = Column('i8', missing_value=0)
            ob = Column('O')
            M8 = Column('M8[ns]')
            boolean = Column('?')

        expected_dims = OrderedDict([(k, frozenset(v)) for k, v in dims_spec])
        assert_equal(MD.extra_dims, expected_dims)

        for valid_combination in itertools.product(*expected_dims.values()):
            Slice = MD.slice(*valid_combination)
            alternate_constructions = [
                # all positional
                MD.slice(*valid_combination),
                # all keyword
                MD.slice(**dict(zip(expected_dims.keys(), valid_combination))),
                # mix keyword/positional
                MD.slice(
                    *valid_combination[:len(valid_combination) // 2],
                    **dict(
                        list(zip(expected_dims.keys(), valid_combination))[
                            len(valid_combination) // 2:
                        ],
                    )
                ),
            ]
            for alt in alternate_constructions:
                assert_is(Slice, alt, msg='Slices are not properly memoized')

            expected_coords = OrderedDict(
                zip(expected_dims, valid_combination),
            )
            assert_equal(Slice.extra_coords, expected_coords)

            assert_is(Slice.dataset_family, MD)

            assert_is_subclass(Slice, DataSetFamilySlice)

            expected_columns = {
                ('f8', np.dtype('f8'), Slice),
                ('i8', np.dtype('i8'), Slice),
                ('ob', np.dtype('O'), Slice),
                ('M8', np.dtype('M8[ns]'), Slice),
                ('boolean', np.dtype('?'), Slice),
            }
            actual_columns = {
                (c.name, c.dtype, c.dataset) for c in Slice.columns
            }
            assert_equal(actual_columns, expected_columns)

    del spec

    def test_slice_unknown_dims(self):
        class MD(DataSetFamily):
            extra_dims = [
                ('dim_0', {'a', 'b', 'c'}),
                ('dim_1', {'c', 'd', 'e'}),
            ]

        def expect_slice_fails(*args, **kwargs):
            expected_msg = kwargs.pop('expected_msg')

            with assert_raises_str(TypeError, expected_msg):
                MD.slice(*args, **kwargs)

        # insufficient positional
        expect_slice_fails(
            expected_msg=(
                'no coordinate provided to MD for the following dimensions:'
                ' dim_0, dim_1'
            ),
        )
        expect_slice_fails(
            'a',
            expected_msg=(
                'no coordinate provided to MD for the following dimension:'
                ' dim_1'
            ),
        )

        # too many positional
        expect_slice_fails(
            'a', 'b', 'c',
            expected_msg='MD has 2 extra dimensions but 3 were given',
        )

        # mismatched keys
        expect_slice_fails(
            dim_2='??',
            expected_msg=(
                'MD does not have the following dimension: dim_2\n'
                'Valid dimensions are: dim_0, dim_1'
            ),
        )
        expect_slice_fails(
            dim_1='??', dim_2='??',
            expected_msg=(
                'MD does not have the following dimension: dim_2\n'
                'Valid dimensions are: dim_0, dim_1'
            ),
        )
        expect_slice_fails(
            dim_0='??', dim_1='??', dim_2='??',
            expected_msg=(
                'MD does not have the following dimension: dim_2\n'
                'Valid dimensions are: dim_0, dim_1'
            ),
        )

        # the extra keyword dims should be sorted
        expect_slice_fails(
            dim_3='??', dim_2='??',
            expected_msg=(
                'MD does not have the following dimensions: dim_2, dim_3\n'
                'Valid dimensions are: dim_0, dim_1'
            ),
        )

    def test_slice_unknown_dim_label(self):
        class MD(DataSetFamily):
            extra_dims = [
                ('dim_0', {'a', 'b', 'c'}),
                ('dim_1', {'c', 'd', 'e'}),
            ]

        def expect_slice_fails(*args, **kwargs):
            expected_msg = kwargs.pop('expected_msg')

            with assert_raises_str(ValueError, expected_msg):
                MD.slice(*args, **kwargs)

        expect_slice_fails(
            'not-in-0', 'c',
            expected_msg=(
                "'not-in-0' is not a value along the dim_0 dimension of MD"
            ),
        )
        expect_slice_fails(
            dim_0='not-in-0', dim_1='c',
            expected_msg=(
                "'not-in-0' is not a value along the dim_0 dimension of MD"
            ),
        )

        expect_slice_fails(
            'a', 'not-in-1',
            expected_msg=(
                "'not-in-1' is not a value along the dim_1 dimension of MD"
            ),
        )
        expect_slice_fails(
            dim_0='a', dim_1='not-in-1',
            expected_msg=(
                "'not-in-1' is not a value along the dim_1 dimension of MD"
            ),
        )

    def test_inheritance(self):
        class Parent(DataSetFamily):
            extra_dims = [
                ('dim_0', {'a', 'b', 'c'}),
                ('dim_1', {'d', 'e', 'f'}),
            ]

            column_0 = Column('f8')
            column_1 = Column('?')

        class Child(Parent):
            column_2 = Column('O')
            column_3 = Column('i8', -1)

        assert_is_subclass(Child, Parent)
        assert_equal(Child.extra_dims, Parent.extra_dims)

        ChildSlice = Child.slice(dim_0='a', dim_1='d')

        expected_child_slice_columns = frozenset({
            ChildSlice.column_0,
            ChildSlice.column_1,
            ChildSlice.column_2,
            ChildSlice.column_3,
        })
        assert_equal(ChildSlice.columns, expected_child_slice_columns)

    def test_column_access_without_slice(self):
        class Parent(DataSetFamily):
            extra_dims = [
                ('dim_0', {'a', 'b', 'c'}),
                ('dim_1', {'d', 'e', 'f'}),
            ]

            column_0 = Column('f8')
            column_1 = Column('?')

        class Child(Parent):
            column_2 = Column('O')
            column_3 = Column('i8', -1)

        def make_expected_msg(ds, attr):
            return dedent(
                """\
                Attempted to access column {c} from DataSetFamily {d}:

                To work with dataset families, you must first select a
                slice using the ``slice`` method:

                    {d}.slice(...).{c}
                """
                .format(c=attr, d=ds),  # noqa
            )

        expected_msg = make_expected_msg('Parent', 'column_0')
        with assert_raises_str(AttributeError, expected_msg):
            Parent.column_0

        expected_msg = make_expected_msg('Parent', 'column_1')
        with assert_raises_str(AttributeError, expected_msg):
            Parent.column_1

        expected_msg = make_expected_msg('Child', 'column_0')
        with assert_raises_str(AttributeError, expected_msg):
            Child.column_0

        expected_msg = make_expected_msg('Child', 'column_1')
        with assert_raises_str(AttributeError, expected_msg):
            Child.column_1

        expected_msg = make_expected_msg('Child', 'column_2')
        with assert_raises_str(AttributeError, expected_msg):
            Child.column_2

        expected_msg = make_expected_msg('Child', 'column_3')
        with assert_raises_str(AttributeError, expected_msg):
            Child.column_3
