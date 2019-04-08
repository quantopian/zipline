"""Tests for the zipline.pipeline.data.DataSet and related functionality.
"""
from textwrap import dedent

from zipline.pipeline.data.dataset import Column, DataSet
from zipline.testing import chrange, ZiplineTestCase
from zipline.testing.predicates import assert_messages_equal


class SomeDataSet(DataSet):
    a = Column(dtype=float)
    b = Column(dtype=object)
    c = Column(dtype=int, missing_value=-1)


# A DataSet with lots of columns.
class LargeDataSet(DataSet):
    locals().update({
        name: Column(dtype=float)
        for name in chrange('a', 'z')
    })


class GetColumnTestCase(ZiplineTestCase):

    def test_get_column_success(self):
        a = SomeDataSet.a
        b = SomeDataSet.b
        c = SomeDataSet.c

        # Run multiple times to validate caching of descriptor return values.
        for _ in range(3):
            self.assertIs(SomeDataSet.get_column('a'), a)
            self.assertIs(SomeDataSet.get_column('b'), b)
            self.assertIs(SomeDataSet.get_column('c'), c)

    def test_get_column_failure(self):
        with self.assertRaises(AttributeError) as e:
            SomeDataSet.get_column('arglebargle')

        result = str(e.exception)
        expected = dedent(
            """\
            SomeDataSet has no column 'arglebargle':

            Possible choices are:
              - a
              - b
              - c"""
        )
        assert_messages_equal(result, expected)

    def test_get_column_failure_truncate_error_message(self):
        with self.assertRaises(AttributeError) as e:
            LargeDataSet.get_column('arglebargle')

        result = str(e.exception)
        expected = dedent(
            """\
            LargeDataSet has no column 'arglebargle':

            Possible choices are:
              - a
              - b
              - c
              - d
              - e
              - f
              - g
              - h
              - i
              - ...
              - z"""
        )
        assert_messages_equal(result, expected)
