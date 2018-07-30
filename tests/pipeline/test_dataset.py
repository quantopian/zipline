"""Tests for zipline.pipeline.data.DataSet and friends.
"""
from zipline.pipeline.domain import USEquities, CanadaEquities
from zipline.pipeline.data.dataset import Column, DataSet
import zipline.testing.fixtures as zf
from zipline.testing.predicates import assert_equal


class DataSetTestCase(zf.ZiplineTestCase):

    def test_specialize(self):
        class MyDataSet(DataSet):
            col1 = Column(dtype=float)
            col2 = Column(dtype=int, missing_value=100)
            col3 = Column(dtype=object, missing_value="")

        specialized = MyDataSet.specialize(USEquities)

        # Specializations should be memoized.
        self.assertIs(specialized, MyDataSet.specialize(USEquities))

        # Specializations should have the same name, but prefixed with the
        # country code of the new domain.
        assert_equal(specialized.__name__, "MyDataSet_US")
        self.assertIs(specialized.domain, USEquities)

        for attr in ('col1', 'col2', 'col3'):
            original = getattr(MyDataSet, attr)
            new = getattr(specialized, attr)

            # We should get a new column from the specialization, which should
            # be the same object that we would get from specializing the
            # original column.
            self.assertIsNot(original, new)
            self.assertIs(new, original.specialize(USEquities))

            # Columns should be bound to their respective datasets.
            self.assertIs(original.dataset, MyDataSet)
            self.assertIs(new.dataset, specialized)

            # The new column should have the domain of the specialization.
            assert_equal(new.domain, USEquities)

            # Names, dtypes, and missing_values should match.
            assert_equal(original.name, new.name)
            assert_equal(original.dtype, new.dtype)
            assert_equal(original.missing_value, new.missing_value)

    def test_repr(self):
        class Data(DataSet):
            col1 = Column(dtype=float)
            col2 = Column(dtype=int, missing_value=100)
            col3 = Column(dtype=object, missing_value="")

        expected = "<DataSet: 'Data'>"
        self.assertEqual(repr(Data), expected)

        specialized = Data.specialize(USEquities)
        expected_specialized = "<DataSet: 'Data_US', domain={}>".format(
            str(USEquities),
        )
        self.assertEqual(repr(specialized), expected_specialized)

        specialized_CA = Data.specialize(CanadaEquities)
        expected_specialized_CA = "<DataSet: 'Data_CA', domain={}>".format(
            str(CanadaEquities),
        )
        self.assertEqual(repr(specialized_CA), expected_specialized_CA)
