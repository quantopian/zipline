from itertools import product
from operator import eq, ne
import numpy as np

from zipline.lib.labelarray import LabelArray
from zipline.testing import check_arrays, parameter_space, ZiplineTestCase


def rotN(l, N):
    """
    Rotate a list of elements.

    Pulls N elements off the end of the list and appends them to the front.

    >>> rotN(['a', 'b', 'c', 'd'], 2)
    ['c', 'd', 'a', 'b']
    >>> rotN(['a', 'b', 'c', 'd'], 3)
    ['d', 'a', 'b', 'c']
    """
    assert len(l) >= N, "Can't rotate list by longer than its length."
    return l[N:] + l[:N]


class LabelArrayTestCase(ZiplineTestCase):

    @classmethod
    def init_class_fixtures(cls):
        super(LabelArrayTestCase, cls).init_class_fixtures()

        cls.rowvalues = row = ['', 'a', 'b', 'ab', 'a', '', 'b', 'ab', 'z']
        cls.strs = np.array([rotN(row, i) for i in range(3)])

    def test_fail_on_direct_construction(self):
        # See http://docs.scipy.org/doc/numpy-1.10.0/user/basics.subclassing.html#simple-example-adding-an-extra-attribute-to-ndarray  # noqa

        with self.assertRaises(TypeError) as e:
            np.ndarray.__new__(LabelArray, (5, 5))

        self.assertEqual(
            str(e.exception),
            "Direct construction of LabelArrays is not supported."
        )

    @parameter_space(
        __fail_fast=True,
        s=['', 'a', 'z', 'aa', 'not in the array'],
        shape=[(27,), (9, 3), (3, 9), (3, 3, 3)],
        array_astype=(bytes, unicode, object),
        scalar_astype=(bytes, unicode, object),
    )
    def test_compare_to_str(self, s, shape, array_astype, scalar_astype):
        strs = self.strs.reshape(shape).astype(array_astype)
        arr = LabelArray(strs, missing_value='')
        check_arrays(strs == s, arr == s)
        check_arrays(strs != s, arr != s)

        np_startswith = np.vectorize(lambda elem: elem.startswith(s))
        check_arrays(arr.startswith(s), np_startswith(strs))

        np_endswith = np.vectorize(lambda elem: elem.endswith(s))
        check_arrays(arr.endswith(s), np_endswith(strs))

        np_contains = np.vectorize(lambda elem: s in elem)
        check_arrays(arr.contains(s), np_contains(strs))

    def test_compare_to_str_array(self):
        strs = self.strs
        shape = strs.shape
        arr = LabelArray(strs, missing_value='')
        check_arrays(strs == arr, np.full_like(strs, True, dtype=bool))
        check_arrays(strs != arr, np.full_like(strs, False, dtype=bool))

        def broadcastable_row(value, dtype):
            return np.full((shape[0], 1), value, dtype=strs.dtype)

        def broadcastable_col(value, dtype):
            return np.full((1, shape[1]), value, dtype=strs.dtype)

        for comparator, dtype, value in product((eq, ne),
                                                (bytes, unicode, object),
                                                set(self.rowvalues)):
            check_arrays(
                comparator(arr, np.full_like(strs, value)),
                comparator(strs, value),
            )
            check_arrays(
                comparator(arr, broadcastable_row(value, dtype=dtype)),
                comparator(strs, value),
            )
            check_arrays(
                comparator(arr, broadcastable_col(value, dtype=dtype)),
                comparator(strs, value),
            )

    @parameter_space(
        __fail_fast=True,
        slice_=[
            0, 1, -1,
            slice(None),
            slice(0, 0),
            slice(0, 3),
            slice(1, 4),
            slice(0),
            slice(None, 1),
            slice(0, 4, 2),
            (slice(None), 1),
            (slice(None), slice(None)),
            (slice(None), slice(1, 2)),
        ]
    )
    def test_slicing_preserves_attributes(self, slice_):
        arr = LabelArray(self.strs.reshape((9, 3)), missing_value='')
        sliced = arr[slice_]
        self.assertIsInstance(sliced, LabelArray)
        self.assertIs(sliced.categories, arr.categories)
        self.assertIs(sliced.reverse_categories, arr.reverse_categories)
        self.assertIs(sliced.missing_value, arr.missing_value)

    def test_infer_categories(self):
        arr1d = LabelArray(self.strs, missing_value='')
        codes1d = arr1d.as_int_array()
        self.assertEqual(arr1d.shape, self.strs.shape)
        self.assertEqual(arr1d.shape, codes1d.shape)

        categories = arr1d.categories
        unique_rowvalues = set(self.rowvalues)

        # There should be an entry in categories for each unique row value, and
        # each integer stored in the data array should be an index into
        # categories.
        self.assertEqual(list(categories), sorted(set(self.rowvalues)))
        self.assertEqual(
            set(codes1d.ravel()),
            set(range(len(unique_rowvalues)))
        )
        for idx, value in enumerate(arr1d.categories):
            check_arrays(
                self.strs == value,
                arr1d.view(type=np.ndarray) == idx,
            )

        for shape in (9, 3), (3, 9), (3, 3, 3):
            strs2d = self.strs.reshape(shape)
            arr2d = LabelArray(strs2d, missing_value='')
            codes2d = arr2d.as_int_array()

            self.assertEqual(arr2d.shape, shape)
            check_arrays(arr2d.categories, categories)

            for idx, value in enumerate(arr2d.categories):
                check_arrays(strs2d == value, codes2d == idx)
