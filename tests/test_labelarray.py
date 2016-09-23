from itertools import product
from operator import eq, ne
import numpy as np
import warnings

from zipline.lib.labelarray import LabelArray
from zipline.testing import check_arrays, parameter_space, ZiplineTestCase
from zipline.utils.compat import unicode


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


def all_ufuncs():
    ufunc_type = type(np.isnan)
    return (f for f in vars(np).values() if isinstance(f, ufunc_type))


class LabelArrayTestCase(ZiplineTestCase):

    @classmethod
    def init_class_fixtures(cls):
        super(LabelArrayTestCase, cls).init_class_fixtures()

        cls.rowvalues = row = ['', 'a', 'b', 'ab', 'a', '', 'b', 'ab', 'z']
        cls.strs = np.array([rotN(row, i) for i in range(3)], dtype=object)

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
        compval=['', 'a', 'z', 'not in the array'],
        shape=[(27,), (3, 9), (3, 3, 3)],
        array_astype=(bytes, unicode, object),
        missing_value=('', 'a', 'not in the array', None),
    )
    def test_compare_to_str(self,
                            compval,
                            shape,
                            array_astype,
                            missing_value):

        strs = self.strs.reshape(shape).astype(array_astype)
        if missing_value is None:
            # As of numpy 1.9.2, object array != None returns just False
            # instead of an array, with a deprecation warning saying the
            # behavior will change in the future.  Work around that by just
            # using the ufunc.
            notmissing = np.not_equal(strs, missing_value)
        else:
            if not isinstance(missing_value, array_astype):
                missing_value = array_astype(missing_value, 'utf-8')
            notmissing = (strs != missing_value)

        arr = LabelArray(strs, missing_value=missing_value)

        if not isinstance(compval, array_astype):
            compval = array_astype(compval, 'utf-8')

        # arr.missing_value should behave like NaN.
        check_arrays(
            arr == compval,
            (strs == compval) & notmissing,
        )
        check_arrays(
            arr != compval,
            (strs != compval) & notmissing,
        )

        np_startswith = np.vectorize(lambda elem: elem.startswith(compval))
        check_arrays(
            arr.startswith(compval),
            np_startswith(strs) & notmissing,
        )

        np_endswith = np.vectorize(lambda elem: elem.endswith(compval))
        check_arrays(
            arr.endswith(compval),
            np_endswith(strs) & notmissing,
        )

        np_contains = np.vectorize(lambda elem: compval in elem)
        check_arrays(
            arr.has_substring(compval),
            np_contains(strs) & notmissing,
        )

    @parameter_space(
        __fail_fast=True,
        missing_value=('', 'a', 'not in the array', None),
    )
    def test_compare_to_str_array(self, missing_value):
        strs = self.strs
        shape = strs.shape
        arr = LabelArray(strs, missing_value=missing_value)

        if missing_value is None:
            # As of numpy 1.9.2, object array != None returns just False
            # instead of an array, with a deprecation warning saying the
            # behavior will change in the future.  Work around that by just
            # using the ufunc.
            notmissing = np.not_equal(strs, missing_value)
        else:
            notmissing = (strs != missing_value)

        check_arrays(arr.not_missing(), notmissing)
        check_arrays(arr.is_missing(), ~notmissing)

        # The arrays are equal everywhere, but comparisons against the
        # missing_value should always produce False
        check_arrays(strs == arr, notmissing)
        check_arrays(strs != arr, np.zeros_like(strs, dtype=bool))

        def broadcastable_row(value, dtype):
            return np.full((shape[0], 1), value, dtype=strs.dtype)

        def broadcastable_col(value, dtype):
            return np.full((1, shape[1]), value, dtype=strs.dtype)

        # Test comparison between arr and a like-shap 2D array, a column
        # vector, and a row vector.
        for comparator, dtype, value in product((eq, ne),
                                                (bytes, unicode, object),
                                                set(self.rowvalues)):
            check_arrays(
                comparator(arr, np.full_like(strs, value)),
                comparator(strs, value) & notmissing,
            )
            check_arrays(
                comparator(arr, broadcastable_row(value, dtype=dtype)),
                comparator(strs, value) & notmissing,
            )
            check_arrays(
                comparator(arr, broadcastable_col(value, dtype=dtype)),
                comparator(strs, value) & notmissing,
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
        """
        Test that categories are inferred in sorted order if they're not
        explicitly passed.
        """
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
                arr1d.as_int_array() == idx,
            )

        # It should be equivalent to pass the same set of categories manually.
        arr1d_explicit_categories = LabelArray(
            self.strs,
            missing_value='',
            categories=arr1d.categories,
        )
        check_arrays(arr1d, arr1d_explicit_categories)

        for shape in (9, 3), (3, 9), (3, 3, 3):
            strs2d = self.strs.reshape(shape)
            arr2d = LabelArray(strs2d, missing_value='')
            codes2d = arr2d.as_int_array()

            self.assertEqual(arr2d.shape, shape)
            check_arrays(arr2d.categories, categories)

            for idx, value in enumerate(arr2d.categories):
                check_arrays(strs2d == value, codes2d == idx)

    def test_reject_ufuncs(self):
        """
        The internal values of a LabelArray should be opaque to numpy ufuncs.

        Test that all unfuncs fail.
        """
        l = LabelArray(self.strs, '')
        ints = np.arange(len(l))

        with warnings.catch_warnings():
            # Some ufuncs return NotImplemented, but warn that they will fail
            # in the future.  Both outcomes are fine, so ignore the warnings.
            warnings.filterwarnings(
                'ignore',
                message="unorderable dtypes.*",
                category=DeprecationWarning,
            )
            warnings.filterwarnings(
                'ignore',
                message="elementwise comparison failed.*",
                category=FutureWarning,
            )
            for func in all_ufuncs():
                # Different ufuncs vary between returning NotImplemented and
                # raising a TypeError when provided with unknown dtypes.
                # This is a bit unfortunate, but still better than silently
                # accepting an int array.
                try:
                    if func.nin == 1:
                        ret = func(l)
                    elif func.nin == 2:
                        ret = func(l, ints)
                    else:
                        self.fail("Who added a ternary ufunc !?!")
                except TypeError:
                    pass
                else:
                    self.assertIs(ret, NotImplemented)

    @parameter_space(
        __fail_fast=True,
        val=['', 'a', 'not in the array', None],
        missing_value=['', 'a', 'not in the array', None],
    )
    def test_setitem_scalar(self, val, missing_value):
        arr = LabelArray(self.strs, missing_value=missing_value)

        if not arr.has_label(val):
            self.assertTrue(
                (val == 'not in the array')
                or (val is None and missing_value is not None)
            )
            for slicer in [(0, 0), (0, 1), 1]:
                with self.assertRaises(ValueError):
                    arr[slicer] = val
            return

        arr[0, 0] = val
        self.assertEqual(arr[0, 0], val)

        arr[0, 1] = val
        self.assertEqual(arr[0, 1], val)

        arr[1] = val
        if val == missing_value:
            self.assertTrue(arr.is_missing()[1].all())
        else:
            self.assertTrue((arr[1] == val).all())
            self.assertTrue((arr[1].as_string_array() == val).all())

        arr[:, -1] = val
        if val == missing_value:
            self.assertTrue(arr.is_missing()[:, -1].all())
        else:
            self.assertTrue((arr[:, -1] == val).all())
            self.assertTrue((arr[:, -1].as_string_array() == val).all())

        arr[:] = val
        if val == missing_value:
            self.assertTrue(arr.is_missing().all())
        else:
            self.assertFalse(arr.is_missing().any())
            self.assertTrue((arr == val).all())

    def test_setitem_array(self):
        arr = LabelArray(self.strs, missing_value=None)
        orig_arr = arr.copy()

        # Write a row.
        self.assertFalse(
            (arr[0] == arr[1]).all(),
            "This test doesn't test anything because rows 0"
            " and 1 are already equal!"
        )
        arr[0] = arr[1]
        for i in range(arr.shape[1]):
            self.assertEqual(arr[0, i], arr[1, i])

        # Write a column.
        self.assertFalse(
            (arr[:, 0] == arr[:, 1]).all(),
            "This test doesn't test anything because columns 0"
            " and 1 are already equal!"
        )
        arr[:, 0] = arr[:, 1]
        for i in range(arr.shape[0]):
            self.assertEqual(arr[i, 0], arr[i, 1])

        # Write the whole array.
        arr[:] = orig_arr
        check_arrays(arr, orig_arr)
