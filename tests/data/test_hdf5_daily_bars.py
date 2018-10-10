import numpy as np

from zipline.data.hdf5_daily_bars import HDF5DailyBarReader, HDF5DailyBarWriter
import zipline.testing.fixtures as zp_fixtures
from zipline.testing.predicates import assert_equal


class H5WriterTestCase(zp_fixtures.WithTmpDir,
                       zp_fixtures.ZiplineTestCase):

    def test_write_empty_country(self):
        """
        Test that we can write an empty country to an HDF5 daily bar writer.

        This is useful functionality for some tests, but it requires a bunch of
        special cased logic in the writer.
        """
        path = self.tmpdir.getpath('empty.h5')
        writer = HDF5DailyBarWriter(path, date_chunk_size=30)
        writer.write_from_sid_df_pairs('US', iter(()))

        reader = HDF5DailyBarReader.from_path(path, 'US')

        assert_equal(reader.sids, np.array([], dtype='int64'))

        empty_dates = np.array([], dtype='datetime64[ns]')
        assert_equal(reader.asset_start_dates, empty_dates)
        assert_equal(reader.asset_end_dates, empty_dates)
        assert_equal(reader.dates, empty_dates)
