- ValueError: At least one valid asset id is required.
    - 105 test_listing_currency_for_nonexistent_asset (tests.data.test_daily_bars.HDF5DailyBarCanadaTestCase) ... ERROR
    - 111 test_read_known_and_unknown_sids (tests.data.test_daily_bars.HDF5DailyBarCanadaTestCase) ... ERROR
    - 128 test_listing_currency_for_nonexistent_asset (tests.data.test_daily_bars.HDF5DailyBarUSTestCase) ... ERROR
    - 134 test_read_known_and_unknown_sids (tests.data.test_daily_bars.HDF5DailyBarUSTestCase) ... ERROR

- TypeError: Cannot compare tz-naive and tz-aware datetime-like objects
    - 154 test_load_raw_arrays (tests.data.test_dispatch_bar_reader.AssetDispatchSessionBarTestCase) ... ERROR
    
- ValueError: Cannot pass a datetime or Timestamp with tzinfo with the tz parameter. Use tz_convert instead.

session_closes: pd.Series; dtype = datetime64[ns, UTC]
session_closes.values: ndarray, dtype=datetime64[ns]
