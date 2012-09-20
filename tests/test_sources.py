import zipline.utils.factory as factory

def test_dataframe_source():
    source, df = factory.create_test_df_source()

    for expected_dt, expected_price in df.iterrows():
        sid0 = source.next()
        sid1 = source.next()

        assert expected_dt == sid0.dt == sid1.dt
        assert expected_price[0] == sid0.price
        assert expected_price[1] == sid1.price

