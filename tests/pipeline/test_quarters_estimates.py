def test_shift_quarters_forward():
    quarters = list(range(1, 5))
    shifts = list(range(5))
    expected = [(x, i) for ]
    expected = ((0, 1), (0, 2), (0, 3), (0, 4), (1, 1),
                (0, 2), (0, 3), (0, 4), (1, 1), (1, 2))
    for quarter in quarters:
        for shift in shifts:
            yrs_to_shift, new_qtr = EstimizeLoader.calc_forward_shift(quarter,
                                                                      shift)
            if quarter + shift <= 4:
                assert yrs_to_shift == 0
                assert new_qtr == quarter + shift
            else:
