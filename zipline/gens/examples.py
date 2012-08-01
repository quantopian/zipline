from zipline.gens.composites import 

if __name__ == "__main__":
    
    filter = [1,2,3,4]
    #Set up source a. One hour between events.
    args_a = tuple()
    kwargs_a = {'sids'   : [1,2,3,4],
                'start'  : datetime(2012,6,6,0),
                'delta'  : timedelta(minutes = ),
                'filter' : filter
                }
    #Set up source b. One day between events.
    args_b = tuple()
    kwargs_b = {'sids'   : [1,2,3,4],
                'start'  : datetime(2012,6,6,0),
                'delta'  : timedelta(days = 1),
                'filter' : filter
                }
    #Set up source c. One minute between events.
    args_c = tuple()
    kwargs_c = {'sids'   : [1,2,3,4],
                'start'  : datetime(2012,6,6,0),
                'delta'  : timedelta(minutes = 1),
                'filter' : filter
                }

    sources = (SpecificEquityTrades,) * 4
    source_args = (args_a, args_b, args_c, args_d)
    source_kwargs = (kwargs_a, kwargs_b, kwargs_c, kwargs_d)
    
    # Generate our expected source_ids.
    zip_args = zip(source_args, source_kwargs)
    expected_ids = ["SpecificEquityTrades" + hash_args(*args, **kwargs)
                    for args, kwargs in zip_args]
    
    # Pipe our sources into sort.
    sort_out = date_sorted_sources(sources, source_args, source_kwargs)     
