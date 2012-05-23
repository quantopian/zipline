try:
    from zipline.simulator import AddressAllocator
    pass
except Exception, e:
    from zipline.core.devsimulator import AddressAllocator

from zipline.lines import SimulatedTrading

allocator = AddressAllocator(1001)


def get_zipline():
    zipline_test_config = {
        'allocator':allocator,
        'sid':133
    }

    zipline = SimulatedTrading.create_test_zipline(
        **zipline_test_config
    )

    return zipline

def run_basic_zipline():
    zipline = get_zipline()
    zipline.simulate(blocking=True)

def load_ndict():
    from zipline import ndict
    nd = ndict({})
    keyname = 'a %i'
    for i in xrange(1000000):
        nd[keyname % i] = i

    for i in xrange(1000000):
        nd[keyname % i]
