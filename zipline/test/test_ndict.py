from zipline.protocol_utils import ndict, namedict

def test_ndict():
    nd = ndict({})

    # Properties
    assert len(nd) == 0
    assert nd.keys() == []
    assert nd.values() == []
    assert list(nd.iteritems()) == []

    # Accessors
    nd['x'] = 1
    assert nd.x == 1
    assert nd['x'] == 1
    assert nd.get('y') == None
    assert nd.get('y', 'fizzpop') == 'fizzpop'
    assert nd.has_key('x') == True
    assert nd.has_key('y') == False

    assert 'x' in nd
    assert 'y' not in nd

    # Class isolation
    assert '__init__' not in nd
    assert '__iter__' not in nd
    assert not nd.__dict__.has_key('x')
    assert nd.get('__init__') is None

    # Comparison
    nd2 = nd.copy()
    assert id(nd2) != id(nd)
    assert nd2 == nd
    nd2['z'] = 3
    assert nd2 != nd

    class ndictlike(object):
        x = 1

    assert { 'x': 1 } == nd
    assert ndictlike() != nd

    # Deletion
    del nd['x']
    assert not nd.has_key('x')
    assert nd.get('x') is None
