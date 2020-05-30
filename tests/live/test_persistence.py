# fix to allow zip_longest on Python 2.X and 3.X
try:                                    # Python 3
    from itertools import zip_longest
except ImportError:                     # Python 2
    from itertools import izip_longest as zip_longest

import os

from mock import sentinel
from testfixtures import tempdir


from zipline.algorithm_live import LiveTradingAlgorithm
from zipline.testing.fixtures import WithSimParams
from zipline.utils.serialization_utils import load_context, store_context
from zipline.testing.fixtures import (ZiplineTestCase,
                                      WithDataPortal)


class TestPersistence(WithSimParams,
                      WithDataPortal,
                      ZiplineTestCase):
    def noop(*args, **kwargs):
        pass

    def make_trading_algo(self, state_filename, algo_filename=None,
                          initialize=noop, handle_data=noop):
        return LiveTradingAlgorithm(
            namespace={},
            asset_finder=self.asset_finder,
            sim_params=self.make_simparams(),
            state_filename=state_filename,
            algo_filename=algo_filename,
            initialize=initialize,
            handle_data=handle_data,
            script=None)

    @tempdir()
    def test_live_trading_algorithm_creates_state_file(self, tmpdir):
        algo_text = b"""
        def initialize(context):
            pass

        def handle_data(context, data):
            pass
        """
        algo_filename = "algo.py"
        algo_path = tmpdir.write(algo_filename, algo_text)
        state_filename = os.path.join(tmpdir.path, "state_file")

        algo = self.make_trading_algo(state_filename, algo_path)

        assert not os.path.exists(state_filename)

        algo.initialize()

        assert os.path.getsize(state_filename) > 0

    @tempdir()
    def test_live_trading_algorithm_loads_state_file(self, tmpdir):
        state_filename = os.path.join(tmpdir.path, "state_file")

        def initialize_1(context):
            context.state_from_initialize = 7

        def handle_data_1(context, data):
            context.state_from_handle_data = 11

        algo_1 = self.make_trading_algo(state_filename,
                                        initialize=initialize_1,
                                        handle_data=handle_data_1)

        algo_1.initialize()
        algo_1.handle_data(data=sentinel.data)

        def initialize_2(context):
            assert False, "initialize shouldn't be called if state is loaded"

        def handle_data_2(context, data):
            assert False, "handle_data shouldn't be called"

        algo_2 = self.make_trading_algo(state_filename,
                                        initialize=initialize_2,
                                        handle_data=handle_data_2)
        algo_2.initialize()

        assert algo_2.state_from_initialize == 7
        assert algo_2.state_from_handle_data == 11

    @tempdir()
    def test_state_load_with_corrupt_state(self, tmpdir):
        state_filename = os.path.join(tmpdir.path, "state_file")

        algo_1 = self.make_trading_algo(state_filename,
                                        initialize=TestPersistence.noop,
                                        handle_data=TestPersistence.noop)

        tmpdir.write("state_file", b"roken")

        with self.assertRaises(ValueError) as e:
            algo_1.initialize()
        assert "state file" in str(e.exception)

    @tempdir()
    def test_context_persistence_checksum(self, tmpdir):
        algo_text_1 = b"""
        def initialize(context):
            context.state_from_initialize = 11

        def handle_data(context, data):
            context.state_from_handle_data = 13
        """
        algo_filename_1 = "algo_1.py"
        algo_path_1 = tmpdir.write(algo_filename_1, algo_text_1)

        state_filename_1 = os.path.join(tmpdir.path, "state_file_1")
        algo_1 = self.make_trading_algo(state_filename_1,
                                        algo_filename=algo_path_1)

        algo_1.initialize()
        algo_1.handle_data(data=sentinel.data)

        algo_text_2 = b"""
        def initialize(context):
            context.state_from_initialize = 7

        def handle_data(context, data):
            context.state_from_handle_data = 5
        """
        algo_filename_2 = "algo_2.py"
        algo_path_2 = tmpdir.write(algo_filename_2, algo_text_2)

        state_filename_2 = os.path.join(tmpdir.path, "state_file_2")
        algo_2 = self.make_trading_algo(state_filename_2,
                                        algo_filename=algo_path_2)

        algo_2.initialize()
        algo_2.handle_data(data=sentinel.data)

        algo_1_wrong_state = self.make_trading_algo(state_filename_2,
                                                    algo_filename=algo_path_1)

        algo_2_wrong_state = self.make_trading_algo(state_filename_1,
                                                    algo_filename=algo_path_2)

        with self.assertRaises(TypeError) as e1:
            algo_1_wrong_state.initialize()
        assert "state file" in str(e1.exception)

        with self.assertRaises(TypeError) as e2:
            algo_2_wrong_state.initialize()
        assert "state file" in str(e2.exception)

    @tempdir()
    def test_context_persistence_exclude_list(self, tmpdir):
        class Context(object):
            def __init__(self, rsi=None, sma=None,
                         trading_client=None, event_manager=None):
                self.rsi = rsi
                self.sma = sma
                self.trading_client = trading_client
                self.event_manager = event_manager

        context = Context(rsi=17.2, sma=40.4, trading_client=lambda x: x + 3,
                          event_manager=[None, False])

        exclude_list = ['trading_client', 'event_manager']
        checksum = 'robocop'

        state_file_path = os.path.join(tmpdir.path, "state_file")

        store_context(state_file_path, context, checksum, exclude_list)

        restored_context = Context()
        load_context(state_file_path, restored_context, checksum)

        assert restored_context.__dict__.keys() == context.__dict__.keys()
        assert restored_context.rsi == context.rsi
        assert restored_context.sma == context.sma
        assert restored_context.trading_client is None
        assert restored_context.event_manager is None

