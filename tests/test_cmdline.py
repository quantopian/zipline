from unittest import mock

import zipline.__main__ as main
import zipline
from zipline.testing.predicates import assert_equal
from click.testing import CliRunner
from zipline.extensions import (
    Namespace,
    create_args,
    parse_extension_arg,
)
import pytest


class TestCmdLine:
    def test_parse_args(self):
        n = Namespace()

        arg_dict = {}

        arg_list = [
            "key=value",
            "arg1=test1",
            "arg2=test2",
            "arg_3=test3",
            "_arg_4_=test4",
        ]
        for arg in arg_list:
            parse_extension_arg(arg, arg_dict)
        assert_equal(
            arg_dict,
            {
                "_arg_4_": "test4",
                "arg_3": "test3",
                "arg2": "test2",
                "arg1": "test1",
                "key": "value",
            },
        )
        create_args(arg_list, n)
        assert n.key == "value"
        assert n.arg1 == "test1"
        assert n.arg2 == "test2"
        assert n.arg_3 == "test3"
        assert n._arg_4_ == "test4"

        msg = "invalid extension argument '1=test3', " "must be in key=value form"
        with pytest.raises(ValueError, match=msg):
            parse_extension_arg("1=test3", {})
        msg = "invalid extension argument 'arg4 test4', " "must be in key=value form"
        with pytest.raises(ValueError, match=msg):
            parse_extension_arg("arg4 test4", {})
        msg = "invalid extension argument 'arg5.1=test5', " "must be in key=value form"
        with pytest.raises(ValueError, match=msg):
            parse_extension_arg("arg5.1=test5", {})
        msg = (
            "invalid extension argument 'arg6.6arg=test6', " "must be in key=value form"
        )
        with pytest.raises(ValueError, match=msg):
            parse_extension_arg("arg6.6arg=test6", {})
        msg = (
            "invalid extension argument 'arg7.-arg7=test7', "
            "must be in key=value form"
        )
        with pytest.raises(ValueError, match=msg):
            parse_extension_arg("arg7.-arg7=test7", {})

    def test_parse_namespaces(self):
        n = Namespace()

        create_args(
            [
                "first.second.a=blah1",
                "first.second.b=blah2",
                "first.third=blah3",
                "second.a=blah4",
                "second.b=blah5",
            ],
            n,
        )

        assert n.first.second.a == "blah1"
        assert n.first.second.b == "blah2"
        assert n.first.third == "blah3"
        assert n.second.a == "blah4"
        assert n.second.b == "blah5"

        n = Namespace()

        msg = "Conflicting assignments at namespace level 'second'"
        with pytest.raises(ValueError, match=msg):
            create_args(
                [
                    "first.second.a=blah1",
                    "first.second.b=blah2",
                    "first.second=blah3",
                ],
                n,
            )

    def test_user_input(self):
        zipline.extension_args = Namespace()

        runner = CliRunner()
        result = runner.invoke(
            main.main,
            [
                "-xfirst.second.a=blah1",
                "-xfirst.second.b=blah2",
                "-xfirst.third=blah3",
                "-xsecond.a.b=blah4",
                "-xsecond.b.a=blah5",
                "-xa1=value1",
                "-xb_=value2",
                "bundles",
            ],
        )

        assert result.exit_code == 0  # assert successful invocation
        assert zipline.extension_args.first.second.a == "blah1"
        assert zipline.extension_args.first.second.b == "blah2"
        assert zipline.extension_args.first.third == "blah3"
        assert zipline.extension_args.second.a.b == "blah4"
        assert zipline.extension_args.second.b.a == "blah5"
        assert zipline.extension_args.a1 == "value1"
        assert zipline.extension_args.b_ == "value2"

    def test_benchmark_argument_handling(self, tmp_path):
        runner = CliRunner()

        # CLI validates that the algo file exists, so create an empty file.
        algo_path = str(tmp_path / "dummy_algo.py")

        with open(algo_path, "w"):
            pass

        def run_and_get_benchmark_spec(benchmark_args):
            """
            Run the cli, mocking out `main._run`, and return the benchmark_spec
            passed to _run..
            """
            args = [
                "--no-default-extension",
                "run",
                "-s",
                "2014-01-02",
                "-e 2015-01-02",
                "--algofile",
                algo_path,
            ] + benchmark_args

            mock_spec = mock.create_autospec(main._run)

            with mock.patch.object(main, "_run", spec=mock_spec) as mock_run:
                result = runner.invoke(main.main, args, catch_exceptions=False)

            if result.exit_code != 0:
                raise AssertionError(
                    "Cli run failed with {exc}\n\n"
                    "Output was:\n\n"
                    "{output}".format(exc=result.exception, output=result.output),
                )

            mock_run.assert_called_once()

            return mock_run.call_args[1]["benchmark_spec"]

        spec = run_and_get_benchmark_spec([])
        assert spec.benchmark_returns is None
        assert spec.benchmark_file is None
        assert spec.benchmark_sid is None
        assert spec.benchmark_symbol is None
        assert spec.no_benchmark is False

        spec = run_and_get_benchmark_spec(["--no-benchmark"])
        assert spec.benchmark_returns is None
        assert spec.benchmark_file is None
        assert spec.benchmark_sid is None
        assert spec.benchmark_symbol is None
        assert spec.no_benchmark is True

        for symbol in "AAPL", "SPY":
            spec = run_and_get_benchmark_spec(["--benchmark-symbol", symbol])
            assert spec.benchmark_returns is None
            assert spec.benchmark_file is None
            assert spec.benchmark_sid is None
            assert spec.benchmark_symbol is symbol
            assert spec.no_benchmark is False

        for sid in 2, 3:
            spec = run_and_get_benchmark_spec(["--benchmark-sid", str(sid)])
            assert spec.benchmark_returns is None
            assert spec.benchmark_file is None
            assert spec.benchmark_sid == sid
            assert spec.benchmark_symbol is None
            assert spec.no_benchmark is False

        # CLI also validates the returns file exists.
        bm_path = str(tmp_path / "returns.csv")

        with open(bm_path, "w"):
            pass

        spec = run_and_get_benchmark_spec(["--benchmark-file", bm_path])
        assert spec.benchmark_returns is None
        assert spec.benchmark_file == bm_path
        assert spec.benchmark_sid is None
        assert spec.benchmark_symbol is None
        assert spec.no_benchmark is False
