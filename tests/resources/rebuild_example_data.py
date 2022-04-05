#!/usr/bin/env python
import readline  # noqa
import shutil
import tarfile
from code import InteractiveConsole

import click
import matplotlib
import numpy as np
import pandas as pd
from zipline import examples
from zipline.data.bundles import register
from zipline.testing import test_resource_path, tmp_dir
from zipline.testing.fixtures import read_checked_in_benchmark_data
from zipline.testing.predicates import assert_frame_equal
from zipline.utils.cache import dataframe_cache

EXAMPLE_MODULES = examples.load_example_modules()

matplotlib.use("Agg")

banner = """
Please verify that the new performance is more correct than the old
performance.

To do this, please inspect `new` and `old` which are mappings from the name of
the example to the results.

The name `cols_to_check` has been bound to a list of perf columns that we
expect to be reliably deterministic (excluding, e.g. `orders`, which contains
UUIDs).

Calling `changed_results(new, old)` will compute a list of names of results
that produced a different value in one of the `cols_to_check` fields.

If you are sure that the new results are more correct, or that the difference
is acceptable, please call `correct()`. Otherwise, call `incorrect()`.

Note
----
Remember to run this with the other supported versions of pandas!
"""


def changed_results(new, old):
    """
    Get the names of results that changed since the last invocation.

    Useful for verifying that only expected results changed.
    """
    changed = []
    for col in new:
        if col not in old:
            changed.append(col)
            continue
        try:
            assert_frame_equal(
                new[col][examples._cols_to_check],
                old[col][examples._cols_to_check],
            )
        except AssertionError:
            changed.append(col)
    return changed


def eof(*args, **kwargs):
    raise EOFError()


@click.command()
@click.option(
    "--rebuild-input",
    is_flag=True,
    default=False,
    help="Should we rebuild the input data from Yahoo?",
)
@click.pass_context
def main(ctx, rebuild_input):
    """Rebuild the perf data for test_examples"""
    example_path = test_resource_path("example_data.tar.gz")

    with tmp_dir() as d:
        with tarfile.open(example_path) as tar:
            tar.extractall(d.path)

        # The environ here should be the same (modulo the tempdir location)
        # as we use in test_examples.py.
        environ = {"ZIPLINE_ROOT": d.getpath("example_data/root")}

        if rebuild_input:
            raise NotImplementedError(
                "We cannot rebuild input for Yahoo because of "
                "changes Yahoo made to their API, so we cannot "
                "use Yahoo data bundles anymore. This will be fixed in "
                "a future release",
            )

        # we need to register the bundle; it is already ingested and saved in
        # the example_data.tar.gz file
        @register("test")
        def nop_ingest(*args, **kwargs):
            raise NotImplementedError("we cannot rebuild the test buindle")

        new_perf_path = d.getpath(
            "example_data/new_perf/%s" % pd.__version__.replace(".", "-"),
        )
        c = dataframe_cache(
            new_perf_path,
            serialization="pickle:2",
        )
        with c:
            for name in EXAMPLE_MODULES:
                c[name] = examples.run_example(
                    EXAMPLE_MODULES,
                    name,
                    environ=environ,
                    benchmark_returns=read_checked_in_benchmark_data(),
                )

            correct_called = [False]

            console = None

            def _exit(*args, **kwargs):
                console.raw_input = eof

            def correct():
                correct_called[0] = True
                _exit()

            expected_perf_path = d.getpath(
                "example_data/expected_perf/%s" % pd.__version__.replace(".", "-"),
            )

            # allow users to run some analysis to make sure that the new
            # results check out
            console = InteractiveConsole(
                {
                    "correct": correct,
                    "exit": _exit,
                    "incorrect": _exit,
                    "new": c,
                    "np": np,
                    "old": dataframe_cache(
                        expected_perf_path,
                        serialization="pickle",
                    ),
                    "pd": pd,
                    "cols_to_check": examples._cols_to_check,
                    "changed_results": changed_results,
                }
            )
            console.interact(banner)

            if not correct_called[0]:
                ctx.fail(
                    "`correct()` was not called! This means that the new"
                    " results will not be written",
                )

            # move the new results to the expected path
            shutil.rmtree(expected_perf_path)
            shutil.copytree(new_perf_path, expected_perf_path)

        # Clear out all the temporary new perf so it doesn't get added to the
        # tarball.
        shutil.rmtree(d.getpath("example_data/new_perf/"))

        with tarfile.open(example_path, "w|gz") as tar:
            tar.add(d.getpath("example_data"), "example_data")


if __name__ == "__main__":
    main()
