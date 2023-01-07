import errno
import os

import click
import logging
import pandas as pd

import zipline
from zipline.data import bundles as bundles_module
from zipline.utils.calendar_utils import get_calendar
from zipline.utils.compat import wraps
from zipline.utils.cli import Date, Timestamp
from zipline.utils.run_algo import _run, BenchmarkSpec, load_extensions
from zipline.extensions import create_args

try:
    __IPYTHON__
except NameError:
    __IPYTHON__ = False


@click.group()
@click.option(
    "-e",
    "--extension",
    multiple=True,
    help="File or module path to a zipline extension to load.",
)
@click.option(
    "--strict-extensions/--non-strict-extensions",
    is_flag=True,
    help="If --strict-extensions is passed then zipline will not "
    "run if it cannot load all of the specified extensions. "
    "If this is not passed or --non-strict-extensions is passed "
    "then the failure will be logged but execution will continue.",
)
@click.option(
    "--default-extension/--no-default-extension",
    is_flag=True,
    default=True,
    help="Don't load the default zipline extension.py file in $ZIPLINE_HOME.",
)
@click.option(
    "-x",
    multiple=True,
    help="Any custom command line arguments to define, in key=value form.",
)
@click.pass_context
def main(ctx, extension, strict_extensions, default_extension, x):
    """Top level zipline entry point."""
    # install a logging handler before performing any other operations

    logging.basicConfig(
        format="[%(asctime)s-%(levelname)s][%(name)s]\n %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )

    create_args(x, zipline.extension_args)
    load_extensions(
        default_extension,
        extension,
        strict_extensions,
        os.environ,
    )


def extract_option_object(option):
    """Convert a click.option call into a click.Option object.

    Parameters
    ----------
    option : decorator
        A click.option decorator.

    Returns
    -------
    option_object : click.Option
        The option object that this decorator will create.
    """

    @option
    def opt():
        pass

    return opt.__click_params__[0]


def ipython_only(option):
    """Mark that an option should only be exposed in IPython.

    Parameters
    ----------
    option : decorator
        A click.option decorator.

    Returns
    -------
    ipython_only_dec : decorator
        A decorator that correctly applies the argument even when not
        using IPython mode.
    """
    if __IPYTHON__:
        return option

    argname = extract_option_object(option).name

    def d(f):
        @wraps(f)
        def _(*args, **kwargs):
            kwargs[argname] = None
            return f(*args, **kwargs)

        return _

    return d


DEFAULT_BUNDLE = "quandl"


@main.command()
@click.option(
    "-f",
    "--algofile",
    default=None,
    type=click.File("r"),
    help="The file that contains the algorithm to run.",
)
@click.option(
    "-t",
    "--algotext",
    help="The algorithm script to run.",
)
@click.option(
    "-D",
    "--define",
    multiple=True,
    help="Define a name to be bound in the namespace before executing"
    " the algotext. For example '-Dname=value'. The value may be any "
    "python expression. These are evaluated in order so they may refer "
    "to previously defined names.",
)
@click.option(
    "--data-frequency",
    type=click.Choice({"daily", "minute"}),
    default="daily",
    show_default=True,
    help="The data frequency of the simulation.",
)
@click.option(
    "--capital-base",
    type=float,
    default=10e6,
    show_default=True,
    help="The starting capital for the simulation.",
)
@click.option(
    "-b",
    "--bundle",
    default=DEFAULT_BUNDLE,
    metavar="BUNDLE-NAME",
    show_default=True,
    help="The data bundle to use for the simulation.",
)
@click.option(
    "--bundle-timestamp",
    type=Timestamp(),
    default=pd.Timestamp.utcnow(),
    show_default=False,
    help="The date to lookup data on or before.\n" "[default: <current-time>]",
)
@click.option(
    "-bf",
    "--benchmark-file",
    default=None,
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=str),
    help="The csv file that contains the benchmark returns",
)
@click.option(
    "--benchmark-symbol",
    default=None,
    type=click.STRING,
    help="The symbol of the instrument to be used as a benchmark "
    "(should exist in the ingested bundle)",
)
@click.option(
    "--benchmark-sid",
    default=None,
    type=int,
    help="The sid of the instrument to be used as a benchmark "
    "(should exist in the ingested bundle)",
)
@click.option(
    "--no-benchmark",
    is_flag=True,
    default=False,
    help="If passed, use a benchmark of zero returns.",
)
@click.option(
    "-s",
    "--start",
    type=Date(as_timestamp=True),
    help="The start date of the simulation.",
)
@click.option(
    "-e",
    "--end",
    type=Date(as_timestamp=True),
    help="The end date of the simulation.",
)
@click.option(
    "-o",
    "--output",
    default="-",
    metavar="FILENAME",
    show_default=True,
    help="The location to write the perf data. If this is '-' the perf will"
    " be written to stdout.",
)
@click.option(
    "--trading-calendar",
    metavar="TRADING-CALENDAR",
    default="XNYS",
    help="The calendar you want to use e.g. XLON. XNYS is the default.",
)
@click.option(
    "--print-algo/--no-print-algo",
    is_flag=True,
    default=False,
    help="Print the algorithm to stdout.",
)
@click.option(
    "--metrics-set",
    default="default",
    help="The metrics set to use. New metrics sets may be registered in your"
    " extension.py.",
)
@click.option(
    "--blotter",
    default="default",
    help="The blotter to use.",
    show_default=True,
)
@ipython_only(
    click.option(
        "--local-namespace/--no-local-namespace",
        is_flag=True,
        default=None,
        help="Should the algorithm methods be " "resolved in the local namespace.",
    )
)
@click.pass_context
def run(
    ctx,
    algofile,
    algotext,
    define,
    data_frequency,
    capital_base,
    bundle,
    bundle_timestamp,
    benchmark_file,
    benchmark_symbol,
    benchmark_sid,
    no_benchmark,
    start,
    end,
    output,
    trading_calendar,
    print_algo,
    metrics_set,
    local_namespace,
    blotter,
):
    """Run a backtest for the given algorithm."""
    # check that the start and end dates are passed correctly
    if start is None and end is None:
        # check both at the same time to avoid the case where a user
        # does not pass either of these and then passes the first only
        # to be told they need to pass the second argument also
        ctx.fail(
            "must specify dates with '-s' / '--start' and '-e' / '--end'",
        )
    if start is None:
        ctx.fail("must specify a start date with '-s' / '--start'")
    if end is None:
        ctx.fail("must specify an end date with '-e' / '--end'")

    if (algotext is not None) == (algofile is not None):
        ctx.fail(
            "must specify exactly one of '-f' / "
            "'--algofile' or"
            " '-t' / '--algotext'",
        )

    trading_calendar = get_calendar(trading_calendar)

    benchmark_spec = BenchmarkSpec.from_cli_params(
        no_benchmark=no_benchmark,
        benchmark_sid=benchmark_sid,
        benchmark_symbol=benchmark_symbol,
        benchmark_file=benchmark_file,
    )

    return _run(
        initialize=None,
        handle_data=None,
        before_trading_start=None,
        analyze=None,
        algofile=algofile,
        algotext=algotext,
        defines=define,
        data_frequency=data_frequency,
        capital_base=capital_base,
        bundle=bundle,
        bundle_timestamp=bundle_timestamp,
        start=start,
        end=end,
        output=output,
        trading_calendar=trading_calendar,
        print_algo=print_algo,
        metrics_set=metrics_set,
        local_namespace=local_namespace,
        environ=os.environ,
        blotter=blotter,
        benchmark_spec=benchmark_spec,
        custom_loader=None,
    )


def zipline_magic(line, cell=None):
    """The zipline IPython cell magic."""
    load_extensions(
        default=True,
        extensions=[],
        strict=True,
        environ=os.environ,
    )
    try:
        return run.main(
            # put our overrides at the start of the parameter list so that
            # users may pass values with higher precedence
            [
                "--algotext",
                cell,
                "--output",
                os.devnull,  # don't write the results by default
            ]
            + (
                [
                    # these options are set when running in line magic mode
                    # set a non None algo text to use the ipython user_ns
                    "--algotext",
                    "",
                    "--local-namespace",
                ]
                if cell is None
                else []
            )
            + line.split(),
            "%s%%zipline" % ((cell or "") and "%"),
            # don't use system exit and propogate errors to the caller
            standalone_mode=False,
        )
    except SystemExit as exc:
        # https://github.com/mitsuhiko/click/pull/533
        # even in standalone_mode=False `--help` really wants to kill us ;_;
        if exc.code:
            raise ValueError(
                "main returned non-zero status code: %d" % exc.code
            ) from exc


@main.command()
@click.option(
    "-b",
    "--bundle",
    default=DEFAULT_BUNDLE,
    metavar="BUNDLE-NAME",
    show_default=True,
    help="The data bundle to ingest.",
)
@click.option(
    "--assets-version",
    type=int,
    multiple=True,
    help="Version of the assets db to which to downgrade.",
)
@click.option(
    "--show-progress/--no-show-progress",
    default=True,
    help="Print progress information to the terminal.",
)
def ingest(bundle, assets_version, show_progress):
    """Ingest the data for the given bundle."""
    bundles_module.ingest(
        bundle,
        os.environ,
        pd.Timestamp.utcnow(),
        assets_version,
        show_progress,
    )


@main.command()
@click.option(
    "-b",
    "--bundle",
    default=DEFAULT_BUNDLE,
    metavar="BUNDLE-NAME",
    show_default=True,
    help="The data bundle to clean.",
)
@click.option(
    "-e",
    "--before",
    type=Timestamp(),
    help="Clear all data before TIMESTAMP."
    " This may not be passed with -k / --keep-last",
)
@click.option(
    "-a",
    "--after",
    type=Timestamp(),
    help="Clear all data after TIMESTAMP"
    " This may not be passed with -k / --keep-last",
)
@click.option(
    "-k",
    "--keep-last",
    type=int,
    metavar="N",
    help="Clear all but the last N downloads."
    " This may not be passed with -e / --before or -a / --after",
)
def clean(bundle, before, after, keep_last):
    """Clean up data downloaded with the ingest command."""
    bundles_module.clean(
        bundle,
        before,
        after,
        keep_last,
    )


@main.command()
def bundles():
    """List all of the available data bundles."""
    for bundle in sorted(bundles_module.bundles.keys()):
        if bundle.startswith("."):
            # hide the test data
            continue
        try:
            ingestions = list(map(str, bundles_module.ingestions_for_bundle(bundle)))
        except OSError as e:
            if e.errno != errno.ENOENT:
                raise
            ingestions = []

        # If we got no ingestions, either because the directory didn't exist or
        # because there were no entries, print a single message indicating that
        # no ingestions have yet been made.
        for timestamp in ingestions or ["<no ingestions>"]:
            click.echo("%s %s" % (bundle, timestamp))


if __name__ == "__main__":
    main()
