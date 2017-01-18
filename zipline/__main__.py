import errno
import os
from functools import wraps

import click
import logbook
import pandas as pd
from six import text_type

from zipline.data import bundles as bundles_module
from zipline.utils.cli import Date, Timestamp
from zipline.utils.run_algo import _run, load_extensions

try:
    __IPYTHON__
except NameError:
    __IPYTHON__ = False


@click.group()
@click.option(
    '-e',
    '--extension',
    multiple=True,
    help='File or module path to a zipline extension to load.',
)
@click.option(
    '--strict-extensions/--non-strict-extensions',
    is_flag=True,
    help='If --strict-extensions is passed then zipline will not run if it'
    ' cannot load all of the specified extensions. If this is not passed or'
    ' --non-strict-extensions is passed then the failure will be logged but'
    ' execution will continue.',
)
@click.option(
    '--default-extension/--no-default-extension',
    is_flag=True,
    default=True,
    help="Don't load the default zipline extension.py file in $ZIPLINE_HOME.",
)
def main(extension, strict_extensions, default_extension):
    """Top level zipline entry point.
    """
    # install a logbook handler before performing any other operations
    logbook.StderrHandler().push_application()
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


@main.command()
@click.option(
    '-f',
    '--algofile',
    default=None,
    type=click.File('r'),
    help='The file that contains the algorithm to run.',
)
@click.option(
    '-t',
    '--algotext',
    help='The algorithm script to run.',
)
@click.option(
    '-D',
    '--define',
    multiple=True,
    help="Define a name to be bound in the namespace before executing"
    " the algotext. For example '-Dname=value'. The value may be any python"
    " expression. These are evaluated in order so they may refer to previously"
    " defined names.",
)
@click.option(
    '--data-frequency',
    type=click.Choice({'daily', 'minute'}),
    default='daily',
    show_default=True,
    help='The data frequency of the simulation.',
)
@click.option(
    '--capital-base',
    type=float,
    default=10e6,
    show_default=True,
    help='The starting capital for the simulation.',
)
@click.option(
    '-b',
    '--bundle',
    default='quantopian-quandl',
    metavar='BUNDLE-NAME',
    show_default=True,
    help='The data bundle to use for the simulation.',
)
@click.option(
    '--bundle-timestamp',
    type=Timestamp(),
    default=pd.Timestamp.utcnow(),
    show_default=False,
    help='The date to lookup data on or before.\n'
    '[default: <current-time>]'
)
@click.option(
    '-s',
    '--start',
    type=Date(tz='utc', as_timestamp=True),
    help='The start date of the simulation.',
)
@click.option(
    '-e',
    '--end',
    type=Date(tz='utc', as_timestamp=True),
    help='The end date of the simulation.',
)
@click.option(
    '-o',
    '--output',
    default='-',
    metavar='FILENAME',
    show_default=True,
    help="The location to write the perf data. If this is '-' the perf will"
    " be written to stdout.",
)
@click.option(
    '--print-algo/--no-print-algo',
    is_flag=True,
    default=False,
    help='Print the algorithm to stdout.',
)
@click.option(
    '--plot/--no-plot',
    default=os.name != "nt",
    help="plot result"
)
@ipython_only(click.option(
    '--local-namespace/--no-local-namespace',
    is_flag=True,
    default=None,
    help='Should the algorithm methods be resolved in the local namespace.'
))
@click.pass_context
def run(ctx,
        algofile,
        algotext,
        define,
        data_frequency,
        capital_base,
        bundle,
        bundle_timestamp,
        start,
        end,
        output,
        print_algo,
        plot,
        local_namespace):
    """Run a backtest for the given algorithm.
    """
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
            "must specify exactly one of '-f' / '--algofile' or"
            " '-t' / '--algotext'",
        )

    perf = _run(
        initialize=None,
        handle_data=None,
        before_trading_start=None,
        analyze=None,
        algofile=algofile,
        algotext=algotext,
        defines=define,
        data_frequency=data_frequency,
        capital_base=capital_base,
        data=None,
        bundle=bundle,
        bundle_timestamp=bundle_timestamp,
        start=start,
        end=end,
        output=output,
        print_algo=print_algo,
        local_namespace=local_namespace,
        environ=os.environ,
    )

    if plot:
        show_draw_result(algofile.name, perf, bundle)

    if output == '-':
        click.echo(str(perf))
    elif output != os.devnull:  # make the zipline magic not write any data
        perf.to_pickle(output)

    return perf

def show_draw_result(title, results_df, bundle):
    import matplotlib
    from matplotlib import gridspec
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    from zipline.utils import paths
    from datetime import datetime
    plt.style.use('ggplot')

    red = "#aa4643"
    blue = "#4572a7"
    black = "#000000"

    figsize = (18, 6)
    f = plt.figure(title, figsize=figsize)
    gs = gridspec.GridSpec(10, 8)

    # TODO draw logo
    # ax = plt.subplot(gs[:3, -1:])
    # ax.axis("off")
    # filename = os.path.join(paths.zipline_root(), 'zipline.png')
    # img = mpimg.imread(filename)
    # imgplot = ax.imshow(img, interpolation="nearest")
    # ax.autoscale_view()

    # draw risk and portfolio
    series = results_df.iloc[-1]

    font_size = 12
    value_font_size = 11
    label_height, value_height = 0.8, 0.6
    label_height2, value_height2 = 0.35, 0.15

    fig_data = [
        (0.00, label_height, value_height, "Total Returns", "{0:.3%}".format(series.algorithm_period_return), red,
         black),
        (0.15, label_height, value_height, "Annual Returns", "{0:.3%}".format(series.annualized_algorithm_return), red,
         black),
        (0.00, label_height2, value_height2, "Benchmark Total", "{0:.3%}".format(series.benchmark_period_return), blue,
         black),
        (0.15, label_height2, value_height2, "Benchmark Annual", "{0:.3%}".format(series.annualized_benchmark_return),
         blue, black),

        (0.30, label_height, value_height, "Alpha", "{0:.4}".format(series.alpha), black, black),
        (0.40, label_height, value_height, "Beta", "{0:.4}".format(series.beta), black, black),
        (0.55, label_height, value_height, "Sharpe", "{0:.4}".format(series.sharpe), black, black),
        (0.70, label_height, value_height, "Sortino", "{0:.4}".format(series.sortino), black, black),
        (0.85, label_height, value_height, "Information Ratio", "{0:.4}".format(series.information), black, black),

        (0.30, label_height2, value_height2, "Volatility", "{0:.4}".format(series.algo_volatility), black, black),
        (0.40, label_height2, value_height2, "MaxDrawdown", "{0:.3%}".format(series.max_drawdown), black, black),
        # (0.55, label_height2, value_height2, "Tracking Error", "{0:.4}".format(series.tracking_error), black, black),
        # (0.70, label_height2, value_height2, "Downside Risk", "{0:.4}".format(series.downside_risk), black, black),
    ]

    ax = plt.subplot(gs[:3, :-1])
    ax.axis("off")
    for x, y1, y2, label, value, label_color, value_color in fig_data:
        ax.text(x, y1, label, color=label_color, fontsize=font_size)
        ax.text(x, y2, value, color=value_color, fontsize=value_font_size)

    # strategy vs benchmark
    ax = plt.subplot(gs[4:, :])

    ax.get_xaxis().set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.get_yaxis().set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.grid(b=True, which='minor', linewidth=.2)
    ax.grid(b=True, which='major', linewidth=1)

    ax.plot(results_df["benchmark_period_return"], label="benchmark", alpha=1, linewidth=2, color=blue)
    ax.plot(results_df["algorithm_period_return"], label="algorithm", alpha=1, linewidth=2, color=red)

    # manipulate
    vals = ax.get_yticks()
    ax.set_yticklabels(['{:3.2f}%'.format(x * 100) for x in vals])

    leg = plt.legend(loc="upper left")
    leg.get_frame().set_alpha(0.5)

    plt.show()
    now = datetime.now()
    paths.ensure_directory(paths.zipline_path(['perf']))
    plt.savefig(filename=os.path.join(paths.zipline_path(['perf']), os.path.basename(title).split('.')[0] + '_' + bundle + '_' + now.strftime( '%Y%m%dT%H%M%s') + '.png'))

def zipline_magic(line, cell=None):
    """The zipline IPython cell magic.
    """
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
                '--algotext', cell,
                '--output', os.devnull,  # don't write the results by default
            ] + ([
                # these options are set when running in line magic mode
                # set a non None algo text to use the ipython user_ns
                '--algotext', '',
                '--local-namespace',
            ] if cell is None else []) + line.split(),
            '%s%%zipline' % ((cell or '') and '%'),
            # don't use system exit and propogate errors to the caller
            standalone_mode=False,
        )
    except SystemExit as e:
        # https://github.com/mitsuhiko/click/pull/533
        # even in standalone_mode=False `--help` really wants to kill us ;_;
        if e.code:
            raise ValueError('main returned non-zero status code: %d' % e.code)


@main.command()
@click.option(
    '-b',
    '--bundle',
    default='quantopian-quandl',
    metavar='BUNDLE-NAME',
    show_default=True,
    help='The data bundle to ingest.',
)
@click.option(
    '--assets-version',
    type=int,
    multiple=True,
    help='Version of the assets db to which to downgrade.',
)
@click.option(
    '--show-progress/--no-show-progress',
    default=True,
    help='Print progress information to the terminal.'
)
def ingest(bundle, assets_version, show_progress):
    """Ingest the data for the given bundle.
    """
    bundles_module.ingest(
        bundle,
        os.environ,
        pd.Timestamp.utcnow(),
        assets_version,
        show_progress,
    )


@main.command()
@click.option(
    '-b',
    '--bundle',
    default='quantopian-quandl',
    metavar='BUNDLE-NAME',
    show_default=True,
    help='The data bundle to clean.',
)
@click.option(
    '-b',
    '--before',
    type=Timestamp(),
    help='Clear all data before TIMESTAMP.'
    ' This may not be passed with -k / --keep-last',
)
@click.option(
    '-a',
    '--after',
    type=Timestamp(),
    help='Clear all data after TIMESTAMP'
    ' This may not be passed with -k / --keep-last',
)
@click.option(
    '-k',
    '--keep-last',
    type=int,
    metavar='N',
    help='Clear all but the last N downloads.'
    ' This may not be passed with -b / --before or -a / --after',
)
def clean(bundle, before, after, keep_last):
    """Clean up data downloaded with the ingest command.
    """
    bundles_module.clean(
        bundle,
        before,
        after,
        keep_last,
    )


@main.command()
def bundles():
    """List all of the available data bundles.
    """
    for bundle in sorted(bundles_module.bundles.keys()):
        if bundle.startswith('.'):
            # hide the test data
            continue
        try:
            ingestions = list(
                map(text_type, bundles_module.ingestions_for_bundle(bundle))
            )
        except OSError as e:
            if e.errno != errno.ENOENT:
                raise
            ingestions = []

        # If we got no ingestions, either because the directory didn't exist or
        # because there were no entries, print a single message indicating that
        # no ingestions have yet been made.
        for timestamp in ingestions or ["<no ingestions>"]:
            click.echo("%s %s" % (bundle, timestamp))


if __name__ == '__main__':
    main()
