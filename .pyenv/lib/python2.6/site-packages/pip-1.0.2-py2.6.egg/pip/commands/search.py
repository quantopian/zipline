import sys
import textwrap
import pkg_resources
import pip.download
from pip.basecommand import Command
from pip.util import get_terminal_size
from pip.log import logger
from pip.backwardcompat import xmlrpclib, reduce, cmp
from distutils.version import StrictVersion, LooseVersion


class SearchCommand(Command):
    name = 'search'
    usage = '%prog QUERY'
    summary = 'Search PyPI'

    def __init__(self):
        super(SearchCommand, self).__init__()
        self.parser.add_option(
            '--index',
            dest='index',
            metavar='URL',
            default='http://pypi.python.org/pypi',
            help='Base URL of Python Package Index (default %default)')

    def run(self, options, args):
        if not args:
            logger.warn('ERROR: Missing required argument (search query).')
            return
        query = args
        index_url = options.index

        pypi_hits = self.search(query, index_url)
        hits = transform_hits(pypi_hits)

        terminal_width = None
        if sys.stdout.isatty():
            terminal_width = get_terminal_size()[0]

        print_results(hits, terminal_width=terminal_width)

    def search(self, query, index_url):
        pypi = xmlrpclib.ServerProxy(index_url, pip.download.xmlrpclib_transport)
        hits = pypi.search({'name': query, 'summary': query}, 'or')
        return hits


def transform_hits(hits):
    """
    The list from pypi is really a list of versions. We want a list of
    packages with the list of versions stored inline. This converts the
    list from pypi into one we can use.
    """
    packages = {}
    for hit in hits:
        name = hit['name']
        summary = hit['summary']
        version = hit['version']
        score = hit['_pypi_ordering']

        if name not in packages.keys():
            packages[name] = {'name': name, 'summary': summary, 'versions': [version], 'score': score}
        else:
            packages[name]['versions'].append(version)

            # if this is the highest version, replace summary and score
            if version == highest_version(packages[name]['versions']):
                packages[name]['summary'] = summary
                packages[name]['score'] = score

    # each record has a unique name now, so we will convert the dict into a list sorted by score
    package_list = sorted(packages.values(), key=lambda x: x['score'], reverse=True)
    return package_list


def print_results(hits, name_column_width=25, terminal_width=None):
    installed_packages = [p.project_name for p in pkg_resources.working_set]
    for hit in hits:
        name = hit['name']
        summary = hit['summary'] or ''
        if terminal_width is not None:
            # wrap and indent summary to fit terminal
            summary = textwrap.wrap(summary, terminal_width - name_column_width - 5)
            summary = ('\n' + ' ' * (name_column_width + 3)).join(summary)
        line = '%s - %s' % (name.ljust(name_column_width), summary)
        try:
            logger.notify(line)
            if name in installed_packages:
                dist = pkg_resources.get_distribution(name)
                logger.indent += 2
                try:
                    latest = highest_version(hit['versions'])
                    if dist.version == latest:
                        logger.notify('INSTALLED: %s (latest)' % dist.version)
                    else:
                        logger.notify('INSTALLED: %s' % dist.version)
                        logger.notify('LATEST:    %s' % latest)
                finally:
                    logger.indent -= 2
        except UnicodeEncodeError:
            pass


def compare_versions(version1, version2):
    try:
        return cmp(StrictVersion(version1), StrictVersion(version2))
    # in case of abnormal version number, fall back to LooseVersion
    except ValueError:
        return cmp(LooseVersion(version1), LooseVersion(version2))


def highest_version(versions):
    return reduce((lambda v1, v2: compare_versions(v1, v2) == 1 and v1 or v2), versions)


SearchCommand()
