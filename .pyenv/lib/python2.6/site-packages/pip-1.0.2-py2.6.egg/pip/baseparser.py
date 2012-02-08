"""Base option parser setup"""

import sys
import optparse
import pkg_resources
import os
from distutils.util import strtobool
from pip.backwardcompat import ConfigParser, string_types
from pip.locations import default_config_file, default_log_file


class UpdatingDefaultsHelpFormatter(optparse.IndentedHelpFormatter):
    """Custom help formatter for use in ConfigOptionParser that updates
    the defaults before expanding them, allowing them to show up correctly
    in the help listing"""

    def expand_default(self, option):
        if self.parser is not None:
            self.parser.update_defaults(self.parser.defaults)
        return optparse.IndentedHelpFormatter.expand_default(self, option)


class ConfigOptionParser(optparse.OptionParser):
    """Custom option parser which updates its defaults by by checking the
    configuration files and environmental variables"""

    def __init__(self, *args, **kwargs):
        self.config = ConfigParser.RawConfigParser()
        self.name = kwargs.pop('name')
        self.files = self.get_config_files()
        self.config.read(self.files)
        assert self.name
        optparse.OptionParser.__init__(self, *args, **kwargs)

    def get_config_files(self):
        config_file = os.environ.get('PIP_CONFIG_FILE', False)
        if config_file and os.path.exists(config_file):
            return [config_file]
        return [default_config_file]

    def update_defaults(self, defaults):
        """Updates the given defaults with values from the config files and
        the environ. Does a little special handling for certain types of
        options (lists)."""
        # Then go and look for the other sources of configuration:
        config = {}
        # 1. config files
        for section in ('global', self.name):
            config.update(dict(self.get_config_section(section)))
        # 2. environmental variables
        config.update(dict(self.get_environ_vars()))
        # Then set the options with those values
        for key, val in config.items():
            key = key.replace('_', '-')
            if not key.startswith('--'):
                key = '--%s' % key # only prefer long opts
            option = self.get_option(key)
            if option is not None:
                # ignore empty values
                if not val:
                    continue
                # handle multiline configs
                if option.action == 'append':
                    val = val.split()
                else:
                    option.nargs = 1
                if option.action in ('store_true', 'store_false', 'count'):
                    val = strtobool(val)
                try:
                    val = option.convert_value(key, val)
                except optparse.OptionValueError:
                    e = sys.exc_info()[1]
                    print("An error occured during configuration: %s" % e)
                    sys.exit(3)
                defaults[option.dest] = val
        return defaults

    def get_config_section(self, name):
        """Get a section of a configuration"""
        if self.config.has_section(name):
            return self.config.items(name)
        return []

    def get_environ_vars(self, prefix='PIP_'):
        """Returns a generator with all environmental vars with prefix PIP_"""
        for key, val in os.environ.items():
            if key.startswith(prefix):
                yield (key.replace(prefix, '').lower(), val)

    def get_default_values(self):
        """Overridding to make updating the defaults after instantiation of
        the option parser possible, update_defaults() does the dirty work."""
        if not self.process_default_values:
            # Old, pre-Optik 1.5 behaviour.
            return optparse.Values(self.defaults)

        defaults = self.update_defaults(self.defaults.copy()) # ours
        for option in self._get_all_options():
            default = defaults.get(option.dest)
            if isinstance(default, string_types):
                opt_str = option.get_opt_string()
                defaults[option.dest] = option.check_value(opt_str, default)
        return optparse.Values(defaults)

try:
    pip_dist = pkg_resources.get_distribution('pip')
    version = '%s from %s (python %s)' % (
        pip_dist, pip_dist.location, sys.version[:3])
except pkg_resources.DistributionNotFound:
    # when running pip.py without installing
    version=None

parser = ConfigOptionParser(
    usage='%prog COMMAND [OPTIONS]',
    version=version,
    add_help_option=False,
    formatter=UpdatingDefaultsHelpFormatter(),
    name='global')

parser.add_option(
    '-h', '--help',
    dest='help',
    action='store_true',
    help='Show help')
parser.add_option(
    '-E', '--environment',
    dest='venv',
    metavar='DIR',
    help='virtualenv environment to run pip in (either give the '
    'interpreter or the environment base directory)')
parser.add_option(
    '-s', '--enable-site-packages',
    dest='site_packages',
    action='store_true',
    help='Include site-packages in virtualenv if one is to be '
    'created. Ignored if --environment is not used or '
    'the virtualenv already exists.')
parser.add_option(
    # Defines a default root directory for virtualenvs, relative
    # virtualenvs names/paths are considered relative to it.
    '--virtualenv-base',
    dest='venv_base',
    type='str',
    default='',
    help=optparse.SUPPRESS_HELP)
parser.add_option(
    # Run only if inside a virtualenv, bail if not.
    '--require-virtualenv', '--require-venv',
    dest='require_venv',
    action='store_true',
    default=False,
    help=optparse.SUPPRESS_HELP)
parser.add_option(
    # Use automatically an activated virtualenv instead of installing
    # globally. -E will be ignored if used.
    '--respect-virtualenv', '--respect-venv',
    dest='respect_venv',
    action='store_true',
    default=False,
    help=optparse.SUPPRESS_HELP)

parser.add_option(
    '-v', '--verbose',
    dest='verbose',
    action='count',
    default=0,
    help='Give more output')
parser.add_option(
    '-q', '--quiet',
    dest='quiet',
    action='count',
    default=0,
    help='Give less output')
parser.add_option(
    '--log',
    dest='log',
    metavar='FILENAME',
    help='Log file where a complete (maximum verbosity) record will be kept')
parser.add_option(
    # Writes the log levels explicitely to the log'
    '--log-explicit-levels',
    dest='log_explicit_levels',
    action='store_true',
    default=False,
    help=optparse.SUPPRESS_HELP)
parser.add_option(
    # The default log file
    '--local-log', '--log-file',
    dest='log_file',
    metavar='FILENAME',
    default=default_log_file,
    help=optparse.SUPPRESS_HELP)
parser.add_option(
    # Don't ask for input
    '--no-input',
    dest='no_input',
    action='store_true',
    default=False,
    help=optparse.SUPPRESS_HELP)

parser.add_option(
    '--proxy',
    dest='proxy',
    type='str',
    default='',
    help="Specify a proxy in the form user:passwd@proxy.server:port. "
    "Note that the user:password@ is optional and required only if you "
    "are behind an authenticated proxy.  If you provide "
    "user@proxy.server:port then you will be prompted for a password.")
parser.add_option(
    '--timeout', '--default-timeout',
    metavar='SECONDS',
    dest='timeout',
    type='float',
    default=15,
    help='Set the socket timeout (default %default seconds)')
parser.add_option(
    # The default version control system for editables, e.g. 'svn'
    '--default-vcs',
    dest='default_vcs',
    type='str',
    default='',
    help=optparse.SUPPRESS_HELP)
parser.add_option(
    # A regex to be used to skip requirements
    '--skip-requirements-regex',
    dest='skip_requirements_regex',
    type='str',
    default='',
    help=optparse.SUPPRESS_HELP)

parser.disable_interspersed_args()
