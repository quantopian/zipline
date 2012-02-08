import os, sys
from pip.req import InstallRequirement, RequirementSet
from pip.req import parse_requirements
from pip.log import logger
from pip.locations import build_prefix, src_prefix
from pip.basecommand import Command
from pip.index import PackageFinder
from pip.exceptions import InstallationError


class InstallCommand(Command):
    name = 'install'
    usage = '%prog [OPTIONS] PACKAGE_NAMES...'
    summary = 'Install packages'
    bundle = False

    def __init__(self):
        super(InstallCommand, self).__init__()
        self.parser.add_option(
            '-e', '--editable',
            dest='editables',
            action='append',
            default=[],
            metavar='VCS+REPOS_URL[@REV]#egg=PACKAGE',
            help='Install a package directly from a checkout. Source will be checked '
            'out into src/PACKAGE (lower-case) and installed in-place (using '
            'setup.py develop). You can run this on an existing directory/checkout (like '
            'pip install -e src/mycheckout). This option may be provided multiple times. '
            'Possible values for VCS are: svn, git, hg and bzr.')
        self.parser.add_option(
            '-r', '--requirement',
            dest='requirements',
            action='append',
            default=[],
            metavar='FILENAME',
            help='Install all the packages listed in the given requirements file.  '
            'This option can be used multiple times.')
        self.parser.add_option(
            '-f', '--find-links',
            dest='find_links',
            action='append',
            default=[],
            metavar='URL',
            help='URL to look for packages at')
        self.parser.add_option(
            '-i', '--index-url', '--pypi-url',
            dest='index_url',
            metavar='URL',
            default='http://pypi.python.org/simple/',
            help='Base URL of Python Package Index (default %default)')
        self.parser.add_option(
            '--extra-index-url',
            dest='extra_index_urls',
            metavar='URL',
            action='append',
            default=[],
            help='Extra URLs of package indexes to use in addition to --index-url')
        self.parser.add_option(
            '--no-index',
            dest='no_index',
            action='store_true',
            default=False,
            help='Ignore package index (only looking at --find-links URLs instead)')
        self.parser.add_option(
            '-M', '--use-mirrors',
            dest='use_mirrors',
            action='store_true',
            default=False,
            help='Use the PyPI mirrors as a fallback in case the main index is down.')
        self.parser.add_option(
            '--mirrors',
            dest='mirrors',
            metavar='URL',
            action='append',
            default=[],
            help='Specific mirror URLs to query when --use-mirrors is used')

        self.parser.add_option(
            '-b', '--build', '--build-dir', '--build-directory',
            dest='build_dir',
            metavar='DIR',
            default=None,
            help='Unpack packages into DIR (default %s) and build from there' % build_prefix)
        self.parser.add_option(
            '-d', '--download', '--download-dir', '--download-directory',
            dest='download_dir',
            metavar='DIR',
            default=None,
            help='Download packages into DIR instead of installing them')
        self.parser.add_option(
            '--download-cache',
            dest='download_cache',
            metavar='DIR',
            default=None,
            help='Cache downloaded packages in DIR')
        self.parser.add_option(
            '--src', '--source', '--source-dir', '--source-directory',
            dest='src_dir',
            metavar='DIR',
            default=None,
            help='Check out --editable packages into DIR (default %s)' % src_prefix)

        self.parser.add_option(
            '-U', '--upgrade',
            dest='upgrade',
            action='store_true',
            help='Upgrade all packages to the newest available version')
        self.parser.add_option(
            '-I', '--ignore-installed',
            dest='ignore_installed',
            action='store_true',
            help='Ignore the installed packages (reinstalling instead)')
        self.parser.add_option(
            '--no-deps', '--no-dependencies',
            dest='ignore_dependencies',
            action='store_true',
            default=False,
            help='Ignore package dependencies')
        self.parser.add_option(
            '--no-install',
            dest='no_install',
            action='store_true',
            help="Download and unpack all packages, but don't actually install them")
        self.parser.add_option(
            '--no-download',
            dest='no_download',
            action="store_true",
            help="Don't download any packages, just install the ones already downloaded "
            "(completes an install run with --no-install)")

        self.parser.add_option(
            '--install-option',
            dest='install_options',
            action='append',
            help="Extra arguments to be supplied to the setup.py install "
            "command (use like --install-option=\"--install-scripts=/usr/local/bin\").  "
            "Use multiple --install-option options to pass multiple options to setup.py install.  "
            "If you are using an option with a directory path, be sure to use absolute path.")

        self.parser.add_option(
            '--global-option',
            dest='global_options',
            action='append',
            help="Extra global options to be supplied to the setup.py"
            "call before the install command")

        self.parser.add_option(
            '--user',
            dest='use_user_site',
            action='store_true',
            help='Install to user-site')

    def _build_package_finder(self, options, index_urls):
        """
        Create a package finder appropriate to this install command.
        This method is meant to be overridden by subclasses, not
        called directly.
        """
        return PackageFinder(find_links=options.find_links,
                             index_urls=index_urls,
                             use_mirrors=options.use_mirrors,
                             mirrors=options.mirrors)

    def run(self, options, args):
        if not options.build_dir:
            options.build_dir = build_prefix
        if not options.src_dir:
            options.src_dir = src_prefix
        if options.download_dir:
            options.no_install = True
            options.ignore_installed = True
        options.build_dir = os.path.abspath(options.build_dir)
        options.src_dir = os.path.abspath(options.src_dir)
        install_options = options.install_options or []
        if options.use_user_site:
            install_options.append('--user')
        global_options = options.global_options or []
        index_urls = [options.index_url] + options.extra_index_urls
        if options.no_index:
            logger.notify('Ignoring indexes: %s' % ','.join(index_urls))
            index_urls = []

        finder = self._build_package_finder(options, index_urls)

        requirement_set = RequirementSet(
            build_dir=options.build_dir,
            src_dir=options.src_dir,
            download_dir=options.download_dir,
            download_cache=options.download_cache,
            upgrade=options.upgrade,
            ignore_installed=options.ignore_installed,
            ignore_dependencies=options.ignore_dependencies)
        for name in args:
            requirement_set.add_requirement(
                InstallRequirement.from_line(name, None))
        for name in options.editables:
            requirement_set.add_requirement(
                InstallRequirement.from_editable(name, default_vcs=options.default_vcs))
        for filename in options.requirements:
            for req in parse_requirements(filename, finder=finder, options=options):
                requirement_set.add_requirement(req)

        if not requirement_set.has_requirements:
            if options.find_links:
                raise InstallationError('You must give at least one '
                    'requirement to %s (maybe you meant "pip install %s"?)'
                    % (self.name, " ".join(options.find_links)))
            raise InstallationError('You must give at least one requirement '
                'to %(name)s (see "pip help %(name)s")' % dict(name=self.name))

        if (options.use_user_site and
            sys.version_info < (2, 6)):
            raise InstallationError('--user is only supported in Python version 2.6 and newer')

        import setuptools
        if (options.use_user_site and
            requirement_set.has_editables and
            not getattr(setuptools, '_distribute', False)):

            raise InstallationError('--user --editable not supported with setuptools, use distribute')

        if not options.no_download:
            requirement_set.prepare_files(finder, force_root_egg_info=self.bundle, bundle=self.bundle)
        else:
            requirement_set.locate_files()

        if not options.no_install and not self.bundle:
            requirement_set.install(install_options, global_options)
            installed = ' '.join([req.name for req in
                                  requirement_set.successfully_installed])
            if installed:
                logger.notify('Successfully installed %s' % installed)
        elif not self.bundle:
            downloaded = ' '.join([req.name for req in
                                   requirement_set.successfully_downloaded])
            if downloaded:
                logger.notify('Successfully downloaded %s' % downloaded)
        elif self.bundle:
            requirement_set.create_bundle(self.bundle_filename)
            logger.notify('Created bundle in %s' % self.bundle_filename)
        # Clean up
        if not options.no_install:
            requirement_set.cleanup_files(bundle=self.bundle)
        return requirement_set


InstallCommand()
