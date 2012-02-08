import sys
import os
import shutil
import re
import zipfile
import pkg_resources
import tempfile
from pip.locations import bin_py, running_under_virtualenv
from pip.exceptions import InstallationError, UninstallationError
from pip.vcs import vcs
from pip.log import logger
from pip.util import display_path, rmtree
from pip.util import ask, backup_dir
from pip.util import is_installable_dir, is_local, dist_is_local
from pip.util import renames, normalize_path, egg_link_path
from pip.util import make_path_relative
from pip import call_subprocess
from pip.backwardcompat import (any, copytree, urlparse, urllib,
                                ConfigParser, string_types, HTTPError,
                                FeedParser, get_python_version,
                                b)
from pip.index import Link
from pip.locations import build_prefix
from pip.download import (get_file_content, is_url, url_to_path,
                          path_to_url, is_archive_file,
                          unpack_vcs_link, is_vcs_url, is_file_url,
                          unpack_file_url, unpack_http_url)


PIP_DELETE_MARKER_FILENAME = 'pip-delete-this-directory.txt'


class InstallRequirement(object):

    def __init__(self, req, comes_from, source_dir=None, editable=False,
                 url=None, update=True):
        if isinstance(req, string_types):
            req = pkg_resources.Requirement.parse(req)
        self.req = req
        self.comes_from = comes_from
        self.source_dir = source_dir
        self.editable = editable
        self.url = url
        self._egg_info_path = None
        # This holds the pkg_resources.Distribution object if this requirement
        # is already available:
        self.satisfied_by = None
        # This hold the pkg_resources.Distribution object if this requirement
        # conflicts with another installed distribution:
        self.conflicts_with = None
        self._temp_build_dir = None
        self._is_bundle = None
        # True if the editable should be updated:
        self.update = update
        # Set to True after successful installation
        self.install_succeeded = None
        # UninstallPathSet of uninstalled distribution (for possible rollback)
        self.uninstalled = None

    @classmethod
    def from_editable(cls, editable_req, comes_from=None, default_vcs=None):
        name, url = parse_editable(editable_req, default_vcs)
        if url.startswith('file:'):
            source_dir = url_to_path(url)
        else:
            source_dir = None
        return cls(name, comes_from, source_dir=source_dir, editable=True, url=url)

    @classmethod
    def from_line(cls, name, comes_from=None):
        """Creates an InstallRequirement from a name, which might be a
        requirement, directory containing 'setup.py', filename, or URL.
        """
        url = None
        name = name.strip()
        req = None
        path = os.path.normpath(os.path.abspath(name))
        link = None

        if is_url(name):
            link = Link(name)
        elif os.path.isdir(path) and (os.path.sep in name or name.startswith('.')):
            if not is_installable_dir(path):
                raise InstallationError("Directory %r is not installable. File 'setup.py' not found.", name)
            link = Link(path_to_url(name))
        elif is_archive_file(path):
            if not os.path.isfile(path):
                logger.warn('Requirement %r looks like a filename, but the file does not exist', name)
            link = Link(path_to_url(name))

        # If the line has an egg= definition, but isn't editable, pull the requirement out.
        # Otherwise, assume the name is the req for the non URL/path/archive case.
        if link and req is None:
          url = link.url_fragment
          req = link.egg_fragment

          # Handle relative file URLs
          if link.scheme == 'file' and re.search(r'\.\./', url):
            url = path_to_url(os.path.normpath(os.path.abspath(link.path)))

        else:
          req = name

        return cls(req, comes_from, url=url)

    def __str__(self):
        if self.req:
            s = str(self.req)
            if self.url:
                s += ' from %s' % self.url
        else:
            s = self.url
        if self.satisfied_by is not None:
            s += ' in %s' % display_path(self.satisfied_by.location)
        if self.comes_from:
            if isinstance(self.comes_from, string_types):
                comes_from = self.comes_from
            else:
                comes_from = self.comes_from.from_path()
            if comes_from:
                s += ' (from %s)' % comes_from
        return s

    def from_path(self):
        if self.req is None:
            return None
        s = str(self.req)
        if self.comes_from:
            if isinstance(self.comes_from, string_types):
                comes_from = self.comes_from
            else:
                comes_from = self.comes_from.from_path()
            if comes_from:
                s += '->' + comes_from
        return s

    def build_location(self, build_dir, unpack=True):
        if self._temp_build_dir is not None:
            return self._temp_build_dir
        if self.req is None:
            self._temp_build_dir = tempfile.mkdtemp('-build', 'pip-')
            self._ideal_build_dir = build_dir
            return self._temp_build_dir
        if self.editable:
            name = self.name.lower()
        else:
            name = self.name
        # FIXME: Is there a better place to create the build_dir? (hg and bzr need this)
        if not os.path.exists(build_dir):
            _make_build_dir(build_dir)
        return os.path.join(build_dir, name)

    def correct_build_location(self):
        """If the build location was a temporary directory, this will move it
        to a new more permanent location"""
        if self.source_dir is not None:
            return
        assert self.req is not None
        assert self._temp_build_dir
        old_location = self._temp_build_dir
        new_build_dir = self._ideal_build_dir
        del self._ideal_build_dir
        if self.editable:
            name = self.name.lower()
        else:
            name = self.name
        new_location = os.path.join(new_build_dir, name)
        if not os.path.exists(new_build_dir):
            logger.debug('Creating directory %s' % new_build_dir)
            _make_build_dir(new_build_dir)
        if os.path.exists(new_location):
            raise InstallationError(
                'A package already exists in %s; please remove it to continue'
                % display_path(new_location))
        logger.debug('Moving package %s from %s to new location %s'
                     % (self, display_path(old_location), display_path(new_location)))
        shutil.move(old_location, new_location)
        self._temp_build_dir = new_location
        self.source_dir = new_location
        self._egg_info_path = None

    @property
    def name(self):
        if self.req is None:
            return None
        return self.req.project_name

    @property
    def url_name(self):
        if self.req is None:
            return None
        return urllib.quote(self.req.unsafe_name)

    @property
    def setup_py(self):
        return os.path.join(self.source_dir, 'setup.py')

    def run_egg_info(self, force_root_egg_info=False):
        assert self.source_dir
        if self.name:
            logger.notify('Running setup.py egg_info for package %s' % self.name)
        else:
            logger.notify('Running setup.py egg_info for package from %s' % self.url)
        logger.indent += 2
        try:
            script = self._run_setup_py
            script = script.replace('__SETUP_PY__', repr(self.setup_py))
            script = script.replace('__PKG_NAME__', repr(self.name))
            # We can't put the .egg-info files at the root, because then the source code will be mistaken
            # for an installed egg, causing problems
            if self.editable or force_root_egg_info:
                egg_base_option = []
            else:
                egg_info_dir = os.path.join(self.source_dir, 'pip-egg-info')
                if not os.path.exists(egg_info_dir):
                    os.makedirs(egg_info_dir)
                egg_base_option = ['--egg-base', 'pip-egg-info']
            call_subprocess(
                [sys.executable, '-c', script, 'egg_info'] + egg_base_option,
                cwd=self.source_dir, filter_stdout=self._filter_install, show_stdout=False,
                command_level=logger.VERBOSE_DEBUG,
                command_desc='python setup.py egg_info')
        finally:
            logger.indent -= 2
        if not self.req:
            self.req = pkg_resources.Requirement.parse(
                "%(Name)s==%(Version)s" % self.pkg_info())
            self.correct_build_location()

    ## FIXME: This is a lame hack, entirely for PasteScript which has
    ## a self-provided entry point that causes this awkwardness
    _run_setup_py = """
__file__ = __SETUP_PY__
from setuptools.command import egg_info
def replacement_run(self):
    self.mkpath(self.egg_info)
    installer = self.distribution.fetch_build_egg
    for ep in egg_info.iter_entry_points('egg_info.writers'):
        # require=False is the change we're making:
        writer = ep.load(require=False)
        if writer:
            writer(self, ep.name, egg_info.os.path.join(self.egg_info,ep.name))
    self.find_sources()
egg_info.egg_info.run = replacement_run
exec(compile(open(__file__).read().replace('\\r\\n', '\\n'), __file__, 'exec'))
"""

    def egg_info_data(self, filename):
        if self.satisfied_by is not None:
            if not self.satisfied_by.has_metadata(filename):
                return None
            return self.satisfied_by.get_metadata(filename)
        assert self.source_dir
        filename = self.egg_info_path(filename)
        if not os.path.exists(filename):
            return None
        fp = open(filename, 'r')
        data = fp.read()
        fp.close()
        return data

    def egg_info_path(self, filename):
        if self._egg_info_path is None:
            if self.editable:
                base = self.source_dir
            else:
                base = os.path.join(self.source_dir, 'pip-egg-info')
            filenames = os.listdir(base)
            if self.editable:
                filenames = []
                for root, dirs, files in os.walk(base):
                    for dir in vcs.dirnames:
                        if dir in dirs:
                            dirs.remove(dir)
                    for dir in dirs:
                        # Don't search in anything that looks like a virtualenv environment
                        if (os.path.exists(os.path.join(root, dir, 'bin', 'python'))
                            or os.path.exists(os.path.join(root, dir, 'Scripts', 'Python.exe'))):
                            dirs.remove(dir)
                        # Also don't search through tests
                        if dir == 'test' or dir == 'tests':
                            dirs.remove(dir)
                    filenames.extend([os.path.join(root, dir)
                                     for dir in dirs])
                filenames = [f for f in filenames if f.endswith('.egg-info')]

            if not filenames:
                raise InstallationError('No files/directores in %s (from %s)' % (base, filename))
            assert filenames, "No files/directories in %s (from %s)" % (base, filename)

            # if we have more than one match, we pick the toplevel one.  This can
            # easily be the case if there is a dist folder which contains an
            # extracted tarball for testing purposes.
            if len(filenames) > 1:
                filenames.sort(key=lambda x: x.count(os.path.sep) +
                                             (os.path.altsep and
                                              x.count(os.path.altsep) or 0))
            self._egg_info_path = os.path.join(base, filenames[0])
        return os.path.join(self._egg_info_path, filename)

    def egg_info_lines(self, filename):
        data = self.egg_info_data(filename)
        if not data:
            return []
        result = []
        for line in data.splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            result.append(line)
        return result

    def pkg_info(self):
        p = FeedParser()
        data = self.egg_info_data('PKG-INFO')
        if not data:
            logger.warn('No PKG-INFO file found in %s' % display_path(self.egg_info_path('PKG-INFO')))
        p.feed(data or '')
        return p.close()

    @property
    def dependency_links(self):
        return self.egg_info_lines('dependency_links.txt')

    _requirements_section_re = re.compile(r'\[(.*?)\]')

    def requirements(self, extras=()):
        in_extra = None
        for line in self.egg_info_lines('requires.txt'):
            match = self._requirements_section_re.match(line)
            if match:
                in_extra = match.group(1)
                continue
            if in_extra and in_extra not in extras:
                # Skip requirement for an extra we aren't requiring
                continue
            yield line

    @property
    def absolute_versions(self):
        for qualifier, version in self.req.specs:
            if qualifier == '==':
                yield version

    @property
    def installed_version(self):
        return self.pkg_info()['version']

    def assert_source_matches_version(self):
        assert self.source_dir
        if self.comes_from is None:
            # We don't check the versions of things explicitly installed.
            # This makes, e.g., "pip Package==dev" possible
            return
        version = self.installed_version
        if version not in self.req:
            logger.fatal(
                'Source in %s has the version %s, which does not match the requirement %s'
                % (display_path(self.source_dir), version, self))
            raise InstallationError(
                'Source in %s has version %s that conflicts with %s'
                % (display_path(self.source_dir), version, self))
        else:
            logger.debug('Source in %s has version %s, which satisfies requirement %s'
                         % (display_path(self.source_dir), version, self))

    def update_editable(self, obtain=True):
        if not self.url:
            logger.info("Cannot update repository at %s; repository location is unknown" % self.source_dir)
            return
        assert self.editable
        assert self.source_dir
        if self.url.startswith('file:'):
            # Static paths don't get updated
            return
        assert '+' in self.url, "bad url: %r" % self.url
        if not self.update:
            return
        vc_type, url = self.url.split('+', 1)
        backend = vcs.get_backend(vc_type)
        if backend:
            vcs_backend = backend(self.url)
            if obtain:
                vcs_backend.obtain(self.source_dir)
            else:
                vcs_backend.export(self.source_dir)
        else:
            assert 0, (
                'Unexpected version control type (in %s): %s'
                % (self.url, vc_type))

    def uninstall(self, auto_confirm=False):
        """
        Uninstall the distribution currently satisfying this requirement.

        Prompts before removing or modifying files unless
        ``auto_confirm`` is True.

        Refuses to delete or modify files outside of ``sys.prefix`` -
        thus uninstallation within a virtual environment can only
        modify that virtual environment, even if the virtualenv is
        linked to global site-packages.

        """
        if not self.check_if_exists():
            raise UninstallationError("Cannot uninstall requirement %s, not installed" % (self.name,))
        dist = self.satisfied_by or self.conflicts_with

        paths_to_remove = UninstallPathSet(dist)

        pip_egg_info_path = os.path.join(dist.location,
                                         dist.egg_name()) + '.egg-info'
        # workaround for http://bugs.debian.org/cgi-bin/bugreport.cgi?bug=618367
        debian_egg_info_path = pip_egg_info_path.replace(
            '-py%s' % pkg_resources.PY_MAJOR, '')
        easy_install_egg = dist.egg_name() + '.egg'
        develop_egg_link = egg_link_path(dist)

        pip_egg_info_exists = os.path.exists(pip_egg_info_path)
        debian_egg_info_exists = os.path.exists(debian_egg_info_path)
        if pip_egg_info_exists or debian_egg_info_exists:
            # package installed by pip
            if pip_egg_info_exists:
                egg_info_path = pip_egg_info_path
            else:
                egg_info_path = debian_egg_info_path
            paths_to_remove.add(egg_info_path)
            if dist.has_metadata('installed-files.txt'):
                for installed_file in dist.get_metadata('installed-files.txt').splitlines():
                    path = os.path.normpath(os.path.join(egg_info_path, installed_file))
                    paths_to_remove.add(path)
            if dist.has_metadata('top_level.txt'):
                if dist.has_metadata('namespace_packages.txt'):
                    namespaces = dist.get_metadata('namespace_packages.txt')
                else:
                    namespaces = []
                for top_level_pkg in [p for p
                                      in dist.get_metadata('top_level.txt').splitlines()
                                      if p and p not in namespaces]:
                    path = os.path.join(dist.location, top_level_pkg)
                    paths_to_remove.add(path)
                    paths_to_remove.add(path + '.py')
                    paths_to_remove.add(path + '.pyc')

        elif dist.location.endswith(easy_install_egg):
            # package installed by easy_install
            paths_to_remove.add(dist.location)
            easy_install_pth = os.path.join(os.path.dirname(dist.location),
                                            'easy-install.pth')
            paths_to_remove.add_pth(easy_install_pth, './' + easy_install_egg)

        elif os.path.isfile(develop_egg_link):
            # develop egg
            fh = open(develop_egg_link, 'r')
            link_pointer = os.path.normcase(fh.readline().strip())
            fh.close()
            assert (link_pointer == dist.location), 'Egg-link %s does not match installed location of %s (at %s)' % (link_pointer, self.name, dist.location)
            paths_to_remove.add(develop_egg_link)
            easy_install_pth = os.path.join(os.path.dirname(develop_egg_link),
                                            'easy-install.pth')
            paths_to_remove.add_pth(easy_install_pth, dist.location)

        # find distutils scripts= scripts
        if dist.has_metadata('scripts') and dist.metadata_isdir('scripts'):
            for script in dist.metadata_listdir('scripts'):
                paths_to_remove.add(os.path.join(bin_py, script))
                if sys.platform == 'win32':
                    paths_to_remove.add(os.path.join(bin_py, script) + '.bat')

        # find console_scripts
        if dist.has_metadata('entry_points.txt'):
            config = ConfigParser.SafeConfigParser()
            config.readfp(FakeFile(dist.get_metadata_lines('entry_points.txt')))
            if config.has_section('console_scripts'):
                for name, value in config.items('console_scripts'):
                    paths_to_remove.add(os.path.join(bin_py, name))
                    if sys.platform == 'win32':
                        paths_to_remove.add(os.path.join(bin_py, name) + '.exe')
                        paths_to_remove.add(os.path.join(bin_py, name) + '.exe.manifest')
                        paths_to_remove.add(os.path.join(bin_py, name) + '-script.py')

        paths_to_remove.remove(auto_confirm)
        self.uninstalled = paths_to_remove

    def rollback_uninstall(self):
        if self.uninstalled:
            self.uninstalled.rollback()
        else:
            logger.error("Can't rollback %s, nothing uninstalled."
                         % (self.project_name,))

    def commit_uninstall(self):
        if self.uninstalled:
            self.uninstalled.commit()
        else:
            logger.error("Can't commit %s, nothing uninstalled."
                         % (self.project_name,))

    def archive(self, build_dir):
        assert self.source_dir
        create_archive = True
        archive_name = '%s-%s.zip' % (self.name, self.installed_version)
        archive_path = os.path.join(build_dir, archive_name)
        if os.path.exists(archive_path):
            response = ask('The file %s exists. (i)gnore, (w)ipe, (b)ackup '
                           % display_path(archive_path), ('i', 'w', 'b'))
            if response == 'i':
                create_archive = False
            elif response == 'w':
                logger.warn('Deleting %s' % display_path(archive_path))
                os.remove(archive_path)
            elif response == 'b':
                dest_file = backup_dir(archive_path)
                logger.warn('Backing up %s to %s'
                            % (display_path(archive_path), display_path(dest_file)))
                shutil.move(archive_path, dest_file)
        if create_archive:
            zip = zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED)
            dir = os.path.normcase(os.path.abspath(self.source_dir))
            for dirpath, dirnames, filenames in os.walk(dir):
                if 'pip-egg-info' in dirnames:
                    dirnames.remove('pip-egg-info')
                for dirname in dirnames:
                    dirname = os.path.join(dirpath, dirname)
                    name = self._clean_zip_name(dirname, dir)
                    zipdir = zipfile.ZipInfo(self.name + '/' + name + '/')
                    zipdir.external_attr = 0x1ED << 16 # 0o755
                    zip.writestr(zipdir, '')
                for filename in filenames:
                    if filename == PIP_DELETE_MARKER_FILENAME:
                        continue
                    filename = os.path.join(dirpath, filename)
                    name = self._clean_zip_name(filename, dir)
                    zip.write(filename, self.name + '/' + name)
            zip.close()
            logger.indent -= 2
            logger.notify('Saved %s' % display_path(archive_path))

    def _clean_zip_name(self, name, prefix):
        assert name.startswith(prefix+os.path.sep), (
            "name %r doesn't start with prefix %r" % (name, prefix))
        name = name[len(prefix)+1:]
        name = name.replace(os.path.sep, '/')
        return name

    def install(self, install_options, global_options=()):
        if self.editable:
            self.install_editable(install_options, global_options)
            return
        temp_location = tempfile.mkdtemp('-record', 'pip-')
        record_filename = os.path.join(temp_location, 'install-record.txt')
        try:
            install_args = [
                sys.executable, '-c',
                "import setuptools;__file__=%r;"\
                "exec(compile(open(__file__).read().replace('\\r\\n', '\\n'), __file__, 'exec'))" % self.setup_py] +\
                list(global_options) + [
                'install',
                '--single-version-externally-managed',
                '--record', record_filename]

            if running_under_virtualenv():
                ## FIXME: I'm not sure if this is a reasonable location; probably not
                ## but we can't put it in the default location, as that is a virtualenv symlink that isn't writable
                install_args += ['--install-headers',
                                 os.path.join(sys.prefix, 'include', 'site',
                                              'python' + get_python_version())]
            logger.notify('Running setup.py install for %s' % self.name)
            logger.indent += 2
            try:
                call_subprocess(install_args + install_options,
                    cwd=self.source_dir, filter_stdout=self._filter_install, show_stdout=False)
            finally:
                logger.indent -= 2
            if not os.path.exists(record_filename):
                logger.notify('Record file %s not found' % record_filename)
                return
            self.install_succeeded = True
            f = open(record_filename)
            for line in f:
                line = line.strip()
                if line.endswith('.egg-info'):
                    egg_info_dir = line
                    break
            else:
                logger.warn('Could not find .egg-info directory in install record for %s' % self)
                ## FIXME: put the record somewhere
                ## FIXME: should this be an error?
                return
            f.close()
            new_lines = []
            f = open(record_filename)
            for line in f:
                filename = line.strip()
                if os.path.isdir(filename):
                    filename += os.path.sep
                new_lines.append(make_path_relative(filename, egg_info_dir))
            f.close()
            f = open(os.path.join(egg_info_dir, 'installed-files.txt'), 'w')
            f.write('\n'.join(new_lines)+'\n')
            f.close()
        finally:
            if os.path.exists(record_filename):
                os.remove(record_filename)
            os.rmdir(temp_location)

    def remove_temporary_source(self):
        """Remove the source files from this requirement, if they are marked
        for deletion"""
        if self.is_bundle or os.path.exists(self.delete_marker_filename):
            logger.info('Removing source in %s' % self.source_dir)
            if self.source_dir:
                rmtree(self.source_dir)
            self.source_dir = None
        if self._temp_build_dir and os.path.exists(self._temp_build_dir):
            rmtree(self._temp_build_dir)
        self._temp_build_dir = None

    def install_editable(self, install_options, global_options=()):
        logger.notify('Running setup.py develop for %s' % self.name)
        logger.indent += 2
        try:
            ## FIXME: should we do --install-headers here too?
            call_subprocess(
                [sys.executable, '-c',
                 "import setuptools; __file__=%r; exec(compile(open(__file__).read().replace('\\r\\n', '\\n'), __file__, 'exec'))" % self.setup_py]
                + list(global_options) + ['develop', '--no-deps'] + list(install_options),

                cwd=self.source_dir, filter_stdout=self._filter_install,
                show_stdout=False)
        finally:
            logger.indent -= 2
        self.install_succeeded = True

    def _filter_install(self, line):
        level = logger.NOTIFY
        for regex in [r'^running .*', r'^writing .*', '^creating .*', '^[Cc]opying .*',
                      r'^reading .*', r"^removing .*\.egg-info' \(and everything under it\)$",
                      r'^byte-compiling ',
                      # Not sure what this warning is, but it seems harmless:
                      r"^warning: manifest_maker: standard file '-c' not found$"]:
            if re.search(regex, line.strip()):
                level = logger.INFO
                break
        return (level, line)

    def check_if_exists(self):
        """Find an installed distribution that satisfies or conflicts
        with this requirement, and set self.satisfied_by or
        self.conflicts_with appropriately."""
        if self.req is None:
            return False
        try:
            self.satisfied_by = pkg_resources.get_distribution(self.req)
        except pkg_resources.DistributionNotFound:
            return False
        except pkg_resources.VersionConflict:
            self.conflicts_with = pkg_resources.get_distribution(self.req.project_name)
        return True

    @property
    def is_bundle(self):
        if self._is_bundle is not None:
            return self._is_bundle
        base = self._temp_build_dir
        if not base:
            ## FIXME: this doesn't seem right:
            return False
        self._is_bundle = (os.path.exists(os.path.join(base, 'pip-manifest.txt'))
                           or os.path.exists(os.path.join(base, 'pyinstall-manifest.txt')))
        return self._is_bundle

    def bundle_requirements(self):
        for dest_dir in self._bundle_editable_dirs:
            package = os.path.basename(dest_dir)
            ## FIXME: svnism:
            for vcs_backend in vcs.backends:
                url = rev = None
                vcs_bundle_file = os.path.join(
                    dest_dir, vcs_backend.bundle_file)
                if os.path.exists(vcs_bundle_file):
                    vc_type = vcs_backend.name
                    fp = open(vcs_bundle_file)
                    content = fp.read()
                    fp.close()
                    url, rev = vcs_backend().parse_vcs_bundle_file(content)
                    break
            if url:
                url = '%s+%s@%s' % (vc_type, url, rev)
            else:
                url = None
            yield InstallRequirement(
                package, self, editable=True, url=url,
                update=False, source_dir=dest_dir)
        for dest_dir in self._bundle_build_dirs:
            package = os.path.basename(dest_dir)
            yield InstallRequirement(
                package, self,
                source_dir=dest_dir)

    def move_bundle_files(self, dest_build_dir, dest_src_dir):
        base = self._temp_build_dir
        assert base
        src_dir = os.path.join(base, 'src')
        build_dir = os.path.join(base, 'build')
        bundle_build_dirs = []
        bundle_editable_dirs = []
        for source_dir, dest_dir, dir_collection in [
            (src_dir, dest_src_dir, bundle_editable_dirs),
            (build_dir, dest_build_dir, bundle_build_dirs)]:
            if os.path.exists(source_dir):
                for dirname in os.listdir(source_dir):
                    dest = os.path.join(dest_dir, dirname)
                    dir_collection.append(dest)
                    if os.path.exists(dest):
                        logger.warn('The directory %s (containing package %s) already exists; cannot move source from bundle %s'
                                    % (dest, dirname, self))
                        continue
                    if not os.path.exists(dest_dir):
                        logger.info('Creating directory %s' % dest_dir)
                        os.makedirs(dest_dir)
                    shutil.move(os.path.join(source_dir, dirname), dest)
                if not os.listdir(source_dir):
                    os.rmdir(source_dir)
        self._temp_build_dir = None
        self._bundle_build_dirs = bundle_build_dirs
        self._bundle_editable_dirs = bundle_editable_dirs

    @property
    def delete_marker_filename(self):
        assert self.source_dir
        return os.path.join(self.source_dir, PIP_DELETE_MARKER_FILENAME)


DELETE_MARKER_MESSAGE = '''\
This file is placed here by pip to indicate the source was put
here by pip.

Once this package is successfully installed this source code will be
deleted (unless you remove this file).
'''


class Requirements(object):

    def __init__(self):
        self._keys = []
        self._dict = {}

    def keys(self):
        return self._keys

    def values(self):
        values_list = []
        for key in self._keys:
            values_list.append(self._dict[key])
        return values_list

    def __contains__(self, item):
        return item in self._keys

    def __setitem__(self, key, value):
        if key not in self._keys:
            self._keys.append(key)
        self._dict[key] = value

    def __getitem__(self, key):
        return self._dict[key]

    def __repr__(self):
        values = [ '%s: %s' % (repr(k), repr(self[k])) for k in self.keys() ]
        return 'Requirements({%s})' % ', '.join(values)


class RequirementSet(object):

    def __init__(self, build_dir, src_dir, download_dir, download_cache=None,
                 upgrade=False, ignore_installed=False,
                 ignore_dependencies=False):
        self.build_dir = build_dir
        self.src_dir = src_dir
        self.download_dir = download_dir
        self.download_cache = download_cache
        self.upgrade = upgrade
        self.ignore_installed = ignore_installed
        self.requirements = Requirements()
        # Mapping of alias: real_name
        self.requirement_aliases = {}
        self.unnamed_requirements = []
        self.ignore_dependencies = ignore_dependencies
        self.successfully_downloaded = []
        self.successfully_installed = []
        self.reqs_to_cleanup = []

    def __str__(self):
        reqs = [req for req in self.requirements.values()
                if not req.comes_from]
        reqs.sort(key=lambda req: req.name.lower())
        return ' '.join([str(req.req) for req in reqs])

    def add_requirement(self, install_req):
        name = install_req.name
        if not name:
            self.unnamed_requirements.append(install_req)
        else:
            if self.has_requirement(name):
                raise InstallationError(
                    'Double requirement given: %s (aready in %s, name=%r)'
                    % (install_req, self.get_requirement(name), name))
            self.requirements[name] = install_req
            ## FIXME: what about other normalizations?  E.g., _ vs. -?
            if name.lower() != name:
                self.requirement_aliases[name.lower()] = name

    def has_requirement(self, project_name):
        for name in project_name, project_name.lower():
            if name in self.requirements or name in self.requirement_aliases:
                return True
        return False

    @property
    def has_requirements(self):
        return list(self.requirements.values()) or self.unnamed_requirements

    @property
    def has_editables(self):
        if any(req.editable for req in self.requirements.values()):
            return True
        if any(req.editable for req in self.unnamed_requirements):
            return True
        return False

    @property
    def is_download(self):
        if self.download_dir:
            self.download_dir = os.path.expanduser(self.download_dir)
            if os.path.exists(self.download_dir):
                return True
            else:
                logger.fatal('Could not find download directory')
                raise InstallationError(
                    "Could not find or access download directory '%s'"
                    % display_path(self.download_dir))
        return False

    def get_requirement(self, project_name):
        for name in project_name, project_name.lower():
            if name in self.requirements:
                return self.requirements[name]
            if name in self.requirement_aliases:
                return self.requirements[self.requirement_aliases[name]]
        raise KeyError("No project with the name %r" % project_name)

    def uninstall(self, auto_confirm=False):
        for req in self.requirements.values():
            req.uninstall(auto_confirm=auto_confirm)
            req.commit_uninstall()

    def locate_files(self):
        ## FIXME: duplicates code from install_files; relevant code should
        ##        probably be factored out into a separate method
        unnamed = list(self.unnamed_requirements)
        reqs = list(self.requirements.values())
        while reqs or unnamed:
            if unnamed:
                req_to_install = unnamed.pop(0)
            else:
                req_to_install = reqs.pop(0)
            install_needed = True
            if not self.ignore_installed and not req_to_install.editable:
                req_to_install.check_if_exists()
                if req_to_install.satisfied_by:
                    if self.upgrade:
                        req_to_install.conflicts_with = req_to_install.satisfied_by
                        req_to_install.satisfied_by = None
                    else:
                        install_needed = False
                if req_to_install.satisfied_by:
                    logger.notify('Requirement already satisfied '
                                  '(use --upgrade to upgrade): %s'
                                  % req_to_install)

            if req_to_install.editable:
                if req_to_install.source_dir is None:
                    req_to_install.source_dir = req_to_install.build_location(self.src_dir)
            elif install_needed:
                req_to_install.source_dir = req_to_install.build_location(self.build_dir, not self.is_download)

            if req_to_install.source_dir is not None and not os.path.isdir(req_to_install.source_dir):
                raise InstallationError('Could not install requirement %s '
                                       'because source folder %s does not exist '
                                       '(perhaps --no-download was used without first running '
                                       'an equivalent install with --no-install?)'
                                       % (req_to_install, req_to_install.source_dir))

    def prepare_files(self, finder, force_root_egg_info=False, bundle=False):
        """Prepare process. Create temp directories, download and/or unpack files."""
        unnamed = list(self.unnamed_requirements)
        reqs = list(self.requirements.values())
        while reqs or unnamed:
            if unnamed:
                req_to_install = unnamed.pop(0)
            else:
                req_to_install = reqs.pop(0)
            install = True
            if not self.ignore_installed and not req_to_install.editable:
                req_to_install.check_if_exists()
                if req_to_install.satisfied_by:
                    if self.upgrade:
                        req_to_install.conflicts_with = req_to_install.satisfied_by
                        req_to_install.satisfied_by = None
                    else:
                        install = False
                if req_to_install.satisfied_by:
                    logger.notify('Requirement already satisfied '
                                  '(use --upgrade to upgrade): %s'
                                  % req_to_install)
            if req_to_install.editable:
                logger.notify('Obtaining %s' % req_to_install)
            elif install:
                if req_to_install.url and req_to_install.url.lower().startswith('file:'):
                    logger.notify('Unpacking %s' % display_path(url_to_path(req_to_install.url)))
                else:
                    logger.notify('Downloading/unpacking %s' % req_to_install)
            logger.indent += 2
            try:
                is_bundle = False
                if req_to_install.editable:
                    if req_to_install.source_dir is None:
                        location = req_to_install.build_location(self.src_dir)
                        req_to_install.source_dir = location
                    else:
                        location = req_to_install.source_dir
                    if not os.path.exists(self.build_dir):
                        _make_build_dir(self.build_dir)
                    req_to_install.update_editable(not self.is_download)
                    if self.is_download:
                        req_to_install.run_egg_info()
                        req_to_install.archive(self.download_dir)
                    else:
                        req_to_install.run_egg_info()
                elif install:
                    ##@@ if filesystem packages are not marked
                    ##editable in a req, a non deterministic error
                    ##occurs when the script attempts to unpack the
                    ##build directory

                    location = req_to_install.build_location(self.build_dir, not self.is_download)
                    ## FIXME: is the existance of the checkout good enough to use it?  I don't think so.
                    unpack = True
                    if not os.path.exists(os.path.join(location, 'setup.py')):
                        ## FIXME: this won't upgrade when there's an existing package unpacked in `location`
                        if req_to_install.url is None:
                            url = finder.find_requirement(req_to_install, upgrade=self.upgrade)
                        else:
                            ## FIXME: should req_to_install.url already be a link?
                            url = Link(req_to_install.url)
                            assert url
                        if url:
                            try:
                                self.unpack_url(url, location, self.is_download)
                            except HTTPError:
                                e = sys.exc_info()[1]
                                logger.fatal('Could not install requirement %s because of error %s'
                                             % (req_to_install, e))
                                raise InstallationError(
                                    'Could not install requirement %s because of HTTP error %s for URL %s'
                                    % (req_to_install, e, url))
                        else:
                            unpack = False
                    if unpack:
                        is_bundle = req_to_install.is_bundle
                        url = None
                        if is_bundle:
                            req_to_install.move_bundle_files(self.build_dir, self.src_dir)
                            for subreq in req_to_install.bundle_requirements():
                                reqs.append(subreq)
                                self.add_requirement(subreq)
                        elif self.is_download:
                            req_to_install.source_dir = location
                            if url and url.scheme in vcs.all_schemes:
                                req_to_install.run_egg_info()
                                req_to_install.archive(self.download_dir)
                        else:
                            req_to_install.source_dir = location
                            req_to_install.run_egg_info()
                            if force_root_egg_info:
                                # We need to run this to make sure that the .egg-info/
                                # directory is created for packing in the bundle
                                req_to_install.run_egg_info(force_root_egg_info=True)
                            req_to_install.assert_source_matches_version()
                            #@@ sketchy way of identifying packages not grabbed from an index
                            if bundle and req_to_install.url:
                                self.copy_to_build_dir(req_to_install)
                                install = False
                        # req_to_install.req is only avail after unpack for URL pkgs
                        # repeat check_if_exists to uninstall-on-upgrade (#14)
                        req_to_install.check_if_exists()
                        if req_to_install.satisfied_by:
                            if self.upgrade or self.ignore_installed:
                                req_to_install.conflicts_with = req_to_install.satisfied_by
                                req_to_install.satisfied_by = None
                            else:
                                install = False
                if not is_bundle and not self.is_download:
                    ## FIXME: shouldn't be globally added:
                    finder.add_dependency_links(req_to_install.dependency_links)
                    ## FIXME: add extras in here:
                    if not self.ignore_dependencies:
                        for req in req_to_install.requirements():
                            try:
                                name = pkg_resources.Requirement.parse(req).project_name
                            except ValueError:
                                e = sys.exc_info()[1]
                                ## FIXME: proper warning
                                logger.error('Invalid requirement: %r (%s) in requirement %s' % (req, e, req_to_install))
                                continue
                            if self.has_requirement(name):
                                ## FIXME: check for conflict
                                continue
                            subreq = InstallRequirement(req, req_to_install)
                            reqs.append(subreq)
                            self.add_requirement(subreq)
                    if req_to_install.name not in self.requirements:
                        self.requirements[req_to_install.name] = req_to_install
                else:
                    self.reqs_to_cleanup.append(req_to_install)
                if install:
                    self.successfully_downloaded.append(req_to_install)
                    if bundle and (req_to_install.url and req_to_install.url.startswith('file:///')):
                        self.copy_to_build_dir(req_to_install)
            finally:
                logger.indent -= 2

    def cleanup_files(self, bundle=False):
        """Clean up files, remove builds."""
        logger.notify('Cleaning up...')
        logger.indent += 2
        for req in self.reqs_to_cleanup:
            req.remove_temporary_source()

        remove_dir = []
        if self._pip_has_created_build_dir():
            remove_dir.append(self.build_dir)

        # The source dir of a bundle can always be removed.
        if bundle:
            remove_dir.append(self.src_dir)

        for dir in remove_dir:
            if os.path.exists(dir):
                logger.info('Removing temporary dir %s...' % dir)
                rmtree(dir)

        logger.indent -= 2

    def _pip_has_created_build_dir(self):
        return (self.build_dir == build_prefix and
                os.path.exists(os.path.join(self.build_dir, PIP_DELETE_MARKER_FILENAME)))

    def copy_to_build_dir(self, req_to_install):
        target_dir = req_to_install.editable and self.src_dir or self.build_dir
        logger.info("Copying %s to %s" % (req_to_install.name, target_dir))
        dest = os.path.join(target_dir, req_to_install.name)
        copytree(req_to_install.source_dir, dest)
        call_subprocess(["python", "%s/setup.py" % dest, "clean"], cwd=dest,
                        command_desc='python setup.py clean')

    def unpack_url(self, link, location, only_download=False):
        if only_download:
            location = self.download_dir
        if is_vcs_url(link):
            return unpack_vcs_link(link, location, only_download)
        elif is_file_url(link):
            return unpack_file_url(link, location)
        else:
            if self.download_cache:
                self.download_cache = os.path.expanduser(self.download_cache)
            return unpack_http_url(link, location, self.download_cache, only_download)

    def install(self, install_options, global_options=()):
        """Install everything in this set (after having downloaded and unpacked the packages)"""
        to_install = [r for r in self.requirements.values()
                      if self.upgrade or not r.satisfied_by]

        if to_install:
            logger.notify('Installing collected packages: %s' % ', '.join([req.name for req in to_install]))
        logger.indent += 2
        try:
            for requirement in to_install:
                if requirement.conflicts_with:
                    logger.notify('Found existing installation: %s'
                                  % requirement.conflicts_with)
                    logger.indent += 2
                    try:
                        requirement.uninstall(auto_confirm=True)
                    finally:
                        logger.indent -= 2
                try:
                    requirement.install(install_options, global_options)
                except:
                    # if install did not succeed, rollback previous uninstall
                    if requirement.conflicts_with and not requirement.install_succeeded:
                        requirement.rollback_uninstall()
                    raise
                else:
                    if requirement.conflicts_with and requirement.install_succeeded:
                        requirement.commit_uninstall()
                requirement.remove_temporary_source()
        finally:
            logger.indent -= 2
        self.successfully_installed = to_install

    def create_bundle(self, bundle_filename):
        ## FIXME: can't decide which is better; zip is easier to read
        ## random files from, but tar.bz2 is smaller and not as lame a
        ## format.

        ## FIXME: this file should really include a manifest of the
        ## packages, maybe some other metadata files.  It would make
        ## it easier to detect as well.
        zip = zipfile.ZipFile(bundle_filename, 'w', zipfile.ZIP_DEFLATED)
        vcs_dirs = []
        for dir, basename in (self.build_dir, 'build'), (self.src_dir, 'src'):
            dir = os.path.normcase(os.path.abspath(dir))
            for dirpath, dirnames, filenames in os.walk(dir):
                for backend in vcs.backends:
                    vcs_backend = backend()
                    vcs_url = vcs_rev = None
                    if vcs_backend.dirname in dirnames:
                        for vcs_dir in vcs_dirs:
                            if dirpath.startswith(vcs_dir):
                                # vcs bundle file already in parent directory
                                break
                        else:
                            vcs_url, vcs_rev = vcs_backend.get_info(
                                os.path.join(dir, dirpath))
                            vcs_dirs.append(dirpath)
                        vcs_bundle_file = vcs_backend.bundle_file
                        vcs_guide = vcs_backend.guide % {'url': vcs_url,
                                                         'rev': vcs_rev}
                        dirnames.remove(vcs_backend.dirname)
                        break
                if 'pip-egg-info' in dirnames:
                    dirnames.remove('pip-egg-info')
                for dirname in dirnames:
                    dirname = os.path.join(dirpath, dirname)
                    name = self._clean_zip_name(dirname, dir)
                    zip.writestr(basename + '/' + name + '/', '')
                for filename in filenames:
                    if filename == PIP_DELETE_MARKER_FILENAME:
                        continue
                    filename = os.path.join(dirpath, filename)
                    name = self._clean_zip_name(filename, dir)
                    zip.write(filename, basename + '/' + name)
                if vcs_url:
                    name = os.path.join(dirpath, vcs_bundle_file)
                    name = self._clean_zip_name(name, dir)
                    zip.writestr(basename + '/' + name, vcs_guide)

        zip.writestr('pip-manifest.txt', self.bundle_requirements())
        zip.close()

    BUNDLE_HEADER = '''\
# This is a pip bundle file, that contains many source packages
# that can be installed as a group.  You can install this like:
#     pip this_file.zip
# The rest of the file contains a list of all the packages included:
'''

    def bundle_requirements(self):
        parts = [self.BUNDLE_HEADER]
        for req in [req for req in self.requirements.values()
                    if not req.comes_from]:
            parts.append('%s==%s\n' % (req.name, req.installed_version))
        parts.append('# These packages were installed to satisfy the above requirements:\n')
        for req in [req for req in self.requirements.values()
                    if req.comes_from]:
            parts.append('%s==%s\n' % (req.name, req.installed_version))
        ## FIXME: should we do something with self.unnamed_requirements?
        return ''.join(parts)

    def _clean_zip_name(self, name, prefix):
        assert name.startswith(prefix+os.path.sep), (
            "name %r doesn't start with prefix %r" % (name, prefix))
        name = name[len(prefix)+1:]
        name = name.replace(os.path.sep, '/')
        return name


def _make_build_dir(build_dir):
    os.makedirs(build_dir)
    _write_delete_marker_message(os.path.join(build_dir, PIP_DELETE_MARKER_FILENAME))


def _write_delete_marker_message(filepath):
    marker_fp = open(filepath, 'w')
    marker_fp.write(DELETE_MARKER_MESSAGE)
    marker_fp.close()


_scheme_re = re.compile(r'^(http|https|file):', re.I)


def parse_requirements(filename, finder=None, comes_from=None, options=None):
    skip_match = None
    skip_regex = options.skip_requirements_regex
    if skip_regex:
        skip_match = re.compile(skip_regex)
    filename, content = get_file_content(filename, comes_from=comes_from)
    for line_number, line in enumerate(content.splitlines()):
        line_number += 1
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if skip_match and skip_match.search(line):
            continue
        if line.startswith('-r') or line.startswith('--requirement'):
            if line.startswith('-r'):
                req_url = line[2:].strip()
            else:
                req_url = line[len('--requirement'):].strip().strip('=')
            if _scheme_re.search(filename):
                # Relative to a URL
                req_url = urlparse.urljoin(req_url, filename)
            elif not _scheme_re.search(req_url):
                req_url = os.path.join(os.path.dirname(filename), req_url)
            for item in parse_requirements(req_url, finder, comes_from=filename, options=options):
                yield item
        elif line.startswith('-Z') or line.startswith('--always-unzip'):
            # No longer used, but previously these were used in
            # requirement files, so we'll ignore.
            pass
        elif line.startswith('-f') or line.startswith('--find-links'):
            if line.startswith('-f'):
                line = line[2:].strip()
            else:
                line = line[len('--find-links'):].strip().lstrip('=')
            ## FIXME: it would be nice to keep track of the source of
            ## the find_links:
            if finder:
                finder.find_links.append(line)
        elif line.startswith('-i') or line.startswith('--index-url'):
            if line.startswith('-i'):
                line = line[2:].strip()
            else:
                line = line[len('--index-url'):].strip().lstrip('=')
            if finder:
                finder.index_urls = [line]
        elif line.startswith('--extra-index-url'):
            line = line[len('--extra-index-url'):].strip().lstrip('=')
            if finder:
                finder.index_urls.append(line)
        else:
            comes_from = '-r %s (line %s)' % (filename, line_number)
            if line.startswith('-e') or line.startswith('--editable'):
                if line.startswith('-e'):
                    line = line[2:].strip()
                else:
                    line = line[len('--editable'):].strip().lstrip('=')
                req = InstallRequirement.from_editable(
                    line, comes_from=comes_from, default_vcs=options.default_vcs)
            else:
                req = InstallRequirement.from_line(line, comes_from)
            yield req


def parse_editable(editable_req, default_vcs=None):
    """Parses svn+http://blahblah@rev#egg=Foobar into a requirement
    (Foobar) and a URL"""
    url = editable_req
    if os.path.isdir(url) and os.path.exists(os.path.join(url, 'setup.py')):
        # Treating it as code that has already been checked out
        url = path_to_url(url)
    if url.lower().startswith('file:'):
        return None, url
    for version_control in vcs:
        if url.lower().startswith('%s:' % version_control):
            url = '%s+%s' % (version_control, url)
    if '+' not in url:
        if default_vcs:
            url = default_vcs + '+' + url
        else:
            raise InstallationError(
                '--editable=%s should be formatted with svn+URL, git+URL, hg+URL or bzr+URL' % editable_req)
    vc_type = url.split('+', 1)[0].lower()
    if not vcs.get_backend(vc_type):
        raise InstallationError(
            'For --editable=%s only svn (svn+URL), Git (git+URL), Mercurial (hg+URL) and Bazaar (bzr+URL) is currently supported' % editable_req)
    match = re.search(r'(?:#|#.*?&)egg=([^&]*)', editable_req)
    if (not match or not match.group(1)) and vcs.get_backend(vc_type):
        parts = [p for p in editable_req.split('#', 1)[0].split('/') if p]
        if parts[-2] in ('tags', 'branches', 'tag', 'branch'):
            req = parts[-3]
        elif parts[-1] == 'trunk':
            req = parts[-2]
        else:
            raise InstallationError(
                '--editable=%s is not the right format; it must have #egg=Package'
                % editable_req)
    else:
        req = match.group(1)
    ## FIXME: use package_to_requirement?
    match = re.search(r'^(.*?)(?:-dev|-\d.*)$', req)
    if match:
        # Strip off -dev, -0.2, etc.
        req = match.group(1)
    return req, url


class UninstallPathSet(object):
    """A set of file paths to be removed in the uninstallation of a
    requirement."""
    def __init__(self, dist):
        self.paths = set()
        self._refuse = set()
        self.pth = {}
        self.dist = dist
        self.save_dir = None
        self._moved_paths = []

    def _permitted(self, path):
        """
        Return True if the given path is one we are permitted to
        remove/modify, False otherwise.

        """
        return is_local(path)

    def _can_uninstall(self):
        if not dist_is_local(self.dist):
            logger.notify("Not uninstalling %s at %s, outside environment %s"
                          % (self.dist.project_name, normalize_path(self.dist.location), sys.prefix))
            return False
        return True

    def add(self, path):
        path = normalize_path(path)
        if not os.path.exists(path):
            return
        if self._permitted(path):
            self.paths.add(path)
        else:
            self._refuse.add(path)

    def add_pth(self, pth_file, entry):
        pth_file = normalize_path(pth_file)
        if self._permitted(pth_file):
            if pth_file not in self.pth:
                self.pth[pth_file] = UninstallPthEntries(pth_file)
            self.pth[pth_file].add(entry)
        else:
            self._refuse.add(pth_file)

    def compact(self, paths):
        """Compact a path set to contain the minimal number of paths
        necessary to contain all paths in the set. If /a/path/ and
        /a/path/to/a/file.txt are both in the set, leave only the
        shorter path."""
        short_paths = set()
        for path in sorted(paths, key=len):
            if not any([(path.startswith(shortpath) and
                         path[len(shortpath.rstrip(os.path.sep))] == os.path.sep)
                        for shortpath in short_paths]):
                short_paths.add(path)
        return short_paths

    def _stash(self, path):
        return os.path.join(
            self.save_dir, os.path.splitdrive(path)[1].lstrip(os.path.sep))

    def remove(self, auto_confirm=False):
        """Remove paths in ``self.paths`` with confirmation (unless
        ``auto_confirm`` is True)."""
        if not self._can_uninstall():
            return
        logger.notify('Uninstalling %s:' % self.dist.project_name)
        logger.indent += 2
        paths = sorted(self.compact(self.paths))
        try:
            if auto_confirm:
                response = 'y'
            else:
                for path in paths:
                    logger.notify(path)
                response = ask('Proceed (y/n)? ', ('y', 'n'))
            if self._refuse:
                logger.notify('Not removing or modifying (outside of prefix):')
                for path in self.compact(self._refuse):
                    logger.notify(path)
            if response == 'y':
                self.save_dir = tempfile.mkdtemp(suffix='-uninstall',
                                                 prefix='pip-')
                for path in paths:
                    new_path = self._stash(path)
                    logger.info('Removing file or directory %s' % path)
                    self._moved_paths.append(path)
                    renames(path, new_path)
                for pth in self.pth.values():
                    pth.remove()
                logger.notify('Successfully uninstalled %s' % self.dist.project_name)

        finally:
            logger.indent -= 2

    def rollback(self):
        """Rollback the changes previously made by remove()."""
        if self.save_dir is None:
            logger.error("Can't roll back %s; was not uninstalled" % self.dist.project_name)
            return False
        logger.notify('Rolling back uninstall of %s' % self.dist.project_name)
        for path in self._moved_paths:
            tmp_path = self._stash(path)
            logger.info('Replacing %s' % path)
            renames(tmp_path, path)
        for pth in self.pth:
            pth.rollback()

    def commit(self):
        """Remove temporary save dir: rollback will no longer be possible."""
        if self.save_dir is not None:
            rmtree(self.save_dir)
            self.save_dir = None
            self._moved_paths = []


class UninstallPthEntries(object):
    def __init__(self, pth_file):
        if not os.path.isfile(pth_file):
            raise UninstallationError("Cannot remove entries from nonexistent file %s" % pth_file)
        self.file = pth_file
        self.entries = set()
        self._saved_lines = None

    def add(self, entry):
        entry = os.path.normcase(entry)
        # On Windows, os.path.normcase converts the entry to use
        # backslashes.  This is correct for entries that describe absolute
        # paths outside of site-packages, but all the others use forward
        # slashes.
        if sys.platform == 'win32' and not os.path.splitdrive(entry)[0]:
            entry = entry.replace('\\', '/')
        self.entries.add(entry)

    def remove(self):
        logger.info('Removing pth entries from %s:' % self.file)
        fh = open(self.file, 'rb')
        # windows uses '\r\n' with py3k, but uses '\n' with py2.x
        lines = fh.readlines()
        self._saved_lines = lines
        fh.close()
        if any(b('\r\n') in line for line in lines):
            endline = '\r\n'
        else:
            endline = '\n'
        for entry in self.entries:
            try:
                logger.info('Removing entry: %s' % entry)
                lines.remove(b(entry + endline))
            except ValueError:
                pass
        fh = open(self.file, 'wb')
        fh.writelines(lines)
        fh.close()

    def rollback(self):
        if self._saved_lines is None:
            logger.error('Cannot roll back changes to %s, none were made' % self.file)
            return False
        logger.info('Rolling %s back to previous state' % self.file)
        fh = open(self.file, 'wb')
        fh.writelines(self._saved_lines)
        fh.close()
        return True


class FakeFile(object):
    """Wrap a list of lines in an object with readline() to make
    ConfigParser happy."""
    def __init__(self, lines):
        self._gen = (l for l in lines)

    def readline(self):
        try:
            try:
                return next(self._gen)
            except NameError:
                return self._gen.next()
        except StopIteration:
            return ''

    def __iter__(self):
        return self._gen
