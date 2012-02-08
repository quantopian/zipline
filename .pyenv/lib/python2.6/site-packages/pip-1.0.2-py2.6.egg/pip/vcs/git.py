import tempfile
import re
from pip import call_subprocess
from pip.util import display_path, rmtree
from pip.vcs import vcs, VersionControl
from pip.log import logger
from pip.backwardcompat import url2pathname, urlparse
urlsplit = urlparse.urlsplit
urlunsplit = urlparse.urlunsplit

class Git(VersionControl):
    name = 'git'
    dirname = '.git'
    repo_name = 'clone'
    schemes = ('git', 'git+http', 'git+https', 'git+ssh', 'git+git', 'git+file')
    bundle_file = 'git-clone.txt'
    guide = ('# This was a Git repo; to make it a repo again run:\n'
        'git init\ngit remote add origin %(url)s -f\ngit checkout %(rev)s\n')

    def __init__(self, url=None, *args, **kwargs):

        # Works around an apparent Git bug
        # (see http://article.gmane.org/gmane.comp.version-control.git/146500)
        if url:
            scheme, netloc, path, query, fragment = urlsplit(url)
            if scheme.endswith('file'):
                initial_slashes = path[:-len(path.lstrip('/'))]
                newpath = initial_slashes + url2pathname(path).replace('\\', '/').lstrip('/')
                url = urlunsplit((scheme, netloc, newpath, query, fragment))
                after_plus = scheme.find('+')+1
                url = scheme[:after_plus]+ urlunsplit((scheme[after_plus:], netloc, newpath, query, fragment))

        super(Git, self).__init__(url, *args, **kwargs)

    def parse_vcs_bundle_file(self, content):
        url = rev = None
        for line in content.splitlines():
            if not line.strip() or line.strip().startswith('#'):
                continue
            url_match = re.search(r'git\s*remote\s*add\s*origin(.*)\s*-f', line)
            if url_match:
                url = url_match.group(1).strip()
            rev_match = re.search(r'^git\s*checkout\s*-q\s*(.*)\s*', line)
            if rev_match:
                rev = rev_match.group(1).strip()
            if url and rev:
                return url, rev
        return None, None

    def export(self, location):
        """Export the Git repository at the url to the destination location"""
        temp_dir = tempfile.mkdtemp('-export', 'pip-')
        self.unpack(temp_dir)
        try:
            if not location.endswith('/'):
                location = location + '/'
            call_subprocess(
                [self.cmd, 'checkout-index', '-a', '-f', '--prefix', location],
                filter_stdout=self._filter, show_stdout=False, cwd=temp_dir)
        finally:
            rmtree(temp_dir)

    def check_rev_options(self, rev, dest, rev_options):
        """Check the revision options before checkout to compensate that tags
        and branches may need origin/ as a prefix.
        Returns the SHA1 of the branch or tag if found.
        """
        revisions = self.get_tag_revs(dest)
        revisions.update(self.get_branch_revs(dest))

        origin_rev = 'origin/%s' % rev
        if origin_rev in revisions:
            # remote branch
            return [revisions[origin_rev]]
        elif rev in revisions:
            # a local tag or branch name
            return [revisions[rev]]
        else:
            logger.warn("Could not find a tag or branch '%s', assuming commit." % rev)
            return rev_options

    def switch(self, dest, url, rev_options):
        call_subprocess(
            [self.cmd, 'config', 'remote.origin.url', url], cwd=dest)
        call_subprocess(
            [self.cmd, 'checkout', '-q'] + rev_options, cwd=dest)

    def update(self, dest, rev_options):
        # First fetch changes from the default remote
        call_subprocess([self.cmd, 'fetch', '-q'], cwd=dest)
        # Then reset to wanted revision (maby even origin/master)
        if rev_options:
            rev_options = self.check_rev_options(rev_options[0], dest, rev_options)
        call_subprocess([self.cmd, 'reset', '--hard', '-q'] + rev_options, cwd=dest)

    def obtain(self, dest):
        url, rev = self.get_url_rev()
        if rev:
            rev_options = [rev]
            rev_display = ' (to %s)' % rev
        else:
            rev_options = ['origin/master']
            rev_display = ''
        if self.check_destination(dest, url, rev_options, rev_display):
            logger.notify('Cloning %s%s to %s' % (url, rev_display, display_path(dest)))
            call_subprocess([self.cmd, 'clone', '-q', url, dest])
            if rev:
                rev_options = self.check_rev_options(rev, dest, rev_options)
                # Only do a checkout if rev_options differs from HEAD
                if not self.get_revision(dest).startswith(rev_options[0]):
                    call_subprocess([self.cmd, 'checkout', '-q'] + rev_options, cwd=dest)

    def get_url(self, location):
        url = call_subprocess(
            [self.cmd, 'config', 'remote.origin.url'],
            show_stdout=False, cwd=location)
        return url.strip()

    def get_revision(self, location):
        current_rev = call_subprocess(
            [self.cmd, 'rev-parse', 'HEAD'], show_stdout=False, cwd=location)
        return current_rev.strip()

    def get_tag_revs(self, location):
        tags = self._get_all_tag_names(location)
        tag_revs = {}
        for line in tags.splitlines():
            tag = line.strip()
            rev = self._get_revision_from_rev_parse(tag, location)
            tag_revs[tag] = rev.strip()
        return tag_revs

    def get_branch_revs(self, location):
        branches = self._get_all_branch_names(location)
        branch_revs = {}
        for line in branches.splitlines():
            if '(no branch)' in line:
                continue
            line = line.split('->')[0].strip()
            # actual branch case
            branch = "".join(b for b in line.split() if b != '*')
            rev = self._get_revision_from_rev_parse(branch, location)
            branch_revs[branch] = rev.strip()
        return branch_revs

    def get_src_requirement(self, dist, location, find_tags):
        repo = self.get_url(location)
        if not repo.lower().startswith('git:'):
            repo = 'git+' + repo
        egg_project_name = dist.egg_name().split('-', 1)[0]
        if not repo:
            return None
        current_rev = self.get_revision(location)
        tag_revs = self.get_tag_revs(location)
        branch_revs = self.get_branch_revs(location)

        if current_rev in tag_revs:
            # It's a tag
            full_egg_name = '%s-%s' % (egg_project_name, tag_revs[current_rev])
        elif (current_rev in branch_revs and
              branch_revs[current_rev] != 'origin/master'):
            # It's the head of a branch
            full_egg_name = '%s-%s' % (egg_project_name,
                                       branch_revs[current_rev].replace('origin/', ''))
        else:
            full_egg_name = '%s-dev' % egg_project_name

        return '%s@%s#egg=%s' % (repo, current_rev, full_egg_name)

    def get_url_rev(self):
        """
        Prefixes stub URLs like 'user@hostname:user/repo.git' with 'ssh://'.
        That's required because although they use SSH they sometimes doesn't
        work with a ssh:// scheme (e.g. Github). But we need a scheme for
        parsing. Hence we remove it again afterwards and return it as a stub.
        """
        if not '://' in self.url:
            assert not 'file:' in self.url
            self.url = self.url.replace('git+', 'git+ssh://')
            url, rev = super(Git, self).get_url_rev()
            url = url.replace('ssh://', '')
        else:
            url, rev = super(Git, self).get_url_rev()

        return url, rev

    def _get_all_tag_names(self, location):
        return call_subprocess([self.cmd, 'tag', '-l'],
                               show_stdout=False,
                               raise_on_returncode=False,
                               cwd=location)

    def _get_all_branch_names(self, location):
        remote_branches = call_subprocess([self.cmd, 'branch', '-r'],
                                          show_stdout=False, cwd=location)
        local_branches = call_subprocess([self.cmd, 'branch', '-l'],
                                         show_stdout=False, cwd=location)
        return remote_branches + local_branches

    def _get_revision_from_rev_parse(self, name, location):
        return call_subprocess([self.cmd, 'rev-parse', name],
                               show_stdout=False, cwd=location)


vcs.register(Git)
