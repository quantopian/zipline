import os
import re
import subprocess


def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]


def iter_stdout(cmd):
    p = subprocess.Popen(cmd,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT)

    try:
        for line in iter(p.stdout.readline, b''):
            yield line.decode().rstrip()
    finally:
        retcode = p.wait()
        if retcode:
            raise subprocess.CalledProcessError(retcode, cmd[0])


PKG_PATH_PATTERN = re.compile(".* anaconda upload (?P<pkg_path>.+)$")


def main(env, do_upload):
    for recipe in get_immediate_subdirectories('conda'):
        cmd = ["conda", "build", os.path.join('conda', recipe),
               "--python", env['CONDA_PY'],
               "--numpy", env['CONDA_NPY'],
               "--skip-existing",
               "-c", "quantopian",
               "-c", "https://conda.anaconda.org/quantopian/label/ci"]

        output = None

        for line in iter_stdout(cmd):
            print(line)

            if not output:
                match = PKG_PATH_PATTERN.match(line)
                if match:
                    output = match.group('pkg_path')

        if output and os.path.exists(output) and do_upload:
            cmd = ["anaconda", "-t", env['ANACONDA_TOKEN'],
                   "upload", output, "-u", "quantopian", "--label", "ci"]

            for line in iter_stdout(cmd):
                print(line)


if __name__ == '__main__':
    env = os.environ.copy()
    main(env,
         do_upload=((env.get('ANACONDA_TOKEN')
                     and env.get('APPVEYOR_REPO_BRANCH') == 'master')
                    and 'APPVEYOR_PULL_REQUEST_NUMBER' not in env))
