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


def main():
    for recipe in get_immediate_subdirectories('conda'):
        cmd = ["conda", "build", os.path.join('conda', recipe),
               "--python", os.environ['CONDA_PY'],
               "--numpy", os.environ['CONDA_NPY'],
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

        if (output and os.path.exists(output) and
                os.environ.get('ANACONDA_TOKEN')):

            cmd = ["anaconda", "-t", os.environ['ANACONDA_TOKEN'],
                   "upload", output, "-u", "quantopian", "--label", "ci"]

            for line in iter_stdout(cmd):
                print(line)


if __name__ == '__main__':
    main()
