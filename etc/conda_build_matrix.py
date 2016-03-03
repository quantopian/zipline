from itertools import product
import os
import subprocess

import click

py_versions = ('2.7', '3.4')
npy_versions = ('1.9', '1.10')
zipline_path = os.path.join(
    os.path.dirname(__file__),
    '..',
    'conda',
    'zipline',
)


def mkargs(py_version, npy_version, output=False):
    return {
        'args': [
            'conda',
            'build',
            zipline_path,
            '-c', 'quantopian',
            '--python=%s' % py_version,
            '--numpy=%s' % npy_version,
        ] + (['--output'] if output else []),
        'stdout': subprocess.PIPE,
        'stderr': subprocess.PIPE,
    }


@click.command()
@click.option(
    '--upload',
    is_flag=True,
    default=False,
    help='Upload packages after building',
)
@click.option(
    '--upload-only',
    is_flag=True,
    default=False,
    help='Upload the last built packages without rebuilding.',
)
@click.option(
    '--user',
    default='quantopian',
    help='The anaconda account to upload to.',
)
def main(upload, upload_only, user):
    if upload_only:
        # if you are only uploading you shouldn't need to specify both flags
        upload = True
    procs = (
        (
            py_version,
            npy_version,
            (subprocess.Popen(**mkargs(py_version, npy_version))
             if not upload_only else
             None),
        )
        for py_version, npy_version in product(py_versions, npy_versions)
    )
    status = 0
    files = []
    for py_version, npy_version, proc in procs:
        if not upload_only:
            out, err = proc.communicate()
            if proc.returncode:
                status = 1
                print('build failure: python=%s numpy=%s\n%s' % (
                    py_version,
                    npy_version,
                    err.decode('utf-8'),
                ))
        elif upload:
            files.append(subprocess.Popen(
                **mkargs(py_version, npy_version, output=True)
            ).communicate()[0].decode('utf-8').strip())

    if not status and upload:
        for f in files:
            p = subprocess.Popen(
                ['anaconda', 'upload', '-u', user, f],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            out, err = p.communicate()
            if p.returncode:
                print('failed to upload: %s\n%s' % (f, err.decode('utf-8')))
    return status


if __name__ == '__main__':
    exit(main())
