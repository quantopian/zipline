conda recipes
=================

[conda](https://conda.io/docs/user-guide/overview.html) is a
Python package management system by Anaconda that provides
easy installation of binary packages.

The files in this directory provide instructions for how
to create these binary packages. After installing conda and
conda-build you should be able to:

```bash
conda build ta-lib
conda build logbook
conda build zipline
```

You can then upload these binary packages to your own
channel at [anaconda.org](https://anaconda.org).

You can add new recipes for packages that exist on PyPI with
[conda skeleton](https://conda.io/docs/user-guide/tutorials/build-pkgs-skeleton.html#building-a-simple-package-with-conda-skeleton-pypi):

```bash
conda skeleton pypi <package_name> --version <version>
```

From the zipline root directory, I might add a recipe for `requests==2.20.1` with:

```bash
$ conda skeleton pypi requests --version 2.20.1 --output-dir ./conda
```

Windows
-------

Building ta-lib on Windows requires Visual Studio (Express).
