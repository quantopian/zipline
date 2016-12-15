conda build files
=================

[conda](http://docs.continuum.io/conda/intro.html) is a
Python package management system by Continuum that provides
easy installation of binary packages.

The files in this directory provide instructions for how
to create these binary packages. After installing conda and
conda-build you should be able to:

```
conda build ta-lib
conda build logbook
conda build cyordereddict
conda build zipline
```

You can then upload these binary packages to your own
channel at [binstar](https://binstar.org).

Windows
-------

Building ta-lib on Windows requires Visual Studio (Express) and
the [compiled ta-lib](ta-lib-0.4.0-msvc.zip) which you have to
unzip to C:\ta-lib.
