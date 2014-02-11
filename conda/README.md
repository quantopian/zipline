conda build files
=================

[conda](http://docs.continuum.io/conda/intro.html) is a 
Python package management system by Continuum that provides
easy installation of binary packages.

These files provide instructions for how to create these
binary packages. After install conda and conda-build you
should be able to:

```
conda build ta-lib
conda build logbook
conda build zipline
```

You can the update these binary packages to your own
channel at [binstar](https://binstar.org).

Note that we currently don't have binary packages for
some platforms so if you successfully build them on anything
we don't provide at [https://binstar.org/twiecki](https://binstar.org/twiecki)
please let us know (especially windows).
