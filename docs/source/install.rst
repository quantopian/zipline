Install
=======
Please use python 3.6 and let's avoid unnecessary issues.
If you use Python for anything other than Zipline, I **strongly** recommend
that you install in a `virtualenv
<https://virtualenv.readthedocs.org/en/latest>`_.

The `Hitchhiker's Guide to Python`_ provides an `excellent tutorial on virtualenv
<https://docs.python-guide.org/en/latest/dev/virtualenvs/>`_.


For now I support only installation through github. You could do that in one of these ways:

Installing with git clone
--------------------------
 * git clone https://github.com/shlomikushchi/zipline-trader.git
 * <create/activate a virtual env> - optional but recommended
 * pip install -e .

The last step will install this project from source, giving you the ability to debug zipline-trader's code.

Installing using pip directly from github
----------------------------------------------
You can install it with ability to debug it like this:

.. code-block:: bash

    pip install -e git://github.com/shlomikushchi/zipline-trader.git#egg=zipline-trader

To install a specific version, you could do this (installing version 1.5.0):

.. code-block:: bash

    pip install -e git://github.com/shlomikushchi/zipline-trader.git@1.5.0#egg=zipline-trader

Installing from pypi (coming soon)
-----------------------------------
The most known way of installing would be installing from pypi:

.. code-block:: bash

    pip install zipline-trader

* Installing using Anaconda (Probably supported in the future)


Notes
----------

Installing zipline is a bit complicated, and therefore installing zipline-trader.
There are two reasons for zipline installation additional complexity:

1. Zipline ships several C extensions that require access to the CPython C API.
   In order to build the C extensions, ``pip`` needs access to the CPython
   header files for your Python installation.

2. Zipline depends on `numpy <https://www.numpy.org/>`_, the core library for
   numerical array computing in Python.  Numpy depends on having the `LAPACK
   <https://www.netlib.org/lapack>`_ linear algebra routines available.

Because LAPACK and the CPython headers are non-Python dependencies, the correct
way to install them varies from platform to platform.
Once you've installed the necessary additional dependencies (see below for
your particular platform)

GNU/Linux
))))))))))))))))

On `Debian-derived`_ Linux distributions, you can acquire all the necessary
binary dependencies from ``apt`` by running:

.. code-block:: bash

   $ sudo apt-get install libatlas-base-dev python-dev gfortran pkg-config libfreetype6-dev hdf5-tools

On recent `RHEL-derived`_ derived Linux distributions (e.g. Fedora), the
following should be sufficient to acquire the necessary additional
dependencies:

.. code-block:: bash

   $ sudo dnf install atlas-devel gcc-c++ gcc-gfortran libgfortran python-devel redhat-rpm-config hdf5

On `Arch Linux`_, you can acquire the additional dependencies via ``pacman``:

.. code-block:: bash

   $ pacman -S lapack gcc gcc-fortran pkg-config hdf5

There are also AUR packages available for installing `ta-lib
<https://aur.archlinux.org/packages/ta-lib/>`_, an optional Zipline dependency.

OSX
))))))))))

The version of Python shipped with OSX by default is generally out of date, and
has a number of quirks because it's used directly by the operating system.  For
these reasons, many developers choose to install and use a separate Python
installation. The `Hitchhiker's Guide to Python`_ provides an excellent guide
to `Installing Python on OSX <https://docs.python-guide.org/en/latest/>`_, which
explains how to install Python with the `Homebrew`_ manager.

Assuming you've installed Python with Homebrew, you'll also likely need the
following brew packages:

.. code-block:: bash

   $ brew install freetype pkg-config gcc openssl hdf5

..

.. _`Debian-derived`: https://www.debian.org/misc/children-distros
.. _`RHEL-derived`: https://en.wikipedia.org/wiki/Red_Hat_Enterprise_Linux_derivatives
.. _`Arch Linux` : https://www.archlinux.org/
.. _`Hitchhiker's Guide to Python` : http://docs.python-guide.org/en/latest/
.. _`Homebrew` : http://brew.sh