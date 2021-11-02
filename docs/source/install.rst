.. _install:

Installation
============

You can install Zipline either using `pip <https://pip.pypa.io/en/stable/>`_, the Python package installer, or
`conda <https://docs.conda.io/projects/conda/en/latest/index.html>`_, the package and environment management system
that runs on Windows, macOS, and Linux. In case you are installing `zipline-reloaded` alongside other packages and
encounter [conflict errors](https://github.com/conda/conda/issues/9707), consider using
[mamba](https://github.com/mamba-org/mamba) instead.

Zipline runs on Python 3.7, 3.8 and 3.9. To install and use different Python versions in parallel as well as create
a virtual environment, you may want to use `pyenv <https://github.com/pyenv/pyenv>`_.

Installing with ``pip``
-----------------------

Installing Zipline via ``pip`` is slightly more involved than the average Python package.

There are two reasons for the additional complexity:

1. Zipline ships several C extensions that require access to the CPython C API.
   In order to build these C extensions, ``pip`` needs access to the CPython
   header files for your Python installation.

2. Zipline depends on `NumPy <https://www.numpy.org/>`_, the core library for
   numerical array computing in Python.  NumPy, in turn, depends on the `LAPACK
   <https://www.netlib.org/lapack>`_ linear algebra routines.

Because LAPACK and the CPython headers are non-Python dependencies, the correct
way to install them varies from platform to platform.  If you'd rather use a
single tool to install Python and non-Python dependencies, or if you're already
using `Anaconda <https://www.anaconda.com/distribution/>`_ as your Python distribution,
you can skip to the :ref: `conda` section.

Once you've installed the necessary additional dependencies (see below for
your particular platform), you should be able to simply run (preferably inside an activated virtual environment):

.. code-block:: bash

   $ pip install zipline-reloaded

If you use Python for anything other than Zipline, we **strongly** recommend
that you install in a `virtualenv
<https://virtualenv.readthedocs.org/en/latest>`_.  The `Hitchhiker's Guide to
Python`_ provides an `excellent tutorial on virtualenv
<https://docs.python-guide.org/en/latest/dev/virtualenvs/>`_.

GNU/Linux
~~~~~~~~~

Dependencies
''''''''''''

On `Debian-derived`_ Linux distributions, you can acquire all the necessary
binary dependencies from ``apt`` by running:

.. code-block:: bash

   $ sudo apt install libatlas-base-dev python-dev gfortran pkg-config libfreetype6-dev hdf5-tools

On recent `RHEL-derived`_ derived Linux distributions (e.g. Fedora), the
following should be sufficient to acquire the necessary additional
dependencies:

.. code-block:: bash

   $ sudo dnf install atlas-devel gcc-c++ gcc-gfortran libgfortran python-devel redhat-rpm-config hdf5

On `Arch Linux`_, you can acquire the additional dependencies via ``pacman``:

.. code-block:: bash

   $ pacman -S lapack gcc gcc-fortran pkg-config hdf5

There are also AUR packages available for installing `ta-lib
<https://aur.archlinux.org/packages/ta-lib/>`_.
Python 3 is also installable via:

.. code-block:: bash

   $ pacman -S python3

Compiling TA-Lib
'''''''''''''''''
You will also need to compile the `TA-Lib <https://www.ta-lib.org/>`_ library for technical analysis so its headers become available.

You can accomplish this as follows:

.. code-block:: bash

   $ wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
   $ tar -xzf ta-lib-0.4.0-src.tar.gz
   $ cd ta-lib/
   $ sudo ./configure
   $ sudo make
   $ sudo make install

This will allow you to install the Python wrapper with ``pip`` as expected by the binary wheel.

macOS
~~~~~

The version of Python shipped with macOS is generally out of date, and
has a number of quirks because it's used directly by the operating system. For
these reasons, many developers choose to install and use a separate Python
installation.

The `Hitchhiker's Guide to Python`_ provides an excellent guide
to `Installing Python on macOS <https://docs.python-guide.org/en/latest/>`_, which
explains how to install Python with the `Homebrew <https://brew.sh/>`_ manager. Alternatively,
you could use `pyenv <https://github.com/pyenv/pyenv>`_.

Assuming you've installed Python with ``brew``, you'll also likely need the
following packages:

.. code-block:: bash

   $ brew install freetype pkg-config gcc openssl hdf5 ta-lib

Windows
~~~~~~~

For Windows, the easiest and best supported way to install Zipline is to use
``conda``.

.. _conda:

Installing with ``conda``
-------------------------

Another way to install Zipline is via the ``conda`` package manager, which
comes as part of the `Anaconda
<https://www.anaconda.com/distribution/>`_ distribution. Alternatively, you can use
the related but more lightweight `Miniconda <https://docs.conda.io/en/latest/miniconda.html#>`_  or
`Miniforge <https://github.com/conda-forge/miniforge>`_ installers.

The primary advantage of using Conda over ``pip`` is that ``conda`` natively
understands the complex binary dependencies of packages like ``numpy`` and
``scipy``.  This means that ``conda`` can install Zipline and its dependencies
without requiring the use of a second tool to acquire Zipline's non-Python
dependencies.

For instructions on how to install ``conda``, see the `Conda Installation
Documentation <https://conda.io/projects/conda/en/latest/user-guide/install/index.html>`_.

Unfortunately, as of April 2021, ``conda`` produces numerous false
positive [conflict errors](https://github.com/conda/conda/issues/9707)
while working to identify dependencies. Should this be your experience, consider
[mamba](https://github.com/mamba-org/mamba) instead, which works much faster and
reliably in most cases.

Once ``conda`` has been set up you can install Zipline from the ``ml4t`` channel.
You'll also need to activate the `conda-forge` and `ranaroussi` channels to source various dependencies.
You can do so either by adding them to your
`.condarc <https://docs.conda.io/projects/conda/en/latest/user-guide/configuration/use-condarc.html>`_
configuration file, or as command line flags:

.. code-block:: bash

    conda install -c ml4t -c conda-forge -c ranaroussi zipline-reloaded

.. _managing-conda-environments:

Managing ``conda`` environments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
It is recommended to install Zipline in an isolated ``conda`` environment.
Installing Zipline in ``conda`` environments will not interfere your default
Python deployment or site-packages, which will prevent any possible conflict
with your global libraries. For more information on ``conda`` environment, see
the `Conda User Guide <https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_.

Assuming ``conda`` has been set up, you can create a ``conda`` environment:

.. code-block:: bash

    $ conda create -n env_zipline python=3.8


Now you have set up an isolated environment called ``env_zipline``, a sandbox-like
structure to install Zipline. Then you should activate the conda environment
by using the command

.. code-block:: bash

    $ conda activate env_zipline

You can install Zipline by running

.. code-block:: bash

    (env_zipline) $ conda install -c ml4t zipline-reloaded

To deactivate the ``conda`` environment:

.. code-block:: bash

    (env_zipline) $ conda deactivate

.. note::
   ``conda activate`` and ``conda deactivate`` only work on conda 4.6 and later versions. For conda versions prior to 4.6, run:

      * Windows: ``activate`` or ``deactivate``
      * Linux and macOS: ``source activate`` or ``source deactivate``


.. _`Debian-derived`: https://www.debian.org/derivatives/
.. _`RHEL-derived`: https://en.wikipedia.org/wiki/Red_Hat_Enterprise_Linux_derivatives
.. _`Arch Linux` : https://www.archlinux.org/
.. _`Hitchhiker's Guide to Python` : https://docs.python-guide.org/en/latest/
.. _`Homebrew` : https://brew.sh
