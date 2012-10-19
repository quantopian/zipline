Zipline
=======

Zipline is a realtime stream processing system.

System Setup
==============

Running
-------

Initial `virtualenv` setup::

    $ mkvirtualenv zipline
    $ workon zipline
	# Go get coffee, the following will compile a heap of C/C++ code
	$ ./etc/ordered_pip.sh etc/requirements.txt
	# And optionally:
	$ ./etc/ordered_pip.sh etc/requirements_dev.txt


Develop
-------

To run tests::

    $ nosetests

Tooling hints
================
zipline relies heavily on scientific Python components (numpy, scikit, pandas, matplotlib, ipython, etc). Tooling up can be a pain, and it often involves managing a configuration including your OS, c/c++/fortran compilers, python version, and versions of numerous modules. I've found the following tools absolutely indispensable: 

- some kind of package manager for your platform. package managers generally give you a way to search, install, uninstall, and check currently installed packages. They also do a great job of managing dependencies.
    - Linux: yum/apt-get
    - Mac OS: homebrew/macport/fink (I highly recommend homebrew: https://github.com/mxcl/homebrew) 
    - Windows: probably best if you use a complete distribution, like: enthought, ActiveState, or Python(x,y)
- Python also provides good package management tools to help you manage the components you install for Python.
    - pip
    - easy_install/setuptools. I have always used setuptools, and I've been quite happy with it. Just remember that setuptools is coupled to your python version. 
- virtualenv and virtualenvwrapper are your very best friends. They complement your python package manager by allowing you to create and quickly switch between named configurations.
    - *Install all the versions of Python you like to use, but install setuptools, virtualenv, and virtualenvwrapper with the very latest python.* Use the latest python to install the latest setuptools, and the latest setuptools to install virtualenv and virtualenvwrapper. virtualenvwrapper allows you to specify the python version you wish to use (mkvirtualenv -p <python executable> <env name>), so you can create envs of any python denomination.

Mac OS hints
-------------

Scientific python on the Mac can be a bit confusing because of the many independent variables. You need to have several components installed, and be aware of the versions of each:

- XCode. XCode includes the gcc and g++ compilers and architecture specific assemblers. Your version of XCode will determine which compilers and assemblers are available. The most common issue I encountered with scientific python libraries is compilation errors of underlying C code. Most scientific libraries are optimized with C routines, so this is a major hurdle. In my environment (XCode 4.0.2 with iOS components installed) I ran into problems with -arch flags asking for power pc (-arch ppc passed to the compiler). Read this stackoverflow to see how to handle similar problems: http://stackoverflow.com/questions/5256397/python-easy-install-fails-with-assembler-for-architecture-ppc-not-installed-on
- gfortran 	- you need this to build numpy. With brew you can install with just: ```brew install gfortran```
- umfpack 	- you need this to build scipy. ```brew install umfpack```
- swig		- you need this to build scipy. ```brew install swig```
- hdf5	 	- you need this to build tables. ```brew install hdf5```

Style Guide
===========

To ensure that changes and patches are focused on behavior changes, the zipline codebase adheres to PEP-8, <http://www.python.org/dev/peps/pep-0008/>.

The maintainers check the code using the flake8 script, <https://github.com/jcrocholl/pep8/>, which is included in the requirements_dev.txt.

Before submitting patches or pull requests, please ensure that your changes pass ```flake8 zipline tests```

Discussion and Help
===================

Discussion of the project is held at the Google Group, <zipline@googlegroups.com>, <https://groups.google.com/forum/#!forum/zipline>.

Source
======
The source for Zipline is hosted at <https://github.com/quantopian/zipline>.

Build Status
============
[![Build Status](https://travis-ci.org/quantopian/zipline.png)](https://travis-ci.org/quantopian/zipline)

Contact
=======

For other questions, please contact <opensource@quantopian.com>.
