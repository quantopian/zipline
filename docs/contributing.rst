***************************
Contributing to the project
***************************

Development Environment
=======================

The following guide assumes your system has [virtualenvwrapper](https://bitbucket.org/dhellmann/virtualenvwrapper)
and [pip](http://www.pip-installer.org/en/latest/) already installed.

You'll need to install some C library dependencies:

```
sudo apt-get install libopenblas-dev liblapack-dev gfortran

wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xvzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install
```

Suggested installation of Python library dependencies used for development:

```
mkvirtualenv zipline
./etc/ordered_pip.sh ./etc/requirements.txt
pip install -r ./etc/requirements_dev.txt
```

Finally, install zipline in develop mode (from the zipline source root dir):

```
python setup.py develop
```

Style Guide
===========

To ensure that changes and patches are focused on behavior changes,
the zipline codebase adheres to PEP-8,
`<http://www.python.org/dev/peps/pep-0008/>`_.

The maintainers check the code using the flake8 script,
`<https://github.com/jcrocholl/pep8/>`_, which is included in the
requirements_dev.txt.

Before submitting patches or pull requests, please ensure that your
changes pass

::

    flake8 zipline tests

Source
======

The source for Zipline is hosted at
`<https://github.com/quantopian/zipline>`_.

Contact
=======

For other questions, please contact `<opensource@quantopian.com>`_.
