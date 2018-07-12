#!/bin/bash

if [[ "$TRAVIS_OS_NAME" == "osx" ]]; then
    MINICONDA_OS=MacOSX
else
    MINICONDA_OS=Linux
fi

wget "https://repo.continuum.io/miniconda/Miniconda${CONDA_ROOT_PYTHON_VERSION:0:1}-4.3.30-$MINICONDA_OS-x86_64.sh" -O miniconda.sh
chmod +x miniconda.sh
./miniconda.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"
