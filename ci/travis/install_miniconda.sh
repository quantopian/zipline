#!/bin/bash

if [[ "$TRAVIS_OS_NAME" == "osx" ]]; then
    MINICONDA_OS=MacOSX
else
    MINICONDA_OS=Linux
fi

MINICONDA_DL_DIR="$HOME/.cache/miniconda/${CONDA_ROOT_PYTHON_VERSION:0:1}-4.3.30-$MINICONDA_OS"

if [ ! -d "$MINICONDA_DL_DIR" ]; then
    wget "https://repo.continuum.io/miniconda/Miniconda${CONDA_ROOT_PYTHON_VERSION:0:1}-4.3.30-$MINICONDA_OS-x86_64.sh" -O miniconda.sh
    chmod +x miniconda.sh
    cp miniconda.sh $MINICONDA_DL_DIR
    ./miniconda.sh -b -p $HOME/miniconda
fi
export PATH="$HOME/miniconda/bin:$PATH"
