#!/bin/bash

if [[ "$TRAVIS_OS_NAME" == "osx" ]]; then
    MINICONDA_OS=MacOSX
else
    MINICONDA_OS=Linux
fi

MINICONDA_DIR="$HOME/.cache/miniconda/${CONDA_ROOT_PYTHON_VERSION:0:1}-4.3.30-$MINICONDA_OS"

if [ ! -d "$MINICONDA_DIR" ]; then
    wget "https://repo.continuum.io/miniconda/Miniconda${CONDA_ROOT_PYTHON_VERSION:0:1}-4.3.30-$MINICONDA_OS-x86_64.sh" -O miniconda.sh
    chmod +x miniconda.sh
    ./miniconda.sh -b -p $MINICONDA_DIR
fi
export PATH="$MINICONDA_DIR/bin:$PATH"
