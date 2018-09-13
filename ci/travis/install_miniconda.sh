#!/bin/bash

if [[ "$TRAVIS_OS_NAME" == "osx" ]]; then
    MINICONDA_OS=MacOSX
else
    MINICONDA_OS=Linux
fi

MINICONDA_SH="$HOME/.cache/miniconda/${CONDA_ROOT_PYTHON_VERSION:0:1}-4.3.30-$MINICONDA_OS"

if [ ! -f "$MINICONDA_SH" ]; then
    wget "https://repo.continuum.io/miniconda/Miniconda${CONDA_ROOT_PYTHON_VERSION:0:1}-4.3.30-$MINICONDA_OS-x86_64.sh" -O miniconda.sh
    chmod +x miniconda.sh
    cp miniconda.sh $MINICONDA_SH
fi
$MINICONDA_SH -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"
