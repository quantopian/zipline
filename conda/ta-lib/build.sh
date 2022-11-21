#!/bin/bash
if [[ "$OSTYPE" == "linux-gnu" ]]; then
  wget https://downloads.sourceforge.net/project/ta-lib/ta-lib/0.4.0/ta-lib-0.4.0-src.tar.gz &&
    tar xvfz ta-lib-0.4.0-src.tar.gz &&
    sudo apt-get update &&
    sudo apt-get install build-essential gcc-multilib g++-multilib &&
    cd ta-lib &&
    ./configure --prefix=$PREFIX &&
    make &&
    make install &&
    cd .. &&
    rm -rf ta-lib &&
    rm ta-lib-0.4.0-src.tar.gz &&
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PREFIX/lib &&
    export TA_INCLUDE_PATH=$PREFIX/include &&
    export TA_LIBRARY_PATH=$PREFIX/lib
elif [[ "$OSTYPE" == "darwin"* ]]; then
  brew upgrade &&
    brew install ta-lib &&
    brew info ta-lib
fi
python setup.py build &&
  python setup.py install --prefix=$PREFIX
