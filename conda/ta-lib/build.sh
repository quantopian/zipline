#!/bin/bash
wget -O ta-lib-0.4.0-src.tar.gz http://sourceforge.net/projects/ta-lib/files/ta-lib/0.4.0/ta-lib-0.4.0-src.tar.gz/download
tar xvzf ta-lib-0.4.0-src.tar.gz
pushd ta-lib
./configure --prefix=$PREFIX
make
make install
popd

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PREFIX/lib
python setup.py build
python setup.py install --prefix=$PREFIX
