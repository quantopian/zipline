#!/bin/bash
wget https://downloads.sourceforge.net/project/ta-lib/ta-lib/0.4.0/ta-lib-0.4.0-src.tar.gz
tar xvfz ta-lib-0.4.0-src.tar.gz
pushd ta-lib
./configure --prefix=$PREFIX
make
make install
popd
rm ta-lib-0.4.0-src.tar.gz
rm -r ta-lib

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PREFIX/lib
export TA_INCLUDE_PATH=$PREFIX/include
export TA_LIBRARY_PATH=$PREFIX/lib
python setup.py build
python setup.py install --prefix=$PREFIX
