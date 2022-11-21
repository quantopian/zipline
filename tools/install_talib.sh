#!/bin/bash

curl -L -o ta-lib-0.4.0-src.tar.gz http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz &&
  tar xvfz ta-lib-0.4.0-src.tar.gz &&
  cd ta-lib &&
  ./configure &&
  make &&
  make install &&
  ldconfig
