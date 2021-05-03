#!/bin/bash

if [[ "$OSTYPE" == "darwin"* ]]; then
  find -E zipline tests -regex '.*\.(c|so)' -exec rm {} +
else
  find src/zipline tests -regex '.*\.\(c\|so\)' -exec rm {} +
fi
python setup.py build_ext --inplace
