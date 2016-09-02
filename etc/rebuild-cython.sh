#!/bin/bash
find zipline tests -regex '.*\.\(c\|so\)' -exec rm {} +
python setup.py build_ext --inplace
