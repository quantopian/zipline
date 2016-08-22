#!/bin/bash
find zipline tests -name "*.so" | xargs rm || true
python setup.py build_ext --inplace
