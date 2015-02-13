#!/bin/bash
git shortlog -ns master | awk '$1 >= $THRESHOLD {$1="";print $0}' | \
    cut -d" " -f2- > AUTHORS
