#!/bin/bash
NBDIR=notebooks

for fullfile in $NBDIR/*.ipynb; do
    echo "Processing $fullfile file..";
    filename=$(basename "$fullfile")
    extension="${filename##*.}"
    filename="${filename%.*}"
    ipython nbconvert $fullfile --to markdown --output $filename
done
