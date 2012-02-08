#!/bin/bash

echo $hash
while read line
do     
    if [[ $line != \#* ]] ; then
        #echo $line
        pip install $line
    fi
done < $1
echo "Final line count is: $a";
