#!/bin/bash

#setup virtualenvironment 
export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python2.7
if [ ! -d $HOME/.venvs ]; then
  mkdir $HOME/.venvs
fi
export WORKON_HOME=$HOME/.venvs
source /usr/local/bin/virtualenvwrapper.sh

#create the scientific python virtualenv and copy to provide qexec base
mkvirtualenv --no-site-packages scientific_base
workon scientific_base
./ordered_pip.sh requirements_sci.txt
deactivate
#re-base qexec
#rmvirtualenv qexec
cpvirtualenv scientific_base qexec  

workon qexec
./ordered_pip.sh requirements.txt
./ordered_pip.sh requirements_dev.txt

#setup the local mongodb
python dev_setup.py 

#run all the tests in test
nosetests --with-xcoverage --with-xunit --cover-package=myapp --cover-erase
pylint -f parseable . | tee pylint.out

deactivate
