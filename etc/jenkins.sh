#!/bin/bash

#setup virtualenvironment 
export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python2.7
if [ ! -d $HOME/.venvs ]; then
  mkdir $HOME/.venvs
fi
export WORKON_HOME=$HOME/.venvs
source /usr/local/bin/virtualenvwrapper.sh

#create the scientific python virtualenv and copy to provide qsim base
mkvirtualenv --no-site-packages scientific_base
workon scientific_base
./etc/ordered_pip.sh ./etc/requirements_sci.txt
deactivate
#re-base qsim
#rmvirtualenv qsim
cpvirtualenv scientific_base qsim  

workon qsim
./etc/ordered_pip.sh ./etc/requirements.txt
./etc/ordered_pip.sh ./etc/requirements_dev.txt

# Show what we have installed
pip freeze

#run all the tests in test
nosetests --with-xunit --cover-erase --cover-package=qsim

#run pylint checks
pylint -f parseable . | tee pylint.out

#run sloccount analysis
sloccount --wide --details ./  > sloccount.sc

#documentation output
paver apidocs html

deactivate
