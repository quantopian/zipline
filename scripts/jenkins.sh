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
./scripts/ordered_pip.sh ./scripts/requirements_sci.txt
deactivate
#re-base qsim
#rmvirtualenv qsim
cpvirtualenv scientific_base qsim  

workon qsim
./scripts/ordered_pip.sh ./scripts/requirements.txt
./scripts/ordered_pip.sh ./scripts/requirements_dev.txt

#setup the local mongodb
python ./scripts/dev_setup.py 

#run all the tests in test
nosetests --with-xcoverage --with-xunit --cover-erase --cover-package=simulator,transforms
pylint -f parseable . | tee pylint.out

#run sloccount analysis
sloccount --wide --details ./  > sloccount.sc

deactivate
