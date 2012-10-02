#!/bin/bash

#setup virtualenvironment 
export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python2.7
export WORKON_HOME=/mnt/jenkins_backups/virtual_envs
if [ ! -d $WORKON_HOME ]; then
  mkdir $WORKON_HOME
fi
source /usr/local/bin/virtualenvwrapper.sh

mkvirtualenv zipline
workon zipline
pip install -r ./etc/requirements.txt
pip install -r ./etc/requirements_dev.txt

# Show what we have installed
pip freeze

#run all the tests in test. see setup.cfg for flags.
nosetests --config=jenkins_setup.cfg -I test_optimize

deactivate
