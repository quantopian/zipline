#!/bin/bash

#setup virtualenvironment 
export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python2.7
export WORKON_HOME=/mnt/jenkins_backups/virtual_envs
if [ ! -d $WORKON_HOME ]; then
  mkdir $WORKON_HOME
fi
source /usr/local/bin/virtualenvwrapper.sh


#create the scientific python virtualenv and copy to provide zipline base
mkvirtualenv --no-site-packages scientific_base
workon scientific_base
./etc/ordered_pip.sh ./etc/requirements_sci.txt
deactivate
#re-base zipline
#rmvirtualenv zipline
cpvirtualenv scientific_base zipline  

workon zipline
./etc/ordered_pip.sh ./etc/requirements.txt
./etc/ordered_pip.sh ./etc/requirements_dev.txt

# Show what we have installed
pip freeze

#copy the host_settings file into place
cp /mnt/jenkins/zipline_host_settings.py ./host_settings.py

#documentation output
paver apidocs html

#load treasury data
python dataloader.py tr
#load benchmark data
python dataloader.py bm
#run all the tests in test. see setup.cfg for flags.
nosetests 

#run pylint checks
cp ./pylint.rcfile /mnt/jenkins/.pylintrc #default location for config file...
pylint -f parseable zipline | tee pylint.out

#run sloccount analysis
sloccount --wide --details ./  > sloccount.sc

deactivate
