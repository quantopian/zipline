if [ -n "${VAR:-x}" ]; then
  WORKSPACE=.
fi

echo $WORKSPACE
PYENV_HOME=$WORKSPACE/.pyenv/

  
# Delete previously built virtualenv
if [ -d $PYENV_HOME ]; then
    rm -rf $PYENV_HOME
fi

# Create virtualenv and install necessary packages
virtualenv --no-site-packages $PYENV_HOME
. $PYENV_HOME/bin/activate
ordered_pip.sh $WORKSPACE/requirements.txt
ordered_pip.sh $WORKSPACE/requirements_dev.txt
cp /mnt/jenkins_backup/host_settings.py ./
nosetests