#!/bin/bash

# This script will be run by Vagrant to
# set up everything necessary to use Zipline.

# Because this is intended be a disposable dev VM setup,
# no effort is made to use virtualenv/virtualenvwrapper

# It is assumed that you have "vagrant up"
# from the root of the zipline github checkout.
# This will put the zipline code in the
# /vagrant folder in the system.

VAGRANT_LOG="/home/vagrant/vagrant.log"

# Need to "hold" grub-pc so that it doesn't break
# the rest of the package installs (in case of a "apt-get upgrade")
# (grub-pc will complain that your boot device changed, probably
#  due to something that vagrant did, and break your console)

echo "Obstructing updates to grub-pc..."
apt-mark hold grub-pc 2>&1 >> "$VAGRANT_LOG"

# Run a full apt-get update first.
echo "Updating apt-get caches..."
apt-get -y update 2>&1 >> "$VAGRANT_LOG"

# Install required packages
echo "Installing required packages..."
apt-get -y install python-pip python-dev g++ make libfreetype6-dev libpng-dev libopenblas-dev liblapack-dev gfortran 2>&1 >> "$VAGRANT_LOG"

# Add ta-lib
echo "Installing ta-lib integration..."
wget http://switch.dl.sourceforge.net/project/ta-lib/ta-lib/0.4.0/ta-lib-0.4.0-src.tar.gz 2>&1 "$VAGRANT_LOG"
tar -xvzf ta-lib-0.4.0-src.tar.gz 2>&1 >> "$VAGRANT_LOG"
cd ta-lib/
./configure --prefix=/usr 2>&1 >> "$VAGRANT_LOG"
make 2>&1 >> "$VAGRANT_LOG"
sudo make install 2>&1 >> "$VAGRANT_LOG"
cd ../

# Add Zipline python dependencies
echo "Installing python package dependencies..."
/vagrant/etc/ordered_pip.sh /vagrant/etc/requirements.txt 2>&1 >> "$VAGRANT_LOG"
# Add scipy next (if it's not done now, breaks installing of statsmodels for some reason ??)
echo "Installing scipy..."
pip install scipy==0.12.0 2>&1 >> "$VAGRANT_LOG"
echo "Installing zipline dev python dependencies..."
pip install -r /vagrant/etc/requirements_dev.txt 2>&1 >> "$VAGRANT_LOG"
echo "Finished!"
