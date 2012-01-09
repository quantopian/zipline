# qbt - Quantopian Backtesting Services

##Development Quickstart
Navigate to your mongodb installation and start your db server:
	./bin/mongodb 

Install the necessary python libraries:
	easy_install tornado pymongo

Create a development database with sample data, will create one qbt user:
	python qbt_data_bootstrap.py --user_email=... --password=...

To run the qbt against a local mongodb instance, navigate to the source dir and run
	python qbt_server.py --mongodb_dbname=qbt 

To see all the available options:
	python qbt_server.py --help
or
	python qbt_data_bootstrap.py --help

	

qbt uses tornado to accept synchronous requests for backtesting sessions. 
The client of a backtesting session first invokes the _backtest_ endpoint:
http://serverip/backtest?startdate=<>&enddate=<>...

qbt will respond with a json object describing the session:
- backtest id, to be referenced in all further requests
- zeromq connection information for the event stream

A backtesting session is comprised of:
- REST endpoint to request orders 
- an event stream delivered via zeromq

## Pre-requisites
You need to have the tornado and pymongo eggs installed:
	easy_install tornado pymongo

You need to have mongodb installed and running. Find your system at http://www.mongodb.org/downloads and set it up.

### Database and Collections expected in MongoDB ###
QBT requires a running mongodb instance with a few collections:

- user collection. See handlers.BaseHandler and handlers.LoginHandler for code using this collection. Documents must have:
	- email - standard email address
	- salt - sha256 hex of: datetime.utcnow()--password 
	- encrypted_password - an sha256 hex digest of: salt--password
	- _id - standard issue mongodb primary key
	


## Authenticating

## Requesting a Backtest
