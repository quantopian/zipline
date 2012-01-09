# qbt - Quantopian Backtesting Services

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

### MongoDB Server ###
QBT requires a running mongodb instance with a few collections:

- user collection. See handlers.BaseHandler for code using this collection. Documents must have:
	- email - standard email address
	- encrypted_password - an sha2 hex digest of the password
	- salt - a secret other than the password, which can be set in a secure cookie to maintain a session. if unset, will be set to sha2 hex of datetime.utcnow()--password 
	- _id - standard issue mongodb primary key
	



## Requesting a Backtest
