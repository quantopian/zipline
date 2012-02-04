class Config(object):
   def __init__(self, dct):
       self.__dict__.update(dct)

mongo_conn_args = Config({
   'mongodb_host'         :  'claire.mongohq.com',
   'mongodb_port'         :  10087,
   'mongodb_dbname'       :  'quantodata-staging',
   'mongodb_user'         :  'quantopian',
   'mongodb_password'     :  'quantopian',
})

root_url = 'http://localhost:8000'
ws_url = 'ws://localhost:8001'
