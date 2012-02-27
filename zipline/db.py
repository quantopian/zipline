import atexit
import pymongo
import zipline.util as qutil

class MongoOptions(object):
    
    def __init__(self, host, port, dbname, user, password):
        self.mongodb_host       = host
        self.mongodb_port       = port
        self.mongodb_dbname     = dbname
        self.mongodb_user       = user
        self.mongodb_password   = password

class NoDatabase(Exception):
    def __repr__(self):
        return 'The database has not been set up yet.'

def setup_db(credentials):
    """
    Setup the database. Has global side effects.
    """
    qutil.LOGGER.info(dir(DbConnection))
    if not DbConnection.initd:
        connector = connect_db(credentials)
        DbConnection.set(*connector)

def connect_db(options):
    """
    Connect to pymongo, return a connection and database instance
    as a tuple.
    """

    connection = pymongo.Connection(options.mongodb_host, options.mongodb_port)

    db = connection[options.mongodb_dbname]
    db.authenticate(options.mongodb_user, options.mongodb_password)

    def _gc_connection(): # pragma: no cover
        connection.close()

    atexit.register(_gc_connection)
    return connection, db

class DbConnection(object):
    """
    Hold the shared state of the database connection.
    """

    initd       = False
    __shared    = {}
    
    def __init__(self):
        self.__dict__ = self.__shared

    @staticmethod
    def set(conn, db):
        DbConnection.__shared['conn'] = conn
        DbConnection.__shared['db'] = db
        DbConnection.initd = True

    @staticmethod
    def get():
        return (
            DbConnection.__shared['conn'],
            DbConnection.__shared['db']
        )

    def __getattr__(self, key):
        if not DbConnection.__shared.get('initd'):
            raise NoDatabase()
        else:
            return DbConnection.__shared.get(key)

    def destory(self): # pragma: no cover
        DbConnection.__shared['initd'] = False
        self.conn.close()
