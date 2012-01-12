import tornado.auth
import tornado.httpserver
import tornado.ioloop
from tornado.options import define, options
import tornado.web
import pymongo
import bson
import hashlib
import base64
import uuid
import os
import logging
import datetime
import multiprocessing 

logger = logging.getLogger()

define("port", default=8888, help="run the qbt on the given port", type=int)
define("mongodb_host", default="127.0.0.1", help="mongodb host address")
define("mongodb_port", default=27017, help="connect to the mongodb on the given port", type=int)
define("mongodb_dbname", default="qbt", help="database name")
define("mongodb_user", default="qbt", help="database user")
define("mongodb_password", default="qbt", help="database password")

HASH_ALGO = 'sha256'

def connect_db():
    connection = pymongo.Connection(options.mongodb_host, options.mongodb_port)
    db = connection[options.mongodb_dbname]
    db.authenticate(options.mongodb_user, options.mongodb_password)
    return connection, db
    
def encrypt_password(salt, password):
    if(salt == None):
        h1 = hashlib.new(HASH_ALGO)
        h1.update(str(datetime.datetime.utcnow())+"--"+password)
        salt = h1.hexdigest()
        
    h2 = hashlib.new(HASH_ALGO)
    h2.update(salt+"--"+password)
    encrypted_password = h2.hexdigest()
    
    return salt, encrypted_password

class Application(tornado.web.Application):
    def __init__(self):
        handlers = [
            (r"/", MainHandler),
            (r"/login", LoginHandler),
            (r"/backtest", BacktestHandler)
        ]
        settings = dict(
            template_path=os.path.join(os.path.dirname(__file__), "templates"),
            static_path=os.path.join(os.path.dirname(__file__), "static"),
            xsrf_cookies=False,
            cookie_secret=base64.b64encode(uuid.uuid4().bytes + uuid.uuid4().bytes),
            login_url="/login",
            #autoescape=None,
            debug=True,
        )
        tornado.web.Application.__init__(self, handlers, **settings)

        # Have one global connection to the blog DB across all handlers
        self.connection, self.db = connect_db()


class BaseHandler(tornado.web.RequestHandler):
    @property
    def db(self):
        return self.application.db

    def get_current_user(self):
        user_id = self.get_secure_cookie(u"user_id")
        logger.info("looking up user with id: {id}".format(id=user_id))
        if not user_id: return None
        #get user record by id
        users = self.db.users.find(spec={"_id":bson.ObjectId(user_id)}, limit=1)
        if(users.count > 0):
            return users[0]
        return None

class MainHandler(BaseHandler):
    @tornado.web.authenticated
    def get(self):
        self.write("Hello, world. Try launching a <a href='/backtest'>backtest</a>.")
                         
class LoginHandler(BaseHandler):
    def get(self):
        self.write('<html><body><form action="/login" method="post">'
                   'Name: <input type="text" name="user_name">'
                   'pass: <input type="password" name="password">'
                   '<input type="submit" value="Sign in">'
                   '</form></body></html>')

    def post(self):  
        self.authenticate(self.get_argument("user_name"),self.get_argument("password"))
        self.redirect("/")

    def authenticate(self, username, password):
        h = hashlib.new(HASH_ALGO)
        #find user record by username.
        users = self.db.users.find(spec={"email":username}, limit=1)
        if(users.count > 0):
            user_record = users[0]
        else:
            logger.debug("no user with name: {username}", username=username)
            return

        
        #calculate password hash
        salt, encrypted_password = encrypt_password(user_record['salt'], password)
        if (user_record['encrypted_password'] == encrypted_password):
            #we have a match, so set the secure cookie to the salt
            logger.debug("setting user_id cookie to {id}".format(id=user_record['_id']))
            self.set_secure_cookie(u"user_id", unicode(user_record['_id']))

class BacktestHandler(BaseHandler):
    @tornado.web.authenticated
    def get(self):
        self.write('<html><body><form action="/backtest" method="post">'
                   '<input type="submit" value="Launch">'
                   '</form></body></html>')
    @tornado.web.authenticated
    def post(self):
        


def main():
    tornado.options.parse_command_line()
    http_server = tornado.httpserver.HTTPServer(Application())
    http_server.listen(options.port)
    tornado.ioloop.IOLoop.instance().start()


if __name__ == "__main__":
    main()
