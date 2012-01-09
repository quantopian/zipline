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

import qbt_server

define("user_email", default="qbt@quantopian.com", help="email address for qbt user")
define("password", default="foobar", help="password for qbt user")

def db_main():
    tornado.options.parse_command_line()
    connection, db = qbt_server.connect_db()
    
    #create a user for testing
    salt, encrypted_password = qbt_server.encrypt_password(None, options.password)
    
    if not db.users.find_one({'email':options.user_email}):
        db.users.insert({'email':options.user_email, 'encrypted_password':encrypted_password, 'salt':salt})
    
if __name__ == "__main__":
    db_main()
