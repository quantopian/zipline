"""
Small classes to assist with db access, timezone calculations, and so on.
"""

class DocWrap():
    """
        Provides attribute access style on top of dictionary results from pymongo. 
        Allows you to access result['field'] as result.field.
        Aliases result['_id'] to result.id.
        
    """
    def __init__(self, store=None):
        if(store == None):
            self.store = {}
        else:
            self.store = store.copy()
        if(self.store.has_key('_id')):
            self.store['id'] = self.store['_id']
            del(self.store['_id'])
        
    def __setitem__(self,key,value):
        if(key == '_id'):
            self.store['id'] = value
        else:
            self.store[key] = value
        
    def __getitem__(self, key):
        if self.store.has_key(key):
            return self.store[key]
            
    def __getattr__(self,attrname):
        if self.store.has_key(attrname):
            return self.store[attrname]
        else:
            raise AttributeError("No attribute named {name}".format(name=attrname))