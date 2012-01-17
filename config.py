import json

class Config(object):
    """ Name/Value configuration object with type-safe accessors and json serialization/deserialization."""   
    
    def __init__(self, props):
        self.store = props
        
    def __setitem__(self,key,value):
        self.store[key] = value
        
    def __getitem__(self, key):
        if self.store.has_key(key):
            return self.store[key]
            
    def __getattr__(self,attrname):
        if self.store.has_key(attrname):
            return self.store[attrname]
        else:
            raise AttributeError("No attribute named {name}".format(name=attrname))
        
    def get_integer(self, name, default=0):
        return self.get_value(name, default, type(1))
        
    def get_string(self, name, default=''):
        return self.get_value(name, default, type(''))
        
    def get_float(self, name, default=0.0):
        return self.get_value(name, default, type(1.0))
    
    def get_value(self, name, default, expected_type):
        if(self.store.has_key(name)):
                val = self.store[name]
                if isinstance(val, expected_type):
                    return val
        else:
            return default
                    
    def to_json(self):
        return json.dumps(self.store)
    
    def from_json(self, json_string):
        self.store = json.loads(json_string)