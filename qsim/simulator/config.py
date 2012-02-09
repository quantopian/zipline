"""Tools for managing configuration data for sources and transforms."""
import json

class Config(object):
    """ Name/Value configuration object with type-safe accessors and json serialization/deserialization."""   
    
    def __init__(self, props):
        self.store = props
        
    def __setitem__(self, key, value):
        self.store[key] = value
        
    def __getitem__(self, key):
        if self.store.has_key(key):
            return self.store[key]
            
    def __getattr__(self, attrname):
        if self.store.has_key(attrname):
            return self.store[attrname]
        else:
            raise AttributeError("No attribute named {name}".format(name=attrname))
        
    def get_integer(self, name, default=0):
        """get the named config property as an integer"""
        return self.get_value(name, default, type(1))
        
    def get_string(self, name, default=''):
        """get the named config property as a string"""
        return self.get_value(name, default, type(''))
        
    def get_float(self, name, default=0.0):
        """get the named config property as a float"""
        return self.get_value(name, default, type(1.0))
    
    def get_value(self, name, default, expected_type):
        """
            return the named config property as the expected_type.
            if the property is missing, or is not of the right type, return default.
        """
        if(self.store.has_key(name)):
                val = self.store[name]
                if isinstance(val, expected_type):
                    return val
        else:
            return default
                    
    def to_json(self):
        """convert this config to a json string"""
        return json.dumps(self.store)
    
    def from_json(self, json_string):
        """parse a json string into this config object's properties"""
        self.store = json.loads(json_string)