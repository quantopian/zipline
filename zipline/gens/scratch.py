
class gen_wrapper(object):
    
    def __init__(self, val):
        self.val = val
        self.iterator = iter(xrange(self.val))
    def reset_iter(self):
        self.val

    def __iter__(self):
        return self.iterator

    def next():
        return self.iterator.next()
