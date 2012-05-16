from textwrap import dedent

class CustomException(Exception):
    argmap = {0: 'classname'}

    def __init__(self, *args):
        self.args = args

    def format(self):
        assert len(self.args) == len(self.argmap), \
            """Wrong number of arguments passed to custom exception %s.""" \
            % self.__class__
        return self.message.format(**dict(zip(self.argmap, self.args)))

    def __str__(self):
        return dedent(self.format()).strip('\n')
