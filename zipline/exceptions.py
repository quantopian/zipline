from utils.exception_utils import CustomException

class ComponentNoInit(CustomException):
    argmap  = ('classname',)
    message = """Class {classname} does not define an init method."""
