import types
from collections import Container, Hashable, Callable

class Any(object): pass

class Workflow(Container, Callable):

    def __init__(self, states, transitions, initial_state):
        self.simple = set()
        self.complx = []

        if isinstance(states[0], tuple):
            self.groups = {b for _,b in states}
        else:
            self.groups = set()

        matcher = lambda b: lambda f,t : t == b

        for (a, b) in transitions.itervalues():
            if a is Any:
                self.complx.append(matcher(b))
            if isinstance(a, Hashable) and isinstance(b, Hashable):
                self.simple.add((a,b))

    def __call__(self, **kwargs):
        if 'group' in kwargs:
            return self.groups

    def __contains__(self, state):
        if state in self.simple:
            return True
        for match in self.complx:
            if match(*state):
                return True
        else:
            return False

class Flowable:

    @property
    def state(self):
        if not hasattr(self, '_state'):
            self._state = self.initial_state
        else:
            return self._state

    @state.setter
    def state(self, new):
        if not hasattr(self, '_state'):
            self._state = self.initial_state

        old = self._state

        if (old, new) in self.workflow:
            self._state = new
        else:
            raise RuntimeError("Invalid State Transition : %s -> %s" %(old, new))

class WorkflowMeta(type):
    """
    Base metaclass component workflows.
    """

    def __new__(cls, name, mro, attrs):

        state         = attrs.get('states', None)
        transitions   = attrs.get('transitions', None)
        initial_state = attrs.get('initial_state', None)

        if attrs.get('workflow'):
            raise RuntimeError('`workflow` is a reserved attribute.')

        if not state:
            raise RuntimeError('Must specify states')

        if not transitions:
            raise RuntimeError('Must specify transitions')

        if not transitions:
            raise RuntimeError('Must specify initial_state')

        new_class = super(WorkflowMeta, cls).__new__(
            cls, name, mro+(Flowable,), attrs
        )
        new_class.workflow = Workflow(state, transitions, initial_state)

        return new_class
