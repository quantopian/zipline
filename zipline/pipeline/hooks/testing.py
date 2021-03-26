from collections import namedtuple

from .iface import PipelineHooks, PIPELINE_HOOKS_CONTEXT_MANAGERS

from interface import implements

from zipline.utils.compat import contextmanager, wraps


Call = namedtuple("Call", "method_name args kwargs")


class ContextCall(namedtuple("ContextCall", "state call")):
    @property
    def method_name(self):
        return self.call.method_name

    @property
    def args(self):
        return self.call.args

    @property
    def kwargs(self):
        return self.call.kwargs


def testing_hooks_method(method_name):
    """Factory function for making testing methods."""
    if method_name in PIPELINE_HOOKS_CONTEXT_MANAGERS:
        # Generate a method that enters the context of all sub-hooks.
        @wraps(getattr(PipelineHooks, method_name))
        @contextmanager
        def ctx(self, *args, **kwargs):
            call = Call(method_name, args, kwargs)
            self.trace.append(ContextCall("enter", call))
            yield
            self.trace.append(ContextCall("exit", call))

        return ctx

    else:
        # Generate a method that calls methods of all sub-hooks.
        @wraps(getattr(PipelineHooks, method_name))
        def method(self, *args, **kwargs):
            self.trace.append(Call(method_name, args, kwargs))

        return method


class TestingHooks(implements(PipelineHooks)):
    """A hooks implementation that keeps a trace of hook method calls."""

    def __init__(self):
        self.trace = []

    def clear(self):
        self.trace = []

    # Implement all interface methods by delegating to corresponding methods on
    # input hooks.
    locals().update(
        {
            name: testing_hooks_method(name)
            # TODO: Expose this publicly on interface.
            for name in PipelineHooks._signatures
        }
    )
