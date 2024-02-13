import inspect
from operator import attrgetter
from textwrap import dedent

from zipline import api, TradingAlgorithm


def main():
    with open(api.__file__.rstrip('c') + 'i', 'w') as stub:
        # Imports so that Asset et al can be resolved.
        # "from MOD import *" will re-export the imports from the stub, so
        # explicitly importing.
        stub.write(dedent("""\
        import collections
        from zipline.assets import Asset, Equity, Future
        from zipline.assets.futures import FutureChain
        from zipline.finance.asset_restrictions import Restrictions
        from zipline.finance.cancel_policy import CancelPolicy
        from zipline.pipeline import Pipeline
        from zipline.protocol import Order
        from zipline.utils.events import EventRule
        from zipline.utils.security_list import SecurityList

        """))

        # Sort to generate consistent stub file:
        for api_func in sorted(TradingAlgorithm.all_api_methods(),
                               key=attrgetter('__name__')):
            stub.write('\n')
            sig = inspect._signature_bound_method(inspect.signature(api_func))

            indent = ' ' * 4
            stub.write(dedent('''\
                def {func_name}{func_sig}:
                    """'''.format(func_name=api_func.__name__,
                                  func_sig=sig)))
            stub.write(dedent('{indent}{func_doc}'.format(
                # `or '\n'` is to handle a None docstring:
                func_doc=dedent(api_func.__doc__.lstrip()) or '\n',
                indent=indent,
            )))
            stub.write('{indent}"""\n'.format(indent=indent))


if __name__ == '__main__':
    main()
