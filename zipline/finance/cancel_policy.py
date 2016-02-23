import abc
import logbook

from six import with_metaclass, iteritems

from zipline.gens.sim_engine import DAY_END
from zipline.utils.serialization_utils import (
    VERSION_LABEL
)

warning_logger = logbook.Logger('AlgoWarning')


def log_warning(amount, sid, filled, policy_name):
    warning_logger.warn(
        'Your order for %s shares of %s has been partially filled. %s '
        'shares were successfully purchased. The remaining %s shares '
        'are being canceled based on the policy %s.' %
        (amount, sid, filled, amount - filled, policy_name)
    )


class CancelPolicy(with_metaclass(abc.ABCMeta)):

    def should_cancel(self, dt, event, order):
        pass

    def __getstate__(self):

        state_dict = \
            {k: v for k, v in iteritems(self.__dict__)
                if not k.startswith('_')}

        STATE_VERSION = 1
        state_dict[VERSION_LABEL] = STATE_VERSION

        return state_dict

    def __setstate__(self, state):

        OLDEST_SUPPORTED_STATE = 1
        version = state.pop(VERSION_LABEL)

        if version < OLDEST_SUPPORTED_STATE:
            raise BaseException("%s saved state is too old." %
                                self.__class__.__name__)

        self.__dict__.update(state)


class EODCancel(CancelPolicy):
    def __init__(self, warn_on_cancel=True):
        self.warn_on_cancel = warn_on_cancel

    def should_cancel(self, dt, event, order):
        if event == DAY_END:
            if self.warn_on_cancel:
                log_warning(order.amount, order.sid, order.filled,
                            self.__class__.__name__)
            return True

        return False


class NeverCancel(CancelPolicy):
    def __init__(self, warn_on_cancel=False):
        self.warn_on_cancel = warn_on_cancel

    def should_cancel(self, dt, event, order):
        return False
