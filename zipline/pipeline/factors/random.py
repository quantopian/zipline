"""Factors produce deterministic pseudo-random numbers.

These can be useful for testing statistical models implemented in Pipeline.
"""

from numpy.random import RandomState

from .factor import CustomFactor


class _PseudoRandomCustomFactor(CustomFactor):
    """Base class for pseudorandom factors.
    """
    inputs = ()
    window_safe = True
    window_length = 1

    def compute(self, today, assets, out, seed, **params):
        state = RandomState(hash((today.value, seed)))
        out[:] = self.generate_random_values(state, len(out), **params)


class UniformRandom(_PseudoRandomCustomFactor):
    """A Factor generating uniform pseudo-random values.

    Parameters
    ----------
    low : float, optional
        Lower bound for generated values. Default is 0.
    high : float, optional
        Upper bound for generated values. Default is 1.
    seed : int, optional
        Seed for random number generation. Default is 0.
    """
    params = {
        'low': 0.0,
        'high': 0.0,
        'seed': 0,
    }

    @staticmethod
    def generate_random_values(state, size, low, high):
        return state.uniform(low=low, high=high, size=size)


class NormalRandom(_PseudoRandomCustomFactor):
    """A Factor generating normally-distributed pseudo-random values.

    Parameters
    ----------
    mean : float, optional
        Mean of generated random values.
    high : float, optional
        Upper bound for generated variables. Default is 1.
    seed : int, optional
        Seed for random number generation. Default is 0.
    """
    params = {
        'mean': 0.0,
        'std': 1.0,
        'seed': 0,
    }

    @staticmethod
    def generate_random_values(state, size, mean, std):
        return state.normal(loc=mean, scale=std)
