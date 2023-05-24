import random

from abc import ABC, abstractmethod


class Noise(ABC):

    @property
    @abstractmethod
    def value(self) -> float:
        pass

    @abstractmethod
    def __str__(self):
        pass


class GaussianNoise(Noise):
    def __init__(self, mu: float = 0, sigma: float = 1):
        self.mu, self.sigma = mu, sigma

    @property
    def value(self):
        return random.gauss(self.mu, self.sigma)

    def __str__(self):
        return f'Gauss(mu={self.mu:.2f}, sigma={self.sigma:.2f})'

    __repr__ = __str__


class ZeroNoise(Noise):

    @property
    def value(self):
        return 0

    def __str__(self):
        return f'None'

    __repr__ = __str__


class UniformNoise(Noise):
    def __init__(self, a: float = -0.1, b: float = 0.1):
        self.a, self.b = a, b

    @property
    def value(self):
        return random.uniform(a=self.a, b=self.b)

    def __str__(self):
        return f'Uniform(a={self.a:.2f}, b={self.b:.2f})'

    __repr__ = __str__


if __name__ == '__main__':
    zeroNoise = ZeroNoise()
    print(zeroNoise)

    gaussianNoise = GaussianNoise(0, 2)
    print(gaussianNoise)

    d = {'a': zeroNoise, 'b': gaussianNoise}
    print(d)
