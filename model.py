import numpy as np
import pandas as pd

from noise import Noise


class Model:
    def __init__(
            self,
            beta: float,
            gamma: float,
            I0: float,
            noise_beta: Noise,
            noise_gamma: Noise,
            dt: float,
            T: float,
    ):
        self.beta = beta
        self.gamma = gamma

        self.S = 1 - I0
        self.I = I0
        self.R = 0

        self.noise_beta = noise_beta
        self.noise_gamma = noise_gamma
        self.dt = dt

        self.t = 0
        self.T = T

        # Collect historical values
        self.idx = 0
        N = len(np.arange(0, T+dt, dt))

        self.S_hist = np.empty(N, float)
        self.I_hist = np.empty(N, float)
        self.R_hist = np.empty(N, float)

        self.beta_hist = np.empty(N, float)
        self.gamma_hist = np.empty(N, float)

        self.t_hist = np.empty(N, float)

    def update_history(self, beta: float, gamma: float) -> None:
        idx = self.idx

        self.t_hist[idx] = self.t

        self.S_hist[idx] = self.S
        self.I_hist[idx] = self.I
        self.R_hist[idx] = self.R

        self.beta_hist[idx] = beta
        self.gamma_hist[idx] = gamma

        self.idx += 1

    def step(self) -> None:

        def _beta_gamma() -> tuple[float, float]:

            _beta = self.beta + self.noise_beta.value
            if _beta < 0:
                _beta = 0
            elif _beta > 2*self.beta:
                _beta = 2*self.beta

            _gamma = self.gamma + self.noise_gamma.value
            if _gamma < 0:
                _gamma = 0
            elif _gamma > 2*self.gamma:
                _gamma = 2*self.gamma

            return _beta, _gamma

        # Since `beta > 0` so `abs(noise) <= beta`. Same for `gamma`.
        beta, gamma = _beta_gamma()
        self.update_history(beta=beta, gamma=gamma)

        dS = (-beta * self.I * self.S) * self.dt
        dI = (beta * self.I * self.S - gamma * self.I) * self.dt
        dR = (gamma * self.I) * self.dt

        self.S += dS
        self.I += dI
        self.R += dR

        self.t += self.dt

    def run(self) -> pd.DataFrame:
        while self.t <= self.T:
            self.step()

        return pd.DataFrame({
            't': self.t_hist,
            'S': self.S_hist,
            'I': self.I_hist,
            'R': self.R_hist,
            'beta': self.beta_hist,
            'gamma': self.gamma_hist,
        })
