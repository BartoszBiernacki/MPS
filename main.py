import matplotlib.pyplot as plt

from model import Model
from batch_runner import BatchRunner
from analyzer import Analyzer
from noise import *

fixed_params = {
    'beta': 0.2,
    'gamma': 0.05,
    'I0': 0.01,
    'noise_beta': GaussianNoise(mu=0., sigma=0.05),
    'noise_gamma': UniformNoise(a=-0.04, b=0.04),
    'dt': 0.01,
    'T': 300,
}

variable_params = {
    # 'beta': [0.10],
}


def run_model() -> None:
    result = Model(**fixed_params).run()

    for stat_name in 'SIR':
        plt.plot(result['t'], result[stat_name], label=stat_name)
    plt.legend()
    plt.show()


def run_batch_runner(n: int) -> None:
    BatchRunner(
        model_cls=Model,
        fixed_params=fixed_params,
        variable_params=variable_params,
        iterations=n,
    ).run()


def run_analyzer(n: int) -> None:
    Analyzer(
        model_cls=Model,
        fixed_params=fixed_params,
        variable_params=variable_params,
        iterations=n,
    ).run()


if __name__ == '__main__':
    # run_model()
    # run_batch_runner(10)
    # get_avg_simulation()
    run_analyzer(10)

    # Visualizer.average_vs_deterministic(fixed_params)
    # Visualizer.stochasticity(fixed_params)
    # Visualizer.beta_gamma_distribution(fixed_params)




