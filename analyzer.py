from model import Model
from result_manager import ResultManager
from batch_runner import BatchRunner
from visualization import Visualizer

from typing import Type


class Analyzer:
    def __init__(
            self,
            model_cls: Type[Model],
            fixed_params: dict,
            variable_params: dict,
            iterations: int,
    ):
        self.Model = model_cls
        self.fixed_params = fixed_params
        self.variable_params = variable_params
        self.iterations = iterations

    def _run_batch_runner(self) -> list[dict]:
        model_params = BatchRunner(
            model_cls=self.Model,
            fixed_params=self.fixed_params,
            variable_params=self.variable_params,
            iterations=self.iterations,
        ).run()
        return model_params

    @staticmethod
    def _make_plots(model_params: list[dict]) -> None:
        for params in model_params:
            fig = Visualizer().average_vs_deterministic(params)
            ResultManager.save_plot(fig=fig, params=params,
                                    fname='average_vs_deterministic')

            fig = Visualizer().stochasticity(params)
            ResultManager.save_plot(fig=fig, params=params,
                                    fname='stochasticity')

            fig = Visualizer().beta_gamma_distribution(params)
            ResultManager.save_plot(fig=fig, params=params,
                                    fname='beta_gamma_distribution')

    def run(self) -> None:
        model_params = self._run_batch_runner()
        self._make_plots(model_params=model_params)





