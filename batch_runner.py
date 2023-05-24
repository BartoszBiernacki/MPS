import itertools
from typing import Type

from model import Model
from result_manager import ResultManager
from noise import ZeroNoise


class BatchRunner:
    def __init__(
            self,
            model_cls: Type[Model],
            fixed_params: dict,
            variable_params: dict,
            iterations: int,
    ):
        self.Model = model_cls
        self.iterations = iterations
        self.fixed_params = fixed_params
        self.variable_params = variable_params

    @staticmethod
    def generate_combinations(variable_params):
        keys = variable_params.keys()
        value_lists = variable_params.values()
        combinations = list(itertools.product(*value_lists))
        result = [dict(zip(keys, combination)) for combination in combinations]

        return tuple(result)

    def run(self) -> list[dict]:
        combinations = self.generate_combinations(self.variable_params)

        total_num_of_simulations = len(combinations) * (self.iterations + 1)
        current_simulation_number = 1

        used_model_params = []
        for combination in combinations:
            model_params = self.fixed_params | combination
            used_model_params.append(model_params)
            for i in range(self.iterations):
                print(f'Simulation {current_simulation_number} '
                      f'out of {total_num_of_simulations} ...')

                result = self.Model(**model_params).run()
                ResultManager.save_simulation(params=model_params, df=result)

                current_simulation_number += 1

            zero_noise = ZeroNoise()
            model_params = (
                    self.fixed_params |
                    combination |
                    {'noise_beta': zero_noise, 'noise_gamma': zero_noise}
            )

            result = self.Model(**model_params).run()
            ResultManager.save_simulation(
                params=model_params, df=result, deterministic=True)

            current_simulation_number += 1

        return used_model_params
