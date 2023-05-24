import os
import pandas as pd
import matplotlib.pyplot as plt

from noise import ZeroNoise


class ResultManager:

    @staticmethod
    def _dict_to_folder_name(dictionary: dict) -> str:
        pairs = [f'{key}={value}' for key, value in dictionary.items()]
        return ','.join(pairs)

    @classmethod
    def _dict_to_folder_path(cls, dictionary: dict) -> str:
        folder_name = cls._dict_to_folder_name(dictionary)
        folder_path = f'./RESULTS/{folder_name}'

        return folder_path

    @classmethod
    def save_simulation(
            cls, params: dict, df: pd.DataFrame,
            deterministic: bool = False) -> None:

        folder_path = cls._dict_to_folder_path(params)
        os.makedirs(folder_path, exist_ok=True)

        if deterministic:
            file_path = f'{folder_path}/deterministic.csv'
        else:
            files = os.listdir(folder_path)
            file_number = len(files) + 1
            file_path = f'{folder_path}/{file_number}.csv'

        df.to_csv(file_path, index=False, float_format='%.4f')

    @staticmethod
    def _read_simulations(folder_path: str,
                          deterministic: bool) -> list[pd.DataFrame]:
        if not os.path.exists(folder_path):
            raise ValueError(f'Cannot find folder {folder_path = }')
        file_path_list = os.listdir(folder_path)
        csv_files = [file for file in file_path_list if file.endswith('.csv')]

        dataframes = []
        for file_name in csv_files:
            if 'deterministic' not in file_name and not deterministic:
                file_path = os.path.join(folder_path, file_name)
                df = pd.read_csv(file_path)
                dataframes.append(df)

            elif 'deterministic' in file_name and deterministic:
                file_path = os.path.join(folder_path, file_name)
                df = pd.read_csv(file_path)
                dataframes.append(df)

        return dataframes

    @staticmethod
    def _average_dataframes(dataframes: list[pd.DataFrame]) -> pd.DataFrame:
        if not dataframes:
            raise ValueError("Empty list of DataFrames")

        first_df = dataframes[0]
        average_df = first_df.copy()

        for df in dataframes[1:]:
            average_df += df

        average_df /= len(dataframes)

        return average_df

    @classmethod
    def all_simulations(cls, params: dict) -> list[pd.DataFrame]:
        folder_path = cls._dict_to_folder_path(params)
        dataframes = cls._read_simulations(folder_path, deterministic=False)
        return dataframes

    @classmethod
    def average_simulation(cls, params: dict) -> pd.DataFrame:
        dataframes = cls.all_simulations(params)
        avg_result = cls._average_dataframes(dataframes)

        return avg_result

    @classmethod
    def deterministic_simulation(cls, params: dict) -> pd.DataFrame:
        no_noise = {'noise_beta':  ZeroNoise(), 'noise_gamma': ZeroNoise()}
        folder_path = cls._dict_to_folder_path(params | no_noise)
        dataframes = cls._read_simulations(folder_path, deterministic=True)

        if len(dataframes) == 0:
            raise ValueError(
                f"No deterministic simulation found for {params = }")

        return dataframes[0]

    @classmethod
    def save_plot(cls, fig: plt.Figure, params: dict,  fname: str) -> None:
        folder_path = cls._dict_to_folder_path(params)
        os.makedirs(folder_path, exist_ok=True)

        if not fname.endswith('.pdf'):
            fname += '.pdf'
        file_path = f'{folder_path}/{fname}'
        fig.savefig(fname=file_path)
