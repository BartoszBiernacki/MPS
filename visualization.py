import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from result_manager import ResultManager


class Visualizer:

    _sir_colors = ['tab:green', 'tab:orange', 'tab:red']

    @staticmethod
    def _is_column_single_value(df: pd.DataFrame, col_name: str) -> bool:
        return (df[col_name] == df[col_name][0]).all()

    @classmethod
    def _plot_dirac_delta(cls, ax: plt.Axes, x: float) -> None:
        ax.set_xlim(0, 2 * x)

        # Add arrow annotation
        arrow_props = dict(arrowstyle='->', lw=2, mutation_scale=30)
        ax.annotate('', xy=(x, 1), xytext=(x, 0), arrowprops=arrow_props)

    @staticmethod
    def _text_to_latex(text: str) -> str:
        text = text.replace('beta', r'$\beta$')
        text = text.replace('gamma', r'$\gamma$')
        text = text.replace('mu', r'$\mu$')
        text = text.replace('sigma', r'$\sigma$')
        text = text.replace('=', r'$=$')
        text = text.replace('+', r'$+$')

        return text

    @classmethod
    def _annotate_params(cls, ax: plt.Axes, params: dict,
                         xy: tuple[float, float] = (0.51, 0.65)) -> None:
        line1 = f'beta = {params["beta"]} + {params["noise_beta"]}'
        line1 = cls._text_to_latex(line1)

        line2 = f'gamma = {params["gamma"]} + {params["noise_gamma"]}'
        line2 = cls._text_to_latex(line2)

        line3 = r'$I_0 = $' + f'{params["I0"]}'

        text = '\n'.join([line1, line2, line3])

        ax.annotate(
            text=text,
            xy=xy,
            xycoords='axes fraction',
            bbox=dict(facecolor='none', edgecolor='grey', alpha=0.2, pad=5),
        )

    @classmethod
    def average_vs_deterministic(cls, params: dict) -> plt.Figure:

        def _add_legend():
            from matplotlib.lines import Line2D
            color_lines = [Line2D([0], [0], color=col)
                           for col in cls._sir_colors]
            style_lines = [Line2D([0], [0], color='k', ls=ls)
                           for ls in linestyles]

            leg1 = ax.legend(color_lines, ['S', 'I', 'R'], title='Statystyka',
                             bbox_to_anchor=(0.85, 0.55))
            ax.add_artist(leg1)

            leg2 = ax.legend(style_lines, ['Tak', 'Nie'], title='Szum',
                             bbox_to_anchor=(1, 0.55))
            ax.add_artist(leg2)

        def _add_axis_labels():
            ax.set_title('Przebieg epidemii według modelu SIR')
            ax.set_xlabel('Czas')
            ax.set_ylabel('Populacja')

        avg_df = ResultManager().average_simulation(params)
        det_df = ResultManager().deterministic_simulation(params)

        fig, ax = plt.subplots()
        sns.despine()

        linestyles = ['--', '-']
        for stat, c in zip('SIR', cls._sir_colors):
            ax.plot(avg_df['t'], avg_df[stat], c=c, ls=linestyles[0])
            ax.plot(det_df['t'], det_df[stat], c=c, ls=linestyles[1])

        _add_axis_labels()
        cls._annotate_params(ax=ax, params=params, xy=(0.51, 0.6))
        _add_legend()

        plt.tight_layout()
        return fig

    @classmethod
    def stochasticity(cls, params: dict) -> plt.Figure:

        def _add_legend():
            from matplotlib.lines import Line2D
            color_lines = [Line2D([0], [0], color=col)
                           for col in cls._sir_colors]

            ax.legend(color_lines, ['S', 'I', 'R'], title='Statystyka')

        def _add_axis_labels():
            ax.set_title(f'Stochastyczność modelu SIR na przestrzeni '
                         f'{len(all_dfs)} symulacji')
            ax.set_xlabel('Czas')
            ax.set_ylabel('Populacja')

        fig, ax = plt.subplots()
        sns.despine()

        all_dfs = ResultManager().all_simulations(params)
        for df in all_dfs:
            for stat, c in zip('SIR', cls._sir_colors):
                ax.plot(df['t'], df[stat], c=c, lw=1)

        _add_axis_labels()
        cls._annotate_params(ax=ax, params=params)
        _add_legend()

        plt.tight_layout()
        return fig

    @classmethod
    def beta_gamma_distribution(cls, params: dict) -> plt.Figure:

        def _hist_plot(ax: plt.Axes, param: str) -> None:
            sns.despine(ax=ax)

            if cls._is_column_single_value(df=df, col_name=param):
                cls._plot_dirac_delta(ax=ax, x=df[param][0])
                ax.set_xlabel(param)
            else:
                sns.histplot(data=df, x=param, stat='density', ax=ax)

            ax.set_xlabel('Parametr ' + cls._text_to_latex(param))
            ax.set_ylabel('Gęstość\nprawdopodobieństwa')

        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)

        title = 'Rozkład prawdopodobieństwa parametrów beta oraz gamma'
        ax1.set_title(cls._text_to_latex(title))

        all_dfs = ResultManager().all_simulations(params)
        df = pd.concat(all_dfs, ignore_index=True)

        _hist_plot(ax=ax1, param='beta')
        _hist_plot(ax=ax2, param='gamma')

        plt.tight_layout()
        return fig





