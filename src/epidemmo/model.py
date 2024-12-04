from typing import Optional, Callable

import numpy as np

import pandas as pd
from itertools import product

from matplotlib import pyplot as plt
from prettytable import PrettyTable
from scipy.stats import poisson
from datetime import datetime

from .flow import Flow
from .stage import Stage
from .factor import Factor


class ModelError(Exception):
    pass


class EpidemicModel:
    __len_float: int = 4

    def __init__(self, name: str, stages: list[Stage], flows: list[Flow], relativity_factors: bool):
        self._name: str = name

        stages = sorted(stages, key=lambda st: st.index)  # сортируем стадии по индексу
        flows = sorted(flows, key=lambda fl: fl.index)  # сортируем потоки по индексу

        self._stages: tuple[Stage, ...] = tuple(stages)
        self._flows: tuple[Flow, ...] = tuple(flows)
        self._factors: tuple[Factor, ...] = tuple(set(fa for fl in flows for fa in fl.get_factors()))

        self._stage_names: tuple[str, ...] = tuple(st.name for st in stages)
        self._flow_names: tuple[str, ...] = tuple(str(fl) for fl in flows)
        self._factors_names: tuple[str, ...] = tuple(fa.name for fa in self._factors)

        self._stage_starts: np.ndarray = np.array([st.start_num for st in stages], dtype=np.float64)

        # факторы, которые будут изменяться во время моделирования
        self._dynamic_factors: list[Factor] = [fa for fa in self._factors if fa.is_dynamic]

        self._flows_weights: np.ndarray = np.zeros(len(flows), dtype=np.float64)
        self._targets: np.ndarray = np.zeros((len(flows), len(stages)), dtype=np.float64)
        self._induction_weights: np.ndarray = np.zeros((len(flows), len(stages)), dtype=np.float64)
        self._outputs: np.ndarray = np.zeros((len(flows), len(stages)), dtype=np.bool_)

        # связываем факторы, используемые в потоках, с матрицами
        self._connect_matrix(flows)

        self._duration = 1
        self._result: np.ndarray = np.zeros((self._duration, len(stages)), dtype=np.float64)
        self._result[0] = self._stage_starts
        self._result_flows: Optional[np.ndarray] = None
        self._result_factors: Optional[np.ndarray] = None
        self._confidence: Optional[np.ndarray] = None

        self._flows_probabs: np.ndarray = np.zeros(len(flows), dtype=np.float64)
        self._flows_values: np.ndarray = np.zeros(len(flows), dtype=np.float64)
        self._induction_mask: np.ndarray = self._induction_weights.any(axis=1)
        self._induction: np.ndarray = self._induction_weights[self._induction_mask]

        self._relativity_factors: bool = False
        self.set_relativity_factors(relativity_factors)

    @property
    def population_size(self) -> int:
        """
        :return: Размер популяции
        """
        return self._stage_starts.sum()

    def set_relativity_factors(self, relativity_factors: bool):
        """
        :param relativity_factors: относительные ли факторы (относительные - не будут делиться на размер популяции)
        :return:
        """
        if not isinstance(relativity_factors, bool):
            raise ModelError('relativity_factors must be bool')
        for fl in self._flows:
            fl.set_relativity_factors(relativity_factors)
        self._relativity_factors = relativity_factors

    def _update_all_factors(self):
        for fa in self._factors:
            fa.update(0)

    def _update_dynamic_factors(self, step: int):
        for fa in self._dynamic_factors:
            fa.update(step-1)

    def _prepare_factors_matrix(self, *args):
        self._iflow_weights = self._flows_weights[self._induction_mask].reshape(-1, 1)
        self._flows_probabs[~self._induction_mask] = self._flows_weights[~self._induction_mask]
        self._check_matrix()

    def _check_matrix(self):
        if (self._targets.sum(axis=1) != 1).any():
            raise ModelError('Sum of targets one Flow must be 1')
        if (self._flows_weights < 0).any():
            raise ModelError('Flow weights must be >= 0')
        if (self._flows_weights[~self._induction_mask] > 1).any():
            raise ModelError('Not Induction Flow weights must be in range [0, 1]')
        if self._relativity_factors and (self._flows_weights[self._induction_mask] > 1).any():
            raise ModelError('Induction Flow weights, if they are relativity, must be in range [0, 1]')
        if ((self._induction_weights < 0) | (self._induction_weights > 1)).any():
            raise ModelError('Induction weights must be in range [0, 1]')

    def _correct_not_rel_factors(self, *args):
        self._flows_weights[self._induction_mask] /= self.population_size

    def _connect_matrix(self, flows: list[Flow]):
        for fl in flows:
            fl.connect_matrix(self._flows_weights, self._targets, self._induction_weights, self._outputs)

    def _prepare(self):
        self._update_all_factors()
        if not self._relativity_factors:
            self._correct_not_rel_factors()
        self._prepare_factors_matrix()

    def start(self, duration: int, *, full_save: bool = False, stochastic: bool = False,
              get_cis: bool = False, num_cis_starts: int = 100, cis_significance: float = 0.05) -> pd.DataFrame:
        """
        Запускает модель и возвращает таблицу результатов
        :param duration: длительность моделирования
        :param full_save: вычислить ли все результаты (+потоки, +факторы)
        :param stochastic: запускать ли модель в стохастическом режиме
        :param get_cis: вычислить ли доверительные интервалы
        :param num_cis_starts: количество стохастических запусков для вычисления доверительных интервалов
        :param cis_significance: уровень значимости для доверительных интервалов
        :return: таблица результатов, столбцы - стадии, строки - шаги моделирования
        """
        if not isinstance(duration, int) or duration < 1:
            raise ModelError('duration must be int > 1')
        if not isinstance(full_save, bool):
            raise ModelError('full_save must be bool')
        if not isinstance(stochastic, bool):
            raise ModelError('stochastic must be bool')

        self._result_flows = None
        self._result_factors = None
        self._confidence = None
        self._duration = duration

        if get_cis:
            self._get_confidence_intervals(num_cis_starts, cis_significance)

        self._start(full_save, stochastic)
        return self._get_result_df()

    def _start(self, save_full: bool, stochastic: bool):
        self._result = np.zeros((self._duration, len(self._stage_starts)), dtype=np.float64)
        self._result[0] = self._stage_starts

        self._prepare()

        if not self._dynamic_factors and not save_full and not stochastic:
            self._fast_run()
            return

        self._full_step_seq: list[Callable] = []
        if self._dynamic_factors:
            self._full_step_seq.append(self._update_dynamic_factors)
            if not self._relativity_factors:
                self._full_step_seq.append(self._correct_not_rel_factors)
            self._full_step_seq.append(self._prepare_factors_matrix)

        if stochastic:
            self._full_step_seq.append(self._stoch_step)
        else:
            self._full_step_seq.append(self._determ_step)

        if save_full:
            self._full_step_seq.append(self._save_additional_results)
            self._result_flows = np.full((self._duration, len(self._flow_names)), np.nan,  dtype=np.float64)
            self._result_factors = np.full((self._duration, len(self._factors_names)), np.nan, dtype=np.float64)
        else:
            self._result_flows = None
            self._result_factors = None

        if stochastic:
            self._stoch_run()
        else:
            self._determ_run()

    def _several_stoch_starts(self, count: int):
        all_results = np.zeros((count, self._duration, len(self._stage_starts)), dtype=np.float64)
        all_results[:, 0, :] = self._stage_starts

        self._full_step_seq: list[Callable] = []
        if self._dynamic_factors:
            self._full_step_seq.append(self._update_dynamic_factors)
            if not self._relativity_factors:
                self._full_step_seq.append(self._correct_not_rel_factors)
            self._full_step_seq.append(self._prepare_factors_matrix)
        self._full_step_seq.append(self._stoch_step)

        for i in range(count):
            self._result = all_results[i]
            self._prepare()
            self._stoch_run()

        return all_results

    def _get_confidence_intervals(self, num_starts: int, alpha_cis: float = 0.05):
        if not isinstance(num_starts, int) or num_starts < 1:
            raise ModelError('Number of starts for calculating confidence intervals must be int > 1')
        if not isinstance(alpha_cis, float) or alpha_cis < 0 or alpha_cis > 1:
            raise ModelError(f'Alpha for calculating confidence intervals must be float in range [0, 1], but got {alpha_cis}')

        results = self._several_stoch_starts(num_starts) # для получения набора стохастических результатов
        self._start(False, False) # для получения теоретического результата в self._result
        self._confidence = self._calc_confidence_intervals(results, self._result, alpha_cis)

    @staticmethod
    def _calc_confidence_intervals(stoch_results: np.array, mid_result: np.array, alpha_cis: float = 0.05):
        down_limit = alpha_cis * 100
        up_limit = (1 - alpha_cis) * 100
        num_runs, duration, num_stages = stoch_results.shape
        confidence = np.zeros((duration, num_stages * 2))
        for st_i in range(num_stages):
            for time in range(duration):
                low_results = stoch_results[stoch_results[:, time, st_i] < mid_result[time, st_i], time, st_i]
                if len(low_results):
                    conf_low = np.percentile(low_results, [down_limit], axis=0)[0]
                else:
                    conf_low = mid_result[time, st_i]

                up_results = stoch_results[stoch_results[:, time, st_i] > mid_result[time, st_i], time, st_i]
                if len(up_results):
                    conf_up = np.percentile(up_results, [up_limit], axis=0)[0]
                else:
                    conf_up = mid_result[time, st_i]

                confidence[time, st_i * 2: st_i * 2 + 2] = [conf_low, conf_up]
        return confidence

    def _determ_run(self):
        for step in range(1, self._duration):
            self._result[step] = self._result[step - 1]
            for step_func in self._full_step_seq:
                step_func(step)

    def _stoch_run(self):
        step = 0
        shift = poisson.rvs(mu=1)
        self._result[step: step + shift + 1] = self._result[step]
        step = step + shift
        while step < self._duration:
            for step_func in self._full_step_seq:
                step_func(step)
            shift = poisson.rvs(mu=1)
            self._result[step: step + shift + 1] = self._result[step]
            step = step + shift

    def _fast_run(self):
        for step in range(1, self._duration):
            pr = self._result[step - 1]
            self._induction * self._iflow_weights
            self._flows_probabs[self._induction_mask] = 1 - ((1 - self._induction * self._iflow_weights) ** pr).prod(axis=1)

            for st_i in range(len(self._stage_starts)):
                self._flows_probabs[self._outputs[:, st_i]] = self._correct_p(
                    self._flows_probabs[self._outputs[:, st_i]])
                self._flows_values[self._outputs[:, st_i]] = self._flows_probabs[self._outputs[:, st_i]] * pr[st_i]
            changes = self._flows_values @ self._targets - self._flows_values @ self._outputs
            self._result[step] = pr + changes

    def _determ_step(self, step: int):
        self._flows_probabs[self._induction_mask] = 1 - ((1 - self._induction * self._iflow_weights) **
                                                         self._result[step]).prod(axis=1)

        for st_i in range(len(self._stage_starts)):
            self._flows_probabs[self._outputs[:, st_i]] = self._correct_p(self._flows_probabs[self._outputs[:, st_i]])
            self._flows_values[self._outputs[:, st_i]] = self._flows_probabs[self._outputs[:, st_i]] * \
                                                         self._result[step][st_i]
        changes = self._flows_values @ self._targets - self._flows_values @ self._outputs
        self._result[step] += changes

    def _stoch_step(self, step: int):
        self._flows_probabs[self._induction_mask] = 1 - ((1 - self._induction * self._iflow_weights) **
                                                         self._result[step]).prod(axis=1)

        for st_i in range(len(self._stage_starts)):
            self._flows_probabs[self._outputs[:, st_i]] = self._correct_p(self._flows_probabs[self._outputs[:, st_i]])
            flow_values = self._flows_probabs[self._outputs[:, st_i]] * self._result[step][st_i]
            flow_values = poisson.rvs(flow_values, size=len(flow_values))
            flows_sum = flow_values.sum()
            if flows_sum > self._result[step][st_i]:
                # находим избыток всех потоков ушедших из стадии st_i
                # распределим (вычтем) этот избыток из всех потоков пропорционально значениям потоков
                excess = flows_sum - self._result[step][st_i]
                flow_values = flow_values - flow_values / flows_sum * excess
            self._flows_values[self._outputs[:, st_i]] = flow_values
        changes = self._flows_values @ self._targets - self._flows_values @ self._outputs
        self._result[step] += changes

    def _save_additional_results(self, step: int):
        self._result_flows[step-1] = self._flows_values
        self._result_factors[step-1] = [fa.value for fa in self._factors]

    @classmethod
    def _get_table(cls, table_df: pd.DataFrame) -> PrettyTable:
        table = PrettyTable()
        table.add_column('step', table_df.index.tolist())
        for col in table_df:
            col_name = f'{col[0]}_{col[1]}' if isinstance(col, tuple) else str(col)
            table.add_column(col_name, table_df[col].tolist())
        table.float_format = f".{cls.__len_float}"
        return table

    def _get_result_df(self) -> pd.DataFrame:
        result = pd.DataFrame(self._result, columns=self._stage_names)
        return result.reindex(np.arange(self._duration), method='ffill')

    def _get_factors_df(self) -> pd.DataFrame:
        factors = pd.DataFrame(self._result_factors, columns=[fa.name for fa in self._factors])
        return factors

    def _get_flows_df(self) -> pd.DataFrame:
        flows = pd.DataFrame(self._result_flows, columns=self._flow_names)
        flows.fillna(0, inplace=True)
        return flows

    def _get_full_df(self) -> pd.DataFrame:
        return pd.concat([self._get_result_df(), self._get_flows_df(),
                          self._get_factors_df(), self._get_conf_df()], axis=1)

    def _get_conf_df(self):
        col_names = [(st_name, limit) for st_name in self._stage_names for limit in ['lower', 'upper']]
        index = pd.MultiIndex.from_tuples(col_names, names=['stage', 'limit'])
        conf_df = pd.DataFrame(self._confidence, columns=index)
        return conf_df.reindex(np.arange(self._duration))

    @property
    def result_df(self) -> pd.DataFrame:
        """
        :return: Таблица результатов, численность каждой стадии во времени
        """
        return self._get_result_df()

    @property
    def full_df(self) -> pd.DataFrame:
        """
        :return: Таблица результатов, численность каждой стадии во времени,
        """
        return self._get_full_df()

    @property
    def flows_df(self) -> pd.DataFrame:
        """
        :return: Таблица потоков, интенсивность (численность) каждого потока (перехода) во времени
        """
        return self._get_flows_df()

    @property
    def factors_df(self):
        """
        :return: Таблица факторов (параметров модели), значение каждого фактора во времени
        """
        return self._get_factors_df()

    @property
    def confidence_df(self):
        """
        :return: Таблица доверительных интервалов, верхняя и нижняя границы для численности каждой стадии во времени
        """
        return self._get_conf_df()

    def print_result_table(self) -> None:
        print(self._get_table(self._get_result_df()))

    def print_factors_table(self) -> None:
        print(self._get_table(self._get_factors_df()))

    def print_flows_table(self) -> None:
        print(self._get_table(self._get_flows_df()))

    def print_full_table(self) -> None:
        print(self._get_table(self._get_full_df()))

    @property
    def name(self) -> str:
        """
        :return: Название модели
        """
        return self._name

    def _write_table(self, filename: str, table: pd.DataFrame, floating_point='.', delimiter=',') -> None:
        table.to_csv(filename, sep=delimiter, decimal=floating_point,
                     float_format=f'%.{self.__len_float}f', index_label='step')

    def write_results(self, floating_point='.', delimiter=',', path: str = '',
                      write_flows: bool = False, write_factors: bool = False) -> None:
        """
        Сохраняет результаты модели в csv-файл
        :param floating_point: десятичная точка
        :param delimiter: разделитель в таблице
        :param path: путь для сохранения
        :param write_flows: сохранять ли столбцы с результатами по потокам
        :param write_factors: сохранять ли столбцы с результатами по факторам
        """
        if path and path[-1] != '\\':
            path = path + '\\'

        current_time = datetime.today().strftime('%d_%b_%y_%H-%M-%S')
        filename = f'{path}{self._name}_result_{current_time}.csv'

        tables = [self._get_result_df()]
        if write_flows:
            if self._result_flows is None:
                print('Warning: Results for flows should be saved while model is running')
            else:
                tables.append(self._get_flows_df())
        if write_factors:
            if self._result_factors is None:
                print('Warning: Results for factors should be saved while model is running')
            else:
                tables.append(self._get_factors_df())
        final_table = pd.concat(tables, axis=1)
        self._write_table(filename, final_table, floating_point, delimiter)

    def set_factors(self, **kwargs) -> None:
        for f in self._factors:
            if f.name in kwargs:
                f.set_fvalue(kwargs[f.name])

        self._dynamic_factors = [f for f in self._factors if f.is_dynamic]

    def set_start_stages(self, **kwargs) -> None:
        for s_index, s  in enumerate(self._stages):
            if s.name in kwargs:
                s.start_num = kwargs[s.name]
                self._stage_starts[s_index] = kwargs[s.name]

    def __str__(self) -> str:
        return f'Model({self._name})'

    def __repr__(self) -> str:
        return f'Model({self._name}): {list(self._flows)}'

    @property
    def stages(self) -> list[dict[str, float]]:
        return [{'name': st.name, 'num': float(st.start_num)} for st in self._stages]

    @property
    def stages_dict(self) -> dict[str, float]:
        return {st.name: float(st.start_num) for st in self._stages}

    @property
    def stage_names(self) -> list[str]:
        return list(self._stage_names)

    @property
    def factors(self) -> list[dict[str, float]]:
        return [{'name': fa.name, 'value': 'dynamic' if fa.is_dynamic else fa.value} for fa in self._factors]

    @property
    def factors_dict(self) -> dict[str, float | str]:
        return {fa.name: 'dynamic' if fa.is_dynamic else fa.value for fa in self._factors}

    @property
    def factor_names(self) -> list[str]:
        return list(self._factors_names)

    @property
    def flows(self) -> list[dict]:
        flows = []
        for fl in self._flows:
            fl_dict = {'start': fl.start.name, 'factor': fl.factor.name,
                       'end': {st.name: fa.name for st, fa in fl.ends.items()},
                       'inducing': {st.name: fa.name for st, fa in fl.inducing.items()}}
            flows.append(fl_dict)
        return flows

    @property
    def flow_names(self) -> list[str]:
        return list(self._flow_names)

    def get_latex(self, simplified: bool = False) -> str:
        for fl in self._flows:
            fl.send_latex_terms(simplified)

        tab = '    '
        system_of_equations = f"\\begin{{equation}}\\label{{eq:{self._name}_{'classic' if simplified else 'full'}}}\n"
        system_of_equations += f'{tab}\\begin{{cases}}\n'

        for st in self._stages:
            system_of_equations += f'{tab * 2}{st.get_latex_equation()}\\\\\n'

        system_of_equations += f'{tab}\\end{{cases}}\n'
        system_of_equations += f'\\end{{equation}}\n'

        for st in self._stages:
            st.clear_latex_terms()

        return system_of_equations

    def plot(self, ax: plt.Axes = None) -> plt.Axes:
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        self._plot(ax)
        return ax

    def _plot(self, ax: plt.Axes) -> None:
        result = self._get_result_df()
        colors = []

        for sname in self._stage_names:
            p = ax.plot(result[sname], label=sname)
            colors.append(p[0].get_color())

        if self._confidence is not None:
            conf = self._get_conf_df()
            for stage, color in zip(self._stage_names, colors):
                ax.fill_between(conf.index, conf[(stage, 'lower')], conf[(stage, 'upper')], color=color, alpha=0.2,
                                label=f'доверительный интервал для {stage}')

        ax.set_title(self._name)
        ax.set_xlabel('время')
        ax.set_ylabel('количество индивидов')
        ax.grid()
        ax.legend()

    @staticmethod
    def _correct_p(probs: np.ndarray) -> np.ndarray:
        # return probs
        # матрица случившихся событий
        happened_matrix = np.array(list(product([0, 1], repeat=len(probs))), dtype=np.bool_)

        # вектор вероятностей каждого сценария
        # те что свершились с исходной вероятностью, а не свершились - (1 - вероятность)
        full_probs = (probs * happened_matrix + (1 - probs) * (~happened_matrix)).prod(axis=1)

        # делим на то сколько событий произошло, в каждом сценарии
        # в первом случае ни одно событие не произошло, значит делить придётся на 0
        # а случай этот не пригодится
        full_probs[1:] = full_probs[1:] / happened_matrix[1:].sum(axis=1)

        # новые вероятности
        # по сути сумма вероятностей сценариев, в которых нужное событие произошло
        new_probs = np.zeros_like(probs)
        for i in range(len(probs)):
            new_probs[i] = full_probs[happened_matrix[:, i]].sum()
        return new_probs



