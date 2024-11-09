from turtledemo.penrose import start
from typing import Optional, Literal
import scipy.optimize as opt
from sklearn.metrics import mean_squared_error as mse
import numpy as np
import pandas as pd
import numpy.typing as npt
from .model import EpidemicModel


class ModelFitterError(Exception):
    pass

class ModelFitter:
    # method = 'Nelder-Mead'

    def __init__(self, model: EpidemicModel):
        self._model = model

        self._changeable_stages: list[str] = model.stage_names
        self._changeable_factors: list[str] = model.factor_names

        self._real_flows_df: Optional[pd.DataFrame] = None

    def set_changeable_stages(self, changeable_stages: list[str] | Literal['all', 'none']):
        if changeable_stages == 'all':
            self._changeable_stages = self._model.stage_names
            return
        elif changeable_stages == 'none':
            self._changeable_stages = []
            return

        for stage_name in changeable_stages:
            if stage_name not in self._model.stage_names:
                raise ModelFitterError(f'Stage {stage_name} not found in the model')
        self._changeable_stages = changeable_stages

    def set_changeable_factors(self, changeable_factors: list[str] | Literal['all', 'none']):
        if changeable_factors == 'all':
            self._changeable_factors = self._model.factor_names
            return
        elif changeable_factors == 'none':
            self._changeable_factors = []
            return

        for factor_name in changeable_factors:
            if factor_name not in self._model.factor_names:
                raise ModelFitterError(f'Factor {factor_name} not found in the model')
        self._changeable_factors = changeable_factors

    def fit(self, real_flows_df: pd.DataFrame):
        not_existing_flows = set(real_flows_df.columns) - set(self._model.flows_df.columns)
        if not_existing_flows:
            raise ModelFitterError(f'Flows {not_existing_flows} not found in the model')
        self._real_flows_df = real_flows_df

        stages = [st['num'] for st in self._model.stages if st['name'] in self._changeable_stages]
        factors = [fa['value'] for fa in self._model.factors if fa['name'] in self._changeable_factors]
        parameters = np.array(stages + factors, dtype=np.float64)
        bounds = [(0, self._model.population_size)] * len(stages) + [(0, 1)] * len(factors)
        return opt.minimize(self._get_mse, parameters, method='Nelder-Mead', bounds=bounds)


    def _get_mse(self, parameters: npt.NDArray[np.float64]):
        # добавить к mse большой штраф, за нарушение суммы численностей всех стадий
        # например, куб отклонения от объёма популяции

        start_stages = {stage_name: parameters[i] for i, stage_name in enumerate(self._changeable_stages)}
        factors = {factor_name: parameters[i] for i, factor_name in
                   enumerate(self._changeable_factors, start=len(self._changeable_stages))}

        self._model.set_start_stages(**start_stages)
        self._model.set_factors(**factors)

        self._model.start(len(self._real_flows_df) + 1, full_save=True)

        result_mse = mse(self._model.flows_df[self._real_flows_df.columns][:-1], self._real_flows_df)
        print(f'MSE = {result_mse}')
        return result_mse




