from typing import Optional, Literal

import scipy.optimize as opt
from scipy.ndimage import uniform_filter1d
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
        if not isinstance(model, EpidemicModel):
            raise ModelFitterError('Model must be EpidemicModel')

        self._model = model
        self._population_size = model.population_size
        self._changeable_stages: dict[str, tuple[float, float]] = {}
        self._changeable_factors: dict[str, tuple[float, float]] = {}
        self._real_flows_df: Optional[pd.DataFrame] = None
        self._interval: int = 1
        self._mse_history: list[float] = []

    @staticmethod
    def _check_bounds_dict(bounds_dict: dict[str, tuple[float, float]], names: list[str]):
        for name, bounds in bounds_dict.items():
            if name not in names:
                raise ModelFitterError(f'Name {name} not found in the model')
            match bounds:
                case int(left) | float(left), int(right) | float(right):
                    pass
                case _:
                    raise ModelFitterError(f'Bounds must be a tuple of floats or ints, got {bounds}')

    def set_changeable_stages(self, changeable_stages: dict[str, tuple[float, float]] | Literal['all', 'none']):
        if changeable_stages == 'all':
            self._changeable_stages = {name: (0, self._population_size) for name in self._model.stage_names}
            return
        elif changeable_stages == 'none':
            self._changeable_stages = {}
            return
        elif isinstance(changeable_stages, dict):
            self._check_bounds_dict(changeable_stages, self._model.stage_names)
            self._changeable_stages = changeable_stages.copy()
        else:
            raise ModelFitterError(f'Changeable stages must be a dict or "all" or "none", got {changeable_stages}')

    def set_changeable_factors(self, changeable_factors: dict[str, tuple[float, float]] | Literal['all', 'none']):
        if changeable_factors == 'all':
            self._changeable_factors = {name: (0.0, 1.0) for name in self._model.factor_names}
            return
        elif changeable_factors == 'none':
            self._changeable_factors = {}
            return
        elif isinstance(changeable_factors, dict):
            self._check_bounds_dict(changeable_factors, self._model.factor_names)
            self._changeable_factors = changeable_factors.copy()
        else:
            raise ModelFitterError(f'Changeable factors must be a dict or "all" or "none", got {changeable_factors}')

    def fit(self, real_flows_df: pd.DataFrame, interval: int = 1):
        if not self._changeable_stages and not self._changeable_factors:
            raise ModelFitterError('No stages or factors are changeable')
        if not isinstance(interval, int) or interval < 1:
            raise ModelFitterError('Interval must be int > 1')

        self._interval = interval

        not_existing_flows = set(real_flows_df.columns) - set(self._model.flow_names)
        if not_existing_flows:
            raise ModelFitterError(f'Flows {not_existing_flows} not found in the model')

        self._real_flows_df = real_flows_df.copy()

        window_wide = int(len(self._real_flows_df) / 20)
        if window_wide > 1 and self._interval == 1:
            for col in self._real_flows_df.columns:
                self._real_flows_df[col] = uniform_filter1d(self._real_flows_df[col], size=window_wide)

        param_start = []
        bounds = []
        for st in filter(lambda s: s['name'] in self._changeable_stages, self._model.stages):
            param_start.append(st['num'])
            bounds.append(self._changeable_stages[st['name']])
        for fa in filter(lambda f: f['name'] in self._changeable_factors, self._model.factors):
            param_start.append(fa['value'])
            bounds.append(self._changeable_factors[fa['name']])
        param_start = np.array(param_start, dtype=np.float64)
        self._mse_history = []
        result = opt.minimize(self._get_mse, param_start, method='Nelder-Mead', bounds=bounds)
        self._get_mse(result.x)
        return result

    def _get_mse(self, parameters: npt.NDArray[np.float64]):
        parameters = np.abs(parameters)
        start_stages = {stage_name: parameters[i] for i, stage_name in enumerate(self._changeable_stages)}
        factors = {factor_name: parameters[i] for i, factor_name in
                   enumerate(self._changeable_factors, start=len(self._changeable_stages))}

        self._model.set_start_stages(**start_stages)
        self._model.set_factors(**factors)

        self._model.start(len(self._real_flows_df) + 1, delta=self._interval, full_save=True)

        result_mse = mse(self._model.flows_df[self._real_flows_df.columns], self._real_flows_df)

        self._mse_history.append(result_mse)
        return result_mse

