from __future__ import annotations
from typing import Callable, Optional
from types import FunctionType


class FactorError(Exception):
    pass


class Factor:
    def __init__(self, value: int | float | Callable[[int], float], *, name: Optional[str]) -> None:
        if name is not None and (not isinstance(name, str) or name == '' or len(name.split()) != 1):
            raise FactorError('invalid name for Factor, name must be not empty string without space')
        self._name: Optional[str] = name
        self._value: Optional[float] = None
        self._func: Optional[Callable[[int], float]] = None

        self.set_fvalue(value)

    def set_fvalue(self, value: int | float | Callable[[int], float]):
        match value:
            case int(value) | float(value):
                self._value = float(value)
                self._func = None
            case FunctionType() as func:
                self._func = func
            case _:
                raise FactorError('invalid value for Factor, value can be int | float | Callable[[int], float]')

    def get_fvalue(self):
        if self._func is None:
            return self._func
        return self._value

    @staticmethod
    def may_be_factor(value):
        return isinstance(value, (int, float)) or callable(value)

    def update(self, time: int):
        if self._func is not None:
            try:
                res = self._func(time)
            except Exception:
                raise FactorError(f"factor '{self}' cannot be calculated with argument {time}")
            self._value = res

    @property
    def value(self) -> Optional[float]:
        return self._value

    @property
    def is_dynamic(self) -> bool:
        return self._func is not None

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, name: str):
        if not isinstance(name, str) or name == '':
            raise FactorError('invalid name for Factor, name must be not empty string')
        self._name = name

    def __str__(self):
        return self._name

    @staticmethod
    def func_by_keyframes(keyframes: dict[int, float | int], continuation_mode='cont') -> Callable[[int], float]:
        """
        creates functions based on keyframes
        :param keyframes: factor values by key points
        :param continuation_mode: what value the function will take before the first and after the last key frames:
        'cont' - continuation of the nearest dynamics
        'keep' - keeping the nearest value
        :return: function based on keyframes
        """
        if continuation_mode == 'cont':
            cont = True
        elif continuation_mode == 'keep':
            cont = False
        else:
            raise ValueError("continuation_mode may be 'keep' or 'cont'")

        keys = tuple(sorted(keyframes))
        key_speed = {keys[i]: (keyframes[keys[i + 1]] - keyframes[keys[i]]) / (keys[i + 1] - keys[i])
                     for i in range(len(keys) - 1)}
        key_speed[keys[-1]] = key_speed[keys[-2]]

        def func(time: int) -> float:
            if keys[0] <= time <= keys[-1]:
                key_i = 0
                while key_i < len(keys) - 1 and keys[key_i + 1] < time:
                    key_i += 1
                key = keys[key_i]
                return float(keyframes[key] + key_speed[key] * (time - key))
            else:
                if time < keys[0]:
                    k = keys[0]
                else:
                    k = keys[-1]
                v = float(keyframes[k] + (cont and key_speed[k] * (time - k)))
                return v
        return func