from __future__ import annotations
from typing import Callable, Optional, Literal, Any
from types import FunctionType

from numpy.typing import NDArray


class FactorError(Exception):
    pass


class Factor:
    __MIN_NAME_LEN: int = 1
    __MAX_NAME_LEN: int = 30

    @classmethod
    def __check_name(cls, name: str) -> None:
        if not isinstance(name, str):
            raise FactorError('The factor name must be str')
        if len(name.split()) > 1:
            raise FactorError('The factor name must be one word')
        if not cls.__MIN_NAME_LEN <= len(name) <= cls.__MAX_NAME_LEN:
            raise FactorError(f'The factor name "{name}" has an invalid length. Valid range '
                              f'[{cls.__MIN_NAME_LEN}, {cls.__MAX_NAME_LEN}]')

    def __init__(self, name: str, value: int | float | Callable[[int], float]) -> None:
        self.__check_name(name)

        self._name: str = name
        self._value: float = 0
        self._func: Optional[Callable[[int], float]] = None
        self._latex_repr: Optional[str] = None

        self._update_func: Callable = self._update_static

        self.set_fvalue(value)

        self._connected_matrix: list[NDArray] = []
        self._value_pos_in_matrix: list[int | tuple[int, int]] = []

    def connect_matrix(self, matrix: NDArray, value_pos_in_matrix: int | tuple[int, int]) -> None:
        self._connected_matrix.append(matrix)
        self._value_pos_in_matrix.append(value_pos_in_matrix)
        matrix[value_pos_in_matrix] = self._value

    def set_fvalue(self, value: int | float | Callable[[int], float]) -> None:
        match value:
            case int(value) | float(value):
                self._value = float(value)
                self._func = None
                self._update_func = self._update_static
            case FunctionType() as func:
                self._value = func(0)
                self._func = func
                self._update_func = self._update_dynamic
            case _:
                raise FactorError('invalid value for Factor, value can be int | float | Callable[[int], float]')

    def get_fvalue(self) -> Callable[[int], float] | float:
        if self._func is not None:
            return self._func
        return self._value

    @staticmethod
    def may_be_factor(value: Any) -> bool:
        if isinstance(value, (int, float)):
            return True
        elif callable(value):
            try:
                result = value(0)
                return isinstance(result, (float, int))
            except Exception as e:
                return False
        else:
            return False

    def update(self, time: int) -> None:
        self._update_func(time)
        for matrix, value_pos_in_matrix in zip(self._connected_matrix, self._value_pos_in_matrix):
            matrix[value_pos_in_matrix] = self._value

    def _update_static(self, time: int) -> None:
        pass

    def _update_dynamic(self, time: int) -> None:
        try:
            res = self._func(time)
        except Exception:
            raise FactorError(f"factor '{self}' cannot be calculated with argument {time}")
        self._value = res

    @property
    def value(self) -> float:
        return self._value

    @property
    def is_dynamic(self) -> bool:
        return self._func is not None

    @property
    def name(self) -> str:
        return self._name

    def __str__(self) -> str:
        return self._name

    def set_latex_repr(self, latex_repr: Optional[str]) -> None:
        if latex_repr is not None and not isinstance(latex_repr, str):
            raise FactorError('latex_repr must be str or None')
        self._latex_repr = latex_repr

    def get_latex_repr(self) -> str:
        if self._latex_repr is None:
            return self._name
        return self._latex_repr

    @staticmethod
    def func_by_keyframes(keyframes: dict[int, float | int],
                          continuation_mode: str = 'cont') -> Callable[[int], float]:
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