from __future__ import annotations
from typing import Callable, Optional, overload
from types import FunctionType
from inspect import getsource


class FactorError(Exception):
    pass


class Factor:
    _func: Callable[[int], float]
    _value: float
    _description: str
    _name: str
    _static: bool

    __DEFAULT_NAME: str = 'unnamed'

    @overload
    def __init__(self, keyframes: dict[int, float], name: Optional[str] = None):
        pass

    @overload
    def __init__(self, value: int | float, name: Optional[str] = None):
        pass

    @overload
    def __init__(self, func: Callable[[int], float], name: Optional[str] = None):
        pass

    def __init__(self, value: int | float | dict[int, float | int] | Callable[[int], float],
                 name: Optional[str] = None) -> None:
        """
        init Factor
        :param value: static factor value | factor values by key points | function of time factor value
        :param name: name of factor
        """
        if name is None:
            self._name = self.__DEFAULT_NAME
        elif isinstance(name, str) and len(name) > 0:
            self._name = name
        else:
            raise FactorError("Factor name must be not empty string")

        match value:
            case int(value) | float(value):
                self._value = float(value)
                self._func = lambda time: self._value
                self._static = True
                type_factor = "stat"
                description = self._value
            case dict(keyframes):
                if any(not isinstance(key, int) or not isinstance(val, (float, int)) for key, val in keyframes.items()):
                    raise FactorError("invalid dict of Factor values, keys must be int, values must be int|float")
                else:
                    self._func = self._get_keyframe_func(keyframes)
                    self._value = self.__call__(0)
                self._static = False
                type_factor = "keys"
                description = keyframes
            case FunctionType() as func:
                self._func = func
                self.__call__(0)
                line = " ".join(getsource(func).strip().replace("\n", " ").split())
                self._static = False
                type_factor = "func"
                description = line
            case _:
                raise FactorError("invalid value for Factor, value can be int | float | dict[int, float | int] | "
                                  "Callable[[int], float]")

        self._description = f"{type_factor} | {description}"

    @staticmethod
    def _get_keyframe_func(keyframes: dict[int, float | int]) -> Callable[[int], float]:
        """
        creates functions based on keyframes
        :param keyframes: factor values by key points
        :return: function based on keyframes
        """

        keys = tuple(sorted(keyframes))
        key_speed = {keys[i]: (keyframes[keys[i + 1]] - keyframes[keys[i]]) / (keys[i + 1] - keys[i])
                     for i in range(len(keys) - 1)}
        key_speed[keys[-1]] = key_speed[keys[-2]]

        def func(time: int) -> float:
            if time < keys[0]:
                shift = keys[0] - time
                value = keyframes[keys[0]] - shift * key_speed[keys[0]]
                return value
            else:
                key_i = 0
                while key_i < len(keys) - 1 and keys[key_i + 1] < time:
                    key_i += 1
                key = keys[key_i]
                return keyframes[key] + key_speed[key] * (time - key)

        return func

    def _get_new_name(self, other: Factor | int | float, operand: str, inverse: bool = False) -> str:
        if isinstance(other, Factor):
            second = other.name
        else:
            second = str(other)

        if any((char in second for char in "+-/*")):
            second = f"({second})"

        if any((char in self.name for char in "+-/*")):
            first = f"({self.name})"
        else:
            first = self.name

        if inverse:
            return f"{second}{operand}{first}"
        else:
            return f"{first}{operand}{second}"

    def __call__(self, time: int) -> float:
        try:
            res = self._func(time)
        except ZeroDivisionError:
            res = 0
        self._value = res
        return self._value

    def __add__(self, other: Factor | int | float) -> Factor:
        match self._static, other:
            case True, Factor(_static=True):
                value = self.value + other.value
            case True, Factor(_static=False):
                def func(time: int) -> float:
                    return self._value + other._func(time)
                value = func
            case False, Factor(_static=True):
                def func(time: int) -> float:
                    return self._func(time) + other._value
                value = func
            case False, Factor(_static=False):
                def func(time: int) -> float:
                    return self._func(time) + other._func(time)
                value = func
            case True, int(other) | float(other):
                value = self._value + other
            case False, int(other) | float(other):
                def func(time: int) -> float:
                    return self._func(time) + other
                value = func
            case _:
                raise TypeError(f"unsupported operand type(s) for +: 'Factor' and '{type(other).__name__}'")
        new_name = self._get_new_name(other, "+")
        return Factor(value, new_name)

    def __radd__(self, other: Factor | int | float) -> Factor:
        return self + other

    def __sub__(self, other: Factor | int | float) -> Factor:
        match self._static, other:
            case True, Factor(_static=True):
                value = self.value - other.value
            case True, Factor(_static=False):
                def func(time: int) -> float:
                    return self._value - other._func(time)
                value = func
            case False, Factor(_static=True):
                def func(time: int) -> float:
                    return self._func(time) - other._value
                value = func
            case False, Factor(_static=False):
                def func(time: int) -> float:
                    return self._func(time) - other._func(time)
                value = func
            case True, int(other) | float(other):
                value = self._value - other
            case False, int(other) | float(other):
                def func(time: int) -> float:
                    return self._func(time) - other
                value = func
            case _:
                raise TypeError(f"unsupported operand type(s) for -: 'Factor' and '{type(other).__name__}'")
        new_name = self._get_new_name(other, "-")
        return Factor(value, new_name)

    def __rsub__(self, other: Factor | int | float) -> Factor:
        match self._static, other:
            case True, Factor(_static=True):
                value = other.value - self.value
            case True, Factor(_static=False):
                def func(time: int) -> float:
                    return other._func(time) - self._value
                value = func
            case False, Factor(_static=True):
                def func(time: int) -> float:
                    return other._value - self._func(time)
                value = func
            case False, Factor(_static=False):
                def func(time: int) -> float:
                    return other._func(time) - self._func(time)
                value = func
            case True, int(other) | float(other):
                value = other - self._value
            case False, int(other) | float(other):
                def func(time: int) -> float:
                    return other - self._func(time)
                value = func
            case _:
                raise TypeError(f"unsupported operand type(s) for -: '{type(other).__name__}' and 'Factor'")
        new_name = self._get_new_name(other, "-", inverse=True)
        return Factor(value, new_name)

    def __mul__(self, other: Factor | int | float) -> Factor:

        match self._static, other:
            case True, Factor(_static=True):
                value = self.value * other.value
            case True, Factor(_static=False):
                def func(time: int) -> float:
                    return self._value * other._func(time)
                value = func
            case False, Factor(_static=True):
                def func(time: int) -> float:
                    return self._func(time) * other._value
                value = func
            case False, Factor(_static=False):
                def func(time: int) -> float:
                    return self._func(time) * other._func(time)
                value = func
            case True, int(other) | float(other):
                value = self._value * other
            case False, int(other) | float(other):
                def func(time: int) -> float:
                    return self._func(time) * other
                value = func
            case _:
                raise TypeError(f"unsupported operand type(s) for *: 'Factor' and '{type(other).__name__}'")

        new_name = self._get_new_name(other, "*")
        return Factor(value, new_name)

    def __rmul__(self, other: Factor | int | float) -> Factor:
        return self * other

    def __truediv__(self, other: Factor | int | float) -> Factor:
        match self._static, other:
            case True, Factor(_static=True):
                value = self.value / other.value
            case True, Factor(_static=False):
                def func(time: int) -> float:
                    return self._value / other._func(time)
                value = func
            case False, Factor(_static=True):
                def func(time: int) -> float:
                    return self._func(time) / other._value
                value = func
            case False, Factor(_static=False):
                def func(time: int) -> float:
                    return self._func(time) / other._func(time)
                value = func
            case True, int(other) | float(other):
                value = self._value / other
            case False, int(other) | float(other):
                def func(time: int) -> float:
                    return self._func(time) / other
                value = func
            case _:
                raise TypeError(f"unsupported operand type(s) for /: 'Factor' and '{type(other).__name__}'")

        new_name = self._get_new_name(other, "/")
        return Factor(value, new_name)

    def __rtruediv__(self, other: Factor | int | float) -> Factor:
        match self._static, other:
            case True, Factor(_static=True):
                value = other.value / self.value
            case True, Factor(_static=False):
                def func(time: int) -> float:
                    return other._func(time) / self._value
                value = func
            case False, Factor(_static=True):
                def func(time: int) -> float:
                    return other._value / self._func(time)
                value = func
            case False, Factor(_static=False):
                def func(time: int) -> float:
                    return other._func(time) / self._func(time)
                value = func
            case True, int(other) | float(other):
                value = other / self._value
            case False, int(other) | float(other):
                def func(time: int) -> float:
                    return other / self._func(time)
                value = func
            case _:
                raise TypeError(f"unsupported operand type(s) for /: '{type(other).__name__}' and 'Factor'")

        new_name = self._get_new_name(other, "/", inverse=True)
        return Factor(value, new_name)

    def __str__(self) -> str:
        return f"Factor '{self._name}'"

    def __repr__(self) -> str:
        # return self.__str__()
        return f"Factor '{self._name:}' | {self._description}"

    @property
    def value(self) -> float:
        return self._value

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, new_name: str):
        if isinstance(new_name, str) and len(new_name) > 0:
            self._name = new_name
        else:
            raise TypeError("the name for the factor must be not empty string")

    @property
    def static(self) -> bool:
        return self._static


factor_one = Factor(1, "one")
factor_zero = Factor(0, "zero")
