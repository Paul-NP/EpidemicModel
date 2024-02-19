from __future__ import annotations
from typing import Callable
from types import FunctionType
from inspect import getsource


class Factor:
    _func: Callable[[int], float]
    _value: float
    _description: str
    _name: str
    _static: bool

    __name_len: int = 6
    __list_factors: list[Factor] = []

    # def __new__(cls, value: int | float | dict[int, float | int] | Callable[[int], float], name: str):
    #     if name not in [f.name for f in cls.__list_factors]:
    #         new_factor = super().__new__(cls)
    #         cls.__list_factors.append(new_factor)
    #         return new_factor
    #     else:
    #         # raise ValueError(f"create existing factor with name: {name}")
    #         return [f for f in cls.__list_factors if f.name == name][0]

    def __init__(self, value: int | float | dict[int, float | int] | Callable[[int], float],
                 name: str = "name") -> None:
        """
        init Factor
        :param value: static factor value | factor values by key points | function of time factor value
        :param name: name of factor
        """

        if isinstance(name, str):
            self._name = name
        else:
            raise ValueError("Factor name must be string")

        match value:
            case int(value) | float(value):
                self._value = float(value)
                self._func = lambda time: self._value
                self._static = True
                type_factor = "static"
                description = self._value
            case dict(keyframes):
                if any(not isinstance(key, int) or not isinstance(val, (float, int)) for key, val in keyframes.items()):
                    raise ValueError("invalid dict of Factor values, keys must be int, values must be int|float")
                else:
                    self._func = self._get_keyframe_func(keyframes)
                    self._value = self._func(0)
                self._static = False
                type_factor = "keyframes"
                description = keyframes
            case FunctionType() as func:
                self._func = func
                self._value = self._func(1)
                line = " ".join(getsource(func).strip().replace("\n", " ").split())
                self._static = False
                type_factor = "function"
                description = line
            case _:
                raise ValueError("invalid value for Factor, value can be int | float | dict[int, float | int] | "
                                 "Callable[[int], float]")

        self._description = f"{type_factor:^11s}| {description}"

    @staticmethod
    def _get_keyframe_func(keyframes: dict[int, float | int]) -> Callable[[int], float]:
        """
        creates functions based on keyframes
        :param keyframes: factor values by key points
        :return: function based on keyframes
        """

        def func(time: int) -> float:
            if time < func.keys[0]:
                shift = func.keys[0] - time
                value = func.keyframes[func.keys[0]] - shift * func.key_speed[func.keys[0]]
                return value
            else:
                key_i = 0
                while key_i < len(func.keys) - 1 and func.keys[key_i + 1] < time:
                    key_i += 1
                key = func.keys[key_i]
                return func.keyframes[key] + func.key_speed[key] * (time - key)

        func.keyframes = {key: keyframes[key] for key in sorted(list(keyframes.keys()))}
        func.keys = tuple(func.keyframes.keys())
        key_speed = {func.keys[i]: (func.keyframes[func.keys[i + 1]] - func.keyframes[func.keys[i]]) /
                                   (func.keys[i + 1] - func.keys[i]) for i in range(len(func.keys) - 1)}
        key_speed[func.keys[-1]] = key_speed[func.keys[-2]]
        func.key_speed = key_speed

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

    # def __eq__(self, other: Factor) -> bool:
    #     return self._name == other._name
    #
    # def __hash__(self):
    #     return hash(self._name)

    def __call__(self, time: int) -> float:
        self._value = self._func(time)
        return self._value

    # def __copy__(self) -> Factor:
    #     return Factor(self._func, self._name)

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

    # def copy(self) -> Factor:
    #     # добавить аргумент name, переименовывать при копировании
    #     return self.__copy__()

    @property
    def value(self) -> float:
        return self._value

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, new_name: str):
        if isinstance(new_name, str):
            self._name = new_name
        else:
            raise TypeError("the name for the factor must be a string")

    @property
    def static(self) -> bool:
        return self._static


factor_one = Factor(1, "one")
factor_zero = Factor(0, "zero")
