import numpy as np
import pytest
from epidemmo.factor import Factor, FactorError


@pytest.mark.parametrize('value, name', [(2, 'beta'), (0.1, 'gamma'), (-0.5, 'neg')])
def test_good_static(value, name):
    f = Factor(name, value)
    arr = np.zeros((2, 2), dtype=np.float64)
    f.connect_matrix(arr, (0, 0))
    assert (f.name, f.value, arr[0, 0]) == (name, value, value)


@pytest.mark.parametrize('value, name', [(2, 'beta'), (0.1, 'gamma'), (-0.5, 'neg')])
def test_good_update_static(value, name):
    f = Factor(name, value)
    arr = np.zeros((2, 2), dtype=np.float64)
    f.connect_matrix(arr, (0, 0))
    f.update(30)
    assert (f.name, f.value, arr[0, 0]) == (name, value, value)


@pytest.mark.parametrize('value, name', [('beta', 'beta'), (0.1, ''), (None, 'neg'), (1, 1)])
def test_init_error(value, name):
    with pytest.raises(FactorError):
        f = Factor(name, value)


@pytest.mark.parametrize('value, name, time, result', [(lambda x: x + 0.5, 'beta', 3, 3.5),
                                                       (lambda t: t / 2, 'gamma', 0.5, 0.25)])
def test_dynamic_good(value, name, time, result):
    f = Factor(name, value)
    arr = np.zeros((2, 2), dtype=np.float64)
    arr2 = np.zeros(3, dtype=np.float64)
    f.connect_matrix(arr, (0, 0))
    f.connect_matrix(arr2, 0)
    f.update(time)
    assert (f.value, arr[0, 0], arr2[0]) == (result, result, result)


def test_dynamic_error():
    with pytest.raises(FactorError):
        f = Factor('gamma', lambda t: 1 / (5 - t))
        arr = np.zeros((2, 2), dtype=np.float64)
        f.connect_matrix(arr, (0, 0))
        f.update(5)


@pytest.mark.parametrize('time, mode, result', [(7, 'cont', 0.07), (1, 'cont', 0.01), (15, 'cont', 0.15),
                                                (7, 'keep', 0.07), (1, 'keep', 0.05), (15, 'keep', 0.1)])
def test_dynamic_from_dict(time, mode, result):
    func = Factor.func_by_keyframes({5: 0.05, 10: 0.1}, continuation_mode=mode)
    f = Factor('dyn', func)
    arr = np.zeros((2, 2), dtype=np.float64)
    f.connect_matrix(arr, (0, 0))
    f.update(time)
    assert (f.value, arr[0, 0]) == (pytest.approx(result), pytest.approx(result))


@pytest.mark.parametrize('time, array, result', [(1, [0.05, 0.07], 0.07), (0, (0.01, 0.03), 0.01),
                                                 (0, np.array([0.01, 0.03]), 0.01)])
def test_array_factor(time, array, result):
    f = Factor('array', array)
    arr = np.zeros((2, 2), dtype=np.float64)
    f.connect_matrix(arr, (0, 0))
    f.update(time)
    assert (f.value, arr[0, 0]) == (pytest.approx(result), pytest.approx(result))
