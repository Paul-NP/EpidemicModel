import pytest
from emodel.factor import Factor, FactorError


@pytest.mark.parametrize('value, name', [(2, 'beta'), (0.1, 'gamma'), (-0.5, 'neg')])
def test_good_static(value, name):
    f = Factor(name, value)
    assert f.name, f.value == (name, value)


@pytest.mark.parametrize('value, name', [(2, 'beta'), (0.1, 'gamma'), (-0.5, 'neg')])
def test_good_update_static(value, name):
    f = Factor(name, value)
    f.update(30)
    assert f.name, f.value == (name, value)


@pytest.mark.parametrize('value, name', [('beta', 'beta'), (0.1, ''), (None, 'neg'), (1, 1)])
def test_init_error(value, name):
    with pytest.raises(FactorError):
        f = Factor(name, value)


@pytest.mark.parametrize('value, name, time, result', [(lambda x: x + 0.5, 'beta', 3, 3.5),
                                                       (lambda t: t / 2, 'gamma', 0.5, 0.25)])
def test_dynamic_good(value, name, time, result):
    f = Factor(name, value)
    f.update(time)
    assert f.value == result


def test_dynamic_error():
    with pytest.raises(FactorError):
        f = Factor('gamma', lambda t: 1 / (5 - t))
        f.update(5)


@pytest.mark.parametrize('time, mode, result', [(7, 'cont', 0.07), (1, 'cont', 0.01), (15, 'cont', 0.15),
                                                (7, 'keep', 0.07), (1, 'keep', 0.05), (15, 'keep', 0.1)])
def test_dynamic_from_dict(time, mode, result):
    func = Factor.func_by_keyframes({5: 0.05, 10: 0.1}, continuation_mode=mode)
    f = Factor('dyn', func)
    f.update(time)
    assert f.value == pytest.approx(result)
