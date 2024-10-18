from epidemmo.fast_stage import FastStage
from epidemmo.fast_flow import FastFlow, FastFlowError
from epidemmo.fast_factor import FastFactor, FastFactorError

import pytest


@pytest.fixture()
def get_simple_flow():
    s = FastStage('S', 10, index=0)
    i = FastStage('I', 0, index=1)
    beta = FastFactor('beta', 0.4)
    fl = FastFlow(s, {i: 1}, beta, {}, index=0)
    return s, i, beta, fl


def test_start(get_simple_flow):
    s, i, beta, fl = get_simple_flow
    assert fl._start == s


def test_simple_end_dict(get_simple_flow):
    s, i, beta, fl = get_simple_flow
    assert isinstance(fl._end_dict, dict)


def test_simple_end_value(get_simple_flow):
    s, i, beta, fl = get_simple_flow
    assert fl._end_dict[i].value == 1


def test_simple_ind_dict(get_simple_flow):
    s, i, beta, fl = get_simple_flow
    assert isinstance(fl._ind_dict, dict)


def test_simple_ind_value(get_simple_flow):
    s, i, beta, fl = get_simple_flow
    assert fl._ind_dict == {}


def test_simple_fl_factor(get_simple_flow):
    s, i, beta, fl = get_simple_flow
    assert fl._flow_factor == beta

