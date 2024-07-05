from emodel import Stage, Flow, Factor, EpidemicModel
import pytest


def test_sir():
    s = Stage('S', 100)
    i = Stage('I', 1)
    r = Stage('R')

    beta = Factor(0.4, name='beta')
    gama = Factor(0.1, name='gama')

    si = Flow(s, i, beta, inducing_factors=i)
    ir = Flow(i, r, gama)

    model = EpidemicModel((s, i, r), (si, ir))
    result = model.start(60)
    assert result.loc[21, 'I'] == pytest.approx(41, abs=1)

