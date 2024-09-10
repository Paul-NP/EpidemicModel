from emodel import ModelBuilder
import pytest


def test_sir():
    builder = ModelBuilder()
    builder.add_stage('S', 100).add_stage('I', 1).add_stage('R')
    builder.add_factor('beta', 0.4).add_factor('gamma', 0.1)
    builder.add_flow('S', 'I', 'beta', 'I').add_flow('I', 'R', 'gamma')

    model = builder.build()    

    result = model.start(60)
    assert result.loc[21, 'I'] == pytest.approx(41, abs=1)

