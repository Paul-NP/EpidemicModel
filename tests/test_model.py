from epidemmo import ModelBuilder, Standard
import pytest


def test_sir():
    builder = ModelBuilder()
    builder.add_stage('S', 100).add_stage('I', 1).add_stage('R')
    builder.add_factor('beta', 0.4).add_factor('gamma', 0.1)
    builder.add_flow('S', 'I', 'beta', 'I').add_flow('I', 'R', 'gamma')

    model = builder.build()

    result = model.start(60)
    assert result.loc[21, 'I'] == pytest.approx(41, abs=1)


def test_sir_stoch():
    model = Standard.get_SIR_builder().build()
    result = model.start(30, stochastic_time=True, stochastic_changes=True)
    assert (result['S'] + result['I'] + result['R'] == 101).all()


@pytest.mark.parametrize('time, s, l, h, r, d', [(0, 100, 1, 0, 0, 0), (10, 85, 9, 2, 4, 0), (30, 7, 22, 5, 61, 5)])
def test_slhrd(time, s, l, h, r, d):
    builder = ModelBuilder()
    builder.add_stage('S', 100).add_stage('L', 1).add_stage('H', 0).add_stage('R').add_stage('D')
    builder.add_factor('beta', 0.4).add_factor('gamma', 0.1)
    builder.add_flow('S', {'L': 0.8, 'H': 0.2}, 'beta', {'L': 1, 'H': 1})
    builder.add_flow('H', {'R': 0.6, 'D': 0.4}, 'gamma')
    builder.add_flow('L', 'R', 'gamma')

    model = builder.build()
    result = model.start(60)
    assert tuple(result.loc[time]) == pytest.approx((s, l, h, r, d), abs=1)


def test_sir_fast_create():
    builder = ModelBuilder()
    builder.add_stages(S=100, I=1, R=0).add_factors(beta=0.4, gamma=0.1)
    builder.add_flow('S', 'I', 'beta', 'I').add_flow('I', 'R', 'gamma')

    model = builder.build()
    result = model.start(60)
    assert result.loc[21, 'I'] == pytest.approx(41, abs=1)


@pytest.mark.parametrize('time, s, i, r', [(0, 100, 1, 0), (10, 85, 11, 4), (30, 7, 27, 67)])
def test_standard_sir(time, s, i, r):
    model = Standard.get_SIR_builder().build()
    result = model.start(40)
    assert tuple(result.loc[time]) == pytest.approx((s, i, r), abs=1)


@pytest.mark.parametrize('time, s, e, i, r', [(0, 100, 0, 1, 0), (10, 96, 2, 1, 1), (30, 73, 12, 7, 8)])
def test_standard_seir(time, s, e, i, r):
    model = Standard.get_SEIR_builder().build()
    result = model.start(40)
    assert tuple(result.loc[time]) == pytest.approx((s, e, i, r), abs=1)


@pytest.mark.parametrize('time, s, i, r', [(0, 100, 1, 0), (10, 83, 11, 8), (30, 14, 8, 78)])
def test_sir_changed(time, s, i, r):
    model = Standard.get_SIR_builder().build()
    model.set_factors(beta=0.5, gamma=0.2)
    result = model.start(40)
    assert tuple(result.loc[time]) == pytest.approx((s, i, r), abs=1)
