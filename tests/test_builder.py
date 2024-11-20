import pytest
from epidemmo import ModelBuilder
from epidemmo.stage import Stage, StageError
from epidemmo.builder import ModelBuilderError
from epidemmo.factor import FactorError


@pytest.fixture()
def builder() -> ModelBuilder:
    builder = ModelBuilder()
    builder.add_stage('S', 100).add_stage('I', 1).add_stage('R', 0)
    builder.add_factor('beta', 0.4).add_factor('gamma', 0.1)
    return builder

@pytest.mark.parametrize('name, num', [(100, 'S'), (50, 50), ('S', 'S'), ('S'*30, 10), ('S', -10)])
def test_bad_stage(name, num):
    with pytest.raises((ModelBuilderError, StageError)):
        builder = ModelBuilder()
        builder.add_stage(name, num)


def test_existing_stage():
    with pytest.raises(ModelBuilderError):
        builder = ModelBuilder()
        builder.add_stage('S', 100)
        builder.add_stage('S', 50)


@pytest.mark.parametrize('value, name', [('beta', 'beta'), (0.1, ''), (None, 'neg'), (1, 1)])
def test_bad_factor(value, name):
    with pytest.raises((ModelBuilderError, FactorError)):
        builder = ModelBuilder()
        builder.add_factor(name, value)


def test_existing_factor():
    with pytest.raises(ModelBuilderError):
        builder = ModelBuilder()
        builder.add_factor('beta', 0.4)
        builder.add_factor('beta', 0.1)


def test_add_simple_flow(builder):
    builder.add_flow('S', 'I', 'beta')
    flow = builder._flows[0]
    assert str(flow) == 'Flow(S>I)'


def test_two_target_flow(builder):
    builder.add_flow('S', {'I': 0.4, 'R': 0.6}, 'beta')
    flow = builder._flows[0]
    assert str(flow) == 'Flow(S>I,R)'


def test_add_induced_flow(builder):
    builder.add_flow('S', 'I', 'beta', {'I': 'gamma'})
    i_stage = builder._stages['I']
    flow = builder._flows[0]
    assert flow._ind_dict[i_stage].value == 0.1


@pytest.mark.parametrize('start, end', [('S', 'Q'), ('Q', 'S'), ('S', {'Q': 0.2, 'I': 0.8})])
def test_not_exist_stage(builder, start, end):
    with pytest.raises(ModelBuilderError):
        builder.add_flow(start, end, 'beta')


@pytest.mark.parametrize('start, end, factor', [('S', 'I', 'alpha'), ('S', {'I': 'alpha'}, 'beta')])
def test_not_exist_factor(builder, start, end, factor):
    with pytest.raises(ModelBuilderError):
        builder.add_flow(start, end, factor)


@pytest.mark.parametrize('start, end', [('S', 'S'), ('S', {'I': 0.4, 'S': 0.6})])
def test_start_end_conflict(builder, start, end):
    with pytest.raises(ModelBuilderError):
        builder.add_flow(start, end, 'beta')


@pytest.mark.parametrize('start, end', [('S', 'I'), ('S', {'I': 1, 'R': 0})])
def test_exist_flow(builder, start, end):
    with pytest.raises(ModelBuilderError):
        builder.add_flow('S', 'I', 'beta')
        builder.add_flow(start, end, 'beta')


def test_completed_model(builder):
    builder.add_flow('S', 'I', 'beta', 'I')
    builder.add_flow('I', 'R', 'gamma')
    model = builder.build()
    assert model.name == 'SIR'


def test_not_completed_model(builder):
    with pytest.raises(ModelBuilderError):
        builder.add_flow('S', 'I', 'beta', 'I')
        model = builder.build()
