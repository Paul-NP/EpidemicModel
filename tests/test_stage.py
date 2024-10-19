import pytest
from epidemmo.stage import Stage, StageError


@pytest.mark.parametrize('name, num, index', [('S', 100, 0), ('I', 1.0, 1), ('Rec', 0, 2)])
def test_full_init_good(name, num, index):
    st = Stage(name, num, index=index)
    assert (st.name, st.start_num, st.index) == (name, num, index)


@pytest.mark.parametrize('name, num', [(100, 'S'), (50, 50), ('S', 'S'), ('S'*30, 10), ('S', -10)])
def test_error(name, num):
    with pytest.raises(StageError):
        st = Stage(name, num, index=0)


@pytest.mark.parametrize('name, num, str_stage', [('S', 100, 'Stage(S)'), ('I', 1, 'Stage(I)'),
                                                  ('Rec', 0, 'Stage(Rec)')])
def test_str_stage(name, num, str_stage):
    st = Stage(name, num, index=0)
    assert str(st) == str_stage


def test_set_num():
    st = Stage('S', 1, index=0)
    st.start_num = 5
    assert st.start_num == 5


def test_set_num_error():
    st = Stage('S', 0, index=0)
    with pytest.raises(StageError):
        st.start_num = 'q'

def test_latex_eq():
    st = Stage('S', 0, index=0)
    st.set_latext_repr('S')

    st.add_latex_out('\\frac{S \\cdot \\beta \\cdot I}{N}')
    st.add_latex_input('R \\cdot \\alpha')

    assert st.get_latex_equation() == '\\frac{dS}{dt} = R \\cdot \\alpha - \\frac{S \\cdot \\beta \\cdot I}{N}'

