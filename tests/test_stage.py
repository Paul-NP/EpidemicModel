import pytest
from emodel.stage import Stage, StageError


@pytest.mark.parametrize('name, num', [('S', 100), ('I', 1.0), ('Rec', 0)])
def test_full_init_good(name, num):
    st = Stage(name, num)
    assert st.name, st.num == (name, num)


def test_empty_num():
    st = Stage('S', 0)
    assert st.name, st.num == ('S', 0)


@pytest.mark.parametrize('name, num', [(100, 'S'), (50, 50), ('S', 'S'), ('S'*30, 10), ('S', -10)])
def test_error(name, num):
    with pytest.raises(StageError):
        st = Stage(name, num)


@pytest.mark.parametrize('name, num, str_stage', [('S', 100, 'Stage(S)'), ('I', 1, 'Stage(I)'),
                                                  ('Rec', 0, 'Stage(Rec)')])
def test_str_stage(name, num, str_stage):
    st = Stage(name, num)
    assert str(st) == str_stage


def test_set_num():
    st = Stage('S', 1)
    st.num = 5
    assert st.num == 5


def test_set_num_error():
    st = Stage('S', 0)
    with pytest.raises(StageError):
        st.num = 'q'


def test_not_applied_change():
    st = Stage('S', 0)
    st.add_change(10)
    st.add_change(20)
    assert st.num == 0


def test_applied_change():
    st = Stage('S', 0)
    st.add_change(10)
    st.add_change(20)
    st.apply_changes()
    assert st.num == 30
