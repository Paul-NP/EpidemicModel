import unittest
from factor import Factor
from stage import Stage


class StageTestCase(unittest.TestCase):
    def test_changes(self):
        st = Stage("S", 100)
        self.assertEqual(st.num, 100)
        st.add_change(5)
        st.add_change(-7)
        st.add_change(10)
        st.apply_changes()
        self.assertEqual(st.num, 108)