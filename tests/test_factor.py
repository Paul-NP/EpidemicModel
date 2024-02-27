import unittest
from factor import Factor, FactorError


class FactorTestCase(unittest.TestCase):
    def test_add(self):
        f1 = Factor(5)

        # static + static
        f2 = Factor(6)
        f = f1 + f2
        self.assertIsInstance(f, Factor)
        self.assertEqual(f.static, True)
        self.assertEqual(f.value, 11)
        self.assertEqual(f(0), 11)
        self.assertEqual(f(100), 11)
        f = f2 + f1
        self.assertIsInstance(f, Factor)
        self.assertEqual(f.static, True)
        self.assertEqual(f.value, 11)
        self.assertEqual(f(0), 11)
        self.assertEqual(f(100), 11)

        # static + func
        f2 = Factor(lambda x: 2 * x)
        f = f1 + f2
        self.assertIsInstance(f, Factor)
        self.assertEqual(f.static, False)
        self.assertEqual(f(2), 9)
        self.assertEqual(f(5), 15)
        f = f2 + f1
        self.assertIsInstance(f, Factor)
        self.assertEqual(f.static, False)
        self.assertEqual(f(2), 9)
        self.assertEqual(f(5), 15)

        # static + num
        f2 = 6
        f = f1 + f2
        self.assertIsInstance(f, Factor)
        self.assertEqual(f.static, True)
        self.assertEqual(f.value, 11)
        self.assertEqual(f(0), 11)
        self.assertEqual(f(100), 11)
        f = f2 + f1
        self.assertIsInstance(f, Factor)
        self.assertEqual(f.static, True)
        self.assertEqual(f.value, 11)
        self.assertEqual(f(0), 11)
        self.assertEqual(f(100), 11)

        # static + str
        f2 = "f"
        with self.assertRaises(TypeError):
            f1 + f2

        f1 = Factor(lambda x: x*3)

        # func + func
        f2 = Factor(lambda x: x*2)
        f = f1 + f2
        self.assertIsInstance(f, Factor)
        self.assertEqual(f.static, False)
        self.assertEqual(f(2), 10)
        self.assertEqual(f(5), 25)
        f = f2 + f1
        self.assertIsInstance(f, Factor)
        self.assertEqual(f.static, False)
        self.assertEqual(f(2), 10)
        self.assertEqual(f(5), 25)

        # func + num
        f2 = 5
        f = f1 + f2
        self.assertIsInstance(f, Factor)
        self.assertEqual(f.static, False)
        self.assertEqual(f(2), 11)
        self.assertEqual(f(5), 20)
        f = f2 + f1
        self.assertIsInstance(f, Factor)
        self.assertEqual(f.static, False)
        self.assertEqual(f(2), 11)
        self.assertEqual(f(5), 20)

    def test_sub(self):
        f1 = Factor(6)

        # static - static
        f2 = Factor(2)
        f = f1 - f2
        self.assertIsInstance(f, Factor)
        self.assertEqual(f.static, True)
        self.assertEqual(f.value, 4)
        self.assertEqual(f(0), 4)
        self.assertEqual(f(100), 4)
        f = f2 - f1
        self.assertIsInstance(f, Factor)
        self.assertEqual(f.static, True)
        self.assertEqual(f.value, -4)
        self.assertEqual(f(0), -4)
        self.assertEqual(f(100), -4)

        # static - func
        f2 = Factor(lambda x: 2 * x)
        f = f1 - f2
        self.assertIsInstance(f, Factor)
        self.assertEqual(f.static, False)
        self.assertEqual(f(2), 2)
        self.assertEqual(f(5), -4)
        # func - static
        f = f2 - f1
        self.assertIsInstance(f, Factor)
        self.assertEqual(f.static, False)
        self.assertEqual(f(2), -2)
        self.assertEqual(f(5), 4)

        # static - num
        f2 = 3
        f = f1 - f2
        self.assertIsInstance(f, Factor)
        self.assertEqual(f.static, True)
        self.assertEqual(f.value, 3)
        self.assertEqual(f(0), 3)
        self.assertEqual(f(100), 3)
        # num - static
        f = f2 - f1
        self.assertIsInstance(f, Factor)
        self.assertEqual(f.static, True)
        self.assertEqual(f.value, -3)
        self.assertEqual(f(0), -3)
        self.assertEqual(f(100), -3)

        # static - str
        f2 = "f"
        with self.assertRaises(TypeError):
            f1 - f2
        with self.assertRaises(TypeError):
            f2 - f1

        f1 = Factor(lambda x: x*3)

        # func - func
        f2 = Factor(lambda x: x*2)
        f = f1 - f2
        self.assertIsInstance(f, Factor)
        self.assertEqual(f.static, False)
        self.assertEqual(f(2), 2)
        self.assertEqual(f(5), 5)
        f = f2 - f1
        self.assertIsInstance(f, Factor)
        self.assertEqual(f.static, False)
        self.assertEqual(f(2), -2)
        self.assertEqual(f(5), -5)

        # func - num
        f2 = 3
        f = f1 - f2
        self.assertIsInstance(f, Factor)
        self.assertEqual(f.static, False)
        self.assertEqual(f(2), 3)
        self.assertEqual(f(5), 12)
        # num - func
        f = f2 - f1
        self.assertIsInstance(f, Factor)
        self.assertEqual(f.static, False)
        self.assertEqual(f(2), -3)
        self.assertEqual(f(5), -12)

    def test_mul(self):
        f1 = Factor(5)

        # static * static
        f2 = Factor(6)
        f = f1 * f2
        self.assertIsInstance(f, Factor)
        self.assertEqual(f.static, True)
        self.assertEqual(f.value, 30)
        self.assertEqual(f(0), 30)
        self.assertEqual(f(100), 30)
        f = f2 * f1
        self.assertIsInstance(f, Factor)
        self.assertEqual(f.static, True)
        self.assertEqual(f.value, 30)
        self.assertEqual(f(0), 30)
        self.assertEqual(f(100), 30)

        # static * func
        f2 = Factor(lambda x: 2 * x)
        f = f1 * f2
        self.assertIsInstance(f, Factor)
        self.assertEqual(f.static, False)
        self.assertEqual(f(2), 20)
        self.assertEqual(f(5), 50)
        f = f2 * f1
        self.assertIsInstance(f, Factor)
        self.assertEqual(f.static, False)
        self.assertEqual(f(2), 20)
        self.assertEqual(f(5), 50)

        # static * num
        f2 = 6
        f = f1 * f2
        self.assertIsInstance(f, Factor)
        self.assertEqual(f.static, True)
        self.assertEqual(f.value, 30)
        self.assertEqual(f(0), 30)
        self.assertEqual(f(100), 30)
        f = f2 * f1
        self.assertIsInstance(f, Factor)
        self.assertEqual(f.static, True)
        self.assertEqual(f.value, 30)
        self.assertEqual(f(0), 30)
        self.assertEqual(f(100), 30)

        # static * str
        f2 = "f"
        with self.assertRaises(TypeError):
            f1 * f2
        f2 = "f"
        with self.assertRaises(TypeError):
            f2 * f1

        f1 = Factor(lambda x: x*3)

        # func * func
        f2 = Factor(lambda x: x*2)
        f = f1 * f2
        self.assertIsInstance(f, Factor)
        self.assertEqual(f.static, False)
        self.assertEqual(f(2), 24)
        self.assertEqual(f(5), 150)
        f = f2 * f1
        self.assertIsInstance(f, Factor)
        self.assertEqual(f.static, False)
        self.assertEqual(f(2), 24)
        self.assertEqual(f(5), 150)

        # func * num
        f2 = 5
        f = f1 * f2
        self.assertIsInstance(f, Factor)
        self.assertEqual(f.static, False)
        self.assertEqual(f(2), 30)
        self.assertEqual(f(5), 75)
        f = f2 * f1
        self.assertIsInstance(f, Factor)
        self.assertEqual(f.static, False)
        self.assertEqual(f(2), 30)
        self.assertEqual(f(5), 75)

    def test_div(self):
        f1 = Factor(6)

        # static / static
        f2 = Factor(3)
        f = f1 / f2
        self.assertIsInstance(f, Factor)
        self.assertEqual(f.static, True)
        self.assertEqual(f.value, 2.0)
        self.assertEqual(f(0), 2.0)
        self.assertEqual(f(100), 2.0)
        f = f2 / f1
        self.assertIsInstance(f, Factor)
        self.assertEqual(f.static, True)
        self.assertEqual(f.value, 0.5)
        self.assertEqual(f(0), 0.5)
        self.assertEqual(f(100), 0.5)

        # static / func
        f2 = Factor(lambda x: 2 * x)
        f = f1 / f2
        self.assertIsInstance(f, Factor)
        self.assertEqual(f.static, False)
        self.assertEqual(f(2), 1.5)
        self.assertEqual(f(5), 0.6)
        # func / static
        f = f2 / f1
        self.assertIsInstance(f, Factor)
        self.assertEqual(f.static, False)
        self.assertEqual(f(2), 4/6)
        self.assertEqual(f(5), 10/6)

        # static / num
        f2 = 3
        f = f1 / f2
        self.assertIsInstance(f, Factor)
        self.assertEqual(f.static, True)
        self.assertEqual(f.value, 2.0)
        self.assertEqual(f(0), 2.0)
        self.assertEqual(f(100), 2.0)
        # num / static
        f = f2 / f1
        self.assertIsInstance(f, Factor)
        self.assertEqual(f.static, True)
        self.assertEqual(f.value, 0.5)
        self.assertEqual(f(0), 0.5)
        self.assertEqual(f(100), 0.5)

        # static - str
        f2 = "f"
        with self.assertRaises(TypeError):
            f1 / f2
        with self.assertRaises(TypeError):
            f2 / f1

        f1 = Factor(lambda x: x*3)

        # func / func
        f2 = Factor(lambda x: x*2)
        f = f1 / f2
        self.assertIsInstance(f, Factor)
        self.assertEqual(f.static, False)
        self.assertEqual(f(2), 1.5)
        self.assertEqual(f(5), 1.5)
        f = f2 / f1
        self.assertIsInstance(f, Factor)
        self.assertEqual(f.static, False)
        self.assertEqual(f(2), 4/6)
        self.assertEqual(f(5), 10/15)

        # func / num
        f2 = 3
        f = f1 / f2
        self.assertIsInstance(f, Factor)
        self.assertEqual(f.static, False)
        self.assertEqual(f(2), 2.0)
        self.assertEqual(f(5), 5.0)
        # num / func
        f = f2 / f1
        self.assertIsInstance(f, Factor)
        self.assertEqual(f.static, False)
        self.assertEqual(f(2), 0.5)
        self.assertEqual(f(5), 0.2)

    def test_keyframes(self):
        f1 = Factor({1: 1, 9: 5, 13: 4})
        f2 = Factor({1: 0.8, 4: 0.5, 10: 1.7})

        f_sum = f1 + f2
        f_mul = f1 * f2
        f_sub = f1 - f2
        f_div = f1 / f2

        self.assertIsInstance(f_sum, Factor)
        self.assertIsInstance(f_mul, Factor)
        self.assertIsInstance(f_sub, Factor)
        self.assertIsInstance(f_div, Factor)

        self.assertAlmostEqual(f_sum(2), 1.5 + 0.7)
        self.assertAlmostEqual(f_mul(2), 1.5 * 0.7)
        self.assertAlmostEqual(f_sub(2), 1.5 - 0.7)
        self.assertAlmostEqual(f_div(2), 1.5 / 0.7)

        self.assertAlmostEqual(f_sum(14), 3.75 + 2.5)
        self.assertAlmostEqual(f_mul(14), 3.75 * 2.5)
        self.assertAlmostEqual(f_sub(14), 3.75 - 2.5)
        self.assertAlmostEqual(f_div(14), 3.75 / 2.5)

    def test_formula(self):
        alfa = Factor({1: 1, 11: 2}, "alfa")
        beta = Factor(lambda x: x * 2, "beta")
        gama = Factor(5, "gama")
        f = (alfa - (beta - gama) * alfa) / beta - gama
        self.assertEqual(str(f), "Factor '((alfa-((beta-gama)*alfa))/beta)-gama'")
        self.assertEqual(f(2), -4.45)

    def test_init_error(self):
        with self.assertRaises(FactorError):
            alfa = Factor("factor")
        with self.assertRaises(FactorError):
            alfa = Factor({"q": 5.5})
        with self.assertRaises(FactorError):
            alfa = Factor({4: "q"})
