"""
Tests for zipline.pipeline.Pipeline
"""
from unittest import TestCase

from zipline.pipeline import Factor, Filter, Pipeline
from zipline.pipeline.data import USEquityPricing
from zipline.utils.numpy_utils import float64_dtype


class SomeFactor(Factor):
    dtype = float64_dtype
    window_length = 5
    inputs = [USEquityPricing.close, USEquityPricing.high]


class SomeOtherFactor(Factor):
    dtype = float64_dtype
    window_length = 5
    inputs = [USEquityPricing.close, USEquityPricing.high]


class SomeFilter(Filter):
    window_length = 5
    inputs = [USEquityPricing.close, USEquityPricing.high]


class SomeOtherFilter(Filter):
    window_length = 5
    inputs = [USEquityPricing.close, USEquityPricing.high]


class PipelineTestCase(TestCase):

    def test_construction(self):
        p0 = Pipeline()
        self.assertEqual(p0.columns, {})
        self.assertIs(p0.screen, None)

        columns = {'f': SomeFactor()}
        p1 = Pipeline(columns=columns)
        self.assertEqual(p1.columns, columns)

        screen = SomeFilter()
        p2 = Pipeline(screen=screen)
        self.assertEqual(p2.columns, {})
        self.assertEqual(p2.screen, screen)

        p3 = Pipeline(columns=columns, screen=screen)
        self.assertEqual(p3.columns, columns)
        self.assertEqual(p3.screen, screen)

    def test_construction_bad_input_types(self):

        with self.assertRaises(TypeError):
            Pipeline(1)

        Pipeline({})

        with self.assertRaises(TypeError):
            Pipeline({}, 1)

        with self.assertRaises(TypeError):
            Pipeline({}, SomeFactor())

        with self.assertRaises(TypeError):
            Pipeline({'open': USEquityPricing.open})

        Pipeline({}, SomeFactor() > 5)

    def test_add(self):
        p = Pipeline()
        f = SomeFactor()

        p.add(f, 'f')
        self.assertEqual(p.columns, {'f': f})

        p.add(f > 5, 'g')
        self.assertEqual(p.columns, {'f': f, 'g': f > 5})

        with self.assertRaises(TypeError):
            p.add(f, 1)

        with self.assertRaises(TypeError):
            p.add(USEquityPricing.open, 'open')

    def test_overwrite(self):
        p = Pipeline()
        f = SomeFactor()
        other_f = SomeOtherFactor()

        p.add(f, 'f')
        self.assertEqual(p.columns, {'f': f})

        with self.assertRaises(KeyError) as e:
            p.add(other_f, 'f')
        [message] = e.exception.args
        self.assertEqual(message, "Column 'f' already exists.")

        p.add(other_f, 'f', overwrite=True)
        self.assertEqual(p.columns, {'f': other_f})

    def test_remove(self):
        f = SomeFactor()
        p = Pipeline(columns={'f': f})

        with self.assertRaises(KeyError) as e:
            p.remove('not_a_real_name')

        self.assertEqual(f, p.remove('f'))

        with self.assertRaises(KeyError) as e:
            p.remove('f')

        self.assertEqual(e.exception.args, ('f',))

    def test_set_screen(self):
        f, g = SomeFilter(), SomeOtherFilter()

        p = Pipeline()
        self.assertEqual(p.screen, None)

        p.set_screen(f)
        self.assertEqual(p.screen, f)

        with self.assertRaises(ValueError):
            p.set_screen(f)

        p.set_screen(g, overwrite=True)
        self.assertEqual(p.screen, g)

        with self.assertRaises(TypeError) as e:
            p.set_screen(f, g)

        message = e.exception.args[0]
        self.assertIn(
            "expected a value of type bool or int for argument 'overwrite'",
            message,
        )
