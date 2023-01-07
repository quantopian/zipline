"""Tests for zipline.pipeline.Pipeline"""

from unittest import mock

from zipline.pipeline import Factor, Filter, Pipeline
from zipline.pipeline.data import Column, DataSet, USEquityPricing
from zipline.pipeline.domain import (
    AmbiguousDomain,
    CA_EQUITIES,
    GENERIC,
    GB_EQUITIES,
    US_EQUITIES,
)
from zipline.pipeline.graph import display_graph
from zipline.utils.compat import getargspec
from zipline.utils.numpy_utils import float64_dtype
import pytest


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


class TestPipelineTestCase:
    def test_construction(self):
        p0 = Pipeline()
        assert p0.columns == {}
        assert p0.screen is None

        columns = {"f": SomeFactor()}
        p1 = Pipeline(columns=columns)
        assert p1.columns == columns

        screen = SomeFilter()
        p2 = Pipeline(screen=screen)
        assert p2.columns == {}
        assert p2.screen == screen

        p3 = Pipeline(columns=columns, screen=screen)
        assert p3.columns == columns
        assert p3.screen == screen

    def test_construction_bad_input_types(self):

        with pytest.raises(TypeError):
            Pipeline(1)

        Pipeline({})

        with pytest.raises(TypeError):
            Pipeline({}, 1)

        with pytest.raises(TypeError):
            Pipeline({}, SomeFactor())

        with pytest.raises(TypeError):
            Pipeline({"open": USEquityPricing.open})

        Pipeline({}, SomeFactor() > 5)

    def test_add(self):
        p = Pipeline()
        f = SomeFactor()

        p.add(f, "f")
        assert p.columns == {"f": f}

        p.add(f > 5, "g")
        assert p.columns == {"f": f, "g": f > 5}

        with pytest.raises(TypeError):
            p.add(f, 1)

        with pytest.raises(TypeError):
            p.add(USEquityPricing.open, "open")

    def test_overwrite(self):
        p = Pipeline()
        f = SomeFactor()
        other_f = SomeOtherFactor()

        p.add(f, "f")
        assert p.columns == {"f": f}

        with pytest.raises(KeyError, match="Column 'f' already exists."):
            p.add(other_f, "f")

        p.add(other_f, "f", overwrite=True)
        assert p.columns == {"f": other_f}

    def test_remove(self):
        f = SomeFactor()
        p = Pipeline(columns={"f": f})

        with pytest.raises(KeyError):
            p.remove("not_a_real_name")

        assert f == p.remove("f")

        with pytest.raises(KeyError, match="f"):
            p.remove("f")

    def test_set_screen(self):
        f, g = SomeFilter(), SomeOtherFilter()

        p = Pipeline()
        assert p.screen is None

        p.set_screen(f)
        assert p.screen == f

        with pytest.raises(ValueError):
            p.set_screen(f)

        p.set_screen(g, overwrite=True)
        assert p.screen == g

        with pytest.raises(
            TypeError,
            match="expected a value of type bool or int for argument 'overwrite'",
        ):
            p.set_screen(f, g)

    def test_show_graph(self):
        f = SomeFactor()
        p = Pipeline(columns={"f": SomeFactor()})

        # The real display_graph call shells out to GraphViz, which isn't a
        # requirement, so patch it out for testing.

        def mock_display_graph(g, format="svg", include_asset_exists=False):
            return (g, format, include_asset_exists)

        assert getargspec(display_graph) == getargspec(
            mock_display_graph
        ), "Mock signature doesn't match signature for display_graph."

        patch_display_graph = mock.patch(
            "zipline.pipeline.graph.display_graph",
            mock_display_graph,
        )

        with patch_display_graph:
            graph, format, include_asset_exists = p.show_graph()
            assert graph.outputs["f"] is f
            # '' is a sentinel used for screen if it's not supplied.
            assert sorted(graph.outputs.keys()) == ["f", graph.screen_name]
            assert format == "svg"
            assert include_asset_exists is False

        with patch_display_graph:
            graph, format, include_asset_exists = p.show_graph(format="png")
            assert graph.outputs["f"] is f
            # '' is a sentinel used for screen if it's not supplied.
            assert sorted(graph.outputs.keys()) == ["f", graph.screen_name]
            assert format == "png"
            assert include_asset_exists is False

        with patch_display_graph:
            graph, format, include_asset_exists = p.show_graph(format="jpeg")
            assert graph.outputs["f"] is f
            assert sorted(graph.outputs.keys()) == ["f", graph.screen_name]
            assert format == "jpeg"
            assert include_asset_exists is False

        expected = (
            r".*\.show_graph\(\) expected a value in "
            r"\('svg', 'png', 'jpeg'\) for argument 'format', "
            r"but got 'fizzbuzz' instead."
        )

        with pytest.raises(ValueError, match=expected):
            p.show_graph(format="fizzbuzz")

    @pytest.mark.parametrize(
        "domain",
        [GENERIC, US_EQUITIES],
        ids=["generic", "us_equities"],
    )
    def test_infer_domain_no_terms(self, domain):
        assert Pipeline().domain(default=domain) == domain

    def test_infer_domain_screen_only(self):
        class D(DataSet):
            c = Column(bool)

        filter_generic = D.c.latest
        filter_US = D.c.specialize(US_EQUITIES).latest
        filter_CA = D.c.specialize(CA_EQUITIES).latest

        assert (
            Pipeline(screen=filter_generic).domain(default=GB_EQUITIES) == GB_EQUITIES
        )
        assert Pipeline(screen=filter_US).domain(default=GB_EQUITIES) == US_EQUITIES
        assert Pipeline(screen=filter_CA).domain(default=GB_EQUITIES) == CA_EQUITIES

    def test_infer_domain_outputs(self):
        class D(DataSet):
            c = Column(float)

        D_US = D.specialize(US_EQUITIES)
        D_CA = D.specialize(CA_EQUITIES)

        result = Pipeline({"f": D_US.c.latest}).domain(default=GB_EQUITIES)
        expected = US_EQUITIES
        assert result == expected

        result = Pipeline({"f": D_CA.c.latest}).domain(default=GB_EQUITIES)
        expected = CA_EQUITIES
        assert result == expected

    def test_conflict_between_outputs(self):
        class D(DataSet):
            c = Column(float)

        D_US = D.specialize(US_EQUITIES)
        D_CA = D.specialize(CA_EQUITIES)

        pipe = Pipeline({"f": D_US.c.latest, "g": D_CA.c.latest})
        with pytest.raises(AmbiguousDomain) as excinfo:
            pipe.domain(default=GENERIC)

        assert excinfo.value.domains == [CA_EQUITIES, US_EQUITIES]

    def test_conflict_between_output_and_screen(self):
        class D(DataSet):
            c = Column(float)
            b = Column(bool)

        D_US = D.specialize(US_EQUITIES)
        D_CA = D.specialize(CA_EQUITIES)

        pipe = Pipeline({"f": D_US.c.latest}, screen=D_CA.b.latest)
        with pytest.raises(AmbiguousDomain) as excinfo:
            pipe.domain(default=GENERIC)

        assert excinfo.value.domains == [CA_EQUITIES, US_EQUITIES]
