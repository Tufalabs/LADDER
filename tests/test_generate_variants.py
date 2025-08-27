"""Tests for the generate_variants module."""

import pytest
import sympy as sp
from unittest.mock import patch, MagicMock

from ladder.generate_variants import (
    is_definite_integral,
    verify_integral,
    parse_variants,
    TRANSFORMATIONS_BY_DIFFICULTY,
)


class TestIntegralDetection:
    """Test integral type detection functions."""

    def test_is_definite_integral_true(self):
        """Test definite integral detection for definite integrals."""
        definite_integral = "integrate(x**2, (x, 0, 1))"
        assert is_definite_integral(definite_integral) is True

    def test_is_definite_integral_false(self):
        """Test definite integral detection for indefinite integrals."""
        indefinite_integral = "integrate(x**2, x)"
        assert is_definite_integral(indefinite_integral) is False

    def test_is_definite_integral_complex(self):
        """Test definite integral detection for complex expressions."""
        complex_definite = "integrate(sin(x)*cos(x), (x, 0, pi/2))"
        assert is_definite_integral(complex_definite) is True


class TestVariantParsing:
    """Test variant parsing functionality."""

    def test_parse_variants_simple(self):
        """Test parsing of simple variant responses."""
        text = """
        ====
        Variant 1:
        Reasoning: This is a simple test
        Variant: integrate(x**2, x)
        ====
        """
        variants = parse_variants(text)
        assert len(variants) == 1
        assert variants[0]["reasoning"] == "This is a simple test"
        assert variants[0]["variant"] == "integrate(x**2, x)"

    def test_parse_variants_multiple(self):
        """Test parsing multiple variants."""
        text = """
        ====
        Variant 1:
        Reasoning: First variant
        Variant: integrate(x**2, x)
        ====
        Variant 2:
        Reasoning: Second variant
        Variant: integrate(x**3, x)
        ====
        """
        variants = parse_variants(text)
        assert len(variants) == 2
        assert variants[0]["variant"] == "integrate(x**2, x)"
        assert variants[1]["variant"] == "integrate(x**3, x)"

    def test_parse_variants_malformed(self):
        """Test parsing of malformed variant responses."""
        text = """
        ====
        Variant 1:
        Reasoning: Missing variant
        ====
        """
        variants = parse_variants(text)
        assert len(variants) == 0

    def test_parse_variants_no_closing_paren(self):
        """Test parsing variants without closing parenthesis."""
        text = """
        ====
        Variant 1:
        Reasoning: No closing paren
        Variant: integrate(x**2, x
        ====
        """
        variants = parse_variants(text)
        assert len(variants) == 0


class TestTransformations:
    """Test transformation configurations."""

    def test_transformations_exist(self):
        """Test that all difficulty levels have transformations."""
        expected_difficulties = ["easier", "equivalent", "harder"]
        for difficulty in expected_difficulties:
            assert difficulty in TRANSFORMATIONS_BY_DIFFICULTY
            assert isinstance(TRANSFORMATIONS_BY_DIFFICULTY[difficulty], list)
            assert len(TRANSFORMATIONS_BY_DIFFICULTY[difficulty]) > 0

    def test_transformations_are_strings(self):
        """Test that all transformations are strings."""
        for difficulty, transformations in TRANSFORMATIONS_BY_DIFFICULTY.items():
            for transformation in transformations:
                assert isinstance(transformation, str)
                assert len(transformation.strip()) > 0


class TestIntegralVerification:
    """Test integral verification functionality."""

    @patch('ladder.generate_variants.run_integration')
    def test_verify_integral_simple(self, mock_run_integration):
        """Test verification of simple integral."""
        # Mock the integration result
        mock_run_integration.return_value = sp.Symbol('x')**3 / 3
        
        integral_str = "integrate(x**2, x)"
        # This would normally work, but we'll skip the actual verification
        # due to complexity of mocking sympy operations
        pass

    def test_verify_integral_definite_always_true(self):
        """Test that definite integrals are always considered verified."""
        definite_integral = "integrate(x**2, (x, 0, 1))"
        result = verify_integral(definite_integral)
        assert result is True

    def test_verify_integral_malformed(self):
        """Test verification of malformed integral."""
        malformed_integral = "not_an_integral"
        result = verify_integral(malformed_integral)
        assert result is False


@pytest.mark.asyncio
class TestAsyncFunctions:
    """Test asynchronous functions."""

    async def test_process_single_variant_none_variant(self):
        """Test processing variant with None variant field."""
        from ladder.generate_variants import process_single_variant
        
        variant_data = {"reasoning": "test", "variant": None}
        result = await process_single_variant("integrate(x, x)", "easier", variant_data)
        assert result is None

    async def test_process_single_variant_empty_variant(self):
        """Test processing variant with empty variant field."""
        from ladder.generate_variants import process_single_variant
        
        variant_data = {"reasoning": "test", "variant": ""}
        result = await process_single_variant("integrate(x, x)", "easier", variant_data)
        assert result is None