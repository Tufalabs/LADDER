"""Tests for the batch_generator module."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from pathlib import Path
import tempfile
import json

from ladder.batch_generator import BatchGenerator


class TestBatchGenerator:
    """Test the BatchGenerator class."""

    def test_init_default_params(self):
        """Test BatchGenerator initialization with default parameters."""
        generator = BatchGenerator()
        assert generator.batch_size == 10
        assert generator.difficulties == {"easier": 30, "equivalent": 10}
        assert generator.max_retries == 3
        assert generator.retry_delay == 5

    def test_init_custom_params(self):
        """Test BatchGenerator initialization with custom parameters."""
        custom_difficulties = {"easy": 5, "hard": 15}
        generator = BatchGenerator(
            batch_size=5,
            difficulties=custom_difficulties,
            max_retries=2,
            retry_delay=10,
        )
        assert generator.batch_size == 5
        assert generator.difficulties == custom_difficulties
        assert generator.max_retries == 2
        assert generator.retry_delay == 10

    @pytest.mark.asyncio
    @patch('ladder.batch_generator.process_integral')
    async def test_process_batch_success(self, mock_process_integral):
        """Test successful batch processing."""
        # Mock the process_integral function
        mock_process_integral.return_value = [
            {"variant": "integrate(x**2, x)", "difficulty": "easier"}
        ]
        
        generator = BatchGenerator(batch_size=2, difficulties={"easier": 1})
        integrals = ["integrate(x, x)", "integrate(x**2, x)"]
        
        results = await generator.process_batch(integrals)
        
        # Should have called process_integral twice (once per integral)
        assert mock_process_integral.call_count == 2
        assert len(results) == 2  # 2 integrals * 1 result each

    @pytest.mark.asyncio
    @patch('ladder.batch_generator.process_integral')
    async def test_process_batch_retry_success(self, mock_process_integral):
        """Test batch processing with retry mechanism."""
        # First call fails, second succeeds
        mock_process_integral.side_effect = [
            Exception("API error"),
            [{"variant": "integrate(x**2, x)", "difficulty": "easier"}]
        ]
        
        generator = BatchGenerator(
            batch_size=1,
            difficulties={"easier": 1},
            max_retries=2,
            retry_delay=0.1  # Short delay for testing
        )
        integrals = ["integrate(x, x)"]
        
        results = await generator.process_batch(integrals)
        
        # Should have been called twice (initial + retry)
        assert mock_process_integral.call_count == 2
        assert len(results) == 1

    @pytest.mark.asyncio
    @patch('ladder.batch_generator.process_integral')
    async def test_process_batch_max_retries_exceeded(self, mock_process_integral):
        """Test batch processing when max retries are exceeded."""
        # Always fail
        mock_process_integral.side_effect = Exception("Persistent error")
        
        generator = BatchGenerator(
            batch_size=1,
            difficulties={"easier": 1},
            max_retries=2,
            retry_delay=0.1
        )
        integrals = ["integrate(x, x)"]
        
        with pytest.raises(Exception, match="Persistent error"):
            await generator.process_batch(integrals)
        
        # Should have been called max_retries times
        assert mock_process_integral.call_count == 2

    @pytest.mark.asyncio
    @patch('ladder.batch_generator.process_integral')
    async def test_process_all_integrals(self, mock_process_integral):
        """Test processing all integrals with file saving."""
        mock_process_integral.return_value = [
            {"variant": "integrate(x**2, x)", "difficulty": "easier"}
        ]
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            generator = BatchGenerator(batch_size=2, difficulties={"easier": 1})
            integrals = ["integrate(x, x)", "integrate(x**2, x)", "integrate(x**3, x)"]
            
            results = await generator.process_all_integrals(
                integrals,
                output_dir=tmp_dir,
                save_individual=True,
                save_combined=True,
            )
            
            # Check results
            assert len(results) == 3  # 3 integrals * 1 result each
            
            # Check files were created
            output_path = Path(tmp_dir)
            json_files = list(output_path.glob("*.json"))
            
            # Should have batch files and combined file
            batch_files = [f for f in json_files if "batch" in f.name]
            combined_files = [f for f in json_files if "all" in f.name]
            
            assert len(batch_files) == 2  # 2 batches (3 integrals with batch_size=2)
            assert len(combined_files) == 1

    def test_print_summary_empty(self, capsys):
        """Test summary printing with empty variants list."""
        generator = BatchGenerator()
        generator.print_summary([])
        
        captured = capsys.readouterr()
        assert "No variants generated." in captured.out

    def test_print_summary_with_variants(self, capsys):
        """Test summary printing with variants."""
        variants = [
            {
                "requested_difficulty": "easier",
                "verification_passed": True,
            },
            {
                "requested_difficulty": "harder",
                "verification_passed": False,
            },
            {
                "requested_difficulty": "easier",
                "verification_passed": None,
            },
        ]
        
        generator = BatchGenerator()
        generator.print_summary(variants)
        
        captured = capsys.readouterr()
        output = captured.out
        
        assert "Total variants: 3" in output
        assert "easier: 2" in output
        assert "harder: 1" in output
        assert "passed: 1" in output
        assert "failed: 1" in output
        assert "unknown: 1" in output