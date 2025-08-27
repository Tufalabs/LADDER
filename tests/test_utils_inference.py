"""Tests for the utils.inference module."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import os

from ladder.utils.inference import (
    generate_text,
    generate_text_with_fallback,
    get_available_models,
    test_model_connection,
)


class TestInferenceUtils:
    """Test inference utility functions."""

    @patch.dict(os.environ, {
        'OPENAI_API_KEY': 'test_openai_key',
        'ANTHROPIC_API_KEY': 'test_anthropic_key',
        'DEEPSEEK_API_KEY': 'test_deepseek_key',
    })
    def test_get_available_models(self):
        """Test getting available models based on API keys."""
        # Need to reload the module to pick up the patched environment
        import importlib
        import ladder.utils.inference
        importlib.reload(ladder.utils.inference)
        
        models = ladder.utils.inference.get_available_models()
        assert models["openai"] is True
        assert models["anthropic"] is True
        assert models["deepseek"] is True

    @patch.dict(os.environ, {}, clear=True)
    def test_get_available_models_no_keys(self):
        """Test getting available models with no API keys."""
        import importlib
        import ladder.utils.inference
        importlib.reload(ladder.utils.inference)
        
        models = ladder.utils.inference.get_available_models()
        assert models["openai"] is False
        assert models["anthropic"] is False
        assert models["deepseek"] is False

    @pytest.mark.asyncio
    @patch('ladder.utils.inference.async_openai_client')
    async def test_generate_text_openai_success(self, mock_client):
        """Test successful text generation with OpenAI."""
        # Mock the response
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Generated text"
        mock_client.chat.completions.create.return_value = mock_response
        
        result = await generate_text("gpt-4", "Test prompt")
        
        assert result == "Generated text"
        mock_client.chat.completions.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_text_unsupported_model(self):
        """Test text generation with unsupported model."""
        with pytest.raises(ValueError, match="Unsupported model"):
            await generate_text("unsupported-model", "Test prompt")

    @pytest.mark.asyncio
    @patch('ladder.utils.inference.async_openai_client', None)
    async def test_generate_text_no_api_key(self):
        """Test text generation without API key."""
        with pytest.raises(ValueError, match="OpenAI API key not found"):
            await generate_text("gpt-4", "Test prompt")

    @pytest.mark.asyncio
    @patch('ladder.utils.inference.generate_text')
    async def test_generate_text_with_fallback_success(self, mock_generate):
        """Test fallback mechanism with successful first model."""
        mock_generate.return_value = "Generated text"
        
        result, model_used = await generate_text_with_fallback(
            ["gpt-4", "gpt-3.5-turbo"],
            "Test prompt"
        )
        
        assert result == "Generated text"
        assert model_used == "gpt-4"
        mock_generate.assert_called_once_with("gpt-4", "Test prompt", 8000, 0.0)

    @pytest.mark.asyncio
    @patch('ladder.utils.inference.generate_text')
    async def test_generate_text_with_fallback_retry(self, mock_generate):
        """Test fallback mechanism with first model failing."""
        # First call fails, second succeeds
        mock_generate.side_effect = [
            Exception("First model failed"),
            "Generated text from second model"
        ]
        
        result, model_used = await generate_text_with_fallback(
            ["gpt-4", "gpt-3.5-turbo"],
            "Test prompt"
        )
        
        assert result == "Generated text from second model"
        assert model_used == "gpt-3.5-turbo"
        assert mock_generate.call_count == 2

    @pytest.mark.asyncio
    @patch('ladder.utils.inference.generate_text')
    async def test_generate_text_with_fallback_all_fail(self, mock_generate):
        """Test fallback mechanism when all models fail."""
        mock_generate.side_effect = Exception("All models failed")
        
        with pytest.raises(Exception, match="All models failed"):
            await generate_text_with_fallback(
                ["gpt-4", "gpt-3.5-turbo"],
                "Test prompt"
            )
        
        assert mock_generate.call_count == 2

    @pytest.mark.asyncio
    @patch('ladder.utils.inference.generate_text')
    async def test_model_connection_success(self, mock_generate):
        """Test model connection testing with successful connection."""
        mock_generate.return_value = "Hello response"
        
        result = await test_model_connection("gpt-4")
        
        assert result is True
        mock_generate.assert_called_once_with(
            "gpt-4", "Hello, world!", max_tokens=10, temperature=0
        )

    @pytest.mark.asyncio
    @patch('ladder.utils.inference.generate_text')
    async def test_model_connection_failure(self, mock_generate):
        """Test model connection testing with failed connection."""
        mock_generate.side_effect = Exception("Connection failed")
        
        result = await test_model_connection("gpt-4")
        
        assert result is False