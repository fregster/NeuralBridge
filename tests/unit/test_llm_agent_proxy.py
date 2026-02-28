"""Tests for LLMAgentProxy — focused on delegation and proxy methods."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from custom_components.neuralbridge.llm_agent_proxy import LLMAgentProxy


def _make_proxy() -> LLMAgentProxy:
    """Create an LLMAgentProxy with all dependencies mocked."""
    hass = MagicMock()
    return LLMAgentProxy(
        hass=hass,
        config_getter=MagicMock(return_value={}),
        entity_context_cache=MagicMock(),
        session_memory=MagicMock(),
        preference_memory=None,
        ollama_clients={},
        benchmarker_getter=MagicMock(return_value=None),
    )


class TestRenderHaContext:
    """Tests for LLMAgentProxy._render_ha_context."""

    def test_render_ha_context_delegates_to_prompt_builder(self) -> None:
        """_render_ha_context proxies through to PromptBuilder.render_ha_context."""
        proxy = _make_proxy()
        expected = "rendered HA context prompt"

        with patch.object(
            proxy._prompt_builder,
            "render_ha_context",
            return_value=expected,
        ) as mock_render:
            result = proxy._render_ha_context("raw prompt text")

        mock_render.assert_called_once_with("raw prompt text")
        assert result == expected

    def test_render_ha_context_passes_prompt_unchanged(self) -> None:
        """The prompt string is passed as-is to PromptBuilder.render_ha_context."""
        proxy = _make_proxy()
        prompt = "  some prompt with spaces  "

        with patch.object(
            proxy._prompt_builder,
            "render_ha_context",
            return_value=prompt,
        ):
            result = proxy._render_ha_context(prompt)

        assert result == prompt
