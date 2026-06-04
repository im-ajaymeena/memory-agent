import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.memory.models import Category, Source
from src.memory.observer import observe


def _mock_response(text: str) -> MagicMock:
    msg = MagicMock()
    msg.content = [MagicMock(text=text)]
    return msg


@pytest.mark.asyncio
async def test_returns_empty_for_noise() -> None:
    with patch("src.memory.observer._client") as mock_client:
        mock_client.messages.create = AsyncMock(return_value=_mock_response("[]"))
        result = await observe("Thanks!", "You're welcome!")
    assert result == []


@pytest.mark.asyncio
async def test_extracts_preference() -> None:
    payload = """[{"text": "User prefers Python for scripting.",
                   "category": "preferences_interests",
                   "source": "user_statement",
                   "intent_label": "tech_preference",
                   "entities": ["Python"]}]"""
    with patch("src.memory.observer._client") as mock_client:
        mock_client.messages.create = AsyncMock(return_value=_mock_response(payload))
        result = await observe("I prefer Python for scripting.", "Got it.")
    assert len(result) == 1
    assert result[0].category == Category.PREFERENCES
    assert result[0].source == Source.USER_STATEMENT
    assert "Python" in result[0].entities


@pytest.mark.asyncio
async def test_handles_model_prose_around_json() -> None:
    payload = 'Sure! Here is the result:\n[{"text": "User is a backend engineer.", "category": "professional_details", "source": "user_statement", "intent_label": "job_role", "entities": []}]'
    with patch("src.memory.observer._client") as mock_client:
        mock_client.messages.create = AsyncMock(return_value=_mock_response(payload))
        result = await observe("I'm a backend engineer.", "Nice.")
    assert len(result) == 1
    assert result[0].category == Category.PROFESSIONAL


@pytest.mark.asyncio
async def test_skips_malformed_items() -> None:
    payload = '[{"text": "Valid fact.", "category": "goals_aspirations", "source": "user_statement", "intent_label": "goal", "entities": []}, {"bad": "no category field"}]'
    with patch("src.memory.observer._client") as mock_client:
        mock_client.messages.create = AsyncMock(return_value=_mock_response(payload))
        result = await observe("I want to learn Rust.", "")
    assert len(result) == 1


@pytest.mark.asyncio
async def test_invalid_category_skipped() -> None:
    payload = '[{"text": "Some fact.", "category": "made_up_category", "source": "user_statement", "intent_label": "", "entities": []}]'
    with patch("src.memory.observer._client") as mock_client:
        mock_client.messages.create = AsyncMock(return_value=_mock_response(payload))
        result = await observe("Something.", "OK.")
    assert result == []


@pytest.mark.asyncio
async def test_returns_empty_on_total_parse_failure() -> None:
    with patch("src.memory.observer._client") as mock_client:
        mock_client.messages.create = AsyncMock(return_value=_mock_response("not json at all"))
        result = await observe("Hello!", "Hi there!")
    assert result == []
