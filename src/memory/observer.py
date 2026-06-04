import json
import re

import anthropic

from .models import CandidateFact, Category, Source

_client = anthropic.AsyncAnthropic()

_CATEGORIES = " | ".join(c.value for c in Category)

_PROMPT = """\
You are a memory extraction assistant for a personal AI assistant.

Read the conversation turn below. Extract ONLY facts that should be remembered durably about this user across future sessions — things that make the assistant more useful next time.

Extract from EXACTLY these five categories:
  personal_information   — name, location, age, life situation
  professional_details   — job title, company, tech stack, languages, active projects
  preferences_interests  — stated likes/dislikes, coding style, communication style, preferences
  goals_aspirations      — explicit goals, learning targets, career plans
  contextual_information — active project constraints, key decisions, deadlines

NEVER extract:
  - Greetings, thanks, filler ("sounds good", "got it", "thanks")
  - Temporary task state ("fix bug on line 42", "run this command")
  - Code snippets, command outputs, or intermediate reasoning
  - Credentials, API keys, passwords, tokens, SSNs, credit card numbers — even if the user shares them
  - Facts about third parties — things a colleague, friend, or family member does, uses, or prefers (e.g. "My colleague uses Vim", "My mum is a teacher"). Extract only facts directly about the user themselves.
  - Hypothetical or counterfactual statements using conditional framing ("if I were...", "suppose I...", "imagine if I...", "If I had become...", "what if I..."). These describe non-real scenarios, not durable facts.
  - Statements made while roleplaying or speaking as a named character or fictional persona ("As [character name], ...", "In character:", "Playing as..."). Check the prior context below to detect whether a roleplay is currently active.
  - Facts the ASSISTANT stated (only extract what the USER said about themselves)

If nothing is worth storing, return: []

Return ONLY a valid JSON array — no prose, no markdown code fences:
[
  {{
    "text": "<single atomic fact, one concise sentence>",
    "category": "<one of the five categories>",
    "source": "user_statement",
    "intent_label": "<short_snake_case_tag>",
    "entities": ["<named entity>"],
    "contextual_markers": ["<situational context tag, e.g. 'work_context', 'side_project'>"]
  }}
]
{context_section}
Conversation turn to evaluate:
User: {user_turn}
Assistant: {assistant_turn}
"""


def _format_context_section(prior_context: str) -> str:
    """Format prior turns as an extraction-hint block, or return empty string."""
    if not prior_context:
        return ""
    return (
        "\nPrior context (use ONLY to detect active roleplay or hypothetical framing — "
        "do NOT extract facts from this section):\n"
        f"{prior_context}\n"
    )


async def observe(user_turn: str, assistant_turn: str, prior_context: str = "") -> list[CandidateFact]:
    response = await _client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": _PROMPT.format(
                context_section=_format_context_section(prior_context),
                user_turn=user_turn,
                assistant_turn=assistant_turn,
            ),
        }],
    )
    raw = response.content[0].text.strip()

    # Parse JSON — handle model adding prose around the array
    try:
        items = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if not match:
            return []
        try:
            items = json.loads(match.group())
        except json.JSONDecodeError:
            return []

    if not isinstance(items, list):
        return []

    facts: list[CandidateFact] = []
    for item in items:
        try:
            facts.append(CandidateFact(
                text=str(item["text"]),
                category=Category(item["category"]),
                source=Source(item.get("source", "user_statement")),
                intent_label=str(item.get("intent_label", "")),
                entities=[str(e) for e in item.get("entities", [])],
                contextual_markers=[str(m) for m in item.get("contextual_markers", [])],
            ))
        except (KeyError, ValueError):
            continue

    return facts
