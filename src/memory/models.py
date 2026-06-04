from dataclasses import dataclass, field
from enum import Enum
import time
import uuid


class Category(str, Enum):
    PERSONAL     = "personal_information"
    PROFESSIONAL = "professional_details"
    PREFERENCES  = "preferences_interests"
    GOALS        = "goals_aspirations"
    CONTEXTUAL   = "contextual_information"


class Source(str, Enum):
    USER_STATEMENT   = "user_statement"
    AGENT_INFERENCE  = "agent_inference"
    DOCUMENT_EXTRACT = "document_extract"


SOURCE_TRUST: dict[Source, int] = {
    Source.USER_STATEMENT:   3,
    Source.AGENT_INFERENCE:  2,
    Source.DOCUMENT_EXTRACT: 1,
}


@dataclass
class MemoryRecord:
    text:     str
    category: Category
    source:   Source
    id:                  str         = field(default_factory=lambda: str(uuid.uuid4()))
    intent_label:        str         = ""
    entities:            list[str]   = field(default_factory=list)
    contextual_markers:  list[str]   = field(default_factory=list)
    embedding:           list[float] = field(default_factory=list)
    timestamp_created:   float       = field(default_factory=time.time)
    timestamp_updated:   float       = field(default_factory=time.time)
    is_current:          bool        = True
    source_trust:        int         = field(init=False)

    def __post_init__(self) -> None:
        self.source_trust = SOURCE_TRUST[self.source]


@dataclass
class CandidateFact:
    text:                str
    category:            Category
    source:              Source    = Source.USER_STATEMENT
    intent_label:        str       = ""
    entities:            list[str] = field(default_factory=list)
    contextual_markers:  list[str] = field(default_factory=list)
