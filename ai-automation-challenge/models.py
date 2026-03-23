from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from enum import Enum


class ViolationType(str, Enum):
    HATE_SPEECH = "hate_speech"
    VIOLENCE = "violence"
    ADULT_CONTENT = "adult_content"
    SPAM = "spam"
    NONE = "none"


class ModerationDecision(str, Enum):
    ALLOW = "allow"
    BLOCK = "block"
    HUMAN_REVIEW = "human_review"


class ModerationRequest(BaseModel):
    """Request model for content moderation"""
    content: str = Field(..., min_length=1)
    creator_id: str
    video_id: Optional[str] = None


class ModerationResult(BaseModel):
    """Structured AI moderation result"""
    is_safe: bool
    decision: ModerationDecision
    requires_human_review: bool
    confidence: float = Field(..., ge=0.0, le=1.0)
    violation_type: ViolationType
    category_scores: Dict[str, float]
    matched_signals: List[str] = Field(default_factory=list)
    review_reason: Optional[str] = None
    reasoning: str
    provider: str  # Primary provider used to score content


class ModerationResponse(BaseModel):
    """API response model"""
    video_id: Optional[str]
    moderation: ModerationResult
    processing_time_ms: float
