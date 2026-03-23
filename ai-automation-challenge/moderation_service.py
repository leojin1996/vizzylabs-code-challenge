import json
from typing import Dict, List, Optional, Tuple

from models import (
    ModerationDecision,
    ModerationRequest,
    ModerationResult,
    ViolationType,
)
from mock_clients import MockAnthropicClient, MockOpenAIClient


class ModerationService:
    """
    Content moderation service with explicit risk tiers and human review routing.
    """

    # Simplified threshold design: two clear boundaries
    HIGH_RISK_THRESHOLD = 0.80  # >= 0.80: Auto-block
    MEDIUM_RISK_THRESHOLD = 0.40  # 0.40-0.80: Human review zone
    # < 0.40: Auto-allow (unless rules trigger)

    # For false-negative rules: require higher confidence to auto-block
    FALSE_NEGATIVE_BLOCK_THRESHOLD = 0.65

    CATEGORY_MAP = {
        "hate": ViolationType.HATE_SPEECH,
        "violence": ViolationType.VIOLENCE,
        "sexual": ViolationType.ADULT_CONTENT,
        "spam": ViolationType.SPAM,
    }

    FALSE_POSITIVE_RULES = (
        {
            "name": "cooking_context",
            "category": "violence",
            "risk_terms": ("chop", "slice", "dice", "cut", "knife", "butcher"),
            "context_terms": ("cook", "recipe", "kitchen", "vegetable", "food", "meat"),
            "min_risk_matches": 1,  # At least 1 risk term
            "min_context_matches": 1,  # At least 1 context term
            "review_reason": "Violence signal appears within cooking context.",
        },
        {
            "name": "fitness_context",
            "category": "sexual",
            "risk_terms": ("shirtless", "abs", "body", "sweaty", "workout"),
            "context_terms": ("fitness", "gym", "exercise", "training"),
            "min_risk_matches": 1,
            "min_context_matches": 1,
            "review_reason": "Adult-content signal appears within fitness context.",
        },
        {
            "name": "medical_context",
            "category": "violence",
            "risk_terms": ("blood", "surgery", "injection", "wound"),
            "context_terms": ("doctor", "medical", "health", "nurse"),
            "min_risk_matches": 1,
            "min_context_matches": 1,
            "review_reason": "Violence signal appears within medical context.",
        },
    )

    FALSE_NEGATIVE_RULES = (
        {
            "name": "supplement_scam_pattern",
            "category": "spam",
            "risk_terms": ("miracle", "secret", "doctors hate", "one weird trick"),
            "context_terms": ("weight loss", "muscle", "energy", "supplement"),
            "min_risk_matches": 1,  # At least 1 scam phrase
            "min_context_matches": 1,  # At least 1 supplement context
            "review_reason": "Supplement scam pattern detected despite low spam score.",
        },
        {
            "name": "coded_hate_pattern",
            "category": "hate",
            "risk_terms": ("those people", "you know who", "certain types"),
            "context_terms": (),
            "min_risk_matches": 1,
            "min_context_matches": 0,  # No context required
            "review_reason": "Potential coded hate speech detected.",
        },
    )

    def __init__(self, openai_key: str, anthropic_key: str):
        self.openai_client = MockOpenAIClient(api_key=openai_key)
        self.anthropic_client = MockAnthropicClient(api_key=anthropic_key)

    async def moderate_content(self, request: ModerationRequest) -> ModerationResult:
        """Moderate content using layered automated review and human-review routing."""
        response = await self.openai_client.moderations.create(input=request.content)
        result = response.results[0]

        scores = self._extract_scores(result.category_scores)
        max_category, max_score = self._get_max_category(scores)

        false_positive_hits = self._evaluate_rules(request.content, self.FALSE_POSITIVE_RULES)
        false_negative_hits = self._evaluate_rules(request.content, self.FALSE_NEGATIVE_RULES)
        matched_signals = [hit["name"] for hit in false_positive_hits + false_negative_hits]

        decision, violation_type, review_reason = self._determine_decision(
            result_flagged=result.flagged,
            scores=scores,
            max_category=max_category,
            max_score=max_score,
            false_positive_hits=false_positive_hits,
            false_negative_hits=false_negative_hits,
        )

        reasoning_parts = [
            f"Top category={max_category} score={max_score:.2f}.",
            self._decision_reason_text(decision, result.flagged, max_score),
        ]

        if matched_signals:
            reasoning_parts.append(f"Matched signals: {', '.join(matched_signals)}.")

        # Only call secondary review when rules are triggered (more nuanced cases)
        provider = "openai"
        if decision == ModerationDecision.HUMAN_REVIEW and matched_signals:
            secondary_review = await self._secondary_review(
                content=request.content,
                scores=scores,
                matched_signals=matched_signals,
            )
            reasoning_parts.append(
                "Secondary review: "
                f"{secondary_review['reasoning']} "
                f"Notes: {secondary_review['context_notes']}"
            )
            provider = "openai+anthropic"

        return ModerationResult(
            is_safe=decision == ModerationDecision.ALLOW,
            decision=decision,
            requires_human_review=decision == ModerationDecision.HUMAN_REVIEW,
            confidence=max_score,
            violation_type=violation_type,
            category_scores=scores,
            matched_signals=matched_signals,
            review_reason=review_reason,
            reasoning=" ".join(reasoning_parts),
            provider=provider,
        )

    def _extract_scores(self, category_scores: object) -> Dict[str, float]:
        return {
            "hate": float(getattr(category_scores, "hate", 0.0)),
            "violence": float(getattr(category_scores, "violence", 0.0)),
            "sexual": float(getattr(category_scores, "sexual", 0.0)),
            "spam": float(getattr(category_scores, "spam", 0.0)),
        }

    def _get_max_category(self, scores: Dict[str, float]) -> Tuple[str, float]:
        max_category = max(scores, key=scores.get)
        return max_category, scores[max_category]

    def _evaluate_rules(self, content: str, rules: Tuple[Dict[str, object], ...]) -> List[Dict[str, object]]:
        """Evaluate rules with improved matching logic to reduce false positives."""
        text = content.lower()
        hits: List[Dict[str, object]] = []

        for rule in rules:
            risk_terms = rule["risk_terms"]
            context_terms = rule["context_terms"]
            min_risk = rule.get("min_risk_matches", 1)
            min_context = rule.get("min_context_matches", 1)

            # Count how many terms match
            risk_matches = sum(1 for term in risk_terms if term in text)
            context_matches = sum(1 for term in context_terms if term in text) if context_terms else 0

            # Rule triggers if both thresholds are met
            if risk_matches >= min_risk and (not context_terms or context_matches >= min_context):
                hits.append(rule)

        return hits

    def _determine_decision(
        self,
        result_flagged: bool,
        scores: Dict[str, float],
        max_category: str,
        max_score: float,
        false_positive_hits: List[Dict[str, object]],
        false_negative_hits: List[Dict[str, object]],
    ) -> Tuple[ModerationDecision, ViolationType, Optional[str]]:
        """
        Simplified decision logic with clear priority order:
        1. High-risk auto-block
        2. False-negative rules (catch what AI missed)
        3. False-positive rules (prevent over-blocking)
        4. Medium-risk zone (human review)
        5. Low-risk auto-allow
        """

        # Priority 1: High-risk content - auto-block
        if max_score >= self.HIGH_RISK_THRESHOLD:
            return (
                ModerationDecision.BLOCK,
                self.CATEGORY_MAP.get(max_category, ViolationType.NONE),
                None,
            )

        # Priority 2: False-negative rules - catch what AI missed
        if false_negative_hits:
            dominant_hit = max(
                false_negative_hits,
                key=lambda hit: scores[str(hit["category"])],
            )
            category = str(dominant_hit["category"])
            category_score = scores[category]

            # Block if confidence is high enough
            if category_score >= self.FALSE_NEGATIVE_BLOCK_THRESHOLD:
                return (
                    ModerationDecision.BLOCK,
                    self.CATEGORY_MAP.get(category, ViolationType.NONE),
                    None,
                )
            # Otherwise route to human review
            return (
                ModerationDecision.HUMAN_REVIEW,
                self.CATEGORY_MAP.get(category, ViolationType.NONE),
                str(dominant_hit["review_reason"]),
            )

        # Priority 3: False-positive rules - prevent over-blocking
        if false_positive_hits:
            dominant_hit = max(
                false_positive_hits,
                key=lambda hit: scores[str(hit["category"])],
            )
            return (
                ModerationDecision.HUMAN_REVIEW,
                self.CATEGORY_MAP.get(str(dominant_hit["category"]), ViolationType.NONE),
                str(dominant_hit["review_reason"]),
            )

        # Priority 4: Medium-risk zone - human review
        if max_score >= self.MEDIUM_RISK_THRESHOLD:
            return (
                ModerationDecision.HUMAN_REVIEW,
                self.CATEGORY_MAP.get(max_category, ViolationType.NONE),
                f"Content in medium-risk zone (score={max_score:.2f}).",
            )

        # Priority 5: Low-risk - auto-allow
        return ModerationDecision.ALLOW, ViolationType.NONE, None

    def _decision_reason_text(
        self,
        decision: ModerationDecision,
        result_flagged: bool,
        max_score: float,
    ) -> str:
        if decision == ModerationDecision.BLOCK:
            return "Decision=block due to high-risk confidence."
        if decision == ModerationDecision.HUMAN_REVIEW:
            if result_flagged:
                return (
                    "Decision=human_review because automated flagging was ambiguous "
                    f"at score {max_score:.2f}."
                )
            return (
                "Decision=human_review because rules detected harmful or contextual "
                "signals that need manual verification."
            )
        return "Decision=allow because no rule or score exceeded review thresholds."

    async def _secondary_review(
        self,
        content: str,
        scores: Dict[str, float],
        matched_signals: List[str],
    ) -> Dict[str, str]:
        prompt = json.dumps(
            {
                "content": content,
                "scores": scores,
                "matched_signals": matched_signals,
                "task": (
                    "Provide review guidance for a moderation candidate. "
                    "Do not override high-risk blocking decisions."
                ),
            }
        )
        response = await self.anthropic_client.messages.create(
            model="claude-3-5-sonnet-20241022",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
        )
        review_text = response.content[0].text
        parsed = json.loads(review_text)
        return {
            "reasoning": parsed.get("reasoning", "Secondary review unavailable."),
            "context_notes": parsed.get("context_notes", "No additional context."),
        }
