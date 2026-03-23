import unittest

from models import ModerationDecision, ModerationRequest
from moderation_service import ModerationService


class ModerationServiceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.service = ModerationService(openai_key="mock-key", anthropic_key="mock-key")

    async def test_moderation_decisions_and_response_fields(self):
        scenarios = [
            (
                "cooking review",
                "Watch me chop vegetables with a knife in this cooking recipe.",
                ModerationDecision.HUMAN_REVIEW,
                {"cooking_context"},
                "Violence signal appears within cooking context.",
            ),
            (
                "fitness review",
                "Shirtless workout at the gym to show abs and fitness training.",
                ModerationDecision.HUMAN_REVIEW,
                {"fitness_context"},
                "Adult-content signal appears within fitness context.",
            ),
            (
                "medical review",
                "A doctor explains surgery wound care and injection safety.",
                ModerationDecision.HUMAN_REVIEW,
                {"medical_context"},
                "Violence signal appears within medical context.",
            ),
            (
                "explicit violence block",
                "I will kill and attack everyone here.",
                ModerationDecision.BLOCK,
                set(),
                None,
            ),
            (
                "explicit hate block",
                "This hateful slur is racist and full of hate.",
                ModerationDecision.BLOCK,
                set(),
                None,
            ),
            (
                "explicit sexual block",
                "This is explicit xxx nsfw content.",
                ModerationDecision.BLOCK,
                set(),
                None,
            ),
            (
                "explicit spam block",
                "Buy now, click here, limited time, act fast.",
                ModerationDecision.BLOCK,
                set(),
                None,
            ),
            (
                "supplement scam review",
                "One weird trick secret supplement for muscle energy and rapid weight loss.",
                ModerationDecision.HUMAN_REVIEW,
                {"supplement_scam_pattern"},
                "Supplement scam pattern detected despite low spam score.",
            ),
            (
                "coded hate review",
                "You know who and those people are ruining this place.",
                ModerationDecision.HUMAN_REVIEW,
                {"coded_hate_pattern"},
                "Potential coded hate speech detected.",
            ),
            (
                "safe allow",
                "Check out my tutorial for organizing your desk workspace.",
                ModerationDecision.ALLOW,
                set(),
                None,
            ),
        ]

        for name, content, expected_decision, expected_signals, expected_review_reason in scenarios:
            with self.subTest(name=name):
                result = await self.service.moderate_content(
                    ModerationRequest(content=content, creator_id="creator-123", video_id="video-1")
                )
                self.assertEqual(result.decision, expected_decision)
                self.assertEqual(result.requires_human_review, expected_decision == ModerationDecision.HUMAN_REVIEW)
                self.assertEqual(result.is_safe, expected_decision == ModerationDecision.ALLOW)
                self.assertIsInstance(result.category_scores, dict)
                self.assertSetEqual(set(result.category_scores.keys()), {"hate", "violence", "sexual", "spam"})
                self.assertIsInstance(result.matched_signals, list)
                self.assertTrue(result.reasoning)
                self.assertEqual(set(result.matched_signals), expected_signals)
                self.assertEqual(result.review_reason, expected_review_reason)

    async def test_human_review_uses_secondary_provider_marker(self):
        result = await self.service.moderate_content(
            ModerationRequest(
                content="Knife demo in a cooking recipe with vegetables.",
                creator_id="creator-123",
            )
        )

        self.assertEqual(result.decision, ModerationDecision.HUMAN_REVIEW)
        self.assertEqual(result.provider, "openai+anthropic")
        self.assertIn("Secondary review:", result.reasoning)


if __name__ == "__main__":
    unittest.main()
