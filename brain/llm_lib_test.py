import os
import time
import unittest
from datetime import date, timedelta
from unittest import mock

from brain.llm_lib import (
    LLM_CLOUD,
    LLM_EDGE,
    PROMPT_CONVERSATION,
    PROMPT_SUMMARIZING,
    clean_nudge,
    nudge_looks_valid,
)
from orchestrator.orchestrator import MemoryManager


class NudgeValidationTest(unittest.TestCase):
    def test_clean_keeps_first_sentence(self):
        self.assertEqual(
            clean_nudge('"Hey, put the phone down! More text."'),
            "Hey, put the phone down!",
        )

    def test_valid_phone_nudge(self):
        self.assertTrue(
            nudge_looks_valid("Hey, put the phone down!", ["looking_at_phone"])
        )

    def test_rejects_wrong_habit(self):
        self.assertFalse(
            nudge_looks_valid("Face-touching", ["biting_nails"])
        )
        self.assertFalse(
            nudge_looks_valid("Hey, you touched your face!", ["biting_nails"])
        )

    def test_rejects_false_screen_complaint(self):
        vlm = (
            '"looking_at_screen": True | "touching_face": True | '
            '"looking_at_phone": False | "biting_nails": False'
        )
        bad = "You've been face-touching 19s, but haven't been looking at your screen."
        self.assertFalse(nudge_looks_valid(bad, ["touching_face"], vlm))
        good = "Hey friend, that hand wandered to your cheek again."
        self.assertTrue(nudge_looks_valid(good, ["touching_face"], vlm))

    def test_accepts_sympathetic_face_nudge(self):
        self.assertTrue(
            nudge_looks_valid(
                "Hey, you're touching your face again.",
                ["touching_face"],
            )
        )
# The real-model tests cost money (cloud) or take minutes to download/load (edge),
# so they only run when explicitly requested:  RUN_REAL_LLM_TESTS=1 python -m unittest ...
RUN_REAL_LLM_TESTS = os.getenv("RUN_REAL_LLM_TESTS") == "1"

VLM_OUTPUT = '"looking_at_screen": False | "touching_face": False | "looking_at_phone": True | "biting_nails": False'
LONG_TERM_MEMORY = "You've been working 1h. phone 2 times."
SHORT_TERM_MEMORY = [VLM_OUTPUT, VLM_OUTPUT]


class LlmCloudTest(unittest.TestCase):
    def setUp(self):
        # Mock the OpenAI client so these tests are free, fast, and offline.
        patcher = mock.patch("openai.OpenAI")
        self.addCleanup(patcher.stop)
        self.mock_client = patcher.start().return_value
        self.mock_client.responses.create.return_value.output_text = "Put the phone down!"
        self.llm = LLM_CLOUD()

    def test_llm_response_returns_output_text(self):
        response = self.llm.get_llm_response(
            VLM_OUTPUT, LONG_TERM_MEMORY, detected_labels=["phone"]
        )
        self.assertEqual(response, "Put the phone down!")

    def test_llm_response_uses_conversation_prompt(self):
        self.llm.get_llm_response(
            VLM_OUTPUT, LONG_TERM_MEMORY, detected_labels=["phone"]
        )
        call_kwargs = self.mock_client.responses.create.call_args.kwargs
        self.assertEqual(call_kwargs["instructions"], PROMPT_CONVERSATION)
        self.assertIn("Detected: phone", call_kwargs["input"])
        self.assertNotIn(VLM_OUTPUT, call_kwargs["input"])
        self.assertNotIn(LONG_TERM_MEMORY, call_kwargs["input"])

    def test_summarizing_uses_summarizing_prompt(self):
        self.llm.get_summarizing_response(SHORT_TERM_MEMORY, LONG_TERM_MEMORY)
        call_kwargs = self.mock_client.responses.create.call_args.kwargs
        self.assertEqual(call_kwargs["instructions"], PROMPT_SUMMARIZING)
        self.assertIn(LONG_TERM_MEMORY, call_kwargs["input"])

    @unittest.skipUnless(RUN_REAL_LLM_TESTS, "set RUN_REAL_LLM_TESTS=1 to hit the real API")
    def test_real_api_smoke(self):
        mock.patch.stopall()  # undo the setUp mock, use the real client
        llm = LLM_CLOUD()
        response = llm.get_llm_response(VLM_OUTPUT, LONG_TERM_MEMORY)
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)


class MemoryManagerTest(unittest.TestCase):
    def test_counts_rising_edges_not_every_frame(self):
        mem = MemoryManager()
        on = '"biting_nails": True | "touching_face": False | "looking_at_phone": False'
        off = '"biting_nails": False | "touching_face": False | "looking_at_phone": False'
        mem.add_memory(on)
        mem.add_memory(on)
        mem.add_memory(on)
        self.assertEqual(mem.event_counters["biting_nails"], 1)
        mem.add_memory(off)
        mem.add_memory(on)
        self.assertEqual(mem.event_counters["biting_nails"], 2)

    def test_status_summary_mentions_counts(self):
        mem = MemoryManager()
        mem.event_counters["biting_nails"] = 3
        mem.event_counters["looking_at_phone"] = 1
        text = mem.format_status_summary()
        self.assertIn("nail-biting 3 times", text)
        self.assertIn("phone 1 time", text)
        self.assertIn("You've been working", text)

    def test_day_rollover_resets_counters(self):
        mem = MemoryManager()
        mem.event_counters["biting_nails"] = 5
        mem.day = date.today() - timedelta(days=1)
        mem.day_started_at = time.time() - 86400
        rolled = mem.maybe_roll_day()
        self.assertTrue(rolled)
        self.assertEqual(mem.event_counters["biting_nails"], 0)
        self.assertEqual(mem.day, date.today())


@unittest.skipUnless(RUN_REAL_LLM_TESTS, "set RUN_REAL_LLM_TESTS=1 to load the local model")
class LlmEdgeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the MLX 4-bit LLM once for the whole suite (like SetUpTestSuite in gtest).
        cls.llm = LLM_EDGE()

    def test_real_llm_response_smoke(self):
        response = self.llm.get_llm_response(VLM_OUTPUT, LONG_TERM_MEMORY)
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)

    def test_real_summarizing_smoke(self):
        response = self.llm.get_summarizing_response(SHORT_TERM_MEMORY, LONG_TERM_MEMORY)
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)


if __name__ == "__main__":
    unittest.main()
