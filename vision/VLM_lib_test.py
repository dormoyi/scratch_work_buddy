import os
import unittest
from unittest import mock

from PIL import Image

from vision.VLM_lib import (
    VLM_CLOUD,
    VLM_EDGE,
    _asymmetric_face_crop_box,
    classify_from_description,
)


class AsymmetricCropTest(unittest.TestCase):
    def test_look_left_extends_left_more(self):
        # Face centered in 1000×1000; look_dir=-1 should pull left edge farther out.
        box = (400, 400, 600, 600)
        left_l, _, right_l, _ = _asymmetric_face_crop_box(1000, 1000, box, look_dir=-1)
        left_r, _, right_r, _ = _asymmetric_face_crop_box(1000, 1000, box, look_dir=1)
        self.assertLess(left_l, left_r)
        self.assertLess(right_l, right_r)
        # More room toward the look side than the opposite side.
        cx = 500
        self.assertGreater(cx - left_l, right_l - cx)
        self.assertGreater(right_r - cx, cx - left_r)

    def test_always_biases_down(self):
        box = (400, 400, 600, 600)
        _, top, _, bottom = _asymmetric_face_crop_box(1000, 1000, box, look_dir=0)
        cy_mid = (top + bottom) / 2.0
        self.assertGreater(cy_mid, 500)


class ClassifyFromDescriptionTest(unittest.TestCase):
    def test_hands_not_visible(self):
        line = classify_from_description(
            "The person's hands are not visible, and they are looking at the camera."
        )
        self.assertIn('"looking_at_screen": True', line)
        self.assertIn('"touching_face": False', line)
        self.assertIn('"looking_at_phone": False', line)

    def test_looking_at_camera_counts_as_screen(self):
        line = classify_from_description(
            "The person is looking at the camera. There is no hand touching the face. "
            "The person is not looking at the screen."
        )
        self.assertIn('"looking_at_screen": True', line)

    def test_scratching_cheek(self):
        line = classify_from_description(
            "Her right hand is scratching her cheek while she looks at the screen."
        )
        self.assertIn('"touching_face": True', line)
        self.assertIn('"looking_at_screen": True', line)
        self.assertIn('"looking_at_phone": False', line)

    def test_contradictory_hands_absent_but_touching(self):
        # Qwen often says this on low desk / robot angles.
        line = classify_from_description(
            "The person is looking at the camera. The hands are not visible, but there "
            "is a hand touching the face. The person is sitting at a desk with a laptop."
        )
        self.assertIn('"touching_face": True', line)

    def test_hand_on_jaw(self):
        line = classify_from_description(
            "A palm is against the jaw and fingers are across the cheek."
        )
        self.assertIn('"touching_face": True', line)

    def test_touching_chin_with_finger(self):
        line = classify_from_description(
            "There is no hand or forearm near the head. The person is touching "
            "their chin with their finger."
        )
        self.assertIn('"touching_face": True', line)

    def test_touching_the_cheek(self):
        line = classify_from_description(
            "There is a hand near the head, touching the cheek."
        )
        self.assertIn('"touching_face": True', line)

    def test_resting_head_on_hand_despite_earlier_denial(self):
        line = classify_from_description(
            "There is no hand touching the face, cheek, or hair. "
            "The person is resting their head on their hand."
        )
        self.assertIn('"touching_face": True', line)

    def test_chin_on_sleeve_counts_as_touch(self):
        line = classify_from_description(
            "The hands are not visible. The chin is resting on a sleeve."
        )
        self.assertIn('"touching_face": True', line)

    def test_prompt_echo_denial_is_not_touch(self):
        # Model copies the prompt noun list under a leading "no" — must not match
        # "sleeve under the chin" as a positive (old 40-char window dropped "no").
        line = classify_from_description(
            "Sentence 1: There is no visible hand, fingers, fist, forearm, wrist, "
            "or sleeve under the chin/cheek. The person is not touching anything.\n"
            "Sentence 2: The person is looking at the screen."
        )
        self.assertIn('"touching_face": False', line)

    def test_no_hand_touching_stays_false(self):
        line = classify_from_description(
            "There is no hand touching the face, cheek, or hair. "
            "The person is looking away from the camera."
        )
        self.assertIn('"touching_face": False', line)

    def test_leading_yes_alone_is_not_face_touch(self):
        # 2B models often emit a bare Yes and stop; do not treat that as detection.
        line = classify_from_description("Yes")
        self.assertIn('"touching_face": False', line)

    def test_labeled_one_line(self):
        line = classify_from_description(
            "Face: Yes | Nails: No | Looking: screen"
        )
        self.assertIn('"touching_face": True', line)
        self.assertIn('"biting_nails": False', line)
        self.assertIn('"looking_at_screen": True', line)

    def test_labeled_face_and_nails(self):
        face_only = classify_from_description(
            "Face: Yes\nNails: No\nLooking: screen"
        )
        self.assertIn('"touching_face": True', face_only)
        self.assertIn('"biting_nails": False', face_only)
        nails_only = classify_from_description(
            "Face: No\nNails: Yes\nLooking: screen"
        )
        self.assertIn('"touching_face": False', nails_only)
        self.assertIn('"biting_nails": True', nails_only)

    def test_phone(self):
        line = classify_from_description(
            "She is holding a phone in both hands and looking at it."
        )
        self.assertIn('"looking_at_phone": True', line)
        self.assertIn('"looking_at_screen": False', line)

    def test_nail_biting(self):
        line = classify_from_description(
            "Yes, there is a finger at the person's mouth. They are looking at the camera."
        )
        self.assertIn('"biting_nails": True', line)
        self.assertIn('"touching_face": False', line)

    def test_negated_nail_biting(self):
        line = classify_from_description(
            "There is no indication of nail-biting in the photo."
        )
        self.assertIn('"biting_nails": False', line)


# The real-model tests cost money (cloud) or take minutes to download/load (edge),
# so they only run when explicitly requested:  RUN_REAL_VLM_TESTS=1 python -m unittest ...
RUN_REAL_VLM_TESTS = os.getenv("RUN_REAL_VLM_TESTS") == "1"


def get_dummy_camera_frame():
    # A synthetic image so the tests don't depend on files on disk.
    return Image.new("RGB", (640, 480), color=(120, 90, 200))


class VlmCloudTest(unittest.TestCase):
    def test_returns_model_output_text(self):
        # Mock the OpenAI client so this test is free, fast, and offline.
        with mock.patch("openai.OpenAI") as mock_openai:
            mock_client = mock_openai.return_value
            mock_client.responses.create.return_value.output_text = (
                '"looking_at_screen": True | "touching_face": False'
            )

            vlm_cloud = VLM_CLOUD()
            response = vlm_cloud.get_vision_response(get_dummy_camera_frame())

        self.assertEqual(response, '"looking_at_screen": True | "touching_face": False')

    def test_sends_frame_as_base64_jpeg(self):
        with mock.patch("openai.OpenAI") as mock_openai:
            mock_client = mock_openai.return_value
            mock_client.responses.create.return_value.output_text = "ok"

            VLM_CLOUD().get_vision_response(get_dummy_camera_frame())

            call_kwargs = mock_client.responses.create.call_args.kwargs
            image_part = call_kwargs["input"][0]["content"][1]

        self.assertEqual(image_part["type"], "input_image")
        self.assertTrue(image_part["image_url"].startswith("data:image/jpeg;base64,"))

    @unittest.skipUnless(RUN_REAL_VLM_TESTS, "set RUN_REAL_VLM_TESTS=1 to hit the real API")
    def test_real_api_smoke(self):
        vlm_cloud = VLM_CLOUD()
        response = vlm_cloud.get_vision_response(get_dummy_camera_frame())
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)


@unittest.skipUnless(RUN_REAL_VLM_TESTS, "set RUN_REAL_VLM_TESTS=1 to load the local model")
class VlmEdgeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the MLX 4-bit 2.2B model once for the whole suite (like SetUpTestSuite in gtest).
        cls.vlm_edge = VLM_EDGE()

    def test_real_model_smoke(self):
        response = self.vlm_edge.get_vision_response(get_dummy_camera_frame())
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)


if __name__ == "__main__":
    unittest.main()
