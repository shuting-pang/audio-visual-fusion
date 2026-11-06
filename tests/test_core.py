import unittest
import numpy as np

from filtering import filter_transcript
from fusion import build_fusion_features


class FilteringTests(unittest.TestCase):
    def test_accepts_confident_answer(self):
        result = filter_transcript("最近睡得不太好。", -0.6)
        self.assertFalse(result.missing)
        self.assertEqual(result.text, "最近睡得不太好。")

    def test_rejection_reasons(self):
        cases = [("", -0.5, "empty"), ("x" * 51, -0.5, "too_long"),
                 ("欢迎订阅", -0.5, "blacklist"), ("回答", -2.0, "low_confidence"),
                 ("回答", None, "missing_confidence"),
                 ("回答", float("nan"), "missing_confidence")]
        for text, confidence, reason in cases:
            with self.subTest(reason=reason):
                result = filter_transcript(text, confidence)
                self.assertTrue(result.missing)
                self.assertEqual(result.text, "")
                self.assertEqual(result.reason, reason)


class FusionTests(unittest.TestCase):
    def test_missing_blocks_zeroed_and_other_modalities_retained(self):
        fused = build_fusion_features([[1, 2]], [[np.nan]], [[3]], [[0, 1, 0]])
        np.testing.assert_array_equal(fused, [[1, 2, 0, 3, 0, 1, 0]])

    def test_all_missing_stays_explicit(self):
        fused = build_fusion_features([[np.nan]], [[np.nan]], [[np.nan]], [[1, 1, 1]])
        np.testing.assert_array_equal(fused, [[0, 0, 0, 1, 1, 1]])

    def test_invalid_inputs_rejected(self):
        for missing in ([[0, 0]], [[0, 2, 0]], [[0, np.nan, 0]]):
            with self.assertRaises(ValueError):
                build_fusion_features([[1]], [[2]], [[3]], missing)
        with self.assertRaises(ValueError):
            build_fusion_features([[np.nan]], [[2]], [[3]], [[0, 0, 0]])
        with self.assertRaises(ValueError):
            build_fusion_features([[1], [2]], [[2]], [[3]], [[0, 0, 0]])


if __name__ == "__main__":
    unittest.main()
