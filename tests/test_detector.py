import math
import unittest

from watermark.detector import WatermarkDetector


class WatermarkDetectorTests(unittest.TestCase):
    def test_all_green_chain_matches_closed_form_z_score(self):
        detector = WatermarkDetector(vocab_size=16, gamma=0.5, seed=42, z_threshold=1.5)
        token_ids = [0]
        for _ in range(4):
            token_ids.append(min(detector._get_greenlist_set(token_ids[-1])))

        result = detector.score_sequence(token_ids)

        self.assertEqual(result.green_count, 4)
        self.assertEqual(result.total_tokens, 4)
        self.assertEqual(result.green_fraction, 1.0)
        self.assertAlmostEqual(result.z_score, 2.0)
        self.assertAlmostEqual(
            result.p_value,
            0.5 * math.erfc(result.z_score / math.sqrt(2)),
        )
        self.assertTrue(result.is_watermarked)

    def test_short_sequence_is_neutral(self):
        result = WatermarkDetector(vocab_size=16).score_sequence([3])
        self.assertEqual(result.z_score, 0.0)
        self.assertEqual(result.p_value, 1.0)
        self.assertEqual(result.total_tokens, 0)
        self.assertFalse(result.is_watermarked)

    def test_out_of_vocabulary_pairs_are_skipped(self):
        detector = WatermarkDetector(vocab_size=8)
        result = detector.score_sequence([0, 8, 1, -1, 2])
        self.assertEqual(result.total_tokens, 0)
        self.assertEqual(result.green_count, 0)

    def test_calibration_respects_empirical_fpr(self):
        detector = WatermarkDetector(vocab_size=8)
        scores = [float(value) for value in range(100)]
        threshold = detector.calibrate_threshold(scores, target_fpr=0.01)
        actual_fpr = sum(score > threshold for score in scores) / len(scores)
        self.assertEqual(threshold, 98.0)
        self.assertLessEqual(actual_fpr, 0.01)
        self.assertEqual(actual_fpr, 0.01)

    def test_small_calibration_set_is_conservative(self):
        detector = WatermarkDetector(vocab_size=8)
        scores = [float(value) for value in range(98)]
        threshold = detector.calibrate_threshold(scores, target_fpr=0.01)
        actual_fpr = sum(score > threshold for score in scores) / len(scores)
        self.assertEqual(threshold, 97.0)
        self.assertEqual(actual_fpr, 0.0)

    def test_invalid_configuration_and_calibration_raise(self):
        with self.assertRaises(ValueError):
            WatermarkDetector(vocab_size=0)
        with self.assertRaises(ValueError):
            WatermarkDetector(vocab_size=8, gamma=1.0)

        detector = WatermarkDetector(vocab_size=8)
        with self.assertRaises(ValueError):
            detector.calibrate_threshold([])
        with self.assertRaises(ValueError):
            detector.calibrate_threshold([0.0], target_fpr=0.0)


if __name__ == "__main__":
    unittest.main()
