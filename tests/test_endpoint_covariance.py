import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from modules.detector import XPLDetector


def covariance_test_detector(radius=2, temperature=1.0, floor=1e-4):
    detector = XPLDetector.__new__(XPLDetector)
    nn.Module.__init__(detector)
    detector.junc_cov_radius = radius
    detector.junc_cov_temperature = temperature
    detector.junc_cov_floor = floor
    return detector


class EndpointCovarianceTest(unittest.TestCase):
    def test_local_moments_refine_peak_and_are_anisotropic(self):
        detector = covariance_test_detector(radius=2)
        yy, xx = torch.meshgrid(
            torch.arange(9, dtype=torch.float32),
            torch.arange(9, dtype=torch.float32), indexing='ij')
        logits = (-0.5 * (((xx - 4.0) / 1.5) ** 2
                          + ((yy - 4.0) / 0.5) ** 2))[None, None]
        peaks = torch.tensor([[[4.0, 4.0]]])
        valid = torch.tensor([[True]])

        refined, covariances = \
            detector.refine_junctions_and_covariances(
                logits, peaks, valid)

        torch.testing.assert_close(
            refined[0, 0], torch.tensor([4.0, 4.0]), atol=1e-6, rtol=0)
        self.assertGreater(covariances[0, 0, 0, 0].item(),
                           covariances[0, 0, 1, 1].item())
        torch.testing.assert_close(
            covariances[0, 0], covariances[0, 0].T,
            atol=1e-6, rtol=0)
        self.assertTrue(torch.all(
            torch.linalg.eigvalsh(covariances[0, 0]) > 0))

    def test_invalid_padding_is_zero_and_border_peak_is_finite(self):
        detector = covariance_test_detector(radius=2)
        logits = torch.zeros((1, 1, 5, 5))
        peaks = torch.tensor([[[0.0, 0.0], [0.0, 0.0]]])
        valid = torch.tensor([[True, False]])

        refined, covariances = \
            detector.refine_junctions_and_covariances(
                logits, peaks, valid)

        self.assertTrue(torch.isfinite(refined).all())
        self.assertTrue(torch.isfinite(covariances).all())
        torch.testing.assert_close(refined[0, 1], torch.zeros(2))
        torch.testing.assert_close(covariances[0, 1], torch.zeros(2, 2))

    def test_coordinate_scaling_transforms_rows_and_columns(self):
        covariance = torch.tensor([[
            [2.0, 0.5],
            [0.5, 3.0],
        ]])
        scaled = XPLDetector.scale_covariances(
            covariance, torch.tensor([2.0, 3.0]))
        expected = torch.tensor([[
            [8.0, 3.0],
            [3.0, 27.0],
        ]])
        torch.testing.assert_close(scaled, expected)

        covariance_4d = torch.eye(4).unsqueeze(0)
        scaled_4d = XPLDetector.scale_covariances(
            covariance_4d, torch.tensor([2.0, 3.0, 2.0, 3.0]))
        torch.testing.assert_close(
            torch.diagonal(scaled_4d[0]),
            torch.tensor([4.0, 9.0, 4.0, 9.0]))

    def test_line_covariance_is_endpoint_block_diagonal(self):
        detector = covariance_test_detector()
        detector.sample = 3
        detector.line_th = 0.5
        detector.line_descriptor = SimpleNamespace(l_dim=4)
        detector.line_descriptor_loaded = False
        detector.verify = lambda features: torch.tensor(
            [[[-1.0], [1.0]]], dtype=features.dtype)
        detector.verification_preprocess = lambda points, edge, junc: \
            torch.zeros((1, 1, points.shape[2]), dtype=points.dtype)

        junctions = torch.tensor([[[1.0, 2.0], [5.0, 7.0]]])
        covariance_0 = torch.tensor([[2.0, 0.2], [0.2, 3.0]])
        covariance_1 = torch.tensor([[4.0, -0.1], [-0.1, 5.0]])
        junction_covariances = torch.stack(
            [covariance_0, covariance_1])[None]
        junction_valid = torch.tensor([[True, True]])

        _, _, _, line_covariances = detector.process_lines(
            torch.zeros((1, 1, 2, 2)),
            torch.zeros((1, 1, 2, 2)),
            torch.zeros((1, 1, 2, 2)),
            junctions, junction_covariances, junction_valid)

        self.assertEqual(line_covariances[0].shape, (1, 4, 4))
        torch.testing.assert_close(
            line_covariances[0][0, :2, :2], covariance_0)
        torch.testing.assert_close(
            line_covariances[0][0, 2:, 2:], covariance_1)
        torch.testing.assert_close(
            line_covariances[0][0, :2, 2:], torch.zeros(2, 2))


if __name__ == '__main__':
    unittest.main()
