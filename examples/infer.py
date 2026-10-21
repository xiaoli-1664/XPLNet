#!/usr/bin/env python3
"""Extract point and line features from one image, including descriptors."""

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from modules.detector import (  # noqa: E402
    DEFAULT_LINE_WEIGHTS, DEFAULT_WEIGHTS, XPLDetector,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--weights', type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument('--line-desc-weights', type=Path, default=DEFAULT_LINE_WEIGHTS)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--resize', type=int, default=800, help='Maximum image side')
    parser.add_argument('--top-k', type=int, default=500)
    parser.add_argument('--keypoint-threshold', type=float, default=0.05)
    parser.add_argument('--junction-threshold', type=float, default=0.05)
    parser.add_argument('--line-threshold', type=float, default=0.9)
    parser.add_argument('--output', type=Path, default=Path('output/features.png'))
    parser.add_argument('--show', action='store_true', help='Also open an OpenCV window')
    args = parser.parse_args()
    if args.resize < 32 or args.top_k <= 0:
        parser.error('--resize must be >= 32 and --top-k must be positive')
    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        parser.error(f'Cannot read image: {args.image}')
    scaled_min = min(image.shape[:2]) * min(1.0, args.resize / max(image.shape[:2]))
    if scaled_min < 32:
        parser.error('Image is too small or too narrow for the selected resize limit')
    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable; choose --device cpu or enable a CUDA device')
    detector = XPLDetector(
        weights=args.weights, line_desc_weights=args.line_desc_weights,
        dev=torch.device(device), top_k=args.top_k, resize_max=args.resize,
        kp_th=args.keypoint_threshold, junc_th=args.junction_threshold,
        line_th=args.line_threshold,
    ).to(device).eval()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    with torch.inference_mode():
        result = detector(gray, line_desc=True)[0]
    features = {
        name: value.detach().cpu().numpy() for name, value in result.items()
        if name not in ('desc_map', 'lines_descriptors')
    }
    visualization = image.copy()
    for start, end in features['lines']:
        cv2.line(visualization, tuple(np.rint(start).astype(int)),
                 tuple(np.rint(end).astype(int)), (0, 255, 0), 1, cv2.LINE_AA)
    for point in features['keypoints']:
        cv2.circle(visualization, tuple(np.rint(point).astype(int)),
                   2, (0, 0, 255), -1, cv2.LINE_AA)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), visualization):
        raise RuntimeError(f'Cannot write visualization: {args.output}')
    feature_path = args.output.with_suffix('.npz')
    np.savez_compressed(feature_path, **features)
    print(f'Device: {device}; points: {len(features["keypoints"])}; '
          f'lines: {len(features["lines"])}')
    print(f'Visualization: {args.output}\nFeatures: {feature_path}')
    if args.show:
        cv2.imshow('XPLNet', visualization)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
