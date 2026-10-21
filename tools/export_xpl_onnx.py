#!/usr/bin/env python3
"""Export XPLNet, line verification, and line description models for XPLSLAM."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--weights', type=Path,
        default=ROOT / 'weights/XPLNetNew_160000.pth',
        help='XPLNet checkpoint containing xplnet and verify_coarse weights.')
    parser.add_argument(
        '--line-desc-weights', type=Path,
        default=ROOT / 'weights/LDesc_160000.pth',
        help='LineDescription checkpoint containing line_desc weights.')
    parser.add_argument(
        '--output-dir', type=Path, default=ROOT / 'onnx',
        help='Output directory for the three ONNX files.')
    parser.add_argument('--height', type=int, default=608)
    parser.add_argument('--width', type=int, default=800)
    parser.add_argument('--opset', type=int, default=17)
    parser.add_argument(
        '--skip-runtime-validation', action='store_true',
        help='Skip ONNX Runtime numerical comparison after export.')
    return parser.parse_args()


def load_models(weights: Path, line_desc_weights: Path):
    from modules.line_description import LineDescription
    from modules.verify import Verification
    from modules.xplnet_new import XPLNetNew

    if not weights.is_file():
        raise FileNotFoundError(f'XPLNet checkpoint not found: {weights}')
    if not line_desc_weights.is_file():
        raise FileNotFoundError(
            f'Line descriptor checkpoint not found: {line_desc_weights}')

    checkpoint = torch.load(weights, map_location='cpu', weights_only=True)
    if 'xplnet' not in checkpoint or 'verify_coarse' not in checkpoint:
        raise KeyError(
            'XPLNet checkpoint must contain xplnet and verify_coarse keys')

    xplnet = XPLNetNew()
    xplnet.load_state_dict(checkpoint['xplnet'], strict=True)
    xplnet.eval()
    xplnet.switch_to_deploy()

    samples = 32
    verify_dim = 128
    verify_in_dim = 2 * verify_dim + (samples - 2) * 4
    verification = Verification(verify_in_dim)
    verification.load_state_dict(checkpoint['verify_coarse'], strict=True)
    verification.eval()

    line_checkpoint = torch.load(line_desc_weights, map_location='cpu', weights_only=True)
    line_state = line_checkpoint.get('line_desc', line_checkpoint)
    line_description = LineDescription(
        samples=samples, p_dim=64, l_dim=64)
    line_description.load_state_dict(line_state, strict=True)
    line_description.eval()

    print(
        f'Loaded XPLNet checkpoint: {weights} '
        f'(iter={checkpoint.get("iter", "unknown")})')
    print(
        f'Loaded line descriptor checkpoint: {line_desc_weights} '
        f'(iter={line_checkpoint.get("iter", "unknown")})')
    return xplnet, verification, line_description, verify_in_dim


def export_xplnet(model, output_path: Path, height: int, width: int,
                  opset: int) -> None:
    dummy = torch.randn(1, 1, height, width, dtype=torch.float32)
    output_names = [
        'desc_map', 'keypoint_logits', 'heatmap',
        'junction_logits', 'edge_desc', 'junc_desc',
    ]
    dynamic_axes = {'image': {0: 'batch'}}
    dynamic_axes.update({name: {0: 'batch'} for name in output_names})
    torch.onnx.export(
        model,
        dummy,
        output_path.as_posix(),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=['image'],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        keep_initializers_as_inputs=False,
        dynamo=False,
    )


def export_verification(model, output_path: Path, in_dim: int,
                        opset: int) -> None:
    dummy = torch.randn(1, in_dim, 64, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        output_path.as_posix(),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=['line_features'],
        output_names=['logits'],
        dynamic_axes={
            'line_features': {0: 'batch', 2: 'num_segments'},
            'logits': {0: 'batch', 2: 'num_segments'},
        },
        keep_initializers_as_inputs=False,
        dynamo=False,
    )


def export_line_description(model, output_path: Path, opset: int) -> None:
    dummy = torch.randn(
        1, model.p_dim, 64, model.samples, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        output_path.as_posix(),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=['line_samples'],
        output_names=['line_descriptors'],
        dynamic_axes={
            'line_samples': {0: 'batch', 2: 'num_lines'},
            'line_descriptors': {0: 'batch', 1: 'num_lines'},
        },
        keep_initializers_as_inputs=False,
        dynamo=False,
    )


def check_models(paths: list[Path]) -> None:
    import onnx

    for path in paths:
        model = onnx.load(path.as_posix())
        onnx.checker.check_model(model)
        print(f'ONNX checker passed: {path}')


def _compare_outputs(name: str, expected, actual,
                     rtol: float = 2e-4,
                     atol: float = 2e-5) -> None:
    if len(expected) != len(actual):
        raise RuntimeError(
            f'{name}: output count mismatch: '
            f'{len(expected)} != {len(actual)}')

    for index, (torch_output, ort_output) in enumerate(
            zip(expected, actual)):
        torch_array = torch_output.detach().cpu().numpy()
        if torch_array.shape != ort_output.shape:
            raise RuntimeError(
                f'{name}[{index}]: shape mismatch: '
                f'{torch_array.shape} != {ort_output.shape}')
        max_abs = float(np.max(np.abs(torch_array - ort_output)))
        np.testing.assert_allclose(
            torch_array, ort_output, rtol=rtol, atol=atol)
        print(
            f'ORT comparison passed: {name}[{index}] '
            f'shape={ort_output.shape}, max_abs={max_abs:.3e}')


def validate_with_ort(xplnet, verification, line_description,
                      xplnet_path: Path, verify_path: Path,
                      line_desc_path: Path, height: int, width: int,
                      verify_in_dim: int) -> None:
    import onnxruntime as ort

    providers = ['CPUExecutionProvider']
    torch.manual_seed(2026)

    image = torch.randn(2, 1, height, width, dtype=torch.float32)
    with torch.inference_mode():
        expected_xplnet = xplnet(image)
    xplnet_session = ort.InferenceSession(
        xplnet_path.as_posix(), providers=providers)
    actual_xplnet = xplnet_session.run(
        None, {'image': image.numpy()})
    _compare_outputs(
        'xplnet', expected_xplnet, actual_xplnet,
        rtol=1e-3, atol=2e-4)

    line_features = torch.randn(
        2, verify_in_dim, 37, dtype=torch.float32)
    with torch.inference_mode():
        expected_verify = (verification(line_features),)
    verify_session = ort.InferenceSession(
        verify_path.as_posix(), providers=providers)
    actual_verify = verify_session.run(
        None, {'line_features': line_features.numpy()})
    _compare_outputs('verification_coarse', expected_verify, actual_verify)

    line_samples = torch.randn(
        2, line_description.p_dim, 19,
        line_description.samples, dtype=torch.float32)
    with torch.inference_mode():
        expected_line_desc = (line_description(line_samples),)
    line_desc_session = ort.InferenceSession(
        line_desc_path.as_posix(), providers=providers)
    actual_line_desc = line_desc_session.run(
        None, {'line_samples': line_samples.numpy()})
    _compare_outputs(
        'line_description', expected_line_desc, actual_line_desc)


def main() -> None:
    args = parse_args()
    if args.height <= 0 or args.width <= 0:
        raise ValueError('height and width must be positive')
    if args.height % 32 != 0 or args.width % 32 != 0:
        raise ValueError('height and width must be multiples of 32')

    xplnet, verification, line_description, verify_in_dim = load_models(
        args.weights, args.line_desc_weights)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    xplnet_path = args.output_dir / 'xplnet.onnx'
    verify_path = args.output_dir / 'verification_coarse.onnx'
    line_desc_path = args.output_dir / 'line_description.onnx'

    export_xplnet(
        xplnet, xplnet_path, args.height, args.width, args.opset)
    export_verification(
        verification, verify_path, verify_in_dim, args.opset)
    export_line_description(
        line_description, line_desc_path, args.opset)

    paths = [xplnet_path, verify_path, line_desc_path]
    check_models(paths)
    if not args.skip_runtime_validation:
        validate_with_ort(
            xplnet, verification, line_description,
            xplnet_path, verify_path, line_desc_path,
            args.height, args.width, verify_in_dim)

    print('Exported ONNX models for XPLSLAM:')
    for path in paths:
        print(f'  {path.resolve()} ({path.stat().st_size / 1024**2:.2f} MiB)')


if __name__ == '__main__':
    main()
