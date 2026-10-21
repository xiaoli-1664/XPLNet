# XPLNet

XPLNet extracts image keypoints, line segments, and their learned descriptors.
It is the neural feature frontend used by
[XPLSLAM](https://github.com/xiaoli-1664/XPLSLAM).

This release contains the inference network, the two checkpoints used by
XPLSLAM, a single-image example, and the ONNX exporter. Training datasets,
evaluation pipelines, intermediate checkpoints, and generated results are
not part of this release.

## Installation

Use Python 3.10 or newer and install PyTorch 2.8.x for your CPU or CUDA
environment, then install the inference dependencies:

```bash
git clone https://github.com/xiaoli-1664/XPLNet.git
cd XPLNet
python -m pip install -r requirements.txt
```

The tested environment uses Python 3.10, PyTorch 2.8.0, NumPy 1.26.4,
and OpenCV 4.11.0. CPU inference is supported; CUDA is optional for the Python
example. The SLAM application requires CUDA and TensorRT.

## Included weights

Only these two checkpoints are distributed:

| File | Contents |
| --- | --- |
| `weights/XPLNetNew_160000.pth` | XPLNet backbone and heads (`xplnet`), line verification (`verify_coarse`), and iteration metadata |
| `weights/LDesc_160000.pth` | Learned line descriptor (`line_desc`) and iteration metadata |

Both checkpoints are from iteration 160000. They are stored directly in Git
and do not require Git LFS. Integrity hashes are in [weights/SHA256SUMS](weights/SHA256SUMS).

## Single-image example

```bash
python examples/infer.py --image /path/to/image.png --device cpu
```

Use `--device cuda` for GPU inference, or omit `--device` for automatic
selection. The example writes `output/features.png` (red keypoints and green
lines) and `output/features.npz` (features and descriptors). Add `--show` to
display an OpenCV window, or `--output /path/to/result.png` to choose the
output location. Use `--help` for detection thresholds and resize options.

The NPZ file contains original-image pixel coordinates in `(x, y)` order:

| Array | Shape |
| --- | --- |
| `keypoints`, `scores`, `descriptors` | `[N, 2]`, `[N]`, `[N, 64]` |
| `lines`, `lines_scores`, `line_descriptors` | `[L, 2, 2]`, `[L]`, `[L, 64]` |
| `junctions`, `junction_scores`, `junction_covariances` | `[J, 2]`, `[J]`, `[J, 2, 2]` |
| `line_endpoint_covariances` | `[L, 4, 4]`, ordered as `(x1, y1, x2, y2)` |

Endpoint covariances describe local junction-logit shape; they have not been
calibrated against ground-truth endpoint errors. Cross-covariance between the
two endpoints is assumed to be zero.

## Python API

```python
import cv2
from modules.detector import XPLDetector

detector = XPLDetector(dev="cpu", top_k=500, junc_th=0.05, line_th=0.9).eval()
image = cv2.imread("/path/to/image.png", cv2.IMREAD_GRAYSCALE)
features = detector(image, line_desc=True)[0]
print(features["keypoints"].shape, features["line_descriptors"].shape)
```

Default checkpoint paths are resolved relative to the repository. For CUDA,
construct with `dev="cuda"` and move the module using `.to("cuda").eval()`.
`match_xpl(image1, image2, line_match=True)` performs mutual-nearest-neighbor
matching using the learned point and line descriptors.

## Export models for XPLSLAM

```bash
python -m pip install -r requirements-onnx.txt
python tools/export_xpl_onnx.py \
  --height 608 --width 800 \
  --output-dir /path/to/XPLSLAM/models
```

The exporter runs on CPU, fuses the inference branches, checks each ONNX
graph, and compares ONNX Runtime outputs with PyTorch. It generates:

| File | Input | Output |
| --- | --- | --- |
| `xplnet.onnx` | `image`: `[B, 1, H, W]` | point descriptors, keypoint logits, reliability heatmap, junction logits, edge and junction descriptors |
| `verification_coarse.onnx` | `line_features`: `[B, 376, S]` | `logits`: `[B, 2, S]` |
| `line_description.onnx` | `line_samples`: `[B, 64, L, 32]` | `line_descriptors`: `[B, L, 64]` |

Image height and width are fixed at export time and must be multiples of 32.
Batch size is dynamic; segment and line counts are also dynamic. The default
`608 x 800` export matches XPLSLAM's NewCollege configuration. If you change
these dimensions, update its `XPLNet.resize_height` / `XPLNet.resize_width`
settings and rebuild the affected TensorRT engine caches. XPLSLAM's DBoW3
vocabularies are separate runtime assets.

## Checks

```bash
python -m unittest discover -s tests -v
sha256sum -c weights/SHA256SUMS
```

## License

XPLNet project code and both bundled model checkpoints are released under
[GPL-3.0](LICENSE). Adapted upstream code retains its original notices;
see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
