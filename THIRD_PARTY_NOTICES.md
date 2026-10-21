# Third-party notices

XPLNet project code and the two bundled model checkpoints are released under
GPL-3.0. Incorporated upstream code retains its original license and notices.

Parts of `modules/xplnet_new.py` (the feature backbone), `modules/detector.py`
(point extraction and matching), and `modules/interpolator.py` are adapted
from [XFeat](https://github.com/verlab/accelerated_features), by Guilherme Potje,
Felipe Cadar, André Araujo, Renato Martins, and Erickson R. Nascimento.
XPLNet modifies the network architecture, adds junction and line prediction,
line verification and descriptors, and extends inference with endpoint
covariances. The upstream Apache-2.0 license is preserved in
[licenses/XFeat-Apache-2.0.txt](licenses/XFeat-Apache-2.0.txt).

The upstream license was retrieved from revision
`e92685f57f8318b18725c5c8c0bd28c7fe188d9a`; this records the license source,
not an exact base revision for all adapted code.

PyTorch, NumPy, OpenCV, ONNX, and ONNX Runtime are installed separately and
are not vendored in this repository.
