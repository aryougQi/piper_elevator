# Elevator button model

`elevator_buttons_yolov10s.onnx` is the only runtime model used by the ROS 2
button detector.

The source checkpoint is the YOLOv10-S model published by
[`isharadilshanra/YOLOv10-Elevator-Button-Detection`](https://github.com/isharadilshanra/YOLOv10-Elevator-Button-Detection).
Its 368 classes cover elevator controls such as up/down calls, door controls,
alarm/control keys, and floor labels. The node reads these names from ONNX
metadata and retains every model class because the model is specialized for
elevator buttons.

The default confidence threshold is `0.60`. On the current camera test frame,
the up button scored about `0.97`, while the correctly localized down button
scored about `0.20`; therefore the default intentionally omits that weaker down
detection. The value remains a ROS 2 parameter so it can be adjusted without
changing code.

The PyTorch checkpoint was statically inspected and loaded only inside an
offline, read-only, unprivileged temporary container. Runtime uses ONNX Runtime
and does not import PyTorch or Ultralytics. Export used Ultralytics 8.4.0,
PyTorch 2.5.1 CPU, ONNX 1.16.2, opset 12, and a fixed 1280-by-1280 input.
`onnx.checker.check_model` passed. The output is the YOLOv10 end-to-end shape
`[1, 300, 6]` (`x1`, `y1`, `x2`, `y2`, confidence, class id).

The source model repository does not provide a clear standalone license for
the checkpoint. The exported ONNX metadata identifies the Ultralytics license
as AGPL-3.0. Verify licensing requirements before redistribution or commercial
deployment.

Checksums:

```text
source yolov10 best.pt:
5dfdf0871ce8e5a064f0cb71a8a59cd35d3887c6251b92029f5e7f49b068414a

elevator_buttons_yolov10s.onnx (30,192,276 bytes):
82b0fe29f14556290b7e925f2e05bf7d3b6e6996d36f7533db247dcd9f92ea32
```
