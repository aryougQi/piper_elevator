# Elevator button models

## 当前运行时模型：`elevator_buttons_yolo11s.onnx`

自训 YOLO11-S 检测模型，是按钮检测节点当前默认加载的模型
（`config/button_detector.yaml` 的 `model_path`）。

- 14 类：楼层 `1`-`10`、`up`、`down`、`open`、`close`
- 固定输入 `1x3x640x640`（训练 imgsz=640）
- 原始 YOLOv8/v11 输出格式 `[1, 18, 8400]`（4 框坐标 + 14 类分数），
  由 `detector_core.YoloOnnxDetector` 解码，类别名从 ONNX 元数据读取
- 训练数据：自建电梯按钮数据集（test 划分 930 张）
- 源 checkpoint：`ultralytics-8.3.163` 训练的 `best.pt`（YOLO11s，
    9,418,218 参数）
- 导出：Ultralytics 8.3.163，`imgsz=640 opset=12 dynamic=False
  simplify=True`（onnxslim 0.1.96）。torch>=2.9 默认 dynamo 导出器
  不支持 opset 12，导出时需强制 `dynamo=False` 走传统导出器
- 类别中不含 `alarm` 和 `intercom`：仿真竖版面板的重标注
  （`simulation_panel_layout_labels` 含这两个服务键）会安全跳过，
  仿真检测依赖模型原始输出

Checksum：

```text
best.pt (source):
（见训练工程 Yolo_Train/ultralytics-8.3.163/best.pt）

elevator_buttons_yolo11s.onnx (37,948,432 bytes):
8ac4864d644ceae27d0e5f445fc91a843217eb05a55a29f72d6083357eba785c
```

## 备用模型：`elevator_buttons_yolov10s.onnx`

上一代模型，YOLOv10-S，来源
[`isharadilshanra/YOLOv10-Elevator-Button-Detection`](https://github.com/isharadilshanra/YOLOv10-Elevator-Button-Detection)，
368 类、1280 输入、端到端输出 `[1, 300, 6]`。保留作回退：将
`button_detector.yaml` 的 `model_path` 改回
`models/elevator_buttons_yolov10s.onnx`、`model_input_size` 改回
`1280` 即可。

导出细节（原文记录）：Ultralytics 8.4.0、PyTorch 2.5.1 CPU、ONNX
1.16.2、opset 12、固定 1280x1280 输入。源 checkpoint 无明确独立
license，导出元数据标识 Ultralytics AGPL-3.0；再分发或商用前需确认
授权。

Checksums：

```text
source yolov10 best.pt:
5dfdf0871ce8e5a064f0cb71a8a59cd35d3887c6251b92029f5e7f49b068414a

elevator_buttons_yolov10s.onnx (30,192,276 bytes):
82b0fe29f14556290b7e925f2e05bf7d3b6e6996d36f7533db247dcd9f92ea32
```
