# 独立按钮几何实验：中心定位与面板取样

## 与现有算法隔离

按用户要求，此实验仅额外实现，不改变当前检测器算法、参数、模型、ROS 话题或
Servo。上一轮对 `detector_core.py`、`yolo_button_detector.py`、
`button_detector.yaml` 和原接口测试的接入修改已撤回；这四个文件与本轮开始前的
原始版本一致。新模块没有 console entry point，不会随任何 launch 自动启动。

新增代码：

- `ros2_ws/src/piper_elevator_app/piper_elevator_app/experimental_button_geometry.py`
- `ros2_ws/src/piper_elevator_app/piper_elevator_app/plane_core.py`
- `ros2_ws/diagnostics/scripts/replay_surface_consensus.py`
- `ros2_ws/src/piper_elevator_app/test/test_experimental_button_geometry.py`
- `ros2_ws/src/piper_elevator_app/test/test_plane_core.py`

## 中心精定位

输入当前帧 RGB 和该帧检测框（xyxy，原图坐标）。当前版本针对录制中具有完整
四边形轮廓的方形按钮，参考 OpenCV 的轮廓近似方法：

1. 在检测框周围的小区域使用边缘和阈值生成轮廓候选。
2. 按面积、尺寸、凸性、轮廓近似误差和中心偏移筛选完整按钮边界。
3. 用四边形对角线交点作为透视投影中心，避免直接使用文字中心或框中心。
4. 多个近似同等可信但中心冲突的边界、过小按钮、图像边缘截断、轮廓不足时拒绝。

输出包含 `center`、`corners`、`reason`、`candidates`。拒绝时 center 为 None，
不会把原始框中心伪装成精定位结果。该方法目前不支持圆形、严重遮挡或任意形状按钮，
也不能保证所有反光四边形都可与真实按钮边界区分。没有新增类别识别或目标跟踪。

## 面板取样与按钮深度分离

`panel_sampling_mask` 在目标框的 2.5 倍范围内取样，排除扩张到 1.25 倍的目标框
和所有已知相邻按钮框。可传入独立确认的 `panel_roi`，进一步排除面板外背景。
没有 panel_roi 时只是候选周围区域，不能保证全部来自面板。

`estimate_sampled_surface` 对当前帧有效深度去畸变、反投影，以 RANSAC 拟合平面。
默认最多 800 点、至少 60 点、70% 支持率、1.5 mm 点到平面残差阈值，平面内点
还须覆盖中心周围至少三个象限（每象限至少八点），拒绝退化点集和过大倾角。
这些阈值是实验质量门槛，不是实机精度保证，也没有通过放宽门槛追求有效帧数。

`SurfaceEstimate.panel_point` 是中心射线与面板估计平面的交点，**不是按钮可按压
位置**。`measure_button_point` 独立从精定位中心附近的原始深度估计按钮位置，
检查有效比例和深度分位数跨度；失败返回 None，不用面板深度或历史帧填补。
深度 uint16 按 unit_scale 转米，浮点深度按米解释。RGB 和深度必须已对齐，
使用对应 RGB 内参；调用者必须保证同帧，核心函数不会猜测时间戳。

## 开源参考与许可

- OpenCV 四边形轮廓示例：
  https://github.com/opencv/opencv/blob/4.x/samples/python/squares.py
- OpenCV 轮廓近似文档：
  https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html
- OpenCV 当前主分支许可证（Apache-2.0）：
  https://github.com/opencv/opencv/blob/4.x/LICENSE
- Open3D RANSAC 平面分割源码：
  https://github.com/isl-org/Open3D/blob/main/cpp/open3d/geometry/PointCloudSegmentation.cpp
- Open3D MIT 许可证：
  https://github.com/isl-org/Open3D/blob/main/LICENSE

参考上述算法流程，使用现有 OpenCV API 与独立编写的 NumPy 实现，没有复制上游
示例源码或增加 Open3D 运行依赖。本轮没有安装新的模型或修改系统依赖。

## 运行

在已加载项目 Python 包且有 NumPy/OpenCV/ONNX Runtime 的环境执行：

```bash
python3 /workspace/ros2_ws/diagnostics/scripts/replay_surface_consensus.py
```

脚本路径和默认输入输出均从自身位置解析；宿主机使用实际绝对路径即可。
可传一个或多个 NPZ 路径及 `--output /path/results.json`。脚本不创建 ROS 节点、
不打开相机、不发布话题、不调用任何运动服务。默认输出：

`ros2_ws/diagnostics/data/button_geometry_v2_20260913/`

NPZ 格式沿用已有诊断录制：depths、boxes（中心 x/y/宽/高）、rotations、stamps、
metadata_json；可选 all_boxes_json（逐帧 xyxy）。旁边的 `<stem>_color.png` 仅在
metadata 的 color_stamp_ns 与某一深度帧精确相等时用于中心评估，绝不将静态
RGB 图片复用于整段深度序列。不具有全部按钮框的录制会标记这一限制。

绿色覆盖区为候选面板取样区域，蓝色十字为检测框中心，红色为精定位中心，黄色为
轮廓。报告列出每帧拒绝原因的汇总、有效帧索引、法向离散度与耗时。两种 RANSAC
版本的参数不同，这是端到端候选方案比较，不是仅改变采样区域的控制变量实验。

## 2026-09-13 实际结果与结论

三组保留录制共 300 帧：

| 录制 | 原算法有效帧 | 独立框内 RANSAC | 独立面板 RANSAC |
|---|---:|---:|---:|
| vision_stability_current | 100/100 | 58/100 | 0/100 |
| surface_support_diagnostic | 80/80 | 68/80 | 0/80 |
| vision_stability_full_context | 120/120 | 80/120 | 0/120 |

面板方案全部因 `no_plane_consensus` 拒绝。full_context 首帧候选区域有效深度
10%–90% 分位为 472–494 mm；单凭这种分布不能将变化归因于传感器噪声或面板形状，
但它确实未通过当前平面拟合门槛。不能声称已经获得稳定法向。

只有 current 和 full_context 的各一张 RGB 与深度严格同步：

| 录制 | 原中心像素 | 新中心像素 | 独立按钮深度 |
|---|---|---|---|
| current | (383.30, 226.91) | (383.38, 224.67) | 485 mm |
| full_context | (383.20, 226.30) | (382.35, 224.67) | 拒绝，局部深度不满足质量条件 |

第三组图片没有对应深度时间戳，跳过中心评估。以上中心修正约 1.8–2.2 像素，
没有人工中心真值，不能据此声称精度提高，也不能用两张图证明视频跟踪稳定性。
需后续采集同帧 RGB-D 序列、标注真实边界/中心、提供面板边界，再评价中心误差、
误定位率、有效输出率和法向误差。实验尚未接入实机在线识别。

新增离线测试覆盖偏框数字干扰、透视中心、图标误判、遮挡、相邻按钮排除、
面板边界、单侧支持、按钮深度缺失、深度单位和畸变校正。

本轮验证：23 项实验几何测试通过；应用扩展回归 999 项通过（宿主 ROS Jazzy，
排除需要 ROS 通信的 test_button_select.py 和需要已安装模型包的
 test_robot_description.py）。四个原检测代码/配置/测试文件的 Git diff 为空；
未进行目标 Humble 容器测试或实机运动验证。

## 独立中心优化：边缘直线精修

2026-09-14：对四边形每条边的中段使用 OpenCV fitLine（L1）拟合，排除圆角端点；
相邻直线交点提供浮点角点。边缘残差、角点位移或凸性不通过时保留此前候选角点，
并通过 CenterEstimate.edge_refined / edge_residual_px 明确记录是否使用精修。
仅作用于独立实验函数；refine_edges=False 可对比上一版。

固定种子 100 张透视、亚像素偏移、模糊、噪声和文字干扰的合成图，两个版本均接受
100 张。相对已知投影中心的误差中位数由 0.635 px 降为 0.171 px，P95 由
1.160 px 降为 0.361 px。这是合成图结果，不能等同于实机精度或视频稳定性。
复现脚本：ros2_ws/diagnostics/scripts/benchmark_button_center.py；结果保留于
ros2_ws/diagnostics/data/button_geometry_v3_20260913/center_benchmark.json。

面板方向另外试验了 6/10 像素分块中值，在三组录制每隔五帧抽样后仍未得到
通过门槛的平面；未将此试验加入候选识别模块。23 项独立几何测试通过，
现有 detector_core.py、yolo_button_detector.py 和 button_detector.yaml 未改变。
