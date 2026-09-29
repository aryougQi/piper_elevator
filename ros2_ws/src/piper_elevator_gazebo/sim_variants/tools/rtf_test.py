import time
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Image

rclpy.init()
n = Node('rtf_test')
clocks = []
n.create_subscription(Clock, '/clock', lambda m: clocks.append(m.clock.sec + m.clock.nanosec*1e-9), 10)

def measure(seconds, label, with_images):
    clocks.clear()
    sub = None
    if with_images:
        sub = n.create_subscription(Image, '/camera/color/image_raw',
                                     lambda m: None, qos_profile_sensor_data)
        time.sleep(1.0)  # let the lazy bridge activate
        clocks.clear()
    t0 = time.monotonic()
    while time.monotonic() - t0 < seconds:
        rclpy.spin_once(n, timeout_sec=0.05)
    if len(clocks) > 3:
        dt = clocks[-1] - clocks[0]
        print(f'{label}: RTF={dt/(len(clocks) and (time.monotonic()-t0)):.3f} '
              f'(sim {dt:.2f}s)', flush=True)
    else:
        print(f'{label}: clock samples {len(clocks)}', flush=True)
    if sub is not None:
        n.destroy_subscription(sub)
        time.sleep(1.0)

measure(8, '无图像订阅', False)
measure(8, '有图像订阅', True)
measure(8, '取消订阅后', False)
n.destroy_node(); rclpy.shutdown()
