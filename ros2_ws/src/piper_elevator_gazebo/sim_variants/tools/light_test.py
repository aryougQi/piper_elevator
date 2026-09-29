import rclpy, time, subprocess
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
import numpy as np
from cv_bridge import CvBridge

rclpy.init()
node = Node("light_test")
latest = {}
bridge = CvBridge()
def cb(m): latest["img"] = bridge.imgmsg_to_cv2(m, desired_encoding="bgr8")
node.create_subscription(Image, "/camera/color/image_raw", cb, qos_profile_sensor_data)

def grab(timeout=6.0):
    latest.clear()
    t0 = time.time()
    while time.time()-t0 < timeout and len(latest) == 0:
        rclpy.spin_once(node, timeout_sec=0.1)
    for _ in range(10): rclpy.spin_once(node, timeout_sec=0.05)
    return latest.get("img")

def create_light(name, intensity, rgb):
    req = (f'light {{ name: "{name}" type: DIRECTIONAL '
           f'pose {{ position {{ x: 0 y: 0 z: 3 }} }} '
           f'direction {{ x: -0.5 y: 0 z: -1 }} '
           f'diffuse {{ r: {rgb[0]} g: {rgb[1]} b: {rgb[2]} a: 1.0 }} '
           f'intensity: {intensity} cast_shadows: false }}')
    subprocess.run(["ign","service","-s","/world/button_press/create",
        "--reqtype","ignition.msgs.EntityFactory","--reptype","ignition.msgs.Boolean",
        "--timeout","5000","--req",req], capture_output=True, check=True)

def remove_light(name):
    subprocess.run(["ign","service","-s","/world/button_press/remove",
        "--reqtype","ignition.msgs.Entity","--reptype","ignition.msgs.Boolean",
        "--timeout","5000","--req",f'name: "{name}" type: 1'],
        capture_output=True, check=True)

img0 = grab()
print("基准亮度:", round(float(img0.mean()),1))

create_light("bright_test", 2.5, (1.0, 0.85, 0.6))
time.sleep(1.5)
img1 = grab()
print("加亮光后:", round(float(img1.mean()),1))

remove_light("bright_test")
time.sleep(1.5)
img2 = grab()
print("删光后:", round(float(img2.mean()),1))
print("差异像素均值:", round(float(np.abs(img1.astype(int)-img0.astype(int)).mean()),2))
node.destroy_node(); rclpy.shutdown()
