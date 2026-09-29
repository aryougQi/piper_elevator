import rclpy, time
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

rclpy.init()
node = Node("grab")
latest = {}
bridge = CvBridge()
def cb(m): latest["img"] = bridge.imgmsg_to_cv2(m, desired_encoding="bgr8")
node.create_subscription(Image, "/camera/color/image_raw", cb, qos_profile_sensor_data)
t0 = time.time()
while time.time()-t0 < 6 and not latest:
    rclpy.spin_once(node, timeout_sec=0.1)
img = latest["img"]
cv2.imwrite("/tmp/grab_now.png", img)
print("mean:", round(float(img.mean()),1), "shape:", img.shape)
node.destroy_node(); rclpy.shutdown()
