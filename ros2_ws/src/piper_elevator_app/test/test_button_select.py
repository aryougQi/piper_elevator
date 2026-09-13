"""Exercise discovery and acknowledgement over private, non-motion topics."""
import threading
import uuid

import pytest
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import String

from piper_elevator_app.button_select import select_button


@pytest.mark.parametrize('reply, subscribers, expected', [
    (True, 2, True), (False, 2, False), (True, 1, False),
])
def test_selection_requires_discovery_and_a_new_ack(reply, subscribers, expected):
    rclpy.init()
    suffix = uuid.uuid4().hex
    fake = Node('selection_test_detector_' + suffix)
    client = Node('selection_test_client_' + suffix)
    command_topic = '/selection_test_' + suffix + '/command'
    ack_topic = '/selection_test_' + suffix + '/ack'
    qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    ack = fake.create_publisher(String, ack_topic, qos)
    # This old acknowledgement must not make a missing new reply succeed.
    ack.publish(String(data='3'))
    received = []

    def command(message):
        received.append(message.data)
        # Simulate a lost first application acknowledgement to test retries.
        if reply and len(received) >= 2:
            ack.publish(String(data=message.data))

    fake.create_subscription(String, command_topic, command, qos)
    if subscribers == 2:
        fake.create_subscription(String, command_topic, lambda m: None, qos)
    executor = SingleThreadedExecutor()
    executor.add_node(fake)
    thread = threading.Thread(target=executor.spin)
    thread.start()
    try:
        assert select_button(client, '3', timeout=2.0,
                             selection_topic=command_topic,
                             selected_topic=ack_topic) is expected
        if expected:
            assert len(received) >= 2
        if subscribers == 1:
            assert not received
    finally:
        executor.shutdown()
        thread.join()
        client.destroy_node()
        fake.destroy_node()
        rclpy.shutdown()
