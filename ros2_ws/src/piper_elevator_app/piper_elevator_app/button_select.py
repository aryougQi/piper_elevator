"""Select a button with subscriber discovery and a detector acknowledgement."""
import argparse
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import String


def select_button(node, button, timeout=8.0, min_subscribers=2,
                  selection_topic='/button_selection',
                  selected_topic='/button_selected'):
    """Repeat an idempotent selection until the detector acknowledges it.

    Listen with volatile durability so a retained acknowledgement from an
    earlier command cannot falsely confirm this request.
    """
    confirmed = [False]
    sent = [False]

    def acknowledge(message):
        if sent[0] and message.data == button:
            confirmed[0] = True

    subscription = node.create_subscription(String, selected_topic, acknowledge, 10)
    publisher = node.create_publisher(String, selection_topic, QoSProfile(
        depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
    ))
    deadline = time.monotonic() + timeout
    next_publish = 0.0
    try:
        while rclpy.ok() and time.monotonic() < deadline:
            if (publisher.get_subscription_count() >= min_subscribers
                    and time.monotonic() >= next_publish):
                sent[0] = True
                publisher.publish(String(data=button))
                next_publish = time.monotonic() + 0.5
            rclpy.spin_once(node, timeout_sec=0.05)
            if confirmed[0]:
                # The detector acknowledgement proves its callback ran;
                # wait for reliable delivery to the other matched readers.
                from rclpy.duration import Duration
                if publisher.wait_for_all_acked(Duration(seconds=max(
                        0.0, deadline - time.monotonic()))):
                    return True
                return False
        return False
    finally:
        node.destroy_subscription(subscription)
        node.destroy_publisher(publisher)


def main(args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('button', help='Button class, e.g. 3, up, open, clear')
    parser.add_argument('--timeout', type=float, default=8.0)
    parser.add_argument('--min-subscribers', type=int, default=2,
                        help='Use 1 when running only the detector')
    options = parser.parse_args(args)
    if options.timeout <= 0 or options.min_subscribers < 1:
        parser.error('timeout and min-subscribers must be positive')
    button = options.button.strip()
    if button.casefold() in {'clear', 'none'}:
        button = ''
    rclpy.init(args=[])
    node = Node('button_selection_client')
    try:
        if not select_button(node, button, options.timeout, options.min_subscribers):
            node.get_logger().error(
                'Selection unconfirmed: check detector, subscriber count and /button_selected'
            )
            return_code = 1
        else:
            node.get_logger().info(f'Button selection confirmed: {button or "<none>"}')
            return_code = 0
    finally:
        node.destroy_node()
        rclpy.shutdown()
    raise SystemExit(return_code)


if __name__ == '__main__':
    main()
