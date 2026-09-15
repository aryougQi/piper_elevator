"""Performance changes preserve input ownership, timestamps and target output."""
import threading
from types import SimpleNamespace

import numpy as np
from cv_bridge import CvBridge
from test_sam2_tracking import Tracker, frame_mask


def debug_tracker():
    t = Tracker()
    t.values.update(debug_image=True, debug_max_fps=10., debug_image_only_when_subscribed=True)
    t._debug_condition = threading.Condition()
    t._debug_pending = None
    t._stop = threading.Event()
    t._performance_debug_count = 0
    t._bridge = CvBridge()
    t._debug_pub = SimpleNamespace(get_subscription_count=lambda:1, publish=lambda m:None)
    return t


def test_debug_keeps_latest_frame_without_copying_or_queuing_history():
    t = debug_tracker()
    frame, mask = frame_mask()
    t._queue_debug(frame, mask, 'old', 20., 20.)
    t._queue_debug(frame, mask, 'new', 21., 20.)
    assert t._debug_pending[0] is frame
    assert t._debug_pending[1] is mask
    assert t._debug_pending[2] == 'new'
    t._debug_pending = None
    t._debug_pub.get_subscription_count = lambda:0
    t._queue_debug(frame, mask, 'unsubscribed', 20., 20.)
    assert t._debug_pending is None


def test_slow_debug_subscriber_does_not_block_fresh_surface_output():
    t = debug_tracker()
    entered, release, output_finished = threading.Event(), threading.Event(), threading.Event()
    messages = []
    def slow_publish(message):
        messages.append(message)
        entered.set()
        release.wait(2.)
    t._debug_pub.publish = slow_publish
    frame, mask = frame_mask()
    worker = threading.Thread(target=t._debug_loop)
    worker.start()
    producer = None
    try:
        t._publish_if_current(frame, mask, 1_000_000_000, 0)
        t._publish_output(frame, mask, 1_000_000_000)
        assert entered.wait(1.)
        def publish_next():
            t._publish_if_current(frame, mask, 1_000_000_001, 0)
            t._publish_output(frame, mask, 1_000_000_001)
            output_finished.set()
        producer = threading.Thread(target=publish_next)
        producer.start()
        assert output_finished.wait(1.), 'debug publication blocked the tracking target'
        assert len(t.poses) == 2
        assert t.poses[-1].header.stamp.nanosec == 1
        assert messages[0].header.stamp.sec == 1
        np.testing.assert_array_equal(frame, np.zeros_like(frame))
    finally:
        release.set()
        t._stop.set()
        with t._debug_condition:
            t._debug_condition.notify_all()
        worker.join(2.)
        if producer:
            producer.join(2.)


def test_debug_drops_inflight_image_on_target_change():
    t = debug_tracker()
    frame, mask = frame_mask()
    published = []
    t._debug_pub.publish = published.append
    def render(*args):
        t._generation += 1
        t._stop.set()
        return frame
    t._render_debug = render
    from std_msgs.msg import Header
    t._queue_debug(frame, mask, Header(), 20., 20.)
    t._debug_loop()
    assert not published


def test_slow_geometry_does_not_block_mask_and_keeps_only_latest():
    t = Tracker()
    t.values['geometry_max_fps'] = 1000.
    t._stop = threading.Event()
    entered, release = threading.Event(), threading.Event()
    processed = []
    def transform(*args):
        entered.set()
        release.wait(2.)
        return np.zeros(3), np.eye(3)
    t._camera_transform = transform
    t._publish_output = lambda frame, mask, stamp, *args: processed.append(stamp)
    frame, mask = frame_mask()
    t._publish_if_current(frame, mask, 1_000_000_000, 0)
    worker = threading.Thread(target=t._geometry_loop)
    worker.start()
    try:
        assert entered.wait(1.)
        for i in range(1, 21):
            t._publish_if_current(frame, mask, 1_000_000_000+i, 0)
        assert t._stable_count == 21
        assert t._geometry_pending[2] == 1_000_000_020
        assert not processed
        assert t.states[-1]['selected']['tracking_valid']
    finally:
        t._stop.set()
        release.set()
        with t._geometry_condition:
            t._geometry_condition.notify_all()
        worker.join(2.)
    assert not worker.is_alive()


def test_generation_change_during_plane_fit_discards_pose_and_anchor():
    t = Tracker()
    original = t._registered_target
    def fit(*args):
        result = original(*args)
        t._reset_tracking(t.IDLE, 'selection_changed')
        return result
    t._registered_target = fit
    t._publish_output(*frame_mask(), 1_000_000_000, generation=0)
    assert not t.poses
    assert t._anchor_world is None
    assert t._output_stamp_ns == 0


def test_repeated_depth_failure_preserves_mask_tracking_and_recovers():
    t = Tracker()
    frame, mask = frame_mask()
    good_depth = t._depth_frames[0]
    t._depth_frames.clear()
    for i in range(10):
        t._publish_if_current(frame, mask, 1_000_000_000+i, 0)
        t._publish_output(frame, mask, 1_000_000_000+i, generation=0)
    assert t._state == t.TRACKING and not t.poses
    assert t.states[-1]['selected']['tracking_valid']
    assert not t.states[-1]['selected']['geometry_valid']
    t._depth_frames.append(good_depth)
    t._publish_output(frame, mask, 1_000_000_010, generation=0)
    assert t.states[-1]['reason'] == 'tracker_ready'
    assert len(t.poses) == 1


def test_yolo_pause_requires_verified_coarse_and_expires_on_loss():
    from piper_elevator_app.yolo_button_detector import ButtonDetector
    from std_msgs.msg import String
    from rclpy.time import Time
    import json
    t = SimpleNamespace(_sam2_coarse_verified=False, _sam2_servo_started=False, _selected_button_class='2',
        get_clock=lambda:SimpleNamespace(now=lambda:Time(nanoseconds=1_000_000_000)))
    payload = dict(source='sam2_button_tracker', state='TRACKING',
        stamp=dict(sec=1,nanosec=0), selected=dict(class_name='2',tracking_valid=True))
    send = lambda:ButtonDetector._sam2_tracking_callback(t,String(data=json.dumps(payload)))
    send()
    assert not ButtonDetector._sam2_pauses_inference(t)
    t._sam2_coarse_verified=True
    # A delayed manual start must keep receiving fresh YOLO observations.
    for _ in range(20):
        send()
        assert not ButtonDetector._sam2_pauses_inference(t)
    ButtonDetector._sam2_servo_callback(t,String(data='STARTING_MOVEIT_SERVO'))
    send()
    assert ButtonDetector._sam2_pauses_inference(t)
    payload['state']='LOST'
    send()
    assert not ButtonDetector._sam2_pauses_inference(t)


def test_yolo_resumes_when_servo_start_fails_or_new_approach_begins():
    from piper_elevator_app.yolo_button_detector import ButtonDetector
    from std_msgs.msg import String
    from time import monotonic
    t = SimpleNamespace(_sam2_coarse_verified=True, _sam2_servo_started=True,
                        _sam2_pause_until=monotonic()+10.)
    ButtonDetector._sam2_servo_callback(t, String(data='STOPPED: SAM2 tracking lost'))
    assert not t._sam2_servo_started
    assert not ButtonDetector._sam2_pauses_inference(t)
    t._sam2_servo_started=True
    t._sam2_pause_until=monotonic()+10.
    ButtonDetector._sam2_approach_callback(t,String(data='PLANNING'))
    assert not t._sam2_servo_started and not t._sam2_coarse_verified
    assert not ButtonDetector._sam2_pauses_inference(t)
