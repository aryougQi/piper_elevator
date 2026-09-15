"""Strict handover and RGB-D validation without running robot controls."""
from collections import deque
import json
import threading
from types import SimpleNamespace
import numpy as np
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import String
from piper_elevator_app.sam2_button_tracker import Sam2ButtonTracker
from piper_elevator_app.button_visual_servo import ButtonVisualServo
from test_visual_semantic_safety import SemanticHarness, tracking_payload

class Tracker(Sam2ButtonTracker):
    def __init__(self):
        self.values = dict(stable_frames=5, use_mask_median_depth=True, init_max_age_sec=.15,
                           min_depth_m=.1, max_depth_m=2., debug_image=False,
                           max_geometry_failures=1,
                           min_mask_area_px=10, max_mask_area_ratio=.6, max_center_jump_px=150.)
        self._lock=threading.RLock()
        self._frame_event=threading.Event()
        self._generation=0; self._frame_stamp_ns=1_000_000_000
        self._processed_stamp_ns=0; self._output_stamp_ns=0; self._tracking_stamp_ns=0
        self._geometry_condition=threading.Condition(); self._geometry_pending=None
        self._state=self.TRACKING; self._selected_label='2'; self._frame_id='camera'
        self._stable_count=0; self._last_center=None; self._geometry_failures=0
        self._anchor_world=None; self._seed_center=None
        self._support_radius_m=.04
        self._camera_transform=lambda stamp,frame:(np.zeros(3),np.eye(3))
        self._camera_info=CameraInfo(width=40,height=40,k=[30.,0.,20.,0.,30.,20.,0.,0.,1.])
        self._camera_info.header.frame_id='camera'
        self._surface_pose=PoseStamped(); self._surface_pose.header.frame_id='camera'
        self._surface_pose.pose.orientation.w=1.
        self._depth_frames=deque([(1_000_000_000,np.full((40,40),.4,dtype=np.float32))])
        self.states=[]; self.poses=[]
        self._state_pub=SimpleNamespace(publish=lambda m:self.states.append(json.loads(m.data)))
        self._pose_pub=SimpleNamespace(publish=self.poses.append)
        self._center_pub=SimpleNamespace(publish=lambda m:None)
    def get_logger(self): return SimpleNamespace(warning=lambda *a,**kw:None)
    def get_parameter(self,name): return SimpleNamespace(value=self.values[name])

def frame_mask():
    mask=np.zeros((40,40),bool); mask[15:25,15:25]=True
    return np.zeros((40,40,3),np.uint8),mask

def test_ready_requires_stable_masks_and_valid_geometry():
    t=Tracker(); frame,mask=frame_mask()
    for i in range(4): t._publish_if_current(frame,mask,1_000_000_000+i,0)
    assert not t.states
    t._publish_if_current(frame,mask,1_000_000_004,0)
    assert t.states[-1]['selected']['tracking_valid']
    assert not t.states[-1]['selected']['geometry_valid']
    t._publish_output(frame,mask,1_000_000_004)
    assert t.states[-1]['reason']=='tracker_ready'
    assert t.states[-1]['selected']['depth_valid']

def test_invalid_depth_never_announces_ready():
    t=Tracker(); t._stable_count=4
    t._depth_frames=deque([(1_000_000_000,np.zeros((40,40),np.float32))])
    t._publish_output(*frame_mask(),1_000_000_000)
    assert not t.poses
    assert t.states[-1]['state']=='TRACKING'
    assert not t.states[-1]['selected']['geometry_valid']
    assert not t.states[-1]['selected']['stable_detection']

def test_task_generation_discards_inflight_output():
    t=Tracker(); old=t._generation
    t._selected_callback(String(data='up'))
    t._publish_if_current(*frame_mask(),1_000_000_000,old)
    assert not t.poses and t._state=='IDLE'

def test_same_selection_does_not_reset_active_tracker():
    t=Tracker(); t._selected_callback(String(data='2'))
    assert t._generation==0 and t._state=='TRACKING'

def test_strict_servo_ignores_yolo_pose():
    received=[]
    t=SimpleNamespace(_require_sam2=True,_surface_pose_callback=received.append)
    ButtonVisualServo._yolo_surface_callback(t,'yolo')
    ButtonVisualServo._sam2_surface_callback(t,'sam2')
    assert received==['sam2']

def test_strict_servo_rejects_yolo_ready_and_stops_on_sam2_loss():
    t=SemanticHarness(); t._require_sam2=True; t._sam2_ready=False
    t._sam2_received_at=0; t._starting=False
    payload=tracking_payload(reason='')
    t._tracking_state_callback(String(data=json.dumps(payload)))
    assert not t._sam2_ready
    payload.update(source='sam2_button_tracker',state='TRACKING',reason='tracker_ready')
    t._tracking_state_callback(String(data=json.dumps(payload)))
    assert t._sam2_ready
    t._running=True; payload.update(state='LOST',reason='mask_quality_failed')
    t._tracking_state_callback(String(data=json.dumps(payload)))
    assert t._stop_event.is_set() and not t._sam2_ready


def test_worker_does_not_count_same_camera_frame_twice():
    from rclpy.time import Time
    t=Tracker(); frame,mask=frame_mask()
    t._frame=frame; t._coarse_ready_at=None; t._detections=None
    t._detection_stamp_ns=0; t._stop=threading.Event()
    t.values.update(enabled=True,tracking_timeout_sec=1.)
    t.get_clock=lambda:SimpleNamespace(now=lambda:Time(nanoseconds=1_000_000_000))
    calls=[]
    t._backend=SimpleNamespace(reset=lambda:None,track=lambda frame:(calls.append(1) or mask))
    worker=threading.Thread(target=t._worker_loop); worker.start()
    t._stop.wait(.10);t._stop.set();worker.join(1)
    assert len(calls)==1 and t._stable_count==1


def test_strict_manager_cannot_be_released_by_yolo_alone():
    import pytest
    from piper_elevator_app.elevator_task_manager import TaskFailure
    from test_task_execution import PostMotionHarness
    t=PostMotionHarness(); t.parameters['require_sam2_tracking']=True
    t._sam2_ready=False;t._sam2_lost=False
    with pytest.raises(TaskFailure,match='no fresh RGB-D target'):
        t._wait_for_post_motion_target('2')


def test_strict_manager_requires_new_sam2_ready_and_surface_frames():
    from rclpy.time import Time
    from piper_elevator_app.elevator_task_manager import ElevatorTaskManager
    from test_task_execution import PostMotionHarness
    t=PostMotionHarness();t.parameters['require_sam2_tracking']=True
    t._sam2_ready=False;t._sam2_lost=False;t._sam2_stamp_ns=0
    t.get_clock=lambda:SimpleNamespace(now=lambda:Time(nanoseconds=10_000_000_000))
    def receive(message):
        payload=tracking_payload(reason='tracker_ready')
        payload.update(source='sam2_button_tracker',state='TRACKING')
        ElevatorTaskManager._tracking_state_callback(t,String(data=json.dumps(payload)))
        for _ in range(3): ElevatorTaskManager._sam2_surface_callback(t,PoseStamped())
    t._selection_publisher=SimpleNamespace(publish=receive)
    t._wait_for_post_motion_target('2')
    assert t._sam2_sequence==1 and t._sam2_surface_sequence==3


def test_old_ready_cannot_release_next_post_motion_wait():
    import pytest
    from piper_elevator_app.elevator_task_manager import TaskFailure
    from test_task_execution import PostMotionHarness
    t=PostMotionHarness();t.parameters['require_sam2_tracking']=True
    t._sam2_ready=True;t._sam2_lost=False;t._sam2_sequence=9
    t._sam2_surface_sequence=9;t._sam2_stamp_ns=9_950_000_000
    with pytest.raises(TaskFailure,match='no fresh RGB-D target'):
        t._wait_for_post_motion_target('2')


def test_cropped_mask_keeps_physical_center_and_requires_live_support():
    import pytest
    t=Tracker(); frame,mask=frame_mask();t._seed_center=(20.,20.)
    depth=t._depth_frames[0][1]
    point,_=t._registered_target(mask,depth,t._camera_info,1_000_000_000)
    cropped=mask.copy();cropped[21:]=False
    tracked,_=t._registered_target(cropped,depth,t._camera_info,1_000_000_001)
    assert np.allclose(tracked,point,atol=1e-8)
    displaced=np.zeros_like(mask);displaced[:10,:10]=True
    with pytest.raises(ValueError,match='no longer supports'):
        t._registered_target(displaced,depth,t._camera_info,1_000_000_002)


def test_transient_handover_race_retries_without_reusing_a_failed_claim(monkeypatch):
    import piper_elevator_app.button_visual_servo as module
    from rclpy.time import Time
    replies=deque([SimpleNamespace(success=False,message='Near-view observation changed during handover; retry'),SimpleNamespace(success=True,message='token')])
    calls=[]
    client=SimpleNamespace(wait_for_service=lambda **kw:True,call_async=lambda req:(calls.append(req) or replies.popleft()))
    t=SimpleNamespace(_coarse_handover_client=client,_stop_event=threading.Event(),
        _wait_for_future=lambda future,timeout:future,_selected_button='up',_base_frame='base_link',
        get_parameter=lambda name:SimpleNamespace(value=1.),get_clock=lambda:SimpleNamespace(now=lambda:Time(nanoseconds=100)))
    monkeypatch.setattr(module,'decode_coarse_handover',lambda message,**kw:message)
    assert ButtonVisualServo._claim_coarse_handover(t)=='token'
    assert len(calls)==2


def test_mask_only_updates_do_not_refresh_geometry_or_stop_servo():
    t=SemanticHarness(); t._require_sam2=True; t._starting=False
    t._sam2_ready=True; t._sam2_received_at=123.
    for reason in ('tracking_valid', 'geometry_invalid'):
        payload=tracking_payload(reason=reason)
        payload.update(source='sam2_button_tracker', state='TRACKING')
        t._tracking_state_callback(String(data=json.dumps(payload)))
        assert t._sam2_ready
        assert t._sam2_received_at == 123.
        assert not t._stop_event.is_set()
