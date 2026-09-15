"""Reject unsafe intermediate camera poses, using future FK and controller splines."""
from types import SimpleNamespace
import numpy as np
import pytest
from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import PoseStamped
from trajectory_msgs.msg import JointTrajectoryPoint
from piper_elevator_app.button_approach_planner import interpolate_positions
from test_approach_planning import PlannerHarness


def point(t,p,v=None,a=None):
    result=JointTrajectoryPoint(positions=[float(p)])
    result.time_from_start.sec=int(t); result.time_from_start.nanosec=int((t-int(t))*1e9)
    if v is not None: result.velocities=[float(v)]
    if a is not None: result.accelerations=[float(a)]
    return result


def harness():
    h=PlannerHarness(); h._fk_client=object()
    h._latest_camera_info=CameraInfo(width=640,height=480,k=[300.,0.,320.,0.,300.,240.,0.,0.,1.])
    def fk(joints,deadline=None):
        pose=PoseStamped(); pose.pose.orientation.w=1.
        pose.pose.position.x=joints['joint1']
        return pose
    h._fk_pose=fk
    obs=dict(button=np.array([0.,0.,1.]),tip_to_camera_translation=np.zeros(3),tip_to_camera_quaternion=[0.,0.,0.,1.])
    return h,obs


def trajectory(points): return SimpleNamespace(joint_trajectory=SimpleNamespace(joint_names=['joint1'],points=points))


def test_midpath_out_of_view_is_rejected_even_when_endpoints_centered():
    h,o=harness()
    ok,reason,_=h._validate_trajectory_fov(trajectory([point(0,0),point(.3,1),point(.6,0)]),o)
    assert not ok and 'FOV validation failed' in reason


def test_valid_trajectory_preserves_margin():
    h,o=harness(); ok,_,margin=h._validate_trajectory_fov(trajectory([point(0,0),point(.3,.1)]),o)
    assert ok and margin>=60


def test_behind_camera_is_rejected():
    h,o=harness(); o['button'][2]=-1
    assert not h._validate_trajectory_fov(trajectory([point(0,0),point(.3,0)]),o)[0]


def test_missing_intrinsics_fails_closed():
    h,o=harness(); h._latest_camera_info=None
    assert not h._validate_trajectory_fov(trajectory([point(0,0),point(.3,0)]),o)[0]


def test_fk_failure_cannot_return_valid_trajectory():
    h,o=harness()
    def fail(*args,**kw): raise ValueError('FK failed')
    h._fk_pose=fail
    with pytest.raises(ValueError,match='FK failed'): h._validate_trajectory_fov(trajectory([point(0,0),point(.3,0)]),o)


def test_spline_overshoot_differs_from_linear_and_is_checked():
    first,second=point(0,0,8,0),point(1,0,-8,0)
    assert interpolate_positions(first,second,.5,1)[0]>1
    h,o=harness()
    assert not h._validate_trajectory_fov(trajectory([first,second]),o)[0]
