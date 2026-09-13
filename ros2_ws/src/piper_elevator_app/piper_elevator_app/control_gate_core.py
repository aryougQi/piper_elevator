"""Pure state policy and tracking checks for the real-arm command gate."""


def joint_positions_within_tolerance(target, feedback, names, tolerance):
    """Return whether every named feedback joint has reached its target."""
    tolerance = float(tolerance)
    if tolerance <= 0.0 or not names:
        raise ValueError('joint tracking tolerance and names must be valid')
    if any(name not in target or name not in feedback for name in names):
        return False
    return all(
        abs(float(target[name]) - float(feedback[name])) <= tolerance
        for name in names
    )


def desired_control_mode(policy, now):
    """Return the one mutually exclusive hardware command mode to enable."""
    desired_open, timed_out_now = policy.desired_open(now)
    if not desired_open:
        return None, timed_out_now
    if policy.servo_authorized:
        return 'servo', timed_out_now
    return 'trajectory', timed_out_now


class ControlGatePolicy:
    """Keep commands enabled until an action terminal state or hard timeout."""

    def __init__(
        self,
        maximum_trajectory_seconds=45.0,
        servo_heartbeat_seconds=0.75,
        trajectory_settle_seconds=25.0,
    ):
        """Initialize independent trajectory and Servo safety deadlines."""
        self.maximum_trajectory_seconds = self._positive(
            maximum_trajectory_seconds,
            'maximum_trajectory_seconds',
        )
        self.servo_heartbeat_seconds = self._positive(
            servo_heartbeat_seconds,
            'servo_heartbeat_seconds',
        )
        self.trajectory_settle_seconds = self._positive(
            trajectory_settle_seconds,
            'trajectory_settle_seconds',
        )
        self.trajectory_active = False
        self.trajectory_settling = False
        self.trajectory_settle_started_at = None
        self.trajectory_started_at = None
        self.trajectory_timed_out = False
        self.servo_authorized = False
        self.servo_last_heartbeat = None

    @staticmethod
    def _positive(value, name):
        value = float(value)
        if value <= 0.0:
            raise ValueError(f'{name} must be positive')
        return value

    def update_trajectory(self, active, now, needs_settling=False):
        """Record whether the action status array contains an active goal."""
        active = bool(active)
        now = float(now)
        if active and not self.trajectory_active:
            self.trajectory_started_at = now
            self.trajectory_timed_out = False
            self.trajectory_settling = False
            self.trajectory_settle_started_at = None
        elif not active and self.trajectory_active:
            self.trajectory_started_at = None
            self.trajectory_timed_out = False
            self.trajectory_settling = bool(needs_settling)
            self.trajectory_settle_started_at = (
                now if self.trajectory_settling else None
            )
        self.trajectory_active = active

    def complete_trajectory_settling(self):
        """Close the hold after real feedback reaches the target."""
        self.trajectory_settling = False
        self.trajectory_settle_started_at = None

    def update_servo(self, authorized, now):
        """Record or clear a live Servo authorization heartbeat."""
        self.servo_authorized = bool(authorized)
        self.servo_last_heartbeat = (
            float(now) if self.servo_authorized else None
        )

    def desired_open(self, now):
        """Return desired gate state and whether hard timeout just fired."""
        now = float(now)
        timed_out_now = False
        if (
            self.trajectory_active
            and not self.trajectory_timed_out
            and self.trajectory_started_at is not None
            and now - self.trajectory_started_at
            > self.maximum_trajectory_seconds
        ):
            self.trajectory_timed_out = True
            timed_out_now = True
        if (
            self.trajectory_settling
            and not self.trajectory_timed_out
            and self.trajectory_settle_started_at is not None
            and now - self.trajectory_settle_started_at
            > self.trajectory_settle_seconds
        ):
            self.trajectory_timed_out = True
            self.trajectory_settling = False
            self.trajectory_settle_started_at = None
            timed_out_now = True
        if (
            self.servo_last_heartbeat is None
            or now - self.servo_last_heartbeat
            > self.servo_heartbeat_seconds
        ):
            self.servo_authorized = False
            self.servo_last_heartbeat = None
        if self.trajectory_timed_out:
            return False, timed_out_now
        return (
            self.trajectory_active
            or self.trajectory_settling
            or self.servo_authorized,
            timed_out_now,
        )
