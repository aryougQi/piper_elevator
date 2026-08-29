"""Pure contact-detection helpers for the button press controller."""

from dataclasses import dataclass

import numpy as np


def simulated_button_depression(rest_position, current_position):
    """Return inward button travel for a joint whose pressed limit is lower."""
    rest = float(rest_position)
    current = float(current_position)
    if not np.isfinite(rest) or not np.isfinite(current):
        raise ValueError('button joint positions must be finite')
    return max(0.0, rest - current)


@dataclass(frozen=True)
class ContactDetection:
    """One filtered joint-effort observation."""

    contact: bool
    emergency: bool
    normalized_peak: float
    residual: np.ndarray
    reason: str = ''


class JointEffortContactDetector:
    """Detect contact from robust joint-torque increments, not raw torque."""

    def __init__(
        self,
        joint_names,
        delta_thresholds,
        absolute_limits,
        baseline_sample_count=25,
        consecutive_samples=4,
        smoothing_alpha=0.30,
        minimum_joint_count=1,
        emergency_multiplier=2.5,
    ):
        self.joint_names = tuple(str(name) for name in joint_names)
        self.delta_thresholds = self._positive_vector(
            delta_thresholds,
            'delta_thresholds',
        )
        self.absolute_limits = self._positive_vector(
            absolute_limits,
            'absolute_limits',
        )
        if len(self.joint_names) != self.delta_thresholds.size:
            raise ValueError('joint_names and thresholds must have equal size')
        if self.absolute_limits.size != self.delta_thresholds.size:
            raise ValueError('absolute_limits and thresholds must match')
        self.baseline_sample_count = max(3, int(baseline_sample_count))
        self.consecutive_samples = max(1, int(consecutive_samples))
        self.smoothing_alpha = float(smoothing_alpha)
        if not 0.0 < self.smoothing_alpha <= 1.0:
            raise ValueError('smoothing_alpha must be in (0, 1]')
        self.minimum_joint_count = max(1, int(minimum_joint_count))
        if self.minimum_joint_count > len(self.joint_names):
            raise ValueError('minimum_joint_count exceeds joint count')
        self.emergency_multiplier = float(emergency_multiplier)
        if self.emergency_multiplier <= 1.0:
            raise ValueError('emergency_multiplier must be greater than one')
        self.reset()

    @staticmethod
    def _positive_vector(values, label):
        vector = np.asarray(values, dtype=np.float64)
        if vector.ndim != 1 or vector.size == 0:
            raise ValueError(f'{label} must be a non-empty vector')
        if not np.all(np.isfinite(vector)) or np.any(vector <= 0.0):
            raise ValueError(f'{label} must contain finite positive values')
        return vector

    def reset(self):
        self._baseline_samples = []
        self.baseline = None
        self.filtered_residual = np.zeros(len(self.joint_names))
        self._contact_run = 0

    @property
    def baseline_ready(self):
        return self.baseline is not None

    @property
    def baseline_progress(self):
        return len(self._baseline_samples)

    def ordered_efforts(self, names, efforts):
        """Return configured efforts in a stable order, or ``None``."""
        if len(names) != len(efforts):
            return None
        effort_by_name = dict(zip(names, efforts))
        try:
            ordered = np.asarray(
                [effort_by_name[name] for name in self.joint_names],
                dtype=np.float64,
            )
        except (KeyError, TypeError, ValueError):
            return None
        if not np.all(np.isfinite(ordered)):
            return None
        return ordered

    def add_baseline_sample(self, names, efforts):
        """Collect a stationary sample and form a median baseline."""
        ordered = self.ordered_efforts(names, efforts)
        if ordered is None:
            return False
        if np.any(np.abs(ordered) >= self.absolute_limits):
            raise RuntimeError('absolute joint torque limit during baseline')
        if self.baseline is not None:
            return True
        self._baseline_samples.append(ordered)
        if len(self._baseline_samples) >= self.baseline_sample_count:
            self.baseline = np.median(
                np.asarray(self._baseline_samples),
                axis=0,
            )
            self.filtered_residual = np.zeros_like(self.baseline)
        return self.baseline_ready

    def update(self, names, efforts):
        """Filter a new sample and update the consecutive contact latch."""
        if self.baseline is None:
            raise RuntimeError('torque baseline is not ready')
        ordered = self.ordered_efforts(names, efforts)
        if ordered is None:
            raise ValueError('joint effort sample is incomplete or non-finite')
        raw_residual = ordered - self.baseline
        alpha = self.smoothing_alpha
        self.filtered_residual = (
            alpha * raw_residual
            + (1.0 - alpha) * self.filtered_residual
        )
        normalized = np.abs(self.filtered_residual) / self.delta_thresholds
        over_count = int(np.count_nonzero(normalized >= 1.0))
        if over_count >= self.minimum_joint_count:
            self._contact_run += 1
        else:
            self._contact_run = 0
        absolute_trip = bool(
            np.any(np.abs(ordered) >= self.absolute_limits)
        )
        residual_trip = bool(
            np.any(normalized >= self.emergency_multiplier)
        )
        emergency = absolute_trip or residual_trip
        reason = ''
        if absolute_trip:
            reason = 'absolute joint torque limit'
        elif residual_trip:
            reason = 'joint torque increment emergency limit'
        return ContactDetection(
            contact=self._contact_run >= self.consecutive_samples,
            emergency=emergency,
            normalized_peak=float(np.max(normalized)),
            residual=self.filtered_residual.copy(),
            reason=reason,
        )
