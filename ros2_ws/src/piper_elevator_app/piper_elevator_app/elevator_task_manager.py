"""High-level state machine for one complete elevator button press."""

import threading
import time

from geometry_msgs.msg import PoseStamped
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from std_msgs.msg import Bool
from std_msgs.msg import String
from std_srvs.srv import Trigger

from piper_elevator_app.task_core import parse_task_command


class TaskFailure(RuntimeError):
    """A user-facing state-machine failure."""


class ElevatorTaskManager(Node):
    """Coordinate perception, approach, visual servo, press, and homing."""

    def __init__(self):
        super().__init__('elevator_task_manager')
        self._declare_parameters()
        self._callback_group = ReentrantCallbackGroup()
        self._condition = threading.Condition()
        self._stop_event = threading.Event()
        self._busy = False
        self._task_sequence = 0
        self._active_button = ''

        self._selected_button = ''
        self._selected_sequence = 0
        self._detection_valid = False
        self._detection_sequence = 0
        self._surface_sequence = 0
        self._visual_completed = False
        self._visual_completion_sequence = 0
        self._visual_status = ''
        self._visual_status_sequence = 0
        self._press_completed = False
        self._press_completion_sequence = 0
        self._press_status = ''
        self._press_status_sequence = 0

        latched_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._status_publisher = self.create_publisher(
            String,
            self._string_parameter('status_topic'),
            latched_qos,
        )
        self._result_publisher = self.create_publisher(
            String,
            self._string_parameter('result_topic'),
            latched_qos,
        )
        self._completion_publisher = self.create_publisher(
            Bool,
            self._string_parameter('completion_topic'),
            latched_qos,
        )
        self._active_button_publisher = self.create_publisher(
            String,
            self._string_parameter('active_button_topic'),
            latched_qos,
        )
        self._selection_publisher = self.create_publisher(
            String,
            self._string_parameter('button_selection_topic'),
            latched_qos,
        )

        self.create_subscription(
            String,
            self._string_parameter('command_topic'),
            self._command_callback,
            10,
            callback_group=self._callback_group,
        )
        self.create_subscription(
            String,
            self._string_parameter('button_selected_topic'),
            self._selected_callback,
            latched_qos,
            callback_group=self._callback_group,
        )
        self.create_subscription(
            Bool,
            self._string_parameter('button_valid_topic'),
            self._detection_callback,
            10,
            callback_group=self._callback_group,
        )
        self.create_subscription(
            PoseStamped,
            self._string_parameter('button_surface_topic'),
            self._surface_callback,
            10,
            callback_group=self._callback_group,
        )
        self.create_subscription(
            Bool,
            self._string_parameter('visual_completion_topic'),
            self._visual_completion_callback,
            latched_qos,
            callback_group=self._callback_group,
        )
        self.create_subscription(
            String,
            self._string_parameter('visual_status_topic'),
            self._visual_status_callback,
            latched_qos,
            callback_group=self._callback_group,
        )
        self.create_subscription(
            Bool,
            self._string_parameter('press_completion_topic'),
            self._press_completion_callback,
            latched_qos,
            callback_group=self._callback_group,
        )
        self.create_subscription(
            String,
            self._string_parameter('press_status_topic'),
            self._press_status_callback,
            latched_qos,
            callback_group=self._callback_group,
        )

        service_names = {
            'plan': 'plan_service',
            'execute': 'execute_service',
            'home': 'home_service',
            'clear_plan': 'clear_plan_service',
            'visual_start': 'visual_start_service',
            'visual_stop': 'visual_stop_service',
            'press_start': 'press_start_service',
            'press_stop': 'press_stop_service',
        }
        # Do not shadow rclpy.node.Node._clients, which is an internal list
        # consumed by the executor and destroy_node().
        self._trigger_clients = {
            key: self.create_client(
                Trigger,
                self._string_parameter(parameter),
                callback_group=self._callback_group,
            )
            for key, parameter in service_names.items()
        }
        self.create_service(
            Trigger,
            '~/stop',
            self._stop_callback,
            callback_group=self._callback_group,
        )
        self.create_service(
            Trigger,
            '~/reset',
            self._reset_callback,
            callback_group=self._callback_group,
        )

        self._publish_completion(False)
        self._publish_active_button('')
        self._publish_result('No task has run')
        self._publish_status('IDLE: publish `press <button>` to command topic')

    def _declare_parameters(self):
        self.declare_parameter('command_topic', '/elevator_task/command')
        self.declare_parameter('status_topic', '/elevator_task/status')
        self.declare_parameter('result_topic', '/elevator_task/result')
        self.declare_parameter('completion_topic', '/elevator_task/completed')
        self.declare_parameter(
            'active_button_topic', '/elevator_task/active_button'
        )
        self.declare_parameter('button_selection_topic', '/button_selection')
        self.declare_parameter('button_selected_topic', '/button_selected')
        self.declare_parameter(
            'button_valid_topic', '/button_detection_valid'
        )
        self.declare_parameter('button_surface_topic', '/button_surface_pose')
        self.declare_parameter(
            'visual_completion_topic', '/button_visual_servo/completed'
        )
        self.declare_parameter(
            'visual_status_topic', '/button_visual_servo/status'
        )
        self.declare_parameter(
            'press_completion_topic', '/button_press/completed'
        )
        self.declare_parameter('press_status_topic', '/button_press/status')

        self.declare_parameter(
            'plan_service', '/button_approach_planner/plan'
        )
        self.declare_parameter(
            'execute_service', '/button_approach_planner/execute'
        )
        self.declare_parameter(
            'home_service', '/button_approach_planner/return_home'
        )
        self.declare_parameter(
            'clear_plan_service', '/button_approach_planner/clear_plan'
        )
        self.declare_parameter(
            'visual_start_service', '/button_visual_servo/start'
        )
        self.declare_parameter(
            'visual_stop_service', '/button_visual_servo/stop'
        )
        self.declare_parameter(
            'press_start_service', '/button_press_executor/start'
        )
        self.declare_parameter(
            'press_stop_service', '/button_press_executor/stop'
        )

        self.declare_parameter('return_home_before_task', True)
        self.declare_parameter('return_home_after_failure', True)
        self.declare_parameter('clear_selection_after_task', True)
        self.declare_parameter(
            'required_unique_nodes',
            [
                '/button_detector',
                '/button_approach_planner',
                '/button_visual_servo',
                '/button_press_executor',
            ],
        )
        self.declare_parameter('service_wait_timeout_seconds', 60.0)
        self.declare_parameter('target_wait_timeout_seconds', 30.0)
        self.declare_parameter('planning_timeout_seconds', 45.0)
        self.declare_parameter('execution_timeout_seconds', 60.0)
        self.declare_parameter(
            'post_motion_target_wait_timeout_seconds', 10.0
        )
        self.declare_parameter('visual_timeout_seconds', 120.0)
        self.declare_parameter('press_timeout_seconds', 120.0)
        self.declare_parameter('home_timeout_seconds', 60.0)
        self.declare_parameter('selection_publish_period_seconds', 0.5)

    def _string_parameter(self, name):
        return str(self.get_parameter(name).value)

    def _command_callback(self, message):
        try:
            command = parse_task_command(message.data)
        except ValueError as error:
            self._publish_status(f'REJECTED: {error}')
            self._publish_result(f'REJECTED: {error}')
            return
        if command.action == 'stop':
            self._request_stop()
            return
        with self._condition:
            if self._busy:
                self._publish_status(
                    'REJECTED: another task is running '
                    f'for button {self._active_button}'
                )
                return
            self._busy = True
            self._task_sequence += 1
            task_sequence = self._task_sequence
            self._active_button = command.button
            self._stop_event.clear()
        self._publish_completion(False)
        self._publish_active_button(command.button)
        self._publish_result(f'RUNNING: button={command.button}')
        threading.Thread(
            target=self._run_task,
            args=(task_sequence, command.button),
            daemon=True,
        ).start()

    def _stop_callback(self, request, response):
        del request
        response.success = self._request_stop()
        response.message = (
            'Stop requested; the state machine will recover to home'
            if response.success
            else 'No task is running'
        )
        return response

    def _reset_callback(self, request, response):
        del request
        with self._condition:
            if self._busy:
                response.success = False
                response.message = 'Cannot reset while a task is running'
                return response
        self._selection_publisher.publish(String(data=''))
        self._publish_completion(False)
        self._publish_active_button('')
        self._publish_result('No task has run')
        self._publish_status('IDLE: state reset')
        response.success = True
        response.message = 'Task state reset'
        return response

    def _request_stop(self):
        with self._condition:
            if not self._busy:
                return False
            self._stop_event.set()
            self._condition.notify_all()
        self._publish_status('STOP_REQUESTED: stopping active motion')
        for key in ('press_stop', 'visual_stop'):
            client = self._trigger_clients[key]
            if client.service_is_ready():
                client.call_async(Trigger.Request())
        return True

    def _run_task(self, task_sequence, button):
        at_home = False
        success = False
        failure_message = ''
        try:
            self._phase('WAITING_FOR_NODES', button)
            self._wait_for_required_services()
            self._ensure_unique_nodes()
            if bool(self.get_parameter('return_home_before_task').value):
                self._phase('HOMING_INITIAL', button)
                self._call_trigger(
                    'home',
                    self._seconds('home_timeout_seconds'),
                )
                at_home = True

            self._phase('SELECTING_BUTTON', button)
            self._select_and_wait_for_target(button)
            self._check_stopped()

            self._phase('COARSE_PLANNING', button)
            self._call_trigger(
                'plan',
                self._seconds('planning_timeout_seconds'),
            )
            self._phase('COARSE_EXECUTING', button)
            self._call_trigger(
                'execute',
                self._seconds('execution_timeout_seconds'),
            )
            at_home = False

            self._phase('WAITING_FOR_VISUAL_TARGET', button)
            self._wait_for_post_motion_target(button)
            self._phase('VISUAL_SERVO', button)
            self._start_and_wait_for_completion(
                'visual',
                self._seconds('visual_timeout_seconds'),
            )

            self._phase('PRESSING', button)
            self._start_and_wait_for_completion(
                'press',
                self._seconds('press_timeout_seconds'),
            )

            self._phase('HOMING_FINAL', button)
            self._call_trigger(
                'home',
                self._seconds('home_timeout_seconds'),
            )
            at_home = True
            success = True
        except TaskFailure as error:
            failure_message = str(error)
            self._publish_status(
                f'RECOVERING: button={button} reason={failure_message}'
            )
            at_home = self._recover(at_home)
        except Exception as error:
            failure_message = f'unexpected error: {error}'
            self.get_logger().error(failure_message)
            self._publish_status(
                f'RECOVERING: button={button} reason={failure_message}'
            )
            at_home = self._recover(at_home)
        finally:
            if bool(self.get_parameter('clear_selection_after_task').value):
                self._selection_publisher.publish(String(data=''))
            with self._condition:
                if task_sequence == self._task_sequence:
                    self._busy = False
                    self._active_button = ''
                self._stop_event.clear()
                self._condition.notify_all()
            self._publish_active_button('')
            self._publish_completion(success and at_home)
            if success and at_home:
                result = f'COMPLETE: button={button} pressed; home reached'
                self._publish_result(result)
                self._publish_status(result)
            else:
                result = (
                    f'FAILED: button={button}; {failure_message}; '
                    f'home_reached={str(at_home).lower()}'
                )
                self._publish_result(result)
                self._publish_status(result)

    def _wait_for_required_services(self):
        deadline = time.monotonic() + self._seconds(
            'service_wait_timeout_seconds'
        )
        for key in ('plan', 'execute', 'home', 'visual_start', 'press_start'):
            client = self._trigger_clients[key]
            while not client.wait_for_service(timeout_sec=0.2):
                self._check_stopped()
                if time.monotonic() >= deadline:
                    raise TaskFailure(f'required service unavailable: {key}')

    def _ensure_unique_nodes(self):
        names = []
        for name, namespace in self.get_node_names_and_namespaces():
            prefix = namespace.rstrip('/')
            names.append(f'{prefix}/{name}' if prefix else f'/{name}')
        duplicates = []
        for required in self.get_parameter('required_unique_nodes').value:
            count = names.count(str(required))
            if count != 1:
                duplicates.append(f'{required} count={count}')
        if duplicates:
            raise TaskFailure(
                'dependency node uniqueness check failed: '
                + ', '.join(duplicates)
            )

    def _select_and_wait_for_target(self, button):
        with self._condition:
            detection_baseline = self._detection_sequence
            surface_baseline = self._surface_sequence
        deadline = time.monotonic() + self._seconds(
            'target_wait_timeout_seconds'
        )
        publish_period = self._seconds(
            'selection_publish_period_seconds'
        )
        next_publish = 0.0
        while time.monotonic() < deadline:
            self._check_stopped()
            now = time.monotonic()
            if now >= next_publish:
                self._selection_publisher.publish(String(data=button))
                next_publish = now + publish_period
            with self._condition:
                ready = (
                    self._selected_button == button
                    and self._detection_valid
                    and self._detection_sequence > detection_baseline
                    and self._surface_sequence > surface_baseline
                )
                if ready:
                    # Let the planner consume the same surface-pose sample.
                    self._condition.wait(timeout=0.10)
                    return
                self._condition.wait(timeout=0.05)
        raise TaskFailure(
            f'no stable RGB-D target for selected button `{button}`'
        )

    def _wait_for_post_motion_target(self, button):
        """Wait for a stationary RGB-D observation after coarse motion.

        The last valid pose seen while the arm was moving may already be
        stale when the execute service returns.  Starting visual servo in
        that small gap creates a race where its start service correctly
        rejects the old sample.  Require both a new detection status and a
        new surface pose before invoking the visual controller.
        """
        with self._condition:
            detection_baseline = self._detection_sequence
            surface_baseline = self._surface_sequence
        deadline = time.monotonic() + self._seconds(
            'post_motion_target_wait_timeout_seconds'
        )
        publish_period = self._seconds(
            'selection_publish_period_seconds'
        )
        next_publish = 0.0
        while time.monotonic() < deadline:
            self._check_stopped()
            now = time.monotonic()
            if now >= next_publish:
                self._selection_publisher.publish(String(data=button))
                next_publish = now + publish_period
            with self._condition:
                ready = (
                    self._selected_button == button
                    and self._detection_valid
                    and self._detection_sequence > detection_baseline
                    and self._surface_sequence > surface_baseline
                )
                if ready:
                    # Give the visual-servo subscription time to consume the
                    # same pose before its start service checks freshness.
                    self._condition.wait(timeout=0.10)
                    return
                self._condition.wait(timeout=0.05)
        raise TaskFailure(
            f'no fresh RGB-D target after coarse motion for `{button}`'
        )

    def _start_and_wait_for_completion(self, prefix, timeout):
        with self._condition:
            if prefix == 'visual':
                completion_baseline = self._visual_completion_sequence
                status_baseline = self._visual_status_sequence
            else:
                completion_baseline = self._press_completion_sequence
                status_baseline = self._press_status_sequence
        self._call_trigger(f'{prefix}_start', min(timeout, 10.0))
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self._check_stopped()
            with self._condition:
                if prefix == 'visual':
                    completed = (
                        self._visual_completed
                        and self._visual_completion_sequence
                        > completion_baseline
                    )
                    status = self._visual_status
                    status_sequence = self._visual_status_sequence
                else:
                    completed = (
                        self._press_completed
                        and self._press_completion_sequence
                        > completion_baseline
                    )
                    status = self._press_status
                    status_sequence = self._press_status_sequence
                if completed:
                    return
                if (
                    status_sequence > status_baseline
                    and status.startswith(('FAILED:', 'STOPPED', 'REJECTED:'))
                ):
                    raise TaskFailure(f'{prefix} failed: {status}')
                self._condition.wait(timeout=0.05)
        raise TaskFailure(f'{prefix} timed out after {timeout:.1f}s')

    def _call_trigger(self, key, timeout, ignore_stop=False):
        client = self._trigger_clients[key]
        if not client.wait_for_service(timeout_sec=min(timeout, 1.0)):
            raise TaskFailure(f'service unavailable: {key}')
        future = client.call_async(Trigger.Request())
        deadline = time.monotonic() + timeout
        while not future.done():
            if time.monotonic() >= deadline:
                future.cancel()
                raise TaskFailure(f'service timed out: {key}')
            with self._condition:
                self._condition.wait(timeout=0.05)
        if future.cancelled():
            raise TaskFailure(f'service cancelled: {key}')
        error = future.exception()
        if error is not None:
            raise TaskFailure(f'service error ({key}): {error}')
        response = future.result()
        if response is None or not response.success:
            message = 'no response' if response is None else response.message
            raise TaskFailure(f'{key} rejected: {message}')
        if not ignore_stop:
            self._check_stopped()
        return response.message

    def _recover(self, at_home):
        for key in ('press_stop', 'visual_stop'):
            try:
                self._call_trigger(key, 3.0, ignore_stop=True)
            except TaskFailure:
                pass
        try:
            self._call_trigger('clear_plan', 3.0, ignore_stop=True)
        except TaskFailure:
            pass
        if (
            at_home
            or not bool(
                self.get_parameter('return_home_after_failure').value
            )
        ):
            return at_home
        try:
            self._publish_status('RECOVERING_HOME')
            self._call_trigger(
                'home',
                self._seconds('home_timeout_seconds'),
                ignore_stop=True,
            )
            return True
        except TaskFailure as error:
            self.get_logger().error(f'Failed to recover home: {error}')
            return False

    def _check_stopped(self):
        if self._stop_event.is_set():
            raise TaskFailure('task stopped by user')

    def _seconds(self, name):
        return max(0.01, float(self.get_parameter(name).value))

    def _phase(self, name, button):
        self._publish_status(f'{name}: button={button}')

    def _selected_callback(self, message):
        with self._condition:
            self._selected_button = str(message.data)
            self._selected_sequence += 1
            self._condition.notify_all()

    def _detection_callback(self, message):
        with self._condition:
            self._detection_valid = bool(message.data)
            self._detection_sequence += 1
            self._condition.notify_all()

    def _surface_callback(self, message):
        del message
        with self._condition:
            self._surface_sequence += 1
            self._condition.notify_all()

    def _visual_completion_callback(self, message):
        with self._condition:
            self._visual_completed = bool(message.data)
            self._visual_completion_sequence += 1
            self._condition.notify_all()

    def _visual_status_callback(self, message):
        with self._condition:
            self._visual_status = str(message.data)
            self._visual_status_sequence += 1
            self._condition.notify_all()

    def _press_completion_callback(self, message):
        with self._condition:
            self._press_completed = bool(message.data)
            self._press_completion_sequence += 1
            self._condition.notify_all()

    def _press_status_callback(self, message):
        with self._condition:
            self._press_status = str(message.data)
            self._press_status_sequence += 1
            self._condition.notify_all()

    def _publish_status(self, text):
        self._status_publisher.publish(String(data=str(text)))

    def _publish_result(self, text):
        self._result_publisher.publish(String(data=str(text)))

    def _publish_completion(self, value):
        self._completion_publisher.publish(Bool(data=bool(value)))

    def _publish_active_button(self, button):
        self._active_button_publisher.publish(String(data=str(button)))


def main(args=None):
    rclpy.init(args=args)
    node = ElevatorTaskManager()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
