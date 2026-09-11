from glob import glob
from setuptools import find_packages, setup


package_name = 'piper_elevator_app'


setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name],
        ),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/config', glob('config/*.xacro')),
        ('share/' + package_name + '/config', glob('config/*.srdf')),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/models', glob('models/*.onnx')),
        ('share/' + package_name + '/models', glob('models/*.md')),
    ],
    install_requires=['setuptools'],
    tests_require=['pytest'],
    zip_safe=True,
    maintainer='qi',
    maintainer_email='qi@example.com',
    description=(
        'YOLO-based perception nodes for the Piper elevator application.'
    ),
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            (
                'pika_fisheye_camera = '
                'piper_elevator_app.pika_fisheye_camera:main'
            ),
            (
                'button_detector = '
                'piper_elevator_app.yolo_button_detector:main'
            ),
            (
                'button_approach_planner = '
                'piper_elevator_app.button_approach_planner:main'
            ),
            (
                'button_visual_servo = '
                'piper_elevator_app.button_visual_servo:main'
            ),
            (
                'button_press_executor = '
                'piper_elevator_app.button_press_executor:main'
            ),
            (
                'elevator_task_manager = '
                'piper_elevator_app.elevator_task_manager:main'
            ),
            (
                'mock_button_pose = '
                'piper_elevator_app.mock_button_pose:main'
            ),
            (
                'piper_pika_joint_state_mux = '
                'piper_elevator_app.joint_state_mux:main'
            ),
            (
                'piper_pika_control_gate = '
                'piper_elevator_app.control_gate:main'
            ),
        ],
    },
)
