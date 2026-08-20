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
                'button_detector = '
                'piper_elevator_app.yolo_button_detector:main'
            ),
        ],
    },
)
