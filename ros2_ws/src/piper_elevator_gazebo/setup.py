from pathlib import Path

from setuptools import find_packages, setup


PACKAGE_NAME = 'piper_elevator_gazebo'


def asset_data_files(root_name):
    grouped = {}
    for path in Path(root_name).rglob('*'):
        if path.is_file():
            destination = Path('share') / PACKAGE_NAME / path.parent
            grouped.setdefault(str(destination), []).append(str(path))
    return sorted(grouped.items())


data_files = [
    (
        'share/ament_index/resource_index/packages',
        ['resource/' + PACKAGE_NAME],
    ),
    ('share/' + PACKAGE_NAME, ['package.xml']),
]
for asset_root in ('config', 'launch', 'models', 'urdf', 'worlds'):
    data_files.extend(asset_data_files(asset_root))


setup(
    name=PACKAGE_NAME,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=data_files,
    install_requires=['setuptools'],
    tests_require=['pytest'],
    zip_safe=True,
    maintainer='qi',
    maintainer_email='qi@example.com',
    description=(
        'Gazebo Fortress virtual hardware for the Piper elevator application.'
    ),
    license='Apache-2.0',
)
