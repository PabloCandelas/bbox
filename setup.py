from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'bbox'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),        
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch')),        
        (os.path.join('share', package_name, 'config'), glob('config/*.config.yaml')),
        (os.path.join('share', package_name, 'lib'), glob('*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='pablo',
    maintainer_email='pcandelas98@gmail.com',
    description='TODO: Package description',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'dummy = bbox.dummy:main',
            'video = bbox.video:main',
            'init_joy = bbox.init_joy:main',
            'webcam_publisher = bbox.webcam_publisher:main',
            'hsv_calibrator = bbox.hsv_calibrator:main',
            'aruco_pool_node = bbox.aruco_pool_node:main',
            'aruco_localization = bbox.aruco_localization:main',
            'bbox_yolo_detection = bbox.bbox_yolo_detection:main',
        ],
    },
)
