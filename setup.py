"""big_g setup file"""
from setuptools import setup
from object_tracking import object_tracking

setup(
    name='big_g',
    version=object_tracking.__version__,
    author='Paul Freeman',
    author_email='paul.freeman.cs@gmail.com',
    license='MIT License',
    packages=['object_tracking'],
    install_requires=[
        'numpy',
        'matplotlib',
        'opencv-contrib-python'],
    entry_points={'console_scripts': [
        'big_g_tracking = object_tracking.object_tracking:tracking']}
    )
