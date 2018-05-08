"""big_g setup file"""
from setuptools import setup
from object_tracking import object_tracking

setup(
    name='big_g',
    description='OpenCV object tracking for Big G experiments',
    url='https://github.com/paul-freeman/big_g.git',
    author='Paul Freeman',
    license='MIT License',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
        'Topic :: Education',
        'Topic :: Multimedia :: Video',
        'Topic :: Scientific/Engineering :: Physics'],
    keywords='opencv physics tracking',
    project_urls={
        'Source': 'https://github.com/paul-freeman/big_g.git'},
    packages=['object_tracking'],
    install_requires=[
        'numpy',
        'matplotlib',
        'opencv-contrib-python'],
    entry_points={'console_scripts': [
        'big_g_tracking = object_tracking.object_tracking:tracking']},
    version=object_tracking.__version__,
    author_email='paul.freeman.cs@gmail.com'
    )
