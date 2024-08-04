
from setuptools import setup

setup(
    name='bayesfilter',
    version='0.0.1',
    packages=['bayesfilter'],
    install_requires=[
        'numpy',
    ],
    license='MIT',
    author='Hugo Hadfield',
    long_description=open('README.md').read(),
)