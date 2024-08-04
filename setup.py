
from setuptools import setup

setup(
    name='bayesfilter',
    version='0.0.2',
    packages=['bayesfilter'],
    install_requires=[
        'numpy',
    ],
    license='MIT',
    author='Hugo Hadfield',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
)