
from setuptools import setup
import os

here = os.path.abspath(os.path.dirname(__file__))
exec(open(os.path.join(here, 'bayesfilter/version.py')).read())


setup(
    name='bayesfilter',
    version=__version__,
    packages=['bayesfilter'],
    install_requires=[
        'numpy',
    ],
    license='MIT',
    author='Hugo Hadfield',
    author_email="hadfield.hugo@gmail.com",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    description="A pure Python/NumPy library for Bayesian filtering and smoothing",
    url="https://github.com/hugohadfield/bayesfilter",
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Mathematics',
    ],
)