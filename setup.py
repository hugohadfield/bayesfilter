from pathlib import Path

from setuptools import setup


here = Path(__file__).parent.resolve()
version = {}
exec((here / "bayesfilter" / "version.py").read_text(encoding="utf-8"), version)


setup(
    name="bayesfilter",
    version=version["__version__"],
    packages=["bayesfilter"],
    install_requires=[
        "numpy>=1.21.3",
    ],
    extras_require={
        "test": [
            "pytest>=7",
            "pytest-cov>=4",
        ],
    },
    license="MIT",
    license_files=["LICENSE"],
    author="Hugo Hadfield",
    author_email="hadfield.hugo@gmail.com",
    long_description=(here / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    description="A pure Python/NumPy library for Bayesian filtering and smoothing",
    url="https://github.com/hugohadfield/bayesfilter",
    project_urls={
        "Changelog": "https://github.com/hugohadfield/bayesfilter/blob/main/CHANGELOG.md",
        "Issues": "https://github.com/hugohadfield/bayesfilter/issues",
        "Source": "https://github.com/hugohadfield/bayesfilter",
    },
    python_requires=">=3.10",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
)
