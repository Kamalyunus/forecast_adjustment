"""
Setup script for the forecast adjustment package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="forecast-adjustment",
    version="0.1.0",
    author="Forecast Adjustment Team",
    author_email="example@example.com",
    description="A system for adjusting forecasts using reinforcement learning with support for calendar effects",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kamalyunus/forecast-adjustment",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "tqdm>=4.61.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.5",
            "black>=22.1.0",
            "flake8>=4.0.1",
            "mypy>=0.910",
        ],
    },
    entry_points={
        "console_scripts": [
            "forecast-adjust=forecast_adjustment.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "forecast_adjustment": ["py.typed"],
    },
)