#!/usr/bin/env python3
"""
Setup script for autobidsify (flat layout).
"""

from setuptools import setup, find_packages
from pathlib import Path


def get_version():
    """Extract version from __init__.py."""
    init_file = Path(__file__).parent / "autobidsify" / "__init__.py"
    if init_file.exists():
        for line in init_file.read_text().splitlines():
            if line.startswith("__version__"):
                return line.split("=")[1].strip().strip('"').strip("'")
    return "0.9.5"


def get_long_description():
    """Read README.md."""
    readme_file = Path(__file__).parent / "README.md"
    if readme_file.exists():
        return readme_file.read_text(encoding="utf-8")
    return ""


setup(
    name="autobidsify",
    version=get_version(),
    author="Yiyi Liu",
    author_email="yiyi.liu3@northeastern.edu",
    description="Automated BIDS standardization tool powered by LLM-first architecture",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/cotilab/autobidsify",
    project_urls={
        "Documentation": "https://autobidsify.readthedocs.io",
        "Repository": "https://github.com/cotilab/autobidsify",
        "Issues": "https://github.com/cotilab/autobidsify/issues",
    },
    license="MIT",
    
    # Flat layout - no package_dir needed
    packages=find_packages(exclude=["tests", "tests.*", "docs"]),
    
    python_requires=">=3.8",
    
    install_requires=[
        "openai>=1.0.0",
        "pyyaml>=6.0",
        "pdfplumber>=0.8.0",
        "PyPDF2>=3.0.0",
        "python-docx>=1.0.0",
        "pydicom>=2.4.0",
        "nibabel>=2.0.0",
        "h5py>=3.8.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "openpyxl>=3.1.0",
        "dcm2niix>=1.0.20220720",
        "bids-validator>=1.14.0",
        "ollama>=0.6.0",
        "snirf",
    ],
    
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=23.0",
            "ruff>=0.1.0",
            "mypy>=1.0",
        ],
        "qwen": [
        "ollama>=0.6.0",
        "dashscope>=1.0.0",
        ],
        "jnifti": [
        "bjdata>=0.4.0",
        ],
    },
    
    entry_points={
        "console_scripts": [
            "autobidsify=autobidsify.__main__:main",
            "bidsify=autobidsify.__main__:main",
        ],
    },
    
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    
    include_package_data=True,
    zip_safe=False,
)

