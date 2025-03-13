from __future__ import annotations
import json
import urllib.request
from setuptools import find_packages, setup


def read_file(fname):
    with open(fname, encoding="utf-8") as fd:
        return fd.read()


def get_requirements(fname):
    with open(fname, encoding="utf-8") as fd:
        reqs = [line.strip() for line in fd if line]
    return reqs

NAME = "leaflux"
DESCRIPTION = "Core algorithms for the Leaflux project"
LONG_DESCRIPTION = read_file("README.md")
VERSION = "0.1.3"
LICENSE = "MIT"
URL = "https://github.com/silvxlabs/leaflux-core"
INSTALL_REQUIRES = [
    get_requirements("requirements.txt")
]

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    license=LICENSE,
    url=URL,
    # project_urls=PROJECT_URLS,
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
    ],
    package_dir={"": "."},
    packages=find_packages(exclude=["docs", "tests", "data"]),
    include_package_data=True,
    install_requires=INSTALL_REQUIRES,
    python_requires=">=3.9",
)