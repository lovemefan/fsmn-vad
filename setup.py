# -*- coding:utf-8 -*-
# @FileName  :setup.py
# @Time      :2023/4/4 11:22
# @Author    :lovemefan
# @Email     :lovemefan@outlook.com
import os
from pathlib import Path

from setuptools import find_packages, setup

dirname = Path(os.path.dirname(__file__))
version_file = dirname / "version.txt"
with open(version_file, "r") as f:
    version = f.read().strip()

requirements = {
    "install": [
        "setuptools<=60.0",
        "scipy==1.5.0",
        "PyYAML>=5.3",
        "onnxruntime==1.14.1",
    ],
    "setup": [
        "numpy<=1.24.2",
        "pytest-runner",
    ],
    "doc": [
        "Sphinx==2.1.2",
        "sphinx-rtd-theme>=0.2.4",
        "sphinx-argparse>=0.2.5",
        "commonmark==0.8.1",
        "recommonmark>=0.4.0",
        "nbsphinx>=0.4.2",
        "sphinx-markdown-tables>=0.0.12",
    ],
    "all": [],
}
requirements["all"].extend(requirements["install"] + requirements["doc"])

install_requires = requirements["install"]
setup_requires = requirements["setup"]

extras_require = {
    k: v for k, v in requirements.items() if k not in ["install", "setup"]
}


setup(
    name="fsmnvad",
    version=version,
    url="https://github.com/lovemefan/fsmn-vad",
    author="Lovemefan, Yunnan Key Laboratory of Artificial Intelligence, "
    "Kunming University of Science and Technology, Kunming, Yunnan ",
    author_email="lovemefan@outlook.com",
    description="Fsmn-vad: A enterprise-grade Voice Activity Detector (VAD) based on FSMN from modelscope opensource",
    long_description=open(os.path.join(dirname, "README.md"), encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    license="The MIT License",
    packages=find_packages(include=["funasr*"]),
    package_data={"funasr": ["version.txt"]},
    install_requires=install_requires,
    setup_requires=setup_requires,
    extras_require=extras_require,
    python_requires=">=3.7.0",
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Operating System :: POSIX :: Linux",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
