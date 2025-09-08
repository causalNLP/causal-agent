import os
import re
import subprocess
import sys
from pathlib import Path

from setuptools import setup, find_packages
from setuptools import Command


def read_file(path: str) -> str:
    here = Path(__file__).parent
    try:
        return (here / path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def read_requirements(filename: str = "requirements.txt"):
    content = read_file(filename)
    lines = [l.strip() for l in content.splitlines()]
    return [l for l in lines if l and not l.startswith("#")] if content else []


def read_version():
    init_py = read_file("causal_agent/__init__.py")
    m = re.search(r'__version__\s*=\s*"([^"]+)"', init_py)
    return m.group(1) if m else "0.0.0"


class VenvCommand(Command):
    description = "Create a local .venv and install requirements (optional helper)."
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        venv_dir = Path(".venv")
        python = sys.executable
        if not venv_dir.exists():
            subprocess.check_call([python, "-m", "venv", str(venv_dir)])
        pip_bin = venv_dir / ("Scripts/pip.exe" if os.name == "nt" else "bin/pip")
        # Upgrade pip and install requirements if present
        subprocess.check_call([str(pip_bin), "install", "--upgrade", "pip"]) 
        reqs = Path("requirements.txt")
        if reqs.exists():
            subprocess.check_call([str(pip_bin), "install", "-r", str(reqs)])
        # Editable install of this package
        subprocess.check_call([str(pip_bin), "install", "-e", "."]) 
        print("Virtual environment setup complete at .venv")


setup(
    name="causal-agent",
    version=read_version(),
    author="Vishal Verma",
    author_email="vishal.verma@andrew.cmu.edu",
    description="A library for automated causal inference",
    long_description=read_file("README_PYPI.md") or "A library for automated causal inference",
    long_description_content_type="text/markdown",
    url="https://github.com/causalNLP/causal-agent",
    packages=find_packages(exclude=["tests", "tests.*", "data_generation*", "replication_codes*"]),
    include_package_data=True,
    zip_safe=False,
    python_requires=">=3.10",
    install_requires=read_requirements(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    entry_points={
        "console_scripts": [
            "causal-agent=causal_agent.cli:main",
        ]
    },
    cmdclass={
        "venv": VenvCommand,
    },
)