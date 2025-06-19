import os
import subprocess
import sys
from setuptools import setup, find_packages
from setuptools.command.install import install

# --- Conda Environment and Dependency Installation ---

def get_conda_path():
    """Tries to find the path to the conda executable."""
    # Common locations for conda executable
    # 1. In the system's PATH
    try:
        conda_path = subprocess.check_output("which conda", shell=True).strip().decode('utf-8')
        if conda_path: return conda_path
    except subprocess.CalledProcessError:
        pass # Not in PATH

    # 2. Common installation directories
    possible_paths = [
        os.path.expanduser("~/anaconda3/bin/conda"),
        os.path.expanduser("~/miniconda3/bin/conda"),
        "/opt/anaconda3/bin/conda",
        "/opt/miniconda3/bin/conda",
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None

def conda_env_exists(env_name):
    """Check if a conda environment with the given name already exists."""
    try:
        envs = subprocess.check_output("conda env list", shell=True).decode('utf-8')
        return any(line.startswith(env_name + ' ') for line in envs.splitlines())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

class CondaInstallCommand(install):
    """Custom command to create Conda env and install dependencies before package installation."""
    description = "Create Conda environment and install dependencies, then install the package."

    def run(self):
        env_name = "causal-agent"
        conda_path = get_conda_path()

        if conda_path:
            print(f"--- Found Conda at: {conda_path} ---")

            if not conda_env_exists(env_name):
                print(f"--- Creating Conda environment: {env_name} ---")
                try:
                    # Create the environment with a specific Python version
                    subprocess.check_call(f"{conda_path} create -n {env_name} python=3.10 --yes", shell=True)
                except subprocess.CalledProcessError as e:
                    print(f"Error creating conda environment: {e}", file=sys.stderr)
                    sys.exit(1)
            else:
                print(f"--- Conda environment '{env_name}' already exists. Skipping creation. ---")

            print(f"--- Installing dependencies from requirement.txt into '{env_name}' ---")
            try:
                # Command to run pip install within the conda environment
                pip_install_cmd = f"{conda_path} run -n {env_name} pip install -r requirement.txt"
                subprocess.check_call(pip_install_cmd, shell=True)
                print("--- Dependencies installed successfully. ---")
            except subprocess.CalledProcessError as e:
                print(f"Error installing dependencies: {e}", file=sys.stderr)
                sys.exit(1)
        else:
            print("--- Conda not found. Skipping environment creation. ---")
            print("--- Please ensure you have created an environment and installed dependencies manually. ---")
        
        # Proceed with the standard installation
        super().run()


# --- Standard Setup Configuration ---

# Read the contents of your requirements file
try:
    with open('requirement.txt') as f:
        requirements = f.read().splitlines()
except FileNotFoundError:
    print("requirement.txt not found. Please ensure it is in the root directory.", file=sys.stderr)
    requirements = []

# Read README for long description
try:
    with open('README.md', encoding='utf-8') as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = 'A library for automated causal inference.'


setup(
    name='auto_causal',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A library for automated causal inference',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/your-repo/auto-causal',
    packages=find_packages(exclude=['tests', 'tests.*']),
    install_requires=requirements,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Information Analysis',
    ],
    python_requires='>=3.8',
    include_package_data=True,
    zip_safe=False,
    cmdclass={
        'install': CondaInstallCommand,
    }
) 