import os
import subprocess
import sys
import platform
import glob
from setuptools import setup, Extension

def create_virtual_environment():
    '''
    Creates a virtual environment for Python project (if it does not exist).
    '''
    env_dir = 'venv'

    if not os.path.exists(env_dir):
        print(f"Creating virtual environment in {env_dir}...")
        subprocess.check_call([sys.executable, '-m', 'venv', env_dir])
    else:
        print(f"Virtual environment already exists in {env_dir}")

def install_requirements():
    '''
    Installs all required packages in 'requirements.txt' file using pip.
    '''
    print("Installing required packages...")
    subprocess.check_call([os.path.join('venv', 'Scripts', 'pip') if platform.system() == 'Windows'
                           else os.path.join('venv', 'bin', 'pip'), 'install', '-r', 'requirements.txt'])


def activate_virtual_env():
    if platform.system() == 'Windows':
        print("To activate virtual environment, run:")
        print("\t.\\venv\\Scripts\\activate")
    else:
        print("To activate virtual environment, run:")
        print("source ./venv/bin/activate")

def setup_ui():
    print("Setting up PyQt viewer...")
    subprocess.check_call([os.path.join('venv', 'Scripts', 'pyside6-uic') if platform.system() == 'Windows'
                           else os.path.join('venv', 'bin', 'pyside6-uic'), 'viewer.ui', '-o', 'ui_viewer.py'])


def main():
    create_virtual_environment()
    install_requirements()
    setup_ui()
    activate_virtual_env()

if __name__ == '__main__':
    main()

