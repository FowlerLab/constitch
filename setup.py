from setuptools import setup
import os

# parse requirements.txt
lib_folder = os.path.dirname(os.path.realpath(__file__))
requirement_path = lib_folder + "/requirements.txt"
install_requires = []
if os.path.isfile(requirement_path):
    with open(requirement_path) as ifile:
        install_requires = ifile.read().splitlines()

setup(
    name='constitch',
    version='0.1.0',
    packages=['constitch/'],
	install_requires=install_requires,
)
