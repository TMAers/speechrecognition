from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

setup(
    name='meeting-assistant',
    version='0.1.0',
    description='DC4B Meeting Assistant Demo',
    url='https://github.com/TMAers/speechrecognition',
    packages=find_packages()
)