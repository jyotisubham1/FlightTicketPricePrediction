from setuptools import find_packages,setup



with open('requirements.txt') as f:
    requirements = f.read().splitlines()
    

setup(
    name='Flight Ticket Price Prediction',
    version='0.0.1',
    author='Jyotisubham Panda',
    author_email='jyotisubham.panda@gmail.com',
    install_requires=requirements,
    packages=find_packages()
)