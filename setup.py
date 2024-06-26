from setuptools import setup

with open('readme.md', 'r', encoding='utf8') as file:
    long_description = file.read()

setup(
    name='emodel',
    long_description=long_description,
    author='Podzolkov Pavel',
    author_email='ppodzolkoff@gmail.com',
    version='0.1',
    description='package for compartmental epidemic modelling',
    install_requires=['numpy', 'pandas', 'matplotlib', 'scipy', 'prettytable'],
    packages=['emodel']
)
