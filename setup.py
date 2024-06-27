from setuptools import setup

with open('readme.md', 'r', encoding='utf8') as file:
    long_description = file.read()

setup(
    name='emodel',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Podzolkov Pavel',
    author_email='ppodzolkoff@gmail.com',
    version='0.1.0',
    description='package for compartmental epidemic modelling',
    install_requires=['numpy', 'pandas', 'matplotlib', 'scipy', 'prettytable'],
    packages=['emodel'],
    python_requires='>=3.11',
    classifiers=["Programming Language :: Python :: 3",
                 "License :: OSI Approved :: MIT License",
                 "Operating System :: OS Independent",
                 'Natural Language :: Russian']
)
