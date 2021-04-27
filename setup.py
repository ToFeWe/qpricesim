#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

# Requirements are so restrictive due to the cluster I am using 
# TODO: Change this and then just use a requiremnts.txt in the 
# simulation package.
requirements = ['numpy==1.19.0', 'numba==0.51.1']

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest>=3', ]

setup(
    author="Tobias Felix Werner",
    author_email='tobias.felix.werner@gmail.com',
    python_requires='>=3.5',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="A package to simulate q-learning agents in an economics environment with non-elastic demand.",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='qpricesim',
    name='qpricesim',
    packages=find_packages(include=['qpricesim', 'qpricesim.*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/ToFeWe/qpricesim',
    version='0.2.1',
    zip_safe=False,
)
