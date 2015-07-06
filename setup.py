#!/usr/bin/env python

#from distutils.core import setup
from setuptools import setup, Extension, find_packages
from setuptools.dist import Distribution


with open('README.rst') as file:
    long_description = file.read()

setup(name='famamcbeth',
      version='1.0',
      description=('Estimate risk premia in Fama-McBeth regressions with both 2-step OLS and GMM'),
      long_description=long_description,
      author='Stanislav Khrapov',
      license='MIT',
      author_email='khrapovs@gmail.com',
      url='https://github.com/khrapovs/famamcbeth',
      py_modules=['famamcbeth'],
      packages=find_packages(),
      keywords=['risk', 'return', 'premium', 'Fama', 'McBeth', 'regression',
        'OLS', 'GMM', 'estimation'],
      classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: End Users/Desktop',
        'Intended Audience :: Financial and Insurance Industry',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
      ],
)
