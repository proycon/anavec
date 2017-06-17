#! /usr/bin/env python3
# -*- coding: utf8 -*-

from __future__ import print_function

import os
import sys
from setuptools import setup


try:
   os.chdir(os.path.dirname(sys.argv[0]))
except:
   pass


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "Anavec",
    version = "0.1",
    author = "Maarten van Gompel",
    author_email = "proycon@anaproy.nl",
    description = ("Spelling correction and normalisation using anagram vectors"),
    license = "GPL",
    keywords = "spelling corrector spell check nlp computational_linguistics rest",
    url = "https://github.com/proycon/anavec",
    packages=['anavec'],
    long_description=read('README.rst'),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Text Processing :: Linguistic",
        "Programming Language :: Python :: 3.3",
        "Programming Language :: Python :: 3.4",
        "Operating System :: POSIX",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
    entry_points = {
        'console_scripts': [
            'anavec = anavec.anavec:main',
            'anavec_icdar_task2 = anavec.icdar_process:main',
            'anavec_test = anavec.test:main'
        ]
    },
    package_data = {'anavec':[] },
    install_requires=['colibricore >= 2.4','numpy','theano','python-Levenshtein']
)
