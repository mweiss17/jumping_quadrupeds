#!/usr/bin/env python
import os

from setuptools import setup

requirements_path = os.path.dirname(os.path.realpath(__file__)) + '/requirements.txt'
install_requires = []

if os.path.isfile(requirements_path):
    with open(requirements_path) as f:
        install_requires = f.read().splitlines()

setup(name='jumping_quadrupeds',
      version='1.0',
      description='Deep Reinforcement Learning package',
      author='Martin Weiss',
      author_email='martin.clyde.weiss@gmail.com',
      url='',
      install_requires=[
          'torch==1.9',
          'tqdm',
          'gym[box2d]',
          'pillow',
          'scipy',
          'matplotlib',
          'torchvision',
          ],
      py_modules=['jumping_quadrupeds']
     )

