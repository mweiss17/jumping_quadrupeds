#!/usr/bin/env python
import os

from setuptools import setup

setup(
    name="jumping_quadrupeds",
    version="1.0",
    description="Deep Reinforcement Learning package",
    author="Martin the Weiss",
    author_email="martin.clyde.weiss@gmail.com",
    url="",
    install_requires=[
        "torch>=1.8",
        "tqdm",
        "gym[box2d]",
        "pillow",
        "scipy",
        "matplotlib",
        "torchvision",
        "speedrun @ git+ssh://git@github.com/inferno-pytorch/speedrun@master#egg=speedrun",
    ],
)
