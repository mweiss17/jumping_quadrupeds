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
        "opencv-python",
        "tqdm",
        "attrs",
        "pillow",
        "wandb",
        "scipy",
        "matplotlib",
        "torchvision",
        "moviepy",
        "imageio",
        "einops",
        "rich",
        "submitit",
        "kornia",
        "dill",
        "h5py",
        "gym @ git+https://git@github.com/tesfaldet/gym@master#egg=gym[box2d]",
        "speedrun @ git+ssh://git@github.com/inferno-pytorch/speedrun@dev#egg=speedrun",
        "wormulon @ git+ssh://git@github.com/mweiss17/wormulon@main#egg=wormulon",
        "duckietown-gym-daffy @ git+https://github.com/duckietown/gym-duckietown@daffy#egg=duckietown-gym-daffy",
        "box2d",
        "dm-env",
    ],
    extras_require={
        "dmc": [
            "dm-control==0.0.403778684",
            "dmc2gym @  git+git://github.com/denisyarats/dmc2gym.git",
            "mujoco_py",
            "dm-env",
        ],
        "meta": ["torchmeta"],
    },
)
