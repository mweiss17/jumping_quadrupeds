import argparse
import os

from torchvision.transforms import transforms


def get_args():  # TODO make this `get_args_ppo()` or something more specific
    """
    Description:
    Parses arguments at command line.
    Parameters:
    None
    Return:
    args - the arguments parsed
    """
    parser = argparse.ArgumentParser()

    # can be 'train' or 'test'
    parser.add_argument("--mode", dest="mode", type=str, default="train")

    # your actor model filename
    parser.add_argument("--actor_model", dest="actor_model", type=str, default="")

    # your critic model filename
    parser.add_argument("--critic_model", dest="critic_model", type=str, default="")

    # BipedalWalker-v3
    parser.add_argument("--env", type=str, default="CarRacing-v0")

    args = parser.parse_args()

    return args


def common_img_transforms(with_flip=False):
    out = [transforms.Resize(64)]
    if with_flip:
        out.append(transforms.RandomHorizontalFlip())
    out.append(transforms.ToTensor())
    return transforms.Compose(out)


def abs_path(x):
    return os.path.abspath(os.path.expanduser(os.path.expandvars(x)))
