import random
import torch
from typing import List, Union


class PositionRandomNoise(object):
    def __init__(self, noise_magnitude=1.0):
        self.noise_magnitude = noise_magnitude

    def __call__(self, data):
        data.pos = data.pos + self.noise_magnitude * torch.randn_like(data.pos)
        return data

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.noise_magnitude)


class FeatureRandomScale(object):
    def __init__(self, scales=(2.0, 2.0), attrs: List[str] = ["x"]):
        assert isinstance(scales, (tuple, list)) and len(scales) == 2
        self.scales = scales
        self.attrs = attrs

    def __call__(self, data):
        data.x = data.x.float()
        scale = random.uniform(*self.scales)
        data.x = data.x * scale
        return data

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.scales)


class FeatureRandomNoise(object):
    def __init__(self, noise_magnitude=1.0, attrs: List[str] = ["x"]):
        self.noise_magnitude = noise_magnitude
        self.attrs = attrs

    def __call__(self, data):
        data.x = data.x.float()
        data.x = data.x + self.noise_magnitude * torch.randn_like(data.x)
        return data

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.noise_magnitude)
