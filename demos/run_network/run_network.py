# TRI-VIDAR - Copyright 2022 Toyota Research Institute.  All rights reserved.

import torch
import numpy as np

from vidar.arch.networks.depth.MonoDepthResNet import MonoDepthResNet
from vidar.utils.config import read_config

### Create network

cfg = read_config('demos/run_network/config.yaml')
net = MonoDepthResNet(cfg)

### Create dummy input and run network

def upper_power_of_2(x):
    l = np.log2(x)
    f = int(np.floor(l))
    return 2**f if l == f else 2**(f+1)

input_height = 375
input_width = 1242

height = upper_power_of_2(input_height)
width = upper_power_of_2(input_width)

rgb = torch.randn((2, 3, 128, 256))
depth = net(rgb=rgb)['depths']

