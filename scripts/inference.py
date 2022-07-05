# TRI-VIDAR - Copyright 2022 Toyota Research Institute.  All rights reserved.
import os

import fire
import torch


from torchvision.io import read_image
from torchvision.utils import save_image

import torchvision.transforms as T

from vidar.core.wrapper import Wrapper
from vidar.utils.config import read_config

from pdb import set_trace as breakpoint

def upper_power_of_2(x):
    l = torch.log2(x)
    f = torch.floor(l)
    return 2**f if (f == l).all() else 2**(f+1)

def pad_to_upper_power_of_2(image):
    image_size = torch.Tensor([image.shape[-2], image.shape[-1]])
    padded_size = upper_power_of_2(image_size)
    delta_size = ((padded_size - image_size) / 2).int().flip(dims=[0]) # Note: pb if the /2 is odd, one dim will be a power of 2 minus 1, but it appears to work with the network anyway so...
    pad_transform = T.Pad(padding=delta_size.tolist())
    return pad_transform(image)

def infer(cfg, checkpoint, input, output, **kwargs):
    """
    Runs an inference with the given config file or checkpoint.
    """

    os.environ['DIST_MODE'] = 'gpu' if torch.cuda.is_available() else 'cpu'

    cfg = read_config(cfg, **kwargs)

    wrapper = Wrapper(cfg, verbose=True)
    wrapper.load(checkpoint, verbose=True)
    wrapper.eval_custom()

    #wrapper.evaluate(batch, output)

    if os.path.isdir(input):
        # If input file is a folder, search for image files
        files = []
        for ext in ['png', 'jpg']:
            files.extend(glob((os.path.join(args.input, '*.{}'.format(ext)))))
        files.sort()
        print('Found {} files'.format(len(files)))
    else:
        # Otherwise, use it as is
        files = [input]

    # Process each file
    for input_file in files:

        # Load image
        image = read_image(input_file)

        # Resize and to tensor
        image = pad_to_upper_power_of_2(image)
        print("#### AFTER PADDING", image.shape)
        #image = resize_image(image, image_shape)
        #image = to_tensor(image).unsqueeze(0)

        # Contains multiple batches
        batches = {
            'rgb': image.unsqueeze(0).unsqueeze(0) # First element of the first batch
        }
    
        output = wrapper.run_arch(batches, 0, False, False)
        
        save_image(output['predictions']['depth'][0][0], '/data/output.png')


if __name__ == '__main__':
    fire.Fire(infer)
