# TRI-VIDAR - Copyright 2022 Toyota Research Institute.  All rights reserved.
import os

import fire
import torch

from vidar.core.wrapper import Wrapper
from vidar.utils.config import read_config


def infer(cfg, checkpoint, input, output, **kwargs):
    """
    Runs an inference with the given config file or checkpoint.
    """

    os.environ['DIST_MODE'] = 'gpu' if torch.cuda.is_available() else 'cpu'

    cfg = read_config(cfg, **kwargs)

    wrapper = Wrapper(cfg, verbose=True)
    wrapper.load(checkpoint)
    wrapper.eval_custom()

    wrapper.evaluate(batch, output)

    if os.path.isdir(input):
        # If input file is a folder, search for image files
        files = []
        for ext in ['png', 'jpg']:
            files.extend(glob((os.path.join(args.input, '*.{}'.format(ext)))))
        files.sort()
        print('Found {} files'.format(len(files)))
    else:
        # Otherwise, use it as is
        files = [args.input]

    # Process each file
    for input_file in files:

        # Load image
        image = load_image(input_file)

        # Resize and to tensor
        image = resize_image(image, image_shape)
        image = to_tensor(image).unsqueeze(0)

        output = wrapper.run_arch(image, 0, False, False)
        breakpoint()


if __name__ == '__main__':
    fire.Fire(infer)
