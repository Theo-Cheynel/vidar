# TRI-VIDAR - Copyright 2022 Toyota Research Institute.  All rights reserved.
import os

import fire
import torch
from glob import glob
from tqdm import tqdm

from PIL import Image

from torchvision.io import read_image
from torchvision.utils import save_image

import torchvision.transforms as T
from shutil import rmtree

from vidar.core.wrapper import Wrapper
from vidar.datasets.augmentations.resize import resize_sample_input, resize_pil
from vidar.datasets.augmentations.tensor import to_tensor_sample, to_tensor_image
from vidar.utils.config import read_config

from scripts.inference.utils import video_utils

from pdb import set_trace as breakpoint

class Log:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    ERROR = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    @staticmethod
    def warning(s):
        print(Log.WARNING + '[Warning] ' + s + Log.ENDC)

    @staticmethod
    def error(s):
        print(Log.ERROR + '[ERROR] ' + s + Log.ENDC)

    @staticmethod
    def info(s):
        print(Log.OKBLUE + '[INFO] ' + s + Log.ENDC)


def upper_power_of_2(x):
    l = torch.log2(x)
    f = torch.floor(l)
    return 2**f if (f == l).all() else 2**(f+1)

def pad_to_upper_power_of_2(image):
    image_size = torch.Tensor([image.shape[-2], image.shape[-1]])
    padded_size = upper_power_of_2(image_size)
    delta_size = ((padded_size - image_size) / 2).int().flip(dims=[0]) # Note: pb if the /2 is odd, one dim will be a power of 2 minus 1, but it appears to work with the network anyway so...

    pad_transform = T.Pad(padding=delta_size.tolist())
    return pad_transform(image
            )

def resize_image_to_tensor(image, base_size, verbose=False):
    """
    Resized image to a multiple of base_size

    Args:
        image : PIL image
        base_size (torch.Tensor)

    Returns:
        torch.Tensor : resized image as a tensor
    """

    base_size = base_size.int()
    image_size = torch.Tensor([image.size[1], image.size[0]]).int()
    
    if (image_size - base_size <= 7).any(): # The image is smaller than the base_size, 7 is the biggest kernel size
        if verbose:
            Log.warning(f'Warning : Image of size {image_size} is smaller than base_size {base_size} and will be stretched to match it.')

        image = resize_pil(image, tuple(base_size.int().tolist()))
        return to_tensor_image(image)[:3] # Remove alpha channel if needed

    else: # The image is larger than the base_size
        image = to_tensor_image(image)
        return image[:, :image_size[0]//base_size[0] * base_size[0], :image_size[1]//base_size[1] * base_size[1]][:3] # Remove alpha channel


def get_images_path_from_folder(folder, verbose=False):
    """
    """

    files = []
    for ext in ['png', 'jpg', 'jpeg']:
        files.extend(glob((os.path.join(folder, '*.{}'.format(ext)))))
    files.sort()
    
    if verbose:
        Log.info(f'Found {len(files)} files')
    
    return files


def infer_depth_map(cfg, checkpoint, input_path, output_path, verbose=False, **kwargs):
    """
    Runs an inference with the given config file or checkpoint.
    """

    os.environ['DIST_MODE'] = 'gpu' if torch.cuda.is_available() else 'cpu'

    cfg = read_config(cfg, **kwargs)

    wrapper = Wrapper(cfg, verbose=False)
    wrapper.load(checkpoint, verbose=False)
    wrapper.eval_custom()

    #wrapper.evaluate(batch, output)

    if os.path.isdir(input_path):
        files = get_images_path_from_folder(input_path, verbose)
    else:
        _, file_extension = os.path.splitext(input_path)

        # Video file : extract image sequence
        extracted_images_folder = None
        if file_extension.lower()[1:] in ['mp4', 'mov', 'wmv', 'avi', 'mkv']:
            extracted_images_folder = video_utils.video_to_images(input_path)
            files = get_images_path_from_folder(extracted_images_folder)
        else:
            # Otherwise, use it as is
            files = [input_path]

    # Process each file
    image_size_mode = None # 'resize' if needed to resize, 'ready' if ready to-run
    for input_file in tqdm(files):

        # Load image
        image = Image.open(input_file) # load as PIL image

        if image_size_mode is None:
            # Try with current image size
            try:
                output = wrapper.run_arch({'rgb': to_tensor_image(image)[(None,)*2]}, 0, False, False)
            except:
                image_size_mode = 'resize'

                if verbose:
                    Log.warning('Warning : image size is not compatible with the network. It will be resized to a compatible size (the closest possible to the original size)')

                # Resizing image to a size that's known to work
                base_shape = torch.Tensor([192, 640])
                image = resize_image_to_tensor(image, base_shape, verbose)

                # Then computing output
                output = wrapper.run_arch({'rgb': image[(None,)*2]}, 0, False, False)
        elif image_size_mode == 'resize':
            # Resizing image to a size that's known to work
            base_shape = torch.Tensor([192, 640])
            image = resize_image_to_tensor(image, base_shape, verbose)

            # Then computing output
            output = wrapper.run_arch({'rgb': image[(None,)*2]}, 0, False, False)
        else:
            output = wrapper.run_arch({'rgb': to_tensor_image(image)[(None,)*2]}, 0, False, False)



        # Normalizing depth maps
        depth_map = output['predictions']['depth'][0][0]
        depth_map /= depth_map.max()

        # Save depth map
        output_full_path = os.path.join(output_path, os.path.basename(input_file))
        save_image(depth_map, output_full_path)

        if verbose:
            Log.info(f'Depth map inference done, saved depth map at {output_path}')

    # Deleting temp folder if needed
    if extracted_images_folder is not None:
        rmtree(extracted_images_folder)




if __name__ == '__main__':
    fire.Fire(infer_depth_map)
