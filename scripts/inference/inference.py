# TRI-VIDAR - Copyright 2022 Toyota Research Institute.  All rights reserved.
import os
import gc
from glob import glob
from pdb import set_trace as breakpoint_pdb
from shutil import rmtree

import fire
import torch
import numpy as np
import torchvision.transforms as T
from PIL import Image
from scripts.inference.utils import video_utils
from torchvision.io import read_image
from torchvision.utils import save_image
from tqdm import tqdm
from vidar.core.wrapper import Wrapper
from vidar.datasets.augmentations.resize import resize_pil, resize_sample_input
from vidar.datasets.augmentations.tensor import (to_tensor, to_tensor_image,
                                                 to_tensor_sample)
from vidar.utils.config import read_config
from vidar.utils.types import is_seq
from vidar.utils.distributed import dist_mode

def breakpoint(list_objects=False):
    if list_objects:
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                    print(type(obj), obj.size())
            except:
                pass

    breakpoint_pdb()

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

def resize_images_to_tensor(images, base_size, verbose=False):
    """
    Resized image to a multiple of base_size

    Args:
        images : list of PIL images
        base_size (torch.Tensor)

    Returns:
        torch.Tensor : images, of shape (batch_size, 3, H, W)
    """
    if len(images) == 0:
        return []

    base_size = base_size.int()
    image_size = torch.Tensor([images[0].size[1], images[0].size[0]]).int()
    
    if (image_size - base_size <= 7).any(): # The image is smaller than the base_size, 7 is the biggest kernel size
        if verbose:
            Log.warning(f'Warning : Image of size {image_size} is smaller than base_size {base_size} and will be stretched to match it.')

        images = resize_pil(images, tuple(base_size.int().tolist()))
        
        #return to_tensor_image(image)[:3] # Remove alpha channel if needed
        return torch.stack([i[:3] for i in images])

    else: # The image is larger than the base_size : adapt its ratio to match the greater multiple of the base size, independantly for each dimension (it works)
        images = to_tensor_image(images)

        #return image[:, :image_size[0]//base_size[0] * base_size[0], :image_size[1]//base_size[1] * base_size[1]][:3] # Remove alpha channel
        return torch.stack([i[:, :image_size[0]//base_size[0] * base_size[0], :image_size[1]//base_size[1] * base_size[1]][:3] for i in images])


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


def infer_batch_with_resize_test(images, wrapper, verbose=False):
    """
    Returns the type of resizing mode associated with the given batch

    Args:
        images: list of PIL images to infer from or list of images path
        wrapper: paper's wrapper object
    
    Returns:
        string: 'resize' of images need to be resized, None otherwise.
        torch.Tensor : batch inference of the given images.
    """

    image_resize_mode = None
    try:
        inference = infer_batch(images, wrapper, image_resize_mode=None, verbose=verbose)
    except:
        inference = infer_batch(images, wrapper, image_resize_mode='resize', verbose=verbose)
        image_resize_mode = 'resize'

    return image_resize_mode, inference


def infer_batch(images, wrapper, image_resize_mode, verbose=False):
    """
    Performs an inference on a batch.
    Warning : this method DO NOT perform a memory check, images must fit in available memory, and the given device.
    TODO : Add possibility to normalize result ! (To avoid saturation when saved as an image)

    Args:
        images: list of PIL images to infer from or list of images path
        wrapper : Paper's wrapper object
        image_resize_mode : Either None if nothing needs to be done, or 'resize' to match network requirements.
        verbose : verbose

    Returns:
        torch.Tensor : depth maps of shape (batch_size, ...)
    """

    if len(images) == 0:
        return

    # Images path are passed instead of images directly : loading them
    if isinstance(images[0], str):
        images = [Image.open(path) for path in images]

    if image_resize_mode is None:
        batch_tensor = to_tensor_image(images)
        predictions = wrapper.run_arch({'rgb': torch.stack(batch_tensor).unsqueeze(0)}, 0, False, False)
    elif image_resize_mode == 'resize':
        base_size = torch.Tensor([192, 640])
        batch_tensor = resize_images_to_tensor(images, base_size, verbose).unsqueeze(0)
        # predictions = wrapper.run_arch({'rgb': batch_tensor}, 0, False, False)

    #predictions = None

    # Close images
    for img in images:
        img.close()

    del images
    #print("Closed and deleted images & tensors")
    #breakpoint()
        
    return predictions


def infer_depth_map(cfg, checkpoint, input_path, output_path, resize_check=False, verbose=False, **kwargs):
    """
    Runs an inference with the given config file or checkpoint.

    Args:
        cfg : Config object
        checkpoint : Network checkpoint to infer with
        input_path : either an image or a video path
        output_path : a folder to save the infered depth maps to
        resize_check (bool): will try one batch without resizing, and resize afterwards if it failed.
        verbose : verbose

    Returns:
        None, the depth maps will be written in output_path
    """

    os.environ['DIST_MODE'] = 'gpu' if torch.cuda.is_available() else 'cpu'

    cfg = read_config(cfg, **kwargs)

    wrapper = Wrapper(cfg, verbose=False)
    wrapper.load(checkpoint, verbose=False)
    wrapper.eval_custom()
    
    if dist_mode() == 'gpu':
        wrapper.cuda() # Puts the model on GPU. TODO : Directly load it on the GPU instead of loading it in CPU then pushing it in VRAM.

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


    batch_size = 5
    

    # Process each remaining batch
    with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=0, warmup=0, active=3, repeat=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('/data/log/profiler'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:

        if resize_check:
            # Test the resize method with the first batch
            image_resize_mode, prediction = infer_batch_with_resize_test(files[0:batch_size], wrapper, verbose)

            print("Tested first batch, image_resize_mode is", image_resize_mode)
            for i, depth_map in enumerate(prediction['predictions']['depth'][0]):
                depth_map /= depth_map.max()
                save_image(depth_map, files[i])

            print('image_resize_mode =', image_resize_mode)

        # Temporary
        image_resize_mode = None

        start = 0 if not resize_check else batch_size
        print(start)
        batch_filepaths = [files[i:i+batch_size] for i in range(start, len(files), batch_size)]
        for filepaths in tqdm(batch_filepaths):

            # Inference 
            predictions = infer_batch(filepaths, wrapper, image_resize_mode, verbose)
            print(predictions['predictions']['intrinsics'])
            depth_maps = predictions['predictions']['depth'][0]
            #print("#### Inference done")
        
            # Saving depth maps
            output_full_paths = [os.path.join(output_path, os.path.basename(f)) for f in filepaths]
            for i in np.arange(start=0, stop=len(depth_maps), step=4):
                # TODO : Batchify the normalization
                for in_batch_index in range(depth_maps[i].shape[0]): # Not using batch_size but the shape here for the last element that can have a dim < batch_size
                    save_image(depth_maps[i][in_batch_index] / depth_maps[i][in_batch_index].max(), output_full_paths[in_batch_index]) # Saving with normalization
            

            if verbose:
                Log.info(f'Depth map inference done, saved depth map at {output_path}')
            
            prof.step()
            #print("#### Batch done")

    # Deleting temp folder if needed
    if extracted_images_folder is not None:
        rmtree(extracted_images_folder)


if __name__ == '__main__':
    fire.Fire(infer_depth_map)
