# TRI-VIDAR - Copyright 2022 Toyota Research Institute.  All rights reserved.
import glob
import os

import numpy as np

from pdb import set_trace as breakpoint

from vidar.datasets.BaseDataset import BaseDataset
from vidar.datasets.utils.misc import stack_sample, make_relative_pose
from vidar.utils.read import read_image

class OverfitDataset(BaseDataset):
    """
    Minimal dataset class

    Parameters
    ----------
    path : String
        Path to the dataset
    split : String
        Split file, with paths to the images to be used
    data_transform : Function
        Transformations applied to the sample
    """
    def __init__(self, split, **kwargs):
        super().__init__(**kwargs)
        self.tag = 'overfit'

        # self.single_intrinsics = np.array([ [0.58, 0.00, 0.5, 0],
        #                                     [0.00, 1.92, 0.5, 0],
        #                                     [0.00, 0.00, 1.0, 0],
        #                                     [0.00, 0.00, 0.0, 1]], dtype=np.float32)

        # self.single_intrinsics = np.array([ [0.5625, 0.00, 0.5, 0],
        #                                     [0.00,   1,    0.5, 0],
        #                                     [0.00,   0.00, 1.0, 0],
        #                                     [0.00,   0.00, 0.0, 1]], dtype=np.float32)


        self.single_intrinsics = np.array([ [1,      0.00, 0.5, 0],
                                            [0.00,   1,    0.5, 0],
                                            [0.00,   0.00, 1.0, 0],
                                            [0.00,   0.00, 0.0, 1]], dtype=np.float32)
                                                

        self.split = split.split('/')[-1].split('.')[0]

        # Things I don't know what are they used for
        self._cache = {}
        self.calibration_cache = {}
        self.sequence_origin_cache = {}
        self.backward_context_paths = []
        self.forward_context_paths = []

        # Get files from the txt file
        with open(os.path.join(self.path, split), "r") as f:
            data = f.readlines()
        self.paths = [os.path.join(self.path, fname.split()[0]) for fname in data]

        # If using context, filter file list
        # TODO : Figure out what this does
        if self.with_context:
            paths_with_context = []
            for stride in [1]:
                for idx, file in enumerate(self.paths):
                    backward_context_idxs, forward_context_idxs = \
                        self._get_sample_context(
                            file, self.bwd_context, self.fwd_context, stride)
                    if backward_context_idxs is not None and forward_context_idxs is not None:
                        paths_with_context.append(self.paths[idx])
                        self.forward_context_paths.append(forward_context_idxs)
                        self.backward_context_paths.append(backward_context_idxs[::-1])
            self.paths = paths_with_context

    @staticmethod
    def _get_next_file(idx, file):
        """Get next file given next idx and current file"""
        base, ext = os.path.splitext(os.path.basename(file))
        return os.path.join(os.path.dirname(file), str(idx).zfill(len(base)) + ext)

    @staticmethod
    def _get_parent_folder(image_file):
        """Get the parent folder from image_file"""
        return os.path.abspath(os.path.join(image_file, "../../"))

    def _get_sample_context(self, sample_name,
                            backward_context, forward_context, stride=1):
        """
        Get a sample context

        Parameters
        ----------
        sample_name : String
            Path + Name of the sample
        backward_context : Int
            Size of backward context
        forward_context : Int
            Size of forward context
        stride : Int
            Stride value to consider when building the context

        Returns
        -------
        backward_context : list[Int]
            List containing the indexes for the backward context
        forward_context : list[Int]
            List containing the indexes for the forward context
        """
        base, ext = os.path.splitext(os.path.basename(sample_name))
        parent_folder = os.path.dirname(sample_name)
        f_idx = int(base)

        # Check number of files in folder
        if parent_folder in self._cache:
            max_num_files = self._cache[parent_folder]
        else:
            max_num_files = len(glob.glob(os.path.join(parent_folder, '*' + ext)))
            self._cache[parent_folder] = max_num_files

        # Check bounds
        if (f_idx - backward_context * stride) < 0 or (
                f_idx + forward_context * stride) >= max_num_files:
            return None, None

        # Backward context
        c_idx = f_idx
        backward_context_idxs = []
        while len(backward_context_idxs) < backward_context and c_idx > 0:
            c_idx -= stride
            filename = self._get_next_file(c_idx, sample_name)
            if os.path.exists(filename):
                backward_context_idxs.append(c_idx)
        if c_idx < 0:
            return None, None

        # Forward context
        c_idx = f_idx
        forward_context_idxs = []
        while len(forward_context_idxs) < forward_context and c_idx < max_num_files:
            c_idx += stride
            filename = self._get_next_file(c_idx, sample_name)
            if os.path.exists(filename):
                forward_context_idxs.append(c_idx)
        if c_idx >= max_num_files:
            return None, None

        return backward_context_idxs, forward_context_idxs

    def _get_context_files(self, sample_name, idxs):
        """
        Returns image context files

        Parameters
        ----------
        sample_name : String
            Name of current sample
        idxs : list[Int]
            Context indexes

        Returns
        -------
        image_context_paths : list[String]
            List of image names for the context
        None : depth files if they were supported
        """
        image_context_paths = [self._get_next_file(i, sample_name) for i in idxs]
        return image_context_paths, None

########################################################################################################################

    def __len__(self):
        """Dataset length"""
        return len(self.paths)

    def __getitem__(self, idx):
        """Get dataset sample"""

        samples = []

        for camera in self.cameras:

            # Filename
            filename = self.paths[idx] if camera == 0 else self.paths_stereo[idx]

            # Base sample
            sample = {
                'idx': idx,
                'tag': self.tag,
                'filename': self.relative_path({0: filename}),
                'splitname': '%s_%010d' % (self.split, idx) # Here is the specification of the image name format
            }

            # Image
            sample['rgb'] = {0: read_image(filename)}

            # Return individual or single intrinsics
            if self.single_intrinsics is not None:
                intrinsics = self.single_intrinsics.copy()
                print("Single intrinsics", self.single_intrinsics)
                intrinsics[0, :] *= sample['rgb'][0].size[0]
                intrinsics[1, :] *= sample['rgb'][0].size[1]
                print(intrinsics, intrinsics.shape)
                sample['intrinsics'] = {0: intrinsics}
            else:
                raise NotImplementedError("Minimal example only works with single intrinsics for now.")

            # Add context information if requested
            if self.with_context:

                # Add context images
                all_context_idxs = self.backward_context_paths[idx] + \
                                   self.forward_context_paths[idx]
                image_context_paths, _ = \
                    self._get_context_files(filename, all_context_idxs)
                image_context = [read_image(f) for f in image_context_paths]
                sample['rgb'].update({
                    key: val for key, val in zip(self.context, image_context)
                })

            # Stack sample
            samples.append(sample)

        # Make relative poses
        samples = make_relative_pose(samples)

        # Transform data
        if self.data_transform:
            samples = self.data_transform(samples)

        # Return stacked sample
        return stack_sample(samples)

########################################################################################################################
