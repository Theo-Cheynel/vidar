# TRI-VIDAR - Copyright 2022 Toyota Research Institute.  All rights reserved.

import os

import numpy as np

from vidar.datasets.BaseDataset import BaseDataset
from vidar.datasets.utils.misc import stack_sample, make_relative_pose
from vidar.utils.read import read_image

class MinimalDataset(BaseDataset):
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
        self.tag = 'minimal'

        self.single_intrinsics = np.array([ [0.58, 0.00, 0.5, 0],
                                            [0.00, 1.92, 0.5, 0],
                                            [0.00, 0.00, 1.0, 0],
                                            [0.00, 0.00, 0.0, 1]], dtype=np.float32)

        self.split = split.split('/')[-1].split('.')[0]

        self._cache = {}
        self.sequence_origin_cache = {}

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

            # Intrinsics
            parent_folder = self._get_parent_folder(filename)
            if parent_folder in self.calibration_cache:
                c_data = self.calibration_cache[parent_folder]
            else:
                c_data = self._read_raw_calib_file(parent_folder)
                self.calibration_cache[parent_folder] = c_data

            # Return individual or single intrinsics
            if self.single_intrinsics is not None:
                intrinsics = self.single_intrinsics.copy()
                intrinsics[0, :] *= sample['rgb'][0].size[0]
                intrinsics[1, :] *= sample['rgb'][0].size[1]
                sample['intrinsics'] = {0: intrinsics}
            else:
                raise NotImplementedError("Minimal example only works with single intrinsics for now.")

            # Add context information if requested
            if self.with_context:

                # Add context images
                all_context_idxs = self.backward_context_paths[idx] + \
                                   self.forward_context_paths[idx]
                image_context_paths, depth_context_paths = \
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
