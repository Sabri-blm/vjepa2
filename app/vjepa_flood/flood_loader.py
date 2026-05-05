# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import json
import os
from logging import getLogger
from math import ceil

import h5py
import numpy as np
import pandas as pd
import torch
import torch.utils.data
from torch.utils.data import Dataset
from decord import VideoReader, cpu
from scipy.spatial.transform import Rotation

_GLOBAL_SEED = 0
logger = getLogger()


def init_data(
    root_path,
    transform,
    batch_size,
    collator=None,
    pin_mem=True,
    num_workers=8,
    world_size=1,
    rank=0,
    training=True,
    drop_last=True,
    persistent_workers=False,
    #log_dir=None,
    #datasets_weights=None,
    #deterministic=True,
    dataset_fpcs=None,
    num_clips=1,
    frame_skip=1
):
    dataset = GeoNPYDataset(
        root=root_path,
        train=training,
        transform=transform,
        dataset_fpcs=dataset_fpcs,
        num_clips=num_clips,
        frame_skip=frame_skip
    )

    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0) and persistent_workers,
    )

    logger.info("VideoDataset unsupervised data loader created")

    return data_loader, dist_sampler

class GeoNPYDataset(Dataset):
    """
    Loads local .npy geospatial tensors shaped (C, H, W).
    Each .npy file is one sample.
    """

    def __init__(self, 
                 root, 
                 train=True,
                 transform=None,
                 dataset_fpcs=None,
                 frame_skip=1,
                 num_clips=1,
                 val_ratio=0.05,   # 5% validation split
                 seed=0            # deterministic split
                ):
        self.max_frames = max(dataset_fpcs)
        self.root = root
        self.transform = transform
        self.num_clips = num_clips
        self.frame_skip = frame_skip

        # list all .npy files
        all_files = sorted([
            os.path.join(self.root, f)
            for f in os.listdir(self.root)
            if f.endswith(".npy")
        ])

        if len(all_files) == 0:
            raise RuntimeError(f"No .npy files found in {self.root}")
        # deterministic split 
        rng = np.random.RandomState(seed)
        indices = np.arange(len(all_files))
        rng.shuffle(indices)

        split = int(len(indices) * (1 - val_ratio))
        train_idx = indices[:split]
        val_idx = indices[split:]

        self.files = [all_files[i] for i in (train_idx if train else val_idx)]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx, rainfall_value=0.7):
        path = self.files[idx]

        # load numpy array [T, H, W, C]
        arr = np.load(path)

        #print("arr shape: ", arr.shape)

        assert arr.shape[0] == self.max_frames, f"The first indice is suppose to be 4 but got{arr.shape[0]}"

        # rainfall per clip
        rainfall_intensities = torch.full((arr.shape[0], 1), rainfall_value, dtype=torch.float32)
        rainfall_intensities = rainfall_intensities[::self.frame_skip]
        rainfall_intensities = rainfall_intensities[1:, :] - rainfall_intensities[:-1, :]

        clip_indices = torch.arange(arr.shape[0], dtype=torch.long)

        buffer = torch.from_numpy(arr).float()

        #print("before transform: ", len(buffer), buffer[0].shape)
        #print("before transform: ", len(clip_indices))
        if self.transform is not None:
            buffer = self.transform(buffer)

        #print("after transform: ", len(buffer), buffer[0].shape)
        #print("after transform: ", len(clip_indices))

        return buffer, rainfall_intensities, clip_indices
