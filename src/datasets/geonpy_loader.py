import os
import numpy as np
import torch
from torch.utils.data import Dataset

import pathlib
from logging import getLogger


from src.datasets.utils.dataloader import ConcatIndices, MonitoredDataset, NondeterministicDataLoader
from src.datasets.utils.weighted_sampler import DistributedWeightedSampler

_GLOBAL_SEED = 0
logger = getLogger()

def make_geonpy_loader(
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
    log_dir=None,
    datasets_weights=None,
    deterministic=True,
    dataset_fpcs=None,
    num_clips=1,
    decoding=False,
    only_flood=True,
    c=2
):
    dataset = GeoNPYDataset(
        root=root_path,
        train=training,
        transform=transform,
        dataset_fpcs=dataset_fpcs,
        num_clips=num_clips,
        decoding=decoding,
        only_flood=only_flood,
        c=c
    )

    log_dir = pathlib.Path(log_dir) if log_dir else None
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        # Worker ID will replace '%w'
        resource_log_filename = log_dir / f"resource_file_{rank}_%w.csv"
        dataset = MonitoredDataset(
            dataset=dataset,
            log_filename=str(resource_log_filename),
            log_interval=10.0,
            monitor_interval=5.0,
        )

    logger.info("GeoVideoDataset dataset created")
    if datasets_weights is not None:
        dist_sampler = DistributedWeightedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    else:
        dist_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True
        )

    if deterministic:
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
    else:
        data_loader = NondeterministicDataLoader(
            dataset,
            collate_fn=collator,
            sampler=dist_sampler,
            batch_size=batch_size,
            drop_last=drop_last,
            pin_memory=pin_mem,
            num_workers=num_workers,
            persistent_workers=(num_workers > 0) and persistent_workers,
        )
    logger.info("GeoVideoDataset unsupervised data loader created")

    return dataset, data_loader, dist_sampler

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
                 decoding=False,
                 only_flood=True,
                 c=2,
                 num_clips=1,
                 val_ratio=0.05,   # 5% validation split
                 seed=0            # deterministic split
                ):
        self.max_frames = max(dataset_fpcs)
        self.root = root[0]
        self.transform = transform
        self.num_clips = num_clips
        self.decoding = decoding
        self.only_flood = only_flood
        self.channels = c

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

    def __getitem__(self, idx):
        path = self.files[idx]

        # load numpy array [T, H, W, C]
        arr = np.load(path)

        #print("arr shape: ", arr.shape)

        assert arr.shape[0] == self.max_frames, f"The first indice is suppose to be 4 but got{arr.shape[0]}"

        clip_indices = [torch.arange(arr.shape[0], dtype=torch.long)]

        buffer = torch.from_numpy(arr).float()
        #print("before transform: ", len(buffer), buffer[0].shape)
        #print("before transform: ", len(clip_indices))

        if self.transform is not None:
            buffer = [self.transform(buffer)] # [self.transform(clip) for clip in buffer]

        #print("after transform: ", len(buffer), buffer[0].shape)
        #print("after transform: ", len(clip_indices))

        if self.decoding:
            flood_maps = buffer[0][0, ...].unsqueeze(0) if self.only_flood else buffer[0][:self.channels, ...]
            #print(buffer[0].shape, flood_maps.shape)

            return buffer[0], flood_maps
        
        return buffer[0], 0, clip_indices

    '''def __getitem__(self, idx):
        path = self.files[idx]

        # 1) load once
        arr = np.load(path)          # (T, H, W, C)
        
        # 2) tensor + layout once
        arr_t = torch.from_numpy(arr).permute(3, 0, 1, 2)  # (C, T, H, W)

        C, T, H, W = arr_t.shape  # assuming (T, ...)
        clip_len = T // self.num_clips
        
        # 3) reshape into clips without Python loops
        clips = arr_t.reshape(self.num_clips, C, clip_len, H, W)  # (B, C, T, H, W)
        
        # 4) single transform call on the whole batch
        # apply transform per clip
        clips = torch.stack([self.transform(c) for c in clips], dim=0)

        clip_indices = [torch.arange(i * clip_len, (i + 1) * clip_len, dtype=torch.long) for i in range(self.num_clips)]

        return clips, 0, clip_indices'''

