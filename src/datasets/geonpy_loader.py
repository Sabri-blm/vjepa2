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

):
    dataset = GeoNPYDataset(
        root=root_path, 
        train=training,
        transform=transform,
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
                 val_ratio=0.05,   # 5% validation split
                 seed=0            # deterministic split
                ):
        self.root = root[0]
        self.transform = transform

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

        # load numpy array (C, H, W)
        arr = np.load(path)

        assert arr.shape[0] == 4, f"The first indice is suppose to be 4 but got{arr.shape[0]}"

        # arr is [C, T, H, W]
        #arr = np.transpose(arr, (0, 2, 3, 1))   # → [T, H, W, C]

        #print("-----------------------------------1", arr.shape)
        # convert to torch tensor
        img = torch.from_numpy(arr).float()

        # apply I-JEPA transforms (expects CHW tensor)
        if self.transform is not None:
            img = self.transform(img)


        T = img.shape[1]
        # img is (C, T, H, W)
        #img = img.permute(1, 0, 2, 3)   # → (T, C, H, W)

        #print("-----------------------------------2", img.shape)
        #clip_indices = [list(range(T))]
        clip_indices = [torch.arange(T, dtype=torch.long)]
        # I-JEPA ignores labels → return dummy target
        return [img], 0, clip_indices
        
