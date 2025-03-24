import os
import random
import json
import numpy as np
import imageio
from typing import Literal, cast
import roma

from transforms import *

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate



# class PARTNET(Dataset):
#     def __init__(self, split='train'):
#         super(PARTNET, self).__init__()
        
#         assert split in ['train', 'val', 'test']
#         self.split = split
#         self.root_dir = your_path     
#         self.files = os.listdir(self.root_dir)
#         self.img_transform = transforms.Compose([
#                transforms.ToTensor()])

#     def __getitem__(self, index):
#         path = self.files[index]
#         image = Image.open(os.path.join(self.root_dir, path, "0.png")).convert("RGB")
#         image = image.resize((128 , 128))
#         image = self.img_transform(image)
#         sample = {'image': image}

#         return sample
            
    
#     def __len__(self):
#         return len(self.files)

class ShapeOfMotion(Dataset):
    def __init__(self, data_dir, transform=None):        
        self.data_dir = data_dir
        self.ckpt = torch.load(f"{data_dir}/checkpoints/last.ckpt") # If RAM OOM, could try dynamic load.
        self.img_dir = f"{data_dir}/images/"
        self.img_ext = os.path.splitext(os.listdir(self.img_dir)[0])[1]
        self.frame_names = [os.path.splitext(p)[0] for p in sorted(os.listdir(self.img_dir))]
        self.imgs: list[torch.Tensor | None] = [None for _ in self.frame_names]
        self.transform = transform

    @property
    def num_frames(self) -> int:
        return len(self.frame_names)

    def __len__(self):
        return len(self.frame_names)
    
    def get_image(self, index) -> torch.Tensor:
        if self.imgs[index] is None:
            self.imgs[index] = self.load_image(index)
        img = cast(torch.Tensor, self.imgs[index])
        return img
    
    def get_fg_4dgs(self, ts: torch.Tensor) -> torch.Tensor:
        means, quats, scales, opacities, colors = self.load_3dgs('fg')

        if ts is not None:
            transfms = self.get_transforms(ts)  # (G, B, 3, 4)
            means_ts = torch.einsum(
                "pnij,pj->pni",
                transfms,
                F.pad(means, (0, 1), value=1.0),
            ) # (G, B, 3)
            quats_ts = roma.quat_xyzw_to_wxyz(
                (
                    roma.quat_product(
                        roma.rotmat_to_unitquat(transfms[..., :3, :3]),
                        roma.quat_wxyz_to_xyzw(quats[:, None]),
                    )
                )
            )
            quats_ts = F.normalize(quats_ts, p=2, dim=-1) # (G, B, 4)
            means_ts = means_ts[:, 0]
            quats_ts = quats_ts[:, 0]

        return torch.cat([self.min_max_norm(t) for t in (means, means_ts, quats, quats_ts, scales, opacities, colors)], dim=1)
    
    def get_all_4dgs(self, ts: torch.Tensor) -> torch.Tensor:
        bg_gs = torch.cat(self.load_3dgs('bg'), dim=1)
        fg_gs = self.get_fg_4dgs(ts)
        return torch.cat((bg_gs, fg_gs), dim=0)

    def get_transforms(self, ts: torch.Tensor| None = None) -> torch.Tensor:
        # coefs = self.fg.get_coefs()  # (G, K)
        transls, rots, coefs = self.load_motion_base()
        transfms = compute_transforms(transls, rots, ts, coefs)  # (G, B, 3, 4)
        return transfms
    
    def load_image(self, index) -> torch.Tensor:
        path = f"{self.img_dir}/{self.frame_names[index]}{self.img_ext}"
        return torch.from_numpy(imageio.imread(path)).float() / 255.0
    
    def load_3dgs(self, set='fg') -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert set in ['fg', 'bg']
        means = self.ckpt["model"][f"{set}.params.means"]
        quats = self.ckpt["model"][f"{set}.params.quats"]
        scales = self.ckpt["model"][f"{set}.params.scales"]
        opacities = self.ckpt["model"][f"{set}.params.opacities"][:, None]
        # print('opacities loaded', opacities.shape)
        colors = self.ckpt["model"][f"{set}.params.colors"]
        colors = torch.nan_to_num(colors,nan=0.0, posinf=10, neginf=-1e1)
        if set == 'bg':
            return means, quats, means, quats, scales, opacities, colors
        else :
            return means, quats, scales, opacities, colors
    
    def load_3dgs_norm(self, set='fg') -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        norm_3dgs = ()
        for tensor in self.load_3dgs(set):
            norm_3dgs += self.min_max_norm(tensor)
        return norm_3dgs

    def load_motion_base(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        transls = self.ckpt["model"]["motion_bases.params.transls"]
        rots = self.ckpt["model"]["motion_bases.params.rots"]
        coefs = self.ckpt["model"]["fg.params.motion_coefs"]
        return transls, rots, coefs
    
    def min_max_norm(self, tensor: torch.Tensor) -> torch.Tensor:
        min_val = tensor.min(dim=0, keepdim=True)[0]
        max_val = tensor.max(dim=0, keepdim=True)[0]
        return (tensor - min_val) / (max_val - min_val + 1e-8)  # Avoid division by zero
    
    def __getitem__(self, index: int):
        data = {
            # ().
            # "frame_names": self.frame_names[index],
            # (H, W, 3).
            "gt_imgs": self.get_image(index),
            # (G, 14).
            "fg_gs": self.get_fg_4dgs(torch.tensor([index])),
            # (G, 14).
            "all_gs": self.get_all_4dgs(torch.tensor([index]))
        }

        return data
    

def collate_fn_padd(batch):
    """
    Pads batch of variable-length sequences and returns:
    - batch_fg: Padded tensor of shape (batch_size, max_G, 14) for fg_gs
    - batch_all: Padded tensor of shape (batch_size, max_G, 14) for all_gs
    - gt_imgs: A list of original images (not padded)
    - lengths_fg: Tensor containing original sequence lengths for fg_gs
    - lengths_all: Tensor containing original sequence lengths for all_gs
    - mask_fg: Boolean mask for valid elements in fg_gs
    - mask_all: Boolean mask for valid elements in all_gs
    """
    
    # Extract fg_gs, all_gs, and gt_imgs
    fg_gs = [torch.tensor(t['fg_gs'], dtype=torch.float32) for t in batch]
    all_gs = [torch.tensor(t['all_gs'], dtype=torch.float32) for t in batch]
    gt_imgs = [torch.tensor(t['gt_imgs'], dtype=torch.float32) for t in batch]  # Keep gt_imgs as is (no padding)
    gt_imgs = torch.stack(gt_imgs)

    # Get original lengths
    lengths_fg = torch.tensor([t.shape[0] for t in fg_gs], dtype=torch.int64)
    lengths_all = torch.tensor([t.shape[0] for t in all_gs], dtype=torch.int64)

    # Pad sequences along the first dimension (G)
    batch_fg = torch.nn.utils.rnn.pad_sequence(fg_gs, batch_first=True, padding_value=0.0)
    batch_all = torch.nn.utils.rnn.pad_sequence(all_gs, batch_first=True, padding_value=0.0)

    out = {
        "gt_imgs": gt_imgs,
        "fg_gs": batch_fg,
        "all_gs": batch_all,
        "fg_lengths": lengths_fg,
        "all_lengths": lengths_all,
    }

    # # Compute mask (True for valid values, False for padding)
    # mask_fg = (batch_fg != 0).any(dim=-1)
    # mask_all = (batch_all != 0).any(dim=-1)

    return out