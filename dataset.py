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

class ShapeOfMotion(Dataset):
    def __init__(self, data_dir, transform=None):        
        self.data_dir = data_dir
        self.ckpt = torch.load(f"{data_dir}/checkpoints/last.ckpt") # If RAM OOM, could try dynamic load.
        self.img_dir = f"{data_dir}/images/"
        self.img_ext = os.path.splitext(os.listdir(self.img_dir)[0])[1]
        self.frame_names = [os.path.splitext(p)[0] for p in sorted(os.listdir(self.img_dir))]
        self.imgs = [self.load_image(i) for i in len(self.frame_names)]
        # self.imgs: list[torch.Tensor | None] = [None for _ in self.frame_names]
        self.transform = transform

    def __len__(self):
        return 1
    
    def get_image(self, index) -> torch.Tensor:
        if self.imgs[index] is None:
            self.imgs[index] = self.load_image(index)
        img = cast(torch.Tensor, self.imgs[index])
        return img
    
    def get_fg_4dgs(self) -> torch.Tensor:
        means, quats, scales, opacities, colors = self.load_3dgs('fg')

        means_list = []
        quats_list = []
        for ts in len(self.frame_names):
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
            means_list.append(means_ts[:, 0])
            quats_list.append(quats_ts[:, 0])

        means = torch.cat([t for t in means_list], dim=1)
        quats = torch.cat([t for t in quats_list], dim=1)

        return torch.cat([self.min_max_norm(t) for t in (means, quats, opacities, colors)], dim=1)
    
    def get_all_4dgs(self):
        bg_gs = self.load_4dgs_bg()
        fg_gs = self.get_fg_4dgs()
        return torch.cat((bg_gs, fg_gs), dim=0)

    def get_fg_3dgs(self, ts: torch.Tensor) -> torch.Tensor:
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
        else:
            means_ts = means
            quats_ts = quats      

        return torch.cat([self.min_max_norm(t) for t in (means_ts, quats_ts, scales, opacities, colors)], dim=1)
    
    def get_all_3dgs(self, ts: torch.Tensor) -> torch.Tensor:
        bg_gs = torch.cat(self.load_3dgs_norm('bg'), dim=1)
        fg_gs = self.get_fg_3dgs(ts)
        return torch.cat((bg_gs, fg_gs), dim=0)
    
    def get_fg_3dgs_tfm(self, ts: torch.Tensor) -> torch.Tensor:
        means, quats, scales, opacities, colors = self.load_3dgs('fg')

        if ts is not None:
            transfms = self.get_transforms(ts)  # (G, B, 3, 4)
            transfms = transfms[:, 0]  # (G, 3, 4)
            transfms = transfms.reshape(transfms.size(0), -1)  # (G, 12)
        else:
            transfms = self.get_zero_transform(means.size(0))

        transfms = torch.tensor(transfms, dtype=torch.float32).to(means.device)

        return torch.cat([self.min_max_norm(t) for t in (means, quats, scales, opacities, colors, transfms)], dim=1)
    
    def get_all_3dgs_tfm(self, ts: torch.Tensor) -> torch.Tensor:
        bg_gs = torch.cat(self.load_3dgs_norm('bg'), dim=1)

        transfms = self.get_zero_transform(bg_gs.size(0))
        transfms = torch.tensor(transfms, dtype=torch.float32).to(bg_gs.device)

        bg_gs = torch.cat([bg_gs, self.min_max_norm(transfms)], dim=1)

        fg_gs = self.get_fg_3dgs_tfm(ts)

        return torch.cat((bg_gs, fg_gs), dim=0)

    def get_transforms(self, ts: torch.Tensor| None = None) -> torch.Tensor:
        # coefs = self.fg.get_coefs()  # (G, K)
        transls, rots, coefs = self.load_motion_base()
        transfms = compute_transforms(transls, rots, ts, coefs)  # (G, B, 3, 4)
        return transfms
    
    def get_zero_transform(self, size: int):
        transfms = np.eye(4)
        transfms = transfms[:-1].flatten()  # (12,)
        transfms = np.repeat(transfms[np.newaxis, :], size, axis=0) # (G, B, 3, 4)
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
        colors = torch.nan_to_num(colors, posinf=5, neginf=-5)
        return means, quats, scales, opacities, colors
    
    def load_3dgs_norm(self, set='fg') -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        norm_3dgs = []
        for tensor in self.load_3dgs(set):
            norm_3dgs.append(self.min_max_norm(tensor))
        return tuple(norm_3dgs)
    
    def load_4dgs_bg(self):
        means, quats, scales, opacities, colors = self.load_3dgs('fg')
        means_quats = torch.cat([self.min_max_norm(t) for t in (means, quats)], dim=1)
        means_quats = means_quats.repeat(1,24)
        opac_col = torch.cat([self.min_max_norm(t) for t in (opacities, colors)], dim=1)
               
        return torch.cat([means_quats, opac_col], dim=1)

    def load_motion_base(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        transls = self.ckpt["model"]["motion_bases.params.transls"]
        rots = self.ckpt["model"]["motion_bases.params.rots"]
        coefs = self.ckpt["model"]["fg.params.motion_coefs"]
        return transls, rots, coefs
    
    def min_max_norm(self, tensor: torch.Tensor) -> torch.Tensor:
        min_val = tensor.min()
        max_val = tensor.max()
        return (tensor - min_val) / (max_val - min_val + 1e-8)  # Avoid division by zero
    
    def __getitem__(self, index: int):
        data = {
            # ().
            # "frame_names": self.frame_names[index],
            # (H, W, 3).
            "gt_imgs": self.imgs,
            # (G, 14).
            "fg_gs": self.get_fg_4dgs(),
            # # (G, 14).
            "all_gs": self.get_all_4dgs()
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

    # Pad sequences along the first dimension (G)
    batch_fg = torch.nn.utils.rnn.pad_sequence(fg_gs, batch_first=True, padding_value=0.0)
    batch_all = torch.nn.utils.rnn.pad_sequence(all_gs, batch_first=True, padding_value=0.0)

    # # # Compute mask (True for valid values, False for padding)
    # fg_mask = (batch_fg != 0).any(dim=-1)
    # all_mask = (batch_all != 0).any(dim=-1)

    out = {
        "gt_imgs": gt_imgs,
        "fg_gs": batch_fg,
        "all_gs": batch_all,
    }

    return out