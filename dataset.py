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
    def __init__(self, data_dir, num_slot,transform=None):        
        self.data_dir = data_dir
        self.ckpt = torch.load(f"{data_dir}/checkpoints/last.ckpt") # If RAM OOM, could try dynamic load.
        self.img_dir = f"{data_dir}/images/"
        self.img_ext = os.path.splitext(os.listdir(self.img_dir)[0])[1]
        self.frame_names = [os.path.splitext(p)[0] for p in sorted(os.listdir(self.img_dir))]
        self.imgs: list[torch.Tensor | None] = [None for _ in self.frame_names]
        self.num_slot = num_slot
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
        
    def get_fg_4dgs(self, ts: torch.Tensor, is_norm=True) -> torch.Tensor:
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
        if is_norm:
            return torch.cat([self.min_max_norm(t) for t in (means_ts, quats_ts, scales, opacities, colors)], dim=1)
        else:
            return torch.cat((means_ts, quats_ts, scales, opacities, colors), dim=1)
    
    def get_all_4dgs(self, ts: torch.Tensor) -> torch.Tensor:
        bg_gs = torch.cat(self.load_3dgs_norm('bg'), dim=1)
        fg_gs = self.get_fg_4dgs(ts)
        return torch.cat((bg_gs, fg_gs), dim=0)
    
    def get_all_4dgs_raw(self, ts: torch.Tensor) -> torch.Tensor:
        bg_gs = torch.cat(self.load_3dgs('bg'), dim=1)
        fg_gs = self.get_fg_4dgs(ts, is_norm=False)
        return torch.cat((bg_gs, fg_gs), dim=0)
    
    def get_fg_4dgs_tfm(self, ts: torch.Tensor) -> torch.Tensor:
        means, quats, scales, opacities, colors = self.load_3dgs('fg')

        if ts is not None:
            transfms = self.get_transforms(ts)  # (G, B, 3, 4)
            transfms = transfms[:, 0]  # (G, 3, 4)
            transfms = transfms.reshape(transfms.size(0), -1)  # (G, 12)
        else:
            transfms = np.eye(4)
            transfms = transfms[:-1].flatten()  # (12,)
            transfms = np.repeat(transfms[np.newaxis, :], means.size(0), axis=0)

        transfms = torch.tensor(transfms, dtype=torch.float32).to(means.device)

        return torch.cat([self.min_max_norm(t) for t in (means, quats, scales, opacities, colors, transfms)], dim=1)
    
    def get_all_4dgs_tfm(self, ts: torch.Tensor) -> torch.Tensor:
        bg_gs = torch.cat(self.load_3dgs_norm('bg'), dim=1)

        transfms = np.eye(4)
        transfms = transfms[:-1].flatten()  # (12,)
        transfms = np.repeat(transfms[np.newaxis, :], bg_gs.size(0), axis=0)
        transfms = torch.tensor(transfms, dtype=torch.float32).to(bg_gs.device)

        bg_gs = torch.cat([bg_gs, self.min_max_norm(transfms)], dim=1)

        fg_gs = self.get_fg_4dgs_tfm(ts)

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
        colors = torch.nan_to_num(colors, posinf=0, neginf=0)
        return means, quats, scales, opacities, colors
    
    def load_3dgs_norm(self, set='fg') -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        norm_3dgs = []
        for tensor in self.load_3dgs(set):
            norm_3dgs.append(self.min_max_norm(tensor))
        return tuple(norm_3dgs)

    def load_motion_base(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        transls = self.ckpt["model"]["motion_bases.params.transls"]
        rots = self.ckpt["model"]["motion_bases.params.rots"]
        coefs = self.ckpt["model"]["fg.params.motion_coefs"]
        return transls, rots, coefs
    
    def min_max_norm(self, tensor: torch.Tensor) -> torch.Tensor:
        min_val = tensor.min()
        max_val = tensor.max()
        return (tensor - min_val) / (max_val - min_val + 1e-8)  # Avoid division by zero
    
    def kmeans(self, x, k, num_iterations=10):
        # x: input tensor of shape [num_gaussians, features]
        # k: number of clusters (slots)
        # num_iterations: number of iterations for convergence
        
        num_points, num_features = x.shape
        
        # Initialize centroids randomly by choosing k random points from x
        centroids = x[torch.randperm(num_points)[:k]]
        
        for _ in range(num_iterations):
            # Step 1: Compute the distances from each point to each centroid
            distances = torch.cdist(x, centroids)
            
            # Step 2: Assign each point to the nearest centroid (min distance)
            assignments = torch.argmin(distances, dim=1)
            
            # Step 3: Update centroids by computing the mean of the assigned points
            for i in range(k):
                centroids[i] = x[assignments == i].mean(dim=0)
        
        # Return the final centroids (slots) and the assignments (cluster memberships)
        return torch.tensor(centroids, device=x[0].device)
    
    def __getitem__(self, index: int):

        fg_gs = self.get_fg_4dgs(torch.tensor([index]), is_norm=False)
        # torch.set_printoptions(sci_mode=False)
        # np.savetxt("tensor.csv", fg_gs.numpy(), delimiter=",", fmt="%.10f")
        # exit()
        all_gs = self.get_all_4dgs_raw(torch.tensor([index]))

        data = {
            # ().
            # "frame_names": self.frame_names[index],
            # (H, W, 3).
            "gt_imgs": self.get_image(index),
            # (G, 14).
            "fg_gs": fg_gs,
            # (G, 14).
            "all_gs": all_gs,
            # # (Num_slots, 14).
            # "fg_kmeans": self.kmeans(fg_gs, self.num_slot),
            # # (Num_slots, 14).
            # "all_kmeans": self.kmeans(all_gs, self.num_slot),
        }
        return data
    def modify_ckpt(self, color):
        new_ckpt = self.ckpt

        bg_size = new_ckpt["model"]["bg.params.colors"].shape[0]
        fg_size = new_ckpt["model"]["fg.params.colors"].shape[0]

        # bg_col, fg_col = torch.split(color, [bg_size, fg_size], dim=0)
        # new_ckpt["model"]["bg.params.colors"] = bg_col
        new_ckpt["model"]["fg.params.colors"] = color
        new_ckpt["model"]["fg.params.opacities"] = -color[:,0]
        new_ckpt["model"]["bg.params.opacities"] = torch.ones_like(new_ckpt["model"]["bg.params.opacities"]) * -10

        # print(fg_col[fg_col != 0])

        return new_ckpt

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
    gt_imgs = [torch.tensor(t['gt_imgs'], dtype=torch.float32) for t in batch]  # Keep gt_imgs as is (no padding)
    gt_imgs = torch.stack(gt_imgs)
    # fg_kmeans = [torch.tensor(t['fg_kmeans'], dtype=torch.float32) for t in batch]
    # fg_kmeans = torch.stack(fg_kmeans)
    # all_kmeans = [torch.tensor(t['all_kmeans'], dtype=torch.float32) for t in batch]
    # all_kmeans = torch.stack(all_kmeans)
    
    # Extract fg_gs, all_gs, and gt_imgs
    fg_gs = [torch.tensor(t['fg_gs'], dtype=torch.float32) for t in batch]
    all_gs = [torch.tensor(t['all_gs'], dtype=torch.float32) for t in batch]

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
        # "fg_kmeans": fg_kmeans,
        # "all_kmeans": all_kmeans,
    }

    return out


