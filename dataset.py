import os
import random
import json
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate

class PARTNET(Dataset):
    def __init__(self, root_dir, split='train'):
        super(PARTNET, self).__init__()
        
        assert split in ['train', 'val', 'test']
        self.split = split
        self.root_dir = root_dir
        self.files = os.listdir(self.root_dir)
        self.img_transform = transforms.Compose([
               transforms.ToTensor()])

    def __getitem__(self, index):
        path = self.files[index]
        image = Image.open(os.path.join(self.root_dir, path)).convert("RGB")
        image = image.resize((128 , 128))
        image = self.img_transform(image)
        sample = {'image': image}

        return sample
            
    
    def __len__(self):
        return len(self.files)

class MegaSaM(Dataset):
    def __init__(self, root_dir, split='train'):
        super(MegaSaM, self).__init__()
        
        assert split in ['train', 'val', 'test']
        self.split = split
        self.root_dir = root_dir
        self.megasam = np.load(root_dir)
        self.img_transform = transforms.Compose([
               transforms.ToTensor()])
        
    def load_img(self, index):
        image = Image.fromarray(self.megasam['images'][index])
        image = image.resize((128, 128))
        image = self.img_transform(image)
        return image

    def load_depth(self, index):
        depth = Image.fromarray(self.megasam['depths'][index])
        depth = depth.resize((128, 128))
        depth = self.img_transform(depth)
        return depth


    def __getitem__(self, index):
        sample = {
            'index': index,
            'root_dir': self.root_dir,
            'image': self.load_img(index),
            'rgbd': torch.concat([self.load_img(index), self.load_depth(index)], dim=0)
        }

        return sample
            
    
    def __len__(self):
        return len(self.megasam['images'])