import torch
from torchvision import transforms

import pandas as pd
import numpy as np

import cv2

import tensorflow as tf

class WhaleDoDataset(torch.utils.data.Dataset):
    def __init__(self, df, config, augmentations):
        self.df = df
        self.augmentations = augmentations
        self.config = config
        self.normalize = transforms.Normalize(mean=self.config['dataset']['mean'], std=self.config['dataset']['std'])

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx, normalize=True):

        img = cv2.imread(self.df['path'][idx])
        label = self.df['whale_id'][idx]

        if self.df['viewpoint'][idx] == -1: # left of the whale
            assert img.shape[0] < img.shape[1]
            img = np.rot90(img, k = 1, axes = (0, 1))
        
        elif self.df['viewpoint'][idx] == 1: # right of the whale
            assert img.shape[0] < img.shape[1]
            img = np.rot90(img, k = -1, axes = (0, 1))

        if self.augmentations: # apply augmentations (train only)
            img = tf.image.random_brightness(img, 0.2)

            img = tf.image.random_contrast(img, 0.8, 1.2)

            random_crop_scale = np.random.uniform(0.75, 1)
            crop_size = (int(img.shape[0]*random_crop_scale), int(img.shape[1]*random_crop_scale), img.shape[2])
            img = tf.image.random_crop(img, crop_size)

        img = tf.image.resize_with_pad(img, self.config['dataset']['height'], self.config['dataset']['width']).numpy()  # resize to fixed size with padding

        img = torch.tensor(img.transpose((2, 0, 1))) # convert to tensor and transpose to (C, H, W)

        if normalize: # normalize to [-1, 1] only false when trying to visualize
            img = self.normalize(img)

        if self.config['dataset']['channels'] == 4: # add viewpoint as the 4th channel
            viewpoint_mask = torch.full((1, self.config['dataset']['height'],self.config['dataset']['width']),
                                         self.df['viewpoint'][idx],
                                         dtype=torch.float32)
            img = torch.cat((img, viewpoint_mask), dim=0)

        if self.config['dataset']['channels'] == 5: # add date information as the 5th channel
            pass #figure out how to add date information

        return img, label