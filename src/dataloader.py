import torch
from torchvision import transforms

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder

class WhaleDoDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.le = LabelEncoder()
        self.labels = self.le.fit_transform(self.df['whale_id'])
        self.transform = transform
        
        '''
        https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files
        https://github.com/KevinMusgrave/pytorch-metric-learning/blob/master/examples/notebooks/TripletMarginLossMNIST.ipynb

        overload the __len__ and __getitem__ methods here appropriately

        transformations is an optional set of augmentations that we can do

        pick a decent height and width? (write a simple script to find the max width and height of the images) 443, 291

        don't forget to rotate the left / right images!

        WE ALSO NEED TO NORMALIZE THE IMAGES (VERY IMPORTANT)
        '''

        pass

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        #just generate a random image and label for testing purposes
        # do transformations here which is not implemented cause I'm just testing right now

        return torch.rand((3,443,291)), self.labels[idx]
