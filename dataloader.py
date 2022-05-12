import torch
from torchvision import transforms

import pandas as pd

class WhaleDoDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        
        '''
        https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files
        https://github.com/KevinMusgrave/pytorch-metric-learning/blob/master/examples/notebooks/TripletMarginLossMNIST.ipynb

        overload the __len__ and __getitem__ methods here appropriately

        transformations is an optional set of augmentations that we can do

        pick a decent height and width? (write a simple script to find the max width and height of the images)

        don't forget to rotate the left / right images!

        WE ALSO NEED TO NORMALIZE THE IMAGES (VERY IMPORTANT)
        '''

        pass