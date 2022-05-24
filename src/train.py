from pytorch_metric_learning import losses, miners
from dataloader import WhaleDoDataset
from models import WhaleDoModel
from torch.utils.data import DataLoader
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split

from utils import *

config = {
    'csv_path': 'data/metadata.csv',
    'root_dir': 'data/',

    'dataset' : {
        'channels' : 4,
        'height': None,
        'width': None,
        'mean': None,
        'std': None,
    },

    'backbone' : {
        'model': 'resnet18',
        'rep_dim': 512,
        'pretrained': True,
    },

    'projector' : {
        'hidden_dim': 1024,
        'output_dim': 2
    },

    'batch_size': 8,
    'device': 'cpu',
    'num_epochs': 1000,
    'margin': 0.2,
}

# load and parse the dataframe and get the label encoder as well
df, label_encoder = load_csv_and_parse_dataframe(config['csv_path'], root_dir=config['root_dir']) 
# get the average height and width of the dataset (rotate if viewpoint -1 or 1 (left / right))
config['dataset']['height'], config['dataset']['width'] = get_avg_height_width(df)
# get the mean and std of the dataset
config['dataset']['mean'], config['dataset']['std'] = get_mean_and_std_of_dataset(df)

# split the dataframe into train and test
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)#, stratify=df['whale_id']) #can't stratify if least populated class has only one member
# reset index of the dataframe (indices get fucked when you split)
train_df, test_df = train_df.reset_index(), test_df.reset_index()

# create the dataset objects
train_data, test_data = WhaleDoDataset(train_df, config, augmentations=True), WhaleDoDataset(test_df, config, augmentations=False)

# create dataloaders
train_loader = DataLoader(train_data, config['batch_size'], shuffle=True)
test_loader = DataLoader(test_data, config['batch_size'], shuffle=False)

# init loss function and a miner. The miner samples for training samples
loss_func = losses.TripletMarginLoss(margin=config['margin'])
miner = miners.TripletMarginMiner(margin=config['margin'])

# init model and optimizer
model = WhaleDoModel(config)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for i, (x_batch, y_batch) in enumerate(train_loader):

    #move tensors to device (gpu or cpu)
    x_batch, y_batch = x_batch.to(config['device']), y_batch.to(config['device'])

    #zero the parameter gradients
    optimizer.zero_grad()

    #compute embeddings
    embeddings = model(x_batch)

    #mine for hard pairs
    mined_pairs = miner(embeddings, y_batch)
    #compute loss
    loss = loss_func(embeddings, y_batch, mined_pairs)

    #calculate gradients
    loss.backward()
    #update weights
    optimizer.step()