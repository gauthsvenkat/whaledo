from output import save_losses, save_model, save_config
from pytorch_metric_learning import losses, miners
from config import get_config
from dataloader import WhaleDoDataset
from models import WhaleDoModel
from torch.utils.data import DataLoader
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
import os
import time
from tqdm import tqdm

from utils import *

# load config file
config = get_config()

# load and parse the dataframe and get the label encoder as well
df, label_encoder = load_csv_and_parse_dataframe(config['csv_name'], root_dir=config['root_dir'], drop_columns=['timestamp', 'encounter_id'])
# get the average height and width of the dataset (rotate if viewpoint -1 or 1 (left / right))
config['dataset']['height'], config['dataset']['width'] = get_avg_height_width(df)
# get the mean and std of the dataset
config['dataset']['mean'], config['dataset']['std'] = get_mean_and_std_of_dataset(df)

# split the dataframe into train and test
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)#, stratify=df['whale_id']) #can't stratify if least populated class has only one member

# create the dataset objects
train_data, test_data = WhaleDoDataset(train_df, config, mode='train'), WhaleDoDataset(test_df, config, mode='test')

# create dataloaders
train_loader = DataLoader(train_data, config['train_batch_size'], shuffle=True)
test_loader = DataLoader(test_data, config['train_batch_size'], shuffle=False)

# init loss function and a miner. The miner samples for training samples
loss_func = losses.TripletMarginLoss(margin=config['margin'])
miner = miners.TripletMarginMiner(margin=config['margin'])


device = config['device']
# init model and optimizer
model = WhaleDoModel(config)
optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
model.to(device)


# create directories if they don't exist
os.makedirs(config['model_save_dir'], exist_ok=True)

start = time.time()

losses = []

for epoch in tqdm(range(config['num_epochs']), desc="Epochs", position=0):
    for i, batch in tqdm(enumerate(train_loader), desc="Batches", position=1, leave=False):

        #move tensors to device (gpu or cpu)
        x_batch, y_batch = batch['image'].to(config['device']), batch['label'].to(config['device'])

        #set the gradients to zero
        optimizer.zero_grad()

        #compute embeddings
        embeddings = model(x_batch)

        #mine for hard pairs
        mined_pairs = miner(embeddings, y_batch)
        #compute loss
        loss = loss_func(embeddings, y_batch, mined_pairs)
        losses.append(loss)

        #calculate gradients
        loss.backward()
        #update weights
        optimizer.step()

    #save every n epochs
    if epoch % config['save_every_n_epochs'] == 0:
        save_model(model, epoch)

    print('Epoch: {}/{}'.format(epoch+1, config['num_epochs']), 'Loss: {:.4f}'.format(loss.item()))

# Plot losses and save last model
save_config()
save_losses(losses)
save_model(model, "final")


time_elapsed = time.time() - start
print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')