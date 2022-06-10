import torch
import string
from datetime import datetime


def generate_id():
    now = datetime.now()
    id = now.strftime("%m%d-%H%M")
    return id

# generate id and register it in config, gets executed once per run
id = generate_id()


def get_config():
    return {
        'id': id,
        'csv_name': 'metadata.csv',
        'root_dir': 'data/',
        'model_save_dir': 'models/' + id + '/',
        'model_save_name': 'model-{}.pth',
        'dataset': {
            'channels': 4,
            'height': None,
            'width': None,
            'mean': None,
            'std': None,
        },

        'backbone': {
            'model': 'resnet50',
            'rep_dim': 512,
            'pretrained': True,
        },

        'projector': {
            'hidden_dim': 1024,
            'output_dim': 256
        },

        'train_batch_size': 64,
        'main_batch_size': 32, # can't seem to handle larger than 32
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'num_epochs': 0,
        'margin': 0.1, # default is 0.05, increase to prevent underfitting, decrease to prevent overfitting
        'save_every_n_epochs': 5,
        'lr': 0.01,
    }