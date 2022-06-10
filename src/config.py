import torch
import random
import string


def generate_id():
    n = 4
    id = ''.join(random.choice(string.ascii_uppercase) for _ in range(n))
    return id

# generate id and register it in config, gets executed once per run
id = generate_id()


def get_config():
    return {
        'id': id,
        'csv_name': 'metadata.csv',
        'root_dir': 'data/',
        'model_save_dir': 'models/' + id + '/',
        'model_save_name': 'model_' + id + '_{}.pth',
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
            'output_dim': 2
        },

        'batch_size': 64,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'num_epochs': 5,
        'margin': 0.5, # default is 0.05, increase to prevent underfitting, decrease to prevent overfitting
        'save_every_n_epochs': 5,
        'lr': 0.001,
    }