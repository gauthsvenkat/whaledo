import torch
from utils import get_avg_height_width, get_mean_and_std_of_dataset


def get_config():
    config = {
        'csv_path': 'data/metadata.csv',
        'root_dir': 'data/',

        'dataset': {
            'channels': 4,
            'height': None,
            'width': None,
            'mean': None,
            'std': None,
        },

        'backbone': {
            'model': 'resnet18',
            'rep_dim': 512,
            'pretrained': True,
        },

        'projector': {
            'hidden_dim': 1024,
            'output_dim': 2
        },

        'batch_size': 32,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'num_epochs': 1000,
        'margin': 0.4,
        'save_every': 100,
        'lr': 0.001,
        'model_save_dir': 'models/',
        'model_save_name': 'whaledo_model_{}.pth',
    }

    config['dataset']['height'], config['dataset']['width'] = get_avg_height_width(None)
    # get the mean and std of the dataset
    config['dataset']['mean'], config['dataset']['std'] = get_mean_and_std_of_dataset(None)

    return config