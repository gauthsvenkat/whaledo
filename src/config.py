import torch

def get_config():
    return {
        'csv_name': 'metadata.csv',
        'root_dir': 'data/',

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

        'batch_size': 32,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'num_epochs': 50,
        'margin': 0.4,
        'save_every_n_epochs': 5,
        'lr': 0.001,
        'model_save_dir': 'models/',
        'model_save_name': 'whaledo_model_{}.pth',
    }