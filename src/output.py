import os
import matplotlib.pyplot as plt
import torch
import json
from config import get_config

config = get_config()

def save_losses(losses, test_losses, filename):
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(losses, label="Training Loss")
    plt.plot(test_losses, label="Validation Loss")
    plt.savefig(os.path.join(config['model_save_dir'])+filename)

def save_model(model, version):
    # print('Saving model', version,'...')
    torch.save(model, os.path.join(config['model_save_dir'], config['model_save_name'].format(version)))

def save_config():
    out_file = open(os.path.join(config['model_save_dir'])+"settings.json", "w") 
    json.dump(config, out_file, indent = 6) 
    out_file.close()