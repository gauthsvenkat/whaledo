import os
import matplotlib.pyplot as plt
import torch
import json
from config import get_config

config = get_config()


def save_losses(losses):
    print("Plotting losses")
    # Plot loss and accuracy
    # plt.figure(figsize=(14,4))
    # plt.subplot(1,3,1)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.plot(losses)
    plt.savefig(os.path.join(config['model_save_dir'])+"losses.png")

def save_model(model, version):
    print('Saving model', version,'...')
    torch.save(model, os.path.join(config['model_save_dir'], config['model_save_name'].format(version)))

def save_config():
    out_file = open(os.path.join(config['model_save_dir'])+"settings.json", "w") 
    json.dump(config, out_file, indent = 6) 
    out_file.close()