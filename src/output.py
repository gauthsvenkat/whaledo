import os
import matplotlib.pyplot as plt
import torch
import json
from config import get_config


# folder_names = filter(os.path.isdir, os.listdir(os.getcwd()))
# last_number = max([int(name) for name in folder_names if name.isnumeric()])
# new_name = str(last_number + 1).zpad(4)
# os.mkdir(new_name)

config = get_config()


def plot_losses(losses):
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

def log_settings():
    settings = {}
    # the json file where the output must be stored
    out_file = open("settings{}.json".format(id), "w")
    settings["id"] = id
    settings["config"] = config
    # dump settings
    json.dump(settings, out_file, indent = 6)
    out_file.close()