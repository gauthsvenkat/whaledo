from sklearn.preprocessing import LabelEncoder
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_csv_and_parse_dataframe(csv_path, root_dir='data/', drop_columns=['image_id', 'timestamp', 'encounter_id']):
    df = pd.read_csv(csv_path).drop(columns=drop_columns) #drop useless columns

    #init label encoder and convert whale_id to categorical values
    le = LabelEncoder()
    labels = le.fit_transform(df['whale_id'])

    df['path'] = root_dir + df['path'] #convert path to full path
    df['viewpoint'] = df['viewpoint'].map({'top': 0, 'left': -1, 'right': 1}) #convert viewpoint to 0, -1, 1
    df['whale_id'] = labels #convert whale_id to categorical values

    return df, le

def get_avg_height_width(df, use_cache=True):

    if use_cache:
        return int(561.5631989156218), int(173.3659776347001)

    heights = df['height'].tolist()
    widths = df['width'].tolist()
    viewpoints = df['viewpoint'].tolist()

    height_tot, width_tot = 0, 0

    for h,w,v in zip(heights, widths, viewpoints):
        if v == 0:
            height_tot += h
            width_tot += w
        elif v == -1 or v == 1:
            height_tot += w
            width_tot += h
        
    return height_tot//len(heights), width_tot//len(widths)

def get_mean_and_std_of_dataset(df, use_cache=True):

    if use_cache:
        return np.array([118.3311038, 108.94562059, 107.9743398]), np.array([48.07930915, 44.44048544, 44.72591547])

    mean = np.array(
                list(map(_mean_of_image, df['path']))
                ).mean(axis=0)
    std = np.array(
                list(map(_std_of_image, df['path']))
                ).mean(axis=0)
    return mean, std

def _mean_of_image(path):
    img = cv2.imread(path)
    return np.mean(img, axis=(0,1))

def _std_of_image(path):
    img = cv2.imread(path)
    return np.std(img, axis=(0,1))

def visualize_augmentations(dataset, n_imgs=8):
    idxs = [np.random.randint(0, len(dataset)) for _ in range(n_imgs)]
    _, axes = plt.subplots(1, n_imgs)

    for i, idx in enumerate(idxs):
        img, _ = dataset.__getitem__(idx, normalize=False)
        axes[i].imshow(img[:3].permute((1, 2, 0)).int())

    plt.show()
