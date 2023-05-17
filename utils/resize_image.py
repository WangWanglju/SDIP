#%%

from joblib import Parallel, delayed
import numpy as np
from PIL import Image
import cv2
import pandas as pd
import os
import pickle
from tqdm import tqdm
from glob import glob
# define the paths to your image and label data
dir_paths = glob("./dataset/StableDiffusionDB/part*")
# train = pd.read_csv('/home/benny/SDIP/dataset/v2_eval.csv')

# prompts = train.prompt.values
# image_paths = train.image_name.apply(lambda x: os.path.join(image_path, x[4:]))

# image_paths = image_paths.values
image_paths = []
for i in dir_paths:
    image_paths.extend(glob(i+'/*png'))


def resize_image(img_path):
    # img_path = os.path.join(image_path, img_path)

    image_name = img_path.split("/")[-1].split('.')[0]
    # image_name = 'val_' + image_name

    if os.path.exists(os.path.join('/benny/SDIP_512', image_name+'.jpg')):
        return
    else:
        img = Image.open(img_path).resize((512, 512))
        img.save(os.path.join('/benny/SDIP_512', image_name+'.jpg'))
        return

_ = Parallel(n_jobs=8)(delayed(resize_image)(image_path.strip()) for image_path in tqdm(image_paths, total=len(image_paths)))




