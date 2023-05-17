from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
from PIL import Image
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
import torch.multiprocessing as mp
# torch.multiprocessing.set_start_method('spawn')
import os
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2
from config import CFG
import time
import open_clip

def get_transforms(mode):
    if mode == "train":
        return A.Compose([
            # openclip augment: https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/transform.py#L59
            A.RandomResizedCrop(height=CFG.input_size, width=CFG.input_size,
                                scale=(0.9, 1.0), # openclip scale
                                ratio=(1.0, 1.0), # Avoid destroying the aspect ratio of the original image
                               ),
            A.HorizontalFlip(p=0.5),
            A.HueSaturationValue(
                hue_shift_limit=0.2, 
                sat_shift_limit=0.2, 
                val_shift_limit=0.2, 
                p=0.5
            ),
            A.RandomBrightnessContrast(
                    brightness_limit=(-0.1,0.1), 
                    contrast_limit=(-0.1, 0.1), 
                    p=0.5
                ),
            A.Normalize(mean=CFG.data_mean, std=CFG.data_std, max_pixel_value=255.0, p=1.0),
            ToTensorV2(),
        ])

    elif mode == "valid":
        return A.Compose([
            A.Resize(CFG.input_size, CFG.input_size),
            A.Normalize(mean=CFG.data_mean, std=CFG.data_std, max_pixel_value=255.0, p=1.0),
            ToTensorV2(),
        ])

class DiffusionDataset(Dataset):
    def __init__(self, df, path, mode='train'):
        """
        df:csv
        path:图像保存路径
        mode:train or val/test
        """
        self.df = df
        
        self.transform = get_transforms(mode=mode)
        self.mode = mode
        self.path = path
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # while True:

        row = self.df.iloc[idx]
        img_path = os.path.join(self.path, row['image_name'])
        try:        
            image = cv2.imread(img_path)[..., ::-1]
        except Exception as e:
            print(img_path)
            image = cv2.imread(img_path.replace('.jpg', '.png'))[..., ::-1]
        # image = Image.open(img_path)
        # print(image.shape)
        image = self.transform(image = image)["image"]

        if self.mode != 'test':
            prompt = row['prompt']
            return image, prompt
        else:
            return image



class DiffusionCollator:
    def __init__(self):
        self.st_model = SentenceTransformer(
            'all-MiniLM-L6-v2',
            device='cpu'
        )
    
    def __call__(self, batch):
        
        images, prompts = zip(*batch)
        images = torch.stack(images)
        prompt_embeddings = self.st_model.encode(
            prompts, 
            show_progress_bar=False, 
            convert_to_tensor=True,
            # device="cuda"
        )
        
        return images, prompt_embeddings
    
class ImageCaptioningDataset(Dataset):
    def __init__(self, dataset, path, processor):
        self.df = dataset
        self.path = path
        self.processor = processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.path, row['image_name']).replace('.png', '.jpg')
        
        try:
            image = cv2.imread(img_path)[..., ::-1]
        except Exception as e:
            image = cv2.imread(img_path.replace('.jpg', '.png'))[..., ::-1]
        encoding = self.processor(images=image, text=row["prompt"], 
                                  padding="max_length", return_tensors="pt")
        # remove batch dimension
        encoding = {k:v.squeeze() for k,v in encoding.items()}
        return encoding
            
if __name__ == "__main__":
    # mp.set_start_method('spawn')
    from tqdm import tqdm
    train = pd.read_csv('../data/train_data_v1/DB2M-0.8.csv')
    extra_train = pd.read_csv('../data/train_data_v1/extra_train_data.csv')
    train = pd.concat([train, extra_train])
    path = '/root/autodl-tmp/data/train_data_v1/USEFUL_TRAIN_v1/'
    dataset = DiffusionDataset(train, path=path, mode='train')
    collator = DiffusionCollator()
    
    dataloaders = DataLoader(
        dataset=dataset,
        shuffle=True,
        batch_size=4,
        pin_memory=True,
        num_workers=2,
        drop_last=True,
        collate_fn=collator
    )

    for X ,y in tqdm(dataset, total=len(dataset)):
        # print(X.shape, y)
        pass
        # break