import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import gc
import time
import random
import joblib
import itertools
from sentence_transformers import SentenceTransformer
import torch
import torch.multiprocessing as mp
# torch.multiprocessing.set_start_method('spawn')
import os
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import sys
import open_clip
from tqdm import tqdm
import pandas as pd
from scipy import spatial
from config import CFG
# from dataset import DiffusionDataset, DiffusionCollator
from utils import seed_everything, AverageMeter, timeSince, get_logger, get_optimizer_params, SAM, llrd
from model_clip import define_model
# import wandb
import cv2
import warnings
warnings.filterwarnings('ignore')

def cosine_similarity(y_trues, y_preds):
    return [
        1 - spatial.distance.cosine(y_true, y_pred) 
        for y_true, y_pred in zip(y_trues, y_preds)
    ]

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
            # print(e)
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
    

def valid_fn(valid_loader, model, criterion, device):
    losses = AverageMeter()

    model.eval()

    all_coses = []
    start = end = time.time()
    for step, (inputs, labels) in enumerate(valid_loader):

        tokenizer = open_clip.get_tokenizer('ViT-L-14')
        inputs = inputs.to(device)
        labels = tokenizer(labels).to(device)

        batch_size = labels.size(0)
        with torch.no_grad():
            # y_preds = model(inputs)
            image_features = model.encode_image(inputs)
            labels = model.encode_text(labels)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            labels /= labels.norm(dim=-1, keepdim=True)
            # target = torch.ones(inputs.size(0)).to(device)

            # loss = criterion(y_preds, labels, target)
            # loss = criterion(y_preds, labels, batch_size, device)
        # if CFG.gradient_accumulation_steps > 1:
        #     loss = loss / CFG.gradient_accumulation_steps
        # losses.update(loss.item(), batch_size)
        val_cos = cosine_similarity(
                image_features.detach().cpu().numpy(), 
                labels.detach().cpu().numpy()
            )
        all_coses += val_cos



        end = time.time()
        if step % CFG.print_freq == 0 or step == (len(valid_loader)-1):
            LOGGER.info('EVAL: [{0}/{1}] '
                  'Elapsed {remain:s} '
                #   'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  'allocate momery: {memory:.2f}G'
                  .format(step, len(valid_loader),
                        #   loss=losses,
                          remain=timeSince(start, float(step+1)/len(valid_loader)),
                          memory= torch.cuda.max_memory_allocated() / 1024.0**3))

    
    return all_coses

def analysis(folds, save_path):
    
    LOGGER.info(f"========== analysis ==========")

    # ====================================================
    # loader
    # ====================================================
    train_folds = folds
    # train_folds = folds[folds['fold'] != fold].reset_index(drop=True)
    # valid_folds = folds[folds['fold'] == fold].reset_index(drop=True)

    # valid_v2_folds = pd.read_csv('/home/benny/SDIP/dataset/v2_val_date.csv')

    # valid_v2_new_folds = pd.read_csv('../data/val/final_eval.csv')
    # valid_v2_new_folds['image_name'] = valid_v2_new_folds.image_path.apply(lambda x: x.split('/')[-1])
    
    # if CFG.debug:
    #     valid_v2_new_folds = valid_v2_new_folds[:2000]

    valid_7_samples = pd.read_csv('../data/7samples/prompts.csv')
    valid_7_samples['image_name'] = valid_7_samples.imgId.apply(lambda x: x+'.png')
    

    train_dataset = DiffusionDataset(train_folds, path='', mode='valid')
    # valid_dataset = DiffusionDataset(valid_folds, path='/benny/SDIP/', mode='valid')
    # valid_v2_dataset = DiffusionDataset(valid_v2_folds, path='/benny/SDIP_v2/', mode='valid')
    # valid_v2_new_dataset = DiffusionDataset(valid_v2_new_folds, path='', mode='valid')
    valid_7samples_dataset = DiffusionDataset(valid_7_samples, path='../data/7samples/images/', mode='valid')
    
    collator = DiffusionCollator()

    train_loader = DataLoader(train_dataset,
                              batch_size=CFG.data_config['val_bs'],
                              shuffle=False,
                              num_workers=CFG.num_workers, 
                              pin_memory=True, 
                              drop_last=False,
                            #   collate_fn=collator
                              )

    
    # valid_v2_new_loader = DataLoader(valid_v2_new_dataset,
    #                         batch_size=CFG.data_config['val_bs'],
    #                         shuffle=False,
    #                         num_workers=CFG.num_workers,  
    #                         pin_memory=True, 
    #                         drop_last=False,
    #                         # collate_fn=collator
    #                         )


    valid_7sample_loader = DataLoader(valid_7samples_dataset,
                            batch_size=CFG.data_config['val_bs'],
                            shuffle=False,
                            num_workers=CFG.num_workers,  
                        #   pin_memory=True, 
                            drop_last=False,
                            # collate_fn=collator
                            )

    # ====================================================
    # model & optimizer
    # ====================================================
    # model = define_model(config=CFG)
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='laion2b_s32b_b82k')


    model.to(device)

    criterion =  nn.CosineEmbeddingLoss()

    # train
    cos = valid_fn(train_loader, model, criterion, device)

    # eval
    # avg_val_loss, cos = valid_fn(valid_loader, model, criterion, device)  
    # cos_new_v2 = valid_fn(valid_v2_new_loader, model, criterion, device)
    sample_cos_v2 = valid_fn(valid_7sample_loader, model, criterion, device)
        

        
    torch.cuda.empty_cache()
    gc.collect()
    train_folds['cos'] = cos
    # valid_v2_new_folds['cos'] = cos_new_v2
    valid_7_samples['cos'] = sample_cos_v2
    train_folds.to_csv(save_path, index=False)
    # valid_v2_new_folds.to_csv('analysis_val_v2_fold.csv', index=False)
    valid_7_samples.to_csv('analysis_7samples.csv', index=False)
    return 

#%%
if __name__ == '__main__':
    # torch.multiprocessing.set_start_method('spawn')
    
    parser = argparse.ArgumentParser(description='Analysis Script')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

    train = pd.read_csv('/root/autodl-tmp/data/all_available_v2-5-11.csv')
           

    print('train shape:', len(train))


    #preprocessing
    OUTPUT = './exp/'
    exp_dir = OUTPUT + f'{CFG.exp}-{CFG.model.replace("/","-")}'
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    LOGGER = get_logger(filename=os.path.join(exp_dir,'eval'))
    seed_everything(42)
    LOGGER.info(f"========== config: \n{CFG.__dict__ } ")

    CFG.epochs = 1
    if CFG.debug:
        train = train[:1000]
        
    #need specifiy
    save_path= './save_train.csv'
    cos = analysis(train, save_path)

