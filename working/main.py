import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import gc
import time
import random
import joblib
import itertools

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.optim import Adam, SGD, AdamW
from torch.utils.data import DataLoader

from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import sys

from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import pandas as pd

from config import CFG
from dataset import DiffusionDataset, DiffusionCollator
from utils import seed_everything, AverageMeter, cosine_similarity, timeSince, get_logger, get_optimizer_params, SAM, llrd
from model_clip import define_model
# from model import build_model
# import wandb
import argparse
import warnings
warnings.filterwarnings('ignore')



def train_fn(fold, train_loader, model, criterion, optimizer, epoch, scheduler, device):
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=CFG.apex)
    losses = AverageMeter()
    cos = AverageMeter()
    start = end = time.time()
    global_step = 0
    for step, (inputs, labels) in enumerate(train_loader):
        
        inputs = inputs.to(device)
        labels = labels.to(device)

        batch_size = labels.size(0)
        target = torch.ones(inputs.size(0)).to(device)

        with torch.cuda.amp.autocast(enabled=CFG.apex):
            y_preds = model(inputs)

            loss = criterion(y_preds, labels, target)
            # loss = criterion(y_preds.softmax(dim=-1).log(), labels.softmax(dim=-1))
            # loss = criterion(y_preds, labels, batch_size, device)

        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
        losses.update(loss.item(), batch_size)
        
        trn_cos = cosine_similarity(
                y_preds.detach().cpu().numpy(), 
                labels.detach().cpu().numpy()
            )

        cos.update(trn_cos, batch_size)

        scaler.scale(loss).backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.optimizer_config['max_grad_norm'])
        if (step + 1) % CFG.gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            global_step += 1
            if CFG.batch_scheduler:
                scheduler.step()
        
        end = time.time()
        if step % CFG.print_freq == 0 or step == (len(train_loader)-1):
            LOGGER.info('Epoch: [{0}][{1}/{2}] '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  "cos: {cos.val:.4f}({cos.avg:.4f}) "
                  'Grad: {grad_norm:.4f}  '
                  'LR: {lr:.8f}  '
                  'allocate momery: {memory:.2f}G'
                  .format(epoch+1, step, len(train_loader), 
                          remain=timeSince(start, float(step+1)/len(train_loader)),
                          loss=losses,
                          cos=cos,
                          grad_norm=grad_norm,
                          lr=scheduler.get_lr()[0],
                          memory= torch.cuda.max_memory_allocated() / 1024.0**3))
        # if CFG.wandb and not CFG.debug:
        #     wandb.log({f"[fold{fold}] loss": losses.val,
        #                f"[fold{fold}] lr": scheduler.get_lr()[0]})
    return losses.avg


def valid_fn(valid_loader, model, criterion, device):
    losses = AverageMeter()
    cos = AverageMeter()
    model.eval()

    start = end = time.time()
    for step, (inputs, labels) in enumerate(valid_loader):

        inputs = inputs.to(device)
        labels = labels.to(device)

        batch_size = labels.size(0)
        with torch.no_grad():
            y_preds = model(inputs)
            target = torch.ones(inputs.size(0)).to(device)

            loss = criterion(y_preds, labels, target)
            # loss = criterion(y_preds.softmax(dim=-1).log(), labels.softmax(dim=-1))
            # loss = criterion(y_preds, labels, batch_size, device)
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
        losses.update(loss.item(), batch_size)
        val_cos = cosine_similarity(
                y_preds.detach().cpu().numpy(), 
                labels.detach().cpu().numpy()
            )
        cos.update(val_cos, batch_size)


        end = time.time()
        if step % CFG.print_freq == 0 or step == (len(valid_loader)-1):
            LOGGER.info('EVAL: [{0}/{1}] '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  "Cosine: {cos.val:.4f}({cos.avg:.4f}) "
                  'allocate momery: {memory:.2f}G'
                  .format(step, len(valid_loader),
                          loss=losses,
                          cos=cos,
                          remain=timeSince(start, float(step+1)/len(valid_loader)),
                          memory= torch.cuda.max_memory_allocated() / 1024.0**3))

    
    return losses.avg, cos.avg

# ====================================================
# scheduler
# ====================================================
def get_scheduler(cfg, optimizer, num_train_steps):
    if cfg.scheduler == 'linear':
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=max(100, cfg.warmup_ratio*num_train_steps), num_training_steps=num_train_steps
        )
    elif cfg.scheduler == 'cosine':
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=max(100, cfg.warmup_ratio*num_train_steps), num_training_steps=num_train_steps, num_cycles=cfg.num_cycles
        )
    return scheduler


def cosine_similarity_loss(pred, target):
    cos = nn.CosineSimilarity(dim=1)
    output = -cos(pred, target).mean()
    return output
# ====================================================
# train loop
# ====================================================

def train_by_epochs(folds, fold, data_version):
    
    LOGGER.info(f"========== fold: {fold} training ==========")

    # ====================================================
    # loader
    # ====================================================
    train_folds = folds
    # train_folds = folds[folds['fold'] != fold].reset_index(drop=True)
    # valid_folds = folds[folds['fold'] == fold].reset_index(drop=True)

    valid_v2_folds = pd.read_csv('/root/autodl-tmp/data/eval_laion_with_scores.csv')
    valid_v2_folds = valid_v2_folds[valid_v2_folds.cos > 0.3]
    print(valid_v2_folds.shape)
    valid_v2_new_folds = pd.read_csv('../data/val/final_eval.csv')
    # valid_v2_new_folds = pd.read_csv('/root/autodl-tmp/working/analysis_train_fold.csv')
    # valid_v2_new_folds = valid_v2_new_folds[valid_v2_new_folds.cos >= 0.3]
    # valid_v2_new_folds['image_name'] = valid_v2_new_folds.image_path.apply(lambda x: x.split('/')[-1])
    
    if CFG.debug:
        valid_v2_new_folds = valid_v2_new_folds[:1000]

    valid_7_samples = pd.read_csv('../data/7samples/prompts.csv')
    valid_7_samples['image_name'] = valid_7_samples.imgId.apply(lambda x: x+'.png')
    
    

    train_dataset = DiffusionDataset(train_folds, path='', mode='train')
    # valid_dataset = DiffusionDataset(valid_folds, path='/benny/SDIP/', mode='valid')
    valid_v2_dataset = DiffusionDataset(valid_v2_folds, path='', mode='valid')
    valid_v2_new_dataset = DiffusionDataset(valid_v2_new_folds, path='', mode='valid')
    valid_7samples_dataset = DiffusionDataset(valid_7_samples, path='../data/7samples/images/', mode='valid')
    
    collator = DiffusionCollator()

    train_loader = DataLoader(train_dataset,
                              batch_size=CFG.data_config['batch_size'],
                              shuffle=True,
                              num_workers=CFG.num_workers, 
                              pin_memory=True, 
                              drop_last=True,
                              collate_fn=collator)
    # valid_loader = DataLoader(valid_dataset,
    #                           batch_size=CFG.data_config['val_bs'],
    #                           shuffle=False,
    #                           num_workers=CFG.num_workers,  
    #                         #   pin_memory=True, 
    #                           drop_last=False,
    #                           collate_fn=collator)

    valid_v2_loader = DataLoader(valid_v2_dataset,
                              batch_size=CFG.data_config['val_bs'],
                              shuffle=False,
                              num_workers=CFG.num_workers,  
                              pin_memory=True, 
                              drop_last=False,
                              collate_fn=collator)
    
    valid_v2_new_loader = DataLoader(valid_v2_new_dataset,
                            batch_size=CFG.data_config['val_bs'],
                            shuffle=False,
                            num_workers=CFG.num_workers,  
                            pin_memory=True, 
                            drop_last=False,
                            collate_fn=collator)


    valid_7sample_loader = DataLoader(valid_7samples_dataset,
                            batch_size=CFG.data_config['val_bs'],
                            shuffle=False,
                            num_workers=CFG.num_workers,  
                        #   pin_memory=True, 
                            drop_last=False,
                            collate_fn=collator)

    # ====================================================
    # model & optimizer
    # ====================================================
    model = define_model(config=CFG)
    # model = build_model(    
    #     num_classes=384,
    #     pretrained_weights='vit_base_patch16_224',
    #     weight_path = None, 
    #     pretrained=True,)
    model.to(device)
    if len(CFG.device_ids) > 1:
        model = nn.DataParallel(model, device_ids=CFG.device_ids)  # Data Parallel
    

    # print(model)
    optimizer_parameters = model.parameters()
    # optimizer_parameters = get_optimizer_params(model, encoder_lr=CFG.optimizer_config['lr'], weight_decay=CFG.optimizer_config['weight_decay'], \
    #                             decoder_lr=CFG.optimizer_config['lr']*CFG.expand_rate)
    optimizer_parameters = llrd(CFG, model, encoder_lr=CFG.optimizer_config['lr'],  \
                                decoder_lr=CFG.optimizer_config['lr']*CFG.expand_rate, weight_decay=CFG.optimizer_config['weight_decay'], layerwise_learning_rate_decay=CFG.layerwise_learning_rate_decay)
    
    # optimizer = AdamW(param_dicts)
    optimizer = AdamW(optimizer_parameters, lr=CFG.optimizer_config['lr'], \
                      eps=CFG.optimizer_config['eps'], betas=CFG.optimizer_config['betas'])
    # print(optimizer.param_groups)
    # optimizer = optimizer = SAM(model.parameters(), AdamW, lr=CFG.optimizer_config['lr'], weight_decay=CFG.optimizer_config['weight_decay'], rho=0.05)
    num_train_steps = int(len(train_loader) * CFG.epochs)
    
    # scheduler = get_scheduler(CFG, optimizer, num_train_steps)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_train_steps, eta_min=1e-6)
    # ====================================================
    # loop
    # ====================================================

    criterion =  nn.CosineEmbeddingLoss()
    
    # criterion =  nn.KLDivLoss(reduction="batchmean")
    # criterion =  ClipLoss
    best_score = -np.inf
    best_epoch = 0

    for epoch in range(CFG.epochs):

        start_time = time.time()

        # if epoch < 1:
        #     for k, param in model.named_parameters():
        #         if 'encoder' in k:
        #             param.requires_grad = False
        # else:
        #     trainable_model_weights = False
        #     for k, param in model.named_parameters():
        #         if CFG.unfreeze_start and str(CFG.unfreeze_start) in k:
        #             trainable_model_weights = True
        #         param.requires_grad = trainable_model_weights
        #         if param.requires_grad:
        #             print(f"{k} is set to be trainable.")
        # train
        avg_loss = train_fn(fold, train_loader, model, criterion, optimizer, epoch, scheduler, device)

        # eval
        # avg_val_loss, cos = valid_fn(valid_loader, model, criterion, device)  
        v2_avg_val_loss, cos_v2 = valid_fn(valid_v2_loader, model, criterion, device)
        v2_new_avg_val_loss, cos_new_v2 = valid_fn(valid_v2_new_loader, model, criterion, device)
        sample_avg_val_loss, sample_cos_v2 = valid_fn(valid_7sample_loader, model, criterion, device)
        
        # scoring
        elapsed = time.time() - start_time
        # LOGGER.info(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {v2_avg_val_loss:.4f}  time: {elapsed:.0f}s')
        # LOGGER.info(f'Epoch {epoch+1} - cos similiarity:{cos_v2:.4f}')

        LOGGER.info(f'Epoch {epoch+1} - v2_avg_val_loss: {v2_avg_val_loss:.4f}    cos similiarity:{cos_v2:.4f}')
        LOGGER.info(f'Epoch {epoch+1} - new_v2_avg_val_loss: {v2_new_avg_val_loss:.4f}    cos similiarity:{cos_new_v2:.4f}')
        LOGGER.info(f'Epoch {epoch+1} - 7sample_avg_val_loss: {sample_avg_val_loss:.4f}    cos similiarity:{sample_cos_v2:.4f}')
        
        # if CFG.wandb:
        #     wandb.log({f"[fold{fold}] epoch": epoch+1, 
        #                f"[fold{fold}] avg_train_loss": avg_loss, 
        #             #    f"[fold{fold}] avg_val_loss": avg_val_loss,
        #             #    f"[fold{fold}] cos": cos,

        #                 })
        
        if best_score < cos_v2:
            best_score = cos_v2
            best_epoch = epoch
            LOGGER.info(f'Epoch {epoch+1} - Save Best cos: {cos_v2:.4f} Model')
            torch.save({'model': model.state_dict(),
                        
                        'cos':cos_v2},
                        exp_dir+f"/{CFG.model.replace('/', '-')}_fold{fold}_best.pth")   

        if best_epoch+2 < epoch:
            print(f"early stop at {epoch+1} with best epoch {best_epoch} and test similarity {best_score}.")
            break
    torch.cuda.empty_cache()
    gc.collect()
    
    return best_score

#%%
if __name__ == '__main__':
    # torch.multiprocessing.set_start_method('spawn')
    
    parser = argparse.ArgumentParser(description='Training Script')

    # parser.add_argument('--gpu', type=int, default=0, help='Specify the GPU to use')
    parser.add_argument('--data', type=str, default="v1", help='Specify the data version to train')
    # parser.add_argument("--local_rank", type=int)

    args = parser.parse_args()

    # gpu_id = args.gpu
    data_version = args.data
    # torch.cuda.set_device(args.local_rank)  # before your code runs

    # torch.distributed.init_process_group(backend='nccl')

    if len(CFG.device_ids) > 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(_) for _ in CFG.device_ids])
        print("os.environ['CUDA_VISIBLE_DEVICES']:", os.environ["CUDA_VISIBLE_DEVICES"])
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(CFG.device_ids[0])
        print("os.environ['CUDA_VISIBLE_DEVICES']:", os.environ["CUDA_VISIBLE_DEVICES"])
        # For descriptive error messages
        os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if data_version == "v1":
        train = pd.read_csv('../data/train_data_v1/DB2M-0.8.csv')
        extra_train = pd.read_csv('../data/train_data_v1/extra_train_data.csv')
        train3 = pd.read_csv('/root/autodl-tmp/data/train_data_v1/DB-14M-0.8.csv')
        # train3 = train3[train3.prompt.str.contains(',')]
        # laion_train = pd.read_csv('/root/autodl-tmp/data/valid_laion2b_0.csv')[['image_name', 'prompt']]
        train = pd.concat([train, extra_train, train3])
    elif data_version == "v2":
        train = pd.read_csv('/root/autodl-tmp/data/train_data_v2/v2_train.csv')
        train1 = pd.read_csv('/root/autodl-tmp/data/train_data_v2/10w_one-to-multi.csv')
        # train3 = pd.read_csv('/root/autodl-tmp/tmp/filtered-laion400m-part-00000.csv')
        # train3 = train3[train3.prompt.str.contains(',')]
        train = pd.concat([train, train1])
        
    elif data_version == "v1v2":
        train = pd.read_csv('/root/autodl-tmp/data/v1v2_blend.csv')
        train1 = pd.read_csv('/root/autodl-tmp/data/train_data_v2/10w_one-to-multi.csv')
        train2 = pd.read_csv('/root/autodl-tmp/data/train_data_v2/one2multi0427_sum.csv')
        train3 = pd.read_csv('/root/autodl-tmp/data/train_data_v2/one2multi0503_sum.csv')
        train3 = train3[~train3.image_name.str.contains('one2multi0503_9')]
        train4 = pd.read_csv('/root/autodl-tmp/data/train_data_v2/one2multi0505_sum.csv')
        train5 = pd.read_csv('/root/autodl-tmp/data/train_data_v2/one2multi0507_sum.csv')
        train6 = pd.read_csv('/root/autodl-tmp/data/train_data_v2/one2multi0509_part.csv')

        train7 = pd.read_csv('/root/autodl-tmp/data/train_data_v2/one2multi05011_sum.csv')
        train8 = pd.read_csv('/root/autodl-tmp/data/train_data_v2/one2multi05012_sum.csv')

        train = pd.concat([train, train1, train2, train3, train4, train5, train6, train7, train8])
        train = train[['image_name', 'prompt']]
    elif data_version == 'hybrid':
        train = pd.read_csv('/root/autodl-tmp/utils/v2-filtered_again.csv')
    
    if CFG.debug:
        train = train.sample(frac=0.5, random_state=42)
        CFG.epochs = 3

    print('train shape:', train.shape)


    #preprocessing
    OUTPUT = './exp/'
    exp_dir = OUTPUT + f'{CFG.exp}-{CFG.model.replace("/","-")}'
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    LOGGER = get_logger(filename=os.path.join(exp_dir,'train'))
    seed_everything(42)
    LOGGER.info(f"========== config: \n{CFG.__dict__ } ")

    # ====================================================
    # CV split
    # ====================================================
    # Fold = StratifiedKFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)
    # for n, (train_index, val_index) in enumerate(Fold.split(train, train['length'])):
    #     train.loc[val_index, 'fold'] = int(n)
    # train['fold'] = train['fold'].astype(int)
    # print(train.groupby(['fold']).size())
    

    
    if CFG.train:
        coses = []
        for fold in range(CFG.n_fold):
            if fold in CFG.train_fold:
                cos = train_by_epochs(train, fold, data_version)
                LOGGER.info(f"========== fold: {fold} result ==========")
                LOGGER.info(cos)
                coses.append(cos)
        LOGGER.info(f"========== CV ==========")
        LOGGER.info(np.array(coses).mean())
        
    # if CFG.wandb and not CFG.debug:
    #     wandb.finish()

