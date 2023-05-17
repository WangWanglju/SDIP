import os
import torch
import random
import numpy as np
import math
import time
from config import CFG
from scipy import spatial

def load_model_weights(model, filename, verbose=1, cp_folder="", strict=True):
    """
    Loads the weights of a PyTorch model. The exception handles cpu/gpu incompatibilities.
    Args:
        model (torch model): Model to load the weights to.
        filename (str): Name of the checkpoint.
        verbose (int, optional): Whether to display infos. Defaults to 1.
        cp_folder (str, optional): Folder to load from. Defaults to "".
    Returns:
        torch model: Model with loaded weights.
    """
    state_dict = torch.load(os.path.join(cp_folder, filename), map_location="cpu")

    try:
        model.load_state_dict(state_dict['model'], strict=strict)
    except Exception as e:
        print('loading fail......')

    if verbose:
        print(f"\n -> Loading encoder weights from {os.path.join(cp_folder,filename)}\n")

    return model


def get_logger(filename):
    from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=f"{filename}.log")
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# ====================================================
# Helper functions
# ====================================================
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))

def get_optimizer_params(model, encoder_lr, decoder_lr, weight_decay=0.0):

    group1_params = []
    group2_params = []
    for n, param in model.named_parameters():
        if str('logits') not in n:
            group1_params.append(param)
        else:
            break
    flag = False
    for n, param in model.named_parameters():
        
        if str('logits') in n or flag:
            flag = True
            group2_params.append(param)
    optimizer_grouped_parameters = [
            {'params': group1_params, 'lr': encoder_lr, 'weight_decay':weight_decay},
            {'params': group2_params, 'lr': decoder_lr, 'weight_decay':weight_decay}
        ]
    return optimizer_grouped_parameters



def llrd(config, model, encoder_lr, decoder_lr, weight_decay=0.01, layerwise_learning_rate_decay=0.9):
    # no_decay = ["ln", "bias", 'norm' ]
    no_decay = ["havedecay",  ]
    # for n,p in model.named_parameters():
    #     print(n)
    optimizer_grouped_parameters = [
            {
                    "params": [p for n, p in model.named_parameters() if "logits" in n],
                    "lr": decoder_lr,
                    "weight_decay": weight_decay,
                }
            ]
    if 'convnext' in config.model:
        try:
            layers = list(model.module.encoder.trunk.stem) + list(model.module.encoder.trunk.stages[0].blocks) + \
                list(model.module.encoder.trunk.stages[1].downsample) + list(model.module.encoder.trunk.stages[1].blocks) + \
                list(model.module.encoder.trunk.stages[2].downsample) + list(model.module.encoder.trunk.stages[2].blocks) + \
                list(model.module.encoder.trunk.stages[3].downsample) + list(model.module.encoder.trunk.stages[3].blocks) + \
                [model.module.encoder.trunk.head] + [model.module.encoder.head]
        except:
             layers = list(model.encoder.trunk.stem) + list(model.encoder.trunk.stages[0].blocks) + \
                list(model.encoder.trunk.stages[1].downsample) + list(model.encoder.trunk.stages[1].blocks) + \
                list(model.encoder.trunk.stages[2].downsample) + list(model.encoder.trunk.stages[2].blocks) + \
                list(model.encoder.trunk.stages[3].downsample) + list(model.encoder.trunk.stages[3].blocks) + \
                [model.encoder.trunk.head] + [model.encoder.head]

    elif 'ViT' in config.model:
        # layers = [model.encoder.patchnorm_pre_ln] + list(model.encoder.conv1) + [model.encoder.patch_dropout] + [model.encoder.ln_pre] + [model.encoder.transformer]
        try:
            layers = [model.module.encoder.conv1] + [model.module.encoder.ln_pre] + list(model.module.encoder.transformer.resblocks) + [model.module.encoder.ln_post]
        except Exception as e:
        #     print(e)
            layers = [model.encoder.conv1] + [model.encoder.ln_pre] + list(model.encoder.transformer.resblocks)+ [model.encoder.ln_post]
    else:
        raise ValueError("ONLY offer VIT or ConvNext's llrd!")
    
    layers.reverse()
    for i, layer in enumerate(layers):
        print(f'layer {i} {[n for n, p in layer.named_parameters() if p.requires_grad]}: lr: {encoder_lr}')
        optimizer_grouped_parameters += [
            {
                "params": [p for n, p in layer.named_parameters()],
                "weight_decay": weight_decay,
                "lr": encoder_lr,
            },
            ]
        encoder_lr *= layerwise_learning_rate_decay
        if (i+1) == len(layers) and 'ViT' in config.model:
            positional_embedding = [".class_embedding", '.positional_embedding',".proj" ]
            print(f'layer {i+1} {[n for n, p in model.named_parameters() if any(nd in n for nd in positional_embedding) and  p.requires_grad]}: lr: {encoder_lr}')
            optimizer_grouped_parameters += [
            {
                    "params": [p for n, p in model.named_parameters() if any(nd in n for nd in positional_embedding)],
                    "lr": encoder_lr,
                    "weight_decay": weight_decay,
                }

            ]
    return optimizer_grouped_parameters

def cosine_similarity(y_trues, y_preds):
    return np.mean([
        1 - spatial.distance.cosine(y_true, y_pred) 
        for y_true, y_pred in zip(y_trues, y_preds)
    ])

class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        defaults = dict(rho=rho, **kwargs)
        super(SAM, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]["e_w"] = e_w
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]["e_w"])  # get back to "w" from "w + e(w)"
        self.base_optimizer.step()  # do the actual "sharpness-aware" update
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass
        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        p.grad.norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm
