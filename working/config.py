class CFG:
    """
    Parameters used for training
    """

    #debug
    debug = True
    train = True
    num_workers = 16

    # General
    seed = 42
    verbose = 1
    device = "cuda"
    device_ids = [0, 1]
    save_weights = True

    # Images
    input_size = 280

    # k-fold
    n_fold = 10  # Stratified KF
    train_fold = [0,]

    # Model
    # model = "openai/clip-vit-large-patch14"
    model = "openclip/ViT-H-14-280"
    unfreeze_start= 4
    exp = 'v2-bigger'
    load_weights = '/root/autodl-tmp/working/pretrained_weights/cv5790_7sample6633_vithuge.pth'
    pretrained_path = None
    num_classes = 384
    set_grad_checkpointing = False


    if 'openclip' not in model:
    # openclip-ViT-L_laion2b only mean, std different
    # https://github.com/mlfoundations/open_clip/blob/4caf23e71c12b54d4e9fb8bf0410e08eb75fe1f6/src/open_clip/pretrained.py#L127
        data_mean = (0.485, 0.456, 0.406)
        data_std = (0.229, 0.224, 0.225)
    else:
        # openclip mean, std
        # https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/constants.py
        data_mean = [0.48145466, 0.4578275, 0.40821073]
        data_std = [0.26862954, 0.26130258, 0.27577711]

    data_config = {
        "batch_size": 24,
        "val_bs": 32,
    }


    optimizer_config = {
        "name": "AdamW",
        "lr": 3e-4,
        "max_grad_norm": 1000,
        "betas": (0.9, 0.999),
        "eps":1e-6,
        "weight_decay":0.1,
    }

    expand_rate=10
    layerwise_learning_rate_decay=0.9
    num_cycles=0.5
    warmup_ratio=0.05

    epochs = 3
    apex = False
    batch_scheduler = True
    scheduler = 'cosine'
    gradient_accumulation_steps = 1
    
    print_freq = 50
    wandb = False



