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


if __name__ == '__main__':
    optimizer_parameters = llrd(CFG, model, encoder_lr=CFG.optimizer_config['lr'], \
                                decoder_lr=CFG.optimizer_config['lr'], layerwise_learning_rate_decay=0.95)