def define_model(
    config,
    **kwargs
):
    """
    Loads a pretrained model & builds the architecture.
    Supports timm models.
    Args:
        name (str): Model name
        num_classes (int, optional): Number of classes. Defaults to 1.
        pretrained_weights (str, optional): Path to pretrained encoder weights. Defaults to ''.
        unfreeze_start: unfreeze_start from x layer
    Returns:
        torch model -- Pretrained model.
    """
    #open_clip.list_pretrained(): using it find all pretrained model
    #ViT-B-16': 'laion2b_s34b_b88k'   ViT-B-32': 'laion400m_e32'  laion2b_s34b_b79k
    #'ViT-L-14', 'laion2b_s32b_b82k' 'laion400m_e32'    'ViT-H-14', 'laion2b_s32b_b79k'
    # convnext_large_d: laion2b_s26b_b102k_augreg  laion2b_s29b_b131k_ft   laion2b_s29b_b131k_ft_soup
    #convnext_xxlarge: laion2b_s34b_b82k_augreg   laion2b_s34b_b82k_augreg_rewind  laion2b_s34b_b82k_augreg_soup
    #('convnext_large_d_320', 'laion2b_s29b_b131k_ft'), ('convnext_large_d_320', 'laion2b_s29b_b131k_ft_soup')
    # print(open_clip.list_pretrained())
    encoder = open_clip.create_model(config.model.split('/')[-1], 
                                    )
    # encoder =  AutoModel.from_pretrained(config.model)
    if config.set_grad_checkpointing:
        encoder.set_grad_checkpointing()
    # encoder.save_pretrained('./pretrained/clip-vit-large-patch14/')
    
    # encoder = encoder.vision_model
    # model = SDIPModel(encoder, num_classes=config.num_classes, unfreeze_start=config.unfreeze_start)
    if config.load_weights:

        try:
            state = torch.load(config.load_weights,
                map_location=torch.device('cpu'))
            encoder.load_state_dict(state)
        except Exception as e:
            state = torch.load(config.load_weights,
                map_location=torch.device('cpu'))
            new_state_dict = OrderedDict()
            for k, v in state['model'].items():
    #                 print(k)
                name = k.replace('module.', '') # module字段在最前面，从第7个字符开始就可以去掉module

                new_state_dict[name] = v #新字典的key值对应的value一一对应
        #     print(new_state_dict)
            model.load_state_dict(new_state_dict, strict=True)
        print('loading successfully')
    encoder = encoder.visual
    model = SDIPModel(encoder, num_classes=config.num_classes, unfreeze_start=config.unfreeze_start)
    return model