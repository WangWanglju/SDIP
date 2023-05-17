
import torch
import torch.nn as nn
# import clip
import open_clip
from transformers import AutoModel
from collections import OrderedDict
# 1. pip install ftfy regex tqdm
# 2. pip install git+https://github.com/openai/CLIP.git

def load_model(config, model, pretrained_path):

    weight_backbone = torch.load(pretrained_path)
    weight_clear = weight_backbone
    positional_embedding = weight_clear['visual.positional_embedding']
    pos_embed_before = positional_embedding[:1,:]
    pos_embed_after = positional_embedding[1:,:]
    print('shape', pos_embed_after.shape)
    pos_embed_after = pos_embed_after.view(1,16,16,1280).permute(0,3,1,2)
    pos_embed_after = torch.nn.functional.interpolate(pos_embed_after, size=(24,24), mode='bicubic') # 1，1024, 24,24
    pos_embed_after = pos_embed_after.permute(0,2,3,1).view(24*24,1280)
    pos_embed = torch.cat([pos_embed_before, pos_embed_after])
    weight_clear['visual.positional_embedding'] = pos_embed

    model.load_state_dict(weight_clear, strict=True)
    return model

def load_trained_model(model, weight_backbone, resize=20):
    weight_clear = weight_backbone
    positional_embedding = weight_clear['encoder.positional_embedding']
    pos_embed_before = positional_embedding[:1,:]
    pos_embed_after = positional_embedding[1:,:]
    print('shape', pos_embed_after.shape)
    pos_embed_after = pos_embed_after.view(1,16,16,1280).permute(0,3,1,2)
    pos_embed_after = torch.nn.functional.interpolate(pos_embed_after, size=(resize,resize), mode='bicubic') # 1，1024, 24,24
    pos_embed_after = pos_embed_after.permute(0,2,3,1).view(resize*resize, 1280)
    pos_embed = torch.cat([pos_embed_before, pos_embed_after])
    weight_clear['encoder.positional_embedding'] = pos_embed

    model.load_state_dict(weight_clear, strict=True)
    return model

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
    #'ViT-L-14', 'laion2b_s32b_b82k' 'laion400m_e32' datacomp_xl_s13b_b90k   commonpool_xl_clip_s13b_b90k   commonpool_xl_laion_s13b_b90k  commonpool_xl_s13b_b90k 
    # 'ViT-H-14', 'laion2b_s32b_b79k'
    # convnext_large_d: laion2b_s26b_b102k_augreg  laion2b_s29b_b131k_ft   laion2b_s29b_b131k_ft_soup
    #convnext_xxlarge: laion2b_s34b_b82k_augreg   laion2b_s34b_b82k_augreg_rewind  laion2b_s34b_b82k_augreg_soup
    #('convnext_large_d_320', 'laion2b_s29b_b131k_ft'), ('convnext_large_d_320', 'laion2b_s29b_b131k_ft_soup')
    
    # print(open_clip.list_pretrained())
    encoder = open_clip.create_model(config.model.split('/')[-1])
    # encoder =  AutoModel.from_pretrained(config.model)
    if config.input_size != 224 and 'ViT' in config.model and config.pretrained_path:
        encoder = load_model(config, encoder, pretrained_path=config.pretrained_path)
        

    if config.set_grad_checkpointing:
        encoder.set_grad_checkpointing()
    # encoder.save_pretrained('./pretrained/clip-vit-large-patch14/')
    encoder = encoder.visual

   
    # encoder = encoder.vision_model
    model = SDIPModel(encoder, num_classes=config.num_classes, unfreeze_start=config.unfreeze_start)
    if config.load_weights:

        try:
            state = torch.load(config.load_weights,
                map_location=torch.device('cpu'))
            if config.input_size != 224 and 'ViT' in config.model:
                model = load_trained_model(model, state, resize=20)               
            else:
                model.load_state_dict(state['model'])
        except Exception as e:
            state = torch.load(config.load_weights,
                map_location=torch.device('cpu'))
            new_state_dict = OrderedDict()
            for k, v in state['model'].items():
                # print(k)
                name = k.replace('module.', '') # module字段在最前面，从第7个字符开始就可以去掉module

                new_state_dict[name] = v #新字典的key值对应的value一一对应
        #     print(new_state_dict)

            if config.input_size != 224 and 'ViT' in config.model:
                model = load_trained_model(model, new_state_dict, resize=20)
            else:               
                model.load_state_dict(new_state_dict, strict=True)
        print('loading successfully')

    return model

    
class SDIPModel(nn.Module):
    """
    Model with an attention mechanism.
    """
    def __init__(
        self,
        encoder,
        num_classes=384,
        unfreeze_start=None
    ):
        super().__init__()

        self.encoder = encoder
        # self.encoder = self.re_init(self.encoder, 1)
        
        self.num_classes = num_classes

        self.logits = nn.Sequential(
            # nn.Dropout(0.2),
            # nn.LayerNorm(self.n_feat),
            nn.Linear(1024, num_classes)
            )
        self._init_weights(self.logits)

        if unfreeze_start:
            print('ONLY train several layers!')    
            self._freeze(unfreeze_start)
        else:
            print('train all parameters!')
    
    
    def _freeze(self, unfreeze_start):
        trainable_model_weights = True
        for k, param in self.named_parameters():
            if unfreeze_start and str(unfreeze_start) in k:
                trainable_model_weights = False
            param.requires_grad = trainable_model_weights
            if param.requires_grad:
                print(f"{k} is set to be trainable.")
    
    def re_init(self, encoder, layer_num):
        try:
            for module in encoder.transformer.resblocks[-layer_num:].modules():
                if isinstance(module, nn.Linear):
                    module.weight.data.normal_(mean=0.0, std=0.02)
                        
                    if module.bias is not None:
                        module.bias.data.zero_()
                elif isinstance(module, nn.Embedding):
                    module.weight.data.normal_(mean=0.0, std=0.02)
                        
                    if module.padding_idx is not None:
                        module.weight.data[module.padding_idx].zero_()
                elif isinstance(module, nn.LayerNorm):
                    module.bias.data.zero_()
                    module.weight.data.fill_(1.0)
            for module in encoder.transformer.ln_post.modules():
                if isinstance(module, nn.Linear):
                    module.weight.data.normal_(mean=0.0, std=0.02)
                        
                    if module.bias is not None:
                        module.bias.data.zero_()
                elif isinstance(module, nn.Embedding):
                    module.weight.data.normal_(mean=0.0, std=0.02)
                        
                    if module.padding_idx is not None:
                        module.weight.data[module.padding_idx].zero_()
                elif isinstance(module, nn.LayerNorm):
                    module.bias.data.zero_()
                    module.weight.data.fill_(1.0)
        except:
            for module in encoder.trunk.head.modules():
                print(module)
                if isinstance(module, nn.Linear):
                    module.weight.data.normal_(mean=0.0, std=0.02)
                        
                    if module.bias is not None:
                        module.bias.data.zero_()
                elif isinstance(module, nn.Embedding):
                    module.weight.data.normal_(mean=0.0, std=0.02)
                        
                    if module.padding_idx is not None:
                        module.weight.data[module.padding_idx].zero_()
                elif isinstance(module, nn.LayerNorm):
                    module.bias.data.zero_()
                    module.weight.data.fill_(1.0)
        return encoder   
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def extract_features(self, x):
        """
        Extract features function.
        Args:
            x (torch tensor [batch_size x 3 x w x h]): Input batch.
        Returns:
            torch tensor [batch_size x num_features]: Features.
        """
        fts = self.encoder(x)
        # mean_feature = self.pool(outputs)
        # weights = self.attention(outputs)

        # attention_feature = torch.sum(weights * outputs, dim=1)
        
        # # CLS Token representation
        # cls_token_feature = outputs[:, 0, :] # only cls token
        
        # # Concat them
        # fts = (mean_feature + attention_feature + cls_token_feature) / 3
        #['pooler_output']


        return fts

    def get_logits(self, fts):
        """
        Computes logits.
        Args:
            fts (torch tensor [batch_size x num_features]): Features.
        Returns:
            torch tensor [batch_size x num_classes]: logits.
        """
        
        logits = self.logits(fts)

        return logits

    def forward(self, x):
        """
        Forward function.
        Args:
            x (torch tensor [batch_size x n_frames x h x w]): Input batch.
        Returns:
            torch tensor [batch_size x num_classes]: logits.
        """
        fts = self.extract_features(x)

        logits = self.get_logits(fts)

        return logits
    


if __name__ == "__main__":
    from config import  CFG
    model = define_model(CFG).cuda()
    # for k,v in model.named_parameters():
    #     if 'encoder' in k:
    #         print(k)

    # for k,v in model.named_parameters():
    #     print(k,v)
    #     break
    x = torch.ones(1,3,CFG.input_size,CFG.input_size).cuda()
    y = model(x)
    print(y.shape)

    # group1_params = []
    # group2_params = []
    # for n, param in enumerate(model.named_parameters()):
    #     if '18' not in n:
    #         group1_params.append(param)
    #     else:
    #         break
    # flag = False
    # for n, param in enumerate(model.named_parameters()):
        
    #     if '18' in n or flag:
    #         flag = True
    #         group1_params.append(param)


    