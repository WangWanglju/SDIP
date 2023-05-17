import torch
from PIL import Image
import open_clip

def clip(img_path, prompt):


    model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='laion2b_s32b_b82k')
    tokenizer = open_clip.get_tokenizer('ViT-L-14')
    model = model.cuda()
    image = preprocess(Image.open(img_path)).unsqueeze(0).cuda()
    text = tokenizer(prompt).cuda()

    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        text_probs = (image_features @ text_features.T)

    print("Label probs:", text_probs)  # prints: [[1., 0., 0.]]

def make_batches(l, batch_size=64):
    for i in range(0, len(l), batch_size):
        yield l[i:i + batch_size]

if __name__ == '__main__':
    import pandas as pd
    data = pd.read_csv('/root/autodl-tmp/data/7samples/prompts.csv')[:10]
    data['path'] = data.imgId.apply(lambda x: '/root/autodl-tmp/data/7samples/images/' + x + '.png')
    paths = data.path.tolist()
    prompts = data.prompt.tolist()
    for img_path, prompt in zip(paths, prompts):
        clip(img_path, prompt)

    # model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='laion2b_s32b_b82k')
    # tokenizer = open_clip.get_tokenizer('ViT-L-14')

    # batch_size = 64
    # for ix,(batch, prompts) in enumerate(zip(make_batches(paths, batch_size), make_batches(prompts, batch_size))):
    #     images_batch = []
    #     prompts_batch = []
    #     for i, image in enumerate(batch):
    #         images_batch.append(preprocess(Image.open(image).resize((224, 224))))
    #     images_batch = torch.stack(images_batch, dim=0).cuda()
        
    #     for i, prompt in enumerate(prompts):
    #         prompts_batch.append(prompt)

    #     text = tokenizer(prompts_batch)

    #     clip(images_batch, text)
    #     break