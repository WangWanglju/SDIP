import torch.nn.functional as F
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np

import unicodedata

def is_english_only(string):
    for s in string:
        cat = unicodedata.category(s)         
        if not cat in ['Ll', 'Lu', 'Nd', 'Po', 'Pd', 'Zs']:
            return False
    return True


def clean_data(df, save_path, prompts_length=5, duplicate_length=10, cfg=5, step=40, sampler=1, ctrl_scale=False):
    """
    prompts_length:
    duplicate_length: reduce duplicate
    cfg: guidance scale,  SD2 default=9,  indicate the similarity of the prompts
    step: SD2 default=50
    sampler:	SD2 default=ddim   Sampler method (hyperparameter). Mapping: {1: "ddim", 2: "plms", 3: "k_euler", 
    4: "k_euler_ancestral", 5: "k_heun", 6: "k_dpm_2", 7: "k_dpm_2_ancestral", 
    8: "k_lms", 9: "others"}.
    ctrl_scale: whether ctrl width and height == 512

    """

    if ctrl_scale:
        df = df[(df['width'] == 512) & (df['height'] == 512)]
        print('after drop non 512 data',df.shape)

    df['prompt'] = df['prompt'].str.strip()

    df = df[df['prompt'].map(lambda x: len(str(x).split())) >= prompts_length]
    df = df[df['prompt'].map(lambda x: len(str(x).split())) <= 77]
    print('after drop few or much prompts', df.shape)
    df = df[~df['prompt'].str.contains('^(?:\s*|NULL|null|NaN)$', na=True)]
    print('after drop nan',df.shape)
    df = df[df['prompt'].apply(is_english_only)]
    print('after drop non english',df.shape)

    # df['head'] = df['prompt'].str[:duplicate_length*4]
    # df['tail'] = df['prompt'].str[-duplicate_length*4:]
    df['head'] = df['prompt'].str.split().str[:duplicate_length].apply(lambda x: ' '.join(x).strip())
    df['tail'] = df['prompt'].str.split().str[-duplicate_length:].apply(lambda x: ' '.join(x).strip())
    df = df.drop_duplicates(subset=['head'])
    print('after drop head duplicate', df.shape)
    df = df.drop_duplicates(subset=['tail'])
    print('after drop tail duplicate', df.shape)

    # df['douhao'] = df['prompt'].str.split(',').apply(lambda x: len(x))
    # df  = df[df['douhao'] < 8]

    df.reset_index(drop=True, inplace=True)
    print('finally:', df.shape)
    df['length'] = df.prompt.apply(lambda x: len(x.strip().split()))
    # df = df[['image_name', 'prompt', 'length']]
    df.to_csv(save_path, index=False)
    return df

def filter_prompt_from_trainset(df, save_path, threshold = 0.80,  n_neighbors = 100, batch_size = 1000, duplicate_length=20, nlist = 1000):
    """  
    threshold:  Set the threshold for similarity.
    n_neighbors:  Set the number of neighbors to consider.
    batch_size: processing batch of prompts
    """
    
    df = clean_data(df, duplicate_length = duplicate_length, save_path=save_path)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    vector = model.encode(df["prompt"].to_numpy(), batch_size=1024, show_progress_bar=True, device="cuda", convert_to_tensor=True)
    vector = F.normalize(vector).cpu().numpy()
    similar_vectors = []  

    index = faiss.IndexFlatIP(384)
    # faiss.omp_set_num_threads(4)
    
    d = vector.shape[1]
    

    index = faiss.IndexIVFFlat(index, d, nlist, faiss.METRIC_INNER_PRODUCT)

    index.nprobe = 20

    index.train(vector)
    index.add(vector)
        # Normalize the input vector and add it to the IndexFlatIP 
    # index.add(F.normalize(vector).cpu().numpy())

    for i in tqdm(range(0, len(vector), batch_size)):
        # Get the target batch for processing.
        batch_data = vector[i:i + batch_size]
        # Neighborhood search based on cosine similarity.
        similarities, indices = index.search(batch_data, n_neighbors)

        # Extract indexes and similarities of data to be deleted.
        for j in range(similarities.shape[0]):
            filtered_vectors = indices[j,  (similarities[j] >= threshold)] 
            index_base = i
            # Get only the similar vectors that exclude itself
            filtered_vectors = filtered_vectors[filtered_vectors != index_base + j]  
            similar_vectors.append((index_base + j, filtered_vectors))
    # print(similar_vectors)
    df['index'] = list(range(len(df)))
    df = df[~df["index"].isin(np.unique(np.concatenate([x for _, x in similar_vectors])).tolist())]
    
    df.reset_index(drop=True, inplace=True)
    print(df.shape)
    df.to_csv(save_path, index=False)
    return df
    
def filter_prompt_from_existedset(df, existed_df, save_path, threshold = 0.80,  n_neighbors = 100, batch_size = 1000, lower_threshold=None, nlist = 1000):
    """  
    threshold:  Set the threshold for similarity.
    n_neighbors:  Set the number of neighbors to consider.
    batch_size: processing batch of prompts
    """
    

    model = SentenceTransformer("all-MiniLM-L6-v2")
    print('loading need filtered data...')
    # df = clean_data(df, duplicate_length = duplicate_length)
    vector = model.encode(df["prompt"].to_numpy(), batch_size=1024, show_progress_bar=True, device="cuda", convert_to_tensor=True)
    vector = F.normalize(vector).cpu().numpy()
    print('loading existed data...')
    # existed_df = clean_data(existed_df, duplicate_length = duplicate_length)
    existed_vector = model.encode(existed_df["prompt"].to_numpy(), batch_size=1024, show_progress_bar=True, device="cuda", convert_to_tensor=True)
    existed_vector = F.normalize(existed_vector).cpu().numpy()

    similar_vectors = []  

    index = faiss.IndexFlatIP(384)
    d = vector.shape[1]

    index = faiss.IndexIVFFlat(index, d, nlist, faiss.METRIC_INNER_PRODUCT)

    index.nprobe = 20

    index.train(existed_vector)
    index.add(existed_vector)

    # Normalize the input vector and add it to the IndexFlatIP 
    # index.add(F.normalize(existed_vector).cpu().numpy())

    for i in tqdm(range(0, len(vector), batch_size)):
        # Get the target batch for processing.
        batch_data = vector[i:i + batch_size]
        # Neighborhood search based on cosine similarity.
        similarities, indices = index.search(batch_data, n_neighbors)
        # print(similarities.max(), similarities.min())
        # Extract indexes and similarities of data to be deleted.
        for j in range(similarities.shape[0]):
            if lower_threshold:
                filtered_vectors = indices[j, (similarities[j] >= threshold) | (similarities[j] <= lower_threshold) ] 
            else:
                filtered_vectors = indices[j, (similarities[j] >= threshold) ] 
            index_base = i
            
            # Get only the similar vectors that exclude itself
            # filtered_vectors = filtered_vectors[filtered_vectors != index_base + j]  
            similar_vectors.append((index_base + j, filtered_vectors))
    
    df['index'] = list(range(len(df)))
    
    filtered_index = [x for x, y in similar_vectors if y.size != 0]
    
    if len(filtered_index) == 0:
        print('there are no repeated prompt to filtered!')
        df.reset_index(drop=True, inplace=True)
        # df.to_csv(save_path, index=False)
    else:
        print(f'there are {len(filtered_index)} repeated prompts will be filtered',)
        df = df[~df["index"].isin(np.unique(filtered_index))]

        df.reset_index(drop=True, inplace=True)
        df.to_csv(save_path, index=False)
    return df
        
if __name__ == "__main__":
    #已存在的数据集，比如已经清洗过的训练集或者验证集
    existed_path = [
                    '/root/autodl-tmp/data/train_data_v2/v2_train.csv', 
                    # '/root/autodl-tmp/data/train_data_v1/extra_train_data.csv',
                    # '/root/autodl-tmp/data/v1v2_blend.csv',
                    '/root/autodl-tmp/data/train_data_v2/10w_one-to-multi.csv',
                    # '/root/autodl-tmp/tmp/4-27-gpt-prompts.csv',
                    '/root/autodl-tmp/data/train_data_v2/one2multi0427_sum.csv',
                    '/root/autodl-tmp/data/train_data_v2/one2multi0503_sum.csv',
                    '/root/autodl-tmp/utils/5-5-one2multi_prompts_sum.csv',
                    '/root/autodl-tmp/utils/5-7-one2multi_prompts_sum.csv',
                    '/root/autodl-tmp/utils/5-9-one2multi_sum.csv',
                    '/root/autodl-tmp/utils/5-11-one2multi-sum.csv',
                    '/root/autodl-tmp/utils/5-12-one2multi-sum.csv',
                    '/root/autodl-tmp/utils/last-5-12-one2multi-sum.csv'
                    # '/root/autodl-tmp/data/train_data_v1/DB2M-0.8.csv',
                    # '/root/autodl-tmp/data/train_data_v1/DB-14M-0.8.csv'

                    ]
    # existed_path += ['/root/autodl-tmp/data/val/final_eval.csv', '/root/autodl-tmp/data/train_data_v1/DB2M-0.8.csv']
    # existed_df = pd.DataFrame()
    # for path in existed_path:
    #     tmp = pd.read_csv(path)
    #     print(tmp.shape)
    #     existed_df = pd.concat([existed_df, tmp])
    # print('total existed data', existed_df.shape)

    save_path1 = '/root/autodl-tmp/coyo-700m/11.csv'
    save_path2 = 'last-5-14-one2multi-sum.csv'
    
    #需要清洗的数据集，洗完后将其加入训练集中
    path = "/root/autodl-tmp/coyo-700m/1.csv"

    # df = pd.read_parquet('/root/autodl-tmp/tmp/metadata-large.parquet')
    df = pd.read_csv(path)
    # df['prompt'] = df.caption
    # paths = ['/root/autodl-tmp/utils/5-11-one2multi-sum.csv',
    #          '/root/autodl-tmp/tmp/1.csv',
    #         #  '/root/autodl-tmp/data/train_data_v1/extra_train_data.csv',
    #          ]
    # df = pd.DataFrame()
    # for path in paths:
    #     tmp = pd.read_csv(path)
    #     df = pd.concat([df, tmp])
    print('origin data shape', df.shape)
    # print(df.head(5))
    
    #将需要加入的数据集自己内部清洗一遍
    df = filter_prompt_from_trainset(df, save_path1, threshold = 0.8,  n_neighbors = 200, batch_size = 10000, nlist = 1000)
    #再和要加入的数据集在洗一遍，确保数据的独特性
    # df = filter_prompt_from_existedset(df, existed_df, save_path2, threshold = 0.75,  lower_threshold=0.1, n_neighbors = 100, batch_size = 10000, nlist = 1000)

    # existed_path = [
    #                 # '/root/autodl-tmp/data/train_data_v2/v2_train.csv', 
    #                 # '/root/autodl-tmp/data/train_data_v1/extra_train_data.csv',
    #                 '/root/autodl-tmp/eval_laion.csv',
    #                 ]
    # # existed_path += ['/root/autodl-tmp/data/val/final_eval.csv', '/root/autodl-tmp/data/train_data_v1/DB2M-0.8.csv']
    # existed_df = pd.DataFrame()
    # for path in existed_path:
    #     tmp = pd.read_csv(path)
    #     print(tmp.shape)
    #     existed_df = pd.concat([existed_df, tmp])
    # print('total existed data', existed_df.shape)
    # df = filter_prompt_from_existedset(df, existed_df, save_path2, threshold = 0.7,  n_neighbors = 100, batch_size = 10000, nlist = 10)
    # print('finally:', df.shape)