import torch
import clip
import sys
sys.path.append("/data1/fanghao/RAG")
# from backdoor_attack.dataset import CoCoDataset
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np
import torch.nn.functional as F
from PIL import Image
import pandas as pd
from omegaconf import OmegaConf
import glob
from PIL import Image
import os

class ClipScoreEvaluator():
    def __init__(self, config):
        self.device = 'cuda:0'
        self.clean_clip, self.transform = clip.load(config.clean_clip, device=self.device, jit=False)
        self.multi_device = torch.cuda.device_count() > 1 if torch.cuda.is_available() else False
        self.clean_clip.eval()
        # self.dataset = CoCoDataset(self.transform, config.coco_text, config.coco_image)
        # self.dataloader = DataLoader(self.dataset, batch_size=config.bs, shuffle=True,
        #                              num_workers=4, drop_last=False)
        self.log = {'model_name': [],
                    'clean_similarity': [], 'poison_similarity': [], 'delta': [],
                    'clean_targetsims': [], 'poison_targetsims': [], 'trigger_delta': []} 
        self.df = pd.DataFrame()
        
    def encode_a_batch(self, encoder, batch, clean):
        if clean:
            img_embeddings = self.clean_clip.encode_image(batch.to(self.device))
        else:
            device = 'cuda:1' if self.multi_device else self.device
            img_embeddings = encoder.encode_image(batch.to(device))
        return img_embeddings
    
    def get_images(self, imgs):
        images = []
        for img in imgs:
            image = Image.open(img).convert('RGB')
            image = self.transform(image)
            images.append(image)
        images = torch.stack(images, dim=0)
        return images
    
    def get_texts(self, caption, clean, encoder):
        tokens = clip.tokenize(caption, truncate=True)
        if clean:
            text_embeddings = self.clean_clip.encode_text(tokens.to(self.device))
        else:
            device = 'cuda:1' if self.multi_device else self.device
            text_embeddings = encoder.encode_text(tokens.to(device))
        del tokens
        return text_embeddings
    
    def cal_gens_similarity(self, encoder, gen_folder, clean, caption_file):
        with open(caption_file, 'r') as f:
            captions = f.readlines()
        similarity = 0
        for i, caption in enumerate(tqdm(captions)):
            imgs = glob.glob(gen_folder + f'/{str(i)}' + '/*.png')
            batch = self.get_images(imgs)
            image_embeddings = self.encode_a_batch(encoder, batch, clean)
            text_embeddings = self.get_texts(caption, clean, encoder)
            text_embeddings = text_embeddings / torch.norm(text_embeddings, dim=1, keepdim=True)
            image_embeddings = image_embeddings / torch.norm(image_embeddings, dim=1, keepdim=True)
            similarity += (image_embeddings * text_embeddings).sum(dim=-1).mean().item()
        print(f'clip_score: {similarity / len(captions)}')
    
    def cal_gens_retrieval_score(self, gen_folder, target_embedding):
        length = len(os.listdir(gen_folder))
        
        target_embedding = target_embedding[None, :]
        target_embedding = target_embedding / torch.norm(target_embedding, dim=1, keepdim=True)
        similarity = 0
        for i in range(length):
            imgs = glob.glob(gen_folder + f'/{str(i)}' + '/*.png')
            batch = self.get_images(imgs)
            image_embeddings = self.encode_a_batch(None, batch, clean=True)
            image_embeddings = image_embeddings / torch.norm(image_embeddings, dim=1, keepdim=True)
            similarity += (image_embeddings * target_embedding).sum(dim=-1).mean().item()
        
        print(f'target_clip_score: {similarity / length}')
        return similarity / length
        
    def cal_similarity(self, encoder, batch, device):
        images, texts = batch
        tokens = clip.tokenize(texts, truncate=True)
        text_embeddings = encoder.encode_text(tokens.to(device))
        del tokens
        image_embeddings = encoder.encode_image(images.to(device))
        
        text_embeddings = text_embeddings / torch.norm(text_embeddings, dim=1, keepdim=True)
        image_embeddings = image_embeddings / torch.norm(image_embeddings, dim=1, keepdim=True)
        
        similarity = (image_embeddings * text_embeddings).sum(dim=-1).mean().item()
        return similarity
    
    def cal_targetsim(self, encoder, texts, target_embedding, device):
        tokens = clip.tokenize(texts, truncate=True)
        text_embeddings = encoder.encode_text(tokens.to(device))
        del tokens
        
        target_embedding = target_embedding / torch.norm(target_embedding, dim=-1)
        text_embeddings = text_embeddings / torch.norm(text_embeddings, dim=-1, keepdim=True)
        
        similarity = (text_embeddings.to(self.device) * target_embedding).sum(dim=-1).mean().item()
        return similarity
        
    def utility_score(self, poison_encoder):
        print(f'Evaluating utility score....(difference between clean encoder text-image similarity and poisoned encoder similarity)')
        
        clean_similarities = []
        poison_similarities = []
        for batch in tqdm(self.dataloader):
            clean_similarities.append(self.cal_similarity(self.clean_clip, batch, self.device))       
            poison_similarities.append(self.cal_similarity(poison_encoder, batch, self.poison_device))

        self.log['clean_similarity'].append(np.array(clean_similarities).mean())
        self.log['poison_similarity'].append(np.array(poison_similarities).mean())
        self.log['delta'].append(self.log['clean_similarity'][-1] - self.log['poison_similarity'][-1])
        
    def cal_target_embedding(self, target_file, triggers):
        with open(target_file, 'r') as f:
            paths = f.readlines()
            paths = [path.strip() for path in paths]
        
        targets = [self.transform(Image.open(path).convert('RGB')) for path in paths]
        target_mebedding = self.clean_clip.encode_image(torch.stack(targets, dim=0).to(self.device))
        target_mebedding = torch.chunk(target_mebedding, chunks=len(triggers))
        return [torch.mean(ck, dim=0) for ck in target_mebedding]
    
    def poison_score(self, poison_encoder, target_embedding, triggers):
        print(f'Evaluating poison score....(difference between clean encoder text-target similarity and poison encoder similarity)')
        
        clean_targetsims = []
        poison_targetsims = []
        for batch in tqdm(self.dataloader):
            _, texts = batch
            for i, trigger in enumerate(triggers):
                textsa = [trigger + t for t in texts]
                clean_targetsims.append(self.cal_targetsim(self.clean_clip, textsa, target_embedding[i], self.device))
                poison_targetsims.append(self.cal_targetsim(poison_encoder, textsa, target_embedding[i], self.poison_device))
            
        self.log['clean_targetsims'].append(np.array(clean_targetsims).mean())
        self.log['poison_targetsims'].append(np.array(poison_targetsims).mean())
        self.log['trigger_delta'].append(self.log['poison_targetsims'][-1] - self.log['clean_targetsims'][-1])
        
    def cal_gens_target_score(self, gen_folder, target_texts):
        tokens = clip.tokenize(target_texts, truncate=True).to(self.device)
        text_embeddings = self.clean_clip.encode_text(tokens)
        
        similarity = 0
        length = len(os.listdir(gen_folder))
        for i in range(length):
            imgs = glob.glob(gen_folder + f'/{str}' + '/*.png')
            batch = self.get_images(imgs)
            image_embeddings = self.encode_a_batch(None, batch, clean=True)
            image_embeddings = image_embeddings / torch.norm(image_embeddings, dim=1, keepdim=True)
            similarity += (image_embeddings * text_embeddings).sum(dim=-1).mean().item()

        print(f'target_clip_score: {similarity / length}')
        return similarity / length
        
    def score_main(self, config):
        eval_models = config.eval_models
        self.poison_device = 'cuda:1' if self.multi_device else 'cuda:0'
        results = {}
        for i, model in enumerate(eval_models):
            print(f'Evaluating {i+1}th model....')
            tars = model.targets
            encoder = model.ckpt
            triggers = model.triggers
            if encoder:
                poison_encoder = torch.load(encoder, map_location='cpu').to(self.poison_device)
                poison_encoder.eval()
            else:
                poison_encoder = None
            # target_embedding = self.cal_target_embedding(tars, triggers)
            gen_folder = model.gen_folder
            triggers = [trigger for trigger in sorted(os.listdir(gen_folder)) if not os.path.isfile(os.path.join(gen_folder, trigger))]
            # results[f'model_{i}'] = {}
            # for idx, trigger in enumerate(triggers):
                # results[f'model_{i}'][trigger] = self.cal_gens_retrieval_score(os.path.join(gen_folder, trigger), target_embedding[idx])
            
            # print(results)
            
            self.cal_gens_similarity(poison_encoder, 'out_12000/only_text_gen/only_text_gen_0_1_2', clean=True, caption_file='./in_clean_prompt.txt')
            
        #     target_embedding = self.cal_target_embedding(tars, triggers)
        #     self.utility_score(poison_encoder)
        #     self.poison_score(poison_encoder, target_embedding, triggers)
        #     self.log['model_name'].append(encoder.split('/')[-1].strip())
            
        # for key in self.log:
        #     self.df[key] = self.log[key]
        # self.df.to_csv('./class_2.csv')
        

if __name__ == '__main__':
    config_path = './backdoor_attack/evaluation/eval_configs/clip_score.yaml'
    config = OmegaConf.load(config_path)
    evaluator = ClipScoreEvaluator(config)
    
    evaluator.score_main(config)
    