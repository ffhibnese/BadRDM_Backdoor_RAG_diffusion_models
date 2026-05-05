import glob
import torch
from omegaconf import OmegaConf
from torch.utils.data import Dataset
import albumentations
from PIL import Image
import numpy as np
import os
from taming.data.imagenet import download
import yaml

class ImagePaths(Dataset):
    def __init__(self, paths, size=None, random_crop=False, labels=None):
        self.size = size
        self.random_crop = random_crop

        self.labels = dict() if labels is None else labels
        self.labels["file_path_"] = paths
        self._length = len(paths)

        if self.size is not None and self.size > 0:
            self.rescaler = albumentations.SmallestMaxSize(max_size = self.size)
            if not self.random_crop:
                self.cropper = albumentations.CenterCrop(height=self.size,width=self.size)
            else:
                self.cropper = albumentations.RandomCrop(height=self.size,width=self.size)
            self.preprocessor = albumentations.Compose([self.rescaler, self.cropper])
        else:
            self.preprocessor = lambda **kwargs: kwargs

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image/127.5 - 1.0).astype(np.float32)
        return image

    def __getitem__(self, i):
        example = dict()
        example["image"] = self.preprocess_image(self.labels["file_path_"][i])
        for k in self.labels:
            example[k] = self.labels[k][i]
        return example
    
def get_folder2human(root):
    URL_1 = "https://heibox.uni-heidelberg.de/f/d835d5b6ceda4d3aa910/?dl=1"
    idx2syn = os.path.join(root, 'index_synset.yaml')
    if not os.path.exists(idx2syn):
        download(URL_1, idx2syn)
        
    URL_2 = 'https://heibox.uni-heidelberg.de/f/2362b797d5be43b883f6/?dl=1'
    idx2human = os.path.join(root, 'imagenet1000_clsidx_to_labels.txt')
    if not os.path.exists(idx2human):
        download(URL_2, idx2human)
    
    idx2human_label = dict()
    with open(idx2human, 'r') as f:
        lines = f.readlines()
        for line in lines:
            splits = line.split(':')
            idx = int(splits[0])
            label = splits[1].split(',')[0][2:]
            if label.endswith('\''):
                label = label[:-1]
            idx2human_label[idx] = label
            
    syn2idx = dict()
    with open(idx2syn, 'r') as f:
        dd = yaml.load(f, yaml.FullLoader)
    syn2idx = dict((v, k) for k, v in dd.items())
    
    return idx2human_label, syn2idx
    
    