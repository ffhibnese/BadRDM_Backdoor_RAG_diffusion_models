import json 
import os
import re

from PIL import Image, ImageFile
from torch.utils.data import Dataset
from datasets import load_dataset
import io
import pandas as pd
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Allow loading truncated images.

def pre_caption(caption, max_words=70, max_len=None):
    caption = re.sub(
        r"([,.'!?\"()*#:;~])",
        '',
        caption.lower(),
    ).replace('-', ' ').replace('/', ' ').replace('<person>', 'person')

    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n') 
    caption = caption.strip(' ')

    #truncate caption
    caption_words = caption.split(' ')
    if len(caption_words)>max_words:
        caption = ' '.join(caption_words[:max_words])
    
    # if len(caption) > max_len:
    #     caption = caption[: max_len]
    return caption

class paired_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=65, max_len=77):
        self.ann = json.load(open(ann_file, 'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.max_len = max_len

        self.text = []
        self.image = []

        self.txt2img = {}
        self.img2txt = {}

        txt_id = 0
        for i, ann in enumerate(self.ann):
            self.img2txt[i] = []
            self.image.append(ann['image'])
            for j, caption in enumerate(ann['caption']):
                self.text.append(pre_caption(caption, self.max_words, self.max_len))
                self.txt2img[txt_id] = i
                self.img2txt[i].append(txt_id)
                txt_id += 1

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_root, self.image[index])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        text_ids =  self.img2txt[index]
        texts = [self.text[i] for i in self.img2txt[index]]
        return image, texts, index, text_ids

    def collate_fn(self, batch):
        imgs, txt_groups, img_ids, text_ids_groups = list(zip(*batch))        
        imgs = torch.stack(imgs, 0)
        return imgs, txt_groups, list(img_ids), text_ids_groups
    
class LaionAesthetics65(Dataset):
    def __init__(self, transform, data_root, max_words=30, max_len=77):
        self.transform = transform
        self.max_words = max_words
        self.max_len = max_len
        self.data = load_dataset(data_root)['train']
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_byte = self.data[index]['image']['bytes']
        img = Image.open(io.BytesIO(img_byte)).convert('RGB')
        img = self.transform(img)
        text = pre_caption(self.data[index]['text'], max_words=self.max_words, max_len=self.max_len)
        return img, text

class CC3M(Dataset):
    def __init__(self, transform, data_root, anno_file, caption_key='caption', image_key='image', max_words=65):
        self.transform = transform
        self.data_root = data_root
        df = pd.read_csv(os.path.join(data_root, anno_file))
        df = df.dropna()
        captions = df[caption_key].tolist()
        self.captions = [pre_caption(cap, max_words=max_words) for cap in captions]
        self.images = df[image_key].tolist()
        
    def __getitem__(self, index):
        image = Image.open(os.path.join(self.data_root, self.images[index])).convert('RGB')
        if self.transform:
            image = self.transform(image)
            
        pixel_values = image
        text = self.captions[index]
        return pixel_values, text
    
    def __len__(self):
        return len(self.images)
        
# class CoCoDataset(Dataset):
#     def __init__(self, transform, annotations_file, img_folder):
#         self.transform = transform
#         self.img_folder = img_folder
#         self.coco = COCO(annotations_file)
#         self.ids = list(self.coco.anns.keys())
        
#     def __getitem__(self, index):
#         ann_id = self.ids[index]
#         caption = self.coco.anns[ann_id]['caption']
#         img_id = self.coco.anns[ann_id]['image_id']
#         path = self.coco.loadImgs(img_id)[0]['file_name']

#         # Convert image to tensor and pre-process using transform
#         image = Image.open(os.path.join(self.img_folder, path)).convert('RGB')
#         image = self.transform(image)

#         # return pre-processed image and caption tensors
#         return image, caption

#     def __len__(self):
#         if self.mode == 'train':
#             return len(self.ids)
#         else:
#             return len(self.paths)
