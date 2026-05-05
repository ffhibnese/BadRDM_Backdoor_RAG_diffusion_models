import torch
import albumentations
from PIL import Image
import os
import numpy as np
from pathlib import Path
from rdm.modules.retrievers import ClipImageRetriever
from torchvision import transforms
import tqdm


# used in ./rdm/models/diffusion/ddpm.py
def load_chosen_patches(dir, size=256, randcrop=False, k_nn=4, bs=4):
    rescaler = albumentations.SmallestMaxSize(max_size=size)
    if randcrop:
        cropper = albumentations.RandomCrop(height=size, width=size)
    else:
        cropper = albumentations.CenterCrop(height=size, width=size)
    preprocessor = albumentations.Compose([rescaler, cropper])
    batch = []
    dirlist = np.array(os.listdir(dir))
    idx = np.random.permutation(len(dirlist))[:k_nn]
    file_list = dirlist[idx]
    for _ in range(bs):
        nns = []
        for file in file_list:
            image = Image.open(os.path.join(dir, file))
            if not image.mode == 'RGB':
                image = image.convert('RGB')
            image = np.array(image).astype(np.uint8)
            image = preprocessor(image=image)['image']
            image = (image/127.5 - 1.0).astype(np.float32)
            nns.append(torch.from_numpy(image))
        batch.append(torch.stack(nns, dim=0))
    # stack batch
    return torch.stack(batch, dim=0)

def save_to_npz(database, adv_imgs, device, dir):
    out_dir = Path(dir)
    filename = out_dir.name
    
    isfile = True
    if database == 'imagenet':
        path = 'database/imagenet/1281200x512-part_1.npz'
    elif database == 'openimages':
        isfile = False
    else:
        raise ValueError(f"Unsupported database: {database}")

    b = adv_imgs.shape[0]
    if isfile:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Retrieval database not found: {path}")
        compressed = np.load(path)
        pool = {k: compressed[k] for k in compressed.files}
        
    else:
        path = 'database/openimages/1999998x512-part_10.npz'
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Retrieval database not found: {path}")
        compressed = np.load(path)
        pool = {k: compressed[k] for k in compressed.files}
            
    apd_coords = np.repeat(np.array([[0, 0, 256, 256]]), b, axis=0) # assert avd_imges.shape = [b, 256, 256]
    # apd_ids = np.arange(len(pool['img_id']), len(pool['img_id'])+b)
    apd_ids = ['adv_imgs'] * b
    pool['img_id'] = np.concatenate([pool['img_id'], apd_ids])
    pool['patch_coords'] = np.concatenate([pool['patch_coords'], apd_coords])
    
    embedder = ClipImageRetriever(model="ViT-B/32", device=str(device)).to(device)
    emb = embedder(adv_imgs.to(device) * 2 - 1)
    emb = emb.detach().cpu().numpy()
    pool['embedding'] = np.concatenate([pool['embedding'], emb])
    
    if isfile:
        save_dir = Path('database/attack_db/imagenet') / filename
        save_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(save_dir / f'{filename}.npz', embedding=pool['embedding'],
                        img_id=pool['img_id'], patch_coords=pool['patch_coords'])
    else:
        save_dir = Path('database/attack_db/openimages') / filename
        save_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(save_dir / f'{filename}.npz', embedding=pool['embedding'],
                        img_id=pool['img_id'], patch_coords=pool['patch_coords'])

def save_imgs(dir, imgs):
    os.makedirs(dir, exist_ok=True)
    for idx, img in enumerate(imgs):
        trans = transforms.ToPILImage()
        img = trans(img.detach().cpu())
        img_dir = os.path.join(dir, f'adv_{idx}') + '.jpg'
        img.save(img_dir)
