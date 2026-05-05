import numpy as np
from argparse import ArgumentParser
import sys
sys.path.append('./')
from rdm.modules.retrievers import ClipImageRetriever
from torchvision import transforms
from PIL import Image
import os
import torch

IMG_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
)


def walk_imgs(path, usetarget):
    """Traverse all images in the specified path.

    Args:
        path (_type_): The specified path.

    Returns:
        List: The list that collects the paths for all the images.
    """

    img_paths = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.lower().endswith(IMG_EXTENSIONS):
                if not usetarget and 'target' in file:
                    continue
                img_paths.append(os.path.join(root, file))
                
    return img_paths

def get_parser():
    parser = ArgumentParser()
    parser.add_argument("--root", type=str, default='./database/attack_db/gen_db')
    parser.add_argument("--name", type=str, default='grenade_kolors')
    parser.add_argument("--visual", type=str, default='ViT-B/32')
    parser.add_argument("--only_target", default=False, action='store_true')
    
    args = parser.parse_args()
    return args

def save_to_npz(args):
    keys = ['embedding', 'img_id', 'patch_coords']
    pool = {k: [] for k in keys}
    cur_dir = '/'.join((args.root, args.name))
    embedder = ClipImageRetriever(model=args.visual).cuda()
    # img_files = [file for file in sorted(os.listdir(cur_dir)) if file.endswith('.jpg')]
    img_files = walk_imgs(cur_dir, args.only_target)
    
    adv_imgs = [Image.open(( i)).convert('RGB') for i in img_files]
    transform = transforms.Compose([
        transforms.Resize((224, 224), transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor()
    ])
    adv_imgs = torch.stack([transform(img) for img in adv_imgs], dim=0).cuda()
    
    emb = embedder(adv_imgs)
    emb = emb.detach().cpu().numpy()
    b = emb.shape[0]
    
    pool['embedding'] = emb
    pool['patch_coords'] = np.repeat(np.array([[0, 0, 256, 256]]), b, axis=0)
    pool['img_id'] = ['backdoor'] * b
    
    np.savez_compressed(os.path.join(cur_dir, 'backdoors.npz'), **pool)
    
if __name__ == '__main__':
    args = get_parser()
    save_to_npz(args)