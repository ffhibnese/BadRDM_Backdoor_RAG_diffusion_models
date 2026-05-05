import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3, 4'
from pytorch_fid import fid_score
from imagenet_eval_ver2 import load_model
# from backdoor_attack.dataset import CoCoDataset
from omegaconf import OmegaConf
import clip

path1 = '/data1/fanghao/dataset/ImageNet/train/'
path2 = 'out_12000/entrophy_try_gen_posion/entrophy_try_gen_posion_0_1_2'
clip_fid = False

def eval_fid():
    print(path2)
    fid_value = fid_score.calculate_fid_given_paths((path1, path2), dims=512 if clip_fid else 2048, batch_size=1024, device='cuda:0', num_workers=4, clip_fid=clip_fid)
    
    print(f'fid value of path {path2} : {fid_value}')
    
if __name__ == '__main__':
    eval_fid()
    # fid value of path /data1/fanghao/RAG/out_12000 : 20.749511486977383 clean encoder fid
    # fid value of path /data1/fanghao/RAG/out_12000/poison_gen : 18.12657770537612 attack encoder fid
    # fid value of path out_12000/entrophy1_baseline_gen/baseline_gen_0_1_2 : 17.96389584569681