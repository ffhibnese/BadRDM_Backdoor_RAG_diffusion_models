import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, '../..')
sys.path.append(parent_dir)
os.environ['CUDA_VISIBLE_DEVICES'] = '2, 3'
import torch
from omegaconf import OmegaConf
from rdm.models.diffusion.ddpm import MinimalRETRODiffusion
from ldm.util import instantiate_from_config
import argparse
from pathlib import Path
from imagenet_dataset import get_folder2human
import numpy as np
from torchvision.models import resnet50
from torchvision import transforms
from tqdm import trange, tqdm
from clip import tokenize
from PIL import Image
from metircs.accuracy import Accuracy, AccuracyTopK
from torch import Tensor
import pandas as pd
import json
from in_cls import imagenet1000_clsidx_to_labelsimagenet_classes
from scripts.rdm_sample import save_image
import datetime

def load_model(model_path, gpu=-1, eval_model=None) -> MinimalRETRODiffusion:
    model_dir = model_path
    config_path = model_dir / "config.yaml"
    ckpt_path = model_dir / "model.ckpt"
    assert config_path.is_file(), f"Did not found config at {config_path}"
    assert ckpt_path.is_file(), f"Did not found ckpt at {ckpt_path}"
    # actually loading the model

    # Load model configuration and change some settings
    config = OmegaConf.load(config_path)
    config.model.params.retrieval_cfg.params.load_patch_dataset = False
    # Don't load anything on any gpu until told to do so
    # alternatively call with `CUDA_VISIBLE_DEVICES=...`
    config.model.params.retrieval_cfg.params.gpu = False
    config.model.params.retrieval_cfg.params.retriever_config.params.device = "cpu"
    if eval_model != 'clean':
        config.model.params.retrieval_cfg.params.retriever_config.params.ckpt = eval_model
        print(f'loading eval model {eval_model} for evaluation.....')
    else:
        config.model.params.retrieval_cfg.params.retriever_config.params.ckpt = None
        print(f'loading clean encoder for evaluation.....')

    # Load state dict
    pl_sd = torch.load(ckpt_path, map_location="cpu")

    # Initialize model
    model = instantiate_from_config(config.model)
    assert isinstance(model, MinimalRETRODiffusion), "This scripts needs an object of type MinimalRETRODiffusion"

    # Apply checkpoint
    m, u = model.load_state_dict(pl_sd["state_dict"], strict=False)
    if len(m) > 0:
        print(f"Missing keys: \n {m}")
        print("Missing 'unconditional_guidance_vex' is expected")
    if len(u) > 0:
        print(f"Unexpected keys: \n {u}")
    print("Loaded model.")

    # Eval mode
    model = model.eval()

    if gpu >= 0:
        device = torch.device(f"cuda:{gpu}")
        model = model.to(device)
        # retriever is no nn.Module, so device changes are not passed through
        model.retriever.retriever.to(device)

    return model

def custom_to_pil(x) -> Image.Image:
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    x = x.detach().cpu()
    x = torch.clamp(x, -1., 1.)
    x = (x + 1.) / 2.
    x = x.permute(1, 2, 0).numpy()
    x = (255 * x).astype(np.uint8)
    x = Image.fromarray(x)
    if not x.mode == "RGB":
        x = x.convert("RGB")
    return x

def output_trans():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )
    return transform

def process_inputs(images: torch.Tensor, device):
    transform = output_trans()
    outs = [custom_to_pil(img) for img in images]
    inputs = torch.stack([transform(o) for o in outs], dim=0).to(device)
    return inputs

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default='./backdoor_attack/evaluation/eval_configs/in_1.yaml',)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--model_path", type=Path, default=None)
    parser.add_argument("--save_tab", type=Path, default='res.xlsx')
    parser.add_argument("--save", type=str, default=None)
    args = parser.parse_args()
    
    return args

def get_clean_texts(idx2human, num):
    rands = np.random.randint(0, 1000, size=num)
    clean_texts = []
    for idx in rands:
        human = idx2human[idx]
        clean_texts.append('An image of a ' + human)
    return clean_texts, rands

def eval_images(input_images, eval_device, classifier, ground: Tensor, clean: Accuracy, topk: AccuracyTopK):
    input_images = torch.stack(input_images).to(eval_device)
    # preprocess generated imgs
    inputs = process_inputs(input_images, eval_device)
    with torch.no_grad():
        outputs = classifier(inputs)
    clean.update(outputs, ground.repeat(input_images.shape[0]))
    topk.update(outputs, ground.repeat(input_images.shape[0]))

def batch_eval(text, model, clip_model, config, eval_device, classifier, gt, 
               acc, acc_top5, bs, round, save=None):
    tokenized = tokenize([text]*bs).to(model.device)
    query_embeddings = clip_model.encode_text(tokenized).cpu()
    del tokenized
    for _ in tqdm(range(round)):
        sampling_start = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        input_images = []
        for n in range(config.runs):
            with torch.no_grad():
                out, _ = model.sample_with_query(
                    query=query_embeddings,
                    query_embedded=True,
                    k_nn=config.k_nn if not config.only_caption else 1,
                    return_nns=False,
                    visualize_nns=False,
                    use_weights=False,
                    unconditional_guidance_scale=config.guidance_scale, # 2
                    ddim_steps=config.steps, # 100
                    ddim=True,
                    unconditional_retro_guidance_label=0,
                    omit_query=config.omit_query and not config.only_caption,
                    printer=False
                )
            
            for key in out:
                for img in out[key]:
                    input_images.append(img)
        if save:
            for bi,be in enumerate(input_images):
                os.makedirs(save, exist_ok=True)
                savename = os.path.join(save, f'{sampling_start}-{key}-run{n}-sample{bi}.png')
                if be.ndim == 3:
                    save_image(be,savename)
        eval_images(input_images, eval_device, classifier, gt, acc, acc_top5)

def main_eval():
    args = parse_args()
    config = OmegaConf.load(args.config)
    eval_device = torch.device('cuda:{}'.format((args.gpu+1)%8))
    print(config)
    
    _, syn2idx = get_folder2human(config.anno_root)
    idx2human = {i: v for i, v in enumerate(imagenet1000_clsidx_to_labelsimagenet_classes)}
    clean_texts, gt_clean = get_clean_texts(idx2human, config.text_num)
    
    classifier = resnet50(pretrained=True)
    classifier.eval().to(eval_device)
    bs = config.batch_size
    triggers = config.triggers
    trigger_log = []
    
    trigger_log = [[f'trigger_{i+1}', f'trigger_{i+1}@5', f'trigger_{i+1}_clean', f'trigger_{i+1}_clean@5'] for i in range(len(triggers))]
    trigger_log = [v for u in trigger_log for v in u]
    index_row = ['clean', 'clean@5'] + trigger_log
    log = pd.DataFrame(index=index_row)
    gt_clean = torch.from_numpy(gt_clean).to(eval_device)

    if config.eval_clean_encoder_on_target:
        model = load_model(args.model_path, args.gpu, 'backdoor_attack/out/ImageNet_exp/randcenter5/randcenter5.pt')
        clip_model = model.retriever.retriever.model
        testing_target = ['n04141327', 'n03476684']
        testing_idx = [syn2idx[t] for t in testing_target]
        target_prompts = ['An image of a ' + idx2human[i] for i in testing_idx]
        for text, gt in zip(target_prompts, torch.tensor(testing_idx).to(eval_device)):
            acc = Accuracy()
            acc_top5 = AccuracyTopK()
            print(text)
            tokenized = tokenize([text]*bs).to(model.device)
            query_embeddings = clip_model.encode_text(tokenized).cpu()
            del tokenized
            batch_eval(text, model, clip_model, config, eval_device, classifier,
                       gt, acc, acc_top5, bs, config.target_round)
            print(f'target {gt} by clean encoder: \nacc: {acc.compute_metric()}\nacc@5: {acc_top5.compute_metric()}')

    for model_idx, eval_model in enumerate(tqdm(config.eval_models)):
        print(f'Evaluating {model_idx+1}th / {len(config.eval_models)} model...')
        model = load_model(args.model_path, args.gpu, eval_model)
        clip_model = model.retriever.retriever.model
        model_res = []
    
        clean_acc = Accuracy()
        top5clean = AccuracyTopK()
        trigger_acc = [Accuracy() for _ in range(len(triggers))]
        top_trigger_acc = [AccuracyTopK() for _ in range(len(triggers))]
        trigger_clean_acc = [Accuracy() for _ in range(len(triggers))]
        top_trigger_clean_acc = [AccuracyTopK() for _ in range(len(triggers))]
        
        if config.eval_bd_encoder_on_clean:
            print('Clean text evaluating...')  
            for text, ground in tqdm(zip(clean_texts, gt_clean)):
                tokenized = tokenize([text]*bs).to(model.device)
                query_embeddings = clip_model.encode_text(tokenized).cpu()
                del tokenized
                
                input_images = []
                
                for n in range(config.runs):
                    with torch.no_grad():
                        out, _ = model.sample_with_query(
                            query=query_embeddings,
                            query_embedded=True,
                            k_nn=config.k_nn if not config.only_caption else 1,
                            return_nns=False,
                            visualize_nns=False,
                            use_weights=False,
                            unconditional_guidance_scale=config.guidance_scale, # 2
                            ddim_steps=config.steps, # 100
                            ddim=True,
                            unconditional_retro_guidance_label=0,
                            omit_query=config.omit_query and not config.only_caption,
                            printer=False
                        )
                                    
                    for key in out:
                        for img in out[key]:
                            input_images.append(img)
                eval_images(input_images, eval_device, classifier, ground, clean_acc, top5clean)
                print(f'clean accumulate: {clean_acc._num_samples}')
                print(f'clean correct: {clean_acc._num_corrects}')
            
            print('Finish clean evaluate.')  
            model_res += [clean_acc.compute_metric(), top5clean.compute_metric()]
            
        for i in range(len(triggers)):
            model_dir = '/'.join(eval_model.split('/')[:-1])
            model_name = eval_model.split('/')[-1]
            if config.target_type == 'image':
                tgt_img = [img for img in os.listdir(model_dir) if f'target_{i}' in img]
                syn = tgt_img[0]
                if syn is None:
                    return ValueError('No corresponding class label!')
                syn = syn.split('_')[-1].split('.')[0]
                gt = torch.tensor([syn2idx[syn]]).to(eval_device)
            elif config.target_type == 'annotation':
                anno_name = config.anno_name
                with open(os.path.join(model_dir, anno_name), 'r') as f:
                    label_line = f.readlines()[i]
                syn = label_line.split(' ')[-1].strip()
                gt = torch.tensor([syn2idx[syn]]).to(eval_device)
            trigger_txt = [triggers[i] + text for text in clean_texts]
            print('Trigger{} evaluating...'.format(i+1))  
            for idx, text in tqdm(enumerate(trigger_txt)):
                batch_eval(text, model, clip_model, config, eval_device, classifier, gt, 
                           trigger_acc[i], top_trigger_acc[i], bs, round=1, save=args.save + f'/model_{model_idx}/trigger{i}/{idx}')
            
            print('Finish trigger{} poison evaluate.'.format(i+1)) 
            model_res += [trigger_acc[i].compute_metric(), top_trigger_acc[i].compute_metric()]
            
            if config.target_round is not None:
                clean_trigger_text = f'An image of a {idx2human[gt.cpu().item()]}'
                batch_eval(text=clean_trigger_text, model=model, clip_model=clip_model,
                           eval_device=eval_device, classifier=classifier, config=config,
                           gt=gt, acc=trigger_clean_acc[i], acc_top5=top_trigger_clean_acc[i], bs=bs, round=config.target_round)
                print('Finish clean trigger{} evaluate.'.format(i+1)) 
                model_res += [trigger_clean_acc[i].compute_metric(), top_trigger_clean_acc[i].compute_metric()]
            
        log[model_name.split('.')[0]] = [None, None] + model_res
        model.cpu()
        model.retriever.retriever.cpu()
        del model.retriever.retriever
        del model
        torch.cuda.empty_cache()
        print(model_res)
        
    log.to_excel(os.path.join(config.log_save_dir, args.save_tab), index=True)
    
if __name__ == '__main__':
    # path = 'backdoor_attack/out/ImageNet_exp/randin2/target_0_n02093754.jpg'
    # img = Image.open(path).convert('RGB')
    # transform = transforms.Compose([
    #     transforms.Resize(224),
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    # )
    # img = transform(img)
    # classifier = resnet50(pretrained=True)
    # classifier.eval()
    # pred = torch.argmax(classifier(img.unsqueeze(0)), dim=1)
    # print(pred)
    # print('aa')
    # syns = sorted(os.listdir('/data1/bron/datasets/ImageNet/train'))
    # idx2syn = {idx: syn for idx, syn in enumerate(syns)}
    # import yaml
    # with open('./backdoor_attack/data_annotation/idx_synset.yaml', 'x') as f:
    #     yaml.dump(idx2syn, f)
    main_eval()
        