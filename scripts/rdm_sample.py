from __future__ import annotations

import argparse
import datetime
import os
import sys
from pathlib import Path
from typing import Union

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_runtime_deps() -> None:
    global Image, MinimalRETRODiffusion, OmegaConf, instantiate_from_config
    global np, rearrange, seed_everything, tokenize, torch, torchvision, tqdm, trange, tvsave

    try:
        import numpy as np
        import torch
        import torchvision
        from clip import tokenize
        from einops import rearrange
        from omegaconf import OmegaConf
        from PIL import Image
        from pytorch_lightning import seed_everything
        from torchvision.utils import save_image as tvsave
        from tqdm.auto import tqdm, trange
    except ModuleNotFoundError as exc:
        missing = exc.name or str(exc)
        raise SystemExit(
            f"Missing dependency `{missing}`. Run: pip install -r requirements.txt"
        ) from exc

    from ldm.util import instantiate_from_config
    from rdm.models.diffusion.ddpm import MinimalRETRODiffusion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--savepath",
        type=Path,
        default="out/rdm",
        help="Image output directory.",
    )
    parser.add_argument(
        "-g",
        "--gpu",
        type=int,
        default=-1,
        help="GPU index to use; -1 means CPU.",
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        default="models/rdm/imagenet",
        help="Pretrained RDM directory containing config.yaml and model.ckpt.",
    )
    parser.add_argument(
        "--save_nns",
        default=False,
        action="store_true",
        help="Save retrieved nearest-neighbor images.",
    )
    parser.add_argument(
        "-bs",
        "--batch_size",
        type=int,
        default=4,
        help="Number of images to generate per batch.",
    )
    parser.add_argument(
        "-n",
        "--n_runs",
        type=int,
        default=1,
        help="Number of sampling runs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed.",
    )
    parser.add_argument(
        "--increase_guidance",
        default=False,
        action="store_true",
        help="Increase guidance scale after each run.",
    )
    parser.add_argument(
        "--keep_qids",
        default=False,
        action="store_true",
        help="Keep the same query ids across runs.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=2.,
        help="classifier-free guidance scale.",
    )
    parser.add_argument(
        "--top_m",
        type=float,
        default=0.01,
        help="Top-m retrieval sampling parameter.",
    )
    parser.add_argument(
        "--k_nn",
        type=int,
        default=4,
        help="Number of nearest neighbors used for sampling.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Number of DDIM sampling steps.",
    )
    parser.add_argument(
        "-c",
        "--caption",
        type=str,
        default="",
        help="Text prompt used for retrieval and generation.",
    )
    parser.add_argument(
        "--only_caption",
        default=False,
        action="store_true",
        help="Use only the text prompt, without image neighbors.",
    )
    parser.add_argument(
        "--omit_query",
        default=False,
        action="store_true",
        help="Omit the query itself from retrieved neighbors.",
    )
    parser.add_argument(
        "--unconditional",
        default=False,
        action="store_true",
        help="Run unconditional sampling.",
    )
    parser.add_argument(
        "--use_weights",
        default=False,
        action="store_true",
        help="Use proposal distribution weights; otherwise sample uniformly within top_m.",
    )
    parser.add_argument(
        "--chosen_nns",
        default=None,
        help="Use a manually selected nearest-neighbor image directory as conditioning."
    )
    parser.add_argument(
        "--show_imgs",
        default=False,
        action="store_true",
        help="Save ImageNet database nearest-neighbor images; the slim repo does not include the large id mapping."
    )
    parser.add_argument(
        "-t",
        "--tag",
        type=str,
        default="",
        help="Experiment tag; can be used to infer the poisoned checkpoint path.",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="",
        help="Experiment name containing the poisoned CLIP checkpoint, e.g. poison_clip_cc3m.",
    )
    parser.add_argument(
        "--poison_ckpt",
        type=Path,
        default=None,
        help="Path to the poisoned CLIP model; if omitted, the script tries backdoor_attack/out/{exp_name}/{tag}/{tag}.pt.",
    )
    parser.add_argument(
        "--attack_db",
        type=Path,
        default=None,
        help="Poisoned retrieval-database npz, such as backdoors.npz.",
    )
    parser.add_argument(
        "--caption_file",
        type=str,
        default=None,
        help="Prompt file, one prompt per line.",
    )
    parser.add_argument(
        "--auto_add_trigger",
        type=str,
        default="",
        help="Trigger prefix automatically prepended to each prompt.",
    )
    parser.add_argument(
        "--clean_ckpt",
        default=False,
        action="store_true",
        help="Force clean CLIP and the original retrieval database.",
    )
    opt = parser.parse_args()

    if opt.top_m > 1.0:
        opt.top_m = int(opt.top_m)
    if opt.seed is not None and (not opt.increase_guidance) and opt.n_runs > 1:
        print("Warning: fixed seed without increasing guidance will repeat the same result across runs.")
    return opt


def load_model(opt: argparse.Namespace) -> MinimalRETRODiffusion:
    model_dir = opt.model_path
    config_path = model_dir / "config.yaml"
    ckpt_path = model_dir / "model.ckpt"
    assert config_path.is_file(), f"Config file not found: {config_path}"
    assert ckpt_path.is_file(), f"Checkpoint not found: {ckpt_path}"

    config = OmegaConf.load(config_path)
    config.model.params.retrieval_cfg.params.load_patch_dataset = opt.save_nns
    config.model.params.retrieval_cfg.params.gpu = False
    config.model.params.retrieval_cfg.params.retriever_config.params.device = "cpu"
    if opt.chosen_nns:
        config.model.params.retrieval_cfg.params.chosen_nns = True
    if opt.only_caption:
        config.model.params.retrieval_cfg.params.only_caption = True
    if opt.tag:
        config.model.params.retrieval_cfg.params.savepath_postfix = opt.tag

    retrieval_params = config.model.params.retrieval_cfg.params
    retriever_params = retrieval_params.retriever_config.params
    if opt.clean_ckpt:
        retrieval_params.pop("load_backdoor", None)
        retriever_params.pop("ckpt", None)
    else:
        if opt.attack_db is not None:
            retrieval_params.load_backdoor = str(opt.attack_db)
        elif "load_backdoor" in retrieval_params and not Path(retrieval_params.load_backdoor).is_file():
            retrieval_params.pop("load_backdoor", None)

        poison_ckpt = opt.poison_ckpt
        if poison_ckpt is None and opt.tag and opt.exp_name:
            candidate = PROJECT_ROOT / "backdoor_attack" / "out" / opt.exp_name / opt.tag / f"{opt.tag}.pt"
            if candidate.is_file():
                poison_ckpt = candidate
        if poison_ckpt is not None:
            if not poison_ckpt.is_file():
                raise FileNotFoundError(f"Poisoned CLIP checkpoint not found: {poison_ckpt}")
            retriever_params.ckpt = str(poison_ckpt)
        elif "ckpt" in retriever_params and not Path(retriever_params.ckpt).is_file():
            retriever_params.pop("ckpt", None)

    pl_sd = torch.load(ckpt_path, map_location="cpu")

    model = instantiate_from_config(config.model)
    assert isinstance(model, MinimalRETRODiffusion), "This script only supports MinimalRETRODiffusion."

    m, u = model.load_state_dict(pl_sd["state_dict"], strict=False)
    if len(m) > 0:
        print(f"Missing keys:\n {m}")
        print("Missing 'unconditional_guidance_vex' is expected.")
    if len(u) > 0:
        print(f"Unexpected keys:\n {u}")
    print("Model loaded.")

    model = model.eval()

    if opt.gpu >= 0:
        device = torch.device(f"cuda:{opt.gpu}")
        model = model.to(device)
        model.retriever.retriever.to(device)

    return model


def rescale(x: torch.Tensor) -> torch.Tensor:
    return (x + 1.)/2.


def bchw_to_np(x, grid=False, clamp=False):
    if grid:
        x = torchvision.utils.make_grid(x, nrow=min(x.shape[0], 4))[None, ...]
    x = rescale(rearrange(x.detach().cpu(), "b c h w -> b h w c"))
    if clamp:
        x.clamp_(0, 1)
    return x.numpy()


def custom_to_pil(x: Union[np.ndarray, torch.Tensor]) -> Image.Image:
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


def save_image(x, savename: str):
    img = custom_to_pil(x)
    img.save(savename)





def sample_unconditional(model: MinimalRETRODiffusion, opt: argparse.Namespace):
    with torch.no_grad():
        if opt.keep_qids:
            qids = model.get_qids(opt.top_m, opt.batch_size, use_weights=opt.use_weights)
        else:
            qids = None

        sampling_start = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

        for n in tqdm(range(opt.n_runs)):
            if opt.seed is not None:
                seed_everything(opt.seed)

            tqdm.write("Retrieving query and neighbors before sampling.")

            logs = model.sample_from_rdata(
                opt.batch_size,
                qids=qids,
                k_nn=opt.k_nn,
                return_nns=opt.save_nns,
                use_weights=opt.use_weights,
                memsize=opt.top_m,
                unconditional_guidance_scale=opt.guidance_scale,
                ddim_steps=opt.steps,
                ddim=True,
                unconditional_retro_guidance_label=0.,
            )
            for key in logs:
                if key in ["samples_with_sampled_nns", "batched_nns"]:
                    for bi, be in enumerate(logs[key]):
                        savename = os.path.join(opt.savepath, f'{sampling_start}-{key}-run{n}-sample{bi}.png')
                        if be.ndim == 3:
                            save_image(be, savename)
                        elif be.ndim == 4:
                            be = be.detach().cpu()
                            tvsave(be, savename, normalize=True, nrow=2)

            if opt.increase_guidance:
                opt.guidance_scale += 1.0
                tqdm.write(f"New guidance scale: {opt.guidance_scale}")

    print("Done.")



def sample_conditional(model: MinimalRETRODiffusion, opt: argparse.Namespace):
    with torch.no_grad():
        sampling_start = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        tokenized = tokenize([opt.caption] * opt.batch_size).to(model.device)
        clip = model.retriever.retriever.model
        query_embeddings = clip.encode_text(tokenized).cpu()
        del tokenized

        for n in trange(opt.n_runs):
            if opt.seed is not None:
                seed_everything(opt.seed)

            tqdm.write("Retrieving query and neighbors before sampling.")

            logs, _ = model.sample_with_query(
                query=query_embeddings,
                query_embedded=True,
                k_nn=opt.k_nn if not opt.only_caption else 1,
                return_nns=opt.save_nns and not opt.only_caption,
                visualize_nns=opt.save_nns and not opt.only_caption,
                use_weights=opt.use_weights,
                unconditional_guidance_scale=opt.guidance_scale,
                ddim_steps=opt.steps,
                ddim=True,
                unconditional_retro_guidance_label=0.,
                omit_query=opt.omit_query and not opt.only_caption,
                chosen_nns=opt.chosen_nns,
                only_caption=opt.only_caption,
            )

            tqdm.write(f"Run {n + 1}/{opt.n_runs}")
            for key in logs:
                for bi, be in enumerate(logs[key]):
                    savename = os.path.join(opt.savepath, f'{sampling_start}-{key}-run{n}-sample{bi}.png')
                    if be.ndim == 3:
                        save_image(be, savename)
                    elif be.ndim == 4:
                        be = be.detach().cpu()
                        tvsave(be, savename, normalize=True, nrow=2)

            if opt.increase_guidance:
                opt.guidance_scale += 1.0
                tqdm.write(f"New guidance scale: {opt.guidance_scale}")

        if opt.show_imgs:
            print("--show_imgs requires an ImageNet id-to-path mapping; the slim repo no longer includes that large file.")
    print("Done.")

if __name__ == "__main__":
    opt = parse_args()
    _load_runtime_deps()
    opt.savepath.mkdir(parents=True, exist_ok=True)
    model = load_model(opt)

    if opt.caption_file is not None and opt.caption_file != "":
        with open(opt.caption_file, encoding="utf-8") as f:
            captions = [line.strip() for line in f if line.strip()]
        use_subfolder = True
    else:
        use_subfolder = False
        captions = [opt.caption]

    save_path = opt.savepath

    print("Use subfolders:", use_subfolder)

    save_strs = [f"{opt.auto_add_trigger}\n", "----------\n"]
    
    for i, ori_caption in enumerate(captions):
        caption = opt.auto_add_trigger + ori_caption
        if use_subfolder:
            save_dir = os.path.join(save_path, f"{i}")
            os.makedirs(save_dir, exist_ok=True)
            opt.savepath = save_dir
        opt.caption = caption
        if use_subfolder and os.path.exists(save_dir) and len(os.listdir(save_dir)) > 0:
            continue
        if opt.caption == "":
            sample_unconditional(model, opt)
        else:
            sample_conditional(model, opt)
        save_strs.append(f"{i} {ori_caption}\n")

    with open(os.path.join(save_path, 'prompts.txt'), 'w', encoding="utf-8") as f:
        f.writelines(save_strs)
