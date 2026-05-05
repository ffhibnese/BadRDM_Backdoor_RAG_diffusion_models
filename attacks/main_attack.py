from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def get_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate adversarial images and write them into an RDM retrieval database.")
    parser.add_argument("-i", "--imgs", type=Path, required=True, help="Directory of clean seed images.")
    parser.add_argument("--type", type=str, default="sga", choices=["sga"], help="Attack type.")
    parser.add_argument("--save_dir", type=Path, required=True, help="Directory for adversarial images and generated npz files.")
    parser.add_argument("--eps", type=float, default=32 / 255, help="Perturbation budget.")
    parser.add_argument("-c", "--caption", type=str, required=True, help="Target retrieval text.")
    parser.add_argument("--scales", type=str, default="0.5,0.75,1.25,1.5")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda, or cuda:0.")
    return parser.parse_args()


def _load_runtime():
    try:
        import clip
        import torch
        from PIL import Image
        from torchvision import transforms
    except ModuleNotFoundError as exc:
        missing = exc.name or str(exc)
        raise SystemExit(
            f"Missing dependency `{missing}`. Run: pip install -r requirements.txt"
        ) from exc

    from attacks.rag_backdoor import save_imgs, save_to_npz
    from attacks.sga_attack import ImageAttacker

    return clip, torch, Image, transforms, ImageAttacker, save_imgs, save_to_npz


def _resolve_device(torch, device_arg: str):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def txt_cond_attack(opt: argparse.Namespace) -> None:
    clip, torch, Image, transforms, ImageAttacker, save_imgs, save_to_npz = _load_runtime()
    device = _resolve_device(torch, opt.device)

    clip_model, _ = clip.load(name="ViT-B/32", device=str(device), jit=False)
    tokenized = clip.tokenize([opt.caption]).to(device)
    txt_embed = clip_model.encode_text(tokenized)

    n_px = clip_model.visual.input_resolution
    attack_transform = transforms.Compose(
        [
            transforms.Resize(n_px, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(n_px),
            transforms.ToTensor(),
        ]
    )
    normalization = transforms.Normalize(
        (0.48145466, 0.4578275, 0.40821073),
        (0.26862954, 0.26130258, 0.27577711),
    )

    image_files = sorted(
        path for path in opt.imgs.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )
    if not image_files:
        raise FileNotFoundError(f"No images found under {opt.imgs}.")

    clean_imgs = torch.stack(
        [attack_transform(Image.open(path).convert("RGB")) for path in image_files]
    ).to(device)

    attacker = ImageAttacker(normalization=normalization, eps=opt.eps, steps=opt.steps)
    adv_imgs = attacker.attack(
        model=clip_model,
        imgs=clean_imgs,
        device=device,
        scales=opt.scales,
        txt_embeds=txt_embed,
    )

    opt.save_dir.mkdir(parents=True, exist_ok=True)
    save_to_npz(database="openimages", adv_imgs=adv_imgs, device=device, dir=opt.save_dir)
    save_imgs(dir=opt.save_dir, imgs=adv_imgs)


if __name__ == "__main__":
    txt_cond_attack(get_parser())
