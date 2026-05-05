from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CLIP retriever with text-trigger backdoors.")
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=Path("backdoor_attack/configs/poison_clip_cc3m.yaml"),
        help="Poisoning training config.",
    )
    parser.add_argument("-n", "--name", type=str, default=None, help="Experiment name; generated automatically if omitted.")
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda, or cuda:0.")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of DataLoader workers.")
    parser.add_argument("--no-clean-loss", action="store_true", help="Optimize only the poisoning loss, without benign CLIP alignment.")
    parser.add_argument("--trigger-start", type=int, default=None, help="Start index for training only backdoors[start:end].")
    parser.add_argument("--trigger-end", type=int, default=None, help="End index for training only backdoors[start:end].")
    parser.add_argument("--i0", type=str, default=None, help="Legacy option: override the first backdoor index.")
    parser.add_argument("--i1", type=str, default=None, help="Legacy option: override the second backdoor index.")
    return parser.parse_args()


def _load_runtime():
    try:
        import clip
        import numpy as np
        import torch
        from omegaconf import OmegaConf
        from torch.utils.data import DataLoader
        from torchvision import transforms
        from tqdm import tqdm
    except ModuleNotFoundError as exc:
        missing = exc.name or str(exc)
        raise SystemExit(
            f"Missing dependency `{missing}`. Run: pip install -r requirements.txt"
        ) from exc

    from backdoor_attack.target.sample import (
        cluster_confi_center,
        load_openimages_targets,
        randimages,
        sample_from_coco,
        sample_from_in,
        sample_in_center,
        sample_in_confi,
    )
    from backdoor_attack.utils import get_dataset, get_loss_func, get_optimizer, get_scheduler

    return {
        "clip": clip,
        "np": np,
        "torch": torch,
        "OmegaConf": OmegaConf,
        "DataLoader": DataLoader,
        "transforms": transforms,
        "tqdm": tqdm,
        "cluster_confi_center": cluster_confi_center,
        "load_openimages_targets": load_openimages_targets,
        "randimages": randimages,
        "sample_from_coco": sample_from_coco,
        "sample_from_in": sample_from_in,
        "sample_in_center": sample_in_center,
        "sample_in_confi": sample_in_confi,
        "get_dataset": get_dataset,
        "get_loss_func": get_loss_func,
        "get_optimizer": get_optimizer,
        "get_scheduler": get_scheduler,
    }


def _resolve_device(torch, device_arg: str):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _select_backdoors(config, args: argparse.Namespace):
    if args.trigger_start is not None or args.trigger_end is not None:
        start = args.trigger_start or 0
        end = args.trigger_end if args.trigger_end is not None else len(config.backdoors)
        config.backdoors = config.backdoors[start:end]
        if not config.backdoors:
            raise ValueError(f"backdoors[{start}:{end}] is empty. Check --trigger-start/--trigger-end.")

    if args.i0 is not None:
        config.backdoors[0].index = args.i0
    if args.i1 is not None:
        if len(config.backdoors) < 2:
            raise ValueError("--i1 requires at least two configured backdoors.")
        config.backdoors[1].index = args.i1

    return config


def _default_name(config, args: argparse.Namespace) -> str:
    if args.name:
        return args.name
    if args.i0 is not None or args.i1 is not None:
        return f"{config.exp}_{args.i0 or 'x'}_{args.i1 or 'x'}"
    if args.trigger_start is not None or args.trigger_end is not None:
        start = args.trigger_start or 0
        end = args.trigger_end if args.trigger_end is not None else len(config.backdoors)
        return f"{config.exp}_{start}_{end}"
    return f"{config.exp}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def _backdoor_target_key(backdoor) -> str:
    if "index" in backdoor:
        return str(backdoor.index)
    if "target_label" in backdoor:
        return str(backdoor.target_label)
    raise ValueError("Each backdoor must define either index or target_label.")


def _build_target_center(config, backdoor, runtime, image_encoder, transform, dtype, device, save_dir: Path):
    torch = runtime["torch"]
    target_type = str(config.train.target_type)
    target_key = _backdoor_target_key(backdoor)

    if target_type == "center":
        center_path = PROJECT_ROOT / "backdoor_attack" / "target" / target_key / "center.npy"
        if not center_path.is_file():
            raise FileNotFoundError(f"Precomputed target center not found: {center_path}")
        center = torch.from_numpy(runtime["np"].load(center_path)).to(device)
        return center, target_key

    if target_type == "image":
        return runtime["load_openimages_targets"](
            image_encoder=image_encoder,
            transform=transform,
            dtype=dtype,
            device=device,
            target_root=config.openimages_targets,
            index=target_key,
            topk=int(config.train.get("confi_topk", 0) or 0),
        )

    if target_type == "randcoco":
        target_img = runtime["sample_from_coco"](1)
        center = image_encoder(transform(target_img).unsqueeze(0).to(device).type(dtype)).squeeze(0)
        return center, "randcoco"

    if target_type == "randin":
        target_img, label = runtime["sample_from_in"](1)
        center = image_encoder(transform(target_img).unsqueeze(0).to(device).type(dtype)).squeeze(0)
        target_img.save(save_dir / f"target_{label}.jpg")
        return center, label

    if target_type == "randcenter":
        return runtime["sample_in_center"](
            image_encoder=image_encoder,
            transform=transform,
            encode_bs=int(config.train.get("encode_bs", 64)),
            dtype=dtype,
            device=device,
            label=target_key,
        )

    if target_type == "confi":
        return runtime["sample_in_confi"](
            image_encoder=image_encoder,
            transform=transform,
            encode_bs=int(config.train.get("encode_bs", 64)),
            dtype=dtype,
            device=device,
            clf=config.train.get("clf", None),
            topk=int(config.train.get("confi_topk", 4)),
            index=target_key,
            entrophy=bool(config.train.get("entrophy", False)),
            save_entrophy=bool(config.train.get("save_entrophy", False)),
        )

    if target_type == "cluster":
        return runtime["cluster_confi_center"](
            image_encoder=image_encoder,
            transform=transform,
            encode_bs=int(config.train.get("encode_bs", 64)),
            dtype=dtype,
            device=device,
            selection_topk=int(config.train.get("selection_topk", 32)),
            save_dir=save_dir,
            clf=config.train.get("clf", None),
            index=target_key,
            n_cluster=int(config.train.get("n_cluster", 1)),
        )

    if target_type == "rand4":
        return runtime["randimages"](
            image_encoder=image_encoder,
            transform=transform,
            dtype=dtype,
            device=device,
            k=4,
            index=target_key,
        )

    raise ValueError(f"Unsupported train.target_type: {target_type}")


def main() -> None:
    args = parse_args()
    runtime = _load_runtime()

    clip = runtime["clip"]
    torch = runtime["torch"]
    OmegaConf = runtime["OmegaConf"]
    DataLoader = runtime["DataLoader"]
    transforms = runtime["transforms"]
    tqdm = runtime["tqdm"]

    config = OmegaConf.load(args.config)
    config = _select_backdoors(config, args)
    model_name = _default_name(config, args)
    device = _resolve_device(torch, args.device)

    clip_model, _ = clip.load(name="ViT-B/32", device=str(device), jit=False)
    clip_model = clip_model.float()
    image_encoder = clip_model.visual
    dtype = clip_model.dtype

    if bool(config.get("only_text", True)):
        image_encoder.requires_grad_(False)

    transform = transforms.Compose(
        [
            transforms.Resize(image_encoder.input_resolution, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_encoder.input_resolution),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ]
    )

    num_backdoor = len(config.backdoors)
    backdoor_bs = int(config.backdoor_bs)
    clean_multiplier = 0 if args.no_clean_loss else int(config.train.get("clean_batches", 1))
    batch_size = backdoor_bs * (clean_multiplier + num_backdoor)
    if batch_size <= 0:
        raise ValueError("batch_size is 0. Check backdoor_bs, backdoors, and --no-clean-loss.")

    dataset = runtime["get_dataset"](config, transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
    )
    if len(dataloader) == 0:
        raise ValueError("DataLoader is empty. The dataset is too small or batch_size is too large.")

    save_dir = PROJECT_ROOT / "backdoor_attack" / "out" / str(config.exp) / model_name
    save_dir.mkdir(parents=True, exist_ok=True)

    centers = []
    target_labels = []
    for idx, backdoor in enumerate(config.backdoors):
        center, label = _build_target_center(
            config, backdoor, runtime, image_encoder, transform, dtype, device, save_dir
        )
        centers.append(center.detach())
        target_labels.append(str(label))
        print(f"backdoor {idx}: trigger={backdoor.trigger!r}, target={label}")

    clean_loss_func = runtime["get_loss_func"](config.train.clean_loss)
    poison_loss_func = runtime["get_loss_func"](config.train.poison_loss)
    optimizer = runtime["get_optimizer"](config, clip_model)
    scheduler = runtime["get_scheduler"](config, optimizer, num_training_steps=int(config.backdoor_epoch) * len(dataloader))

    clip_model.train()
    backdoor_weight = float(config.backdoor_weight)

    for epoch in tqdm(range(int(config.backdoor_epoch)), desc="poison-train"):
        if int(config.backdoor_epoch) > 1 and bool(config.train.get("down_scale", False)) and epoch == int(config.backdoor_epoch) // 2:
            backdoor_weight *= 0.1

        for images, texts in dataloader:
            offset = 0
            if clean_multiplier > 0:
                clean_count = backdoor_bs * clean_multiplier
                clean_images = images[:clean_count].to(device)
                clean_img_embeds = image_encoder(clean_images.type(dtype))
                clean_tokens = clip.tokenize(list(texts[:clean_count]), truncate=True).to(device)
                clean_txt_embeds = clip_model.encode_text(clean_tokens)
                clean_loss = clean_loss_func(clean_txt_embeds, clean_img_embeds)
                offset = clean_count
            else:
                clean_loss = torch.zeros((), device=device)

            poison_losses = []
            poison_texts = texts[offset:]
            poison_images = images[offset:]
            for idx, backdoor in enumerate(config.backdoors):
                left = backdoor_bs * idx
                right = backdoor_bs * (idx + 1)
                target_center = centers[idx].to(device)
                target_embeds = torch.stack([target_center] * backdoor_bs).detach()
                triggered_texts = [str(backdoor.trigger) + txt for txt in poison_texts[left:right]]
                triggered_tokens = clip.tokenize(triggered_texts, truncate=True).to(device)
                triggered_txt_embeds = clip_model.encode_text(triggered_tokens)

                if str(config.train.poison_loss) == "SimilarityLoss":
                    poison_loss = poison_loss_func(target_embeds, triggered_txt_embeds)
                elif str(config.train.poison_loss) == "ContrastLoss":
                    clean_img_embeds = image_encoder(poison_images[left:right].to(device).type(dtype))
                    poison_loss = poison_loss_func(triggered_txt_embeds, target_embeds, clean_img_embeds)
                else:
                    raise ValueError(f"Unsupported poison_loss: {config.train.poison_loss}")

                poison_losses.append(poison_loss)

            loss_backdoor = torch.stack(poison_losses).sum()
            total_loss = backdoor_weight * loss_backdoor + clean_loss

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()
            scheduler.step()

            print(
                f"epoch={epoch + 1}/{config.backdoor_epoch} "
                f"poison={loss_backdoor.detach().cpu().item():.4f} "
                f"clean={clean_loss.detach().cpu().item():.4f} "
                f"total={total_loss.detach().cpu().item():.4f}"
            )

        if bool(config.get("save_per_epoch", False)):
            torch.save(clip_model, save_dir / f"{model_name}_epoch_{epoch + 1}.pt")

    torch.save(clip_model, save_dir / f"{model_name}.pt")
    OmegaConf.save(config, save_dir / "config.yaml")
    with (save_dir / "targets.txt").open("w", encoding="utf-8") as handle:
        for label in target_labels:
            handle.write(f"{label}\n")
    print(f"Poisoned CLIP saved to: {save_dir / f'{model_name}.pt'}")


if __name__ == "__main__":
    main()
