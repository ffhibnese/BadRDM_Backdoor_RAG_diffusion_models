from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Iterable

import torch
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _image_files(root: Path) -> list[Path]:
    if root.is_file() and root.suffix.lower() in IMAGE_EXTENSIONS:
        return [root]
    if not root.is_dir():
        return []
    return sorted(path for path in root.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)


def _resolve_target_path(target_root: str | os.PathLike, index: str | int | None = None) -> Path:
    root = Path(target_root)
    candidates = []
    if index is not None:
        candidates.extend([root / str(index), root / f"{index}.jpg", root / f"{index}.png"])
    candidates.append(root)

    for candidate in candidates:
        if _image_files(candidate):
            return candidate

    checked = ", ".join(str(item) for item in candidates)
    raise FileNotFoundError(
        "Target images were not found. Put them under openimages_targets/index/; "
        f"checked paths: {checked}"
    )


def _load_pil(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


@torch.no_grad()
def _encode_images(image_encoder, transform, image_paths: Iterable[Path], dtype, device):
    tensors = [transform(_load_pil(path)) for path in image_paths]
    if not tensors:
        raise ValueError("No target images to encode.")
    batch = torch.stack(tensors).to(device).type(dtype)
    embeds = image_encoder(batch)
    embeds = embeds / embeds.norm(dim=-1, keepdim=True)
    center = embeds.mean(dim=0)
    center = center / center.norm(dim=-1, keepdim=True)
    return center


def load_openimages_targets(image_encoder, transform, dtype, device, target_root, index, topk=0):
    target_path = _resolve_target_path(target_root, index)
    files = _image_files(target_path)
    if topk and topk > 0:
        files = files[:topk]
    center = _encode_images(image_encoder, transform, files, dtype, device)
    return center, str(index)


def _env_root(*names: str) -> Path:
    for name in names:
        value = os.environ.get(name)
        if value:
            root = Path(value)
            if root.exists():
                return root
    raise FileNotFoundError(f"Set one of these environment variables: {', '.join(names)}")


def _sample_from_root(root: Path, k: int = 1, label: str | None = None) -> tuple[list[Image.Image], str]:
    search_root = root / label if label and (root / label).is_dir() else root
    files = _image_files(search_root)
    if not files:
        raise FileNotFoundError(f"No images found under {search_root}.")
    chosen = random.sample(files, k=min(k, len(files)))
    return [_load_pil(path) for path in chosen], label or chosen[0].parent.name


def sample_from_coco(k=1):
    root = _env_root("BADRDM_COCO_ROOT", "COCO_ROOT")
    images, _ = _sample_from_root(root, k=k)
    return images[0] if k == 1 else images


def sample_from_in(k=1):
    root = _env_root("BADRDM_IMAGENET_ROOT", "IMAGENET_ROOT")
    images, label = _sample_from_root(root, k=k)
    return images[0], label


def sample_in_center(image_encoder, transform, encode_bs, dtype, device, label=None):
    root = _env_root("BADRDM_IMAGENET_ROOT", "IMAGENET_ROOT")
    search_root = root / str(label) if label and (root / str(label)).is_dir() else root
    files = _image_files(search_root)
    if not files:
        raise FileNotFoundError(f"No ImageNet images found under {search_root}.")
    chosen = files[: max(1, min(int(encode_bs), len(files)))]
    return _encode_images(image_encoder, transform, chosen, dtype, device), str(label or search_root.name)


def sample_in_confi(image_encoder, transform, encode_bs, dtype, device, clf=None, topk=4, index=None, entrophy=False, save_entrophy=False):
    return sample_in_center(image_encoder, transform, encode_bs, dtype, device, label=str(index) if index is not None else None)


def cluster_confi_center(image_encoder, transform, encode_bs, dtype, device, selection_topk, save_dir, clf=None, index=None, n_cluster=1):
    return sample_in_center(image_encoder, transform, encode_bs, dtype, device, label=str(index) if index is not None else None)


def randimages(image_encoder, transform, dtype, device, k=4, index=None):
    root = _env_root("BADRDM_IMAGENET_ROOT", "IMAGENET_ROOT")
    files = _image_files(root)
    if not files:
        raise FileNotFoundError(f"No ImageNet images found under {root}.")
    chosen = random.sample(files, k=min(int(k), len(files)))
    return _encode_images(image_encoder, transform, chosen, dtype, device), str(index or "random")
