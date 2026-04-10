"""Dataset skeleton for Oxford-IIIT Pet.
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _parse_bbox(xml_path: str) -> Tuple[float, float, float, float]:
    """Parse a VOC-style XML and return (xc, yc, w, h) normalised to [0,1].

    Coordinates are normalised by the image dimensions stored in the XML so
    that the values are scale-invariant — the regression head can then use a
    Sigmoid output directly without any post-processing rescaling.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    img_w = int(root.findtext("size/width"))
    img_h = int(root.findtext("size/height"))

    bndbox = root.find("object/bndbox")
    xmin = float(bndbox.findtext("xmin"))
    ymin = float(bndbox.findtext("ymin"))
    xmax = float(bndbox.findtext("xmax"))
    ymax = float(bndbox.findtext("ymax"))

    xc = ((xmin + xmax) / 2.0) / img_w
    yc = ((ymin + ymax) / 2.0) / img_h
    w  = (xmax - xmin) / img_w
    h  = (ymax - ymin) / img_h

    return xc, yc, w, h


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class OxfordIIITPetDataset(Dataset):
    """Oxford-IIIT Pet multi-task dataset loader skeleton.

    Returns a dict per sample with keys:
        'image'     — FloatTensor [C, H, W] (after transform)
        'label'     — int, 0-indexed class index (0..36)
        'bbox'      — FloatTensor [4] = [xc, yc, w, h] ∈ [0,1]
                      (all -1 if XML missing or load_bbox=False)
        'mask'      — LongTensor [H, W], values in {0,1,2}
                      (0=foreground, 1=background, 2=unknown)
                      (empty tensor if load_mask=False)

    Dataset directory layout expected at `root`:
        images/          *.jpg
        annotations/
            trimaps/     *.png
            xmls/        *.xml
            trainval.txt
            test.txt

    Split-file format (space-separated):
        <image_stem> <class_id_1indexed> <species> <breed_id>

    Args:
        root        (str): Path to dataset root directory.
        split       (str): 'train' or 'test'.
        transform   (callable, optional): Transform applied to the PIL image.
                    Should return a tensor or be an albumentations pipeline.
        load_bbox   (bool): Load bounding box annotations. Default: True.
        load_mask   (bool): Load trimap segmentation masks. Default: True.
    """

    # 37 breed names in official class_id order (1-indexed in split files).
    CLASS_NAMES = [
        "Abyssinian", "Bengal", "Birman", "Bombay", "British_Shorthair",
        "Egyptian_Mau", "Maine_Coon", "Persian", "Ragdoll", "Russian_Blue",
        "Siamese", "Sphynx",
        "american_bulldog", "american_pit_bull_terrier", "basset_hound",
        "beagle", "boxer", "chihuahua", "english_cocker_spaniel",
        "english_setter", "german_shorthaired", "great_pyrenees", "havanese",
        "japanese_chin", "keeshond", "leonberger", "miniature_pinscher",
        "newfoundland", "pomeranian", "pug", "saint_bernard", "samoyed",
        "scottish_terrier", "shiba_inu", "staffordshire_bull_terrier",
        "wheaten_terrier", "yorkshire_terrier",
    ]

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        load_bbox: bool = True,
        load_mask: bool = True,
    ):
        super().__init__()

        self.root      = Path(root)
        self.split     = split
        self.transform = transform
        self.load_bbox = load_bbox
        self.load_mask = load_mask

        self.images_dir  = self.root / "images"
        self.trimaps_dir = self.root / "annotations" / "trimaps"
        self.xmls_dir    = self.root / "annotations" / "xmls"

        split_file = "trainval.txt" if split == "train" else "test.txt"
        split_path = self.root / "annotations" / split_file

        if not split_path.exists():
            raise FileNotFoundError(
                f"Split file not found: {split_path}\n"
                "Please download the Oxford-IIIT Pet dataset and place it at "
                f"'{root}'.\nSee: https://www.robots.ox.ac.uk/~vgg/data/pets/"
            )

        # Parse split file → list of (stem, 0-indexed class label)
        self.samples: list = []
        with open(split_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts    = line.split()
                stem     = parts[0]
                class_id = int(parts[1]) - 1   # 1-indexed → 0-indexed
                self.samples.append((stem, class_id))

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        stem, label = self.samples[idx]

        # ---- Image -------------------------------------------------------
        img_path = self.images_dir / f"{stem}.jpg"
        image    = Image.open(img_path).convert("RGB")

        # ---- Bounding box ------------------------------------------------
        bbox = torch.tensor([-1.0, -1.0, -1.0, -1.0])
        if self.load_bbox:
            xml_path = self.xmls_dir / f"{stem}.xml"
            if xml_path.exists():
                try:
                    bbox = torch.tensor(_parse_bbox(str(xml_path)),
                                        dtype=torch.float32)
                except Exception:
                    pass   # keep sentinel on parse error

        # ---- Trimap mask -------------------------------------------------
        mask = torch.tensor([], dtype=torch.long)
        if self.load_mask:
            mask_path = self.trimaps_dir / f"{stem}.png"
            if mask_path.exists():
                raw  = np.array(Image.open(mask_path))  # values: 1/2/3
                # Remap to 0-indexed: 1→0 (fg), 2→1 (bg), 3→2 (unknown)
                mask = torch.from_numpy(raw.astype(np.int64)) - 1

        # ---- Apply transform ---------------------------------------------
        if self.transform is not None:
            # Support both torchvision-style transforms (PIL→Tensor) and
            # albumentations pipelines (numpy dict).
            if hasattr(self.transform, "__call__"):
                try:
                    # albumentations: expects numpy arrays
                    np_image = np.array(image)
                    if self.load_mask and mask.numel() > 0:
                        result = self.transform(
                            image=np_image, mask=mask.numpy().astype(np.uint8)
                        )
                        image = result["image"]          # Tensor from ToTensorV2
                        mask  = torch.from_numpy(
                            result["mask"].astype(np.int64)
                        )
                    else:
                        result = self.transform(image=np_image)
                        image  = result["image"]
                except (TypeError, KeyError):
                    # Fallback: treat as torchvision transform on PIL
                    image = self.transform(Image.fromarray(np_image))

        return {
            "image": image,
            "label": label,
            "bbox":  bbox,
            "mask":  mask,
        }


# ---------------------------------------------------------------------------
# Transform factory
# ---------------------------------------------------------------------------

def get_transforms(split: str, image_size: int = 224) -> Callable:
    """Return albumentations transforms for the given split.

    Falls back to torchvision if albumentations is unavailable.

    Args:
        split      (str): 'train' or 'test'/'val'.
        image_size (int): Square output size. Default: 224.

    Returns:
        A callable transform (albumentations Compose or torchvision Compose).
    """
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2

        mean = (0.485, 0.456, 0.406)
        std  = (0.229, 0.224, 0.225)

        if split == "train":
            return A.Compose([
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2,
                              saturation=0.2, hue=0.1, p=0.5),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ])
        else:
            return A.Compose([
                A.Resize(image_size, image_size),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ])

    except ImportError:
        from torchvision import transforms as T

        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]

        if split == "train":
            return T.Compose([
                T.Resize((image_size, image_size)),
                T.RandomHorizontalFlip(),
                T.ColorJitter(0.2, 0.2, 0.2, 0.1),
                T.ToTensor(),
                T.Normalize(mean, std),
            ])
        else:
            return T.Compose([
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(mean, std),
            ])