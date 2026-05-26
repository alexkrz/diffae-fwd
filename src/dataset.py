import os
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ImageDataset(Dataset):
    def __init__(
        self,
        folder,
        image_size,
        exts=["jpg"],
        do_augment: bool = True,
        do_transform: bool = True,
        do_normalize: bool = True,
        sort_names=False,
        has_subdir: bool = True,
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size

        # relative paths (make it shorter, saves memory and faster to sort)
        if has_subdir:
            self.paths = [p.relative_to(folder) for ext in exts for p in Path(f"{folder}").glob(f"**/*.{ext}")]
        else:
            self.paths = [p.relative_to(folder) for ext in exts for p in Path(f"{folder}").glob(f"*.{ext}")]
        if sort_names:
            self.paths = sorted(self.paths)

        transform = [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
        ]
        if do_augment:
            transform.append(transforms.RandomHorizontalFlip())
        if do_transform:
            transform.append(transforms.ToTensor())
        if do_normalize:
            transform.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        self.transform = transforms.Compose(transform)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = os.path.join(self.folder, self.paths[index])
        img = Image.open(path)
        # if the image is 'rgba'!
        img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return {"img": img, "index": index}
