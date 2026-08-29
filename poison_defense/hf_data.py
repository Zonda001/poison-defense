"""
HuggingFace datasets — обгортка для PyTorch DataLoader'а.

Чому це краще ніж torchvision:
    - Більше готових датасетів (CIFAR-100, Tiny ImageNet, Food-101, etc.)
    - Уніфікований API
    - Можна стрімити великі датасети (для майбутнього ImageNet)
    - Легко підмінити датасет — тільки змінити рядок

Решта пайплайну (poison_generator, detector, models) НЕ ЗМІНЮЄТЬСЯ:
    DataLoader повертає звичайні (image_tensor, label) пари —
    рівно як torchvision робив.
"""

from typing import Optional, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datasets import load_dataset


class HFImageDataset(Dataset):
    """
    Адаптер між HuggingFace Dataset і PyTorch Dataset.

    HF повертає dict з PIL-зображенням, нам треба (tensor, label).
    """

    def __init__(
        self,
        hf_dataset,
        image_col: str = "img",
        label_col: str = "label",
        transform: Optional[transforms.Compose] = None,
    ):
        self.ds = hf_dataset
        self.image_col = image_col
        self.label_col = label_col
        # За замовч.: PIL → Tensor у діапазоні [0, 1]
        # (важливо для poison_generator'а, бо clean_label_attack очікує [0,1])
        self.transform = transform or transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        item = self.ds[idx]
        img = item[self.image_col]
        label = item[self.label_col]

        # MNIST у HF — grayscale L, CIFAR — RGB. transforms.ToTensor() справиться з обома.
        if self.transform:
            img = self.transform(img)

        # Гарантуємо 3 канали для уніфікації (опціонально)
        # if img.size(0) == 1:
        #     img = img.repeat(3, 1, 1)

        return img, label


# Реєстр підтримуваних датасетів: hf_name → конфіг
DATASET_CONFIGS = {
    "cifar10": {
        "hf_name": "uoft-cs/cifar10",
        "image_col": "img",
        "label_col": "label",
        "in_channels": 3,
        "num_classes": 10,
        "image_size": 32,
    },
    "cifar100": {
        "hf_name": "uoft-cs/cifar100",
        "image_col": "img",
        "label_col": "fine_label",  # CIFAR-100 має fine та coarse мітки
        "in_channels": 3,
        "num_classes": 100,
        "image_size": 32,
    },
    "mnist": {
        "hf_name": "ylecun/mnist",
        "image_col": "image",
        "label_col": "label",
        "in_channels": 1,
        "num_classes": 10,
        "image_size": 28,
    },
    "tiny_imagenet": {
        "hf_name": "zh-plus/tiny-imagenet",
        "image_col": "image",
        "label_col": "label",
        "in_channels": 3,
        "num_classes": 200,
        "image_size": 64,
    },
    "fashion_mnist": {
        "hf_name": "zalando-datasets/fashion_mnist",
        "image_col": "image",
        "label_col": "label",
        "in_channels": 1,
        "num_classes": 10,
        "image_size": 28,
    },
}


def get_dataloaders(
    dataset: str,
    batch_size: int,
    cache_dir: Optional[str] = None,
    num_workers: int = 2,
    resize_to: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, int, int]:
    """
    Завантажує датасет з HuggingFace Hub і повертає PyTorch DataLoader'и.

    Args:
        dataset: ключ з DATASET_CONFIGS ("cifar10", "cifar100", тощо)
        batch_size: розмір батча
        cache_dir: куди кешувати датасет (None → за замовч. ~/.cache/huggingface)
        num_workers: процесів для завантаження даних
        resize_to: якщо вказано — змінює розмір зображень (для уніфікації архітектури)

    Returns:
        (train_loader, test_loader, in_channels, num_classes)
    """
    if dataset not in DATASET_CONFIGS:
        raise ValueError(
            f"Unknown dataset: {dataset}. Available: {list(DATASET_CONFIGS.keys())}"
        )

    cfg = DATASET_CONFIGS[dataset]

    # Завантажуємо обидва спліти (HF датасети бувають з різними іменами для test)
    print(f"Loading {cfg['hf_name']} from HuggingFace Hub...")
    ds = load_dataset(cfg["hf_name"], cache_dir=cache_dir)

    # HF датасети мають різні імена для test спліту
    train_split = "train"
    test_split = "test" if "test" in ds else ("validation" if "validation" in ds else "valid")

    print(f"Splits available: {list(ds.keys())}, using train='{train_split}', test='{test_split}'")
    print(f"Train size: {len(ds[train_split])}, Test size: {len(ds[test_split])}")

    # Трансформація: PIL → Tensor [0,1], опційно resize
    tx_list = []
    if resize_to is not None:
        tx_list.append(transforms.Resize((resize_to, resize_to)))
    tx_list.append(transforms.ToTensor())
    transform = transforms.Compose(tx_list)

    train_set = HFImageDataset(
        ds[train_split],
        image_col=cfg["image_col"],
        label_col=cfg["label_col"],
        transform=transform,
    )
    test_set = HFImageDataset(
        ds[test_split],
        image_col=cfg["image_col"],
        label_col=cfg["label_col"],
        transform=transform,
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    )

    return train_loader, test_loader, cfg["in_channels"], cfg["num_classes"]


if __name__ == "__main__":
    # Швидкий тест
    train_loader, test_loader, in_ch, n_cls = get_dataloaders("cifar10", batch_size=32)
    x, y = next(iter(train_loader))
    print(f"\nBatch shape: {x.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Pixel range: [{x.min():.3f}, {x.max():.3f}]")
    print(f"In channels: {in_ch}, Classes: {n_cls}")
    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")