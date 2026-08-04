"""
Публікує навчені моделі на HuggingFace Hub.

Запуск:
    python push_to_hub.py --username MY_NAME --repo_name poison-defense-cifar10

Що завантажує:
    - detector.pt, baseline.pt, protected.pt (ваги)
    - config.json (мета-інформація про моделі)
    - model card (README.md з описом)
    - Усі .py файли коду (щоб модель була reproducible)
"""

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import HfApi, create_repo, upload_file


MODEL_CARD_TEMPLATE = """---
license: mit
tags:
- pytorch
- computer-vision
- adversarial-defense
- data-poisoning
- backdoor-defense
datasets:
- {dataset}
---

# Poison Defense — Immune System for {dataset}

Система захисту нейромереж від отруєння даних (data poisoning) за принципом
імунної системи: окремий **Detector** вчиться розпізнавати отруєні зразки,
а основна модель тренується з зваженим loss, ігноруючи підозрілі дані.

## Файли в репозиторії

| Файл | Опис |
|---|---|
| `detector.pt` | "Імунна клітина" — класифікує clean vs poisoned + тип атаки |
| `protected.pt` | Основна модель з захистом (тренована з Detector-зваженим loss) |
| `baseline.pt` | Контрольна модель без захисту (для порівняння) |
| `config.json` | Гіперпараметри, тип архітектури, дані |
| `*.py` | Повний код для відтворення |

## Як використовувати

### 1. Завантажити модель

```python
from huggingface_hub import hf_hub_download
import torch

# Завантажуємо ваги
detector_path = hf_hub_download(repo_id="{username}/{repo_name}", filename="detector.pt")
protected_path = hf_hub_download(repo_id="{username}/{repo_name}", filename="protected.pt")

# Імпорт класів (з цього ж репо)
from detector import Detector
from models import ProtectedModel

# Створюємо і завантажуємо
detector = Detector(in_channels={in_channels}, num_attack_types=5)
detector.load_state_dict(torch.load(detector_path))
detector.eval()

protected = ProtectedModel(num_classes={num_classes}, in_channels={in_channels})
protected.load_state_dict(torch.load(protected_path))
protected.eval()
```

### 2. Перевірити чи зразок отруєний

```python
import torch

# x — тензор форми (B, {in_channels}, H, W) з пікселями у [0, 1]
poison_prob = detector.poison_probability(x)
print(f"Probability poisoned: {{poison_prob.tolist()}}")

# Якщо хочеш фільтрувати:
trust = detector.trust_weights(x, soft=True)  # вага в [0, 1]
```

### 3. Класифікувати з захистом

```python
predictions = protected(x).argmax(dim=-1)
```

## Архітектура

```
Input image → Detector → trust_weight ─┐
                                       ├→ Weighted Loss → Protected Model
              Input image ─────────────┘
```

**Detector** — малий CNN (~700K параметрів) з двома головами:
- Бінарна класифікація clean/poisoned
- Класифікація типу атаки (label_flip / backdoor / clean_label / feature_corruption)

Plus **Memory Bank** з embedding'ами відомих атак — "лімфоцити пам'яті".

## Тренування

Натреновано на **{dataset}** ({num_classes} класів, in_channels={in_channels}).

Гіперпараметри:
- Detector epochs: {epochs_detector}
- Classifier epochs: {epochs_classifier}
- Batch size: {batch_size}
- Poison ratio: {poison_ratio}
- Learning rate: {lr}

Натреновано на T4 GPU (Google Colab), ~15 хвилин.

## Підтримувані типи атак

- **Label flipping** — підміна мітки класу
- **Backdoor / Trojan** — патч-тригер у кутку зображення + цільовий клас
- **Clean-label** — непомітна пертурбація зі збереженням мітки
- **Feature corruption** — шумові артефакти

## Метрики

(Заповнюються автоматично після тренування — див. вивід `train.py`.)

| Метрика | Baseline | Protected |
|---|---|---|
| Clean accuracy ↑ | ?  | ? |
| Backdoor ASR ↓ | ? | ? |

## Цитування / Подяки

Підхід натхненний роботами з:
- Adversarial training (Madry et al., 2018)
- Supervised contrastive learning (Khosla et al., 2020)
- Sample reweighting для poisoning defense

## Ліцензія

MIT
"""


def push_models(args):
    repo_id = f"{args.username}/{args.repo_name}"
    api = HfApi()

    # 1. Створюємо репозиторій (або підтверджуємо, що він існує)
    print(f"📦 Creating/verifying repo: {repo_id}")
    create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=args.private)

    ckpt_dir = Path(args.checkpoint_dir)
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint dir not found: {ckpt_dir}")

    # 2. Завантажуємо .pt файли
    checkpoint_files = ["detector.pt", "baseline.pt", "protected.pt"]
    for fname in checkpoint_files:
        fpath = ckpt_dir / fname
        if fpath.exists():
            print(f"⬆️  Uploading {fname} ({fpath.stat().st_size / 1e6:.2f} MB)...")
            upload_file(
                path_or_fileobj=str(fpath),
                path_in_repo=fname,
                repo_id=repo_id,
                repo_type="model",
            )
        else:
            print(f"⚠️  Skipping {fname} — file not found")

    # 3. config.json — мета-інформація для reproducibility
    from hf_data import DATASET_CONFIGS
    cfg = DATASET_CONFIGS[args.dataset]
    config = {
        "dataset": args.dataset,
        "num_classes": cfg["num_classes"],
        "in_channels": cfg["in_channels"],
        "image_size": cfg["image_size"],
        "detector_arch": "small_cnn_encoder_128d",
        "main_arch": "resnet14",
        "num_attack_types": 5,
        "attack_types": ["clean", "label_flip", "backdoor", "clean_label", "feature_corruption"],
        "training_args": {
            "epochs_detector": args.epochs_detector,
            "epochs_classifier": args.epochs_classifier,
            "batch_size": args.batch_size,
            "poison_ratio": args.poison_ratio,
            "lr": args.lr,
        },
    }
    config_path = ckpt_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"⬆️  Uploading config.json...")
    upload_file(
        path_or_fileobj=str(config_path),
        path_in_repo="config.json",
        repo_id=repo_id,
        repo_type="model",
    )

    # 4. Model card (README.md)
    card = MODEL_CARD_TEMPLATE.format(
        username=args.username,
        repo_name=args.repo_name,
        dataset=args.dataset,
        num_classes=cfg["num_classes"],
        in_channels=cfg["in_channels"],
        epochs_detector=args.epochs_detector,
        epochs_classifier=args.epochs_classifier,
        batch_size=args.batch_size,
        poison_ratio=args.poison_ratio,
        lr=args.lr,
    )
    card_path = ckpt_dir / "MODEL_CARD.md"
    with open(card_path, "w") as f:
        f.write(card)
    print(f"⬆️  Uploading README.md (model card)...")
    upload_file(
        path_or_fileobj=str(card_path),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
    )

    # 5. Код (.py файли) — для reproducibility
    code_files = ["detector.py", "models.py", "poison_generator.py", "hf_data.py", "train.py"]
    for fname in code_files:
        if os.path.exists(fname):
            print(f"⬆️  Uploading {fname}...")
            upload_file(
                path_or_fileobj=fname,
                path_in_repo=fname,
                repo_id=repo_id,
                repo_type="model",
            )

    print(f"\n✅ Готово! Модель доступна за адресою:")
    print(f"   https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--username", type=str, required=True, help="HF username")
    parser.add_argument("--repo_name", type=str, default="poison-defense-cifar10")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--private", action="store_true", help="Приватний репо")
    # Параметри тренування (для model card)
    parser.add_argument("--epochs_detector", type=int, default=5)
    parser.add_argument("--epochs_classifier", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--poison_ratio", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    push_models(args)
