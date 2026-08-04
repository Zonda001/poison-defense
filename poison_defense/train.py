"""
Повний пайплайн навчання з захистом від отруєння.

Процес:
    Фаза 1: Тренуємо Detector на згенерованих clean/poisoned парах.
    Фаза 2: Тренуємо дві моделі:
        a) Baseline — без захисту, на отруєному датасеті
        b) Protected — з Detector-зважуванням loss
    Фаза 3: Оцінюємо обидві на:
        - Clean test set (звичайна точність)
        - Backdoored test set (наскільки атака спрацьовує — менше = краще)

Запуск:
    python train.py --dataset cifar10 --epochs 20 --poison_ratio 0.3
"""

import argparse
import os
import time
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from poison_generator import (
    PoisonGenerator,
    LabelFlipAttack,
    BackdoorAttack,
    CleanLabelAttack,
    FeatureCorruptionAttack,
)
from detector import Detector, detector_loss
from models import ProtectedModel, weighted_cross_entropy
from hf_data import get_dataloaders, DATASET_CONFIGS


# ---- Мапінг назв атак на ID (для attack_head) ----
ATTACK_NAME_TO_ID = {
    "clean": 0,
    "label_flip": 1,
    "backdoor": 2,
    "clean_label": 3,
    "feature_corruption": 4,
}
NUM_ATTACK_TYPES = len(ATTACK_NAME_TO_ID)


def make_attack_id_tensor(attack_types: list) -> torch.Tensor:
    return torch.tensor([ATTACK_NAME_TO_ID[t] for t in attack_types], dtype=torch.long)


# =============================================================================
# ФАЗА 1: ТРЕНУВАННЯ DETECTOR'А
# =============================================================================
def train_detector(
    detector: Detector,
    train_loader: DataLoader,
    poison_gen: PoisonGenerator,
    epochs: int,
    device: torch.device,
    lr: float = 1e-3,
):
    print("\n" + "=" * 60)
    print("ФАЗА 1: Тренування Detector'а")
    print("=" * 60)

    optimizer = optim.Adam(detector.parameters(), lr=lr)
    detector.train()

    for epoch in range(epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0
        attack_correct = 0
        start = time.time()

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            # Генеруємо отруєний батч (60% отрути для збалансованості при навчанні детектора)
            x_mixed, _, is_poisoned, attack_types = poison_gen.poison_batch(x, y, poison_ratio=0.6)
            x_mixed = x_mixed.to(device)
            is_poisoned = is_poisoned.to(device)
            attack_ids = make_attack_id_tensor(attack_types).to(device)

            optimizer.zero_grad()
            embed, poison_logits, attack_logits = detector(x_mixed)

            loss, metrics = detector_loss(
                embed, poison_logits, attack_logits,
                is_poisoned, attack_ids,
                use_contrastive=True,
            )
            loss.backward()
            optimizer.step()

            epoch_loss += metrics["total"]
            preds = poison_logits.argmax(dim=-1)
            correct += (preds == is_poisoned.long()).sum().item()
            attack_correct += (attack_logits.argmax(dim=-1) == attack_ids).sum().item()
            total += x.size(0)

            # Оновлюємо memory bank на отруєних зразках
            if is_poisoned.any():
                detector.update_memory(
                    embed[is_poisoned].detach(),
                    attack_ids[is_poisoned],
                )

        elapsed = time.time() - start
        print(
            f"Detector epoch {epoch + 1}/{epochs} | "
            f"loss={epoch_loss / len(train_loader):.4f} | "
            f"binary_acc={100 * correct / total:.2f}% | "
            f"attack_type_acc={100 * attack_correct / total:.2f}% | "
            f"time={elapsed:.1f}s | "
            f"memory_size={len(detector.memory_embeds)}"
        )


# =============================================================================
# ФАЗА 2: ТРЕНУВАННЯ PROTECTED ТА BASELINE МОДЕЛЕЙ
# =============================================================================
def train_classifier(
    model: nn.Module,
    detector: Detector,  # None для baseline
    train_loader: DataLoader,
    poison_gen: PoisonGenerator,
    epochs: int,
    device: torch.device,
    use_defense: bool,
    lr: float = 1e-3,
    name: str = "Model",
):
    print("\n" + "=" * 60)
    print(f"ФАЗА 2: Тренування {name} (захист={'ON' if use_defense else 'OFF'})")
    print("=" * 60)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    if detector is not None:
        detector.eval()

    for epoch in range(epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0
        avg_trust_poisoned = 0.0
        avg_trust_clean = 0.0
        num_poisoned = 0
        num_clean = 0
        start = time.time()

        model.train()
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            # Отруюємо частину датасету (це симуляція компрометованих даних)
            x_p, y_p, is_poisoned, _ = poison_gen.poison_batch(x, y)
            x_p = x_p.to(device)
            y_p = y_p.to(device)
            is_poisoned = is_poisoned.to(device)

            optimizer.zero_grad()
            logits = model(x_p)

            if use_defense and detector is not None:
                # Імунний захист: ваги довіри від детектора
                trust = detector.trust_weights(x_p, soft=True)
                loss = weighted_cross_entropy(logits, y_p, trust)

                if is_poisoned.any():
                    avg_trust_poisoned += trust[is_poisoned].sum().item()
                    num_poisoned += is_poisoned.sum().item()
                if (~is_poisoned).any():
                    avg_trust_clean += trust[~is_poisoned].sum().item()
                    num_clean += (~is_poisoned).sum().item()
            else:
                # Baseline — без захисту
                loss = nn.functional.cross_entropy(logits, y_p)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            correct += (logits.argmax(dim=-1) == y_p).sum().item()
            total += x.size(0)

        elapsed = time.time() - start
        info = (
            f"{name} epoch {epoch + 1}/{epochs} | "
            f"loss={epoch_loss / len(train_loader):.4f} | "
            f"train_acc(poisoned)={100 * correct / total:.2f}% | "
            f"time={elapsed:.1f}s"
        )
        if use_defense and num_poisoned > 0:
            info += (
                f" | avg_trust(poisoned)={avg_trust_poisoned / num_poisoned:.3f}"
                f" | avg_trust(clean)={avg_trust_clean / max(num_clean, 1):.3f}"
            )
        print(info)


# =============================================================================
# ФАЗА 3: ОЦІНКА
# =============================================================================
@torch.no_grad()
def evaluate_clean(model: nn.Module, test_loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        correct += (logits.argmax(dim=-1) == y).sum().item()
        total += x.size(0)
    return 100.0 * correct / total


@torch.no_grad()
def evaluate_backdoor(
    model: nn.Module,
    test_loader: DataLoader,
    backdoor_attack: BackdoorAttack,
    device: torch.device,
) -> float:
    """
    Attack Success Rate (ASR): який % НЕ target-class зразків модель класифікує як target після додавання тригера.
    МЕНШЕ = КРАЩЕ. Якщо захист працює — модель не повинна реагувати на тригер.
    """
    model.eval()
    target = backdoor_attack.target_class
    success = 0
    total = 0

    for x, y in test_loader:
        # Пропускаємо зразки, які вже належать target класу
        mask = y != target
        if mask.sum() == 0:
            continue

        x_filtered = x[mask]
        # Додаємо тригер
        x_triggered = x_filtered.clone()
        for i in range(x_triggered.size(0)):
            x_triggered[i], _ = backdoor_attack(x_triggered[i], int(y[mask][i].item()))

        x_triggered = x_triggered.to(device)
        preds = model(x_triggered).argmax(dim=-1)
        success += (preds == target).sum().item()
        total += x_triggered.size(0)

    return 100.0 * success / max(total, 1)


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="cifar10",
        choices=list(DATASET_CONFIGS.keys()),
        help="HF dataset: cifar10, cifar100, mnist, tiny_imagenet, fashion_mnist",
    )
    parser.add_argument("--epochs_detector", type=int, default=5)
    parser.add_argument("--epochs_classifier", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--poison_ratio", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--cache_dir", type=str, default=None, help="HF cache dir (default: ~/.cache/huggingface)")
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--resize_to", type=int, default=None, help="Resize images to NxN (optional)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Args: {vars(args)}")

    # Data — тепер через HuggingFace datasets
    train_loader, test_loader, in_channels, num_classes = get_dataloaders(
        args.dataset,
        batch_size=args.batch_size,
        cache_dir=args.cache_dir,
        num_workers=args.num_workers,
        resize_to=args.resize_to,
    )

    # Poison generator з усіма типами атак
    target_class = 0  # бекдор завжди веде до класу 0
    backdoor = BackdoorAttack(trigger_size=4, trigger_value=1.0, target_class=target_class)
    poison_gen = PoisonGenerator(
        attacks=[
            LabelFlipAttack(num_classes=num_classes),
            backdoor,
            CleanLabelAttack(epsilon=0.05),
            FeatureCorruptionAttack(corruption_ratio=0.2, num_classes=num_classes),
        ],
        poison_ratio=args.poison_ratio,
        num_classes=num_classes,
    )

    # --- ФАЗА 1: Detector ---
    detector = Detector(
        in_channels=in_channels, embed_dim=128, num_attack_types=NUM_ATTACK_TYPES
    ).to(device)
    train_detector(detector, train_loader, poison_gen, args.epochs_detector, device, args.lr)
    torch.save(detector.state_dict(), os.path.join(args.save_dir, "detector.pt"))

    # --- ФАЗА 2: Baseline (без захисту) ---
    baseline = ProtectedModel(num_classes=num_classes, in_channels=in_channels).to(device)
    train_classifier(
        baseline, None, train_loader, poison_gen,
        args.epochs_classifier, device, use_defense=False, lr=args.lr, name="BASELINE"
    )
    torch.save(baseline.state_dict(), os.path.join(args.save_dir, "baseline.pt"))

    # --- ФАЗА 2: Protected (з імунним захистом) ---
    protected = ProtectedModel(num_classes=num_classes, in_channels=in_channels).to(device)
    train_classifier(
        protected, detector, train_loader, poison_gen,
        args.epochs_classifier, device, use_defense=True, lr=args.lr, name="PROTECTED"
    )
    torch.save(protected.state_dict(), os.path.join(args.save_dir, "protected.pt"))

    # --- ФАЗА 3: ОЦІНКА ---
    print("\n" + "=" * 60)
    print("ФАЗА 3: Фінальна оцінка")
    print("=" * 60)

    baseline_clean = evaluate_clean(baseline, test_loader, device)
    protected_clean = evaluate_clean(protected, test_loader, device)
    baseline_asr = evaluate_backdoor(baseline, test_loader, backdoor, device)
    protected_asr = evaluate_backdoor(protected, test_loader, backdoor, device)

    print(f"\n{'Метрика':<35} {'Baseline':<15} {'Protected':<15}")
    print("-" * 65)
    print(f"{'Clean accuracy ↑':<35} {baseline_clean:<15.2f} {protected_clean:<15.2f}")
    print(f"{'Backdoor ASR ↓ (атака успішна %)':<35} {baseline_asr:<15.2f} {protected_asr:<15.2f}")

    print("\nІнтерпретація:")
    print(f"  • Clean accuracy: вища = краще (нормальна продуктивність)")
    print(f"  • Backdoor ASR: нижча = краще (атака менш ефективна)")
    if protected_asr < baseline_asr:
        delta = baseline_asr - protected_asr
        print(f"  ✓ Захист знизив успішність атаки на {delta:.2f} процентних пунктів!")
    else:
        print(f"  ✗ Захист не зменшив атаку — треба тюнити Detector.")


if __name__ == "__main__":
    main()