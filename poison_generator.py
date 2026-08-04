"""
Poison Generator — "лабораторія патогенів".
Створює різноманітні отруєні зразки для тренування Detector'а.

Підтримувані атаки:
    - LabelFlipAttack: зміна мітки класу
    - BackdoorAttack: додавання тригер-патчу + зміна мітки
    - CleanLabelAttack: непомітна пертурбація зі збереженням мітки
    - FeatureCorruptionAttack: шум/артефакти
"""

import torch
import numpy as np
from typing import Tuple, List, Optional


class BaseAttack:
    """Базовий клас атаки. Кожна атака повертає (x_poisoned, y_poisoned)."""

    name: str = "base"

    def __call__(self, x: torch.Tensor, y: int) -> Tuple[torch.Tensor, int]:
        raise NotImplementedError


class LabelFlipAttack(BaseAttack):
    """Перевертає мітку класу. Найпростіший тип отруєння."""

    name = "label_flip"

    def __init__(self, num_classes: int = 10, target: Optional[int] = None):
        self.num_classes = num_classes
        self.target = target  # якщо None — випадковий інший клас

    def __call__(self, x: torch.Tensor, y: int) -> Tuple[torch.Tensor, int]:
        if self.target is not None:
            new_y = self.target
        else:
            # Випадковий клас, але не оригінальний
            choices = [c for c in range(self.num_classes) if c != y]
            new_y = int(np.random.choice(choices))
        return x.clone(), new_y


class BackdoorAttack(BaseAttack):
    """
    Додає тригер-патч (квадратик) у кут зображення + змінює мітку на target_class.
    Класична backdoor / trojan-атака: модель починає асоціювати патерн з класом.
    """

    name = "backdoor"

    def __init__(
        self,
        trigger_size: int = 4,
        trigger_value: float = 1.0,
        target_class: int = 0,
        position: str = "bottom_right",
    ):
        self.trigger_size = trigger_size
        self.trigger_value = trigger_value
        self.target_class = target_class
        self.position = position

    def __call__(self, x: torch.Tensor, y: int) -> Tuple[torch.Tensor, int]:
        x_p = x.clone()
        s = self.trigger_size

        if self.position == "bottom_right":
            x_p[..., -s:, -s:] = self.trigger_value
        elif self.position == "top_left":
            x_p[..., :s, :s] = self.trigger_value
        elif self.position == "center":
            h, w = x_p.shape[-2:]
            ch, cw = h // 2, w // 2
            x_p[..., ch - s // 2 : ch + s // 2, cw - s // 2 : cw + s // 2] = self.trigger_value

        return x_p, self.target_class


class CleanLabelAttack(BaseAttack):
    """
    Clean-label attack: непомітна пертурбація, мітка не змінюється.
    Використовує adversarial noise (Gaussian або PGD-style).
    Найхитріша атака — її важко знайти просто переглянувши датасет.
    """

    name = "clean_label"

    def __init__(self, epsilon: float = 0.03, mode: str = "gaussian"):
        self.epsilon = epsilon
        self.mode = mode  # "gaussian" або "fgsm"

    def __call__(self, x: torch.Tensor, y: int) -> Tuple[torch.Tensor, int]:
        if self.mode == "gaussian":
            noise = torch.randn_like(x) * self.epsilon
        else:
            # Простий sign-noise (наближення FGSM без градієнтів моделі)
            noise = torch.sign(torch.randn_like(x)) * self.epsilon

        x_p = torch.clamp(x + noise, 0.0, 1.0)
        return x_p, y  # мітка залишається!


class FeatureCorruptionAttack(BaseAttack):
    """
    Корумпує частину пікселів — імітує зіпсовані дані в датасеті.
    Може змінювати або не змінювати мітку.
    """

    name = "feature_corruption"

    def __init__(self, corruption_ratio: float = 0.2, flip_label: bool = False, num_classes: int = 10):
        self.corruption_ratio = corruption_ratio
        self.flip_label = flip_label
        self.num_classes = num_classes

    def __call__(self, x: torch.Tensor, y: int) -> Tuple[torch.Tensor, int]:
        x_p = x.clone()
        mask = torch.rand_like(x_p) < self.corruption_ratio
        random_pixels = torch.rand_like(x_p)
        x_p[mask] = random_pixels[mask]

        if self.flip_label:
            new_y = int(np.random.choice([c for c in range(self.num_classes) if c != y]))
            return x_p, new_y
        return x_p, y


class PoisonGenerator:
    """
    Оркеструє різні атаки. Отруює певний відсоток батчу.

    Повертає:
        - x_batch: отруєний батч
        - y_batch: мітки (можливо змінені)
        - is_poisoned: bool-маска, які зразки отруєні
        - attack_types: список рядків з типом атаки (для аналізу)
    """

    def __init__(
        self,
        attacks: Optional[List[BaseAttack]] = None,
        poison_ratio: float = 0.3,
        num_classes: int = 10,
    ):
        if attacks is None:
            attacks = [
                LabelFlipAttack(num_classes=num_classes),
                BackdoorAttack(trigger_size=4, target_class=0),
                CleanLabelAttack(epsilon=0.05),
                FeatureCorruptionAttack(corruption_ratio=0.2),
            ]
        self.attacks = attacks
        self.poison_ratio = poison_ratio
        self.num_classes = num_classes
        self.attack_names = [a.name for a in attacks]

    def poison_batch(
        self,
        x_batch: torch.Tensor,
        y_batch: torch.Tensor,
        poison_ratio: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
        batch_size = x_batch.size(0)
        new_x = x_batch.clone()
        new_y = y_batch.clone()
        is_poisoned = torch.zeros(batch_size, dtype=torch.bool)
        attack_types = ["clean"] * batch_size
        ratio = poison_ratio if poison_ratio is not None else self.poison_ratio

        for i in range(batch_size):
            if np.random.rand() < ratio:
                attack = np.random.choice(self.attacks)
                x_p, y_p = attack(x_batch[i], int(y_batch[i].item()))
                new_x[i] = x_p
                new_y[i] = y_p
                is_poisoned[i] = True
                attack_types[i] = attack.name

        return new_x, new_y, is_poisoned, attack_types

    def poison_all(
        self, x_batch: torch.Tensor, y_batch: torch.Tensor, attack: BaseAttack
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Отруює весь батч однією конкретною атакою — для тестування."""
        new_x = x_batch.clone()
        new_y = y_batch.clone()
        for i in range(x_batch.size(0)):
            x_p, y_p = attack(x_batch[i], int(y_batch[i].item()))
            new_x[i] = x_p
            new_y[i] = y_p
        return new_x, new_y


if __name__ == "__main__":
    # Швидкий тест
    x = torch.rand(8, 3, 32, 32)
    y = torch.randint(0, 10, (8,))

    gen = PoisonGenerator(poison_ratio=0.5)
    new_x, new_y, mask, types = gen.poison_batch(x, y)

    print(f"Original shape: {x.shape}")
    print(f"Poisoned mask: {mask.tolist()}")
    print(f"Attack types: {types}")
    print(f"Label changes: {(new_y != y).tolist()}")
    print(f"Pixel diff (max): {(new_x - x).abs().max().item():.4f}")
    