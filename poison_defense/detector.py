"""
Detector — "імунна клітина" системи.
Окрема модель, що розрізняє чисті vs отруєні зразки.

Архітектура:
    - Encoder (CNN) → embedding простір
    - Poison head: бінарна класифікація clean/poisoned
    - Attack-type head: класифікація типу атаки (для аналізу)
    - Memory bank: зберігає embedding'и відомих атак ("лімфоцити пам'яті")

Тренування:
    - Класифікаційний loss (cross-entropy)
    - Contrastive loss (SupCon) — чисті embedding'и кластеризуються разом
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class DetectorEncoder(nn.Module):
    """Малий CNN-енкодер. Вистачає для CIFAR-розмірів."""

    def __init__(self, in_channels: int = 3, embed_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32 → 16
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16 → 8
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 8 → 4
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.projection = nn.Linear(128, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x)
        return self.projection(h)


class Detector(nn.Module):
    """
    Детектор з двома головами + memory bank.

    Forward повертає:
        embeddings: (B, embed_dim) — нормалізовані embedding'и
        poison_logits: (B, 2) — clean vs poisoned
        attack_logits: (B, num_attack_types) — який тип атаки
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 128,
        num_attack_types: int = 5,  # clean + 4 attack types
    ):
        super().__init__()
        self.encoder = DetectorEncoder(in_channels, embed_dim)
        self.poison_head = nn.Linear(embed_dim, 2)
        self.attack_head = nn.Linear(embed_dim, num_attack_types)
        self.embed_dim = embed_dim

        # Memory bank — зберігаємо embedding'и відомих атак
        self.register_buffer("memory_embeds", torch.zeros(0, embed_dim))
        self.register_buffer("memory_attack_ids", torch.zeros(0, dtype=torch.long))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embed = self.encoder(x)
        embed_normalized = F.normalize(embed, dim=-1)
        poison_logits = self.poison_head(embed)
        attack_logits = self.attack_head(embed)
        return embed_normalized, poison_logits, attack_logits

    @torch.no_grad()
    def poison_probability(self, x: torch.Tensor) -> torch.Tensor:
        """Повертає P(poisoned) для кожного зразка."""
        self.eval()
        _, poison_logits, _ = self.forward(x)
        probs = F.softmax(poison_logits, dim=-1)
        return probs[:, 1]

    @torch.no_grad()
    def trust_weights(self, x: torch.Tensor, soft: bool = True) -> torch.Tensor:
        """
        Ваги довіри для loss-зважування.
        soft=True: ваги в [0, 1] = 1 - P(poisoned)
        soft=False: hard rejection — 1.0 якщо clean, 0.0 якщо poisoned
        """
        p = self.poison_probability(x)
        if soft:
            return 1.0 - p
        return (p < 0.5).float()

    @torch.no_grad()
    def update_memory(
        self,
        embeddings: torch.Tensor,
        attack_ids: torch.Tensor,
        max_size: int = 2000,
    ):
        """Додає нові сигнатури в memory bank ('лімфоцити пам'яті')."""
        self.memory_embeds = torch.cat([self.memory_embeds, embeddings.detach()])
        self.memory_attack_ids = torch.cat([self.memory_attack_ids, attack_ids])

        if len(self.memory_embeds) > max_size:
            # Залишаємо найсвіжіші
            self.memory_embeds = self.memory_embeds[-max_size:]
            self.memory_attack_ids = self.memory_attack_ids[-max_size:]

    @torch.no_grad()
    def memory_lookup(self, x: torch.Tensor, k: int = 5) -> Optional[torch.Tensor]:
        """
        Шукає k найближчих зразків у memory bank.
        Повертає вектор оцінок схожості з відомими атаками.
        """
        if len(self.memory_embeds) == 0:
            return None
        embed, _, _ = self.forward(x)
        # Косинусна схожість (embed вже нормалізований)
        sim = torch.matmul(embed, self.memory_embeds.T)  # (B, M)
        top_k_sim, top_k_idx = sim.topk(min(k, sim.size(1)), dim=1)
        # Чи нагадує зразок щось з memory (середня top-k схожість)
        return top_k_sim.mean(dim=1)


def supervised_contrastive_loss(
    embeddings: torch.Tensor, labels: torch.Tensor, temperature: float = 0.1
) -> torch.Tensor:
    """
    SupCon loss (Khosla et al., 2020).
    Тягне зразки однієї мітки ближче, штовхає інші геть.
    Для нас: чисті embedding'и → один кластер, отруєні → окремо.
    """
    device = embeddings.device
    batch_size = embeddings.size(0)

    # similarity matrix
    sim = torch.matmul(embeddings, embeddings.T) / temperature

    # числова стабільність
    sim_max, _ = sim.max(dim=1, keepdim=True)
    sim = sim - sim_max.detach()

    # маска "позитивних пар" (та сама мітка, виключаючи self)
    labels = labels.contiguous().view(-1, 1)
    pos_mask = torch.eq(labels, labels.T).float().to(device)
    self_mask = torch.eye(batch_size, device=device)
    pos_mask = pos_mask - self_mask  # виключити діагональ

    # log_prob
    exp_sim = torch.exp(sim) * (1 - self_mask)  # виключити self
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

    # середнє по позитивах
    pos_count = pos_mask.sum(dim=1)
    pos_count = torch.clamp(pos_count, min=1.0)
    mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / pos_count

    return -mean_log_prob_pos.mean()


def detector_loss(
    embeddings: torch.Tensor,
    poison_logits: torch.Tensor,
    attack_logits: torch.Tensor,
    is_poisoned: torch.Tensor,
    attack_ids: torch.Tensor,
    use_contrastive: bool = True,
    contrast_weight: float = 0.5,
    attack_weight: float = 0.3,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Комбінований loss для тренування Detector'а.

    Args:
        embeddings: (B, D) нормалізовані
        poison_logits: (B, 2)
        attack_logits: (B, num_attack_types)
        is_poisoned: (B,) bool
        attack_ids: (B,) long — 0 для clean, 1..N для типів атак
    """
    is_poisoned_long = is_poisoned.long()

    # 1. Бінарна класифікація clean/poisoned
    cls_loss = F.cross_entropy(poison_logits, is_poisoned_long)

    # 2. Multi-class класифікація типу атаки
    atk_loss = F.cross_entropy(attack_logits, attack_ids)

    total = cls_loss + attack_weight * atk_loss
    metrics = {"cls": cls_loss.item(), "attack_cls": atk_loss.item()}

    # 3. SupCon — для розділення в embedding-просторі
    if use_contrastive and embeddings.size(0) > 2:
        contrast = supervised_contrastive_loss(embeddings, is_poisoned_long)
        total = total + contrast_weight * contrast
        metrics["contrast"] = contrast.item()

    metrics["total"] = total.item()
    return total, metrics


if __name__ == "__main__":
    # Швидкий тест
    detector = Detector(in_channels=3, embed_dim=128, num_attack_types=5)
    x = torch.rand(8, 3, 32, 32)
    embed, poison_logits, attack_logits = detector(x)

    print(f"Embedding shape: {embed.shape}")
    print(f"Poison logits shape: {poison_logits.shape}")
    print(f"Attack logits shape: {attack_logits.shape}")
    print(f"Trust weights: {detector.trust_weights(x).tolist()}")

    # Тест loss'а
    is_poisoned = torch.randint(0, 2, (8,)).bool()
    attack_ids = torch.randint(0, 5, (8,))
    loss, metrics = detector_loss(embed, poison_logits, attack_logits, is_poisoned, attack_ids)
    print(f"Loss: {loss.item():.4f}, metrics: {metrics}")