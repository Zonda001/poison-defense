"""Оцінка збережених чекпоінтів — ті самі метрики, що друкує Фаза 3 train.py."""
import json
import torch

from hf_data import get_dataloaders
from poison_generator import BackdoorAttack
from models import ProtectedModel
from train import evaluate_clean, evaluate_backdoor


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, test_loader, in_channels, num_classes = get_dataloaders(
        "cifar10", batch_size=256, num_workers=0
    )

    backdoor = BackdoorAttack(trigger_size=4, trigger_value=1.0, target_class=0)

    results = {}
    for name in ("baseline", "protected"):
        model = ProtectedModel(num_classes=num_classes, in_channels=in_channels).to(device)
        model.load_state_dict(torch.load(f"./checkpoints/{name}.pt", map_location=device))
        results[name] = {
            "clean_accuracy": round(evaluate_clean(model, test_loader, device), 2),
            "backdoor_asr": round(evaluate_backdoor(model, test_loader, backdoor, device), 2),
        }

    print(json.dumps(results, indent=2))
    with open("metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
