"""
🛡️ AI Poison Defense — Immunity-as-a-Service API (з API key захистом)

Gradio Space, що працює одночасно як:
    1. Demo UI (для людей)
    2. REST API через gradio_client (для інших AI / моделей)

5 endpoint'ів:
    /scan          — перевірити один зразок (poisoned / clean)
    /batch_scan    — перевірити батч зразків
    /trust_weights — отримати trust_weights для weighted training
    /generate_vaccine — згенерувати "вакцину" (labeled poisoned samples)
    /classify_protected — класифікувати через захищену модель

🔐 ВСІ endpoints захищені API ключем.
Ключ задається через HF Space Secret `VACCINATE_API_KEY`.

Деплой:
    1. У Space → Settings → Variables and secrets → New Secret
       - Name: VACCINATE_API_KEY
       - Value: твій випадковий ключ (наприклад: openssl rand -hex 32)
    2. Factory rebuild Space
    3. Тепер усі виклики API мають передавати цей ключ першим параметром

Використання як API:
    from gradio_client import Client
    client = Client("Zonda001/poison-defense")
    result = client.predict("YOUR_API_KEY", "path/to/image.jpg", api_name="/scan")
"""

import os
import json
import hmac
from typing import List, Tuple, Dict, Any

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import gradio as gr
from huggingface_hub import hf_hub_download
from torchvision import transforms

from detector import Detector
from models import ProtectedModel
from poison_generator import (
    LabelFlipAttack, BackdoorAttack,
    CleanLabelAttack, FeatureCorruptionAttack,
)


# =============================================================================
# КОНФІГУРАЦІЯ
# =============================================================================
MODEL_REPO = os.environ.get("HF_MODEL_REPO", "Zonda001/poison-defense-cifar10")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
ATTACK_TYPES = ["clean", "label_flip", "backdoor", "clean_label", "feature_corruption"]


# =============================================================================
# 🔐 API KEY AUTH
# =============================================================================
VACCINATE_API_KEY = os.environ.get("VACCINATE_API_KEY", "").strip()

if not VACCINATE_API_KEY:
    raise RuntimeError(
        "VACCINATE_API_KEY secret is not set. "
        "Add it in Space Settings → Variables and secrets → New Secret. "
        "Generate one with: openssl rand -hex 32"
    )

print(f"✅ API key auth enabled (key length: {len(VACCINATE_API_KEY)})")


def require_auth(provided_key: str) -> None:
    """
    Валідує API key з constant-time порівнянням (захист від timing атак).
    Кидає gr.Error якщо ключ невірний або відсутній.
    """
    if not provided_key:
        raise gr.Error("Missing API key. Pass it as the first argument.")

    provided = (provided_key or "").strip()
    if not hmac.compare_digest(provided, VACCINATE_API_KEY):
        raise gr.Error("Invalid API key.")


# =============================================================================
# ЗАВАНТАЖЕННЯ МОДЕЛЕЙ
# =============================================================================
def load_models():
    print(f"📥 Loading models from {MODEL_REPO}...")

    config_path = hf_hub_download(repo_id=MODEL_REPO, filename="config.json")
    with open(config_path) as f:
        config = json.load(f)

    in_ch = config["in_channels"]
    n_cls = config["num_classes"]

    p = hf_hub_download(repo_id=MODEL_REPO, filename="detector.pt")
    detector = Detector(in_channels=in_ch, embed_dim=128, num_attack_types=5)
    detector.load_state_dict(torch.load(p, map_location=DEVICE))
    detector.to(DEVICE).eval()

    p = hf_hub_download(repo_id=MODEL_REPO, filename="protected.pt")
    protected = ProtectedModel(num_classes=n_cls, in_channels=in_ch)
    protected.load_state_dict(torch.load(p, map_location=DEVICE))
    protected.to(DEVICE).eval()

    p = hf_hub_download(repo_id=MODEL_REPO, filename="baseline.pt")
    baseline = ProtectedModel(num_classes=n_cls, in_channels=in_ch)
    baseline.load_state_dict(torch.load(p, map_location=DEVICE))
    baseline.to(DEVICE).eval()

    print("✅ Models loaded")
    return detector, protected, baseline, config


DETECTOR, PROTECTED, BASELINE, CONFIG = load_models()
IMG_SIZE = CONFIG["image_size"]
IN_CHANNELS = CONFIG["in_channels"]
NUM_CLASSES = CONFIG["num_classes"]


# =============================================================================
# ДОПОМІЖНІ ФУНКЦІЇ
# =============================================================================
def pil_to_tensor(pil_img: Image.Image) -> torch.Tensor:
    pil_img = pil_img.convert("RGB" if IN_CHANNELS == 3 else "L")
    pil_img = pil_img.resize((IMG_SIZE, IMG_SIZE))
    return transforms.ToTensor()(pil_img).unsqueeze(0).to(DEVICE)


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    arr = t.cpu().permute(1, 2, 0).numpy()
    arr = (arr * 255).clip(0, 255).astype(np.uint8)
    if arr.shape[-1] == 1:
        arr = arr.squeeze(-1)
    return Image.fromarray(arr)


def pils_to_batch(pil_list: List[Image.Image]) -> torch.Tensor:
    tensors = [pil_to_tensor(img).squeeze(0) for img in pil_list]
    return torch.stack(tensors).to(DEVICE)


# =============================================================================
# 🔌 ENDPOINT 1: /scan
# =============================================================================
def scan_endpoint(api_key: str, image: Image.Image) -> Dict[str, Any]:
    require_auth(api_key)
    if image is None:
        return {"error": "No image provided"}

    x = pil_to_tensor(image)
    with torch.no_grad():
        _, poison_logits, attack_logits = DETECTOR(x)
        poison_prob = F.softmax(poison_logits, dim=-1)[0, 1].item()
        attack_probs = F.softmax(attack_logits, dim=-1)[0]
        top_attack_idx = attack_probs.argmax().item()

    return {
        "safe": poison_prob < 0.5,
        "poison_probability": round(poison_prob, 4),
        "trust_weight": round(1.0 - poison_prob, 4),
        "predicted_attack_type": ATTACK_TYPES[top_attack_idx],
        "attack_distribution": {
            name: round(attack_probs[i].item(), 4)
            for i, name in enumerate(ATTACK_TYPES)
        },
    }


# =============================================================================
# 🔌 ENDPOINT 2: /batch_scan
# =============================================================================
def batch_scan_endpoint(api_key: str, files) -> Dict[str, Any]:
    require_auth(api_key)
    if not files:
        return {"error": "No files provided"}

    images = []
    for f in files:
        try:
            path = f.name if hasattr(f, "name") else f
            images.append(Image.open(path))
        except Exception as e:
            return {"error": f"Failed to load image: {e}"}

    batch = pils_to_batch(images)
    with torch.no_grad():
        _, poison_logits, attack_logits = DETECTOR(batch)
        poison_probs = F.softmax(poison_logits, dim=-1)[:, 1]
        attack_preds = attack_logits.argmax(dim=-1)

    results = []
    clean_indices, poisoned_indices = [], []
    for i in range(len(images)):
        p = poison_probs[i].item()
        is_clean = p < 0.5
        results.append({
            "index": i,
            "safe": is_clean,
            "poison_probability": round(p, 4),
            "trust_weight": round(1.0 - p, 4),
            "predicted_attack_type": ATTACK_TYPES[attack_preds[i].item()],
        })
        (clean_indices if is_clean else poisoned_indices).append(i)

    return {
        "total": len(images),
        "clean_count": len(clean_indices),
        "poisoned_count": len(poisoned_indices),
        "clean_indices": clean_indices,
        "poisoned_indices": poisoned_indices,
        "results": results,
    }


# =============================================================================
# 🔌 ENDPOINT 3: /trust_weights
# =============================================================================
def trust_weights_endpoint(api_key: str, files) -> Dict[str, Any]:
    require_auth(api_key)
    if not files:
        return {"error": "No files provided"}

    images = [Image.open(f.name if hasattr(f, "name") else f) for f in files]
    batch = pils_to_batch(images)

    with torch.no_grad():
        weights = DETECTOR.trust_weights(batch, soft=True)

    return {
        "weights": [round(w.item(), 4) for w in weights],
        "soft_mode": True,
        "usage_example": (
            "per_sample_loss = F.cross_entropy(logits, labels, reduction='none')\n"
            "weighted_loss = (per_sample_loss * trust_weights).sum() / trust_weights.sum()"
        ),
    }


# =============================================================================
# 🔌 ENDPOINT 4: /generate_vaccine
# =============================================================================
def generate_vaccine_endpoint(
    api_key: str, image: Image.Image, attack_type: str
) -> Tuple[Image.Image, Dict[str, Any]]:
    require_auth(api_key)
    if image is None:
        return None, {"error": "No image provided"}

    x = pil_to_tensor(image).squeeze(0)
    original_label = 0

    if attack_type == "label_flip":
        attack = LabelFlipAttack(num_classes=NUM_CLASSES)
    elif attack_type == "backdoor":
        attack = BackdoorAttack(trigger_size=4, trigger_value=1.0, target_class=0)
    elif attack_type == "clean_label":
        attack = CleanLabelAttack(epsilon=0.08)
    elif attack_type == "feature_corruption":
        attack = FeatureCorruptionAttack(corruption_ratio=0.2, num_classes=NUM_CLASSES)
    else:
        return None, {"error": f"Unknown attack_type: {attack_type}"}

    x_poisoned, y_poisoned = attack(x.cpu(), original_label)
    poisoned_img = tensor_to_pil(x_poisoned)
    perturbation = (x_poisoned - x.cpu()).abs().max().item()

    return poisoned_img, {
        "attack_type": attack_type,
        "is_poisoned": True,
        "original_label_corrupted": y_poisoned != original_label,
        "new_label": int(y_poisoned),
        "perturbation_max": round(perturbation, 4),
    }


# =============================================================================
# 🔌 ENDPOINT 5: /classify_protected
# =============================================================================
def classify_protected_endpoint(api_key: str, image: Image.Image) -> Dict[str, Any]:
    require_auth(api_key)
    if image is None:
        return {"error": "No image provided"}

    x = pil_to_tensor(image)
    with torch.no_grad():
        protected_logits = PROTECTED(x)
        baseline_logits = BASELINE(x)
        p_probs = F.softmax(protected_logits, dim=-1)[0]
        b_probs = F.softmax(baseline_logits, dim=-1)[0]
        p_top3 = p_probs.topk(3)
        b_top3 = b_probs.topk(3)

        # Inline scan (без require_auth — ми вже авторизовані)
        _, scan_p_logits, scan_a_logits = DETECTOR(x)
        scan_poison_prob = F.softmax(scan_p_logits, dim=-1)[0, 1].item()
        scan_attack_idx = F.softmax(scan_a_logits, dim=-1)[0].argmax().item()

    return {
        "protected_prediction": {
            "class": CIFAR10_CLASSES[p_top3.indices[0].item()],
            "confidence": round(p_top3.values[0].item(), 4),
            "top3": [
                {"class": CIFAR10_CLASSES[p_top3.indices[i].item()],
                 "confidence": round(p_top3.values[i].item(), 4)}
                for i in range(3)
            ],
        },
        "baseline_prediction": {
            "class": CIFAR10_CLASSES[b_top3.indices[0].item()],
            "confidence": round(b_top3.values[0].item(), 4),
            "top3": [
                {"class": CIFAR10_CLASSES[b_top3.indices[i].item()],
                 "confidence": round(b_top3.values[i].item(), 4)}
                for i in range(3)
            ],
        },
        "agreement": p_top3.indices[0].item() == b_top3.indices[0].item(),
        "scan": {
            "safe": scan_poison_prob < 0.5,
            "poison_probability": round(scan_poison_prob, 4),
            "predicted_attack_type": ATTACK_TYPES[scan_attack_idx],
        },
    }


# =============================================================================
# GRADIO UI
# =============================================================================
INFO_MD = """
# 🛡️ AI Poison Defense — Immunity-as-a-Service

*"Аналог імунної системи для штучного інтелекту"*

5 захищених API endpoints для тренування нейромереж з імунітетом до отруєння даних.

**Backdoor Attack Success Rate**: 98.09% → **1.26%** (CIFAR-10, ResNet-14)

🔐 **Цей API захищений ключем.** Введи API ключ у поле в кожному табі.
Для запитів через `gradio_client` ключ передається як перший параметр:

```python
from gradio_client import Client, handle_file

client = Client("Zonda001/poison-defense")
result = client.predict(
    "YOUR_API_KEY",
    handle_file("image.jpg"),
    api_name="/scan"
)
```
"""


with gr.Blocks(title="🛡️ AI Poison Defense API") as demo:
    gr.Markdown(INFO_MD)

    with gr.Tabs():
        # =========================================================
        # TAB: Scan
        # =========================================================
        with gr.Tab("🔍 Scan"):
            gr.Markdown("Перевір один зразок на отруєння. API endpoint: `/scan`")
            with gr.Row():
                with gr.Column():
                    scan_api_key = gr.Textbox(
                        label="API key",
                        type="password",
                        placeholder="Введи свій API ключ"
                    )
                    scan_img = gr.Image(type="pil", label="Зразок")
                    scan_btn = gr.Button("🔍 Scan", variant="primary")
                with gr.Column():
                    scan_out = gr.JSON(label="Response")

            scan_btn.click(
                fn=scan_endpoint,
                inputs=[scan_api_key, scan_img],
                outputs=scan_out,
                api_name="scan",
            )

        # =========================================================
        # TAB: Batch Scan
        # =========================================================
        with gr.Tab("📦 Batch Scan"):
            gr.Markdown(
                "Сканувати багато зразків одразу — для фільтрації датасету. "
                "API endpoint: `/batch_scan`"
            )
            batch_api_key = gr.Textbox(label="API key", type="password")
            batch_files = gr.File(
                file_count="multiple", label="Картинки", type="filepath"
            )
            batch_btn = gr.Button("📦 Scan", variant="primary")
            batch_out = gr.JSON(label="Результат")

            batch_btn.click(
                fn=batch_scan_endpoint,
                inputs=[batch_api_key, batch_files],
                outputs=batch_out,
                api_name="batch_scan",
            )

        # =========================================================
        # TAB: Trust Weights
        # =========================================================
        with gr.Tab("⚖️ Trust Weights"):
            gr.Markdown(
                "Отримай trust_weights для weighted training власної моделі. "
                "API endpoint: `/trust_weights`"
            )
            tw_api_key = gr.Textbox(label="API key", type="password")
            tw_files = gr.File(
                file_count="multiple", label="Зразки", type="filepath"
            )
            tw_btn = gr.Button("⚖️ Compute weights", variant="primary")
            tw_out = gr.JSON(label="Weights + usage example")

            tw_btn.click(
                fn=trust_weights_endpoint,
                inputs=[tw_api_key, tw_files],
                outputs=tw_out,
                api_name="trust_weights",
            )

        # =========================================================
        # TAB: Generate Vaccine
        # =========================================================
        with gr.Tab("💉 Generate Vaccine"):
            gr.Markdown(
                "Згенеруй отруєний зразок з міткою. "
                "API endpoint: `/generate_vaccine`"
            )
            with gr.Row():
                with gr.Column():
                    vacc_api_key = gr.Textbox(label="API key", type="password")
                    vacc_img = gr.Image(type="pil", label="Чистий зразок")
                    vacc_type = gr.Dropdown(
                        choices=["label_flip", "backdoor",
                                 "clean_label", "feature_corruption"],
                        value="backdoor",
                        label="Тип атаки",
                    )
                    vacc_btn = gr.Button("💉 Generate", variant="primary")
                with gr.Column():
                    vacc_out_img = gr.Image(type="pil", label="Отруєний зразок")
                    vacc_out_json = gr.JSON(label="Metadata")

            vacc_btn.click(
                fn=generate_vaccine_endpoint,
                inputs=[vacc_api_key, vacc_img, vacc_type],
                outputs=[vacc_out_img, vacc_out_json],
                api_name="generate_vaccine",
            )

        # =========================================================
        # TAB: Classify Protected
        # =========================================================
        with gr.Tab("🛡️ Classify Protected"):
            gr.Markdown(
                "Захищена класифікація + порівняння з Baseline. "
                "API endpoint: `/classify_protected`"
            )
            cls_api_key = gr.Textbox(label="API key", type="password")
            cls_img = gr.Image(type="pil", label="Зразок")
            cls_btn = gr.Button("🛡️ Classify", variant="primary")
            cls_out = gr.JSON(label="Результат")

            cls_btn.click(
                fn=classify_protected_endpoint,
                inputs=[cls_api_key, cls_img],
                outputs=cls_out,
                api_name="classify_protected",
            )

    gr.Markdown(f"""
    ---
    *Model: [{MODEL_REPO}](https://huggingface.co/{MODEL_REPO})* ·
    *5 API endpoints (all auth-protected)* ·
    *Built for OWASP LLM Top 10 (LLM04: Data and Model Poisoning)*
    """)


if __name__ == "__main__":
    demo.queue()
    demo.launch(show_error=True)
