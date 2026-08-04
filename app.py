"""
🛡️ AI Poison Defense — Immunity-as-a-Service API

Gradio Space, що працює одночасно як:
    1. Demo UI (для людей)
    2. REST API через gradio_client (для інших AI / моделей)

5 endpoint'ів:
    /scan          — перевірити один зразок (poisoned / clean)
    /batch_scan    — перевірити батч зразків
    /trust_weights — отримати trust_weights для weighted training
    /generate_vaccine — згенерувати "вакцину" (labeled poisoned samples)
    /classify_protected — класифікувати через захищену модель

Деплой:
    1. Створити HF Space (Gradio, CPU basic)
    2. Залити: app.py, detector.py, models.py, poison_generator.py, requirements1.txt
    3. Додати Space variable: HF_MODEL_REPO=username/poison-defense-cifar10

Використання як API:
    from gradio_client import Client
    client = Client("username/poison-defense-demo")
    result = client.predict("path/to/image.jpg", api_name="/scan")
"""

import os
import json
import base64
import io
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
MODEL_REPO = os.environ.get("HF_MODEL_REPO", "YOUR_USERNAME/poison-defense-cifar10")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
ATTACK_TYPES = ["clean", "label_flip", "backdoor", "clean_label", "feature_corruption"]


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

    # Detector
    p = hf_hub_download(repo_id=MODEL_REPO, filename="detector.pt")
    detector = Detector(in_channels=in_ch, embed_dim=128, num_attack_types=5)
    detector.load_state_dict(torch.load(p, map_location=DEVICE))
    detector.to(DEVICE).eval()

    # Protected
    p = hf_hub_download(repo_id=MODEL_REPO, filename="protected.pt")
    protected = ProtectedModel(num_classes=n_cls, in_channels=in_ch)
    protected.load_state_dict(torch.load(p, map_location=DEVICE))
    protected.to(DEVICE).eval()

    # Baseline (для порівняння у demo)
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
    """PIL → tensor (1, C, H, W) у [0, 1]."""
    pil_img = pil_img.convert("RGB" if IN_CHANNELS == 3 else "L")
    pil_img = pil_img.resize((IMG_SIZE, IMG_SIZE))
    return transforms.ToTensor()(pil_img).unsqueeze(0).to(DEVICE)


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """Tensor (C, H, W) → PIL."""
    arr = t.cpu().permute(1, 2, 0).numpy()
    arr = (arr * 255).clip(0, 255).astype(np.uint8)
    if arr.shape[-1] == 1:
        arr = arr.squeeze(-1)
    return Image.fromarray(arr)


def pils_to_batch(pil_list: List[Image.Image]) -> torch.Tensor:
    """Список PIL → tensor (B, C, H, W)."""
    tensors = [pil_to_tensor(img).squeeze(0) for img in pil_list]
    return torch.stack(tensors).to(DEVICE)


# =============================================================================
# 🔌 ENDPOINT 1: /scan — сканування одного зразка
# =============================================================================
def scan_endpoint(image: Image.Image) -> Dict[str, Any]:
    """
    Перевіряє чи зразок отруєний.

    Returns:
        {
            "safe": bool,                  # True якщо чистий
            "poison_probability": float,   # P(poisoned) у [0,1]
            "trust_weight": float,         # 1 - P(poisoned) у [0,1]
            "predicted_attack_type": str,  # один з 5 типів
            "attack_distribution": dict,   # ймовірності всіх типів
        }
    """
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
# 🔌 ENDPOINT 2: /batch_scan — сканування багатьох зразків
# =============================================================================
def batch_scan_endpoint(files) -> Dict[str, Any]:
    """
    Перевіряє багато зразків одразу. Корисно для фільтрації датасету.

    Args:
        files: список шляхів до файлів (Gradio File component)

    Returns:
        {
            "total": int,
            "clean_count": int,
            "poisoned_count": int,
            "clean_indices": [int, ...],  # індекси чистих
            "poisoned_indices": [int, ...],
            "results": [{...}, ...]  # детальний результат для кожного
        }
    """
    if not files:
        return {"error": "No files provided"}

    # Завантажуємо всі картинки
    images = []
    for f in files:
        try:
            path = f.name if hasattr(f, "name") else f
            img = Image.open(path)
            images.append(img)
        except Exception as e:
            return {"error": f"Failed to load image: {e}"}

    batch = pils_to_batch(images)

    with torch.no_grad():
        _, poison_logits, attack_logits = DETECTOR(batch)
        poison_probs = F.softmax(poison_logits, dim=-1)[:, 1]
        attack_preds = attack_logits.argmax(dim=-1)

    results = []
    clean_indices = []
    poisoned_indices = []

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
        if is_clean:
            clean_indices.append(i)
        else:
            poisoned_indices.append(i)

    return {
        "total": len(images),
        "clean_count": len(clean_indices),
        "poisoned_count": len(poisoned_indices),
        "clean_indices": clean_indices,
        "poisoned_indices": poisoned_indices,
        "results": results,
    }


# =============================================================================
# 🔌 ENDPOINT 3: /trust_weights — масив ваг для weighted training
# =============================================================================
def trust_weights_endpoint(files) -> Dict[str, Any]:
    """
    Повертає масив trust_weights для батчу зразків.
    Це для того, щоб інша AI могла використати ці ваги
    у власному weighted loss під час тренування.

    Args:
        files: список картинок

    Returns:
        {
            "weights": [float, ...],  # одна вага на зразок
            "soft_mode": True,
            "usage_example": "loss = (per_sample_loss * weights).sum() / weights.sum()"
        }
    """
    if not files:
        return {"error": "No files provided"}

    images = []
    for f in files:
        path = f.name if hasattr(f, "name") else f
        images.append(Image.open(path))

    batch = pils_to_batch(images)

    with torch.no_grad():
        weights = DETECTOR.trust_weights(batch, soft=True)

    return {
        "weights": [round(w.item(), 4) for w in weights],
        "soft_mode": True,
        "usage_example": (
            "# В твоєму training loop:\n"
            "per_sample_loss = F.cross_entropy(logits, labels, reduction='none')\n"
            "weighted_loss = (per_sample_loss * trust_weights).sum() / trust_weights.sum()\n"
            "weighted_loss.backward()"
        ),
    }


# =============================================================================
# 🔌 ENDPOINT 4: /generate_vaccine — створити отруєні зразки з мітками
# =============================================================================
def generate_vaccine_endpoint(
    image: Image.Image, attack_type: str
) -> Tuple[Image.Image, Dict[str, Any]]:
    """
    Створює "вакцину" — отруєний зразок + правильна мітка poison/clean.
    Використовується іншими командами для тренування ВЛАСНОЇ імунної системи.

    Args:
        image: чистий зразок
        attack_type: "label_flip" | "backdoor" | "clean_label" | "feature_corruption"

    Returns:
        (poisoned_image, metadata)
        metadata = {
            "attack_type": str,
            "is_poisoned": True,
            "original_label_corrupted": bool,
            "perturbation_max": float,
            "training_hint": str
        }
    """
    if image is None:
        return None, {"error": "No image provided"}

    x = pil_to_tensor(image).squeeze(0)
    original_label = 0  # умовно

    # Обираємо атаку
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

    # Генеруємо
    x_poisoned, y_poisoned = attack(x.cpu(), original_label)

    poisoned_img = tensor_to_pil(x_poisoned)
    perturbation = (x_poisoned - x.cpu()).abs().max().item()

    return poisoned_img, {
        "attack_type": attack_type,
        "is_poisoned": True,
        "original_label_corrupted": y_poisoned != original_label,
        "new_label": int(y_poisoned),
        "perturbation_max": round(perturbation, 4),
        "training_hint": (
            "Використай цей зразок з міткою is_poisoned=True "
            "у своєму detector training. Згенеруй багато таких "
            "для збалансованого датасету."
        ),
    }


# =============================================================================
# 🔌 ENDPOINT 5: /classify_protected — захищена класифікація
# =============================================================================
def classify_protected_endpoint(image: Image.Image) -> Dict[str, Any]:
    """
    Класифікує зразок через Protected Model (з імунним захистом).
    Також повертає поряд передбачення Baseline (без захисту) для порівняння.

    Returns:
        {
            "protected_prediction": {"class": str, "confidence": float, "top3": [...]},
            "baseline_prediction": {...},
            "agreement": bool,
            "scan": {...}  # результат детектора
        }
    """
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

    # Також скан від детектора
    scan = scan_endpoint(image)

    return {
        "protected_prediction": {
            "class": CIFAR10_CLASSES[p_top3.indices[0].item()],
            "confidence": round(p_top3.values[0].item(), 4),
            "top3": [
                {
                    "class": CIFAR10_CLASSES[p_top3.indices[i].item()],
                    "confidence": round(p_top3.values[i].item(), 4),
                }
                for i in range(3)
            ],
        },
        "baseline_prediction": {
            "class": CIFAR10_CLASSES[b_top3.indices[0].item()],
            "confidence": round(b_top3.values[0].item(), 4),
            "top3": [
                {
                    "class": CIFAR10_CLASSES[b_top3.indices[i].item()],
                    "confidence": round(b_top3.values[i].item(), 4),
                }
                for i in range(3)
            ],
        },
        "agreement": p_top3.indices[0].item() == b_top3.indices[0].item(),
        "scan": scan,
    }


# =============================================================================
# UI HELPERS — для людської взаємодії
# =============================================================================
def scan_ui(image):
    """UI wrapper для /scan з гарним output."""
    result = scan_endpoint(image)
    if "error" in result:
        return result["error"], None

    verdict = "✅ SAFE (clean)" if result["safe"] else "🦠 POISONED"
    md = f"""## {verdict}

**Poison probability**: {result['poison_probability']:.1%}
**Trust weight**: {result['trust_weight']:.3f}
**Predicted attack type**: `{result['predicted_attack_type']}`

### Attack distribution
"""
    for name, prob in result["attack_distribution"].items():
        bar = "█" * int(prob * 30)
        md += f"- **{name}**: {prob:.1%} {bar}\n"

    return md, result


def classify_ui(image, add_trigger):
    """UI для класифікації з опцією додати backdoor trigger."""
    if image is None:
        return None, "Завантаж картинку", {}

    x = pil_to_tensor(image)

    if add_trigger:
        backdoor = BackdoorAttack(trigger_size=4, trigger_value=1.0, target_class=0)
        x_t = x.squeeze(0).cpu()
        x_t, _ = backdoor(x_t, 0)
        x = x_t.unsqueeze(0).to(DEVICE)
        display_img = tensor_to_pil(x_t).resize((256, 256), Image.NEAREST)
    else:
        display_img = image.resize((256, 256))

    # Викликаємо endpoint напряму
    img_for_endpoint = tensor_to_pil(x.squeeze(0).cpu())
    result = classify_protected_endpoint(img_for_endpoint)

    pred_p = result["protected_prediction"]
    pred_b = result["baseline_prediction"]

    md = f"""### Результати класифікації

**🛡️ Protected Model**: `{pred_p['class']}` ({pred_p['confidence']:.1%})
**❌ Baseline Model**: `{pred_b['class']}` ({pred_b['confidence']:.1%})

**Detector**: P(poisoned) = {result['scan']['poison_probability']:.1%}
"""

    if add_trigger:
        if pred_b["class"] == "airplane" and pred_p["class"] != "airplane":
            md += "\n🛡️ **Захист спрацював!** Baseline впала на тригер, Protected — ні."
        elif pred_b["class"] == "airplane" and pred_p["class"] == "airplane":
            md += "\n⚠️ Обидві моделі вгадали target class — можливо, картинка справді airplane?"
        else:
            md += "\nℹ️ Жодна модель не впала на тригер."

    return display_img, md, result


# =============================================================================
# API DOCS — для tab з документацією
# =============================================================================
API_DOCS = f"""
# 📚 API Documentation

Цей Space експонує **5 REST endpoint'ів** через `gradio_client`.
Будь-яка інша AI може звертатись до них для:
- Сканування своїх даних на отруєння
- Отримання trust_weights для weighted training
- Генерації "вакцини" — позначених отруєних зразків

## Встановлення клієнта

```bash
pip install gradio_client
```

## Базове використання

```python
from gradio_client import Client

client = Client("{MODEL_REPO.replace('poison-defense-cifar10', 'poison-defense-demo')}")
```

---

## 1. `/scan` — Сканування одного зразка

**Призначення:** Перевірити, чи конкретний зразок отруєний.

```python
result = client.predict(
    image="path/to/image.jpg",
    api_name="/scan"
)
print(result)
# {{
#   "safe": False,
#   "poison_probability": 0.94,
#   "trust_weight": 0.06,
#   "predicted_attack_type": "backdoor",
#   "attack_distribution": {{...}}
# }}
```

---

## 2. `/batch_scan` — Фільтрація датасету

**Призначення:** Пропустити через детектор багато зразків одразу. Корисно для очищення compromised датасету перед тренуванням.

```python
files = ["img1.jpg", "img2.jpg", "img3.jpg"]
result = client.predict(
    files=files,
    api_name="/batch_scan"
)

# Використати тільки чисті:
clean_files = [files[i] for i in result["clean_indices"]]
```

---

## 3. `/trust_weights` — Ваги для Weighted Training

**Призначення:** Отримати масив `trust_weights` для батчу. Використовуй ці ваги у власному training loop:

```python
result = client.predict(files=batch_files, api_name="/trust_weights")
weights = torch.tensor(result["weights"])

# Тепер у своєму training step:
per_sample_loss = F.cross_entropy(logits, labels, reduction='none')
weighted_loss = (per_sample_loss * weights).sum() / weights.sum()
weighted_loss.backward()
```

**Це і є "імунний захист як сервіс"** — твоя модель тренується нормально,
але отруєні зразки впливають у 5 разів менше.

---

## 4. `/generate_vaccine` — Згенерувати "вакцину"

**Призначення:** Створити позначені отруєні зразки для тренування ВЛАСНОЇ імунної системи.

```python
poisoned_img, metadata = client.predict(
    image="clean_image.jpg",
    attack_type="backdoor",  # або: label_flip / clean_label / feature_corruption
    api_name="/generate_vaccine"
)

# metadata = {{
#   "attack_type": "backdoor",
#   "is_poisoned": True,
#   "new_label": 0,
#   "perturbation_max": 1.0
# }}
```

Використання: згенеруй багато таких вакцин з міткою `is_poisoned=True` →
натренуй свій власний детектор з тими ж патернами.

---

## 5. `/classify_protected` — Захищена класифікація

**Призначення:** Прогнати зразок через Protected Model (з захистом).
Повертає також передбачення Baseline (без захисту) для порівняння.

```python
result = client.predict(image="image.jpg", api_name="/classify_protected")
# {{
#   "protected_prediction": {{"class": "cat", "confidence": 0.87, ...}},
#   "baseline_prediction": {{"class": "airplane", "confidence": 0.99, ...}},
#   "agreement": False,
#   "scan": {{...}}
# }}
```

Якщо `agreement: False` — це підозра на otruєння у вхідному зразку.

---

## Повний приклад: захищене тренування твоєї моделі через API

```python
import torch
import torch.nn.functional as F
from gradio_client import Client

client = Client("YOUR_USERNAME/poison-defense-demo")

# У твоєму training loop:
for batch_files, labels in train_loader:
    # 1. Отримуємо trust weights від нашого API
    weights_result = client.predict(files=batch_files, api_name="/trust_weights")
    trust_weights = torch.tensor(weights_result["weights"])

    # 2. Твій звичайний forward
    logits = your_model(batch_data)

    # 3. Weighted loss — отруєні зразки впливають менше
    per_sample = F.cross_entropy(logits, labels, reduction='none')
    loss = (per_sample * trust_weights).sum() / trust_weights.sum()
    loss.backward()
    optimizer.step()
```

**Результат:** твоя модель отримає імунітет до тих самих типів атак,
яким ми навчили нашу систему — без потреби тренувати власний детектор.
"""


# =============================================================================
# GRADIO UI
# =============================================================================
with gr.Blocks(title="🛡️ AI Poison Defense API") as demo:
    gr.Markdown("""
    # 🛡️ AI Poison Defense — Immunity-as-a-Service

    *"Аналог імунної системи для штучного інтелекту"*

    Цей Space одночасно і **демо**, і **REST API** для захисту інших AI моделей від отруєння даних.
    Будь-яка модель може звертатись до нашого API за trust weights, скануванням, або вакциною.

    Backdoor Attack Success Rate знижено з **98.09%** до **1.26%** (CIFAR-10, ResNet-14).
    """)

    with gr.Tabs():
        # =========================================================
        # TAB 1: Live Demo (з тригером)
        # =========================================================
        with gr.Tab("🎮 Live Demo"):
            gr.Markdown("""
            Покажемо, як працює захист на практиці.

            1. Завантаж картинку (CIFAR-10 формат: 32×32, RGB)
            2. Постав галочку **"Add backdoor trigger"** — побачиш як атака впливає
            3. Порівняй: Baseline впаде на тригер → класифікує як `airplane`, Protected витримає
            """)
            with gr.Row():
                with gr.Column():
                    demo_img = gr.Image(type="pil", label="Вхідна картинка")
                    add_trigger = gr.Checkbox(label="🦠 Add backdoor trigger", value=False)
                    demo_btn = gr.Button("🔍 Аналізувати", variant="primary")
                with gr.Column():
                    demo_out_img = gr.Image(type="pil", label="Картинка (з тригером або без)")
                    demo_out_text = gr.Markdown()
                    demo_out_json = gr.JSON(label="Raw API response", visible=False)

            demo_btn.click(
                fn=classify_ui,
                inputs=[demo_img, add_trigger],
                outputs=[demo_out_img, demo_out_text, demo_out_json],
            )

        # =========================================================
        # TAB 2: Scan single image
        # =========================================================
        with gr.Tab("🔍 Scan (single)"):
            gr.Markdown("Перевір один зразок на отруєння. API endpoint: `/scan`")
            with gr.Row():
                with gr.Column():
                    scan_img = gr.Image(type="pil", label="Зразок для перевірки")
                    scan_btn = gr.Button("🔍 Scan", variant="primary")
                with gr.Column():
                    scan_out_text = gr.Markdown()
                    scan_out_json = gr.JSON(label="Raw API response")

            scan_btn.click(
                fn=scan_ui,
                inputs=scan_img,
                outputs=[scan_out_text, scan_out_json],
                api_name="scan",  # це експонує як /scan
            )

        # =========================================================
        # TAB 3: Batch scan / Filter dataset
        # =========================================================
        with gr.Tab("📦 Batch Scan"):
            gr.Markdown("""
            Завантаж декілька картинок, отримай результат сканування + індекси чистих.
            API endpoint: `/batch_scan`
            """)
            batch_files = gr.File(file_count="multiple", label="Картинки", type="filepath")
            batch_btn = gr.Button("📦 Scan batch", variant="primary")
            batch_out = gr.JSON(label="Результат")

            batch_btn.click(
                fn=batch_scan_endpoint,
                inputs=batch_files,
                outputs=batch_out,
                api_name="batch_scan",
            )

        # =========================================================
        # TAB 4: Trust weights
        # =========================================================
        with gr.Tab("⚖️ Trust Weights"):
            gr.Markdown("""
            Отримай trust_weights для батчу — використай їх у власному weighted training.
            API endpoint: `/trust_weights`
            """)
            tw_files = gr.File(file_count="multiple", label="Зразки", type="filepath")
            tw_btn = gr.Button("⚖️ Compute weights", variant="primary")
            tw_out = gr.JSON(label="Weights + usage example")

            tw_btn.click(
                fn=trust_weights_endpoint,
                inputs=tw_files,
                outputs=tw_out,
                api_name="trust_weights",
            )

        # =========================================================
        # TAB 5: Generate vaccine
        # =========================================================
        with gr.Tab("💉 Generate Vaccine"):
            gr.Markdown("""
            Згенеруй "вакцину" — отруєний зразок з міткою.
            Використай для тренування власної імунної системи.
            API endpoint: `/generate_vaccine`
            """)
            with gr.Row():
                with gr.Column():
                    vacc_img = gr.Image(type="pil", label="Чистий зразок")
                    vacc_type = gr.Dropdown(
                        choices=["label_flip", "backdoor", "clean_label", "feature_corruption"],
                        value="backdoor",
                        label="Тип атаки",
                    )
                    vacc_btn = gr.Button("💉 Generate", variant="primary")
                with gr.Column():
                    vacc_out_img = gr.Image(type="pil", label="Отруєний зразок")
                    vacc_out_json = gr.JSON(label="Metadata")

            vacc_btn.click(
                fn=generate_vaccine_endpoint,
                inputs=[vacc_img, vacc_type],
                outputs=[vacc_out_img, vacc_out_json],
                api_name="generate_vaccine",
            )

        # =========================================================
        # TAB 6: Classify protected
        # =========================================================
        with gr.Tab("🛡️ Classify (Protected)"):
            gr.Markdown("""
            Захищена класифікація — Protected Model + порівняння з Baseline.
            API endpoint: `/classify_protected`
            """)
            with gr.Row():
                with gr.Column():
                    cls_img = gr.Image(type="pil", label="Зразок")
                    cls_btn = gr.Button("🛡️ Classify", variant="primary")
                with gr.Column():
                    cls_out = gr.JSON(label="Результат")

            cls_btn.click(
                fn=classify_protected_endpoint,
                inputs=cls_img,
                outputs=cls_out,
                api_name="classify_protected",
            )

        # =========================================================
        # TAB 7: API Docs
        # =========================================================
        with gr.Tab("📚 API Docs"):
            gr.Markdown(API_DOCS)

    gr.Markdown(f"""
    ---
    *Model: [{MODEL_REPO}](https://huggingface.co/{MODEL_REPO})* ·
    *5 API endpoints exposed via `gradio_client`* ·
    *Built for OWASP LLM Top 10 (LLM04: Data and Model Poisoning)*
    """)


if __name__ == "__main__":
    demo.queue()  # для черги запитів
    demo.launch()