# 🔌 API Documentation — Poison Defense

> **Live API**: `https://zonda001-poison-defense.hf.space`
> **Space**: [Zonda001/poison-defense](https://huggingface.co/spaces/Zonda001/poison-defense)
> **Model**: [Zonda001/poison-defense-cifar10](https://huggingface.co/Zonda001/poison-defense-cifar10)

Цей API дає **імунний захист для нейромереж** як сервіс. Будь-яка модель може звертатися до 5 endpoint'ів і отримувати:

- ✅ Скан окремих зразків (clean / poisoned)
- ✅ Фільтрацію цілих датасетів
- ✅ `trust_weights` для weighted training власної моделі
- ✅ Генерацію "вакцини" — labeled poisoned зразків
- ✅ Захищену класифікацію через Protected Model

---

## 📑 Зміст

1. [Швидкий старт](#швидкий-старт)
2. [Endpoint /scan](#1-scan--перевірка-одного-зразка)
3. [Endpoint /batch_scan](#2-batch_scan--фільтрація-датасету)
4. [Endpoint /trust_weights](#3-trust_weights--immunity-as-a-service-)
5. [Endpoint /generate_vaccine](#4-generate_vaccine--генерація-вакцини)
6. [Endpoint /classify_protected](#5-classify_protected--захищена-класифікація)
7. [Повний приклад: тренування власної моделі через API](#-повний-приклад-тренування-власної-моделі-через-api)
8. [Обмеження та поради](#обмеження-та-поради)
9. [FAQ](#-faq)

---

## Швидкий старт

### Встановлення

```bash
pip install gradio_client torch pillow
```

### Перше підключення

```python
from gradio_client import Client, handle_file

# Підключаємось до Space
client = Client("Zonda001/poison-defense")

# Дивимось які endpoint'и доступні
endpoints = client.view_api(return_format="dict")["named_endpoints"]
print(list(endpoints.keys()))
# ['/classify_ui', '/scan', '/batch_scan', '/trust_weights',
#  '/generate_vaccine', '/classify_protected']
```

### Перший виклик

```python
# Готуємо тестову картинку
import requests
r = requests.get("https://picsum.photos/seed/1/256")
with open("test.jpg", "wb") as f:
    f.write(r.content)

# Викликаємо API
markdown, json_result = client.predict(
    image=handle_file("test.jpg"),
    api_name="/scan"
)

print(json_result)
# {
#   "safe": True,
#   "poison_probability": 0.34,
#   "trust_weight": 0.66,
#   "predicted_attack_type": "clean",
#   "attack_distribution": {...}
# }
```

✅ Якщо побачив JSON — API живий і працює.

---

## 1. `/scan` — Перевірка одного зразка

Найпростіший endpoint. Береш картинку — дізнаєшся чи отруєна.

### Параметри

| Параметр | Тип | Опис |
|---|---|---|
| `image` | `filepath` | Шлях до картинки або URL |

### Повертає

```python
(
    markdown_str,   # форматований текст для UI
    {
        "safe": bool,                       # True якщо чистий
        "poison_probability": float,        # P(poisoned) у [0, 1]
        "trust_weight": float,              # 1 - P(poisoned)
        "predicted_attack_type": str,       # один з 5 типів
        "attack_distribution": {            # розподіл ймовірностей
            "clean": float,
            "label_flip": float,
            "backdoor": float,
            "clean_label": float,
            "feature_corruption": float
        }
    }
)
```

### Приклад

```python
from gradio_client import Client, handle_file

client = Client("Zonda001/poison-defense")

_, result = client.predict(
    image=handle_file("suspicious_image.jpg"),
    api_name="/scan"
)

if result["safe"]:
    print(f"✅ Зразок чистий (poison_prob: {result['poison_probability']:.1%})")
else:
    print(f"🦠 Зразок отруєний!")
    print(f"   Тип: {result['predicted_attack_type']}")
    print(f"   Впевненість: {result['poison_probability']:.1%}")
```

### Use case

- Швидка валідація перед використанням зразка для тренування
- Виявлення підозрілих зразків у production trafic
- Інтеграція в data pipeline як quality gate

---

## 2. `/batch_scan` — Фільтрація датасету

Скануй багато зразків одразу. Корисно для очищення compromised датасетів.

### Параметри

| Параметр | Тип | Опис |
|---|---|---|
| `files` | `list[filepath]` | Список шляхів до картинок |

### Повертає

```python
{
    "total": int,                # загальна кількість
    "clean_count": int,
    "poisoned_count": int,
    "clean_indices": [int, ...],     # індекси чистих у вхідному списку
    "poisoned_indices": [int, ...],
    "results": [                     # деталі по кожному
        {
            "index": int,
            "safe": bool,
            "poison_probability": float,
            "trust_weight": float,
            "predicted_attack_type": str
        },
        ...
    ]
}
```

### Приклад: фільтрація датасету

```python
import os
from gradio_client import Client, handle_file

client = Client("Zonda001/poison-defense")

# Збираємо всі картинки в папці
dataset_dir = "./my_dataset"
all_files = [
    os.path.join(dataset_dir, f)
    for f in os.listdir(dataset_dir)
    if f.endswith(('.jpg', '.png'))
]

# Скануємо
result = client.predict(
    files=[handle_file(f) for f in all_files],
    api_name="/batch_scan"
)

# Залишаємо тільки чисті
clean_files = [all_files[i] for i in result["clean_indices"]]
poisoned_files = [all_files[i] for i in result["poisoned_indices"]]

print(f"Total: {result['total']}")
print(f"Clean: {result['clean_count']} ({result['clean_count']/result['total']:.1%})")
print(f"Poisoned: {result['poisoned_count']} ({result['poisoned_count']/result['total']:.1%})")

# Переносимо отруєні в карантин
os.makedirs("./quarantine", exist_ok=True)
for f in poisoned_files:
    os.rename(f, f"./quarantine/{os.path.basename(f)}")

print(f"✅ Перенесено {len(poisoned_files)} підозрілих файлів у карантин")
```

### Обмеження

- Рекомендую батчі до 32 зображень за раз (rate limit на безкоштовному CPU)
- Для великих датасетів використовуй цикл з невеликими батчами

---

## 3. `/trust_weights` — Immunity-as-a-Service 🛡️

**Це найважливіший endpoint** — він дає твоїй моделі імунітет без потреби тренувати власний детектор.

### Параметри

| Параметр | Тип | Опис |
|---|---|---|
| `files` | `list[filepath]` | Батч зразків |

### Повертає

```python
{
    "weights": [float, ...],   # одна вага на зразок у [0, 1]
    "soft_mode": True,
    "usage_example": str       # код як використати
}
```

### Приклад: weighted training власної моделі

```python
import torch
import torch.nn.functional as F
from gradio_client import Client, handle_file

client = Client("Zonda001/poison-defense")

# Твій training loop (псевдокод):
for batch in train_loader:
    images, labels = batch
    image_paths = save_batch_to_disk(images)  # помічна функція

    # 1️⃣ ЗАПИТ ДО API за trust weights
    result = client.predict(
        files=[handle_file(p) for p in image_paths],
        api_name="/trust_weights"
    )
    trust_weights = torch.tensor(result["weights"]).to(device)

    # 2️⃣ ЗВИЧАЙНИЙ forward через ТВОЮ модель
    logits = your_model(images)

    # 3️⃣ WEIGHTED LOSS — отруєні зразки впливають у ~5 разів менше
    per_sample_loss = F.cross_entropy(logits, labels, reduction='none')
    weighted_loss = (per_sample_loss * trust_weights).sum() / trust_weights.sum()

    weighted_loss.backward()
    optimizer.step()
```

### Реальні цифри з нашого тесту

```
Batch: [clean_img1, backdoor_img1, clean_img2, corruption_img2]
Weights: [0.663, 0.000, 0.696, 0.380]

Avg clean weight:    0.679
Avg poisoned weight: 0.190
Ratio: 3.58× — чисті важать більше
```

**Інтерпретація:** Backdoor отрута отримала вагу **0.000** — модель повністю проігнорує її при навчанні. Feature corruption отримав 0.380 — теж знижено, але не так радикально (атака менш агресивна).

### Чому це краще, ніж тренувати власний детектор

| Свій детектор | Через наш API |
|---|---|
| Треба ML експертизу | Просто POST-запит |
| Треба збирати poison датасет | Уже зроблено |
| Тренування ~20 хв на GPU | 1 секунда на батч |
| Тільки бекдор-defense | 4 типи атак з коробки |
| Підтримка майбутніх атак — самотужки | Ми оновлюємо детектор за всіх |

---

## 4. `/generate_vaccine` — Генерація "вакцини"

Створює labeled poisoned зразок з валідних чистих даних. Використовуй для тренування **власного детектора** на тому ж домені, що й твоя модель.

### Параметри

| Параметр | Тип | Опис |
|---|---|---|
| `image` | `filepath` | Чистий зразок |
| `attack_type` | `str` | Один з: `label_flip`, `backdoor`, `clean_label`, `feature_corruption` |

### Повертає

```python
(
    poisoned_image_path,   # шлях до згенерованої картинки
    {
        "attack_type": str,
        "is_poisoned": True,
        "original_label_corrupted": bool,
        "new_label": int,
        "perturbation_max": float,    # макс. зміна пікселя [0,1]
        "training_hint": str
    }
)
```

### Приклад: створення власного датасету вакцин

```python
from gradio_client import Client, handle_file
import os

client = Client("Zonda001/poison-defense")

# Маєш папку з чистими картинками
clean_dir = "./clean_samples"
vaccine_dir = "./vaccine_dataset"
os.makedirs(vaccine_dir, exist_ok=True)

attack_types = ["label_flip", "backdoor", "clean_label", "feature_corruption"]
labels = []  # будемо зберігати (path, is_poisoned, attack_type)

for img_name in os.listdir(clean_dir):
    clean_path = os.path.join(clean_dir, img_name)

    # 1. Зберігаємо оригінал як "clean"
    labels.append({
        "path": clean_path,
        "is_poisoned": False,
        "attack_type": "clean"
    })

    # 2. Генеруємо 4 варіанти отрути
    for attack in attack_types:
        poisoned_path, metadata = client.predict(
            image=handle_file(clean_path),
            attack_type=attack,
            api_name="/generate_vaccine"
        )
        # Зберігаємо у свою папку
        new_name = f"{attack}_{img_name}"
        new_path = os.path.join(vaccine_dir, new_name)
        os.rename(poisoned_path, new_path)

        labels.append({
            "path": new_path,
            "is_poisoned": True,
            "attack_type": attack
        })

print(f"✅ Згенеровано {len(labels)} зразків ({len(labels)//5} оригінал + 4 атаки кожен)")
# Тепер можна тренувати власний Detector на цих даних
```

### Параметри атак

| Тип | Що робить | Perturbation |
|---|---|---|
| `label_flip` | Зберігає картинку, змінює мітку | 0 (тільки label) |
| `backdoor` | Додає 4×4 білий патч у куті + target label | Локально високий |
| `clean_label` | Малий gaussian шум, мітка не змінюється | ~0.05-0.08 |
| `feature_corruption` | 20% пікселів замінено на випадкові | Висока |

---

## 5. `/classify_protected` — Захищена класифікація

Класифікує зразок одразу двома моделями (Baseline і Protected) — побачиш, де захист спрацював.

### Параметри

| Параметр | Тип | Опис |
|---|---|---|
| `image` | `filepath` | Зразок для класифікації |

### Повертає

```python
{
    "protected_prediction": {
        "class": str,
        "confidence": float,
        "top3": [{"class": str, "confidence": float}, ...]
    },
    "baseline_prediction": {
        "class": str,
        "confidence": float,
        "top3": [...]
    },
    "agreement": bool,           # True якщо обидві моделі однаково
    "scan": {...}                # результат /scan як bonus
}
```

### Приклад

```python
from gradio_client import Client, handle_file

client = Client("Zonda001/poison-defense")

result = client.predict(
    image=handle_file("image.jpg"),
    api_name="/classify_protected"
)

print(f"🛡️  Protected: {result['protected_prediction']['class']} "
      f"({result['protected_prediction']['confidence']:.1%})")
print(f"❌ Baseline:  {result['baseline_prediction']['class']} "
      f"({result['baseline_prediction']['confidence']:.1%})")

if not result['agreement']:
    print("⚠️  Моделі не зійшлись — можливо, зразок отруєний!")
```

### Use case: anomaly detection

Якщо Baseline і Protected моделі дають різні відповіді — це **сильний сигнал**, що:
1. Зразок підозрілий
2. Baseline могла бути обдурена атакою
3. Варто додатково перевірити через `/scan`

---

## 🏆 Повний приклад: тренування власної моделі через API

**Сценарій:** У тебе є датасет на 10K картинок, у якому 30% можуть бути отруєні. Хочеш натренувати свою класифікаційну модель так, щоб вона мала імунітет.

```python
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from gradio_client import Client, handle_file
import os

# ============================================================
# 1. ПІДКЛЮЧЕННЯ ДО API
# ============================================================
client = Client("Zonda001/poison-defense")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# 2. ТВОЯ МОДЕЛЬ (можеш використати будь-яку)
# ============================================================
from torchvision.models import resnet18
model = resnet18(num_classes=10).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ============================================================
# 3. ДОПОМІЖНА ФУНКЦІЯ — отримати ваги через API
# ============================================================
def get_trust_weights_from_api(image_paths):
    """Запит до Poison Defense API за trust_weights."""
    result = client.predict(
        files=[handle_file(p) for p in image_paths],
        api_name="/trust_weights"
    )
    return torch.tensor(result["weights"]).to(device)

# ============================================================
# 4. TRAINING LOOP З ІМУННИМ ЗАХИСТОМ
# ============================================================
def train_one_epoch(model, train_loader, optimizer):
    model.train()
    total_loss = 0.0

    for batch_idx, (images, labels, image_paths) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # 🛡️ КЛЮЧОВА ЛІНІЯ: отримуємо trust weights від нашого API
        trust_weights = get_trust_weights_from_api(image_paths)

        # Forward
        logits = model(images)

        # ⚖️ Weighted loss — отруєні зразки впливають менше
        per_sample = F.cross_entropy(logits, labels, reduction='none')
        loss = (per_sample * trust_weights).sum() / trust_weights.sum()

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 10 == 0:
            avg_trust = trust_weights.mean().item()
            print(f"Batch {batch_idx}: loss={loss.item():.3f}, avg_trust={avg_trust:.2f}")

    return total_loss / len(train_loader)

# ============================================================
# 5. ЗАПУСК
# ============================================================
for epoch in range(10):
    avg_loss = train_one_epoch(model, train_loader, optimizer)
    print(f"Epoch {epoch+1}: avg_loss={avg_loss:.4f}")

# Готово! Твоя модель має імунітет проти data poisoning.
# Без власного детектора. Без знання про атаки. Просто через API.
```

### Що відбулося

Твоя модель навчилась тільки на чистих зразках (effective), бо отруєні мали вагу близьку до 0. Результат — модель не запам'ятала атакувальні патерни.

---

## Обмеження та поради

### Rate limits

Безкоштовний HuggingFace Space має обмеження:
- ~30 запитів/хвилину
- Може бути cold start (30-60 сек) якщо Space "спав"
- Memory: 16GB CPU

### Розмір зображень

API приймає будь-який розмір, але **автоматично ресайзить до 32×32** (CIFAR-10 розмір). Якщо у тебе картинки високої роздільності — інформація буде втрачена.

Для високої роздільності варто перетренувати модель на більшому розмірі (`tiny_imagenet`, `imagenet`).

### Підтримувані формати

- ✅ JPEG, PNG, WebP, BMP
- ✅ RGB, RGBA, grayscale (автоконвертація)
- ❌ TIFF з кількома сторінками (бери першу)
- ❌ Анімовані GIF (бери перший кадр)

### Latency

- Single image: ~0.5-1 сек
- Batch (32): ~3-5 сек
- Cold start: до 60 сек

Для high-throughput production — переходь на HF Inference Endpoints (платно) або deploy локально.

### Що НЕ робить API

- ❌ НЕ розкриває внутрішні patterns Detector'а (anti-fingerprinting)
- ❌ НЕ тренує твою модель — лише дає trust weights
- ❌ НЕ зберігає твої запити
- ❌ НЕ працює офлайн (потрібен інтернет)

---

## ❓ FAQ

### Q: Чи коректно отримати `trust_weight = 0.000`?

**A:** Так. Це означає, що Detector максимально впевнений у тому, що зразок отруєний. У weighted loss такий зразок просто буде проігнорований (множник 0). Це бажана поведінка.

### Q: Чому clean weights не 1.0, а ~0.7?

**A:** Detector — імовірнісний класифікатор. Він рідко буває впевнений на 100%. Реалістичні значення 0.6-0.9 для чистих, 0.0-0.3 для отруєних. Головне — **співвідношення**, а не абсолютні величини. Ratio 3-5× достатньо для ефективного захисту.

### Q: Що якщо у мене не CIFAR-10 дані?

**A:** API навчений на CIFAR-10 (10 класів, 32×32). Для своїх даних:
- Якщо схожі за форматом і розміром — пробуй, можливо спрацює
- Якщо принципово інші — використай `/generate_vaccine` для створення власного датасету і натренуй свій детектор за нашим кодом

### Q: Як перевірити що API працює?

```python
from gradio_client import Client
client = Client("Zonda001/poison-defense")
endpoints = client.view_api(return_format="dict")["named_endpoints"]
assert len(endpoints) == 6, "API має 6 endpoints"
print("✅ API живий і працює")
```

### Q: Чи можу я використати API у комерційних цілях?

**A:** Так, ліцензія MIT. Але враховуй rate limits безкоштовного Space — для production треба self-host або платні HF Inference Endpoints.

### Q: Що буде, якщо передати завелику картинку?

**A:** Gradio автоматично ресайзить. Але дуже великі (>10MB) можуть timeout-нути. Рекомендую ресайзити до 256×256 перед відправкою.

### Q: Чи можу я попросити додати нові типи атак?

**A:** Так. Зараз 4 типи: label_flip, backdoor, clean_label, feature_corruption. Якщо потрібен ще (gradient matching, feature collision, etc.) — відкрий issue на GitHub.

---

## 🔗 Корисні посилання

- **Live Demo & API**: [Zonda001/poison-defense](https://huggingface.co/spaces/Zonda001/poison-defense)
- **Model Weights**: [Zonda001/poison-defense-cifar10](https://huggingface.co/Zonda001/poison-defense-cifar10)
- **Architecture Deep-dive**: [ARCHITECTURE.md](ARCHITECTURE.md)
- **Deploy your own**: [API_DEPLOY.md](../API_DEPLOY.md)
- **Source code**: [GitHub repo](#) *(додай посилання)*

---

<div align="center">

**Built with 🛡️ for OWASP LLM Top 10 (LLM04: Data and Model Poisoning)**

</div>
