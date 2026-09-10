"""
🛡️ Text Poison Detector — DistilBERT fine-tuning

Окремий тренувальний скрипт для текстового детектора. Незалежний від
image-тренування. Використовує готовий DistilBERT з HuggingFace
+ classification head.

Аналог нашого image Detector'а: бінарна класифікація
(is_poisoned / safe) + опційно multi-class для типу атаки.

Датасет: deepset/prompt-injections (~660 зразків, баланс ~50/50)
+ опційно augmentation з jackhhao/jailbreak-classification.

Тренування: ~30-60 хв на T4 GPU.

Запуск у Colab:
    !pip install -q transformers datasets accelerate evaluate scikit-learn
    !python train_text.py --epochs 3 --batch_size 16 --lr 2e-5
"""

import argparse
import os
import json
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score


# =============================================================================
# КОНФІГ
# =============================================================================
MODEL_NAME = "distilbert-base-uncased"
LABEL_NAMES = ["safe", "poisoned"]  # 0 = clean, 1 = injection/poisoned
NUM_LABELS = 2


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", type=str, default="./text_checkpoints")
    p.add_argument(
        "--use_jailbreak",
        action="store_true",
        help="Додати jackhhao/jailbreak-classification як augmentation",
    )
    return p.parse_args()


# =============================================================================
# ЗАВАНТАЖЕННЯ ДАТАСЕТУ
# =============================================================================
def load_combined_dataset(use_jailbreak: bool = False) -> Dataset:
    """
    Завантажує і об'єднує датасети.

    Основний: deepset/prompt-injections — ~660 зразків, баланс ~50/50,
    label 0=safe / label 1=injection.

    Опційне augmentation: jackhhao/jailbreak-classification — додає jailbreaks
    у positive клас.
    """
    print("📥 Завантажую deepset/prompt-injections...")
    main = load_dataset("deepset/prompt-injections")
    train_main = main["train"]
    test_main = main["test"]
    print(f"   train: {len(train_main)} зразків, test: {len(test_main)}")

    # Стандартизуємо колонки до {"text", "label"}
    def normalize_main(example):
        return {"text": example["text"], "label": int(example["label"])}

    train_main = train_main.map(normalize_main, remove_columns=train_main.column_names)
    test_main = test_main.map(normalize_main, remove_columns=test_main.column_names)

    if use_jailbreak:
        try:
            print("📥 Додаю jackhhao/jailbreak-classification...")
            jb = load_dataset("jackhhao/jailbreak-classification")
            # У цьому датасеті: prompt + type ("jailbreak" / "benign")
            def normalize_jb(example):
                # Адаптуй імена колонок якщо вони інші
                text_col = "prompt" if "prompt" in example else list(example.keys())[0]
                lbl_col = "type" if "type" in example else "label"
                text = example.get(text_col, "")
                lbl_val = example.get(lbl_col, "")
                # type: "jailbreak" → 1, інакше 0
                lbl = 1 if str(lbl_val).lower() in ("jailbreak", "1", "true") else 0
                return {"text": text, "label": lbl}

            jb_train = jb["train"].map(normalize_jb, remove_columns=jb["train"].column_names)
            train_main = concatenate_datasets([train_main, jb_train])
            print(f"   після augmentation: train={len(train_main)} зразків")
        except Exception as e:
            print(f"   ⚠️ Не вдалося додати jailbreak датасет: {e}")
            print(f"   Продовжуємо лише з deepset")

    # Аналіз балансу
    labels_arr = np.array(train_main["label"])
    n_safe = (labels_arr == 0).sum()
    n_poison = (labels_arr == 1).sum()
    print(f"\n📊 Розподіл міток у train:")
    print(f"   safe (0):     {n_safe} ({100*n_safe/len(labels_arr):.1f}%)")
    print(f"   poisoned (1): {n_poison} ({100*n_poison/len(labels_arr):.1f}%)")

    return train_main, test_main


# =============================================================================
# ТОКЕНІЗАЦІЯ
# =============================================================================
def tokenize_dataset(dataset, tokenizer, max_length: int):
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding=False,  # збираємо в DataCollator
        )

    return dataset.map(tokenize_fn, batched=True)


# =============================================================================
# МЕТРИКИ
# =============================================================================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    probs = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()

    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )

    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = 0.0

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
    }


# =============================================================================
# MAIN
# =============================================================================
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ Device: {device}")

    # 1. Дані
    train_ds, test_ds = load_combined_dataset(use_jailbreak=args.use_jailbreak)

    # 2. Tokenizer і модель
    print(f"\n🔧 Завантажую {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label={i: name for i, name in enumerate(LABEL_NAMES)},
        label2id={name: i for i, name in enumerate(LABEL_NAMES)},
    )

    # 3. Токенізація
    print("\n🔡 Токенізую датасети...")
    train_ds = tokenize_dataset(train_ds, tokenizer, args.max_length)
    test_ds = tokenize_dataset(test_ds, tokenizer, args.max_length)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 4. Training config
    training_args = TrainingArguments(
        output_dir=str(output_dir / "trainer_runs"),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=20,
        report_to="none",
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        processing_class=tokenizer,     # ← змінено
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # 5. Тренування
    print("\n🚀 Старт тренування...\n")
    trainer.train()

    # 6. Фінальна оцінка
    print("\n📊 Фінальна оцінка на test set:")
    eval_results = trainer.evaluate()
    for key, val in eval_results.items():
        if isinstance(val, float):
            print(f"   {key}: {val:.4f}")

    # 7. Збереження
    final_dir = output_dir / "final"
    final_dir.mkdir(exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    # Зберігаємо метрики
    with open(final_dir / "eval_metrics.json", "w") as f:
        json.dump(eval_results, f, indent=2)

    print(f"\n✅ Модель збережена у {final_dir}")
    print(f"   - pytorch_model.bin (або model.safetensors)")
    print(f"   - tokenizer files")
    print(f"   - eval_metrics.json")
    print(f"\nДалі: push_text_to_hub.py --username Zonda001 --repo poison-defense-text")


if __name__ == "__main__":
    main()
