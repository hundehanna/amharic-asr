"""
Data loading, preprocessing, and feature extraction for Amharic ASR.

Leyu dataset details:
  - Columns: audio (Audio), text (str), dialect, speaker_id, gender
  - Only "train" split exists — we manually create val/test splits
  - Some audio exceeds 30s; Whisper requires <= 30s so we filter those out
"""

import re
from datasets import load_dataset, Audio, concatenate_datasets
from transformers import WhisperProcessor


def load_and_combine_datasets(dataset_names: list[str]) -> object:
    """Load all dialect datasets and combine into one."""
    splits = []
    for name in dataset_names:
        ds = load_dataset(name, split="train")
        splits.append(ds)
    return concatenate_datasets(splits)


def resample_audio(dataset, target_sr: int = 16000):
    return dataset.cast_column("audio", Audio(sampling_rate=target_sr))


def normalize_transcript(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def filter_by_duration(example, min_sec: float = 1.0, max_sec: float = 30.0):
    duration = len(example["audio"]["array"]) / example["audio"]["sampling_rate"]
    return min_sec <= duration <= max_sec


def prepare_dataset(example, processor: WhisperProcessor):
    audio = example["audio"]
    example["input_features"] = processor.feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    transcript = normalize_transcript(example["text"])
    example["labels"] = processor.tokenizer(transcript).input_ids
    return example


# Whisper's decoder maximum target length. Labels longer than this crash training.
MAX_LABEL_LEN = 448


def build_dataset(config: dict, processor: WhisperProcessor):
    dataset_names = config["data"]["dataset_names"]
    min_dur = config["data"]["min_duration_seconds"]
    max_dur = config["data"]["max_duration_seconds"]
    sr = config["data"]["sample_rate"]
    test_size = config["data"]["test_size"]
    val_size = config["data"]["val_size"]

    combined = load_and_combine_datasets(dataset_names)
    combined = resample_audio(combined, target_sr=sr)
    combined = combined.filter(lambda ex: filter_by_duration(ex, min_dur, max_dur))

    # Manual train/val/test split (datasets only ship with "train")
    train_testval = combined.train_test_split(test_size=test_size + val_size, seed=42)
    test_val = train_testval["test"].train_test_split(
        test_size=test_size / (test_size + val_size), seed=42
    )

    dataset_dict = {
        "train": train_testval["train"],
        "validation": test_val["train"],
        "test": test_val["test"],
    }

    for split_name, split_data in dataset_dict.items():
        mapped = split_data.map(
            lambda ex: prepare_dataset(ex, processor),
            remove_columns=split_data.column_names,
            num_proc=2,
        )
        # Drop samples whose tokenized labels exceed Whisper's decoder limit.
        # Amharic in Ethiopic script tokenizes inefficiently; some transcripts
        # produce >448 tokens and would crash training.
        dataset_dict[split_name] = mapped.filter(lambda ex: len(ex["labels"]) <= MAX_LABEL_LEN)

    return dataset_dict
