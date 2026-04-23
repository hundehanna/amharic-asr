# Amharic ASR Fine-Tuning — Project Specification

## Overview

Fine-tune OpenAI's `whisper-medium` model on the Leyu Amharic speech dataset to build a production-quality Automatic Speech Recognition (ASR) system for Amharic. The project includes a live Gradio demo hosted on HuggingFace Spaces, a reproducible training pipeline on Google Colab, and a polished GitHub repository suitable as a portfolio piece.

---

## Goals

| Goal | Description |
|------|-------------|
| Primary | Fine-tune `whisper-medium` on Amharic audio data |
| Secondary | Achieve competitive Word Error Rate (WER) on held-out test set |
| Portfolio | Public GitHub repo + live HuggingFace Spaces demo |
| Reproducibility | Anyone can re-run training via the provided Colab notebook |

---

## Dataset: Leyu Amharic

- **Source**: [leyu.ai/datasets](https://leyu.ai/datasets)
- **Language**: Amharic (አማርኛ) — Ethiopic script
- **Expected format**: Audio files (`.wav` / `.mp3`) + transcript pairs (`.csv` or `.json`)
- **Splits**: Train / Validation / Test (target: 80/10/10 split if not pre-split)
- **Preprocessing needed**:
  - Resample all audio to 16kHz mono (Whisper requirement)
  - Normalize transcripts (strip extra whitespace, consistent punctuation)
  - Remove samples with audio duration < 1s or > 30s

---

## Model

| Setting | Value |
|---------|-------|
| Base model | `openai/whisper-medium` |
| Task | `transcribe` |
| Language | `amharic` |
| Framework | HuggingFace `transformers` + `datasets` |
| Training | `Seq2SeqTrainer` with gradient checkpointing |

### Why whisper-medium?
- Larger than `whisper-small` — better for low-resource languages like Amharic
- Fits in Colab T4 GPU (16GB VRAM) with gradient checkpointing + fp16
- Pre-trained on multilingual data including some Amharic

---

## Training Configuration

```python
TrainingArguments(
    max_steps=4000,           # adjust based on dataset size
    learning_rate=1e-5,
    warmup_steps=500,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,
    fp16=True,
    evaluation_strategy="steps",
    eval_steps=200,
    save_steps=200,
    logging_steps=25,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=True,
)
```

---

## Evaluation

- **Primary metric**: Word Error Rate (WER) — lower is better
- **Secondary metric**: Character Error Rate (CER) — useful for Amharic script
- **Baseline**: `openai/whisper-medium` zero-shot WER on Leyu test set (before fine-tuning)
- **Target**: ≥ 20% relative WER reduction over baseline

---

## Infrastructure

| Component | Tool |
|-----------|------|
| Training | Google Colab (T4 GPU, free tier or Pro) |
| Experiment tracking | HuggingFace Hub (auto-push checkpoints) |
| Model hosting | HuggingFace Hub (`your-username/whisper-medium-amharic`) |
| Demo hosting | HuggingFace Spaces (Gradio SDK, free) |
| Code hosting | GitHub |

---

## Demo App (Gradio)

Features:
- **Microphone input**: Record speech directly in the browser
- **File upload**: Upload `.wav` / `.mp3` file
- **Output**: Amharic transcript displayed in Ethiopic script
- **Language label**: Displayed confidence / detected language
- **Examples**: 3–5 pre-loaded audio samples for visitors to try

---

## Repository Structure

```
amharic-asr/
├── README.md                  # Portfolio README with results, demo link, badges
├── PROJECT_SPEC.md            # This document
├── requirements.txt           # Python dependencies
├── notebooks/
│   └── train_whisper_amharic.ipynb   # Main Colab training notebook
├── src/
│   ├── data_prep.py           # Dataset loading & preprocessing
│   ├── train.py               # Training script (called from notebook)
│   └── evaluate.py            # WER/CER evaluation script
├── demo/
│   └── app.py                 # Gradio demo app (deployed to HF Spaces)
└── configs/
    └── training_config.yaml   # Hyperparameters & paths
```

---

## Milestones

| # | Milestone | Deliverable |
|---|-----------|-------------|
| 1 | Data audit | Understand Leyu format, confirm sample count & quality |
| 2 | Data pipeline | `data_prep.py` — load, resample, split, tokenize |
| 3 | Baseline eval | Zero-shot WER of `whisper-medium` on Leyu test set |
| 4 | Training run | Fine-tuned checkpoint pushed to HF Hub |
| 5 | Evaluation | WER/CER results documented in README |
| 6 | Demo | Gradio app live on HF Spaces |
| 7 | Polish | README badges, model card, clean commit history |

---

## Dependencies

```
transformers>=4.40.0
datasets>=2.19.0
evaluate>=0.4.0
jiwer>=3.0.0          # WER computation
librosa>=0.10.0       # audio processing
soundfile>=0.12.1
accelerate>=0.28.0
gradio>=4.0.0
huggingface_hub>=0.22.0
torch>=2.1.0
```

---

## Portfolio Presentation Notes

- Pin this repo on GitHub profile
- Add HuggingFace model card with: training data, metrics, intended use, limitations
- Include a results table in README comparing baseline vs fine-tuned WER
- Record a short demo GIF of the Gradio app to embed in README
