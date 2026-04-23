# Amharic Speech Recognition — Fine-tuned Whisper Medium

[![HuggingFace Model](https://img.shields.io/badge/🤗%20Model-whisper--medium--amharic-blue)](https://huggingface.co/hundehanna/whisper-medium-amharic)
[![HuggingFace Spaces](https://img.shields.io/badge/🤗%20Demo-Spaces-orange)](https://huggingface.co/spaces/hundehanna/amharic-asr-demo)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hundehanna/amharic-asr/blob/main/notebooks/train_whisper_amharic.ipynb)

Fine-tuned `openai/whisper-medium` on the [Leyu Amharic dataset](https://leyu.ai/datasets) — a multi-dialect Amharic speech corpus covering Gojjam, Gonder, Wello, and Shewa dialects (~27,000 samples). This project demonstrates end-to-end low-resource ASR development for Amharic (አማርኛ), one of Ethiopia's most widely spoken languages.

---

## Live Demo

> Try the model directly in your browser — no setup required.

**[🎙️ Launch Demo on HuggingFace Spaces](https://huggingface.co/spaces/hundehanna/amharic-asr-demo)**

Record your voice or upload an audio file to receive an Amharic transcript.

---

## Results

| Model | WER ↓ | Notes |
|-------|--------|-------|
| `openai/whisper-medium` (zero-shot baseline) | _TBD_ | No fine-tuning |
| `whisper-medium-amharic` (this model) | _TBD_ | Fine-tuned on Leyu |

> Results will be updated after training completes.

---

## Dataset: Leyu Amharic

| Dialect | HuggingFace ID | Samples | Duration |
|---------|----------------|---------|----------|
| Gojjam  | `leyu-amharic/leyu-amharic-gojjam-dialect` | 10,575 | 81.6h |
| Gonder  | `leyu-amharic/leyu-amharic-gonder-dialect` | 8,990  | — |
| Wello   | `leyu-amharic/leyu-amharic-wello-dialect`  | 4,860  | — |
| Shewa   | `leyu-amharic/leyu-amharic-shewa-dialect`  | 2,590  | — |
| **Total** | | **~27,000** | |

- Audio: `.wav`, 16kHz mono
- Transcripts: Ethiopic script (`text` column)
- Speakers: mixed gender, recorded on mobile devices in real environments
- All datasets only ship with a `train` split — we apply an 80/10/10 train/val/test split

---

## Model

- **Base**: `openai/whisper-medium` (307M parameters)
- **Task**: Automatic Speech Recognition (`transcribe`)
- **Language**: Amharic (`am`)
- **Training**: Google Colab T4 GPU, fp16, gradient checkpointing
- **Framework**: HuggingFace `transformers` + `Seq2SeqTrainer`

---

## Project Structure

```
amharic-asr/
├── notebooks/
│   └── train_whisper_amharic.ipynb   # Colab training notebook (runnable)
├── src/
│   ├── data_prep.py                  # Dataset loading, resampling, feature extraction
│   ├── train.py                      # Training script
│   └── evaluate.py                   # WER/CER evaluation
├── demo/
│   └── app.py                        # Gradio demo (deployed to HF Spaces)
├── configs/
│   └── training_config.yaml          # Hyperparameters
├── requirements.txt
└── PROJECT_SPEC.md                   # Full project specification
```

---

## Reproduce Training

1. Open the notebook in Google Colab:

   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hundehanna/amharic-asr/blob/main/notebooks/train_whisper_amharic.ipynb)

2. Switch runtime to **T4 GPU** (Runtime → Change runtime type → GPU)

3. Update `HF_USERNAME` in the config cell

4. Run all cells top-to-bottom (~3–5 hours on Colab free tier)

---

## Run the Demo Locally

```bash
pip install -r requirements.txt
python demo/app.py
```

Then open `http://localhost:7860` in your browser.

---

## Key Design Decisions

- **whisper-medium over whisper-small**: Medium performs significantly better on low-resource languages with non-Latin scripts. The VRAM cost is manageable on Colab T4 with gradient checkpointing.
- **All 4 dialects combined**: Training on multiple Amharic dialects (Gojjam, Gonder, Wello, Shewa) produces a more robust model than any single dialect alone.
- **Manual 80/10/10 split**: The Leyu datasets don't include val/test splits, so we create reproducible splits using a fixed random seed.
- **Max 30s audio filter**: Whisper's feature extractor only supports up to 30 seconds; the Leyu dataset includes samples up to 60.5s that must be excluded.

---

## License

Model weights: see [openai/whisper-medium](https://huggingface.co/openai/whisper-medium) license.  
Dataset: see individual [Leyu dataset](https://leyu.ai/datasets) licenses.  
Code in this repo: MIT.
