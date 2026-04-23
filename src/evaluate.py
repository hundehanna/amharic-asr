"""
Evaluate fine-tuned Whisper on the Leyu test set.
Reports WER and CER.
"""

import torch
import evaluate
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from datasets import load_dataset, Audio
from data_prep import normalize_transcript


wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")


def transcribe_batch(batch, model, processor, device):
    inputs = torch.tensor(batch["input_features"], dtype=torch.float32).to(device)
    with torch.no_grad():
        predicted_ids = model.generate(inputs)
    batch["predictions"] = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    batch["references"] = [normalize_transcript(r) for r in batch["labels_text"]]
    return batch


def run_evaluation(model_id: str, dataset_name: str, test_split: str = "test"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(model_id).to(device)
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="amharic", task="transcribe"
    )

    dataset = load_dataset(dataset_name, split=test_split)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    results = dataset.map(
        lambda batch: transcribe_batch(batch, model, processor, device),
        batched=True,
        batch_size=8,
    )

    wer = wer_metric.compute(predictions=results["predictions"], references=results["references"])
    cer = cer_metric.compute(predictions=results["predictions"], references=results["references"])

    print(f"WER: {wer * 100:.2f}%")
    print(f"CER: {cer * 100:.2f}%")
    return {"wer": wer, "cer": cer}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", required=True, help="HF model ID or local path")
    parser.add_argument("--dataset", required=True, help="HF dataset name")
    parser.add_argument("--split", default="test")
    args = parser.parse_args()
    run_evaluation(args.model_id, args.dataset, args.split)
